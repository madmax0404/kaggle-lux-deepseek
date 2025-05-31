# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import math
import os
import textwrap
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from transformers.utils import is_peft_available
from transformers.utils.deprecation import deprecate_kwarg

from trl.core import masked_mean, masked_whiten
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.ppo_config import PPOConfig
from trl.trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    generate_model_card,
    get_comet_experiment_url,
    get_reward,
    log_table_to_comet_experiment,
    peft_module_casting_to_bf16,
    prepare_deepspeed,
    print_rich_table,
    truncate_response,
)


if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_wandb_available():
    import wandb


INVALID_LOGPROB = 1.0


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

    def forward(self, **kwargs):
        output = self.critic_backbone(**kwargs)
        logits = self.value_model.score(output.hidden_states[-1])
        return self.policy(**kwargs), logits


class ModifiedPPOTrainer(Trainer):
    _tag_names = ["trl", "ppo"]

    @deprecate_kwarg("config", "0.15.0", "args", warn_if_greater_or_equal_version=True, raise_if_both_names=True)
    @deprecate_kwarg(
        "tokenizer", "0.15.0", "processing_class", warn_if_greater_or_equal_version=True, raise_if_both_names=True
    )
    @deprecate_kwarg("policy", "0.15.0", "model", warn_if_greater_or_equal_version=True, raise_if_both_names=True)
    def __init__(
        self,
        args: PPOConfig,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ],
        model: nn.Module,
        model_2: nn.Module,
        reward_model: nn.Module,
        reward_functions: list,
        game_env = None,
        value_model: Optional[nn.Module] = None,
        value_model_2: Optional[nn.Module] = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        # less commonly used
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[list[TrainerCallback]] = None,
        peft_config: Optional["PeftConfig"] = None,
        num_games_to_train: int = 1,
    ) -> None:

        self.args = args
        self.processing_class = processing_class
        self.policy_model = model
        self.policy_model_2 = model_2

        # Define the collator if not provided
        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)

        # Handle stop token settings: update policy model's generation_config to use provided stop token
        if args.stop_token and args.stop_token_id:
            raise ValueError("You cannot set both `stop_token` and `stop_token_id`.")
        elif args.stop_token:
            if args.stop_token == "eos":
                self.policy_model.generation_config.eos_token_id = self.stop_token_id = processing_class.eos_token_id
                self.policy_model_2.generation_config.eos_token_id = self.stop_token_id = processing_class.eos_token_id
            else:
                raise ValueError(
                    f"Unknown `stop_token` {args.stop_token}. Allowed values are: `'eos'` and `None` (no stop token)."
                )
        else:
            self.policy_model.generation_config.eos_token_id = self.stop_token_id = args.stop_token_id  # None or int
            self.policy_model_2.generation_config.eos_token_id = self.stop_token_id = args.stop_token_id  # None or int

        # peft support
        if not is_peft_available() and peft_config is not None:
            raise ImportError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_confg, we merge and unload it first
            if isinstance(self.policy_model, PeftModel):
                self.policy_model = self.policy_model.merge_and_unload()
            if isinstance(self.policy_model_2, PeftModel):
                self.policy_model_2 = self.policy_model_2.merge_and_unload()
            
            # get peft model with the given config
            self.policy_model = get_peft_model(self.policy_model, peft_config)
            if args.bf16 and getattr(self.policy_model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(self.policy_model)

            self.policy_model_2 = get_peft_model(self.policy_model_2, peft_config)
            if args.bf16 and getattr(self.policy_model_2, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(self.policy_model_2)

        self.is_peft_model = is_peft_available() and isinstance(self.policy_model, PeftModel)
        self.is_peft_model_2 = is_peft_available() and isinstance(self.policy_model_2, PeftModel)
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        self.reward_model = reward_model
        self.value_model = value_model
        self.value_model_2 = value_model_2
        self.data_collator = data_collator
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_2, self.lr_scheduler_2 = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        if args.whiten_rewards:
            assert (
                args.local_mini_batch_size >= 8
            ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = args.local_batch_size

        #########
        # setup model, optimizer, and others
        #########
        for module in [self.policy_model, self.policy_model_2, self.value_model, self.value_model_2, self.reward_model]:
            if module is not None:
                disable_dropout_in_model(module)
        self.model = PolicyAndValueWrapper(self.policy_model, self.value_model)
        self.model_2 = PolicyAndValueWrapper(self.policy_model_2, self.value_model_2)
        self.model.config = self.policy_model.config  # needed for pushing to hub
        self.model_2.config = self.policy_model_2.config  # needed for pushing to hub
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.callback_handler_2 = CallbackHandler(
            self.callbacks, self.model_2, self.processing_class, self.optimizer_2, self.lr_scheduler_2
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )
        # copy 2
        self.control_2 = TrainerControl()
        self.state_2 = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler_2.callbacks + [self.control_2] if isinstance(cb, ExportableState)
            ],
        )


        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)
        if hasattr(self.model_2, "add_model_tags"):
            self.model_2.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        self.model_2, self.optimizer_2, self.dataloader = accelerator.prepare(self.model_2, self.optimizer_2, self.dataloader) # edit it later
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            self.reward_model = prepare_deepspeed(
                self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
            )
        else:
            self.reward_model = self.reward_model.to(self.accelerator.device)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with self.accelerator.unwrap_model(
            self.model.policy
        ).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.model_adapter_name or "default")

        # copy 2
        with self.accelerator.unwrap_model(
            self.model_2.policy
        ).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model_2.policy.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model_2.policy.set_adapter(self.model_adapter_name or "default")

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        backup_model = self.model
        backup_model_2 = self.model_2
        self.model = self.model.policy  # save only the policy
        self.model_2 = self.model_2.policy  # save only the policy

        if self.is_deepspeed_enabled:
            backup_deepspeed = self.deepspeed
            self.deepspeed = self.model
            self.deepspeed_2 = self.model_2

        super().save_model(output_dir, _internal_call)

        self.model = backup_model
        self.model_2 = backup_model_2

        if self.is_deepspeed_enabled:
            self.deepspeed = backup_deepspeed

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        optimizer_2 = self.optimizer_2
        model = self.model
        model_2 = self.model_2
        reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        approxkl_stats_2 = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats_2 = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats_2 = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats_2 = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats_2 = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        entropy_stats_2 = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        ratio_stats_2 = torch.zeros(stats_shape, device=device)
        model.train()
        model_2.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches * args.num_mini_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # copy 2
        self.state_2.global_step = 0
        self.state_2.episode = 0
        self.state_2.max_steps = args.num_total_batches * args.num_mini_batches
        self.state_2.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        # copy 2
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state_2.logging_steps = math.ceil(self.state_2.max_steps * args.logging_steps)
            else:
                self.state_2.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state_2.eval_steps = math.ceil(self.state_2.max_steps * args.eval_steps)
            else:
                self.state_2.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state_2.save_steps = math.ceil(self.state_2.max_steps * args.save_steps)
            else:
                self.state_2.save_steps = args.save_steps
        self.control_2 = self.callback_handler_2.on_train_begin(args, self.state_2, self.control_2)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.deepspeed_2 = self.model_2
            self.model_wrapped = self.model
            self.model_wrapped_2 = self.model_2

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["input_ids"].to(device) # edit later
                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                scores = []
                sequence_lengths = []
                values = []

                # copy 2
                queries_2 = data["input_ids"].to(device) # edit later
                context_length_2 = queries_2.shape[1]
                responses_2 = []
                postprocessed_responses_2 = []
                logprobs_2 = []
                scores_2 = []
                sequence_lengths_2 = []
                values_2 = []

                with unwrap_model_for_generation(
                    self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    query_responses, logitss = batch_generation(
                        unwrapped_model.policy,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )
                # copy 2
                with unwrap_model_for_generation(
                    self.model_2, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model_2:
                    query_responses_2, logitss_2 = batch_generation(
                        unwrapped_model_2.policy,
                        queries_2,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )

                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    all_logprob = F.log_softmax(logits, dim=-1)
                    logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    del logits, all_logprob
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1
                    unwrapped_value_model = accelerator.unwrap_model(model).value_model
                    full_value, _, _ = get_reward(
                        unwrapped_value_model, query_response, processing_class.pad_token_id, context_length
                    )

                    value = full_value[:, context_length - 1 : -1].squeeze(-1)
                    _, score, _ = get_reward(
                        reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                    )

                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                    values.append(value)

                # copy 2
                for i in range(0, queries_2.shape[0], args.local_rollout_forward_batch_size):
                    query_2 = queries_2[i : i + args.local_rollout_forward_batch_size]
                    query_response_2 = query_responses_2[i : i + args.local_rollout_forward_batch_size]
                    response_2 = query_response_2[:, context_length_2:]
                    logits_2 = logitss_2[i : i + args.local_rollout_forward_batch_size]
                    all_logprob_2 = F.log_softmax(logits_2, dim=-1)
                    logprob_2 = torch.gather(all_logprob_2, 2, response_2.unsqueeze(-1)).squeeze(-1)
                    del logits_2, all_logprob_2
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response_2 = response_2
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response_2 = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response_2
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response_2 = torch.cat((query_2, postprocessed_response_2), 1)
                    sequence_length_2 = first_true_indices(postprocessed_response_2 == processing_class.pad_token_id) - 1
                    unwrapped_value_model_2 = accelerator.unwrap_model(model_2).value_model
                    full_value_2, _, _ = get_reward(
                        unwrapped_value_model_2, query_response_2, processing_class.pad_token_id, context_length_2
                    )

                    value_2 = full_value_2[:, context_length_2 - 1 : -1].squeeze(-1)
                    _, score_2, _ = get_reward(
                        reward_model, postprocessed_query_response_2, processing_class.pad_token_id, context_length_2
                    )

                    responses_2.append(response_2)
                    postprocessed_responses_2.append(postprocessed_response_2)
                    logprobs_2.append(logprob_2)
                    sequence_lengths_2.append(sequence_length_2)
                    scores_2.append(score_2)
                    values_2.append(value_2)

                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                values = torch.cat(values, 0)

                # copy 2
                responses_2 = torch.cat(responses_2, 0)
                postprocessed_responses_2 = torch.cat(postprocessed_responses_2, 0)
                logprobs_2 = torch.cat(logprobs_2, 0)
                sequence_lengths_2 = torch.cat(sequence_lengths_2, 0)
                scores_2 = torch.cat(scores_2, 0)
                values_2 = torch.cat(values_2, 0)

                del (logprob, logprob_2, full_value, full_value_2, value, value_2, score, score_2, unwrapped_model, unwrapped_model_2)
                torch.cuda.empty_cache()
                gc.collect()

                # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                contain_eos_token = torch.any(postprocessed_responses == self.processing_class.eos_token_id, dim=-1)
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # copy 2
                contain_eos_token_2 = torch.any(postprocessed_responses_2 == self.processing_class.eos_token_id, dim=-1)
                if self.args.missing_eos_penalty is not None:
                    scores_2[~contain_eos_token_2] -= self.args.missing_eos_penalty
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                sequence_lengths_p1 = sequence_lengths + 1
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # copy 2
                response_idxs_2 = torch.arange(responses_2.shape[1], device=responses_2.device).repeat(responses_2.shape[0], 1)
                padding_mask_2 = response_idxs_2 > sequence_lengths_2.unsqueeze(1)
                logprobs_2 = torch.masked_fill(logprobs_2, padding_mask_2, INVALID_LOGPROB)
                sequence_lengths_p1_2 = sequence_lengths_2 + 1
                padding_mask_p1_2 = response_idxs_2 > (sequence_lengths_p1_2.unsqueeze(1))
                values_2 = torch.masked_fill(values_2, padding_mask_p1_2, 0)

                # 4. compute rewards
                kl = logprobs - logprobs_2
                non_score_reward = -args.kl_coef * kl
                rewards = non_score_reward.clone()
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                rewards[[actual_start, actual_end]] += scores

                # copy 2
                kl_2 = logprobs_2 - logprobs
                non_score_reward_2 = -args.kl_coef * kl_2
                rewards_2 = non_score_reward_2.clone()
                actual_start_2 = torch.arange(rewards_2.size(0), device=rewards_2.device)
                actual_end_2 = torch.where(sequence_lengths_p1_2 < rewards_2.size(1), sequence_lengths_p1_2, sequence_lengths_2)
                rewards_2[[actual_start_2, actual_end_2]] += scores_2

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                    #copy 2
                    rewards_2 = masked_whiten(rewards_2, mask=~padding_mask_p1_2, shift_mean=False)
                    rewards_2 = torch.masked_fill(rewards_2, padding_mask_p1_2, 0)

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]

                # copy 2
                lastgaelam_2 = 0
                advantages_reversed_2 = []
                gen_length_2 = responses_2.shape[1]

                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)

                # copy 2
                for t in reversed(range(gen_length_2)):
                    nextvalues_2 = values_2[:, t + 1] if t < gen_length_2 - 1 else 0.0
                    delta_2 = rewards_2[:, t] + args.gamma * nextvalues_2 - values_2[:, t]
                    lastgaelam_2 = delta_2 + args.gamma * args.lam * lastgaelam_2
                    advantages_reversed_2.append(lastgaelam_2)

                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)

                # copy 2
                advantages_2 = torch.stack(advantages_reversed_2[::-1], axis=1)
                returns_2 = advantages_2 + values_2
                advantages_2 = masked_whiten(advantages_2, ~padding_mask_2)
                advantages_2 = torch.masked_fill(advantages_2, padding_mask_2, 0)

                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_advantage = advantages[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]
                            mb_return = returns[micro_batch_inds]
                            mb_values = values[micro_batch_inds]

                            output, vpred_temp = forward(model, mb_query_responses, processing_class.pad_token_id)
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7
                            new_all_logprobs = F.log_softmax(logits, dim=-1)
                            new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )
                            vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                            vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
                            vpredclipped = torch.clamp(
                                vpred,
                                mb_values - args.cliprange_value,
                                mb_values + args.cliprange_value,
                            )
                            vf_losses1 = torch.square(vpred - mb_return)
                            vf_losses2 = torch.square(vpredclipped - mb_return)
                            vf_loss_max = torch.max(vf_losses1, vf_losses2)
                            vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
                            vf_clipfrac = masked_mean(
                                (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
                            )
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                            loss = pg_loss + args.vf_coef * vf_loss
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                pg_clipfrac = masked_mean(
                                    (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                                )
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    vf_clipfrac
                                )
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()

                        # copy 2
                        with accelerator.accumulate(model_2):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_advantage_2 = advantages_2[micro_batch_inds]
                            mb_responses_2 = responses_2[micro_batch_inds]
                            mb_query_responses_2 = query_responses_2[micro_batch_inds]
                            mb_logprobs_2 = logprobs_2[micro_batch_inds]
                            mb_return_2 = returns_2[micro_batch_inds]
                            mb_values_2 = values_2[micro_batch_inds]

                            output_2, vpred_temp_2 = forward(model_2, mb_query_responses_2, processing_class.pad_token_id)
                            logits_2 = output_2.logits[:, context_length - 1 : -1]
                            logits_2 /= args.temperature + 1e-7
                            new_all_logprobs_2 = F.log_softmax(logits_2, dim=-1)
                            new_logprobs_2 = torch.gather(new_all_logprobs_2, 2, mb_responses_2.unsqueeze(-1)).squeeze(-1)
                            new_logprobs_2 = torch.masked_fill(
                                new_logprobs_2, padding_mask_2[micro_batch_inds], INVALID_LOGPROB
                            )
                            vpred_2 = vpred_temp_2[:, context_length - 1 : -1].squeeze(-1)
                            vpred_2 = torch.masked_fill(vpred_2, padding_mask_p1_2[micro_batch_inds], 0)
                            vpredclipped_2 = torch.clamp(
                                vpred_2,
                                mb_values_2 - args.cliprange_value,
                                mb_values_2 + args.cliprange_value,
                            )
                            vf_losses1_2 = torch.square(vpred_2 - mb_return_2)
                            vf_losses2_2 = torch.square(vpredclipped_2 - mb_return_2)
                            vf_loss_max_2 = torch.max(vf_losses1_2, vf_losses2_2)
                            vf_loss_2 = 0.5 * masked_mean(vf_loss_max_2, ~padding_mask_p1_2[micro_batch_inds])
                            vf_clipfrac_2 = masked_mean(
                                (vf_losses2_2 > vf_losses1_2).float(), ~padding_mask_p1_2[micro_batch_inds]
                            )
                            logprobs_diff_2 = new_logprobs_2 - mb_logprobs_2
                            ratio_2 = torch.exp(logprobs_diff_2)
                            pg_losses_2 = -mb_advantage_2 * ratio_2
                            pg_losses2_2 = -mb_advantage_2 * torch.clamp(ratio_2, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max_2 = torch.max(pg_losses_2, pg_losses2_2)
                            pg_loss_2 = masked_mean(pg_loss_max_2, ~padding_mask_2[micro_batch_inds])
                            loss_2 = pg_loss_2 + args.vf_coef * vf_loss_2
                            accelerator.backward(loss_2)
                            optimizer_2.step()
                            optimizer_2.zero_grad()
                            with torch.no_grad():
                                pg_clipfrac_2 = masked_mean(
                                    (pg_losses2_2 > pg_losses_2).float(), ~padding_mask_2[micro_batch_inds]
                                )
                                prob_dist_2 = torch.nn.functional.softmax(logits_2, dim=-1)
                                entropy_2 = torch.logsumexp(logits_2, dim=-1) - torch.sum(prob_dist_2 * logits_2, dim=-1)
                                approxkl_2 = 0.5 * (logprobs_diff_2**2).mean()
                                approxkl_stats_2[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl_2
                                pg_clipfrac_stats_2[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac_2
                                )
                                pg_loss_stats_2[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss_2
                                vf_loss_stats_2[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss_2
                                vf_clipfrac_stats_2[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    vf_clipfrac_2
                                )
                                entropy_stats_2[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy_2.mean()
                                ratio_stats_2[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio_2.mean()
                        
                            
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, vpred_temp, logits, new_all_logprobs, new_logprobs, vpred, vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,
                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                        mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    del (
                        output_2, vpred_temp_2, logits_2, new_all_logprobs_2, new_logprobs_2, vpred_2, vpredclipped_2,
                        vf_losses1_2, vf_losses2_2, vf_loss_2, vf_clipfrac_2, logprobs_diff_2, ratio_2, pg_losses_2, pg_losses2_2, pg_loss_max_2,
                        pg_loss_2, loss_2, pg_clipfrac_2, prob_dist_2, entropy_2, approxkl_2, mb_return_2,
                        mb_advantage_2, mb_values_2, mb_responses_2, mb_query_responses_2, mb_logprobs_2,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                rlhf_reward = mean_non_score_reward + scores.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = (
                    self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
                )
                metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather_for_metrics(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather_for_metrics(vf_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

                # copy 2
                mean_kl_2 = kl_2.sum(1).mean()
                mean_entropy_2 = (-logprobs_2).sum(1).mean()
                mean_non_score_reward_2 = non_score_reward_2.sum(1).mean()
                rlhf_reward_2 = mean_non_score_reward_2 + scores_2.mean()
                eps_2 = int(self.state_2.episode / (time.time() - start_time))
                metrics_2 = {}
                metrics_2["eps_2"] = eps_2
                metrics_2["objective/kl_2"] = self.accelerator.gather_for_metrics(mean_kl_2).mean().item()
                metrics_2["objective/entropy_2"] = self.accelerator.gather_for_metrics(mean_entropy_2).mean().item()
                metrics_2["objective/non_score_reward_2"] = (
                    self.accelerator.gather_for_metrics(mean_non_score_reward_2).mean().item()
                )
                metrics_2["objective/rlhf_reward_2"] = self.accelerator.gather_for_metrics(rlhf_reward_2).mean().item()
                metrics_2["objective/scores_2"] = self.accelerator.gather_for_metrics(scores_2.mean()).mean().item()
                metrics_2["policy/approxkl_avg_2"] = self.accelerator.gather_for_metrics(approxkl_stats_2).mean().item()
                metrics_2["policy/clipfrac_avg_2"] = self.accelerator.gather_for_metrics(pg_clipfrac_stats_2).mean().item()
                metrics_2["loss/policy_avg_2"] = self.accelerator.gather_for_metrics(pg_loss_stats_2).mean().item()
                metrics_2["loss/value_avg_2"] = self.accelerator.gather_for_metrics(vf_loss_stats_2).mean().item()
                metrics_2["val/clipfrac_avg_2"] = self.accelerator.gather_for_metrics(vf_clipfrac_stats_2).mean().item()
                metrics_2["policy/entropy_avg_2"] = self.accelerator.gather_for_metrics(entropy_stats_2).mean().item()
                metrics_2["val/ratio_2"] = self.accelerator.gather_for_metrics(ratio_stats_2).mean().item()
                metrics_2["val/ratio_var_2"] = self.accelerator.gather_for_metrics(ratio_stats_2).var().item()
                metrics_2["val/num_eos_tokens_2"] = (responses_2 == processing_class.eos_token_id).sum().item()
                metrics_2["lr_2"] = self.lr_scheduler_2.get_last_lr()[0]
                metrics_2["episode_2"] = self.state_2.episode
                self.state_2.epoch = self.state_2.episode / self.train_dataset_len  # used by self.log
                self.state_2.global_step += 1
                self.log(metrics_2)

            my_output_dir = self.args.output_dir


            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores, metrics, non_score_reward

            self.args.output_dir = my_output_dir + "_2"

            # copy 2
            self.lr_scheduler_2.step()
            self.control_2 = self.callback_handler_2.on_step_end(args, self.state_2, self.control_2)
            if self.control_2.should_save:
                self._save_checkpoint(model_2, trial=None)
                self.control_2 = self.callback_handler_2.on_save(self.args, self.state_2, self.control_2)
            del kl_2, mean_kl_2, mean_entropy_2, mean_non_score_reward_2, scores_2, metrics_2, non_score_reward_2
            torch.cuda.empty_cache()
            gc.collect()

            self.args.output_dir = my_output_dir

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                torch.cuda.empty_cache()
            del (
                query_responses,
                responses,
                postprocessed_responses,
                logprobs,
                values,
                sequence_lengths,
                contain_eos_token,
                sequence_lengths_p1,
                response_idxs,
                padding_mask,
                padding_mask_p1,
                rewards,
                actual_start,
                actual_end,
                advantages,
                returns,
            )

            # copy 2
            del (
                query_responses_2,
                responses_2,
                postprocessed_responses_2,
                logprobs_2,
                values_2,
                sequence_lengths_2,
                contain_eos_token_2,
                sequence_lengths_p1_2,
                response_idxs_2,
                padding_mask_2,
                padding_mask_p1_2,
                rewards_2,
                actual_start_2,
                actual_end_2,
                advantages_2,
                returns_2,
            )
            
            torch.cuda.empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        # copy 2
        self.control_2 = self.callback_handler_2.on_train_end(args, self.state_2, self.control_2)
        if self.control_2.should_save:
            self._save_checkpoint(model_2, trial=None, metrics=None)
            self.control_2 = self.callback_handler_2.on_save(self.args, self.state_2, self.control_2)

    def generate_completions(self, sampling: bool = False):
        args = self.args
        processing_class = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            for batch in self.eval_dataloader:
                query = batch["input_ids"]
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _ = batch_generation(
                        unwrapped_model.policy,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
                        )
                    table["query"].extend(
                        gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                    )
                    table["model response"].extend(
                        gather_object(processing_class.batch_decode(postprocessed_response))
                    )

                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    _, score, _ = get_reward(
                        self.reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                    )
                    table["score"].extend(self.accelerator.gather_for_metrics(score).float().cpu().numpy())

                if sampling:
                    break
        
        # copy 2
        table_2 = defaultdict(list)
        with unwrap_model_for_generation(
            self.model_2, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model_2:
            for batch in self.eval_dataloader:
                query_2 = batch["input_ids"] # edit later
                with torch.no_grad():
                    context_length_2 = query_2.shape[1]
                    query_response_2, _ = batch_generation(
                        unwrapped_model_2.policy,
                        query_2,
                        query_2.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    response_2 = query_response_2[:, context_length_2:]
                    postprocessed_response_2 = response_2
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response_2 = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response_2
                        )
                    table_2["query_2"].extend(
                        gather_object(processing_class.batch_decode(query_2, skip_special_tokens=True))
                    )
                    table_2["model response_2"].extend(
                        gather_object(processing_class.batch_decode(postprocessed_response_2))
                    )

                    postprocessed_query_response_2 = torch.cat((query_2, postprocessed_response_2), 1)
                    _, score_2, _ = get_reward(
                        self.reward_model, postprocessed_query_response_2, processing_class.pad_token_id, context_length_2
                    )
                    table_2["score_2"].extend(self.accelerator.gather_for_metrics(score_2).float().cpu().numpy())

                if sampling:
                    break

        df = pd.DataFrame(table)
        df_2 = pd.DataFrame(table_2)

        if self.accelerator.is_main_process:
            print_rich_table(df.iloc[0 : 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(
                    name="completions.csv",
                    table=df,
                )
            
            # copy 2
            print_rich_table(df_2.iloc[0 : 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions_2": wandb.Table(dataframe=df_2)})

            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(
                    name="completions_2.csv",
                    table=df_2,
                )

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent("""\
        @article{mziegler2019fine-tuning,
            title        = {{Fine-Tuning Language Models from Human Preferences}},
            author       = {Daniel M. Ziegler and Nisan Stiennon and Jeffrey Wu and Tom B. Brown and Alec Radford and Dario Amodei and Paul F. Christiano and Geoffrey Irving},
            year         = 2019,
            eprint       = {arXiv:1909.08593}
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="PPO",
            trainer_citation=citation,
            paper_title="Fine-Tuning Language Models from Human Preferences",
            paper_id="1909.08593",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
