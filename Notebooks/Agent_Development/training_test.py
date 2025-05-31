# %%
import re
import torch
import torch.nn as nn
#import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from trl import PPOConfig#, PPOTrainer
#import luxai_s3
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
#from luxai_s3.params import EnvParams
import numpy as np
from datasets import load_dataset, Dataset
#from peft import LoraConfig, get_peft_model
import os
#from accelerate import infer_auto_device_map
import gc
#import copy
gc.enable()

#from stable_baselines3 import PPO
#import gymnasium as gym
#import gym

# %%
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["FLASH_ATTENTION"] = "1"
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.cache_size_limit = 64
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
np.set_printoptions(linewidth=200)
# Configure CUDA memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False

# Enable gradient checkpointing
os.environ["PYTORCH_ATTENTION_USE_MEMORY_EFFICIENT_ATTENTION"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%
# Load and prep dataset

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# %%
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

#dataset = get_gsm8k_questions()

# %%
policy_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# ✅ Load tokenizer
policy_tokenizer = AutoTokenizer.from_pretrained(policy_model_name)

# ✅ Ensure pad token is set correctly
policy_tokenizer.pad_token = policy_tokenizer.eos_token

value_model_name = "Qwen/Qwen2-0.5B"

# ✅ Load tokenizer
value_tokenizer = AutoTokenizer.from_pretrained(value_model_name)

# ✅ Ensure pad token is set correctly
value_tokenizer.pad_token = value_tokenizer.eos_token

# ✅ Optimized quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,  # ✅ Add nested quantization for better memory usage
    bnb_4bit_quant_storage="bfloat16"  # Enable quantized storage
)

# %%
policy_autoconfig = AutoConfig.from_pretrained(policy_model_name)
policy_autoconfig.max_position_embeddings = 11000
policy_autoconfig.use_cache = False

# %%
value_autoconfig = AutoConfig.from_pretrained(value_model_name)
value_autoconfig.max_position_embeddings = 11000
value_autoconfig.use_cache = False

# %%
def create_policy_model():
    model = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        trust_remote_code=True,
        device_map="auto",  # Let Accelerate handle device placement
        #device_map={"0": "14GiB", "cpu": "64GiB"},  # Let Accelerate handle device placement
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        config=policy_autoconfig,
        attn_implementation="flash_attention_2",
        #use_cache=False,  # Disable KV cache during training
        low_cpu_mem_usage=True
    )

    # Enable memory efficient features
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    return model

# %%
def create_value_model():
    model = AutoModelForCausalLM.from_pretrained(
        value_model_name,
        trust_remote_code=True,
        device_map="auto",  # Let Accelerate handle device placement
        #device_map={"0": "14GiB", "cpu": "64GiB"},  # Let Accelerate handle device placement
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        config=value_autoconfig,
        attn_implementation="flash_attention_2",
        #use_cache=False,  # Disable KV cache during training
        low_cpu_mem_usage=True
    )

    # Enable memory efficient features
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    return model

# %%
env = RecordEpisode(
    LuxAIS3GymEnv(numpy_output=True)
)

# %%
# Reward functions
def strict_format_reward_func(completion) -> float:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<answer>\n.*?\n</answer>\n$"
    match = re.match(pattern, completion)

    return 0.5 if match else 0.0

def soft_format_reward_func(completion) -> float:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<answer>.*?</answer>"
    match = re.match(pattern, completion)

    return 0.5 if match else 0.0

def count_xml(text) -> float:
    count = 0.0
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001

    return count

def xmlcount_reward_func(completion) -> float:

    return count_xml(completion)

def answer_format_reward_func(completion) -> float:
    r"""
    Computes a reward based on whether the answer text (extracted from the completion)
    follows the required format:
    
    Expected format (one line per unit, 16 total):
      Unit 0: 1
      Unit 1: 2
      Unit 2: 5, 2, 2
      Unit 3: 0
      Unit 4: 5, 1, 1
      Unit 5: 5, -1, -2
      Unit 6: 5, -2, 2
      Unit 7: 5, 0, 0
      Unit 8: 4
      Unit 9: 0
      Unit 10: 3
      Unit 11: 2
      Unit 12: 1
      Unit 13: 0
      Unit 14: 5, -4, 5
      Unit 15: 5, 3, -3

    Each line is expected to match:
      ^Unit\s+([0-9]+):\s+((?:[0-4])|(?:5,\s*-?\d+,\s*-?\d+))$
    
    Returns:
      A list of scores (floats), one per completion.
    """
    # extract_xml_answer should extract the text between the <answer> tags.
    answer = extract_xml_answer(completion)
    
    # Updated regex pattern:
    answer_pattern = re.compile(
        r"^Unit\s+([0-9]+):\s+((?:[0-4])|(?:5,\s*-?\d+,\s*-?\d+))$"
    )

    answer_score = 0.0
    # Split the answer into lines and remove any extra whitespace
    lines = [line.strip() for line in answer.strip().split("\n") if line.strip()]
    # Penalize if we do not have exactly 16 lines (one per unit)
    if len(lines) != 16:
        answer_score -= 0.2  # adjust penalty as desired
    
    for line in lines:
        match = answer_pattern.match(line)
        if match:
            # Reward for a valid line
            answer_score += 0.5 / 16
            unit_number = int(match.group(1))
            # Ensure unit numbers are within the valid range (0 to 15)
            if unit_number < 0 or unit_number > 15:
                answer_score -= 0.1 / 16
        else:
            # Penalize for any line that doesn't match the required format
            answer_score -= 0.1

    return answer_score

def point_gain_reward_func(reward_score) -> float:

    return reward_score

def match_won_reward_func(match_won) -> float:

    return 100.0 if match_won else 0.0

def match_lost_reward_func(match_lost) -> float:

    return -10.0 if match_lost else 0.0

def game_won_reward_func(game_won) -> float:

    return 1000.0 if game_won else 0.0

def game_lost_reward_func(game_lost) -> float:

    return -100.0 if game_lost else 0.0

# %%
num_games_to_train = 5

# %%
output_dir="outputs/DeepSeek-R1-Distill-Qwen-1.5B-PPO"
run_name="DeepSeek-R1-Distill-Qwen-1.5B-PPO-20250217_01"

training_args = PPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    batch_size=1,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    bf16=True,
    gradient_accumulation_steps=16,
    num_sample_generations=0,
    max_grad_norm=0.1,
    num_train_epochs=1,
    save_steps=100,
    log_on_each_node=False,
    report_to="none",
    num_ppo_epochs=1,
    cliprange=0.2,
    vf_coef=1.0,
    kl_coef=0.01,
    prediction_loss_only=True,
    gradient_checkpointing=True,
    #reward_model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    optim="adamw_torch_fused",
    #use_cpu=True,
    max_steps=1,
    #eval_steps=1,
    #eval_accumulation_steps=8,
    #accelerator_config={"num_processes": 8},
    per_device_train_batch_size=1,
    #per_device_eval_batch_size=1,
    torch_empty_cache_steps=1,
    #torch_compile=True,
    #torch_compile_mode="default",
    total_episodes=num_games_to_train,
    micro_batch_size=1,
    mini_batch_size=1,
    local_batch_size=1,
    response_length=230,
    ds3_gather_for_generation=False,
)

# %%
training_args.num_mini_batches

# %%
training_args.mini_batch_size

# %%
policy_model_1 = create_policy_model()
policy_model_2 = create_policy_model()
value_model_1 = create_value_model()
value_model_2 = create_value_model()

# %%
class ValueHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        # A linear layer that maps the hidden state (for each token) to a scalar value.
        self.linear = nn.Linear(hidden_size, 1).to('cuda', dtype=torch.bfloat16)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Convert hidden state representations to per-token scalar value estimates.

        Args:
            hidden_states: Tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            Tensor of shape (batch_size, sequence_length) containing scalar value estimates.
        """
        # Apply the linear layer token-wise. The output will have shape (batch_size, sequence_length, 1)
        value = self.linear(hidden_states)
        # Remove the last dimension to have shape (batch_size, sequence_length)
        value = value.squeeze(-1)
        
        if torch.isnan(value).any() or torch.isinf(value).any():
            raise ValueError("Value head produced NaN or Inf values!")
        return value

# %%
value_model_1.config.hidden_size

# %%
value_model_1.score = ValueHead(value_model_1.config.hidden_size)
value_model_2.score = ValueHead(value_model_2.config.hidden_size)

# %%
from Modified_PPO_Trainer.ppo_trainer_20250218_01 import ModifiedPPOTrainer

# %%
trainer = ModifiedPPOTrainer(
    model=policy_model_1,
    model_2=policy_model_2,
    value_model=value_model_1,
    value_model_2=value_model_2,
    processing_class=policy_tokenizer,
    args=training_args,
    reward_functions=[
        strict_format_reward_func,
        soft_format_reward_func,
        xmlcount_reward_func,
        answer_format_reward_func,
        point_gain_reward_func,
        match_won_reward_func,
        match_lost_reward_func,
        game_won_reward_func,
        game_lost_reward_func
    ],
    game_env=env,
    num_games_to_train=num_games_to_train
)

# %%
trainer.train()


