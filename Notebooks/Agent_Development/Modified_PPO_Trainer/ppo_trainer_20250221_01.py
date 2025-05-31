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
import time
from collections import defaultdict
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
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
from transformers.utils.deprecation import deprecate_kwarg

from trl.core import masked_mean, masked_whiten
from trl.trainer.ppo_config import PPOConfig
from trl.trainer.utils import (
    OnlineTrainerState,
    disable_dropout_in_model,
    first_true_indices,
    truncate_response,
)
import re
import json

if is_wandb_available():
    import wandb


INVALID_LOGPROB = 1.0

game_rules = {'content': "\nYour job is to play a game according to the given rules and game state information (observation).\nYou can play the game by choosing an action for each unit you have at each step.\nmax_new_tokens is only 256, so you must provide unit actions according to the below Action Format section within the token limit, or else you won't be able to provide actions for units at all.\nHere is the game rules in JSON format:\n" + \
json.dumps(
{
    "Game Objective": [
        "Two teams compete in a best-of-5 match sequence (called a game).",
        "Each match lasts 100 time steps.",
        "Teams control units to gain relic points on the map while preventing the opposing team from doing the same.",
        "The team with the most relic points at the end of a match wins that match.",
        "The team that wins the most matches wins the game.",
        "Strategy: Explore more in early matches to learn the map and opponent behavior, then exploit this knowledge to win later matches."
    ],
    "Map Features": {
        "description": "The map is a 24x24 2D grid, randomly generated but consistent across matches in a game (no full regeneration between matches).",
        "Key map features": {
            "Unknown Tiles": [
                "Not visible until a unit is within sensor range (randomized 2-4 tiles).",
                "Can be any type of tile (empty, asteroid, nebula, etc.).",
                "Represented as -1 in map tile types."
            ],
            "Empty Tiles": [
                "Units can move onto these tiles.",
                "Represented as 0 in map tile types.",
                "Represented as 0 in map tile types."
            ],
            "Asteroid Tiles": [
                "Impassable; units cannot move or spawn on them.",
                "May move symmetrically over time.",
                "If an asteroid moves onto a unit, the unit survives and can still act if an adjacent non-asteroid tile is available.",
                "Represented as 1 in map tile types."
            ],
            "Nebula Tiles": {
                "description": "Passable but affect units.",
                "Vision Reduction": [
                    "Reduces unit vision (randomized 0-3).",
                    "Units on nebula tiles may not see themselves or other tiles if vision reduction is strong."
                ],
                "Energy Reduction": "Reduces unit energy (randomized 0, 10, 25).",
                "additional": [
                    "May move symmetrically over time.",
                    "Represented as 2 in map tile types."
                ]
            },
            "Energy Tiles": [
                "Emit energy fields that units can harvest.",
                "Positions and energy values change over time (symmetric movement).",
                "Energy value at a tile is calculated based on the distance to energy tiles using randomized functions.",
                "Not represented in map tile types; energy values are provided in observations."
            ],
            "Relic Nodes": [
                "Units near relic nodes gain relic points for their team.",
                "Only specific tiles near relic nodes yield points (hidden, discovered through trial and error).",
                "Relic node positions are visible within sensor range, but point-yielding tiles are not.",
                "Multiple units on a tile yield at most 1 point per tile (stacking units is risky due to sap actions).",
                "Point-yielding tiles are defined by a random 5x5 mask centered on the relic node.",
                "Not represented in map tile types; relic node positions are provided in observations."
            ]
        }
    },
    "Units": {
        "description": [
            "Each team has up to 16 units (IDs 0-15).",
            "Units start with 100 energy (max 400).",
            "Units spawn in one of the map corners based on their team."
        ],
        "Energy": [
            "Recharged by energy tiles.",
            "Reduced by nebula tiles (cannot go below 0 from map features).",
            "Opposing units can reduce energy below 0, causing the unit to be removed and respawn later."
        ],
        "Actions": {
            "description": "Units can take 1 action per timestep.",
            "Actions": {
                "Move Actions (6 options)": [
                    "0: Center (no movement, free).",
                    "1: Up.",
                    "2: Right.",
                    "3: Down.",
                    "4: Left.",
                    "5: Sap (ranged attack, see below)."
                ],
                "movementNotes": [
                    "Moving (except center) costs energy (randomized 1-5).",
                    "Cannot move onto asteroid tiles or off the map (energy is still consumed if attempted)."
                ]
            },
            "Sap Action (5)": [
                "Ranged attack (range 3-7 tiles, randomized).",
                "Targets a tile (dx, dy relative to unit position, within sap range).",
                "Reduces energy of enemy units on the target tile by a sap cost (randomized 30-50).",
                "Reduces energy of enemy units on adjacent tiles by sap cost * dropoff factor (0.25, 0.5, or 1).",
                "Costs the unit energy equal to the sap cost.",
                "Risky: Missing wastes energy, but effective against stacked enemy units."
            ]
        }
    },
    "Vision": {
        "description": "Team vision is the combined vision of all units, represented as a boolean mask over the map.",
        "sensorRange": "Each unit has a sensor range (randomized 2-4 tiles).",
        "Vision Power Calculation": {
            "calculation": "For each unit, compute vision power for tiles within sensor range: Vision power = 1 + sensor_range - min(dx, dy) for each tile (x+dx, y+dy).",
            "nebulaEffect": "Nebula tiles reduce vision power by their vision reduction value (0-3).",
            "lowVision": "If vision power is too low, units cannot see tile details.",
            "overlapBonus": "Overlapping unit vision increases vision power linearly, helping to see through nebula tiles."
        },
        "nebulaVisibility": "Units on nebula tiles may not see themselves or other tiles if vision reduction is strong, but can see beyond nebula tiles if vision power is sufficient."
    },
    "Collisions and Energy Void Fields": {
        "Collisions": [
            "If units from opposing teams occupy the same tile at the end of a turn, the team with higher total energy survives; the other team's units are removed.",
            "If tied, all units on the tile are removed."
        ],
        "Energy Void Fields": [
            "Each unit generates an energy void field affecting adjacent enemy units (up, right, down, left).",
            "Energy void strength = unit energy * void factor (randomized 0.0625, 0.125, 0.25, 0.375).",
            "Enemy units on a tile lose energy equal to the void strength divided by the number of units on that tile.",
            "Removed units (from collisions) do not contribute to void fields.",
            "Encourages stacking units to reduce void field impact."
        ]
    },
    "Win Conditions": {
        "Match Win": [
            "Team with the most relic points at the end of 100 time steps wins.",
            "If tied, team with more total unit energy wins.",
            "If still tied, winner is chosen randomly."
        ],
        "Game Win": "Team that wins the most matches (best-of-5) wins the game."
    },
    "Match Resolution Order": {
        "steps": [
            "Move units with enough energy.",
            "Execute sap actions for units with enough energy.",
            "Resolve collisions and apply energy void fields.",
            "Update unit energy based on map position (energy fields, nebula tiles).",
            "Spawn units; remove units with energy < 0.",
            "Compute team vision masks and mask observations.",
            "Move map objects (asteroids, nebula tiles, energy tiles).",
            "Compute new team relic points."
        ],
        "notes": "Matches run for 100 steps, but 101 frames of observations are provided (first frame is empty or previous match's final state; actions on it are ignored)."
    },
    "Action Format": {
        "description": "Units take 1 action per timestep.",
        "submissionFormat": "<answer>\nUnit 0: action (0-5), dx, dy (if sap, else 0, 0)\nUnit 1: action (0-5), dx, dy (if sap, else 0, 0)\n... \nUnit 15: action (0-5), dx, dy (if sap, else 0, 0)\n</answer>",
        "instructions": [
            "If action is sap (5), provide dx, dy (relative coordinates within sap range).",
            "Only take actions for available units; invalid actions are ignored."
        ],
        "example": "<answer>\nUnit 0: 1, 0, 0\nUnit 1: 2, 0, 0\nUnit 2: 5, 2, 2\n... \nUnit 15: 5, 3, -3\n</answer>"
    },
    "Game Parameters": {
        "description": "Many parameters are randomized and not revealed to teams (discovered through exploration).",
        "Key randomized parameters": {
            "Unit move cost": "1-5",
            "Unit sensor range": "2-4",
            "Nebula vision reduction": "0-3",
            "Nebula energy reduction": "0, 0, 10, 25",
            "Sap cost": "30-50",
            "Sap range": "3-7",
            "Sap dropoff factor": "0.25, 0.5, 1",
            "Energy void factor": "0.0625, 0.125, 0.25, 0.375",
            "Map drift": "Map drift speeds and magnitudes for nebula, energy tiles, and asteroid."
        },
        "consistency": "Parameters remain consistent within a game (across matches)."
    }
})+"\n", 'role': 'system'}


def default_converter(o):
    # Convert numpy scalars to native Python types.
    if isinstance(o, np.generic):
        return o.item()
    # If it's an ndarray, convert it to a list.
    if isinstance(o, np.ndarray):
        return o.tolist()
    # Otherwise, raise a TypeError.
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

def prep_llm_input(env_cfg, obs, team_id, enemy_team_id, enemy_spawn_location, map_explored_status):
    # Build your structured dictionary (data) as before.
    data = {
        "Game state information": {
            "Environment configuration": {
                "Maximum number of units": env_cfg['max_units'],
                "Total number of matches": env_cfg['match_count_per_episode'],
                "Maximum number of steps per match": env_cfg['max_steps_in_match'],
                "Map height": env_cfg['map_height'],
                "Map width": env_cfg['map_width'],
                "Number of teams in this game": env_cfg['num_teams'],
                "Unit Movement cost": env_cfg['unit_move_cost'],
                "Unit sap action cost": env_cfg['unit_sap_cost'],
                "Unit sap action maximum range": env_cfg['unit_sap_range'],
                "Unit sensor (vision) range": env_cfg['unit_sensor_range']
            },
            "Observation": {
                "Unit position warning": "-1, -1 means the unit is not spawned yet, got destroyed, or not visible.",
                "Unit positions": {
                    "My units": [],
                    "Enemy units": []
                },
                "Unit energies": {
                    "My units": [],
                    "Enemy units": []
                },
                "Unit visibility": {
                    "My units": [],
                    "Enemy units": []
                },
                "Sensor (vision) mask": [],
                "Map explored status": [],
                "Map features": {
                    "Energy": [],
                    "Tile type": []
                },
                "Relic nodes": {
                    "Positions": [],
                    "Warning": "-1, -1 means the relic node is not yet discovered.",
                    "Visibility": []
                },
                "Points": {
                    "My points": obs['team_points'][team_id],
                    "Enemy's points": obs['team_points'][enemy_team_id]
                },
                "Match wins": {
                    "My number of match wins": obs['team_wins'][team_id],
                    "Enemy's number of match wins": obs['team_wins'][enemy_team_id]
                },
                "Current step": obs['steps'],
                "Current match step": obs['match_steps'],
                "Enemy spawn location": None
            }
        }
    }
    
    # Populate my team unit positions
    obs_my_unit_positions = obs['units']['position'][team_id]
    for i in range(obs_my_unit_positions.shape[0]):
        pos = obs_my_unit_positions[i]
        data["Game state information"]["Observation"]["Unit positions"]["My units"].append({
            "Unit": i,
            "Position": [pos[0], pos[1]]
        })
    
    # Populate enemy team unit positions
    obs_enemy_unit_positions = obs['units']['position'][enemy_team_id]
    for i in range(obs_enemy_unit_positions.shape[0]):
        pos = obs_enemy_unit_positions[i]
        data["Game state information"]["Observation"]["Unit positions"]["Enemy units"].append({
            "Unit": i,
            "Position": [pos[0], pos[1]]
        })
    
    # Populate unit energies
    obs_my_unit_energys = obs['units']['energy'][team_id]
    for i in range(obs_my_unit_energys.shape[0]):
        energy = obs_my_unit_energys[i]
        data["Game state information"]["Observation"]["Unit energies"]["My units"].append({
            "Unit": i,
            "Energy": energy
        })
    
    obs_enemy_unit_energys = obs['units']['energy'][enemy_team_id]
    for i in range(obs_enemy_unit_energys.shape[0]):
        energy = obs_enemy_unit_energys[i]
        data["Game state information"]["Observation"]["Unit energies"]["Enemy units"].append({
            "Unit": i,
            "Energy": energy
        })
    
    # Populate unit visibility (masks)
    obs_my_units_mask = obs['units_mask'][team_id]
    for i in range(obs_my_units_mask.shape[0]):
        mask = obs_my_units_mask[i]
        data["Game state information"]["Observation"]["Unit visibility"]["My units"].append({
            "Unit": i,
            "Visibility": mask.tolist() if hasattr(mask, "tolist") else mask
        })
    
    obs_enemy_units_mask = obs['units_mask'][enemy_team_id]
    for i in range(obs_enemy_units_mask.shape[0]):
        mask = obs_enemy_units_mask[i]
        data["Game state information"]["Observation"]["Unit visibility"]["Enemy units"].append({
            "Unit": i,
            "Visibility": mask.tolist() if hasattr(mask, "tolist") else mask
        })
    
    # Populate sensor mask rows
    sensor_mask_T = obs['sensor_mask'].T
    for i in range(sensor_mask_T.shape[0]):
        row = sensor_mask_T[i]
        data["Game state information"]["Observation"]["Sensor (vision) mask"].append(
            row.tolist() if hasattr(row, "tolist") else row
        )
    
    # Populate map explored status rows
    for i in range(map_explored_status.shape[0]):
        row = map_explored_status[i]
        data["Game state information"]["Observation"]["Map explored status"].append(
            row.tolist() if hasattr(row, "tolist") else row
        )
    
    # Populate map features - energy
    obs_map_features_energy = obs['map_features']['energy'].T
    for i in range(obs_map_features_energy.shape[0]):
        row = obs_map_features_energy[i]
        data["Game state information"]["Observation"]["Map features"]["Energy"].append(
            row.tolist() if hasattr(row, "tolist") else row
        )
    
    # Populate map features - tile types
    obs_map_features_tile_type = obs['map_features']['tile_type'].T
    for i in range(obs_map_features_tile_type.shape[0]):
        row = obs_map_features_tile_type[i]
        data["Game state information"]["Observation"]["Map features"]["Tile type"].append(
            row.tolist() if hasattr(row, "tolist") else row
        )
    
    # Populate relic node positions
    obs_relic_nodes = obs['relic_nodes']
    for i in range(obs_relic_nodes.shape[0]):
        pos = obs_relic_nodes[i]
        data["Game state information"]["Observation"]["Relic nodes"]["Positions"].append({
            "Node": i,
            "Position": [pos[0], pos[1]]
        })
    
    # Populate relic node masks
    obs_relic_nodes_mask = obs['relic_nodes_mask']
    for i in range(obs_relic_nodes_mask.shape[0]):
        mask = obs_relic_nodes_mask[i]
        data["Game state information"]["Observation"]["Relic nodes"]["Visibility"].append({
            "Node": i,
            "Visibility": mask.tolist() if hasattr(mask, "tolist") else mask
        })
    
    # Set enemy spawn location info
    if enemy_spawn_location is None:
        data["Game state information"]["Observation"]["Enemy spawn location"] = "not yet discovered"
    else:
        data["Game state information"]["Observation"]["Enemy spawn location"] = [enemy_spawn_location[0], enemy_spawn_location[1]]
    
    # Convert the dictionary to a JSON string using the custom default converter.
    return {
        'content': "\nHere is the game state information (observation) in JSON format:\n" + json.dumps(data, default=default_converter),
        'role': 'user'
    }


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
        model_1: nn.Module,
        model_2: nn.Module,
        reward_functions: list,
        game_env = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        num_games_to_train: int = 1,
    ) -> None:

        self.args = args
        self.processing_class = processing_class
        self.model_1 = model_1
        self.model_2 = model_2
        self.num_games_to_train = num_games_to_train
        self.game_env = game_env
        self.reward_functions = reward_functions
        self._metrics = defaultdict(list)
        self.is_fsdp_enabled = False
        self.is_deepspeed_enabled = False

        # Define the collator if not provided
        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)

        # Handle stop token settings: update policy model's generation_config to use provided stop token
        if self.args.stop_token and self.args.stop_token_id:
            raise ValueError("You cannot set both `stop_token` and `stop_token_id`.")
        elif self.args.stop_token:
            if self.args.stop_token == "eos":
                self.model_1.base.generation_config.eos_token_id = self.stop_token_id = processing_class.eos_token_id
                self.model_2.base.generation_config.eos_token_id = self.stop_token_id = processing_class.eos_token_id
            else:
                raise ValueError(
                    f"Unknown `stop_token` {self.args.stop_token}. Allowed values are: `'eos'` and `None` (no stop token)."
                )
        else:
            self.model_1.base.generation_config.eos_token_id = self.stop_token_id = self.args.stop_token_id  # None or int
            self.model_2.base.generation_config.eos_token_id = self.stop_token_id = self.args.stop_token_id  # None or int

        self.model_adapter_name = self.args.model_adapter_name
        self.ref_adapter_name = self.args.ref_adapter_name

        self.data_collator = data_collator
        self.optimizer, self.lr_scheduler = (None, None)
        self.optimizer_2, self.lr_scheduler_2 = (None, None)
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47
        plugin = GradientAccumulationPlugin(sync_with_dataloader=False, num_steps=self.args.gradient_accumulation_steps, sync_each_batch=True, adjust_scheduler=True)
        accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_plugin=plugin)
        accelerator_2 = Accelerator(mixed_precision="bf16", gradient_accumulation_plugin=plugin)
        self.args.world_size = accelerator.num_processes
        self.args.num_total_batches = num_games_to_train * 505

        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = time_tensor.item()  # avoid different timestamps across processes
        self.args.run_name = f"{self.args.exp_name}__{self.args.seed}__{time_int}"
        self.local_seed = self.args.seed + accelerator.process_index * 100003  # Prime
        if self.args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, self.args.num_total_batches // self.args.num_sample_generations)

        #########
        # setup model, optimizer, and others
        #########
        for module in [self.model_1, self.model_2]:
            if module is not None:
                disable_dropout_in_model(module)
        self.create_optimizer_and_scheduler(
            num_training_steps=self.args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model_1, self.processing_class, self.optimizer, self.lr_scheduler
        )
        # copy 2
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
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model_1, "add_model_tags"):
            self.model_1.add_model_tags(self._tag_names)
        # copy 2
        if hasattr(self.model_2, "add_model_tags"):
            self.model_2.add_model_tags(self._tag_names)

        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(self.args.seed)
        self.model_1, self.optimizer, self.lr_scheduler = accelerator.prepare(self.model_1, self.optimizer, self.lr_scheduler)
        self.model_2, self.optimizer_2, self.lr_scheduler_2 = accelerator_2.prepare(self.model_2, self.optimizer_2, self.lr_scheduler_2) # edit it later
        self.model_1 = torch.compile(self.model_1)
        self.model_2 = torch.compile(self.model_2)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.device = accelerator.device
        self.accelerator = accelerator
        self.accelerator_2 = accelerator_2

        self.generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            min_new_tokens=222,
            temperature=(self.args.temperature + 1e-7),
            top_k=20,
            top_p=0.95,
            do_sample=True,
            eos_token_id=self.processing_class.eos_token_id,  # Ensure EOS token stops generation
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=False,
        )

        print("-" * 10 + " Optimizer")
        print(self.optimizer)
        print(self.optimizer_2)
        print("-" * 10 + " LR Scheduler")
        print(self.lr_scheduler)
        print(self.lr_scheduler_2)

    def train(self):
        env = self.game_env
        args = self.args

        self.accelerator.print("===training policy===")
        start_time = time.time()
    
        self.model_1.train()
        self.model_2.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches * args.num_mini_batches
        # copy 2
        self.state_2.global_step = 0
        self.state_2.episode = 0
        self.state_2.max_steps = args.num_total_batches * args.num_mini_batches
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

        
        for game_number in range(1, self.num_games_to_train + 1):
            print(f"Game number: {game_number}")
            self.state.episode += 1 * args.batch_size

            # swap models
            self.model_1, self.model_2 = self.model_2, self.model_1
            self.optimizer, self.optimizer_2 = self.optimizer_2, self.optimizer
            self.lr_scheduler, self.lr_scheduler_2 = self.lr_scheduler_2, self.lr_scheduler
            self.accelerator, self.accelerator_2 = self.accelerator_2, self.accelerator

            obs_all, info = env.reset()
            env_cfg = info['params']

            self.sap_range = env_cfg["unit_sap_range"]

            player_0_previous_score = 0.0
            player_1_previous_score = 0.0

            match_number = 1

            first_spawn = False

            player_0_spawn_location = None
            player_1_spawn_location = None

            player_0_map_explored_status = np.zeros((env_cfg["map_height"], env_cfg["map_width"]), dtype=bool)
            player_1_map_explored_status = np.zeros((env_cfg["map_height"], env_cfg["map_width"]), dtype=bool)

            game_ended = False

            player_0_match_won_num = 0
            player_1_match_won_num = 0

            while game_ended is not True:

                player_0_current_score = obs_all['player_0']['team_points'][0]
                player_1_current_score = obs_all['player_1']['team_points'][1]

                player_0_reward_score = player_0_current_score - player_0_previous_score
                player_1_reward_score = player_1_current_score - player_1_previous_score

                player_0_previous_score = player_0_current_score
                player_1_previous_score = player_1_current_score

                current_match_step = obs_all["player_0"]["match_steps"]

                player_0_match_won = False
                player_0_match_lost = False
                player_1_match_won = False
                player_1_match_lost = False

                player_0_game_won = False
                player_0_game_lost = False
                player_1_game_won = False
                player_1_game_lost = False

                if current_match_step == 100:
                    if player_0_current_score > player_1_current_score:
                        player_0_match_won = True
                        player_1_match_lost = True
                        player_0_match_won_num += 1
                    elif player_0_current_score < player_1_current_score:
                        player_0_match_lost = True
                        player_1_match_won = True
                        player_1_match_won_num += 1

                if player_0_match_won_num >= 3:
                    game_ended = True
                    print("Player 0 won the game.")
                    break

                if player_1_match_won_num >= 3:
                    game_ended = True
                    print("Player 1 won the game.")
                    break

                player_0_unit_positions = np.array(obs_all['player_0']["units"]["position"][0])
                player_1_unit_positions = np.array(obs_all['player_1']["units"]["position"][1])

                player_0_unit_mask = np.array(obs_all['player_0']["units_mask"][0])
                player_1_unit_mask = np.array(obs_all['player_1']["units_mask"][1])

                player_0_map_features = obs_all['player_0']['map_features']
                player_1_map_features = obs_all['player_1']['map_features']

                player_0_current_map_tile_type = player_0_map_features['tile_type'].T
                player_1_current_map_tile_type = player_1_map_features['tile_type'].T

                player_0_map_explored_status[player_0_current_map_tile_type != -1] = True
                player_1_map_explored_status[player_1_current_map_tile_type != -1] = True

                player_0_available_unit_ids = np.where(player_0_unit_mask)[0]
                player_1_available_unit_ids = np.where(player_1_unit_mask)[0]

                if player_0_available_unit_ids.shape[0] == 0:
                    pass
                else:
                    if first_spawn == False:
                        player_0_first_unit_id = player_0_available_unit_ids[0]
                        player_0_first_unit_pos = player_0_unit_positions[player_0_first_unit_id]
                        player_0_spawn_location = (player_0_first_unit_pos[0], player_0_first_unit_pos[1])
                        player_1_first_unit_id = player_1_available_unit_ids[0]
                        player_1_first_unit_pos = player_1_unit_positions[player_1_first_unit_id]
                        player_1_spawn_location = (player_1_first_unit_pos[0], player_1_first_unit_pos[1])
                        first_spawn = True

                player_0_llm_input = prep_llm_input(env_cfg, obs_all['player_0'], 0, 1, player_1_spawn_location, player_0_map_explored_status)
                player_1_llm_input = prep_llm_input(env_cfg, obs_all['player_1'], 1, 0, player_0_spawn_location, player_1_map_explored_status)

                self.model_1.to(self.device)
                self.model_2.to("cpu")

                player_0_logprob, player_0_actions, player_0_sequence_length, player_0_sequence_length_p1, player_0_padding_mask_p1, player_0_score, player_0_response, player_0_value, \
                player_0_padding_mask, player_0_query_response, player_0_context_length \
                = self.response_processing(
                    self.model_1, player_0_llm_input, player_0_reward_score, player_0_match_won, player_0_match_lost, player_0_game_won, player_0_game_lost
                )

                self.model_1.to("cpu")
                self.model_2.to(self.device)

                player_1_logprob, player_1_actions, player_1_sequence_length, player_1_sequence_length_p1, player_1_padding_mask_p1, player_1_score, player_1_response, player_1_value, \
                player_1_padding_mask, player_1_query_response, player_1_context_length \
                = self.response_processing(
                    self.model_2, player_1_llm_input, player_1_reward_score, player_1_match_won, player_1_match_lost, player_1_game_won, player_1_game_lost
                )

                self.model_1.to(self.device)
                self.model_2.to("cpu")
                player_1_query_response = player_1_query_response.to("cpu")

                torch.cuda.empty_cache()
                gc.collect()

                self.train_step(
                    self.model_1, self.optimizer, player_0_logprob, player_1_logprob, player_0_sequence_length, player_0_sequence_length_p1, player_0_padding_mask_p1, player_0_score,
                    player_0_response, player_0_value, player_0_padding_mask, player_0_query_response, player_0_context_length, self.accelerator
                )

                del (
                    player_0_score,
                    player_0_query_response,
                    player_0_response,
                    player_0_value,
                    player_0_sequence_length,
                    player_0_sequence_length_p1,
                    player_0_padding_mask,
                    player_0_padding_mask_p1,
                )


                self.model_1.to("cpu")
                self.model_2.to(self.device)
                player_1_query_response = player_1_query_response.to(self.device)

                torch.cuda.empty_cache()
                gc.collect()

                self.train_step(
                    self.model_2, self.optimizer_2, player_1_logprob, player_0_logprob, player_1_sequence_length, player_1_sequence_length_p1, player_1_padding_mask_p1, player_1_score,
                    player_1_response, player_1_value, player_1_padding_mask, player_1_query_response, player_1_context_length, self.accelerator_2
                )

                # save checkpoint
                self.lr_scheduler.step()
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                if obs_all["player_0"]["steps"] % self.args.save_steps == 0:
                    torch.save(self.model_1.state_dict(), self.args.output_dir + f"/model_1_weights_game_{game_number}_step_{obs_all["player_0"]["steps"]}.pth")

                # copy 2
                self.lr_scheduler_2.step()
                self.control_2 = self.callback_handler_2.on_step_end(args, self.state_2, self.control_2)
                if obs_all["player_0"]["steps"] % self.args.save_steps == 0:
                    torch.save(self.model_2.state_dict(), self.args.output_dir + f"/model_2_weights_game_{game_number}_step_{obs_all["player_0"]["steps"]}.pth")
                
                if args.num_sample_generations > 0 and (game_number - 1) % self.sample_generations_freq == 0:
                    self.generate_completions(sampling=True)
                    # torch.cuda.empty_cache()

                del (
                    player_0_logprob,
                )

                # copy 2
                del (
                    player_1_score,
                    player_1_query_response,
                    player_1_response,
                    player_1_logprob,
                    player_1_value,
                    player_1_sequence_length,
                    player_1_sequence_length_p1,
                    player_1_padding_mask,
                    player_1_padding_mask_p1,
                )

                if match_number >= 5 and current_match_step == 100:
                    game_ended = True
                    print("Game ended.")
                    break

                if current_match_step == 100:
                    match_number += 1

                obs_all, _, _, _, _ = env.step({
                    "player_0": player_0_actions,
                    "player_1": player_1_actions
                })

                torch.cuda.empty_cache()
                gc.collect()


        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            torch.save(self.model_1.state_dict(), self.args.output_dir + f"/model_1_weights_final_save.pth")

        # copy 2
        self.control_2 = self.callback_handler_2.on_train_end(args, self.state_2, self.control_2)
        if self.control_2.should_save:
            torch.save(self.model_2.state_dict(), self.args.output_dir + f"/model_2_weights_final_save.pth")


    def response_processing(self, model, llm_input, reward_score, match_won, match_lost, game_won, game_lost):

        chat_text = self.processing_class.apply_chat_template([game_rules, llm_input], tokenize=False)
        print("-" * 10 + " Chat text " + "-" * 10)
        print(chat_text)
        print("-" * 30)
        tokens = self.processing_class(chat_text, return_tensors="pt", max_length=False, truncation=False, padding=False, add_special_tokens=True).to(self.device)
        model.base.gradient_checkpointing_disable()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                context_length = tokens['input_ids'].shape[1]
                print("Context length:", context_length)
                print("-" * 10 + " Generate start")
                gen_output = model.base.generate(
                    **tokens,
                    generation_config=self.generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    # output_hidden_states=True,
                    pad_token_id=self.processing_class.pad_token_id,
                    # stopping_criteria=self.stopping_criteria,
                )
                print("-" * 10 + " Generate end")

                logits = torch.stack(gen_output.scores, dim=1)

                outputs = model.base(gen_output.sequences[:, context_length:], output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                value = model.value_head(hidden_states).squeeze(-1)  # (batch, gen_length)
                # value = value[:, context_length:]

                query_response = torch.cat((tokens['input_ids'], gen_output.sequences[:, context_length:]), dim=1)

                response = query_response[:, context_length:]
                all_logprob = F.log_softmax(logits, dim=-1)
                logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)

                # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                postprocessed_response = response
                if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                    postprocessed_response = truncate_response(
                        self.stop_token_id, self.processing_class.pad_token_id, response
                    )

                # Response Processing 2. run reward model on the truncated responses
                postprocessed_query_response = torch.cat((tokens["input_ids"], postprocessed_response), 1)
                sequence_length = first_true_indices(postprocessed_response == self.processing_class.pad_token_id) - 1

                completion_ids = postprocessed_query_response[:, context_length:]
                completion = self.processing_class.decode(completion_ids[0], skip_special_tokens=False)

                print("-" * 10 + " Model output " + "-" * 10)
                print(completion)
                print("-" * 30)

                func_rewards = torch.zeros(1, len(self.reward_functions), device=self.device)

                for i, reward_func in enumerate(self.reward_functions):
                    reward_func_name = reward_func.__name__
                    if reward_func_name == "point_gain_reward_func":
                        reward = reward_func(reward_score)
                    elif reward_func_name == "match_won_reward_func":
                        reward = reward_func(match_won)
                    elif reward_func_name == "match_lost_reward_func":
                        reward = reward_func(match_lost)
                    elif reward_func_name == "game_won_reward_func":
                        reward = reward_func(game_won)
                    elif reward_func_name == "game_lost_reward_func":
                        reward = reward_func(game_lost)
                    elif reward_func_name == "answer_format_reward_func":
                        reward = reward_func(completion, self.sap_range)
                    else:
                        reward = reward_func(completion)
                    func_rewards[0, i] = reward
                    # self._metrics[f"player_0_rewards/{reward_func_name}"].append(reward)

                score = func_rewards.sum(dim=1)

                actions = self.get_action_from_answer(answer=completion)

                # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                contain_eos_token = torch.any(postprocessed_response == self.processing_class.eos_token_id, dim=-1)
                if self.args.missing_eos_penalty is not None:
                    score[~contain_eos_token] -= self.args.missing_eos_penalty
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(response.shape[1], device=response.device).repeat(response.shape[0], 1)
                padding_mask = response_idxs > sequence_length.unsqueeze(1)
                logprob = torch.masked_fill(logprob, padding_mask, INVALID_LOGPROB)
                sequence_length_p1 = sequence_length + 1
                padding_mask_p1 = response_idxs > (sequence_length_p1.unsqueeze(1))
                value = torch.masked_fill(value, padding_mask_p1, 0)

        del (
            response_idxs, contain_eos_token, postprocessed_response, postprocessed_query_response, hidden_states, outputs, gen_output, tokens, logits, all_logprob,
            completion_ids, completion, func_rewards, reward, reward_func, reward_func_name, chat_text
        )
        torch.cuda.empty_cache()
        gc.collect()

        return logprob, actions, sequence_length, sequence_length_p1, padding_mask_p1, score, response, value, padding_mask, query_response, context_length
        
    def train_step(
            self, model, optimizer, my_logprob, enemy_logprob, sequence_length, sequence_length_p1, padding_mask_p1, score, response, value, padding_mask, query_response, context_length, accelerator
        ):
        args = self.args
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                # 4. compute rewards
                if my_logprob.shape[1] != enemy_logprob.shape[1]:
                    min_logprob_len = min(my_logprob.shape[1], enemy_logprob.shape[1])
                    my_logprob = my_logprob[:, :min_logprob_len]
                    enemy_logprob = enemy_logprob[:, :min_logprob_len]

                kl = my_logprob - enemy_logprob
                non_score_reward = -args.kl_coef * kl
                reward = non_score_reward.clone()
                actual_start = torch.arange(reward.size(0), device=reward.device)
                actual_end = torch.where(sequence_length_p1 < reward.size(1), sequence_length_p1, sequence_length)
                reward[[actual_start, actual_end]] += score

                # 5. whiten rewards
                if args.whiten_rewards:
                    reward = masked_whiten(reward, mask=~padding_mask_p1, shift_mean=False)
                    reward = torch.masked_fill(reward, padding_mask_p1, 0)

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = response.shape[1]

                for t in reversed(range(gen_length)):
                    nextvalue = value[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = reward[:, t] + args.gamma * nextvalue - value[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)

                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + value
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)

            model.base.gradient_checkpointing_enable()
            time.sleep(1)
            with accelerator.accumulate(model):
                attention_mask = (query_response != self.processing_class.pad_token_id).float()
                logits, vpred_temp = model(query_response, attention_mask=attention_mask)
                logits = logits[:, context_length - 1 : -1, :]  # (batch, gen_length, vocab_size)
                vpred = vpred_temp[:, context_length - 1 : -1]  # (batch, gen_length)
                logits /= args.temperature + 1e-7
                new_all_logprobs = F.log_softmax(logits, dim=-1)
                new_logprobs = torch.gather(new_all_logprobs, 2, response.unsqueeze(-1)).squeeze(-1)
                new_logprobs = torch.masked_fill(
                    new_logprobs, padding_mask, INVALID_LOGPROB
                )
                vpred = torch.masked_fill(vpred, padding_mask_p1, 0)
                vpredclipped = torch.clamp(
                    vpred,
                    value - args.cliprange_value,
                    value + args.cliprange_value,
                )
                vf_losses1 = torch.square(vpred - returns)
                vf_losses2 = torch.square(vpredclipped - returns)
                vf_loss_max = torch.max(vf_losses1, vf_losses2)
                vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1)
                logprobs_diff = new_logprobs - my_logprob
                ratio = torch.exp(logprobs_diff)
                pg_losses = -advantages * ratio
                pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                pg_loss_max = torch.max(pg_losses, pg_losses2)
                pg_loss = masked_mean(pg_loss_max, ~padding_mask)
                loss = pg_loss + args.vf_coef * vf_loss

                del (
                    enemy_logprob, kl, non_score_reward, reward, actual_start, actual_end, lastgaelam, advantages_reversed, advantages, returns, my_logprob, value, query_response,
                    attention_mask, response, padding_mask, padding_mask_p1, sequence_length, sequence_length_p1, score, context_length, 
                )

                torch.cuda.empty_cache()
                gc.collect()

                print("-" * 10 + ' Backward start')
                accelerator.backward(loss)
                print("-" * 10 + ' Backward end')
                optimizer.step()
                optimizer.zero_grad()
        
        del (
            logits, vpred_temp, vpred, new_all_logprobs, new_logprobs,
            vpredclipped, vf_losses1, vf_losses2, vf_loss_max, vf_loss, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max, pg_loss
        )

        torch.cuda.empty_cache()
        gc.collect()

        return

    # convert answer to action
    def get_action_from_answer(self, answer):
        action = np.zeros((16, 3), dtype=int)
        for line in answer.split("\n"):
            match = re.match(r"Unit (\d+): (\d+)(?:, (-?\d+), (-?\d+))?", line.strip())
            if match:
                unit, act, dx, dy = match.groups(default=0)
                unit, act = int(unit), int(act)
                if unit < 0 or unit >= 16:
                    continue
                dx, dy = int(dx) if dx else 0, int(dy) if dy else 0
                action[unit] = [act, dx, dy]
        return action