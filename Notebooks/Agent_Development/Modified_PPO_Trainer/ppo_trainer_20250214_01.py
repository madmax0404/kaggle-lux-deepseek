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

game_rules = """
!!!GAME RULES!!!
Environment
Two teams compete against each other on a 2D map in a best of 5 match sequence (called a game) with each match lasting 100 time steps. Both teams have a pool of units they can control to gain points around the map while also trying to prevent the other team from doing the same.
A core objective of this game is a balanced strategy of exploration and exploitation. It is recommended to explore more in the first match or two before leveraging gained knowledge about the map and opponent behavior to win the latter matches.
Map
The map is a randomly generated 2D grid of size 24x24. There are several core features that make up the map: Unknown Tiles, Empty Tiles, Asteroid Tiles, Nebula Tiles, Energy Tiles, Relic Nodes, and Relic Fragments. Notably, in a game, the map is never regenerated completely between matches. Whatever is the state of the map at the end of one match is what is used for the next match.
Unknown Tiles
These are tiles that are not visible. They can be any type of tile but are not visible to you until a unit is within sensor range of that tile.
Empty Tiles
These are empty tiles in space without anything special about them. Units and tiles can be placed/move onto these tiles.
Asteroid Tiles
Asteroid tiles are impassable tiles that block anything from moving/spawning onto them. These tiles might move around over time during the map in a symmetric fashion. Sometimes asteroid tiles might move on top of existing units. In the game the unit is not removed as a result of this and can still take actions and move around provided there is an non asteroid tile adjacent to it.
Nebula Tiles
Nebula tiles are passable tiles with a number of features. These tiles might move around over time during the map in a symmetric fashion.
Vision Reduction: Nebula tiles can reduce/block vision of units. Because of vision reduction it is even possible for a unit to be unable to see itself while still being able to move! See Vision section below for more details on how team vision is determined. All nebula tiles have the same vision reduction value called params.nebula_tile_vision_reduction which is randomized from 0 to 3.
Energy Reduction: Nebula tiles can reduce the energy of units that end their turn on them. All nebula tiles have the same energy reduction value called params.nebula_tile_energy_reduction.
In Map Tile Types, empty tiles are represented by 0, asteroid tiles by 1, nebula tiles by 2, and unknown tiles by -1.
Energy Tiles
Energy tiles are mysterious objects that emit energy fields which can be harvested by units. These tiles might move around over time during the map in a symmetric fashion. In code, what actually occurs in each game is energy tiles are randomly generated on the map symmetrically and a random function is generated for each tile. Each energy tile's function is a function of distance. The energy value of a tile on a map is determined to be the sum of the energy tile functions applied to the distance between tile and each tile.
Relic Nodes
Relic nodes are objects in space that enable ships to go near it to gain team points. These relic nodes however are ancient and thus fragmented. As a result, only certain tiles near the relic nodes when a friendly ship is on it will gain points. The tiles that yield points are always hidden and can only be discovered by trial and error by moving around the relic nodes. Relic node positions themselves can be observed if withins sensor range. The tiles around relic nodes can overlap with tiles of other relic nodes but will not yield extra points if that occurs and is treated as one tile.
In code, a random 5x5 configuration / mask centered on the relic node is generated indicating which tiles yield points and which don't. Multiple ships can stack on one tile but will only yield at most one point per tile. Note that ship stacking can be risky due to the sapping action.
Units
Units in the game are ships that can move one tile in 5 directions (center, up, right, down, left) and perform a ranged energy sapping action. Units can overlap with other friendly units if they move onto the same tile. Units have a energy property which determines whether they can perform actions and start with 100 energy and can have a max of 400 energy. Energy is recharged via the energy field of the map. They always spawn on one of the two corners of the map depending on which team they are on.
Note that nebula tiles and energy fields can modify the energy of a unit when it is on that tile. However they can never reduce the energy of a unit below 0, only opposing units can do that which will then remove the unit from the game to be respawned at a later timestep. Unit IDs range from 0 to params.max_units - 1 for each team, and are recycled when units are spawned in if a previous one was removed.
Move Actions
All move actions except moving center cost params.unit_move_cost energy to perform. Moving center is always free (a zero action). Attempting to move off the edge of the map results in no movement occuring but energy is still consumed. Units cannot move onto tiles with an impassible feature like an asteroid tile.
Sap Actions
The sap action lets a unit target a specific tile on the map within a range called params.unit_sap_range and reduces the energy of each opposition unit on the target tile by params.unit_sap_cost while also costing unit_sap_cost energy to use. Moreover, any opposition units on the 8 adjacent tiles to the target tile are also sapped and their energy is reduced by params.unit_sap_cost * params.unit_sap_dropoff_factor.
Sap actions are submitted to the game engine / environment as a delta x and delta y value relative to the unit's current position. The delta x and delta y value magnitudes must both be <= params.unit_sap_range, so the sap range is a square around the unit.
Generally sap actions are risky since a single miss means your ships lose energy while the opponent does not. The area of effect can mitigate this risk somewhat depending on game parameters. Sap actions can however prove very valuable when opposition ships are heavily stacked and get hit as sapping the stacked tile hits every ship on the tile.
Vision
A team's vision is the combined vision of all units on that team. Team vision is essentially a boolean mask / matrix over the 2D map indicating whether that tile's information is visible to the team. In this game, you can think of each unit having an "eye in the sky" sattelite that is capturing information about the units surroundings, but this sattelite has reduced accuracy the farther away the tile is from the unit.
To determine which map tiles are visible to a team, we compute a vision power value for each tile on the map. For each unit on a team, we check each tile within the unit's sensor range and add 1 + params.unit_sensor_range - min(dx, dy) to the vision power map at tile (x+dx, y+dy) where (x,y) is the unit's position and (dx, dy) is the offset from the unit's position and abs(dx) <= params.unit_sensor_range and abs(dy) <= params.unit_sensor_range.
Nebula tiles have a vision reduction value of params.nebula_tile_vision_reduction. This number is reduced from every tile's vision power if that tile is a nebula tile.
When a unit is near a nebula tile, it can't see details about some nebula tiles, but it can see tiles beyond nebula tiles.
When a unit is inside a nebula tile, if the nebula vision reduction is powerful enough, the unit cannot even see itself or any other nebula tiles.
Unit vision can overlap and increase the vision power linearly, which can help handle the situations like above when you cannot see anything.
Collisions / Energy Void Fields
In close quarters, units can impact each other in two ways, via direct collisions or by being adjacent to each other and sapping energy via their energy void fields.
In the event of two or more units from opposing teams occupy the same tile at the end of a turn, the team with the highest aggregate energy among its units on that tile survive, while the units of the opposing teams are removed from the game. If it is a tie, all units are removed from the game.
Furthermore, each unit generates an "energy void" field around itself that affects all cardinally (up, right, down left) adjacent opposition units. To determine how exactly each unit is affected by these energy void fields, we compute a 2D map for each team indicating the energy void strength at each tile. A unit contributes to tiles adjacent to itself a energy void strength equal to the total amount of energy the unit has at the start of the turn multiplied by params.unit_energy_void_factor rounded down. After a energy void map is computed for each team, a unit's energy is reduced by the energy void strength of the tile it is on divided by the total number of units on that tile. Note that units removed due to collisions do not contribute to the energy void field.
The energy void fields generally encourage stacking units to better spread out energy sapped by energy void fields of opposition units.
Win Conditions
To win the game, the team must have won the most matches out of the 5 match sequence.
To win a match, the team must have gained more relic points than the other team at the end of the match. If the relic points scores are tied, then the match winner is decided by who has more total unit energy. If that is also tied then the winner is chosen at random.
Match Resolution Order
At each time step of a match, we run the following steps in order:
1. Move all units that have enough energy to move
2. Execute the sap actions of all units that have enough energy to do so
3. Resolve collisions and apply energy void fields
4. Update the energy of all units based on their position (energy fields and nebula tiles)
5. Spawn units for all teams. Remove units that have less than 0 energy.
6. Determine the team vision / sensor masks for all teams and mask out observations accordingly
7. Environment objects like asteroids/nebula tiles/energy tiles move around in space
8. Compute new team points
Note that each match runs for params.max_steps_in_match steps and you take that many actions that affect the game. However, you will actually receive params.max_steps_in_match + 1 frames of observations since the very first frame will either be empty or the previous match's final observation (actions on these observations will not do anything).
Game Parameters
The full set of game parameters can be found here in the codebase.
Randomized Game Parameters / Map Generation
There are a number of randomized game paramteres which can modify and even disable/enable certain game mechanics. None of these game parameters are changed between matches in a game. The majority of these parameters are also not given to the teams themselves and must be discovered through exploration.
env_params_ranges = dict(
    map_type=[1],
    unit_move_cost=list(range(1, 6)), # list(range(x, y)) = [x, x+1, x+2, ... , y-1]
    unit_sensor_range=list(range(2, 5)),
    nebula_tile_vision_reduction=list(range(0,4)),
    nebula_tile_energy_reduction=[0, 0, 10, 25],
    unit_sap_cost=list(range(30, 51)),
    unit_sap_range=list(range(3, 8)),
    unit_sap_dropoff_factor=[0.25, 0.5, 1],
    unit_energy_void_factor=[0.0625, 0.125, 0.25, 0.375],
    # map randomizations
    nebula_tile_drift_speed=[-0.05, -0.025, 0.025, 0.05],
    energy_tile_drift_speed=[0.01, 0.02, 0.03, 0.04, 0.05],
    energy_tile_drift_magnitude=list(range(3, 6))
)
These parameter ranges (and other parameters) are subject to change in the beta phase of this competition as we gather feedback and data.
There are 6 actions that can be taken by a unit in this game: 0 = center, 1 = up, 2 = right, 3 = down, 4 = left, 5 = sap.
So your answer should be in this format:
Unit 0: action(from 0 to 5)
Unit 1: action(from 0 to 5)
Unit 2: action(from 0 to 5)
Unit 3: action(from 0 to 5)
Unit 4: action(from 0 to 5)
Unit 5: action(from 0 to 5)
Unit 6: action(from 0 to 5)
Unit 7: action(from 0 to 5)
Unit 8: action(from 0 to 5)
Unit 9: action(from 0 to 5)
Unit 10: action(from 0 to 5)
Unit 11: action(from 0 to 5)
Unit 12: action(from 0 to 5)
Unit 13: action(from 0 to 5)
Unit 14: action(from 0 to 5)
Unit 15: action(from 0 to 5)
However, if you choose to sap(5), you should also provide the direction of the sap, which is a pair of integers (dx, dy) where dx and dy are the relative coordinates of the target tile from the unit's current position. The magnitudes of dx and dy must be less than or equal to the unit's sap range. For example, if unit 3 is at (5, 5) and you want to sap the tile at (7, 7), your answer for unit 3 should be 5, 2, 2.
Also, you can only take actions for the units that are available to you in the current timestep. If you take an action for a unit that is not available to you, the game engine will ignore that action.
Additionally, you can only take one action per unit per timestep. If you take multiple actions for a single unit in a timestep, the game engine will ignore all but the first action.
So, below is an example of a valid answer:
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
In the above example, unit 0 moves up, unit 1 moves right, unit 2 saps at (2, 2) relative to its current position, unit 3 does nothing, unit 4 saps at (1, 1) relative to its current position, unit 5 saps at (-1, -2) relative to its current position, unit 6 saps at (-2, 2) relative to its current position, unit 7 saps at (0, 0) relative to its current position, unit 8 moves left, unit 9 does nothing, unit 10 moves down, unit 11 moves right, unit 12 moves up, unit 13 does nothing, unit 14 saps at (-4, 5) relative to its current position, and unit 15 saps at (3, -3) relative to its current position.
"""

def prep_llm_input(env_cfg, obs):

    game_state_info = "\n!!!GAME STATE INFORMATION!!!"

    ### env_cfg information
    env_cfg_info = "\nENVIRONMENT CONFIGURATION:"
    max_units = f"\nMaximum possible number of units for each team: {env_cfg['max_units']}."
    match_count_per_episode = f"\nNumber of matches per game: {env_cfg['match_count_per_episode']}."
    max_steps_in_match = f"\nNumber of steps per match: {env_cfg['max_steps_in_match']}."
    map_height = f"\nMap height: {env_cfg['map_height']}."
    map_width = f"\nMap width: {env_cfg['map_width']}."
    num_teams = f"\nNumber of teams: {env_cfg['num_teams']}."
    unit_move_cost = f"\nUnit move energy cost: {env_cfg['unit_move_cost']}."
    unit_sap_cost = f"\nUnit sap energy cost: {env_cfg['unit_sap_cost']}."
    unit_sap_range = f"\nUnit sap range: {env_cfg['unit_sap_range']}."
    unit_sensor_range = f"\nUnit sensor range: {env_cfg['unit_sensor_range']}."

    ### obs information
    obs_info = "\nOBSERVATION:"
    unit_position_warning = "\nUnit position: -1, -1 means the unit is not spawned yet or not visible."

    # unit positions
    unit_position_info = "\nUnit Positions:"
    obs_my_unit_positions = obs['units']['position'][self.team_id]
    my_unit_positions_list = []
    for i in range(obs_my_unit_positions.shape[0]):
        pos = obs_my_unit_positions[i]
        my_unit_positions_list.append(f"\nMy unit {i} position: {pos[0]}, {pos[1]}.")
    my_unit_positions = "".join(my_unit_positions_list)

    obs_enemy_unit_positions = obs['units']['position'][self.enemy_team_id]
    enemy_unit_positions_list = []
    for i in range(obs_enemy_unit_positions.shape[0]):
        pos = obs_enemy_unit_positions[i]
        enemy_unit_positions_list.append(f"\nEnemy unit {i} position: {pos[0]}, {pos[1]}.")
    enemy_unit_positions = "".join(enemy_unit_positions_list)

    # unit energys
    unit_energy_info = "\nUnit Energys:"
    obs_my_unit_energys = obs['units']['energy'][self.team_id]
    my_unit_energys_list = []
    for i in range(obs_my_unit_energys.shape[0]):
        energy = obs_my_unit_energys[i]
        my_unit_energys_list.append(f"\nMy unit {i} energy: {energy}.")
    my_unit_energys = "".join(my_unit_energys_list)

    obs_enemy_unit_energys = obs['units']['energy'][self.enemy_team_id]
    enemy_unit_energys_list = []
    for i in range(obs_enemy_unit_energys.shape[0]):
        energy = obs_enemy_unit_energys[i]
        enemy_unit_energys_list.append(f"\nEnemy unit {i} energy: {energy}.")
    enemy_unit_energys = "".join(enemy_unit_energys_list)

    # unit masks
    unit_mask_info = "\nUnit Visibility:"
    obs_my_units_mask = obs['units_mask'][self.team_id]
    my_units_mask_list = []
    for i in range(obs_my_units_mask.shape[0]):
        mask = obs_my_units_mask[i]
        my_units_mask_list.append(f"\nMy unit {i} visibility: {mask}.")
    my_units_mask = "".join(my_units_mask_list)

    obs_enemy_units_mask = obs['units_mask'][self.enemy_team_id]
    enemy_units_mask_list = []
    for i in range(obs_enemy_units_mask.shape[0]):
        mask = obs_enemy_units_mask[i]
        enemy_units_mask_list.append(f"\nEnemy unit {i} visibility: {mask}.")
    enemy_units_mask = "".join(enemy_units_mask_list)

    # sensor mask
    sensor_mask_info = "\nSensor Mask:"
    obs_sensor_mask = obs['sensor_mask']
    sensor_mask_list = []
    for i in range(obs_sensor_mask.shape[0]):
        sensor_mask_list.append(f"\nSensor mask row {i}: {str(obs_sensor_mask[i]).replace("[", "").replace("]", "")}.")
    sensor_mask = "".join(sensor_mask_list)

    # map features - energy
    map_features_energy_info = "\nMap Energys:"
    obs_map_features_energy = obs['map_features']['energy']
    map_features_energy_list = []
    for i in range(obs_map_features_energy.shape[0]):
        map_features_energy_list.append(f"\nMap energy row {i}: {str(obs_map_features_energy[i]).replace("[", "").replace("]", "")}.")
    map_features_energy = "".join(map_features_energy_list)

    # map features - tile_type
    map_features_tile_type_info = "\nMap Tile Types:"
    obs_map_features_tile_type = obs['map_features']['tile_type']
    map_features_tile_type_list = []
    for i in range(obs_map_features_tile_type.shape[0]):
        map_features_tile_type_list.append(f"\nMap tile type row {i}: {str(obs_map_features_tile_type[i]).replace("[", "").replace("]", "")}.")
    map_features_tile_type = "".join(map_features_tile_type_list)

    # relic nodes
    relic_node_info = "\nRelic Node positions:"
    relic_node_warning = "\nRelic node position: -1, -1 means the relic node is not yet discoverd."
    obs_relic_nodes = obs['relic_nodes']
    relic_nodes_list = []
    for i in range(obs_relic_nodes.shape[0]):
        relic_nodes_list.append(f"\nRelic node {i} position: {obs_relic_nodes[i][0]}, {obs_relic_nodes[i][1]}.")
    relic_nodes = "".join(relic_nodes_list)

    # relic nodes mask
    relic_node_mask_info = "\nRelic Node Visibility:"
    obs_relic_nodes_mask = obs['relic_nodes_mask']
    relic_nodes_mask_list = []
    for i in range(obs_relic_nodes_mask.shape[0]):
        relic_nodes_mask_list.append(f"\nRelic node {i} visibility: {obs_relic_nodes_mask[i]}.")
    relic_nodes_mask = "".join(relic_nodes_mask_list)

    # team points
    my_team_points = f"\nMy current point for this match is: {obs['team_points'][self.team_id]}."
    enemy_team_points = f"\nEnemy current point for this match is: {obs['team_points'][self.enemy_team_id]}."

    # team wins
    my_team_wins = f"\nI have won {obs['team_wins'][self.team_id]} matches."
    enemy_team_wins = f"\nEnemy has won {obs['team_wins'][self.enemy_team_id]} matches."

    # steps
    steps = f"\nThis is step {obs['steps']} of the game."

    # match_steps
    match_steps = f"\nThis is step {obs['match_steps']} of the match."

    if self.enemy_spawn_location is None:
        enemy_spawn_location_warning = "\nEnemy spawn location: not yet discovered."
    else:
        enemy_spawn_location_warning = f"\nEnemy spawn location: {self.enemy_spawn_location[0]}, {self.enemy_spawn_location[1]}."
    
    all_variables = "".join([
        game_state_info, env_cfg_info, max_units, match_count_per_episode, max_steps_in_match, map_height, map_width, num_teams, unit_move_cost, unit_sap_cost, unit_sap_range, unit_sensor_range,
        obs_info,
        unit_position_warning, unit_position_info, my_unit_positions, enemy_unit_positions,
        unit_energy_info, my_unit_energys, enemy_unit_energys,
        unit_mask_info, my_units_mask, enemy_units_mask,
        sensor_mask_info, sensor_mask,
        map_features_energy_info, map_features_energy,
        map_features_tile_type_info, map_features_tile_type,
        relic_node_info, relic_node_warning, relic_nodes,
        relic_node_mask_info, relic_nodes_mask,
        my_team_points, enemy_team_points, my_team_wins, enemy_team_wins, steps, match_steps, enemy_spawn_location_warning
    ])

    return all_variables


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
        self.num_games_to_train = num_games_to_train
        self.game_env = game_env

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
        # copy 2
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
        num_games_to_train = self.num_games_to_train
        env = self.game_env

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


        
        for epoch in range(num_games_to_train):

            obs_all, info = env.reset()
            env_cfg = info['params']

            previous_score_player_0 = 0.0
            previous_score_player_1 = 0.0

            current_score_player_0 = obs_all['player_0']['team_points'][0]
            current_score_player_1 = obs_all['player_1']['team_points'][1]

            reward_score_player_0 = current_score_player_0 - previous_score_player_0
            reward_score_player_1 = current_score_player_1 - previous_score_player_1

            previous_score_player_0 = current_score_player_0
            previous_score_player_1 = current_score_player_1

            player_0_llm_input = prep_llm_input(env_cfg, obs_all['player_0'])
            player_1_llm_input = prep_llm_input(env_cfg, obs_all['player_1'])

            player_0_input = "".join([game_rules, player_0_llm_input])
            player_1_input = "".join([game_rules, player_1_llm_input])

            player_0_tokens = processing_class(player_0_input, return_tensors="pt", max_length=10000, truncation=False, padding="max_length")
            player_1_tokens = processing_class(player_1_input, return_tensors="pt", max_length=10000, truncation=False, padding="max_length")


            with torch.no_grad():
                player_0_queries = player_0_tokens["input_ids"].to(device)
                player_1_queries = player_1_tokens["input_ids"].to(device)

                content_length_player_0 = player_0_queries.shape[1]
                content_length_player_1 = player_1_queries.shape[1]

                player_0_responses = []
                player_0_postprocessed_responses = []
                player_0_logprobs = []
                player_0_scores = []
                player_0_sequence_lengths = []
                player_0_values = []

                player_1_responses = []
                player_1_postprocessed_responses = []
                player_1_logprobs = []
                player_1_scores = []
                player_1_sequence_lengths = []
                player_1_values = []

                with unwrap_model_for_generation(
                    self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    query_responses, logitss = batch_generation(
                        unwrapped_model.policy,
                        player_0_queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )

                for i in range(0, player_0_queries.shape[0], args.local_rollout_forward_batch_size):
                    query = player_0_queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, content_length_player_0:]
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
                        unwrapped_value_model, query_response, processing_class.pad_token_id, content_length_player_0
                    )

                    value = full_value[:, content_length_player_0 - 1 : -1].squeeze(-1)
                    _, score, _ = get_reward(
                        reward_model, postprocessed_query_response, processing_class.pad_token_id, content_length_player_0
                    )

                    player_0_responses.append(response)
                    player_0_postprocessed_responses.append(postprocessed_response)
                    player_0_logprobs.append(logprob)
                    player_0_sequence_lengths.append(sequence_length)
                    player_0_scores.append(score)
                    player_0_values.append(value)

                player_0_responses = torch.cat(player_0_responses, 0)
                player_0_postprocessed_responses = torch.cat(player_0_postprocessed_responses, 0)
                player_0_logprobs = torch.cat(player_0_logprobs, 0)
                player_0_sequence_lengths = torch.cat(player_0_sequence_lengths, 0)
                player_0_scores = torch.cat(player_0_scores, 0)
                player_0_values = torch.cat(player_0_values, 0)

                # copy 2
                with unwrap_model_for_generation(
                    self.model_2, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model_2:
                    query_responses_2, logitss_2 = batch_generation(
                        unwrapped_model_2.policy,
                        player_1_queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )

                # copy 2
                for i in range(0, player_1_queries.shape[0], args.local_rollout_forward_batch_size):
                    query_2 = player_1_queries[i : i + args.local_rollout_forward_batch_size]
                    query_response_2 = query_responses_2[i : i + args.local_rollout_forward_batch_size]
                    response_2 = query_response_2[:, content_length_player_1:]
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
                        unwrapped_value_model_2, query_response_2, processing_class.pad_token_id, content_length_player_1
                    )

                    value_2 = full_value_2[:, content_length_player_1 - 1 : -1].squeeze(-1)
                    _, score_2, _ = get_reward(
                        reward_model, postprocessed_query_response_2, processing_class.pad_token_id, content_length_player_1
                    )

                    player_1_responses.append(response_2)
                    player_1_postprocessed_responses.append(postprocessed_response_2)
                    player_1_logprobs.append(logprob_2)
                    player_1_sequence_lengths.append(sequence_length_2)
                    player_1_scores.append(score_2)
                    player_1_values.append(value_2)

                player_1_responses = torch.cat(player_1_responses, 0)
                player_1_postprocessed_responses = torch.cat(player_1_postprocessed_responses, 0)
                player_1_logprobs = torch.cat(player_1_logprobs, 0)
                player_1_sequence_lengths = torch.cat(player_1_sequence_lengths, 0)
                player_1_scores = torch.cat(player_1_scores, 0)
                player_1_values = torch.cat(player_1_values, 0)

                del (logprob, logprob_2, full_value, full_value_2, value, value_2, score, score_2, unwrapped_model, unwrapped_model_2)
                torch.cuda.empty_cache()
                gc.collect()

                # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                contain_eos_token = torch.any(player_0_postprocessed_responses == self.processing_class.eos_token_id, dim=-1)
                if self.args.missing_eos_penalty is not None:
                    player_0_scores[~contain_eos_token] -= self.args.missing_eos_penalty
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # copy 2
                contain_eos_token_2 = torch.any(player_1_postprocessed_responses == self.processing_class.eos_token_id, dim=-1)
                if self.args.missing_eos_penalty is not None:
                    player_1_scores[~contain_eos_token_2] -= self.args.missing_eos_penalty
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(player_0_responses.shape[1], device=player_0_responses.device).repeat(player_0_responses.shape[0], 1)
                padding_mask = response_idxs > player_0_sequence_lengths.unsqueeze(1)
                player_0_logprobs = torch.masked_fill(player_0_logprobs, padding_mask, INVALID_LOGPROB)
                sequence_lengths_p1 = player_0_sequence_lengths + 1
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                player_0_values = torch.masked_fill(player_0_values, padding_mask_p1, 0)

                # copy 2
                response_idxs_2 = torch.arange(player_1_responses.shape[1], device=player_1_responses.device).repeat(player_1_responses.shape[0], 1)
                padding_mask_2 = response_idxs_2 > player_1_sequence_lengths.unsqueeze(1)
                player_1_logprobs = torch.masked_fill(player_1_logprobs, padding_mask_2, INVALID_LOGPROB)
                sequence_lengths_p1_2 = player_1_sequence_lengths + 1
                padding_mask_p1_2 = response_idxs_2 > (sequence_lengths_p1_2.unsqueeze(1))
                player_1_values = torch.masked_fill(player_1_values, padding_mask_p1_2, 0)

                # 4. compute rewards
                kl = player_0_logprobs - player_1_logprobs
                non_score_reward = -args.kl_coef * kl
                rewards = non_score_reward.clone()
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, player_0_sequence_lengths)
                rewards[[actual_start, actual_end]] += player_0_scores

                # copy 2
                kl_2 = player_1_logprobs - player_0_logprobs
                non_score_reward_2 = -args.kl_coef * kl_2
                rewards_2 = non_score_reward_2.clone()
                actual_start_2 = torch.arange(rewards_2.size(0), device=rewards_2.device)
                actual_end_2 = torch.where(sequence_lengths_p1_2 < rewards_2.size(1), sequence_lengths_p1_2, player_1_sequence_lengths)
                rewards_2[[actual_start_2, actual_end_2]] += player_1_scores

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
                gen_length = player_0_responses.shape[1]

                # copy 2
                lastgaelam_2 = 0
                advantages_reversed_2 = []
                gen_length_2 = player_1_responses.shape[1]

                for t in reversed(range(gen_length)):
                    nextvalues = player_0_values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - player_0_values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)

                # copy 2
                for t in reversed(range(gen_length_2)):
                    nextvalues_2 = player_1_values[:, t + 1] if t < gen_length_2 - 1 else 0.0
                    delta_2 = rewards_2[:, t] + args.gamma * nextvalues_2 - player_1_values[:, t]
                    lastgaelam_2 = delta_2 + args.gamma * args.lam * lastgaelam_2
                    advantages_reversed_2.append(lastgaelam_2)

                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + player_0_values
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)

                # copy 2
                advantages_2 = torch.stack(advantages_reversed_2[::-1], axis=1)
                returns_2 = advantages_2 + player_1_values
                advantages_2 = masked_whiten(advantages_2, ~padding_mask_2)
                advantages_2 = torch.masked_fill(advantages_2, padding_mask_2, 0)

                torch.cuda.empty_cache()








        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():

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
