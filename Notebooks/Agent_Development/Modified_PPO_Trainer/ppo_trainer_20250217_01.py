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
from accelerate.utils import broadcast, gather_object, GradientAccumulationPlugin
# from datasets import Dataset
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

answer_format = '\nRespond in the following format:\n<answer>\n...\n</answer>\n'

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

def prep_llm_input(env_cfg, obs, team_id, enemy_team_id, enemy_spawn_location, map_explored_status):

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
    obs_my_unit_positions = obs['units']['position'][team_id]
    my_unit_positions_list = []
    for i in range(obs_my_unit_positions.shape[0]):
        pos = obs_my_unit_positions[i]
        my_unit_positions_list.append(f"\nMy unit {i} position: {pos[0]}, {pos[1]}.")
    my_unit_positions = "".join(my_unit_positions_list)

    obs_enemy_unit_positions = obs['units']['position'][enemy_team_id]
    enemy_unit_positions_list = []
    for i in range(obs_enemy_unit_positions.shape[0]):
        pos = obs_enemy_unit_positions[i]
        enemy_unit_positions_list.append(f"\nEnemy unit {i} position: {pos[0]}, {pos[1]}.")
    enemy_unit_positions = "".join(enemy_unit_positions_list)

    # unit energys
    unit_energy_info = "\nUnit Energys:"
    obs_my_unit_energys = obs['units']['energy'][team_id]
    my_unit_energys_list = []
    for i in range(obs_my_unit_energys.shape[0]):
        energy = obs_my_unit_energys[i]
        my_unit_energys_list.append(f"\nMy unit {i} energy: {energy}.")
    my_unit_energys = "".join(my_unit_energys_list)

    obs_enemy_unit_energys = obs['units']['energy'][enemy_team_id]
    enemy_unit_energys_list = []
    for i in range(obs_enemy_unit_energys.shape[0]):
        energy = obs_enemy_unit_energys[i]
        enemy_unit_energys_list.append(f"\nEnemy unit {i} energy: {energy}.")
    enemy_unit_energys = "".join(enemy_unit_energys_list)

    # unit masks
    unit_mask_info = "\nUnit Visibility:"
    obs_my_units_mask = obs['units_mask'][team_id]
    my_units_mask_list = []
    for i in range(obs_my_units_mask.shape[0]):
        mask = obs_my_units_mask[i]
        my_units_mask_list.append(f"\nMy unit {i} visibility: {mask}.")
    my_units_mask = "".join(my_units_mask_list)

    obs_enemy_units_mask = obs['units_mask'][enemy_team_id]
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

    # map explored status
    map_explored_status_info = "\nMap Explored Status:"
    map_explored_status_list = []
    for i in range(map_explored_status.shape[0]):
        map_explored_status_list.append(f"\nMap explored status row {i}: {str(map_explored_status[i]).replace("[", "").replace("]", "")}.")
    map_explored_status_input = "".join(map_explored_status_list)

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
    my_team_points = f"\nMy current point for this match is: {obs['team_points'][team_id]}."
    enemy_team_points = f"\nEnemy current point for this match is: {obs['team_points'][enemy_team_id]}."

    # team wins
    my_team_wins = f"\nI have won {obs['team_wins'][team_id]} matches."
    enemy_team_wins = f"\nEnemy has won {obs['team_wins'][enemy_team_id]} matches."

    # steps
    steps = f"\nThis is step {obs['steps']} of the game."

    # match_steps
    match_steps = f"\nThis is step {obs['match_steps']} of the match."

    if enemy_spawn_location is None:
        enemy_spawn_location_warning = "\nEnemy spawn location: not yet discovered."
    else:
        enemy_spawn_location_warning = f"\nEnemy spawn location: {enemy_spawn_location[0]}, {enemy_spawn_location[1]}."
    
    all_variables = "".join([
        game_state_info, env_cfg_info, max_units, match_count_per_episode, max_steps_in_match, map_height, map_width, num_teams, unit_move_cost, unit_sap_cost, unit_sap_range, unit_sensor_range,
        obs_info,
        unit_position_warning, unit_position_info, my_unit_positions, enemy_unit_positions,
        unit_energy_info, my_unit_energys, enemy_unit_energys,
        unit_mask_info, my_units_mask, enemy_units_mask,
        sensor_mask_info, sensor_mask, map_explored_status_info, map_explored_status_input,
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
        self.reward_functions = reward_functions
        self._metrics = defaultdict(list)

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

        self.value_model = value_model
        self.value_model_2 = value_model_2
        self.data_collator = data_collator
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_2, self.lr_scheduler_2 = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # calculate various batch sizes
        #########
        plugin = GradientAccumulationPlugin(sync_with_dataloader=False, num_steps=args.gradient_accumulation_steps, sync_each_batch=False)
        accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_plugin=plugin)
        self.accelerator = accelerator
        # args.world_size = accelerator.num_processes
        # args.local_batch_size = (
        #     args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        # )
        # args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        # args.batch_size = int(args.local_batch_size * args.world_size)
        # args.mini_batch_size = exact_div(
        #     args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        # )
        # args.local_mini_batch_size = exact_div(
        #     args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        # )
        # if args.whiten_rewards:
        #     assert (
        #         args.local_mini_batch_size >= 8
        #     ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
        # # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        # args.num_total_batches = math.ceil(
        #     args.total_episodes / args.batch_size
        # )  # we may train for more than `total_episodes`
        args.num_total_batches = num_games_to_train * 505

        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)

        #########
        # setup model, optimizer, and others
        #########
        for module in [self.policy_model, self.policy_model_2, self.value_model, self.value_model_2]:
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
            os.makedirs(self.args.output_dir + "_2", exist_ok=True)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)
        # copy 2
        if hasattr(self.model_2, "add_model_tags"):
            self.model_2.add_model_tags(self._tag_names)

        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.lr_scheduler = accelerator.prepare(self.model, self.optimizer, self.lr_scheduler)
        self.model_2, self.optimizer_2, self.lr_scheduler_2 = accelerator.prepare(self.model_2, self.optimizer_2, self.lr_scheduler_2) # edit it later
        torch.manual_seed(self.local_seed)  # reset the local seed again


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
        processing_class = self.processing_class
        device = accelerator.device
        num_games_to_train = self.num_games_to_train
        env = self.game_env

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
        player_0_approxkl_stats = torch.zeros(stats_shape, device=device)
        player_1_approxkl_stats = torch.zeros(stats_shape, device=device)
        player_0_pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        player_1_pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        player_0_pg_loss_stats = torch.zeros(stats_shape, device=device)
        player_1_pg_loss_stats = torch.zeros(stats_shape, device=device)
        player_0_vf_loss_stats = torch.zeros(stats_shape, device=device)
        player_1_vf_loss_stats = torch.zeros(stats_shape, device=device)
        player_0_vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        player_1_vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        player_0_entropy_stats = torch.zeros(stats_shape, device=device)
        player_1_entropy_stats = torch.zeros(stats_shape, device=device)
        player_0_ratio_stats = torch.zeros(stats_shape, device=device)
        player_1_ratio_stats = torch.zeros(stats_shape, device=device)

        # model = torch.compile(model)
        # model_2 = torch.compile(model_2)
    
        model.train()
        model_2.train()

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

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.deepspeed_2 = self.model_2
            self.model_wrapped = self.model
            self.model_wrapped_2 = self.model_2


        
        for game_number in range(1, num_games_to_train + 1):
            self.state.episode += 1 * args.batch_size

            obs_all, info = env.reset()
            env_cfg = info['params']

            # max_steps_per_match = env_cfg["max_steps_in_match"] + 1

            previous_score_player_0 = 0.0
            previous_score_player_1 = 0.0

            match_number = 1

            first_spawn = False

            player_0_spawn_location = None
            player_1_spawn_location = None

            player_0_map_explored_status = np.zeros((env_cfg["map_height"], env_cfg["map_width"]), dtype=int)
            player_1_map_explored_status = np.zeros((env_cfg["map_height"], env_cfg["map_width"]), dtype=int)

            game_ended = False

            player_0_match_won_num = 0
            player_1_match_won_num = 0

            while game_ended is not True:

                current_score_player_0 = obs_all['player_0']['team_points'][0]
                current_score_player_1 = obs_all['player_1']['team_points'][1]

                reward_score_player_0 = current_score_player_0 - previous_score_player_0
                reward_score_player_1 = current_score_player_1 - previous_score_player_1

                previous_score_player_0 = current_score_player_0
                previous_score_player_1 = current_score_player_1

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
                    if current_score_player_0 > current_score_player_1:
                        player_0_match_won = True
                        player_1_match_lost = True
                        player_0_match_won_num += 1
                    elif current_score_player_0 < current_score_player_1:
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

                player_0_current_map_tile_type = player_0_map_features['tile_type']
                player_1_current_map_tile_type = player_1_map_features['tile_type']

                player_0_map_explored_status[player_0_current_map_tile_type != -1] = 1
                player_1_map_explored_status[player_1_current_map_tile_type != -1] = 1

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

                player_0_input = "".join([answer_format, game_rules, player_0_llm_input])
                player_1_input = "".join([answer_format, game_rules, player_1_llm_input])

                player_0_tokens = processing_class(player_0_input, return_tensors="pt", max_length=10000, truncation=False, padding="max_length").to(device)
                player_1_tokens = processing_class(player_1_input, return_tensors="pt", max_length=10000, truncation=False, padding="max_length").to(device)

                with torch.no_grad():
                    # player_0_queries = player_0_tokens["input_ids"].to(device)
                    # player_1_queries = player_1_tokens["input_ids"].to(device)

                    player_0_context_length = player_0_tokens['input_ids'].shape[1]
                    player_1_context_length = player_1_tokens['input_ids'].shape[1]

                    with unwrap_model_for_generation(
                        self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                    ) as player_0_unwrapped_model:                    
                        player_0_gen_output = player_0_unwrapped_model.policy.generate(
                            **player_0_tokens,
                            generation_config=generation_config,
                            return_dict_in_generate=True,
                            output_scores=True,
                            pad_token_id=processing_class.pad_token_id,
                        )

                    player_0_logits = torch.stack(player_0_gen_output.scores, 1)
                    player_0_query_response = torch.cat((player_0_tokens['input_ids'], player_0_gen_output.sequences[:, player_0_context_length:]), dim=1)
                    player_0_response = player_0_query_response[:, player_0_context_length:]
                    player_0_all_logprob = F.log_softmax(player_0_logits, dim=-1)
                    player_0_logprob = torch.gather(player_0_all_logprob, 2, player_0_response.unsqueeze(-1)).squeeze(-1)
                    # del player_0_logits, player_0_all_logprob, player_0_llm_input, player_0_input
                    #torch.cuda.empty_cache()
                    #gc.collect()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    player_0_postprocessed_response = player_0_response
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        player_0_postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, player_0_response
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    player_0_postprocessed_query_response = torch.cat((player_0_tokens["input_ids"], player_0_postprocessed_response), 1)
                    player_0_sequence_length = first_true_indices(player_0_postprocessed_response == processing_class.pad_token_id) - 1
                    player_0_unwrapped_value_model = accelerator.unwrap_model(model).value_model
                    player_0_full_value, _, _ = get_reward(
                        player_0_unwrapped_value_model, player_0_query_response, processing_class.pad_token_id, player_0_context_length
                    )
                    player_0_value = player_0_full_value[:, player_0_context_length - 1 : -1].squeeze(-1)

                    player_0_completion_ids = player_0_postprocessed_query_response[:, player_0_context_length:]
                    player_0_completion = self.processing_class.decode(player_0_completion_ids[0], skip_special_tokens=True)

                    print(player_0_completion)

                    player_0_func_rewards = torch.zeros(1, len(self.reward_functions), device=device)

                    for i, reward_func in enumerate(self.reward_functions):
                        reward_func_name = reward_func.__name__
                        if reward_func_name == "point_gain_reward_func":
                            reward = reward_func(reward_score_player_0)
                        elif reward_func_name == "match_won_reward_func":
                            reward = reward_func(player_0_match_won)
                        elif reward_func_name == "match_lost_reward_func":
                            reward = reward_func(player_0_match_lost)
                        elif reward_func_name == "game_won_reward_func":
                            reward = reward_func(player_0_game_won)
                        elif reward_func_name == "game_lost_reward_func":
                            reward = reward_func(player_0_game_lost)
                        else:
                            reward = reward_func(player_0_completion)
                        player_0_func_rewards[0, i] = reward
                        self._metrics[f"player_0_rewards/{reward_func_name}"].append(reward)

                    player_0_score = player_0_func_rewards.sum(dim=1)

                    player_0_actions = self.get_action_from_answer(answer=player_0_completion)

                    # del (
                    #     player_0_full_value, player_0_unwrapped_model, player_0_unwrapped_value_model, player_0_completion_ids, player_0_func_rewards, player_0_tokens
                    # )
                    #torch.cuda.empty_cache()
                    #gc.collect()


                    # copy 2
                    with unwrap_model_for_generation(
                        self.model_2, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                    ) as player_1_unwrapped_model:                    
                        player_1_gen_output = player_1_unwrapped_model.policy.generate(
                            **player_1_tokens,
                            generation_config=generation_config,
                            return_dict_in_generate=True,
                            output_scores=True,
                            pad_token_id=processing_class.pad_token_id,
                        )

                    player_1_logits = torch.stack(player_1_gen_output.scores, 1)
                    player_1_query_response = torch.cat((player_1_tokens['input_ids'], player_1_gen_output.sequences[:, player_1_context_length:]), dim=1)
                    player_1_response = player_1_query_response[:, player_1_context_length:]
                    player_1_all_logprob = F.log_softmax(player_1_logits, dim=-1)
                    player_1_logprob = torch.gather(player_1_all_logprob, 2, player_1_response.unsqueeze(-1)).squeeze(-1)
                    # del player_1_logits, player_1_all_logprob, player_1_llm_input, player_1_input
                    #torch.cuda.empty_cache()
                    #gc.collect()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    player_1_postprocessed_response = player_1_response
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        player_1_postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, player_1_response
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    player_1_postprocessed_query_response = torch.cat((player_1_tokens["input_ids"], player_1_postprocessed_response), 1)
                    player_1_sequence_length = first_true_indices(player_1_postprocessed_response == processing_class.pad_token_id) - 1
                    player_1_unwrapped_value_model = accelerator.unwrap_model(model_2).value_model
                    player_1_full_value, _, _ = get_reward(
                        player_1_unwrapped_value_model, player_1_query_response, processing_class.pad_token_id, player_1_context_length
                    )
                    player_1_value = player_1_full_value[:, player_1_context_length - 1 : -1].squeeze(-1)

                    player_1_completion_ids = player_1_postprocessed_query_response[:, player_1_context_length:]
                    player_1_completion = self.processing_class.decode(player_1_completion_ids[0], skip_special_tokens=True)

                    player_1_func_rewards = torch.zeros(1, len(self.reward_functions), device=device)

                    for i, reward_func in enumerate(self.reward_functions):
                        reward_func_name = reward_func.__name__
                        if reward_func_name == "point_gain_reward_func":
                            reward = reward_func(reward_score_player_1)
                        elif reward_func_name == "match_won_reward_func":
                            reward = reward_func(player_1_match_won)
                        elif reward_func_name == "match_lost_reward_func":
                            reward = reward_func(player_1_match_lost)
                        elif reward_func_name == "game_won_reward_func":
                            reward = reward_func(player_1_game_won)
                        elif reward_func_name == "game_lost_reward_func":
                            reward = reward_func(player_1_game_lost)
                        else:
                            reward = reward_func(player_1_completion)
                        player_1_func_rewards[0, i] = reward
                        self._metrics[f"player_1_rewards/{reward_func_name}"].append(reward)

                    player_1_score = player_1_func_rewards.sum(dim=1)

                    player_1_actions = self.get_action_from_answer(answer=player_1_completion)

                    # del (
                    #     player_1_full_value, player_1_unwrapped_model, player_1_unwrapped_value_model, player_1_completion_ids, player_1_func_rewards, player_1_tokens
                    # )
                    #torch.cuda.empty_cache()
                    #gc.collect()


                    # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
                    # Completions not passing that filter will receive a lower score.
                    player_0_contain_eos_token = torch.any(player_0_postprocessed_response == self.processing_class.eos_token_id, dim=-1)
                    if self.args.missing_eos_penalty is not None:
                        player_0_score[~player_0_contain_eos_token] -= self.args.missing_eos_penalty
                    # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                    # copy 2
                    player_1_contain_eos_token = torch.any(player_1_postprocessed_response == self.processing_class.eos_token_id, dim=-1)
                    if self.args.missing_eos_penalty is not None:
                        player_1_score[~player_1_contain_eos_token] -= self.args.missing_eos_penalty
                    # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                    # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                    player_0_response_idxs = torch.arange(player_0_response.shape[1], device=player_0_response.device).repeat(player_0_response.shape[0], 1)
                    player_0_padding_mask = player_0_response_idxs > player_0_sequence_length.unsqueeze(1)
                    player_0_logprob = torch.masked_fill(player_0_logprob, player_0_padding_mask, INVALID_LOGPROB)
                    player_0_sequence_length_p1 = player_0_sequence_length + 1
                    player_0_padding_mask_p1 = player_0_response_idxs > (player_0_sequence_length_p1.unsqueeze(1))
                    player_0_value = torch.masked_fill(player_0_value, player_0_padding_mask_p1, 0)

                    # copy 2
                    player_1_response_idxs = torch.arange(player_1_response.shape[1], device=player_1_response.device).repeat(player_1_response.shape[0], 1)
                    player_1_padding_mask = player_1_response_idxs > player_1_sequence_length.unsqueeze(1)
                    player_1_logprob = torch.masked_fill(player_1_logprob, player_1_padding_mask, INVALID_LOGPROB)
                    player_1_sequence_length_p1 = player_1_sequence_length + 1
                    player_1_padding_mask_p1 = player_1_response_idxs > (player_1_sequence_length_p1.unsqueeze(1))
                    player_1_value = torch.masked_fill(player_1_value, player_1_padding_mask_p1, 0)

                    # 4. compute rewards
                    player_0_kl = player_0_logprob - player_1_logprob
                    player_0_non_score_reward = -args.kl_coef * player_0_kl
                    player_0_reward = player_0_non_score_reward.clone()
                    player_0_actual_start = torch.arange(player_0_reward.size(0), device=player_0_reward.device)
                    player_0_actual_end = torch.where(player_0_sequence_length_p1 < player_0_reward.size(1), player_0_sequence_length_p1, player_0_sequence_length)
                    player_0_reward[[player_0_actual_start, player_0_actual_end]] += player_0_score

                    # copy 2
                    player_1_kl = player_1_logprob - player_0_logprob
                    player_1_non_score_reward = -args.kl_coef * player_1_kl
                    player_1_reward = player_1_non_score_reward.clone()
                    player_1_actual_start = torch.arange(player_1_reward.size(0), device=player_1_reward.device)
                    player_1_actual_end = torch.where(player_1_sequence_length_p1 < player_1_reward.size(1), player_1_sequence_length_p1, player_1_sequence_length)
                    player_1_reward[[player_1_actual_start, player_1_actual_end]] += player_1_score

                    # 5. whiten rewards
                    if args.whiten_rewards:
                        player_0_reward = masked_whiten(player_0_reward, mask=~player_0_padding_mask_p1, shift_mean=False)
                        player_0_reward = torch.masked_fill(player_0_reward, player_0_padding_mask_p1, 0)

                        #copy 2
                        player_1_reward = masked_whiten(player_1_reward, mask=~player_1_padding_mask_p1, shift_mean=False)
                        player_1_reward = torch.masked_fill(player_1_reward, player_1_padding_mask_p1, 0)

                    # 6. compute advantages and returns
                    player_0_lastgaelam = 0
                    player_0_advantages_reversed = []
                    player_0_gen_length = player_0_response.shape[1]

                    # copy 2
                    player_1_lastgaelam = 0
                    player_1_advantages_reversed = []
                    player_1_gen_length = player_1_response.shape[1]

                    for t in reversed(range(player_0_gen_length)):
                        player_0_nextvalue = player_0_value[:, t + 1] if t < player_0_gen_length - 1 else 0.0
                        player_0_delta = player_0_reward[:, t] + args.gamma * player_0_nextvalue - player_0_value[:, t]
                        player_0_lastgaelam = player_0_delta + args.gamma * args.lam * player_0_lastgaelam
                        player_0_advantages_reversed.append(player_0_lastgaelam)

                    # copy 2
                    for t in reversed(range(player_1_gen_length)):
                        player_1_nextvalue = player_1_value[:, t + 1] if t < player_1_gen_length - 1 else 0.0
                        player_1_delta = player_1_reward[:, t] + args.gamma * player_1_nextvalue - player_1_value[:, t]
                        player_1_lastgaelam = player_1_delta + args.gamma * args.lam * player_1_lastgaelam
                        player_1_advantages_reversed.append(player_1_lastgaelam)

                    player_0_advantages = torch.stack(player_0_advantages_reversed[::-1], axis=1)
                    player_0_returns = player_0_advantages + player_0_value
                    player_0_advantages = masked_whiten(player_0_advantages, ~player_0_padding_mask)
                    player_0_advantages = torch.masked_fill(player_0_advantages, player_0_padding_mask, 0)

                    # copy 2
                    player_1_advantages = torch.stack(player_1_advantages_reversed[::-1], axis=1)
                    player_1_returns = player_1_advantages + player_1_value
                    player_1_advantages = masked_whiten(player_1_advantages, ~player_1_padding_mask)
                    player_1_advantages = torch.masked_fill(player_1_advantages, player_1_padding_mask, 0)

                    #torch.cuda.empty_cache()
                    #gc.collect()

                # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
                
                gradient_accumulation_idx = 0
                with accelerator.accumulate(model):
                    player_0_mb_advantage = player_0_advantages
                    player_0_mb_responses = player_0_response
                    player_0_mb_query_responses = player_0_query_response
                    #player_0_mb_query_responses[0] = player_0_mb_query_responses[0].to("cuda", dtype=torch.bfloat16)
                
                    player_0_mb_logprobs = player_0_logprob
                    player_0_mb_return = player_0_returns
                    player_0_mb_values = player_0_value

                    player_0_output, player_0_vpred_temp = forward(model, player_0_mb_query_responses, processing_class.pad_token_id)
                    player_0_logits = player_0_output.logits[:, player_0_context_length - 1 : -1]
                    player_0_logits /= args.temperature + 1e-7
                    player_0_new_all_logprobs = F.log_softmax(player_0_logits, dim=-1)
                    player_0_new_logprobs = torch.gather(player_0_new_all_logprobs, 2, player_0_mb_responses.unsqueeze(-1)).squeeze(-1)
                    player_0_new_logprobs = torch.masked_fill(
                        player_0_new_logprobs, player_0_padding_mask, INVALID_LOGPROB
                    )
                    player_0_vpred = player_0_vpred_temp[:, player_0_context_length - 1 : -1].squeeze(-1)
                    player_0_vpred = torch.masked_fill(player_0_vpred, player_0_padding_mask_p1, 0)
                    player_0_vpredclipped = torch.clamp(
                        player_0_vpred,
                        player_0_mb_values - args.cliprange_value,
                        player_0_mb_values + args.cliprange_value,
                    )
                    player_0_vf_losses1 = torch.square(player_0_vpred - player_0_mb_return)
                    player_0_vf_losses2 = torch.square(player_0_vpredclipped - player_0_mb_return)
                    player_0_vf_loss_max = torch.max(player_0_vf_losses1, player_0_vf_losses2)
                    player_0_vf_loss = 0.5 * masked_mean(player_0_vf_loss_max, ~player_0_padding_mask_p1)
                    player_0_vf_clipfrac = masked_mean(
                        (player_0_vf_losses2 > player_0_vf_losses1).float(), ~player_0_padding_mask_p1
                    )
                    player_0_logprobs_diff = player_0_new_logprobs - player_0_mb_logprobs
                    player_0_ratio = torch.exp(player_0_logprobs_diff)
                    player_0_pg_losses = -player_0_mb_advantage * player_0_ratio
                    player_0_pg_losses2 = -player_0_mb_advantage * torch.clamp(player_0_ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                    player_0_pg_loss_max = torch.max(player_0_pg_losses, player_0_pg_losses2)
                    player_0_pg_loss = masked_mean(player_0_pg_loss_max, ~player_0_padding_mask)
                    player_0_loss = player_0_pg_loss + args.vf_coef * player_0_vf_loss
                    accelerator.backward(player_0_loss)
                    print('backward')
                    optimizer.step()
                    optimizer.zero_grad()
                    # with torch.no_grad():
                    #     player_0_pg_clipfrac = masked_mean(
                    #         (player_0_pg_losses2 > player_0_pg_losses).float(), ~player_0_padding_mask[micro_batch_inds]
                    #     )
                    #     player_0_prob_dist = torch.nn.functional.softmax(player_0_logits, dim=-1)
                    #     player_0_entropy = torch.logsumexp(player_0_logits, dim=-1) - torch.sum(player_0_prob_dist * player_0_logits, dim=-1)
                    #     player_0_approxkl = 0.5 * (player_0_logprobs_diff**2).mean()
                    #     player_0_approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = player_0_approxkl
                    #     player_0_pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                    #         player_0_pg_clipfrac
                    #     )
                    #     player_0_pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = player_0_pg_loss
                    #     player_0_vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = player_0_vf_loss
                    #     player_0_vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                    #         player_0_vf_clipfrac
                    #     )
                    #     player_0_entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = player_0_entropy.mean()
                    #     player_0_ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = player_0_ratio.mean()

                # del (
                #     player_0_output, player_0_vpred_temp, player_0_logits, player_0_new_all_logprobs, player_0_new_logprobs, player_0_vpred, player_0_vpredclipped,
                #     player_0_vf_losses1, player_0_vf_losses2, player_0_vf_loss, player_0_vf_clipfrac, player_0_logprobs_diff, player_0_ratio, player_0_pg_losses, player_0_pg_losses2,
                #     player_0_pg_loss_max,
                #     player_0_pg_loss, player_0_loss, # player_0_pg_clipfrac, player_0_prob_dist, player_0_entropy, player_0_approxkl,
                #     player_0_mb_return,
                #     player_0_mb_advantage, player_0_mb_values, player_0_mb_responses, player_0_mb_query_responses, player_0_mb_logprobs,
                # )

                # copy 2
                with accelerator.accumulate(model_2):
                    player_1_mb_advantage = player_1_advantages
                    player_1_mb_responses = player_1_response
                    player_1_mb_query_responses = player_1_query_response
                    #player_1_mb_query_responses[0] = player_1_mb_query_responses[0].to("cuda", dtype=torch.bfloat16)
                    player_1_mb_logprobs = player_1_logprob
                    player_1_mb_return = player_1_returns
                    player_1_mb_values = player_1_value

                    player_1_output, player_1_vpred_temp = forward(model_2, player_1_mb_query_responses, processing_class.pad_token_id)
                    player_1_logits = player_1_output.logits[:, player_1_context_length - 1 : -1]
                    player_1_logits /= args.temperature + 1e-7
                    player_1_new_all_logprobs = F.log_softmax(player_1_logits, dim=-1)
                    player_1_new_logprobs = torch.gather(player_1_new_all_logprobs, 2, player_1_mb_responses.unsqueeze(-1)).squeeze(-1)
                    player_1_new_logprobs = torch.masked_fill(
                        player_1_new_logprobs, player_1_padding_mask, INVALID_LOGPROB
                    )
                    player_1_vpred = player_1_vpred_temp[:, player_1_context_length - 1 : -1].squeeze(-1)
                    player_1_vpred = torch.masked_fill(player_1_vpred, player_1_padding_mask_p1, 0)
                    player_1_vpredclipped = torch.clamp(
                        player_1_vpred,
                        player_1_mb_values - args.cliprange_value,
                        player_1_mb_values + args.cliprange_value,
                    )
                    player_1_vf_losses1 = torch.square(player_1_vpred - player_1_mb_return)
                    player_1_vf_losses2 = torch.square(player_1_vpredclipped - player_1_mb_return)
                    player_1_vf_loss_max = torch.max(player_1_vf_losses1, player_1_vf_losses2)
                    player_1_vf_loss = 0.5 * masked_mean(player_1_vf_loss_max, ~player_1_padding_mask_p1)
                    player_1_vf_clipfrac = masked_mean(
                        (player_1_vf_losses2 > player_1_vf_losses1).float(), ~player_1_padding_mask_p1
                    )
                    player_1_logprobs_diff = player_1_new_logprobs - player_1_mb_logprobs
                    player_1_ratio = torch.exp(player_1_logprobs_diff)
                    player_1_pg_losses = -player_1_mb_advantage * player_1_ratio
                    player_1_pg_losses2 = -player_1_mb_advantage * torch.clamp(player_1_ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                    player_1_pg_loss_max = torch.max(player_1_pg_losses, player_1_pg_losses2)
                    player_1_pg_loss = masked_mean(player_1_pg_loss_max, ~player_1_padding_mask)
                    player_1_loss = player_1_pg_loss + args.vf_coef * player_1_vf_loss
                    accelerator.backward(player_1_loss)
                    optimizer_2.step()
                    optimizer_2.zero_grad()
                    # with torch.no_grad():
                    #     player_1_pg_clipfrac = masked_mean(
                    #         (player_1_pg_losses2 > player_1_pg_losses).float(), ~player_1_padding_mask
                    #     )
                    #     player_1_prob_dist = torch.nn.functional.softmax(player_1_logits, dim=-1)
                    #     player_1_entropy = torch.logsumexp(player_1_logits, dim=-1) - torch.sum(player_1_prob_dist * player_1_logits, dim=-1)
                    #     player_1_approxkl = 0.5 * (player_1_logprobs_diff**2).mean()
                    #     player_1_approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = player_1_approxkl
                    #     player_1_pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                    #         player_1_pg_clipfrac
                    #     )
                    #     player_1_pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = player_1_pg_loss
                    #     player_1_vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = player_1_vf_loss
                    #     player_1_vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                    #         player_1_vf_clipfrac
                    #     )
                    #     player_1_entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = player_1_entropy.mean()
                    #     player_1_ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = player_1_ratio.mean()
                
                        # del everything and empty cache
                        # fmt: off
                # del (
                #     player_1_output, player_1_vpred_temp, player_1_logits, player_1_new_all_logprobs, player_1_new_logprobs, player_1_vpred, player_1_vpredclipped,
                #     player_1_vf_losses1, player_1_vf_losses2, player_1_vf_loss, player_1_vf_clipfrac, player_1_logprobs_diff, player_1_ratio, player_1_pg_losses, player_1_pg_losses2,
                #     player_1_pg_loss_max,
                #     player_1_pg_loss, player_1_loss, # player_1_pg_clipfrac, player_1_prob_dist, player_1_entropy, player_1_approxkl, 
                #     player_1_mb_return,
                #     player_1_mb_advantage, player_1_mb_values, player_1_mb_responses, player_1_mb_query_responses, player_1_mb_logprobs,
                # )
                # fmt: on
                #torch.cuda.empty_cache()
                #gc.collect()

                # with torch.no_grad():
                #     player_0_mean_kl = player_0_kl.sum(1).mean()
                #     player_0_mean_entropy = (-player_0_logprob).sum(1).mean()
                #     player_0_mean_non_score_reward = player_0_non_score_reward.sum(1).mean()
                #     player_0_rlhf_reward = player_0_mean_non_score_reward + player_0_score.mean()
                #     player_0_eps = int(self.state.episode / (time.time() - start_time))
                #     player_0_metrics = {}
                #     player_0_metrics["player_0_eps"] = player_0_eps
                #     player_0_metrics["player_0_objective/kl"] = self.accelerator.gather_for_metrics(player_0_mean_kl).mean().item()
                #     player_0_metrics["player_0_objective/entropy"] = self.accelerator.gather_for_metrics(player_0_mean_entropy).mean().item()
                #     player_0_metrics["player_0_objective/non_score_reward"] = (
                #         self.accelerator.gather_for_metrics(player_0_mean_non_score_reward).mean().item()
                #     )
                #     player_0_metrics["player_0_objective/rlhf_reward"] = self.accelerator.gather_for_metrics(player_0_rlhf_reward).mean().item()
                #     player_0_metrics["player_0_objective/score"] = self.accelerator.gather_for_metrics(player_0_score.mean()).mean().item()
                #     player_0_metrics["player_0_policy/approxkl_avg"] = self.accelerator.gather_for_metrics(player_0_approxkl_stats).mean().item()
                #     player_0_metrics["player_0_policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(player_0_pg_clipfrac_stats).mean().item()
                #     player_0_metrics["player_0_loss/policy_avg"] = self.accelerator.gather_for_metrics(player_0_pg_loss_stats).mean().item()
                #     player_0_metrics["player_0_loss/value_avg"] = self.accelerator.gather_for_metrics(player_0_vf_loss_stats).mean().item()
                #     player_0_metrics["player_0_val/clipfrac_avg"] = self.accelerator.gather_for_metrics(player_0_vf_clipfrac_stats).mean().item()
                #     player_0_metrics["player_0_policy/entropy_avg"] = self.accelerator.gather_for_metrics(player_0_entropy_stats).mean().item()
                #     player_0_metrics["player_0_val/ratio"] = self.accelerator.gather_for_metrics(player_0_ratio_stats).mean().item()
                #     player_0_metrics["player_0_val/ratio_var"] = self.accelerator.gather_for_metrics(player_0_ratio_stats).var().item()
                #     player_0_metrics["player_0_val/num_eos_tokens"] = (player_0_response == processing_class.eos_token_id).sum().item()
                #     player_0_metrics["player_0_lr"] = self.lr_scheduler.get_last_lr()[0]
                #     player_0_metrics["player_0_episode"] = self.state.episode
                #     self.state.epoch = self.state.episode / num_games_to_train  # used by self.log # edit later
                #     self.state.global_step += 1
                #     self.log(player_0_metrics)

                #     # copy 2
                #     player_1_mean_kl = player_1_kl.sum(1).mean()
                #     player_1_mean_entropy = (-player_1_logprob).sum(1).mean()
                #     player_1_mean_non_score_reward = player_1_non_score_reward.sum(1).mean()
                #     player_1_rlhf_reward = player_1_mean_non_score_reward + player_1_score.mean()
                #     player_1_eps = int(self.state_2.episode / (time.time() - start_time))
                #     player_1_metrics = {}
                #     player_1_metrics["player_1_eps"] = player_1_eps
                #     player_1_metrics["player_1_objective/kl"] = self.accelerator.gather_for_metrics(player_1_mean_kl).mean().item()
                #     player_1_metrics["player_1_objective/entropy"] = self.accelerator.gather_for_metrics(player_1_mean_entropy).mean().item()
                #     player_1_metrics["player_1_objective/non_score_reward"] = (
                #         self.accelerator.gather_for_metrics(player_1_mean_non_score_reward).mean().item()
                #     )
                #     player_1_metrics["player_1_objective/rlhf_reward"] = self.accelerator.gather_for_metrics(player_1_rlhf_reward).mean().item()
                #     player_1_metrics["player_1_objective/score"] = self.accelerator.gather_for_metrics(player_1_score.mean()).mean().item()
                #     player_1_metrics["player_1_policy/approxkl_avg"] = self.accelerator.gather_for_metrics(player_1_approxkl_stats).mean().item()
                #     player_1_metrics["player_1_policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(player_1_pg_clipfrac_stats).mean().item()
                #     player_1_metrics["player_1_loss/policy_avg"] = self.accelerator.gather_for_metrics(player_1_pg_loss_stats).mean().item()
                #     player_1_metrics["player_1_loss/value_avg"] = self.accelerator.gather_for_metrics(player_1_vf_loss_stats).mean().item()
                #     player_1_metrics["player_1_val/clipfrac_avg"] = self.accelerator.gather_for_metrics(player_1_vf_clipfrac_stats).mean().item()
                #     player_1_metrics["player_1_policy/entropy_avg"] = self.accelerator.gather_for_metrics(player_1_entropy_stats).mean().item()
                #     player_1_metrics["player_1_val/ratio"] = self.accelerator.gather_for_metrics(player_1_ratio_stats).mean().item()
                #     player_1_metrics["player_1_val/ratio_var"] = self.accelerator.gather_for_metrics(player_1_ratio_stats).var().item()
                #     player_1_metrics["player_1_val/num_eos_tokens"] = (player_1_response == processing_class.eos_token_id).sum().item()
                #     player_1_metrics["player_1_lr"] = self.lr_scheduler_2.get_last_lr()[0]
                #     player_1_metrics["player_1_episode"] = self.state_2.episode
                #     self.state_2.epoch = self.state_2.episode / num_games_to_train  # used by self.log
                #     self.state_2.global_step += 1
                #     self.log(player_1_metrics)

                # save checkpoint
                my_output_dir = self.args.output_dir

                self.lr_scheduler.step()
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                if self.control.should_save:
                    self._save_checkpoint(model, trial=None)
                    self.control = self.callback_handler.on_save(self.args, self.state, self.control)
                # del player_0_kl, player_0_score, player_0_non_score_reward #, player_0_mean_kl, player_0_mean_entropy, player_0_mean_non_score_reward, player_0_metrics,

                self.args.output_dir = my_output_dir + "_2"

                # copy 2
                self.lr_scheduler_2.step()
                self.control_2 = self.callback_handler_2.on_step_end(args, self.state_2, self.control_2)
                if self.control_2.should_save:
                    self._save_checkpoint(model_2, trial=None)
                    self.control_2 = self.callback_handler_2.on_save(self.args, self.state_2, self.control_2)
                # del player_1_kl, player_1_score, player_1_non_score_reward #, player_1_mean_kl, player_1_mean_entropy, player_1_mean_non_score_reward, player_1_metrics,
                #torch.cuda.empty_cache()
                #gc.collect()

                self.args.output_dir = my_output_dir
                
                if args.num_sample_generations > 0 and (game_number - 1) % self.sample_generations_freq == 0:
                    self.generate_completions(sampling=True)
                    #torch.cuda.empty_cache()
                # del (
                #     player_0_query_response,
                #     player_0_response,
                #     player_0_postprocessed_response,
                #     player_0_logprob,
                #     player_0_value,
                #     player_0_sequence_length,
                #     player_0_contain_eos_token,
                #     player_0_sequence_length_p1,
                #     player_0_response_idxs,
                #     player_0_padding_mask,
                #     player_0_padding_mask_p1,
                #     player_0_reward,
                #     player_0_actual_start,
                #     player_0_actual_end,
                #     player_0_advantages,
                #     player_0_returns,
                # )

                # # copy 2
                # del (
                #     player_1_query_response,
                #     player_1_response,
                #     player_1_postprocessed_response,
                #     player_1_logprob,
                #     player_1_value,
                #     player_1_sequence_length,
                #     player_1_contain_eos_token,
                #     player_1_sequence_length_p1,
                #     player_1_response_idxs,
                #     player_1_padding_mask,
                #     player_1_padding_mask_p1,
                #     player_1_reward,
                #     player_1_actual_start,
                #     player_1_actual_end,
                #     player_1_advantages,
                #     player_1_returns,
                # )
                
                #torch.cuda.empty_cache()
                #gc.collect()

                if match_number >= 5 and current_match_step == 100:
                    game_ended = True
                    break

                if current_match_step == 100:
                    match_number += 1

                obs_all, _, _, _, _ = env.step({
                    "player_0": player_0_actions,
                    "player_1": player_1_actions
                })

        
        my_output_dir = self.args.output_dir

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        self.args.output_dir = my_output_dir + "_2"

        # copy 2
        self.control_2 = self.callback_handler_2.on_train_end(args, self.state_2, self.control_2)
        if self.control_2.should_save:
            self._save_checkpoint(model_2, trial=None, metrics=None)
            self.control_2 = self.callback_handler_2.on_save(self.args, self.state_2, self.control_2)
        
        self.args.output_dir = my_output_dir

    # convert answer to action
    def get_action_from_answer(self, answer):
        action = np.zeros((16, 3), dtype=int)
        answer = answer.split("\n")
        for i, unit_answer in enumerate(answer):
            if "Unit" in unit_answer:
                unit_number = int(unit_answer.split("Unit ")[1].split(":")[0])
                unit_action = unit_answer.split(": ")[1].split(",")
                if len(unit_action) == 1:
                    action[unit_number] = [int(unit_action[0]), 0, 0]
                else:
                    action[unit_number] = [int(unit_action[0]), int(unit_action[1]), int(unit_action[2])]
        return action

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