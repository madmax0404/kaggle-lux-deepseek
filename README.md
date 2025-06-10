# Kaggle
# NeurIPS 2024 - Lux AI Season 3
https://www.kaggle.com/competitions/lux-ai-season-3

---
## 기술 스택 (Tech Stack)

* **Language**: Python
* **LLM Model**: DeepSeek-R1-Distill-Qwen-1.5B (https://huggingface.co/deepseek-ai/DeepSeek-R1#deepseek-r1-distill-models)
* **LLM Library**: HuggingFace Transformers (https://huggingface.co/docs/transformers/index)
* **Deep Learning Framework**: PyTorch
* **LLM Reinforcement Learning (강화학습) Library**: HuggingFace TRL (https://huggingface.co/docs/trl/index)
* **Reinforcement Learning Algorithm (강화학습 알고리즘)**: PPO (https://huggingface.co/docs/trl/ppo_trainer)
* **Numerical Computing**: NumPy
* **Distributed Training Library**: HuggingFace Accelerate (https://huggingface.co/docs/transformers/accelerate)
* **OS**: Linux (Ubuntu Desktop 24.04 LTS)
* **IDE**: VSCode, Jupyter Notebook

---

## 1. Project Overview

Lux AI Challenge hosted a competition on Kaggle that the goal was to create and/or train AI bots to play a novel multi-agent 1v1 game against other submitted agents. This project aimed to fine-tune a small DeepSeek model via reinforcement learning on my local computer to make it compete in the competition.

## 2. Problem Statement

The participants of the competition were to create an agent for the game (either via developing algorithms or via reinforcement learning) and compete each other. Each player's agents are placed on a 24x24 tile map where they control their ships to collect resources. A game has 5 matches and each match has 100 steps. The player with more resources at the 100th step wins the match and whoever wins 3 matches first wins the game. The game also included various other mechanics like attacking enemy ships(called sap in the description), fog of war, and different tile types.

See here for detailed game mechanics: https://www.kaggle.com/competitions/lux-ai-season-3/overview

The key challenges were:
  * Optimizing multi-variable resource gathering and allocation strategies.
  * Developing an agent capable of adapting to random game mechanics, including map generation and vision reduction caused by nebula tiles.
  * Balancing exploration and exploitation strategies to gather points effectively while anticipating opponent moves.

The problem was notably complex, characterized by a large state space and the need for long-term planning. Adding to this complexity were distributed decision-making across multiple units, intricate resource management at the unit level, and a highly dynamic environment. Further challenges arose from partial observability, a need for multi-variable optimization, randomized game mechanics, and a complex action space with intricate interactions, all while demanding robust opponent modeling and real-time decision-making.

## 3. Dataset

The core files can be downloaded from the below competition page. The page also describes how to install the game engine library. The game environment can be generated using the game engine.

https://www.kaggle.com/competitions/lux-ai-season-3/data

## 4. Methodology & Approach

Since it was not a trivial task, data exploration, trial and error were done throughout the project. However, the works may be divided into the following parts:

1.  **Data Loading & Initial Exploration:**
    * Loaded and understood the various observations, actions, and game parameters.
    * Performed initial sanity checks and basic statistical analyses.


























