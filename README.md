# NeurIPS 2024 — Lux AI Season 3 — LLM Agent (DeepSeek-R1-Distill-Qwen-1.5B)

Kaggle Competition
[https://www.kaggle.com/competitions/lux-ai-season-3](https://www.kaggle.com/competitions/lux-ai-season-3)

---

## Overview

**Lux AI Season 3** is a NeurIPS 2024 competition hosted on Kaggle where participants build AI bots to play a complex 1-v-1 resource-gathering strategy game. This project is an experimental exploration of using a **Large Language Model (LLM)** as the **core** of a competitive game agent.

Instead of a specialized neural network, I fine-tuned a **1.5B-parameter LLM (DeepSeek-R1-Distill-Qwen-1.5B)** with **reinforcement learning (PPO)** to act as the game agent. The goal was to examine the feasibility of an **LLM-based strategic agent** in a complex, partially observable, multi-agent environment. This unorthodox direction serves as a **proof of concept**, highlighting both the **potential** and the **challenges** of applying LLMs to strategy-game AI.

---

## Tech Stack

* **Language:** Python
* **Deep Learning Framework:** PyTorch
* **Large Language Model:** **DeepSeek-R1-Distill-Qwen-1.5B** — a distilled 1.5B-parameter Qwen model from DeepSeek-AI. Chosen for strong reasoning per parameter, distilled from a larger RL-trained model.
  [https://openlaboratory.ai/models/deepseek-r1-qwen-1_5b](https://openlaboratory.ai/models/deepseek-r1-qwen-1_5b)
* **LLM Libraries:** Hugging Face Transformers for integration, Accelerate for device management, BitsAndBytes for **4-bit quantization** (to fit within GPU memory).
* **Reinforcement Learning:** Hugging Face **TRL** (Transformer Reinforcement Learning) using **PPO** for reward-driven fine-tuning of the LLM.
* **Game Environment:** Lux AI Season 3 (`luxai_s3` Python package). The engine is JAX-based but wrapped for Python, exposing the game state and reward mechanics.
* **Tools & Platform:** Jupyter Notebooks (including Kaggle Notebooks) and VS Code. Training on Ubuntu Linux with CUDA-enabled GPUs.
* **Visualization:** TensorBoard
* **OS:** Linux (Ubuntu Desktop 24.04 LTS)

---

## Problem

Lux AI Season 3 is played on a **24×24** grid. Each player controls a **fleet** that gathers **energy resources** scattered across the map. A full game comprises up to **five matches**, each lasting **100 turns**; the first to win **three** matches wins the game.

> ▶️ **[View Example Replay](Notebooks/Agent_Development/replay_my_agent.html)** — download and open locally.

Key mechanics:

* **Resource Mining:** Collect energy across the map.
* **Storage & Management:** Efficiently store and spend harvested energy.
* **Ship Movement:** Navigate ships across tiles.
* **Combat:** Collisions can “absorb” energy from enemy ships.

The environment features **fog of war** (limited vision via nebula tiles) and **diverse tile types** (e.g., asteroids that block movement).

Designing a strong agent is difficult due to:

* **Large State Space & Partial Observability:** Each ship has limited sensor range. Maps are randomized per episode, increasing uncertainty and requiring adaptation.
* **Complex Action Space:** Each ship can move (four directions), mine, attack, etc., and multiple ships act simultaneously. Coordinating the fleet creates combinatorial complexity and demands long-horizon planning.
* **Resource Management:** Balance **exploration** (find new resources/enemies) with **exploitation** (harvest known resources efficiently). Energy is both a win condition and the currency for actions.
* **Adversarial Interaction:** Success depends on modeling the opponent, defending resources, and opportunistic engagements—requiring strong opponent modeling and real-time strategy updates.

Traditionally, such problems are tackled with specialized RL agents or heuristics. Here, we push beyond that by **handling the complexity with a general-purpose LLM agent**.

---

## Approach & Methodology

Conventional solutions employ task-specific networks (e.g., CNN/MLP tuned to the game state). In contrast, this project introduces an **LLM-based agent**, which required reframing the problem and designing a new training strategy.

> ![Model Architecture](images/Screenshot%20from%202025-06-21%2014-44-04.png)
> ▲ DeepSeek-R1-Distill-Qwen-1.5B architecture

The process comprised several stages:

* **Feasibility Check:**
  First, I validated whether an LLM can play a complex game at all. Using prompt engineering, I tested browser-based models (e.g., GPT-4o, DeepSeek-R1) to play simplified versions, confirming **basic feasibility**.

  > ![Proof-of-Concept](images/Screenshot%20from%202025-06-14%2013-48-32.png)
  > ▲ Proof-of-concept: early feasibility validation

* **Environment Understanding & Data Exploration:**
  Integrated the Lux AI environment, inspected observations and mechanics extensively, and ran sanity checks and statistics (e.g., resource node distributions, typical fleet sizes). Understanding the **raw observation structure** was crucial to convert features into an **LLM-friendly** format.

* **LLM Agent Design & Prompt Engineering:**
  The core challenge: mapping **structured game state** to a **sequence input** the LLM can understand. I designed a **turn-wise prompt schema** encoding ship sensor info, current energy, nearby resources/threats, and other key context.
  The LLM outputs **action decisions**, which are **decoded** into game commands. This is essentially **prompt engineering**: specifying I/O formats so the LLM interprets the situation and proposes valid actions. Due to token budgets, prompts were kept concise and iteratively refined (e.g., ensuring the output grammar matches the expected action format).

  > ![Prompt Engineering Example](images/Screenshot%20from%202025-06-14%2012-38-21.png)
  > ▲ Prompt-engineering example

* **RL Fine-Tuning with PPO (Self-Play):**
  After integration, the LLM was fine-tuned via **PPO** using Hugging Face TRL. Rather than supervised labels, the model updated from **reward signals** derived from game results. To stabilize training, I adopted **self-play**, pitting two LLM agents against each other so difficulty scaled with the agent’s current skill. After each game or batch, PPO updated the policy using win/loss and intermediate scores.
  Given high outcome variance, I used smaller batch sizes with frequent updates. **Reward shaping** (e.g., intermediate rewards for resource collection or enemy destruction) helped in the sparse-reward setting. Metrics such as episode return, win rate, and policy loss were monitored continuously; parameters were tuned to prevent divergence.

* **Experimentation & Iteration:**
  Being a research-style project, iterative tuning was essential: different prompt formats, model hyperparameters, and training setups (pure online vs. replay buffers of past states) were tested. To curb verbosity and stochastic drift, I constrained the **action vocabulary** and enforced **short outputs**. Early failures (e.g., invalid or inefficient actions) informed prompt rules and lightweight output validators.

---

## Results & Key Observations

**Performance:**
While this novel approach did not reach top leaderboard positions, training showed **steady progress**. Over time, the LLM agent exhibited **increasingly sensible behavior**—improving resource collection and avoiding clearly poor actions (e.g., gratuitous collisions).
Self-play led to a **gradual rise in average episode reward** and **win rate** against earlier checkpoints, indicating genuine policy improvement. That said, the final agent remained weaker than specialized rules-based or task-specific RL agents—reflecting the difficulty for a **small LLM** to fully master the game under limited training.

### Key Findings

* **Feasibility of LLM Agents:**
  With sufficient **reasoning ability**, an LLM can be adapted via RL to make decisions in a game environment. The distilled DeepSeek-R1-Distill-Qwen-1.5B—trained for reasoning—carried useful prior structure for **logical planning**, aiding multi-step inference in the game.

* **Prompt Design Matters:**
  How information is **presented** to the LLM heavily affects performance. Prompts that highlight key features (e.g., “Ship A at (3,5) low energy, enemy nearby”) consistently outperformed overly simple or verbose descriptions. In LLM-based agents, **prompt engineering is as critical** as network design in conventional agents.

* **LLM + RL is Hard:**
  Training instability and **sample inefficiency** were major hurdles. LLM outputs have high variance, which translates into noisy reward signals. Massive self-play and **reward smoothing** helped but didn’t fully solve stability. Credit assignment in long games was also non-trivial; reward shaping helps, but mis-shaping can mislead the agent.

* **Scalability & Resources:**
  Even at **1.5B parameters**, running the model in the game loop is computationally expensive. 4-bit quantization and PyTorch optimizations improved throughput, but training remained slower than with compact specialized models. This underscores the need for **efficient training frameworks** when applying larger models to RL.

**Conclusion:**
This project demonstrates that an **LLM can be fine-tuned with RL** to act in a complex environment—serving as a **proof of concept**. Although performance trails specialized solutions, the agent **learned strategy via trial and error**. It lays groundwork for future research: larger models, integration with **planning algorithms**, or **human-feedback-based** reward shaping could make LLM agents far more competitive in complex domains.

In short, this repository showcases an innovative integration of **state-of-the-art LLMs and RL** in a dynamic game setting (Hugging Face ecosystem, PPO, and a JAX-backed environment). It highlights both the **strengths** of LLM reasoning and the **practical limits** encountered when applying LLMs to embodied decision-making.

---

## How to Reproduce

1. **Clone the repository**

   ```bash
   git clone https://github.com/madmax0404/kaggle-lux-deepseek.git
   cd kaggle-lux-deepseek
   ```

2. **Download the dataset**

   * Join the competition: **NeurIPS 2024 — Lux AI Season 3**
     [https://www.kaggle.com/competitions/lux-ai-season-3](https://www.kaggle.com/competitions/lux-ai-season-3)
   * Download the data and place it in the appropriate directory.

3. **Create a virtual environment & install dependencies**

   ```bash
   conda create -n kaggle_lux_deepseek python=3.12  # or venv
   conda activate kaggle_lux_deepseek
   pip install -r requirements.txt
   ```

4. **Run Jupyter Notebook**

   ```bash
   jupyter notebook Notebooks
   ```

   Follow the notebooks to run preprocessing, training, and evaluation.

---

## Project Structure

```
kaggle-lux-deepseek/
├── Notebooks/
│   ├── Agent_Development/                        # Agent development & experiments
│   │   ├── Modified_PPO_Trainer/                 # Custom tweaks to HuggingFace PPO Trainer
│   │   └── DeepSeek-R1-Distill-Qwen-1.5B*.ipynb  # Model-specific experiment notebooks
│   └── EDA/                                      # Exploratory data analysis
└── images/
```

---

## Acknowledgements

Thanks to the **Lux AI Challenge** and **Kaggle** for the dataset and competition platform.

This project was supported by the following open-source tools: Python, PyTorch, DeepSeek, Hugging Face, TensorBoard, pandas, numpy, matplotlib, seaborn, Jupyter, SciPy.

All data usage complies with the competition rules and licenses.

---

## License

Code © 2025 **Jongyun Han (Max)**. Released under the **MIT License**.
See the LICENSE file for details.

**Note:** Datasets are **NOT** redistributed in this repository.
Please download them from the official Kaggle competition page and comply with the competition rules/EULA.