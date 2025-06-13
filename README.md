# NeurIPS 2024 - Lux AI Season 3 – LLM Agent (DeepSeek-R1-Distill-Qwen-1.5B)
### https://www.kaggle.com/competitions/lux-ai-season-3

## Tech Stack (기술 스택)
* **Programming Language**: Python
* **Machine Learning Framework**: PyTorch
* **Large Language Model**: DeepSeek-R1-Distill-Qwen-1.5B – a distilled 1.5B-parameter Qwen model from DeepSeek-AI. This compact LLM was chosen for its strong reasoning capabilities relative to size, achieved by distillation from a larger RL-trained model. (https://openlaboratory.ai/models/deepseek-r1-qwen-1_5b)
* **LLM Libraries**: Hugging Face Transformers for model integration, with Accelerate for device management and BitsAndBytes for 4-bit quantization (to efficiently load the model under GPU memory constraints).
* **Reinforcement Learning**: Hugging Face TRL (Transformer Reinforcement Learning) library using the Proximal Policy Optimization (PPO) algorithm. This allowed us to fine-tune the LLM with reward signals.
* **Environment**: Lux AI Season 3 game environment (luxai_s3 Python package) for simulation. The environment is JAX-based but wrapped for Python usage, providing the game’s state and reward mechanics.
* **Tooling & Platform**: Jupyter Notebooks (Kaggle Notebooks) and VS Code for development and experimentation. Training was conducted on an Ubuntu Linux system with CUDA support for GPU acceleration.

---

# 한국어

## 개요
Lux AI 시즌 3는 Kaggle에서 진행되는 NeurIPS 2024 대회로, 참가자들은 복잡한 1대1 자원 수집 게임을 플레이하는 AI 봇을 개발합니다. 이 프로젝트는 대규모 언어 모델(LLM)을 경쟁용 AI 에이전트의 핵심으로 사용하는 실험적인 탐구입니다.

일반적인 전문화된 신경망 대신, 저는 15억 개의 매개변수를 가진 LLM(DeepSeek-R1-Distill-Qwen-1.5B)을 강화 학습(PPO)을 통해 미세 조정하여 게임 에이전트를 제어했습니다. 목표는 복잡한 다중 에이전트 환경에서 LLM 기반 전략 에이전트의 실현 가능성을 조사하는 것이었습니다. 이 틀에 얽매이지 않는 접근 방식은 전략 게임 AI에 LLM을 적용하는 데 있어 잠재력과 과제를 모두 강조하는 개념 증명(proof-of-concept) 역할을 합니다.

## 문제
Lux AI 시즌 3 게임은 두 명의 플레이어가 24x24 격자 타일 위에서 경쟁하며, 각 플레이어는 함대를 조종하여 맵 곳곳에 흩어져 있는 에너지 자원을 수집합니다. 전체 게임은 최대 5번의 매치로 구성되며, 각 매치는 100턴 동안 진행됩니다. 5번의 매치 중 3번을 먼저 이기는 플레이어가 게임에서 승리합니다.

주요 게임 메커니즘은 다음과 같습니다.
* **자원 채굴**: 맵에서 에너지를 수집합니다.
* **보관소 관리**: 채굴한 에너지를 효율적으로 저장하고 활용합니다.
* **선박 이동**: 함선을 맵 위에서 움직입니다.
* **전투**: 선박이 적 선박과 충돌하여 에너지를 '흡수'할 수 있습니다.
또한, 환경은 **전장의 안개(성운 타일을 통한 제한된 시야)**와 **다양한 타일 유형(예: 이동을 방해하는 소행성)**을 특징으로 합니다.

이 게임을 위한 성공적인 에이전트를 설계하는 것은 여러 요인 때문에 매우 어렵습니다. 광대한 상태 공간 및 부분 관측 가능성: 각 선박은 센서 범위가 제한적이어서 에이전트는 불완전한 정보로 결정을 내려야 합니다. 맵은 에피소드마다 무작위로 생성되므로 불확실성이 추가되고 적응력이 필요합니다.
* **복잡한 행동 공간**: 각 선박은 네 방향 이동, 채굴, 공격 등 다양한 행동을 할 수 있으며, 여러 선박이 동시에 행동합니다. 함대 전체의 행동을 조정하는 것은 조합론적 복잡성을 야기하며 장기적인 계획이 필요합니다.
* **자원 관리**: 에이전트는 탐사(새로운 자원이나 상대를 찾는 것)와 활용(알려진 자원을 효율적으로 수확하는 것) 사이에서 균형을 맞춰야 합니다. 수집된 에너지는 승리 조건이자 행동을 위한 통화이므로 전략적인 할당이 중요합니다.
* **상대방과의 상호작용**: 이 게임은 대결형 게임입니다. 성공은 상대방의 움직임을 예측하고, 자원을 방어하며, 기회주의적인 전투를 수행하는 데 달려 있습니다. 이는 강력한 상대방 모델링과 실시간으로 전략을 조정하는 능력을 요구합니다.

이러한 도전 과제들은 이 문제를 매우 복잡하게 만들며, 일반적으로 전문화된 강화 학습(RL) 에이전트나 휴리스틱 알고리즘으로 해결됩니다. 저희 프로젝트는 범용 LLM 에이전트로 이러한 복잡성을 처리함으로써 기존의 한계를 뛰어넘고자 합니다.

---

# English

## Overview
Lux AI Season 3 is a NeurIPS 2024 competition on Kaggle where participants develop AI bots to play a complex 1v1 resource-gathering game. This project is an experimental exploration of using a large language model (LLM) as the core of an AI agent for the competition. Instead of the typical specialized neural networks, I fine-tuned a 1.5 billion-parameter LLM (DeepSeek-R1-Distill-Qwen-1.5B) via reinforcement learning (PPO) to control the game agent. The goal was to investigate the viability of an LLM-based strategy agent in a complex, multi-agent environment. This unconventional approach serves as a proof-of-concept, highlighting both the potential and challenges of applying LLMs to strategic game AI.

## Problem Statement (Lux AI Game)
In the Lux AI Season 3 game, two players compete on a 24x24 grid of tiles. Each player controls a fleet of ships to collect energy resources scattered across the map. A full game consists of up to 5 matches, each 100 turns long; the player who wins 3 out of 5 matches wins the game. Key game mechanics include: resource mining, deposit management, ship movement, and combat (ships can “sap” energy from enemy ships by colliding). The environment also features fog of war (limited visibility via nebula tiles) and varied tile types (e.g. asteroids that block movement).

Designing a successful agent for this game is challenging due to several factors:
* **Large State Space & Partial Observability**: Each ship has a limited sensor range, so agents must make decisions with incomplete information. The map is randomly generated each episode, adding uncertainty and requiring adaptability.
* **Complex Action Space**: Each ship can take different actions (move in four directions, mine, attack, etc.), and multiple ships act simultaneously. Coordinating actions across a fleet introduces combinatorial complexity and a need for long-term planning.
* **Resource Management**: Agents must balance exploration (searching for new resources or opponents) and exploitation (efficiently harvesting known resources). Energy gathered is both the win condition and the currency for actions, so strategic allocation is critical.
* **Opponent Interaction**: The game is adversarial – success depends on anticipating opponent moves, defending resources, and opportunistic combat. This demands robust opponent modeling and the ability to adapt strategy in real-time.

These challenges make the problem notably complex, typically tackled with specialized RL agents or heuristic algorithms. Our project pushes the envelope by attempting to handle this complexity with a generalist LLM agent.

## Approach & Methodology
Generally, past solutions use custom neural networks (e.g. CNNs or MLPs) tailored to game state features. In contrast, this project uses an LLM-based agent, which required careful problem re-formulation and training strategy.

The process can be divided into a few major steps:
* **Environment Understanding & Data Exploration**: I first integrated the Lux AI environment into our pipeline and performed extensive exploration of its observations and mechanics. This involved loading the game engine and observing state representations (maps, ship statuses, sensor inputs). I conducted sanity checks and statistical analysis on game data (e.g., distribution of resource nodes, typical ship counts, etc.) to ground our intuition. Understanding the observation structure was crucial, since I needed to convert these raw features into a format suitable for an LLM.
* **LLM Agent Design & Prompt Engineering**: A core challenge was mapping the structured game state into natural language or another sequential input format for the LLM. I designed a prompt schema that encodes relevant information about the game state at each turn. For example, the prompt might include summaries of each ship’s sensor readings, current energy, and nearby resources or threats. The LLM was expected to output an action decision, which I decoded into game commands. This step was essentially prompt engineering – crafting the input-output specification so that the language model could interpret the game situation and propose valid actions. I kept the prompts concise due to token limitations, and iteratively refined the format based on the model’s responses (e.g., ensuring the model’s output syntax matched the game’s expected action format).
* **Reinforcement Learning Fine-Tuning (PPO Self-Play)**: With the LLM integrated, I fine-tuned it using reinforcement learning. I leveraged the Hugging Face TRL library’s PPO implementation, which allowed us to update the model’s weights based on a reward signal rather than supervised labels. The reward was derived from game outcomes – encouraging the model to choose actions that lead to higher energy gains and wins. To stabilize training, I employed self-play: two instances of the LLM agent played against each other in simulated matches. Self-play provided a curriculum of increasingly challenging scenarios, as the agent effectively learned from playing against its current skill level. After each game (or batch of games), the model’s policy was updated via PPO, using the difference in outcome (win/loss or intermediate score) as feedback. Key hyperparameters included a small batch size and frequent model updates, given the high variance in game outcomes. I also utilized techniques like reward shaping (assigning intermediate rewards for collecting resources or destroying enemy ships) to guide the learning process in such a sparse reward environment. Throughout training, I monitored metrics such as total reward per episode, win rates, and policy loss, adjusting parameters to prevent divergence.
* **Experimentation & Iteration**: As this was an experimental project, a lot of iterative tuning was involved. I experimented with different prompt formats, model hyperparameters, and training setups (for instance, testing both fully online training and a replay buffer of past game states). I also tried various strategies to deal with the LLM’s verbosity and stochasticity – such as constraining the action vocabulary and using shorter generation lengths for decisions. Each iteration revealed insights that informed the next: for example, early tests showed the vanilla LLM often produced invalid or suboptimal actions, which led us to refine the prompt instructions and incorporate simple validation rules on the LLM’s output.

## Results & Insights
**Performance**: Given the novelty of this approach, the final agent was not yet a top contender in the competition, but it demonstrated learning progress and provided valuable insights. Over the course of training, the LLM agent began to show reasonable behaviors – for instance, improving its resource collection efficiency and avoiding obviously bad moves (like crashing ships needlessly). The self-play setup led to a gradual increase in average reward per game, indicating the agent was learning from experience. I observed the win-rate of one LLM agent against its earlier versions improving over time, suggesting that the policy was indeed evolving. However, the agent’s skill leveled off below that of specialized rule-based agents, reflecting the difficulty for a small LLM to master the full complexity of the game with limited training.

### Key Findings:
**This experiment highlighted several points:**
* **Feasibility of LLM Agents**: An LLM with relevant reasoning skills can be adapted (via RL fine-tuning) to make decisions in a game environment. This is encouraging, as it opens the door to using a single general model for diverse tasks. The DeepSeek-R1-Distill-Qwen-1.5B model, distilled from a reasoning-focused parent model, brought useful priors (like logical reasoning and planning) into the game scenario. I found that these priors helped the agent reason about multi-step outcomes to some extent.
* **Prompt Design is Critical**: The way information was presented to the LLM greatly affected its performance. A well-crafted prompt that highlights important features (e.g., “Ship A at (3,5) low energy, enemy nearby”) yielded much better decisions than a raw or overly verbose description. This underlines that, for LLM-based agents, prompt engineering becomes as important as network architecture is for traditional agents.
* **Challenges of RL with LLMs**: Training instability and sample inefficiency were significant hurdles. Language models tend to be high-variance in their outputs, so the reward signal was noisy. I mitigated this with a large number of self-play games and by smoothing rewards, but stability remained an issue. Additionally, the credit assignment (attributing a win or loss to specific actions) is hard in long games. Techniques like reward shaping were necessary but had to be tuned carefully to not mislead the agent.
* **Scalability and Resources**: Running an LLM (even a 1.5B parameter one) inside a game loop is computationally expensive. I used 4-bit quantization and optimized PyTorch settings to maximize inference speed. Even so, training was slow compared to lightweight models. This experiment underscores the need for efficient training frameworks when applying large models to reinforcement learning scenarios.

**Conclusion**: This project served as a proof-of-concept that reinforcement learning can fine-tune a language model to operate in a complex game environment. While the LLM agent did not outperform specialized solutions, it validated the concept that an LLM can learn game strategies via trial and error. The work is highly experimental, but it provides a stepping stone for future research. With larger models or more advanced techniques (such as combining with planning algorithms or using human feedback for reward tuning), LLM-based agents might become more competitive in complex environments. In summary, this repository showcases an innovative integration of modern LLMs with reinforcement learning in a gaming context. It emphasizes the technical stack (Hugging Face ecosystem, PPO, JAX environment) and the methodology of training a novel agent. The insights gained here contribute to understanding how far I can push LLMs beyond traditional language tasks – highlighting both their strengths in reasoning and the practical challenges when applied to dynamic, embodied decision-making tasks.

## Project Structure
    kaggle-lux-deepseek/
    ├── Notebooks/
    │   ├── Agent_Development/                        # 에이전트 개발 및 실험
    │   │   ├── DeepSeek-R1-Distill-Qwen-1.5B*.ipynb  # 모델별 실험 노트북
    │   │   ├── agent/                                # 핵심 에이전트 구현
    │   │   │   ├── agent.py                          # 메인 에이전트 로직
    │   │   │   └── train.py                          # 훈련 스크립트
    │   │   └── lux/                                  # Lux AI 환경 래퍼
    │   ├── EDA/                                      # 탐색적 데이터 분석
    │   └── Testing_Agents/                           # 에이전트 테스트 및 검증

