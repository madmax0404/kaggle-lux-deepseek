# NeurIPS 2024 - Lux AI Season 3 – LLM 에이전트 (DeepSeek-R1-Distill-Qwen-1.5B)

Kaggle 대회

https://www.kaggle.com/competitions/lux-ai-season-3

---

## 개요
Lux AI 시즌 3는 Kaggle에서 진행되는 NeurIPS 2024 대회로, 참가자들은 복잡한 1대1 자원 수집 게임을 플레이하는 AI 봇을 개발합니다. 이 프로젝트는 대규모 언어 모델(LLM)을 경쟁용 AI 에이전트의 핵심으로 사용하는 실험적인 탐구입니다.

일반적인 전문화된 신경망 대신, 저는 15억 개의 매개변수를 가진 LLM(DeepSeek-R1-Distill-Qwen-1.5B)을 강화 학습(PPO)을 통해 파인 튜닝하여 게임 에이전트를 만들었습니다. 목표는 복잡한 다중 에이전트 환경에서 LLM 기반 전략 에이전트의 실현 가능성을 조사하는 것이었습니다. 이 틀에 얽매이지 않는 접근 방식은 전략 게임 AI에 LLM을 적용하는 데 있어 잠재력과 과제를 모두 강조하는 개념 증명(proof-of-concept) 역할을 합니다.

---

## 기술 스택
* **언어**: Python
* **딥러닝 프레임워크**: PyTorch
* **Large Language Model**: DeepSeek-R1-Distill-Qwen-1.5B – a distilled 1.5B-parameter Qwen model from DeepSeek-AI. This compact LLM was chosen for its strong reasoning capabilities relative to size, achieved by distillation from a larger RL-trained model. (https://openlaboratory.ai/models/deepseek-r1-qwen-1_5b)
* **LLM 라이브러리들**: Hugging Face Transformers for model integration, with Accelerate for device management and BitsAndBytes for 4-bit quantization (to efficiently load the model under GPU memory constraints).
* **강화 학습**: Hugging Face TRL (Transformer Reinforcement Learning) library using the Proximal Policy Optimization (PPO) algorithm. This allowed us to fine-tune the LLM with reward signals.
* **게임 환경**: Lux AI Season 3 game environment (luxai_s3 Python package) for simulation. The environment is JAX-based but wrapped for Python usage, providing the game’s state and reward mechanics.
* **툴 및 플랫폼**: Jupyter Notebooks (Kaggle Notebooks) and VS Code for development and experimentation. Training was conducted on an Ubuntu Linux system with CUDA support for GPU acceleration.
* **시각화**: TensorBoard
* **운영체제**: Linux (Ubuntu Desktop 24.04 LTS)

---

## 문제
Lux AI 시즌 3 게임은 두 명의 플레이어가 24x24 격자 타일 위에서 경쟁하며, 각 플레이어는 함대를 조종하여 맵 곳곳에 흩어져 있는 에너지 자원을 수집합니다. 전체 게임은 최대 5번의 매치로 구성되며, 각 매치는 100턴 동안 진행됩니다. 5번의 매치 중 3번을 먼저 이기는 플레이어가 게임에서 승리합니다.

> ▶️ [예시 리플레이 보기](<Notebooks/Agent_Development/replay_my_agent.html>): 보려면 다운받아서 실행해 주세요.

주요 게임 메커니즘은 다음과 같습니다.
* **자원 채굴**: 맵에서 에너지를 수집합니다.
* **보관소 관리**: 채굴한 에너지를 효율적으로 저장하고 활용합니다.
* **선박 이동**: 함선을 맵 위에서 움직입니다.
* **전투**: 선박이 적 선박과 충돌하여 에너지를 '흡수'할 수 있습니다.
또한, 환경은 **전장의 안개**(성운 타일을 통한 제한된 시야)와 **다양한 타일 유형**(예: 이동을 방해하는 소행성)을 특징으로 합니다.

이 게임을 위한 성공적인 에이전트를 설계하는 것은 여러 요인 때문에 매우 어렵습니다. 광대한 상태 공간 및 부분 관측 가능성: 각 선박은 센서 범위가 제한적이어서 에이전트는 불완전한 정보로 결정을 내려야 합니다. 맵은 에피소드마다 무작위로 생성되므로 불확실성이 추가되고 적응력이 필요합니다.
* **복잡한 행동 공간**: 각 선박은 네 방향 이동, 채굴, 공격 등 다양한 행동을 할 수 있으며, 여러 선박이 동시에 행동합니다. 함대 전체의 행동을 조정하는 것은 조합론적 복잡성을 야기하며 장기적인 계획이 필요합니다.
* **자원 관리**: 에이전트는 탐사(새로운 자원이나 상대를 찾는 것)와 활용(알려진 자원을 효율적으로 수확하는 것) 사이에서 균형을 맞춰야 합니다. 수집된 에너지는 승리 조건이자 행동을 위한 통화이므로 전략적인 할당이 중요합니다.
* **상대방과의 상호작용**: 이 게임은 대결형 게임입니다. 성공은 상대방의 움직임을 예측하고, 자원을 방어하며, 기회주의적인 전투를 수행하는 데 달려 있습니다. 이는 강력한 상대방 모델링과 실시간으로 전략을 조정하는 능력을 요구합니다.

이러한 도전 과제들은 이 문제를 매우 복잡하게 만들며, 일반적으로 전문화된 강화 학습(RL) 에이전트나 휴리스틱 알고리즘으로 해결됩니다. 저희 프로젝트는 범용 LLM 에이전트로 이러한 복잡성을 처리함으로써 기존의 한계를 뛰어넘고자 합니다.

---

## 접근 방식 및 방법론

일반적으로, 기존의 솔루션들은 게임 상태의 특성에 맞게 커스텀 신경망(CNN이나 MLP 등)을 사용합니다. 반면, 이 프로젝트에서는 LLM(대형 언어 모델) 기반 에이전트를 도입했으며, 이를 위해 문제 재정의와 훈련 전략 수립이 필요했습니다.

> ![Model Architecture](<images/Screenshot from 2025-06-21 14-44-04.png>)
> 
> ▲DeepSeek-R1-Distill-Qwen-1.5B 아키텍쳐

프로세스는 몇 가지 주요 단계로 나눌 수 있습니다:

* **아이디어의 실현 가능성 검즘**: 먼저, LLM이 이런 복잡한 게임을 플레이하는 것이 가능한지를 검증할 필요가 있었습니다. 웹 브라우저용 GPT-4o, DeepSeek-R1 등의 모델들에게 프롬프트 엔지니어링으로 게임 플레이를 시켜봤고, 충분히 실현 가능하다는 것을 확인했습니다.

> ![Proof-of-Concept](<images/Screenshot from 2025-06-14 13-48-32.png>)
>
> ▲아이디어 실현 가능성 검증 예시

* **환경 이해 및 데이터 탐색**: Lux AI 환경을 파이프라인에 통합하고, 관측값과 메커니즘을 광범위하게 탐색했습니다. 게임 엔진을 로드하여 상태 표현(맵, 선박 상태, 센서 입력 등)을 관찰하고, 직관을 얻기 위해 데이터에 대한 sanity check 및 통계 분석(예: 자원 노드 분포, 일반적인 선박 수 등)을 진행했습니다. 관측 구조를 정확히 이해하는 것이 매우 중요했는데, 이는 raw feature들을 LLM에 적합한 형태로 변환해야 했기 때문입니다.

* **LLM 에이전트 설계 및 프롬프트 엔지니어링**: 가장 핵심적인 과제는 구조화된 게임 상태를 자연어 또는 LLM이 이해할 수 있는 시퀀스 입력 형태로 매핑하는 것이었습니다. 각 턴마다 게임 상태에 관한 정보를 담는 프롬프트 스키마를 설계했으며, 예를 들어 각 선박의 센서 정보, 현재 에너지, 주변 자원 및 위협 요약 등이 프롬프트에 포함되었습니다. LLM은 행동 결정을 출력하며, 이는 다시 게임 명령으로 디코딩되었습니다. 이 과정은 본질적으로 프롬프트 엔지니어링으로, 언어 모델이 게임 상황을 해석하고 올바른 행동을 제안할 수 있도록 입출력 사양을 설계하는 일이었습니다. 토큰 제한 때문에 프롬프트를 최대한 간결하게 유지했으며, 모델의 반응을 보면서 포맷을 반복적으로 개선했습니다(예: 모델의 출력 문법이 게임의 예상 액션 포맷과 일치하는지 등).

> ![Prompt Engineering Example](<images/Screenshot from 2025-06-14 12-38-21.png>)
>
> ▲프롬프트 엔지니어링 예시

* **강화학습 파인튜닝(PPO 셀프플레이)**: LLM을 통합한 뒤에는 강화학습으로 파인튜닝을 진행했습니다. Hugging Face의 TRL 라이브러리의 PPO 구현을 활용하여, 지도 학습 레이블 대신 보상 신호를 기반으로 모델 가중치를 업데이트했습니다. 보상은 게임 결과에서 추출되었으며, 에너지 획득과 승리에 기여하는 행동을 유도했습니다. 훈련의 안정화를 위해 셀프플레이를 도입했는데, 두 LLM 에이전트 인스턴스가 시뮬레이션된 경기에서 서로 대결하며, 현재 실력에 맞는 점진적으로 어려워지는 시나리오에서 학습할 수 있도록 했습니다. 각 경기(또는 게임 배치) 이후, PPO로 정책을 업데이트했고, 결과(승패 또는 중간 점수)의 차이를 피드백으로 사용했습니다. 주요 하이퍼파라미터로는 게임 결과의 분산이 크기 때문에 작은 배치 사이즈와 잦은 업데이트를 사용했습니다. 또한 보상 설계(reward shaping, 예: 자원 수집이나 적 선박 파괴에 중간 보상을 부여)를 적용해 희소 보상 환경에서도 학습이 잘 이루어지도록 유도했습니다. 훈련 내내 에피소드당 총 보상, 승률, 정책 손실 등의 지표를 모니터링하며, 발산을 방지하기 위해 파라미터를 조정했습니다.

* **실험 및 반복**: 이 프로젝트는 실험적인 성격이 강해 반복적인 튜닝이 필수적이었습니다. 다양한 프롬프트 포맷, 모델 하이퍼파라미터, 훈련 방식을 실험했고(예: 완전 온라인 학습 vs. 과거 게임 상태의 리플레이 버퍼 등), LLM의 장황함과 확률적 성향을 다루기 위한 전략(예: 행동 어휘 제한, 짧은 출력 사용 등)도 시도해봤습니다. 각 반복마다 얻은 인사이트가 다음 단계의 개선으로 이어졌으며, 예를 들어 초기 테스트에서 일반 LLM이 잘못된 행동이나 비효율적인 행동을 자주 내놓는다는 점을 발견해, 프롬프트 지침을 다듬고 LLM 출력에 간단한 검증 규칙을 추가하게 되었습니다.

---

## 결과 및 주요 관찰
**성능**: 이 방식이 새로웠던 만큼, 최종 에이전트가 대회 상위권에 오르진 못했지만, 학습의 진전과 여러 가치 있는 인사이트를 제공했습니다. 훈련이 진행될수록 LLM 에이전트는 점차 합리적인 행동을 보이기 시작했는데, 예를 들어 자원 수집 효율을 높이고, 명백히 나쁜 행동(예: 불필요하게 선박을 충돌시키는 것 등)을 피하는 모습을 관찰할 수 있었습니다.

셀프플레이 방식 덕분에 게임당 평균 보상이 점진적으로 증가했으며, 이는 에이전트가 경험을 통해 학습하고 있음을 보여줍니다. 또한 LLM 에이전트가 이전 버전과의 대결에서 승률이 점차 상승하는 것을 관찰할 수 있었고, 이를 통해 정책이 실제로 진화하고 있음을 확인할 수 있었습니다.
하지만 에이전트의 실력은 특화된 규칙 기반 에이전트보다는 낮은 수준에서 머물렀는데, 이는 소형 LLM이 제한된 학습으로 게임의 복잡성을 완전히 마스터하기 어렵다는 점을 반영합니다.

### 주요 발견 사항:
이번 실험을 통해 몇 가지 중요한 점이 드러났습니다:

* **LLM 에이전트의 실현 가능성**: 적절한 추론 능력을 갖춘 LLM은 강화학습 파인튜닝을 통해 게임 환경에서 의사결정을 내리도록 적응시킬 수 있습니다. 이는 하나의 범용 모델을 다양한 작업에 활용할 수 있는 가능성을 보여주기 때문에 매우 고무적입니다. 이번에 사용한 DeepSeek-R1-Distill-Qwen-1.5B 모델은 추론 능력에 중점을 둔 상위 모델에서 디스틸된 것으로, 논리적 추론이나 계획 같은 유용한 사전 지식을 게임 환경에도 가져왔습니다. 이런 사전 지식 덕분에 에이전트가 어느 정도 다단계 결과를 추론하는 데 도움이 되었습니다.

* **프롬프트 설계의 중요성**: LLM에게 정보를 어떻게 제시하느냐가 성능에 큰 영향을 미쳤습니다. 예를 들어, "Ship A at (3,5) low energy, enemy nearby"처럼 중요한 특징을 명확하게 부각하는 프롬프트가, 단순하거나 지나치게 장황한 설명보다 훨씬 더 나은 행동 결정을 이끌어냈습니다. 이는 LLM 기반 에이전트에서는 프롬프트 엔지니어링이 기존 에이전트에서 신경망 구조만큼이나 중요해진다는 점을 시사합니다.

* **LLM과 RL의 어려움**: 학습 불안정성과 샘플 비효율성이 큰 장애물이었습니다. 언어 모델은 출력의 분산이 크기 때문에 보상 신호도 잡음이 많았고, 이를 완화하기 위해 셀프플레이 게임을 대량으로 돌리고 보상을 스무딩 처리했지만, 완전한 안정성 확보에는 어려움이 있었습니다. 또한 긴 게임에서 승패를 특정 행동에 연결짓는(credit assignment) 것도 쉽지 않았습니다. 리워드 셰이핑 같은 기법이 필요했지만, 잘못하면 오히려 에이전트를 혼란스럽게 만들 수 있어 주의가 필요했습니다.

* **확장성과 자원 문제**: 1.5B 파라미터 규모의 LLM조차 게임 루프 안에서 돌리는 것은 연산 비용이 상당히 높았습니다. 추론 속도를 극대화하기 위해 4비트 양자화와 PyTorch 최적화를 적용했지만, 경량 모델에 비해 훈련 속도가 현저히 느렸습니다. 이 실험을 통해, 대형 모델을 강화학습에 적용할 때는 효율적인 학습 프레임워크가 반드시 필요하다는 점을 다시 한 번 확인할 수 있었습니다.

**결론**:
이번 프로젝트는 강화학습을 통해 언어 모델을 복잡한 게임 환경에서 동작하도록 파인튜닝할 수 있다는 개념을 입증하는 ‘개념 증명(proof-of-concept)’ 역할을 했습니다. 비록 LLM 에이전트가 특화된 솔루션들보다 뛰어난 성능을 내진 못했지만, LLM이 시행착오를 통해 게임 전략을 학습할 수 있음을 확인했습니다. 이 작업은 매우 실험적인 단계이지만, 향후 연구를 위한 중요한 디딤돌이 됩니다. 더 큰 모델이나, 플래닝 알고리즘과의 결합, 혹은 인간 피드백을 활용한 보상 조정 등 더 발전된 기법을 도입한다면, LLM 기반 에이전트도 복잡한 환경에서 훨씬 경쟁력 있게 발전할 수 있을 것입니다.

요약하자면, 이 레포지토리는 최신 LLM과 강화학습을 게임 환경에서 혁신적으로 통합한 사례를 보여줍니다. (Hugging Face 생태계, PPO, JAX 환경 등) 기술 스택과 새로운 에이전트 훈련 방법론이 강조되어 있으며, 이번 경험을 통해 LLM을 기존의 언어 작업을 넘어 어디까지 확장할 수 있는지에 대한 인사이트도 얻을 수 있었습니다. 특히 LLM의 추론 능력이라는 강점과, 실제로 역동적이고 구현된 의사결정 과제에 적용할 때 마주치는 현실적 한계들을 모두 조명했습니다.

---

## 프로젝트 실행 방법

분석 및 모델 훈련 재현 방법:

1.  **Repository 복제:**
    ```bash
    git clone [https://github.com/madmax0404/kaggle-lux-deepseek.git](https://github.com/madmax0404/kaggle-lux-deepseek.git)
    cd kaggle-lux-deepseek
    ```
2.  **데이터셋 다운로드:**
    * 캐글에서 대회에 참가하세요. [NeurIPS 2024 - Lux AI Season 3](https://www.kaggle.com/competitions/lux-ai-season-3)
    * 데이터를 다운받은 후 알맞은 디렉토리에 저장하세요.
3.  **가상 환경을 생성하고 필요한 라이브러리들을 설치해주세요:**
    ```bash
    conda create -n kaggle_lux_deepseek python=3.12 # or venv
    conda activate kaggle_lux_deepseek
    pip install -r requirements.txt
    ```
4.  **Jupyter Notebook을 실행해주세요:**
    ```bash
    jupyter notebook Notebooks
    ```
    데이터 처리, 모델 훈련 및 평가를 실행하려면 노트북의 단계를 따르세요.

---

## 프로젝트 구조
    kaggle-lux-deepseek/
    ├── Notebooks/
    │   ├── Agent_Development/                        # 에이전트 개발 및 실험
    │   │   ├── Modified_PPO_Trainer/                 # HuggingFace PPO_Trainer 파일 수정하여 사용
    │   │   └── DeepSeek-R1-Distill-Qwen-1.5B*.ipynb  # 모델별 실험 노트북
    │   └── EDA/                                      # 탐색적 데이터 분석
    └── images/

---

## Acknowledgements

데이터셋과 대회 플랫폼을 제공한 Lux AI Challenge와와 Kaggle에 감사드립니다.

본 프로젝트는 다음 오픈소스의 도움을 받았습니다: Python, PyTorch, DeepSeek, Hugging Face, TensorBoard, pandas, numpy, matplotlib, seaborn, Jupyter, SciPy.

모든 데이터 이용은 대회 규정과 라이선스를 준수합니다.

---

## License

Code © 2025 Jongyun Han (Max). Released under the MIT License.
See the LICENSE file for details.

Note: Datasets are NOT redistributed in this repository.
Please download them from the official Kaggle competition page
and comply with the competition rules/EULA.
