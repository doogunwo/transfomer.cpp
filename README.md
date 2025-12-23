transfomer.cpp는 LLM 추론의 핵심인 Transformer 아키텍처를 C++로 바닥부터 구현해보는 프로젝트입니다.
llama.cpp를 모방하여, CPU 환경에서 최적의 성능을 내는 것을 목표로 합니다.

### 🚀 프로젝트 현황 (Milestones)

[ ] Weight Export: 학습된 .pt 가중치를 전용 바이너리 포맷(TFCP)으로 추출하는 모듈 구현 중. \
[ ] Inference Engine: mmap 기반 로더 및 C++ 추론 커널 구현 예정. \

### 📂 디렉토리 구조 (Project Structure)

| 분류 | 경로 | 역할 및 주요 기능 |
| :--- | :--- | :--- |
| **Inference Engine** | `engine/` | 고성능 추론 엔진 소스 (main.cpp, model.cpp, loader.cpp) |
| **Headers** | `include/` | MTP 기반 텐서 추상화 및 SIMD(NEON) 가속 커널 헤더 |
| **Models** | `models/` | 추출된 TFCP 규격 바이너리 가중치 저장소 |
| **Scripts** | `scripts/` | PyTorch 가중치 추출(Export) 및 변환 유틸리티 |
| **Training** | `train/` | MPS 가속 학습 파이프라인 및 모델 레이어(Layer/) 정의 |
| **Build & Test** | `Makefile`, `tests/` | 최적화 빌드 설정 및 연산 정확도 유닛 테스트 |

### ### 🛠 가중치 바이너리 규격 (TFCP v1 Design)

| 구분 | 필드명 | 타입 | 설명 |
| :--- | :--- | :--- | :--- |
| **Header** | Magic Number | `char[4]` | `0x54464350` ("TFCP") 식별자 |
| | Version | `int32` | 포맷 버전 (현재 v1) |
| | Hparams | `int32[5]` | d_model, n_heads, n_layers, vocab_size, max_seq |
| **Tensors** | Name Length | `int32` | 텐서 이름의 길이 |
| | Name | `char[n]` | 레이어 식별 이름 (예: `dec.attn.weight`) |
| | Rank | `int32` | 차원 수 (예: 2D Tensor = 2) |
| | Shape | `int32[rank]` | 각 차원의 크기 (M, N) |
| | Data | `float32[]` | 정렬된(Aligned) 실제 가중치 값 |

### 🏃 시작하기 (Quick Start)

1. 환경 설정
cd train
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


2. 학습 실행 (Mac 기준)
python3 train_mac.py