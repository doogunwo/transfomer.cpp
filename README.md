transfomer.cpp는 LLM 추론의 핵심인 Transformer 아키텍처를 C++로 바닥부터 구현해보는 프로젝트입니다.
llama.cpp를 모방하여, CPU 환경에서 최적의 성능을 내는 것을 목표로 합니다.

### 🚀 프로젝트 현황 (Milestones)

[x] Model Modeling: PyTorch 기반의 Transformer(Encoder-Decoder) 모델 설계 완료.
[x] MPS Acceleration: Apple Silicon(M1/M2/M3) GPU를 활용한 Mac 전용 학습 파이프라인 train_mac.py 구현
[x] Project Architecture: 엔진(engine), 헤더(include), 스크립트(scripts) 기반의 C++ 프로젝트 구조 정리
[ ] Weight Export: 학습된 .pt 가중치를 전용 바이너리 포맷(TFCP)으로 추출하는 모듈 구현 중.
[ ] Inference Engine: mmap 기반 로더 및 C++ 추론 커널 구현 예정.

### 📂 디렉토리 구조 (Project Structure)
.
├── engine/           # 고성능 추론 엔진 (연산 커널 및 실행 로직)
│   ├── main.cpp      # CLI 엔트리 포인트 및 추론 제어
│   ├── model.cpp     # C++ 기반 Transformer 추론 구현
│   └── loader.cpp    # mmap 기반 가중치 로딩 엔진
├── include/          # MTP 기반 텐서 추상화 및 헤더 파일
│   ├── tensor.hpp    # 템플릿 기반 텐서 클래스
│   └── kernels.hpp   # NEON/SIMD 가속 연산 커널
├── models/           # 추출된 바이너리 가중치(.bin) 저장소 (Git 제외)
├── scripts/          # 가중치 변환(Export) 및 배포용 스크립트
├── tests/            # 연산 정확도 검증을 위한 유닛 테스트
├── train/            # PyTorch 기반 학습 환경 (Mac/DML 지원)
│   ├── Layer/        # 트랜스포머 핵심 레이어(Attention, FFN 등) 구현부
│   ├── train_mac.py  # Apple Silicon 가속 학습 스크립트
│   ├── transformer.py # 모델 전체 아키텍처 조립
│   └── DataLoader.py # 데이터 전처리 및 로딩 파이프라인
├── Makefile          # 빌드 시스템 (Clang/OpenMP/SIMD 최적화 설정)
└── README.md

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