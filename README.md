
# Bit-Serial Dynamic Inference Engine for CPU

CPU 환경에서 Bit-slicing(비트 평면) 및 Bit-serial computation을 활용하여 선형 레이어 연산을 가속하는 실험적 추론 엔진. \
부동소수점 곱셈(GEMM)을 Bitwise 연산(AND, POPCOUNT, SHIFT)으로 대체 \ 

1. 핵심 아이디어 \
INT4 Quantization: FP32 가중치를 정밀도 손실을 최소화하여 4비트 정수형으로 압축 \
Bit-Plane Slicing: 각 비트 자릿수($2^3, 2^2, 2^1, 2^0$)별로 별도의 4개 비트 평면(Bit-planes), 즉 "Switch Boards"를 생성 \
CPU가 AVX2 SIMD 명령어로 한 번에 256개의 가중치 비트를 읽어와 Bitwise AND 및 Popcount 연산을 수행할 수 있게 합니다.\

2. Layered Switch Boards (Bit-Plane Storage) \
가중치를 단순한 숫자가 아닌, 4개의 비트 평면(Bit-planes)으로 물리적으로 분리하여 저장 \

Board 3 (MSB): 가장 큰 값을 결정하는 상위 비트 평면 \
Board 0 (LSB): 정밀도를 결정하는 하위 비트 평면 \

3. Dynamic Speculation (Early Exit) \
상위 비트(MSB)부터 우선 연산하여 결과값이 임계치에 도달하면 하위 비트 연산을 생략합니다. 이를 통해 정밀도(Accuracy)와 지연시간(Latency)을 실시간으로 조절할 수 있습니다. \