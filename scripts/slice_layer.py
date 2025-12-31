import torch
import numpy as np
import struct
import os

# =========================================================
# 1. 경로 및 설정
# =========================================================
# 입력 모델 경로
INPUT_MODEL_PATH = "../models/pretrained_opus_de_en.pt"
# 출력 파일 경로
OUTPUT_GTF_PATH = "../models/model_hybrid.gtf"

MAGIC = b"GTF_HYB" # Magic Number (Hybrid Format)
VERSION = 1
ALIGNMENT = 32     # 메모리 정렬 (32바이트)

# =========================================================
# 2. 핵심 엔진: FP32 -> INT4 -> Bit Slicing (Switch Boards)
# =========================================================
def create_switch_boards(weights_fp32):
    """
    [입력] FP32 가중치 (1D Array)
    [출력] (Scale bytes, [Board3, Board2, Board1, Board0])
    """
    # 1. Padding (32의 배수, 블록화 및 AVX 최적화 위함)
    n = len(weights_fp32)
    target_len = ((n + 31) // 32) * 32 
    if target_len > n:
        weights_fp32 = np.pad(weights_fp32, (0, target_len - n), 'constant')

    blocks = weights_fp32.reshape(-1, 32)
    
    # 2. Scale 계산 (Max / 7.0) -> INT4 범위 매핑용
    max_vals = np.max(np.abs(blocks), axis=1)
    scales = max_vals / 7.0
    scales[scales == 0] = 1.0
    
    # 3. Quantize (Round & Offset +8)
    # -7.0 ~ +7.0 범위를 1 ~ 15 정수로 변환 (0은 -Max/Outlier 처리용으로 비워둠)
    # 여기서는 0.0을 8로 매핑하는 Offset Binary 방식 사용
    scales_reshaped = scales[:, np.newaxis]
    q_blocks = np.round(blocks / scales_reshaped) + 8
    q_data = np.clip(q_blocks, 0, 15).astype(np.uint8).flatten()
    
    # 4. ★ Bit Slicing (4장의 보드로 찢기) ★
    # 15(1111) -> B3(1), B2(1), B1(1), B0(1)
    boards = []
    # MSB(3) 부터 LSB(0) 순서로 추출 (중요한 순서대로 저장)
    for b in range(3, -1, -1): 
        bits = (q_data >> b) & 1
        # packbits: 8개의 0/1을 1바이트로 압축
        boards.append(np.packbits(bits))
        
    return scales.astype(np.float32).tobytes(), boards

# =========================================================
# 3. 메인 변환 루프
# =========================================================
def main():
    print(f"Loading {INPUT_MODEL_PATH}...")
    if not os.path.exists(INPUT_MODEL_PATH):
        print(f"입력 파일이 없습니다: {INPUT_MODEL_PATH}")
        return

    model_state = torch.load(INPUT_MODEL_PATH, map_location="cpu", weights_only=True)
    
    binary_blob = bytearray()
    tensor_info = []
    current_offset = 0 

    print(f"Exporting to {OUTPUT_GTF_PATH}...")
    print("   Applying Hybrid Quantization Rules...")

    for name, tensor in model_state.items():
        data_fp32 = tensor.detach().cpu().float().numpy().flatten()
        shape = list(tensor.shape)
        
        # -------------------------------------------------
        # 타깃팅 룰 적용 ((O) vs (X))
        # -------------------------------------------------
        # 1. 2차원 이상 행렬인가? (len(shape) >= 2)
        # 2. 이름에 'weight'가 있는가?
        # 3. 'norm', 'bias'는 제외
        is_switch_target = (len(shape) >= 2) and ('weight' in name) and \
                           ('norm' not in name) and ('bias' not in name)

        tensor_blob = bytearray()
        
        if is_switch_target:
            # [Type 1] Switch Board (INT4 Bit-Sliced)
            scales_bytes, boards = create_switch_boards(data_fp32)
            
            tensor_blob.extend(scales_bytes)
            for b in boards:
                tensor_blob.extend(b.tobytes())
            
            t_type = 1 
            
        else:
            # [Type 0] Raw Float (FP32 유지)
            tensor_blob.extend(data_fp32.tobytes())
            t_type = 0 

        # 메모리 정렬 (Padding)
        pad_len = (ALIGNMENT - (len(tensor_blob) % ALIGNMENT)) % ALIGNMENT
        tensor_blob.extend(b'\x00' * pad_len)

        # 메타데이터 기록
        tensor_info.append({
            'name': name,
            'type': t_type,    # 0:Raw, 1:Switch
            'offset': current_offset,
            'size': len(tensor_blob),
            'shape': shape
        })
        
        binary_blob.extend(tensor_blob)
        current_offset += len(tensor_blob)

    # ---------------------------------------------------------
    # 파일 쓰기
    # ---------------------------------------------------------
    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(OUTPUT_GTF_PATH), exist_ok=True)

    with open(OUTPUT_GTF_PATH, "wb") as f:
        # 1. Header Write
        f.write(struct.pack('<4sIII', MAGIC, VERSION, len(tensor_info), 0))
        
        # 2. Tensor Map Write
        for info in tensor_info:
            # Name (64s)
            name_bytes = info['name'].encode('utf-8')[:63]
            f.write(struct.pack('<64s', name_bytes))
            
            # Shape (4 dims -> 16 bytes)
            dims = info['shape'] + [1]*(4-len(info['shape']))
            f.write(struct.pack('<IIII', *dims[:4]))
            
            # Info (Type, Offset, Size -> 20 bytes)
            f.write(struct.pack('<IQQ', info['type'], info['offset'], info['size']))
            
            # Entry Padding (Total 128 bytes)
            f.write(b'\x00' * 28)

        # 3. Data Blob Write
        f.write(binary_blob)

    # ---------------------------------------------------------
    # 결과 요약
    # ---------------------------------------------------------
    file_size_mb = os.path.getsize(OUTPUT_GTF_PATH) / 1024 / 1024
    n_switch = sum(1 for t in tensor_info if t['type'] == 1)
    n_raw = sum(1 for t in tensor_info if t['type'] == 0)

    print("\n[Success] GTF Hybrid Model Created!")
    print(f"Path: {OUTPUT_GTF_PATH}")
    print(f"Size: {file_size_mb:.2f} MB")
    print("-" * 40)
    print(f"Switch Boards (Compressed): {n_switch} tensors")
    print(f"Raw Floats (Preserved):    {n_raw} tensors")
    print("-" * 40)

if __name__ == "__main__":
    main()