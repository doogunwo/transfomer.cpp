import torch
import numpy as np
import struct
import os

# =========================================================
# 1. 경로 및 설정
# =========================================================
INPUT_MODEL_PATH = "../models/pretrained_opus_de_en.pt"
OUTPUT_GTF_PATH = "../models/model_hybrid.gtf"

MAGIC = b"GTFH" # 딱 4바이트로 맞춤
VERSION = 1
ALIGNMENT = 32     # AVX2/512 최적화 정렬 (32바이트)

# =========================================================
# 2. 핵심 알고리즘: 4-Bit Plane Slicing
# =========================================================
def create_switch_boards(weights_fp32):
    """
    [알고리즘 설명]
    1. FP32 -> INT4 (0~15) 값으로 변환합니다.
    2. 변환된 정수 배열을 4번 순회하며 각 비트 위치(3,2,1,0)의 값만 추출합니다.
    3. 추출된 비트(0,1)들을 8개씩 묶어 1바이트로 압축(Packing)합니다.
    
    [출력] 
    (Scale(FP32), [Board_MSB, Board_2, Board_1, Board_LSB])
    """
    # -----------------------------------------------------
    # Step A: 전처리 (Padding & Block Quantization)
    # -----------------------------------------------------
    # 1. Padding (32의 배수)
    n = len(weights_fp32)
    target_len = ((n + 31) // 32) * 32 
    if target_len > n:
        weights_fp32 = np.pad(weights_fp32, (0, target_len - n), 'constant')

    blocks = weights_fp32.reshape(-1, 32)
    
    # 2. Scale 계산 (Max / 7.0) -> Symmetric Quantization
    max_vals = np.max(np.abs(blocks), axis=1)
    scales = max_vals / 7.0
    scales[scales == 0] = 1.0
    
    # 3. Quantize (FP32 -> INT4 Integer)
    # 0.0을 정수 8(1000)로 매핑하는 Offset Binary 방식
    scales_reshaped = scales[:, np.newaxis]
    q_blocks = np.round(blocks / scales_reshaped) + 8
    q_data = np.clip(q_blocks, 0, 15).astype(np.uint8).flatten()
    
    # -----------------------------------------------------
    # Step B: ★ Bit Plane Conversion (핵심 알고리즘) ★
    # -----------------------------------------------------
    # 기존에 하나로 뭉쳐있던 q_data(INT4)를 4개의 독립된 배열로 찢습니다.
    boards = []
    
    # MSB(비트 3) 부터 LSB(비트 0) 순서로 추출
    for b in range(3, -1, -1): 
        # 1. Bit Extraction: 해당 비트 위치(b)의 값만 남김 (0 또는 1)
        # 예: 값 13(1101)이고 b=2라면 -> (1101 >> 2) & 1 -> 1
        bits = (q_data >> b) & 1
        
        # 2. Bit Packing: 8개의 비트를 1바이트로 압축
        # 예: [1,1,1,1,0,0,0,0] -> 0xF0 (240)
        # 이렇게 하면 용량이 원본 대비 1/8이 됩니다.
        packed_board = np.packbits(bits)
        boards.append(packed_board)
        
    # 결과: 4장의 압축된 보드와 스케일 값 반환
    return scales.astype(np.float32).tobytes(), boards

# =========================================================
# 3. 메인 프로세스
# =========================================================
def main():
    print(f"Loading Source: {INPUT_MODEL_PATH}")
    if not os.path.exists(INPUT_MODEL_PATH):
        print(f"Error: Input file not found at {INPUT_MODEL_PATH}")
        return

    model_state = torch.load(INPUT_MODEL_PATH, map_location="cpu", weights_only=True)
    
    binary_blob = bytearray()
    tensor_info = []
    current_offset = 0 

    print(f"Starting Conversion to Bit-Sliced Format...")
    print(f"Output Target: {OUTPUT_GTF_PATH}")

    for name, tensor in model_state.items():
        data_fp32 = tensor.detach().cpu().float().numpy().flatten()
        shape = list(tensor.shape)
        
        # -------------------------------------------------
        # 타깃 레이어 필터링
        # -------------------------------------------------
        # 2차원 행렬(Matrix)이면서 가중치(weight)인 것만 4-Plane 변환
        is_switch_target = (len(shape) >= 2) and ('weight' in name) and \
                           ('norm' not in name) and ('bias' not in name)

        tensor_blob = bytearray()
        
        if is_switch_target:
            # [Type 1] 4-Plane Bit Slicing 적용
            scales_bytes, boards = create_switch_boards(data_fp32)
            
            # 저장 순서: [Scale] -> [Board 3] -> [Board 2] -> [Board 1] -> [Board 0]
            tensor_blob.extend(scales_bytes)
            for b in boards:
                tensor_blob.extend(b.tobytes())
            
            t_type = 1 
            
        else:
            # [Type 0] 변환 없이 FP32 원본 저장
            tensor_blob.extend(data_fp32.tobytes())
            t_type = 0 

        # 메모리 정렬 (Padding for Alignment)
        pad_len = (ALIGNMENT - (len(tensor_blob) % ALIGNMENT)) % ALIGNMENT
        tensor_blob.extend(b'\x00' * pad_len)

        # 메타데이터 생성
        tensor_info.append({
            'name': name,
            'type': t_type,    # 1: Sliced, 0: Raw
            'offset': current_offset,
            'size': len(tensor_blob),
            'shape': shape
        })
        
        binary_blob.extend(tensor_blob)
        current_offset += len(tensor_blob)

    # ---------------------------------------------------------
    # GTF 파일 생성 (Write to Disk)
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_GTF_PATH), exist_ok=True)

    with open(OUTPUT_GTF_PATH, "wb") as f:
        # 1. Header
        f.write(struct.pack('<4sIII', MAGIC, VERSION, len(tensor_info), 0))
        
        # 2. Tensor Table
        for info in tensor_info:
            name_bytes = info['name'].encode('utf-8')[:63]
            f.write(struct.pack('<64s', name_bytes))
            
            dims = info['shape'] + [1]*(4-len(info['shape']))
            f.write(struct.pack('<IIII', *dims[:4]))
            
            f.write(struct.pack('<IQQ', info['type'], info['offset'], info['size']))
            f.write(b'\x00' * 28) # Entry Padding

        # 3. Binary Payload
        f.write(binary_blob)

    # 결과 리포트
    file_size_mb = os.path.getsize(OUTPUT_GTF_PATH) / 1024 / 1024
    n_switch = sum(1 for t in tensor_info if t['type'] == 1)
    
    print("\n Conversion Complete.")
    print(f"   - File: {OUTPUT_GTF_PATH}")
    print(f"   - Size: {file_size_mb:.2f} MB")
    print(f"   - Sliced Tensors: {n_switch} (converted to 4 planes)")

if __name__ == "__main__":
    main()