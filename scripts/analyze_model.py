import struct
import os

GTF_PATH = "../models/model_hybrid.gtf"

def analyze_gtf_switch_planes(path, target_layer_name=None):
    with open(path, "rb") as f:
        # 1. Header & Map 파싱 (기존 로직)
        header = f.read(16)
        magic, version, count, _ = struct.unpack('<4sIII', header)
        
        for _ in range(count):
            entry = f.read(128)
            unpacked = struct.unpack('<64sIIII IQQ 28x', entry)
            name = unpacked[0].decode('utf-8').rstrip('\x00')
            t_type, offset, size = unpacked[5], unpacked[6], unpacked[7]
            dims = unpacked[1:5]

            # 분석할 첫 번째 SWITCH 레이어 선택
            if t_type == 1:
                if target_layer_name and name != target_layer_name:
                    continue
                
                print(f"📊 Analyzing Switch Board: {name}")
                print(f"   Shape: {list(dims)} | Total Size: {size} bytes")
                
                # 2. 비트 플레인 크기 계산
                # INT4 = 4 planes. 만약 1비트가 1개의 Weight를 대변한다면:
                num_elements = dims[0] * dims[1] * dims[2] * dims[3]
                plane_size_bytes = num_elements // 8  # 비트 단위 패킹 가정
                
                print(f"   Expected Elements: {num_elements}")
                print(f"   Calculated Plane Size: {plane_size_bytes} bytes per Board")
                
                # 3. 데이터 로드 및 플레인별 샘플링
                f.seek(offset)
                
                # Scale Factor가 맨 앞에 있다고 가정 (예: 4바이트 FP32)
                scale = struct.unpack('<f', f.read(4))[0]
                print(f"   Detected Scale Factor: {scale}")

                # 4개의 Board(3~0)를 순차적으로 읽어 패턴 확인
                for board_id in range(3, -1, -1):
                    board_data = f.read(16) # 각 보드의 앞 16바이트(128비트)만 확인
                    bits = "".join([format(b, '08b') for b in board_data])
                    print(f"   [Board {board_id} (2^{board_id})] First 128 bits: {bits[:64]}...")
                
                break # 하나만 분석 후 종료

if __name__ == "__main__":
    analyze_gtf_switch_planes(GTF_PATH)