import struct
import os
import sys

# =========================================================
# 설정
# =========================================================
GTF_PATH = "../models/model_hybrid.gtf"

def read_gtf_structure(path):
    if not os.path.exists(path):
        print(f"❌ 파일을 찾을 수 없습니다: {path}")
        return

    print(f"🔍 Inspecting: {path}")
    file_size = os.path.getsize(path)
    print(f"📦 File Size: {file_size / 1024 / 1024:.2f} MB")
    print("-" * 100)

    with open(path, "rb") as f:
        # -------------------------------------------------
        # 1. Header Read (16 bytes)
        # -------------------------------------------------
        # Magic(4s) + Version(I) + TensorCount(I) + Reserved(I)
        header_data = f.read(16)
        if len(header_data) < 16:
            print("❌ Error: 헤더를 읽기에 파일이 너무 짧습니다.")
            return

        magic, version, count, reserved = struct.unpack('<4sIII', header_data)

        # 디코딩 시 에러 방지 (try-except 혹은 safe decode)
        try:
            magic_str = magic.decode('utf-8')
        except:
            magic_str = str(magic)

        print(f"🧩 Magic   : {magic_str}")
        print(f"🔢 Version : {version}")
        print(f"📚 Tensors : {count}")
        
        # [수정됨] 실제 파일의 Magic Number인 'GTFH'를 확인하도록 변경
        # 기존 코드의 b"GTF_HYB"는 7바이트라 4s 포맷과 맞지 않았습니다.
        valid_magics = [b"GTFH", b"GTF1"] 
        
        if magic not in valid_magics:
            print(f"❌ Error: 유효한 GTF 파일이 아닙니다. (Found: {magic})")
            return

        print("-" * 100)
        print(f"{'Tensor Name':<50} | {'Type':<8} | {'Shape':<20} | {'Size (KB)':<10} | {'Offset'}")
        print("=" * 100)

        # -------------------------------------------------
        # 2. Tensor Map Loop
        # -------------------------------------------------
        
        type_1_count = 0  # Switch
        type_0_count = 0  # Float

        for i in range(count):
            entry_data = f.read(128)
            if len(entry_data) < 128:
                print(f"⚠️ Warning: 텐서 맵 엔트리 읽기 실패 (Index: {i})")
                break
            
            # 구조: Name(64s) + Shape(4I) + Type(I) + Offset(Q) + Size(Q) + Padding(28x)
            unpacked = struct.unpack('<64sIIII IQQ 28x', entry_data)
            
            name_bytes = unpacked[0]
            dims = unpacked[1:5] # Shape (4개 정수)
            t_type = unpacked[5]
            offset = unpacked[6]
            size = unpacked[7]
            
            # 이름 디코딩 (Null 문자 제거)
            name = name_bytes.decode('utf-8', errors='ignore').rstrip('\x00')
            
            # Shape 포맷팅 (0이 나오기 전까지만 표시)
            valid_dims = [d for d in dims if d > 0]
            if not valid_dims: valid_dims = [0] # 스칼라 등의 경우 대비
            shape_str = str(valid_dims)
            
            # 타입 이름 변환
            if t_type == 1:
                type_str = "SWITCH" # Bit Sliced
                type_1_count += 1
            else:
                type_str = "FLOAT"  # Raw FP32
                type_0_count += 1

            # 사이즈 KB 변환
            size_kb = size / 1024
            
            print(f"{name:<50} | {type_str:<8} | {shape_str:<20} | {size_kb:>9.2f} | {offset}")

        print("=" * 100)
        print(f"✅ Summary:")
        print(f"   - Switch Board Layers (Compressed): {type_1_count}")
        print(f"   - Raw Float Layers (Preserved):     {type_0_count}")
        print(f"   - Total Layers:                     {type_1_count + type_0_count}")
        
        # -------------------------------------------------
        # 3. 데이터 영역 검증
        # -------------------------------------------------
        expected_data_start = 16 + (count * 128)
        current_pos = f.tell()
        
        if current_pos == expected_data_start:
             print(f"   - Data section starts correctly at byte {current_pos}")
        else:
             print(f"   ⚠️ Warning: Meta/Data boundary mismatch (Pos: {current_pos}, Exp: {expected_data_start})")

if __name__ == "__main__":
    read_gtf_structure(GTF_PATH)