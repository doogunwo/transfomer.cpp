import struct
import os

GTF_PATH = "../models/model_hybrid.gtf"

def count_gtf_parameters(path):
    total_params = 0
    layer_count = 0
    
    print(f"📂 Reading GTF File: {path}")
    
    with open(path, "rb") as f:
        # Header 파싱
        header = f.read(16)
        if len(header) < 16:
            print("Error: Invalid Header")
            return
            
        magic, version, count, _ = struct.unpack('<4sIII', header)
        print(f"   Magic: {magic.decode()}, Count: {count} entries")
        
        # 각 엔트리 순회
        for _ in range(count):
            entry = f.read(128)
            unpacked = struct.unpack('<64sIIII IQQ 28x', entry)
            
            # dims: [N, C, H, W]
            dims = unpacked[1:5]
            
            # 레이어의 파라미터 수 = 차원들의 곱
            num_elements = dims[0] * dims[1] * dims[2] * dims[3]
            total_params += num_elements
            layer_count += 1

    # 결과 출력
    print(f"---------------------------------------------")
    print(f"📊 GTF Total Parameters Check")
    print(f"---------------------------------------------")
    
    if total_params >= 1e9:
        print(f"Total Size: {total_params / 1e9:.2f}B (Billion)")
    elif total_params >= 1e6:
        print(f"Total Size: {total_params / 1e6:.2f}M (Million)")
    else:
        print(f"Total Size: {total_params:,}")
        
    print(f"Total Layers Scanned: {layer_count}")
    print(f"---------------------------------------------")

if __name__ == "__main__":
    if os.path.exists(GTF_PATH):
        count_gtf_parameters(GTF_PATH)
    else:
        print(f"File not found: {GTF_PATH}")