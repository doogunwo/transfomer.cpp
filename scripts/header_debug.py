import struct
import os

GTF_PATH = "../models/model_hybrid.gtf"

def hexdump_header():
    if not os.path.exists(GTF_PATH):
        print(f"❌ 파일이 없습니다: {GTF_PATH}")
        return

    print(f"🔍 Checking Header of: {GTF_PATH}")
    
    with open(GTF_PATH, "rb") as f:
        # 딱 16바이트만 읽어옵니다.
        raw_bytes = f.read(16)
    
    print("-" * 40)
    print(f"Raw Bytes (Hex): {raw_bytes.hex(' ')}") 
    print(f"Raw Bytes (Str): {raw_bytes}")
    print("-" * 40)

    # 4바이트 매직넘버 해석
    try:
        magic_part = raw_bytes[:4]
        print(f"👉 Magic Number 부분: {magic_part}")
        
        if magic_part == b'GTFH':
            print("✅ 상태: 정상 (GTFH)")
        elif magic_part == b'GTF_':
            print("❌ 상태: 잘림 (GTF_) -> 파일이 갱신되지 않았음")
        else:
            print(f"⚠️ 상태: 알 수 없음 ({magic_part})")
            
    except Exception as e:
        print(f"Error parsing: {e}")

if __name__ == "__main__":
    hexdump_header()