import torch
import platform
import time

def check_mac_setup():
    print("="*50)
    print("Mac Deep Learning Environment Check")
    print("="*50)

    # 1. 시스템 정보 확인
    print(f"[System] OS: {platform.system()} {platform.release()}")
    print(f"[System] Processor: {platform.processor()}")
    print(f"[System] Python version: {platform.python_version()}")
    
    # 2. PyTorch 정보 확인
    print(f"[PyTorch] Version: {torch.__version__}")
    
    # 3. MPS(Metal Performance Shaders) 가속 확인
    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()
    
    print(f"[MPS] is_available: {mps_available}")
    print(f"[MPS] is_built: {mps_built}")

    if not mps_available:
        print("\n - 결과: MPS를 사용할 수 없습니다.")
        print("   - macOS 12.3 이상인지 확인하세요.")
        print("   - Apple Silicon(M1, M2, M3) 또는 최신 AMD GPU Mac인지 확인하세요.")
        return

    # 4. 실제 GPU 연산 테스트 (Matrix Multiplication)
    print("\n[Test] Running a simple Matrix Multiplication on GPU...")
    device = torch.device("mps")
    
    try:
        # 큰 행렬 생성
        size = 4000
        x = torch.randn(size, size, device=device)
        y = torch.randn(size, size, device=device)
        
        # GPU 연산 시작 시간 측정
        start_time = time.time()
        
        # 행렬 곱셈 수행
        z = torch.matmul(x, y)
        
        # MPS는 비동기 방식이므로 결과를 동기화하여 정확한 시간 측정
        # (현재 PyTorch 버전에서는 자동 동기화되나 명시적 확인을 위해 결과값 참조)
        _ = z[0, 0].item() 
        
        end_time = time.time()
        
        print(f"   - 결과: 성공!")
        print(f"   - GPU 연산 시간 ({size}x{size}): {end_time - start_time:.4f} seconds")

    except Exception as e:
        print(f"- 연산 중 오류 발생: {e}")

if __name__ == "__main__":
    check_mac_setup()