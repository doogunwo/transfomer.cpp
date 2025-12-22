import torch # type: ignore
import torch_directml # type: ignore
# .\.venv\Scripts\python.exe check_dml.py
# 1. DirectML 디바이스 가져오기
dml = torch_directml.device()

print(f"--- AMD 가속 테스트 ---")
print(f"DirectML Device: {dml}")

# 2. 텐서 연산 테스트
try:
    a = torch.tensor([1.0, 2.0]).to(dml)
    b = torch.tensor([3.0, 4.0]).to(dml)
    c = a + b
    print(f"연산 결과: {c}")
    print("- 성공! AMD 내장 GPU 가속이 작동 중입니다.")
except Exception as e:
    print(f"- 실패: {e}")