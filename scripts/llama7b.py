import os
from huggingface_hub import snapshot_download

# ================= 설정 =================
# 1. 다운로드할 모델 ID
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"

# 2. 저장할 로컬 경로 (절대 경로 권장)
# 예: 현재 폴더 아래 'models/llama-2-7b'에 저장
LOCAL_DIR = "./models/llama-2-7b-chat-hf"

# 3. Hugging Face 토큰 (로그인 안 되어 있으면 입력 필요)
# 터미널에서 huggingface-cli login을 했다면 None으로 두어도 됨
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxx"
# ========================================

def download_model():
    print(f"[INFO] '{MODEL_ID}' 다운로드 시작...")
    print(f"[INFO] 저장 경로: {os.path.abspath(LOCAL_DIR)}")

    # snapshot_download는 리포지토리의 모든 파일을 로컬로 복사함
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        token=HF_TOKEN,
        local_dir_use_symlinks=False, # True면 캐시 바로가기 생성, False면 실제 파일 복사 (독립적 관리 원하면 False)
        ignore_patterns=["*.msgpack", "*.h5", ".git*"] # 불필요한 텐서플로우/Flax 가중치 제외
    )
    
    print("[INFO] 다운로드 완료.")

if __name__ == "__main__":
    download_model()