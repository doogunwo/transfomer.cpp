import os
from huggingface_hub import hf_hub_download

# ================= 설정 =================
# GGUF 변환 장인인 TheBloke의 리포 사용
REPO_ID = "TheBloke/Llama-2-7b-Chat-GGUF"
FILENAME = "llama-2-7b-chat.Q4_K_M.gguf" # 약 4.08GB

# 저장할 위치
LOCAL_DIR = "./models"
# ========================================

def download_gguf():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"[INFO] GGUF 다운로드 시작: {FILENAME}")
    
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False # 실제 파일 다운로드
    )
    
    print(f"[INFO] 다운로드 완료: {model_path}")

if __name__ == "__main__":
    download_gguf()