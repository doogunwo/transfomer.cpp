import torch
from transformers import AutoModelForSeq2SeqLM

# 1. Hugging Face에서 사전 학습된 모델 다운로드
MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"
print(f"📥 Downloading {MODEL_NAME}...")

# 실제 모델 객체 로드
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# 2. 가중치(state_dict)만 추출하여 .pt로 저장
# 이 파일이 생성되면 바로 export_weights.py를 돌릴 수 있습니다.
SAVE_PATH = "pretrained_opus_de_en.pt"
torch.save(model.state_dict(), SAVE_PATH)

print(f"✅ Saved pre-trained weights to: {SAVE_PATH}")
print("이제 이 파일을 scripts/export_weights.py의 입력값으로 사용하세요!")