from llama_cpp import Llama

# 1. 모델 로드
# n_gpu_layers=-1 : 가능한 모든 레이어를 GPU에 올림 (VRAM 부족하면 숫자 조절, 예: 30)
llm = Llama(
    model_path="../models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=4096,       # 컨텍스트 길이
    n_gpu_layers=-1,  # GPU 가속 활성화 (핵심)
    verbose=True      # 로딩 로그 보기
)

# 2. 추론
prompt = "System Software Engineer로서 갖춰야 할 핵심 역량 3가지를 알려줘."
formatted_prompt = f"[INST] {prompt} [/INST]"

print(f"\n[질문]: {prompt}")

output = llm(
    formatted_prompt,
    max_tokens=512,
    stop=["</s>"],
    echo=False
)

print(f"[답변]:\n{output['choices'][0]['text']}")