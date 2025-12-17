import torch

# token_ids: List[int] or 1D torch.LongTensor
data = torch.tensor(token_ids, dtype=torch.long)   # [N]
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(batch_size: int, block_size: int):
    # 랜덤 시작 위치들
    ix = torch.randint(0, data.size(0) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])         # [B, T]
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])     # [B, T]
    return x.to(device), y.to(device)

# 학습 루프 예시
for step in range(1000):
    x, y = get_batch(batch_size=8, block_size=512)
    logits = model(x)  # [B, T, V]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


