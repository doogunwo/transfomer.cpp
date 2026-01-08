import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer

# blocks.py에서 재사용
from Layer.blocks import (
    InputEmbedding,
    PositionalEncoding,
    MultiHeadAttention,
    PositionwiseFeedForward,
    SublayerConnection,
)

# -----------------------------
# 0) Device: Mac(MPS) -> CUDA -> CPU
# -----------------------------
def get_device():
    if torch.backends.mps.is_available():
        print("[Device] Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("[Device] Using CUDA")
        return torch.device("cuda")
    print("[Device] Using CPU")
    return torch.device("cpu")

device = get_device()

# -----------------------------
# 1) Mask utils (bool mask: True = mask out)
# -----------------------------
def make_padding_mask(tokens: torch.Tensor, pad_id: int):
    # tokens: [B, T] -> [B,1,1,T]
    return (tokens == pad_id).unsqueeze(1).unsqueeze(2)

def make_causal_mask(T: int, device):
    # [T, T] upper triangle => 미래 가리기
    mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # [1,1,T,T]

def combine_masks(pad_mask, causal_mask):
    # pad_mask: [B,1,1,T], causal_mask: [1,1,T,T] -> [B,1,T,T]
    if pad_mask.dim() == 4 and pad_mask.size(2) == 1:
        pad_mask = pad_mask.expand(-1, -1, causal_mask.size(2), -1)
    return pad_mask | causal_mask

# -----------------------------
# 2) Decoder-only Model (LLaMA/GPT 스타일)
# -----------------------------
class DecoderOnlyLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x, attn_mask):
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, attn_mask)[0])
        x = self.sublayer2(x, self.ffn)
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_head=8,
        num_layers=4,
        d_ff=1024,
        max_len=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = InputEmbedding(d_model, vocab_size)
        self.pe = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderOnlyLayer(d_model, n_head, d_ff, dropout) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attn_mask=None):
        x = self.embedding(input_ids)  # [B,T,d]
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, attn_mask)
        logits = self.lm_head(x)       # [B,T,V]
        return logits

# -----------------------------
# 3) Dataset + Tokenizer
#    (Decoder-only로 번역을 학습: "de + EOS + en" 이어쓰기)
# -----------------------------
MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="./hf_cache")

pad_id = tokenizer.pad_token_id
eos_id = tokenizer.eos_token_id
bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else eos_id

MAX_SRC_LEN = 128
MAX_TRG_LEN = 128

raw = load_dataset("opus_books", "de-en", cache_dir="./hf_cache")

# ids = [BOS] + src + [EOS] + trg + [EOS]
# loss는 trg(+마지막 EOS)에서만 계산하도록 src 구간 라벨을 ignore 처리
def preprocess(ex):
    src_text = ex["translation"]["de"]
    trg_text = ex["translation"]["en"]

    src = tokenizer(src_text, truncation=True, max_length=MAX_SRC_LEN, add_special_tokens=True)
    trg = tokenizer(trg_text, truncation=True, max_length=MAX_TRG_LEN, add_special_tokens=True)

    src_ids = src["input_ids"]
    trg_ids = trg["input_ids"]

    ids = [bos_id] + src_ids + [eos_id] + trg_ids
    if len(ids) == 0 or ids[-1] != eos_id:
        ids.append(eos_id)

    # labels에서 ignore할 prefix 길이 = (src_ids 길이) + (구분 EOS 1개)
    # labels는 shift된 ids[1:] 기준이므로, src 토큰들 + 구분 EOS까지 총 (len(src_ids)+1)개 ignore
    ignore_prefix = len(src_ids) + 1

    return {
        "ids": ids,
        "ignore_prefix": ignore_prefix,
    }

split = raw["train"].train_test_split(test_size=0.05, seed=42)
train_ds = split["train"].map(preprocess, remove_columns=split["train"].column_names)
valid_ds = split["test"].map(preprocess, remove_columns=split["test"].column_names)

print("[Data] train columns:", train_ds.column_names)
print("[Data] valid columns:", valid_ds.column_names)

IGNORE_INDEX = -100  # CrossEntropyLoss 기본 ignore_index와 동일하게 맞춤

def collate(batch):
    # 각 샘플에서 input_ids, labels 생성 (shift)
    inputs = []
    labels = []
    for b in batch:
        ids = b["ids"]
        ign = b["ignore_prefix"]

        inp = ids[:-1]
        lab = ids[1:]

        # src + 구분 EOS 구간은 loss 계산 제외
        lab = lab[:]  # copy
        for i in range(min(ign, len(lab))):
            lab[i] = IGNORE_INDEX

        inputs.append({"input_ids": inp})

        # labels는 tokenizer.pad로 못 깔끔하게 처리되니 직접 pad
        labels.append(lab)

    # input pad
    padded_inp = tokenizer.pad(inputs, padding=True, return_tensors="pt")["input_ids"]  # [B,T]

    # labels pad (길이 맞춰 IGNORE_INDEX로 패딩)
    max_len = padded_inp.size(1)
    padded_lab = torch.full((len(labels), max_len), IGNORE_INDEX, dtype=torch.long)
    for i, lab in enumerate(labels):
        L = min(len(lab), max_len)
        padded_lab[i, :L] = torch.tensor(lab[:L], dtype=torch.long)

    return padded_inp, padded_lab

BATCH_SIZE = 2

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate,
    pin_memory=False,
)

valid_loader = DataLoader(
    valid_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate,
    pin_memory=False,
)

# -----------------------------
# 4) Model / Optim / Loss
# -----------------------------
vocab_size = len(tokenizer)

model = DecoderOnlyTransformer(
    vocab_size=vocab_size,
    d_model=256,
    n_head=8,
    num_layers=4,
    d_ff=1024,
    max_len=2048,
    dropout=0.1,
).to(device)

print("[Model] param device:", next(model.parameters()).device)

criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

# CUDA면 AMP 사용, MPS/CPU면 비활성
use_amp = (device.type == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# -----------------------------
# 5) Train / Eval
# -----------------------------
def train_one_epoch(epoch_idx: int, log_every: int = 50):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for step, (input_ids, labels) in enumerate(train_loader, start=1):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        T = input_ids.size(1)
        pad_mask = make_padding_mask(input_ids, pad_id).to(device)
        causal_mask = make_causal_mask(T, device)
        attn_mask = combine_masks(pad_mask, causal_mask)  # [B,1,T,T]

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast(True):
                logits = model(input_ids, attn_mask=attn_mask)  # [B,T,V]
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, attn_mask=attn_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            optimizer.step()

        # MPS 메모리 압박 완화용(원래 네 코드 스타일 유지)
        if device.type == "mps":
            torch.mps.empty_cache()

        total_loss += float(loss.item())

        if step % log_every == 0:
            elapsed = time.time() - t0
            avg = total_loss / step
            ips = step / max(1e-9, elapsed)
            ppl = math.exp(avg) if avg < 50 else float("inf")
            print(f"[Train] epoch={epoch_idx} step={step}/{len(train_loader)} "
                  f"loss={loss.item():.4f} avg={avg:.4f} ppl={ppl:.2f} it/s={ips:.2f}")

    return total_loss / max(1, len(train_loader))

@torch.no_grad()
def evaluate(epoch_idx: int):
    model.eval()
    total_loss = 0.0

    for step, (input_ids, labels) in enumerate(valid_loader, start=1):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        T = input_ids.size(1)
        pad_mask = make_padding_mask(input_ids, pad_id).to(device)
        causal_mask = make_causal_mask(T, device)
        attn_mask = combine_masks(pad_mask, causal_mask)

        logits = model(input_ids, attn_mask=attn_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        total_loss += float(loss.item())

    avg = total_loss / max(1, len(valid_loader))
    ppl = math.exp(avg) if avg < 50 else float("inf")
    print(f"[Valid] epoch={epoch_idx} avg_loss={avg:.4f} ppl={ppl:.2f}")
    return avg

# -----------------------------
# 6) Run
# -----------------------------
EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    tr_loss = train_one_epoch(epoch, log_every=50)
    va_loss = evaluate(epoch)
    print(f"[Epoch] {epoch}/{EPOCHS} train_loss={tr_loss:.4f} valid_loss={va_loss:.4f}")

os.makedirs("../models", exist_ok=True)
save_path = "../models/decoder_only_opus_books_de_en.pt"
torch.save(model.state_dict(), save_path)
print(f"saved: {save_path}")
