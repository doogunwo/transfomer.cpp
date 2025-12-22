import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer

from transformer import Transformer

# -----------------------------
# 0) Device: Mac 실리콘 먼저
# -----------------------------
def get_device():
    # 1. Apple Silicon GPU 확인
    if torch.backends.mps.is_available():
        print("[Device] Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    
    # 2. NVIDIA GPU 확인
    if torch.cuda.is_available():
        print("[Device] Using CUDA")
        return torch.device("cuda")

    # 3. 기본 CPU
    print("[Device] Using CPU")
    return torch.device("cpu")

device = get_device()

# -----------------------------
# 1) Mask utils (bool mask: True = mask out)
# -----------------------------
def make_padding_mask(tokens: torch.Tensor, pad_id: int):
    # tokens: [B,S] -> [B,1,1,S]
    return (tokens == pad_id).unsqueeze(1).unsqueeze(2)

def make_causal_mask(T: int, device):
    # 처음부터 device(mps)에 직접 생성하여 데이터 전송 오버헤드 방지
    mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)

def combine_masks(pad_mask, causal_mask):
    if pad_mask.dim() == 4 and pad_mask.size(2) == 1:
        pad_mask = pad_mask.expand(-1, -1, causal_mask.size(2), -1)
    return pad_mask | causal_mask

# -----------------------------
# 2) Dataset + Tokenizer
# -----------------------------
MODEL_NAME = "Helsinki-NLP/opus-mt-de-en" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="./hf_cache")

pad_id = tokenizer.pad_token_id
eos_id = tokenizer.eos_token_id
bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else eos_id

MAX_SRC_LEN = 128
MAX_TRG_LEN = 128

raw = load_dataset("opus_books", "de-en", cache_dir="./hf_cache")

def preprocess(ex):
    src_text = ex["translation"]["de"]
    trg_text = ex["translation"]["en"]

    src = tokenizer(src_text, truncation=True, max_length=MAX_SRC_LEN, add_special_tokens=True)
    trg = tokenizer(trg_text, truncation=True, max_length=MAX_TRG_LEN - 1, add_special_tokens=True)

    trg_ids = [bos_id] + trg["input_ids"]
    if len(trg_ids) == 0 or trg_ids[-1] != eos_id:
        trg_ids.append(eos_id)

    return {"src_ids": src["input_ids"], "trg_ids": trg_ids}

split = raw["train"].train_test_split(test_size=0.05, seed=42)
train_ds = split["train"].map(preprocess, remove_columns=split["train"].column_names)
valid_ds = split["test"].map(preprocess, remove_columns=split["test"].column_names)

print("[Data] train columns:", train_ds.column_names)
print("[Data] valid columns:", valid_ds.column_names)

def collate(batch):
    src_batch = [{"input_ids": x["src_ids"]} for x in batch]
    trg_batch = [{"input_ids": x["trg_ids"]} for x in batch]

    src_pad = tokenizer.pad(src_batch, padding=True, return_tensors="pt")
    trg_pad = tokenizer.pad(trg_batch, padding=True, return_tensors="pt")
    return src_pad["input_ids"], trg_pad["input_ids"]

BATCH_SIZE = 2 # iGPU면 16은 부담일 수 있어. 먼저 8 추천
train_loader = DataLoader(
    train_ds, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate,
    pin_memory=False  # 지원안됨
)

valid_loader = DataLoader(
    valid_ds, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    collate_fn=collate, 
    pin_memory=False  # 추가
)

# -----------------------------
# 3) Model
# -----------------------------
vocab_size = len(tokenizer)

model = Transformer(
    src_vocab_size=vocab_size,
    trg_vocab_size=vocab_size,
    d_model=256,
    n_head=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    d_ff=1024,
    max_len=2048,
    dropout=0.1,
).to(device)

print("[Model] param device:", next(model.parameters()).device)

criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# -----------------------------
# 4) Train / Eval (배치/에폭 출력 포함)
# -----------------------------
def train_one_epoch(epoch_idx: int, log_every: int = 50):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for step, (src_ids, trg_ids) in enumerate(train_loader, start=1):
        src_ids = src_ids.to(device)
        trg_ids = trg_ids.to(device)

        trg_in = trg_ids[:, :-1]
        trg_y  = trg_ids[:, 1:]

        src_mask = make_padding_mask(src_ids, pad_id).to(device)
        trg_pad_mask = make_padding_mask(trg_in, pad_id).to(device)
        trg_causal = make_causal_mask(trg_in.size(1), device)
        trg_mask = combine_masks(trg_pad_mask, trg_causal)

        optimizer.zero_grad(set_to_none=True)
        logits = model(src_ids, trg_in, src_mask=src_mask, trg_mask=trg_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), trg_y.reshape(-1))
        loss.backward()
        optimizer.step()
        torch.mps.empty_cache()# 캐시 비우기

        total_loss += loss.item()

        if step % log_every == 0:
            elapsed = time.time() - t0
            avg = total_loss / step
            ips = step / max(1e-9, elapsed)
            print(f"[Train] epoch={epoch_idx} step={step}/{len(train_loader)} "
                  f"loss={loss.item():.4f} avg={avg:.4f} it/s={ips:.2f}")

    return total_loss / max(1, len(train_loader))

@torch.no_grad()
def evaluate(epoch_idx: int):
    model.eval()
    total_loss = 0.0

    for step, (src_ids, trg_ids) in enumerate(valid_loader, start=1):
        src_ids = src_ids.to(device)
        trg_ids = trg_ids.to(device)

        trg_in = trg_ids[:, :-1]
        trg_y  = trg_ids[:, 1:]

        src_mask = make_padding_mask(src_ids, pad_id).to(device)
        trg_pad_mask = make_padding_mask(trg_in, pad_id).to(device)
        trg_causal = make_causal_mask(trg_in.size(1), device)
        trg_mask = combine_masks(trg_pad_mask, trg_causal)

        logits = model(src_ids, trg_in, src_mask=src_mask, trg_mask=trg_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), trg_y.reshape(-1))
        total_loss += loss.item()

    avg = total_loss / max(1, len(valid_loader))
    print(f"[Valid] epoch={epoch_idx} avg_loss={avg:.4f}")
    return avg

# -----------------------------
# 5) Run
# -----------------------------
EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    tr_loss = train_one_epoch(epoch, log_every=50)
    va_loss = evaluate(epoch)
    print(f"[Epoch] {epoch}/{EPOCHS} train_loss={tr_loss:.4f} valid_loss={va_loss:.4f}")

torch.save(model.state_dict(), "transformer_opus_books_de_en.pt")
print("saved: transformer_opus_books_de_en.pt")
