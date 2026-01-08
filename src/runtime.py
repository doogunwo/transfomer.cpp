# src/runtime.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from gguf import read_gguf, summarize_model


@dataclass
class RuntimeConfig:
    # 추후 확장 포인트(지금은 크게 의미 없음)
    verbose: bool = False

class Runtime:
    def __init__(self, cfg: Optional[RuntimeConfig] = None):
        self.cfg = cfg or RuntimeConfig()

        self.model_path: Optional[Path] = None
        self.header = None
        self.kv: Dict[str, Any] = {}
        self.tensors = []
        self.model_info: Dict[str, Any] = {}

    def load_model(self, gguf_path: str) -> Dict[str, Any]:
        """
        GGUF 파일을 '로드'한다 = 파싱(header/kv/tensors)하고 요약 정보를 만든다.
        (실제 weight를 GPU로 올리는 건 다음 단계)
        """
        p = Path(gguf_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"GGUF model not found: {p}")

        hdr, kv, tensors = read_gguf(str(p))

        self.model_path = p
        self.header = hdr
        self.kv = kv
        self.tensors = tensors
        self.model_info = summarize_model(kv)

        if self.cfg.verbose:
            print(f"[Runtime] loaded GGUF: {p}")
            print(f"[Runtime] version={hdr.version} n_tensors={hdr.n_tensors} n_kv={hdr.n_kv}")

        return self.model_info

    def kv_cache_suggestion(self) -> Dict[str, Any]:
        """
        contiguous KV-cache를 위한 기본 파라미터 힌트.
        GGUF 메타 키가 부족하면 None이 될 수 있음.
        """
        info = self.model_info or {}
        n_layers = info.get("n_layers")
        n_heads = info.get("n_heads")
        n_kv_heads = info.get("n_kv_heads") or n_heads
        n_embd = info.get("embedding")
        ctx = info.get("context_length")

        head_dim = None
        if isinstance(n_embd, int) and isinstance(n_heads, int) and n_heads > 0:
            head_dim = n_embd // n_heads

        return {
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "max_seq_len": ctx,
            "dtype": "fp16",
            "layout": "contiguous(K,V) on GPU",
            "note": "If any field is None, infer it from tensor shapes or set manually.",
        }

    def generate(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0) -> str:
        """
        (stub) 나중에 여기서:
        - tokenizer
        - prefill/decode 루프
        - C++/CUDA runtime 호출
        을 붙일 예정.
        """
        if self.model_path is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # stub output
        return (
            f"[generate:stub]\n"
            f"model={self.model_path}\n"
            f"prompt={prompt}\n"
            f"max_new_tokens={max_new_tokens} temperature={temperature}\n"
            f"(next step: run actual decode loop)\n"
        )
