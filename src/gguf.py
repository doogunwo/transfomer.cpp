# src/gguf_reader.py
from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional


# GGUF value types (as used by gguf/ggml ecosystem)
GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12


@dataclass
class GGUFHeader:
    magic: int
    version: int
    n_tensors: int
    n_kv: int


@dataclass
class TensorInfo:
    name: str
    n_dims: int
    dims: Tuple[int, ...]
    ggml_type: int
    offset: int


def _read_exact(f: io.BufferedReader, n: int) -> bytes:
    b = f.read(n)
    if b is None or len(b) != n:
        raise EOFError(f"Unexpected EOF: need {n} bytes, got {0 if b is None else len(b)}")
    return b


def _u32(f) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _u64(f) -> int:
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _i8(f) -> int:
    return struct.unpack("<b", _read_exact(f, 1))[0]


def _u8(f) -> int:
    return struct.unpack("<B", _read_exact(f, 1))[0]


def _i16(f) -> int:
    return struct.unpack("<h", _read_exact(f, 2))[0]


def _u16(f) -> int:
    return struct.unpack("<H", _read_exact(f, 2))[0]


def _i32(f) -> int:
    return struct.unpack("<i", _read_exact(f, 4))[0]


def _f32(f) -> float:
    return struct.unpack("<f", _read_exact(f, 4))[0]


def _i64(f) -> int:
    return struct.unpack("<q", _read_exact(f, 8))[0]


def _f64(f) -> float:
    return struct.unpack("<d", _read_exact(f, 8))[0]


def _read_str(f) -> str:
    n = _u64(f)
    raw = _read_exact(f, n)
    return raw.decode("utf-8", errors="replace")


def _read_kv_value(f, vtype: int) -> Any:
    if vtype == GGUF_TYPE_UINT8:
        return _u8(f)
    if vtype == GGUF_TYPE_INT8:
        return _i8(f)
    if vtype == GGUF_TYPE_UINT16:
        return _u16(f)
    if vtype == GGUF_TYPE_INT16:
        return _i16(f)
    if vtype == GGUF_TYPE_UINT32:
        return _u32(f)
    if vtype == GGUF_TYPE_INT32:
        return _i32(f)
    if vtype == GGUF_TYPE_FLOAT32:
        return _f32(f)
    if vtype == GGUF_TYPE_BOOL:
        return bool(_u8(f))
    if vtype == GGUF_TYPE_STRING:
        return _read_str(f)
    if vtype == GGUF_TYPE_UINT64:
        return _u64(f)
    if vtype == GGUF_TYPE_INT64:
        return _i64(f)
    if vtype == GGUF_TYPE_FLOAT64:
        return _f64(f)
    if vtype == GGUF_TYPE_ARRAY:
        elem_type = _u32(f)
        n = _u64(f)
        arr = []
        # parse elements
        for _ in range(n):
            arr.append(_read_kv_value(f, elem_type))
        return {"elem_type": elem_type, "values": arr}
    raise ValueError(f"Unknown GGUF value type: {vtype}")


def read_gguf(path: str) -> Tuple[GGUFHeader, Dict[str, Any], List[TensorInfo]]:
    with open(path, "rb") as f:
        # header
        magic = _u32(f)
        version = _u32(f)
        n_tensors = _u64(f)
        n_kv = _u64(f)
        hdr = GGUFHeader(magic, version, n_tensors, n_kv)

        # kvs
        kv: Dict[str, Any] = {}
        for _ in range(hdr.n_kv):
            key = _read_str(f)
            vtype = _u32(f)
            val = _read_kv_value(f, vtype)
            kv[key] = val

        # tensor infos
        tensors: List[TensorInfo] = []
        for _ in range(hdr.n_tensors):
            name = _read_str(f)
            n_dims = _u32(f)
            dims = tuple(_u64(f) for _ in range(n_dims))
            ggml_type = _u32(f)
            offset = _u64(f)
            tensors.append(TensorInfo(name, n_dims, dims, ggml_type, offset))

        return hdr, kv, tensors


def pick_int(kv: Dict[str, Any], keys: List[str]) -> Optional[int]:
    for k in keys:
        if k in kv and isinstance(kv[k], int):
            return int(kv[k])
    return None


def pick_str(kv: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in kv and isinstance(kv[k], str):
            return kv[k]
    return None


def summarize_model(kv: Dict[str, Any]) -> Dict[str, Any]:
    # Common llama.cpp-style keys (decoder-only families often use these names even if not "llama")
    arch = pick_str(kv, ["general.architecture", "architecture", "model.architecture"])
    name = pick_str(kv, ["general.name", "model.name", "general.basename"])
    ctx  = pick_int(kv, ["llama.context_length", "context_length", "ctx_len", "n_ctx"])
    n_layer = pick_int(kv, ["llama.block_count", "block_count", "n_layer", "layers"])
    n_head  = pick_int(kv, ["llama.attention.head_count", "head_count", "n_head"])
    n_kv_head = pick_int(kv, ["llama.attention.head_count_kv", "head_count_kv", "n_kv_head"])
    n_embd = pick_int(kv, ["llama.embedding_length", "embedding_length", "n_embd", "d_model"])
    rope_theta = kv.get("llama.rope.freq_base", None)
    rms_eps = kv.get("llama.attention.layer_norm_rms_epsilon", None)

    # Your custom model may store different keys; we still output what we can.
    out = {
        "name": name,
        "architecture": arch,
        "context_length": ctx,
        "n_layers": n_layer,
        "n_heads": n_head,
        "n_kv_heads": n_kv_head,
        "embedding": n_embd,
        "rope_theta": rope_theta,
        "rms_eps": rms_eps,
    }
    return out
