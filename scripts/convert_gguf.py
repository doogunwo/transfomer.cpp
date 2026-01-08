#!/usr/bin/env python3
# convert_gguf.py
#
# Convert a PyTorch .pt checkpoint (state_dict) into a minimal GGUF v3 file.
# - Writes each tensor in state_dict as a GGUF tensor with the same name.
# - Supports output dtype: f16 or f32 (no quantization).
#
# References:
# - GGUF v3 spec: header.version must be 3, offsets are relative to tensor_data, alignment via general.alignment.
# - ggml_type: F32=0, F16=1 (others omitted here).

from __future__ import annotations

import argparse
import os
import struct
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import torch


# ----------------------------
# GGUF constants (subset)
# ----------------------------
GGUF_MAGIC = 0x47475546  # 'GGUF' little-endian
GGUF_VERSION = 3

# gguf metadata value types (same numeric IDs as ggml/gguf)
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_STRING = 8

# ggml tensor types (subset)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


def align_up(x: int, a: int) -> int:
    return x + ((a - (x % a)) % a)


def pack_u32(x: int) -> bytes:
    return struct.pack("<I", x)


def pack_u64(x: int) -> bytes:
    return struct.pack("<Q", x)


def pack_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return pack_u64(len(b)) + b


@dataclass
class KV:
    key: str
    vtype: int
    value: Any

    def to_bytes(self) -> bytes:
        out = bytearray()
        out += pack_str(self.key)
        out += pack_u32(self.vtype)

        if self.vtype == GGUF_TYPE_UINT32:
            out += pack_u32(int(self.value))
        elif self.vtype == GGUF_TYPE_UINT64:
            out += pack_u64(int(self.value))
        elif self.vtype == GGUF_TYPE_STRING:
            out += pack_str(str(self.value))
        else:
            raise ValueError(f"Unsupported KV type: {self.vtype}")
        return bytes(out)


@dataclass
class TensorInfo:
    name: str
    shape: Tuple[int, ...]
    ggml_type: int
    offset: int  # relative to data section start
    nbytes: int

    def header_bytes(self) -> bytes:
        # gguf_tensor_info_t:
        # name: gguf string
        # n_dimensions: u32
        # dimensions: u64[n_dimensions]
        # type: u32 (ggml_type)
        # offset: u64
        out = bytearray()
        out += pack_str(self.name)
        out += pack_u32(len(self.shape))
        for d in self.shape:
            out += pack_u64(int(d))
        out += pack_u32(int(self.ggml_type))
        out += pack_u64(int(self.offset))
        return bytes(out)


def extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """
    Accept common checkpoint layouts:
      - state_dict itself: {name: Tensor}
      - {"state_dict": ...}
      - {"model_state_dict": ...}
      - {"model": ...} (sometimes nested)
    """
    if isinstance(obj, dict):
        # direct state_dict?
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj  # type: ignore[return-value]

        for k in ("state_dict", "model_state_dict", "model"):
            if k in obj:
                cand = obj[k]
                if isinstance(cand, dict) and cand and all(isinstance(v, torch.Tensor) for v in cand.values()):
                    return cand  # type: ignore[return-value]
                # sometimes {"model": {"state_dict": ...}}
                if isinstance(cand, dict):
                    try:
                        return extract_state_dict(cand)
                    except Exception:
                        pass

    raise ValueError(
        "Could not find a state_dict-like mapping in the .pt file. "
        "Expected a dict of {str: Tensor} or a dict containing 'state_dict'/'model_state_dict'/'model'."
    )


def tensor_to_bytes(t: torch.Tensor, out_dtype: str) -> Tuple[bytes, Tuple[int, ...], int]:
    """
    Returns (raw_bytes, shape, ggml_type).
    """
    t = t.detach().cpu()

    # Make it contiguous in row-major
    if not t.is_contiguous():
        t = t.contiguous()

    if out_dtype == "f16":
        ggml_type = GGML_TYPE_F16
        if not t.is_floating_point():
            t = t.to(torch.float16)
        else:
            t = t.to(torch.float16)
        # ensure little-endian
        b = t.numpy().astype("<f2", copy=False).tobytes(order="C")
    elif out_dtype == "f32":
        ggml_type = GGML_TYPE_F32
        if not t.is_floating_point():
            t = t.to(torch.float32)
        else:
            t = t.to(torch.float32)
        b = t.numpy().astype("<f4", copy=False).tobytes(order="C")
    else:
        raise ValueError("--dtype must be one of: f16, f32")

    shape = tuple(int(x) for x in t.shape)
    return b, shape, ggml_type


def build_kvs(alignment: int, arch: str, name: str) -> List[KV]:
    # Minimal but spec-friendly keys
    return [
        KV("general.architecture", GGUF_TYPE_STRING, arch),
        KV("general.name", GGUF_TYPE_STRING, name),
        KV("general.alignment", GGUF_TYPE_UINT32, alignment),
    ]


def write_gguf(
    out_path: str,
    state_dict: Dict[str, torch.Tensor],
    dtype: str,
    alignment: int,
    arch: str,
    model_name: str,
) -> None:
    # 1) Convert tensors to raw bytes first (so we can compute offsets)
    tensors_raw: List[Tuple[str, bytes, Tuple[int, ...], int]] = []
    for name, t in state_dict.items():
        raw, shape, ggml_type = tensor_to_bytes(t, dtype)
        tensors_raw.append((name, raw, shape, ggml_type))

    # 2) Prepare KV + tensor infos with offsets
    kvs = build_kvs(alignment=alignment, arch=arch, name=model_name)

    # Compute per-tensor offsets (relative to data section start)
    cur_off = 0
    infos: List[TensorInfo] = []
    for name, raw, shape, ggml_type in tensors_raw:
        cur_off = align_up(cur_off, alignment)
        infos.append(
            TensorInfo(
                name=name,
                shape=shape,
                ggml_type=ggml_type,
                offset=cur_off,
                nbytes=len(raw),
            )
        )
        cur_off += len(raw)

    # 3) Serialize header + kvs + tensor infos
    header = bytearray()
    header += pack_u32(GGUF_MAGIC)
    header += pack_u32(GGUF_VERSION)
    header += pack_u64(len(infos))  # tensor_count
    header += pack_u64(len(kvs))    # metadata_kv_count

    meta_blob = bytearray()
    for kv in kvs:
        meta_blob += kv.to_bytes()

    ti_blob = bytearray()
    for ti in infos:
        ti_blob += ti.header_bytes()

    # 4) Pad to alignment to start data section
    pre_data = header + meta_blob + ti_blob
    pre_data_len = len(pre_data)
    data_start_len = align_up(pre_data_len, alignment)
    pad_len = data_start_len - pre_data_len
    pre_data += b"\x00" * pad_len

    # 5) Write file: pre_data + tensor data (each tensor padded to alignment)
    with open(out_path, "wb") as f:
        f.write(pre_data)

        # data section cursor (relative)
        rel = 0
        for (name, raw, _shape, _ggml_type), ti in zip(tensors_raw, infos):
            # pad up to ti.offset
            if rel < ti.offset:
                f.write(b"\x00" * (ti.offset - rel))
                rel = ti.offset

            # write tensor bytes
            f.write(raw)
            rel += len(raw)

            # no need to pad here; next tensor loop pads to its offset

    print(f"[OK] wrote: {out_path}")
    print(f"     tensors: {len(infos)} | dtype: {dtype} | alignment: {alignment}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="Input .pt (state_dict or checkpoint dict)")
    ap.add_argument("--output", "-o", required=True, help="Output .gguf")
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f16", help="Tensor storage dtype")
    ap.add_argument("--alignment", type=int, default=32, help="general.alignment and tensor data alignment")
    ap.add_argument("--arch", default="transfomer", help="general.architecture (lowercase recommended)")
    ap.add_argument("--name", default=None, help="general.name (default: output basename)")

    args = ap.parse_args()

    in_path = args.input
    out_path = args.output
    alignment = int(args.alignment)

    ckpt = torch.load(in_path, map_location="cpu")
    sd = extract_state_dict(ckpt)

    model_name = args.name
    if model_name is None:
        model_name = os.path.splitext(os.path.basename(out_path))[0]

    write_gguf(
        out_path=out_path,
        state_dict=sd,
        dtype=args.dtype,
        alignment=alignment,
        arch=args.arch,
        model_name=model_name,
    )


if __name__ == "__main__":
    main()
