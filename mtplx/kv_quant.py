"""Plain paged-KV quantization helpers.

This is intentionally separate from TurboQuant.  TurboQuant depends on the
external vLLM-Metal encode/attention kernels; this module provides an in-tree
q8/q4 storage mode that can always fall back through MLX SDPA after dequant.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


MODES = {"q8", "int8", "q4", "int4"}


@dataclass(frozen=True)
class PagedKVQuantConfig:
    mode: str = "q8"

    @property
    def normalized_mode(self) -> str:
        raw = self.mode.strip().lower().replace("-", "_")
        if raw in {"int8", "q8_0"}:
            return "q8"
        if raw in {"int4", "q4_0"}:
            return "q4"
        return raw

    @property
    def bits(self) -> int:
        return 4 if self.normalized_mode == "q4" else 8


def env_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def config_from_env() -> PagedKVQuantConfig | None:
    raw = (
        os.environ.get("MTPLX_VLLM_METAL_PAGED_KV_QUANT")
        or os.environ.get("MTPLX_PAGED_KV_QUANT")
        or ""
    ).strip().lower().replace("-", "_")
    if raw in {"", "0", "false", "no", "off", "none"}:
        return None
    if raw in {"q8_0", "int8"}:
        raw = "q8"
    if raw in {"q4_0", "int4"}:
        raw = "q4"
    if raw not in {"q8", "q4"}:
        raise ValueError(
            f"Unsupported paged KV quantization mode {raw!r}; available=['off', 'q8', 'q4']"
        )
    return PagedKVQuantConfig(mode=raw)


def packed_dim(head_dim: int, bits: int) -> int:
    head_dim = int(head_dim)
    bits = int(bits)
    if bits == 8:
        return head_dim
    if bits == 4:
        if head_dim % 2:
            raise ValueError(f"q4 paged KV quantization requires even head_dim, got {head_dim}")
        return head_dim // 2
    raise ValueError(f"unsupported paged KV quantization bits={bits}")


def quantize_symmetric(x: Any, *, bits: int) -> tuple[Any, Any]:
    import mlx.core as mx

    bits = int(bits)
    qmax = 127 if bits == 8 else 7
    max_abs = mx.max(mx.abs(x.astype(mx.float32)), axis=-1, keepdims=True)
    scale = mx.maximum(max_abs / float(qmax), mx.array(1.0e-6, dtype=mx.float32))
    q = mx.round(x.astype(mx.float32) / scale)
    q = mx.clip(q, -float(qmax), float(qmax))
    if bits == 8:
        return q.astype(mx.int8), scale.astype(mx.float16)
    if bits == 4:
        unsigned = (q + 8).astype(mx.uint8)
        even = unsigned[..., 0::2]
        odd = unsigned[..., 1::2]
        packed = mx.bitwise_or(even, mx.left_shift(odd, 4)).astype(mx.uint8)
        return packed, scale.astype(mx.float16)
    raise ValueError(f"unsupported paged KV quantization bits={bits}")


def dequantize_symmetric(q: Any, scale: Any, *, bits: int, head_dim: int) -> Any:
    import mlx.core as mx

    bits = int(bits)
    if bits == 8:
        return q.astype(mx.float32) * scale.astype(mx.float32)
    if bits == 4:
        low = mx.bitwise_and(q, 0x0F)
        high = mx.bitwise_and(mx.right_shift(q, 4), 0x0F)
        stacked = mx.stack([low, high], axis=-1).reshape(*q.shape[:-1], int(head_dim))
        signed = stacked.astype(mx.int16) - 8
        return signed.astype(mx.float32) * scale.astype(mx.float32)
    raise ValueError(f"unsupported paged KV quantization bits={bits}")


def compression_ratio(*, head_dim: int, bits: int) -> float:
    head_dim = int(head_dim)
    bits = int(bits)
    # Two fp16 tensors, key + value.
    fp16_bytes = 2 * head_dim * 2
    # Two quantized tensors plus one fp16 scale for K and one for V.
    quant_bytes = 2 * packed_dim(head_dim, bits) + 2 * 2
    return float(fp16_bytes) / float(quant_bytes)
