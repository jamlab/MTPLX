from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


KV_QUANT_MODES = ("off", "q8", "q4")


@dataclass(frozen=True)
class ResolvedAPIKey:
    value: str | None
    source: str

    @property
    def required(self) -> bool:
        return bool(self.value)


def normalize_paged_kv_quantization(value: object | None, *, allow_none: bool = False) -> str | None:
    if value is None:
        if allow_none:
            return None
        return "off"
    raw = str(value).strip().lower().replace("-", "_")
    if raw in ("", "none", "false", "0", "disabled", "disable"):
        return "off"
    if raw in ("off", "q8", "q4"):
        return raw
    if raw in ("8", "8bit", "int8", "uint8", "q8_0"):
        return "q8"
    if raw in ("4", "4bit", "int4", "uint4", "q4_0"):
        return "q4"
    choices = ", ".join(KV_QUANT_MODES)
    raise ValueError(f"unsupported paged KV quantization mode {value!r}; expected one of: {choices}")


def paged_kv_quantization_env(mode: object | None) -> dict[str, str]:
    canonical = normalize_paged_kv_quantization(mode)
    return {
        "MTPLX_VLLM_METAL_PAGED_KV_QUANT": canonical,
        "MTPLX_PAGED_KV_QUANT": canonical,
    }


def apply_paged_kv_quantization_env(mode: object | None, env: dict[str, str] | None = None) -> str:
    canonical = normalize_paged_kv_quantization(mode)
    target = os.environ if env is None else env
    target.update(paged_kv_quantization_env(canonical))
    return canonical


def resolve_api_key(
    *,
    explicit_api_key: str | None = None,
    api_key_file: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> ResolvedAPIKey:
    explicit = _clean_secret(explicit_api_key)
    if explicit:
        return ResolvedAPIKey(explicit, "flag")

    if api_key_file:
        path = Path(api_key_file).expanduser()
        secret = _clean_secret(path.read_text(encoding="utf-8"))
        if not secret:
            raise ValueError(f"API key file is empty: {path}")
        return ResolvedAPIKey(secret, "file")

    source_env = os.environ if env is None else env
    api_key = _clean_secret(source_env.get("MTPLX_API_KEY"))
    if api_key:
        return ResolvedAPIKey(api_key, "env:MTPLX_API_KEY")

    legacy = _clean_secret(source_env.get("MTPLX_AUTH"))
    if legacy:
        return ResolvedAPIKey(legacy, "env:MTPLX_AUTH")

    return ResolvedAPIKey(None, "none")


def _clean_secret(value: object | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None
