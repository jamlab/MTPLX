"""Hardware-aware verified default model selection for product CLI paths."""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mtplx.constants import DEFAULT_RUNTIME_MODEL_DIR
from mtplx.hardware import classify_apple_silicon_generation, detect_apple_silicon
from mtplx.profiles import (
    DEFAULT_FP16_PUBLIC_MODEL_ID,
    DEFAULT_FP16_HF_MODEL_ID,
    DEFAULT_HF_MODEL_ID,
    DEFAULT_MODEL_ID,
    DEFAULT_PUBLIC_MODEL_ID,
    LEGACY_OPTIMIZED_PUBLIC_MODEL_ID,
    QUALITY_HF_MODEL_ID,
    QUALITY_PUBLIC_MODEL_ID,
)


DEFAULT_MODEL_VARIANT_ENV = "MTPLX_DEFAULT_MODEL_VARIANT"
QUALITY_MODEL_ENV = "MTPLX_OPTIMIZED_QUALITY_MODEL"
DEFAULT_MODEL_VARIANTS = frozenset({"auto", "bf16", "fp16"})
_LEGACY_APPLE_FP16_GENERATIONS = frozenset({"m1", "m2"})
_NEWER_APPLE_BF16_GENERATIONS = frozenset({"m3", "m4", "m5"})
OPTIMIZED_QUALITY_LABEL = "Qwen3.6 27B MTPLX Optimized Quality"
OPTIMIZED_QUALITY_DESCRIPTION = "Flat8 target with INT8 MTP sidecar"
_OPTIMIZED_QUALITY_LOCAL_CANDIDATES = (
    "~/Documents/MTPLX/hf-staging/Qwen3.6-27B-MTPLX-Optimized-Quality",
    "~/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Quality",
)
_VERIFIED_DEFAULT_LOCAL_NAMES = frozenset(
    {
        "Qwen3.6-27B-MTPLX-Optimized-Speed",
        "Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed",
        "Qwen3.6-27B-MTPLX-Optimized-Speed-FP16",
        "Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed-FP16",
    }
)


@dataclass(frozen=True)
class DefaultModelSelection:
    model: str
    hf_model: str
    variant: str
    precision: str
    chip_generation: str
    chip: str
    reason: str
    auto_selected: bool
    env_override: str | None = None

    @property
    def display_name(self) -> str:
        suffix = "FP16" if self.variant == "fp16" else "BF16"
        return f"Qwen3.6 27B Optimized Speed {suffix}"

    @property
    def label(self) -> str:
        return f"{self.hf_model}  ·  {self.precision}  ·  {self.reason}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "hf_model": self.hf_model,
            "variant": self.variant,
            "precision": self.precision,
            "chip_generation": self.chip_generation,
            "chip": self.chip,
            "reason": self.reason,
            "auto_selected": self.auto_selected,
            "env_override": self.env_override,
            "display_name": self.display_name,
            "label": self.label,
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_complete_local_model(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (
        (path / "mtplx_pair.json").is_file()
        and (path / "target").is_dir()
        and (path / "assistant").is_dir()
    ):
        return True
    if not (path / "config.json").is_file():
        return False
    has_weights = any(path.glob("model-*.safetensors")) or (path / "model.safetensors").is_file()
    has_mtp = (path / "mtp.safetensors").is_file()
    return has_weights and has_mtp


def optimized_quality_model_ref() -> str:
    env_ref = str(os.environ.get(QUALITY_MODEL_ENV) or "").strip()
    candidates = (env_ref, *_OPTIMIZED_QUALITY_LOCAL_CANDIDATES) if env_ref else _OPTIMIZED_QUALITY_LOCAL_CANDIDATES
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if _is_complete_local_model(path):
            return str(path)
    return QUALITY_HF_MODEL_ID


def is_optimized_quality_model_ref(model: str | Path | None) -> bool:
    if model is None:
        return False
    text = str(model).strip()
    if not text:
        return False
    if text == QUALITY_HF_MODEL_ID:
        return True
    return "qwen3.6-27b-mtplx-optimized-quality" in text.lower()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _artifact_role_model_id(role: str) -> str | None:
    normalized = role.strip().lower().replace("_", "-")
    if not normalized:
        return None
    if "quality" in normalized:
        return QUALITY_PUBLIC_MODEL_ID
    if "fp16" in normalized or "float16" in normalized:
        return DEFAULT_FP16_PUBLIC_MODEL_ID
    if "speed" in normalized or "flat4" in normalized or "maximum-speed" in normalized:
        return DEFAULT_PUBLIC_MODEL_ID
    return None


def _public_model_id_from_metadata(path: Path) -> str | None:
    runtime = _read_json(path / "mtplx_runtime.json")
    for key in ("public_model_id", "served_model_id", "model_id"):
        value = runtime.get(key)
        if isinstance(value, str) and value.strip():
            return _sanitize_public_model_id(value)
    precision = runtime.get("precision_variant")
    if isinstance(precision, str) and precision.strip().lower() in {"fp16", "float16"}:
        return DEFAULT_FP16_PUBLIC_MODEL_ID
    role = runtime.get("artifact_role")
    if isinstance(role, str):
        inferred = _artifact_role_model_id(role)
        if inferred:
            return inferred

    config = _read_json(path / "config.json")
    quantization = config.get("quantization") or config.get("quantization_config")
    if isinstance(quantization, dict):
        text = json.dumps(quantization, sort_keys=True).lower()
        if "bits\": 8" in text and "language_model" in text:
            return QUALITY_PUBLIC_MODEL_ID
        bits = quantization.get("bits")
        if bits == 4:
            return DEFAULT_PUBLIC_MODEL_ID
    return None


def _sanitize_public_model_id(value: str) -> str:
    lowered = str(value).strip().lower()
    lowered = lowered.replace("_", "-")
    lowered = re.sub(r"[^a-z0-9.-]+", "-", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered).strip("-.")
    return lowered or DEFAULT_PUBLIC_MODEL_ID


def _public_model_id_from_name(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None
    lowered = text.replace("\\", "/").lower()
    if "qwen3.6-27b-mtplx-optimized-quality" in lowered:
        return QUALITY_PUBLIC_MODEL_ID
    if "qwen3.6-27b-mtplx-optimized-speed-fp16" in lowered:
        return DEFAULT_FP16_PUBLIC_MODEL_ID
    if "qwen3.6-27b-mtplx-optimized-speed" in lowered:
        return DEFAULT_PUBLIC_MODEL_ID
    legacy_names = {
        "qwen3.6-27b-mtplx-optimized",
        "youssofal--qwen3.6-27b-mtplx-optimized",
    }
    basename = Path(text).name.lower()
    if basename in legacy_names or lowered.endswith("/qwen3.6-27b-mtplx-optimized"):
        return LEGACY_OPTIMIZED_PUBLIC_MODEL_ID
    return None


def public_model_id_for_ref(
    model: str | Path | None,
    *,
    default_model_id: str = DEFAULT_PUBLIC_MODEL_ID,
) -> str:
    """Return the served OpenAI model id for the selected artifact.

    The default public id is only used when no model was provided. Once a
    concrete repo/path exists, MTPLX should report that artifact instead of
    silently claiming the speed default.
    """

    if model is None:
        return default_model_id
    text = str(model).strip()
    if not text:
        return default_model_id
    path = Path(text).expanduser()
    if path.is_dir():
        inferred = _public_model_id_from_metadata(path)
        if inferred:
            return inferred
    inferred = _public_model_id_from_name(text)
    if inferred:
        return inferred
    basename = Path(text).name or text.split("/")[-1]
    return _sanitize_public_model_id(basename)


def _normalize_variant(value: str | None) -> tuple[str, str | None]:
    raw = str(value or "").strip().lower()
    if raw in {"", "auto"}:
        return "auto", None
    aliases = {
        "bf16": "bf16",
        "bfloat16": "bf16",
        "bfloat": "bf16",
        "fp16": "fp16",
        "float16": "fp16",
        "f16": "fp16",
    }
    normalized = aliases.get(raw)
    if normalized is None:
        return "auto", raw
    return normalized, raw


def _hardware_generation(hardware: Mapping[str, Any]) -> str:
    generation = str(hardware.get("apple_silicon_generation") or "").strip().lower()
    if generation:
        return generation
    return classify_apple_silicon_generation(
        str(hardware.get("chip") or ""),
        system=str(hardware.get("system") or ""),
        machine=str(hardware.get("machine") or ""),
    )


def select_default_model(
    *,
    variant_override: str | None = None,
    hardware: Mapping[str, Any] | None = None,
) -> DefaultModelSelection:
    """Select the verified default model for this machine.

    Auto policy is intentionally simple and visible:
    M1/M2 -> FP16, M3/M4/M5/unknown -> BF16.
    """

    env_value = variant_override if variant_override is not None else os.environ.get(DEFAULT_MODEL_VARIANT_ENV)
    requested_variant, invalid_override = _normalize_variant(env_value)
    hardware_info = dict(detect_apple_silicon() if hardware is None else hardware)
    generation = _hardware_generation(hardware_info)
    chip = str(hardware_info.get("chip") or "").strip()

    if requested_variant == "fp16":
        variant = "fp16"
        reason = f"forced by {DEFAULT_MODEL_VARIANT_ENV}=fp16"
        auto_selected = False
    elif requested_variant == "bf16":
        variant = "bf16"
        reason = f"forced by {DEFAULT_MODEL_VARIANT_ENV}=bf16"
        auto_selected = False
    elif generation in _LEGACY_APPLE_FP16_GENERATIONS:
        variant = "fp16"
        reason = "selected for M1/M2 Apple Silicon"
        auto_selected = True
    else:
        variant = "bf16"
        auto_selected = True
        if generation in _NEWER_APPLE_BF16_GENERATIONS:
            reason = "selected for newer Apple Silicon"
        elif generation == "intel":
            reason = "selected because this is not Apple Silicon"
        else:
            reason = "selected because hardware is unknown"

    if invalid_override is not None and requested_variant == "auto":
        reason = f"{reason}; ignored invalid {DEFAULT_MODEL_VARIANT_ENV}={invalid_override}"

    hf_model = DEFAULT_FP16_HF_MODEL_ID if variant == "fp16" else DEFAULT_HF_MODEL_ID
    precision = "FP16" if variant == "fp16" else "BF16"
    return DefaultModelSelection(
        model=hf_model,
        hf_model=hf_model,
        variant=variant,
        precision=precision,
        chip_generation=generation,
        chip=chip,
        reason=reason,
        auto_selected=auto_selected,
        env_override=env_value if env_value else None,
    )


def verified_default_refs() -> set[str]:
    root = _repo_root()
    refs = {
        DEFAULT_HF_MODEL_ID,
        DEFAULT_FP16_HF_MODEL_ID,
        DEFAULT_MODEL_ID,
        str(DEFAULT_RUNTIME_MODEL_DIR),
        str((root / DEFAULT_RUNTIME_MODEL_DIR).resolve()),
    }
    return {ref for ref in refs if ref}


def is_verified_default_model_ref(model: str | Path | None) -> bool:
    if model is None:
        return True
    text = str(model).strip()
    if not text:
        return True
    refs = verified_default_refs()
    if text in refs:
        return True
    if text.startswith(("~", "/", "./", "../")):
        path = Path(text).expanduser()
        if path.name in _VERIFIED_DEFAULT_LOCAL_NAMES:
            return True
        try:
            expanded = str(path.resolve())
        except OSError:
            expanded = str(path)
        return expanded in refs
    return False
