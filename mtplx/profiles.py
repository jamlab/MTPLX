"""Public MTPLX runtime profiles.

Profiles are product policy, not benchmark folklore.  They are deliberately
kept free of MLX imports so the CLI can explain the available modes in a fresh
environment before the runtime stack is installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, MutableMapping


ProfileName = str

DEFAULT_PROFILE_NAME = "performance-cold"
PROFILE_CHOICES = (
    "stable",
    "performance-cold",
    "sustained",
    "exact",
    "max-diagnostic",
)

DEFAULT_HF_MODEL_ID = "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
DEFAULT_MODEL_ID = DEFAULT_HF_MODEL_ID
DEFAULT_PUBLIC_MODEL_ID = "mtplx-qwen36-27b-optimized-speed"
NATIVE_MTP_60_MLX_FORK_COMMIT = "2377a99f"
NATIVE_MTP_60_MLX_FORK_FRAGMENT = "mlx-mtplx-0.31.2-qmm"


NATIVE_MTP_60_FAST_PATH_ENV = {
    "MTPLX_LAZY_VERIFY_LOGITS": "1",
    "MTPLX_BATCH_TARGET_ARRAYS": "1",
    "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
    "MTPLX_DROP_EVENTS": "1",
    "MTPLX_SKIP_VERIFY_SNAPSHOT": "1",
}

EXACT_PAGED_ATTENTION_ENV = {
    "MTPLX_VLLM_METAL_PAGED_ATTN": "1",
    "MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE": "16",
    "MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS": "1024",
    "MTPLX_VLLM_METAL_PAGED_ATTN_IMPL": "mlx_vector_paged",
    "MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN": "1",
    "MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD": "2048",
    "MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE": "512",
}

LONG_RESPONSE_STAGED_ENV = {
    "MTPLX_DROP_EVENTS": "1",
    "MTPLX_EVAL_STATE_ROOTS_ON_COMMIT": "1",
    "MTPLX_EVAL_STATE_ROOTS_INCLUDE_MTP": "1",
    "MTPLX_EVAL_STATE_ROOTS_INCLUDE_LIVE": "1",
    "MTPLX_TARGET_LAYER_EVAL_SCHEDULE": "2048:16,8192:8",
    "MTPLX_TARGET_LAYER_EVAL_CONTEXT_THRESHOLD": "0",
    "MTPLX_TARGET_LAYER_EVAL_MAX_Q": "8",
}

SUSTAINED_PREFILL_ENV = {
    "MTPLX_SUSTAINED_PREFILL": "1",
    "MTPLX_PREFILL_CHUNK_SIZE": "2048",
    "MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS": "0",
    "MTPLX_DYNAMIC_PAGED_KV": "1",
    "MTPLX_VLLM_METAL_PAGED_ATTN": "1",
    "MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE": "16",
    "MTPLX_VLLM_METAL_PAGED_ATTN_IMPL": "mlx_vector_paged",
    "MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN": "1",
    "MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD": "2048",
    "MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE": "512",
    "MTPLX_CLEAR_CACHE_EVERY": "0",
}


@dataclass(frozen=True)
class SamplerDefaults:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20


@dataclass(frozen=True)
class DraftLMHeadRequirement:
    bits: int
    group_size: int
    mode: str


@dataclass(frozen=True)
class RuntimeProfile:
    name: ProfileName
    runtime_profile: str
    summary: str
    env: tuple[tuple[str, str], ...]
    sampler: SamplerDefaults = SamplerDefaults()
    model_id: str = DEFAULT_MODEL_ID
    benchmark_ids: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    required_mlx_fork_commit: str | None = None
    required_mlx_fork_fragment: str | None = None
    draft_lm_head: DraftLMHeadRequirement | None = None
    draft_sampler: SamplerDefaults | None = None
    qa_only: bool = False
    fan_control_allowed: bool = False
    clock_anchor_allowed: bool = False
    product_claim_eligible: bool = True

    def env_dict(self) -> dict[str, str]:
        return dict(self.env)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "runtime_profile": self.runtime_profile,
            "summary": self.summary,
            "env": self.env_dict(),
            "sampler": {
                "temperature": self.sampler.temperature,
                "top_p": self.sampler.top_p,
                "top_k": self.sampler.top_k,
            },
            "model_id": self.model_id,
            "benchmark_ids": list(self.benchmark_ids),
            "caveats": list(self.caveats),
            "required_mlx_fork_commit": self.required_mlx_fork_commit,
            "required_mlx_fork_fragment": self.required_mlx_fork_fragment,
            "draft_lm_head": (
                None
                if self.draft_lm_head is None
                else {
                    "bits": self.draft_lm_head.bits,
                    "group_size": self.draft_lm_head.group_size,
                    "mode": self.draft_lm_head.mode,
                }
            ),
            "draft_sampler": (
                None
                if self.draft_sampler is None
                else {
                    "temperature": self.draft_sampler.temperature,
                    "top_p": self.draft_sampler.top_p,
                    "top_k": self.draft_sampler.top_k,
                }
            ),
            "qa_only": self.qa_only,
            "fan_control_allowed": self.fan_control_allowed,
            "clock_anchor_allowed": self.clock_anchor_allowed,
            "product_claim_eligible": self.product_claim_eligible,
        }


def _items(mapping: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(mapping.items()))


def _merge_env(*mappings: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    env: dict[str, str] = {}
    for mapping in mappings:
        env.update(mapping)
    return _items(env)


STABLE_PROFILE = RuntimeProfile(
    name="stable",
    runtime_profile="long_response_exact_staged",
    summary=(
        "Stable Mode: exact/staged long-reply path with no fan control. Hidden "
        "from first-run onboarding, but available by flag for compatibility."
    ),
    env=_merge_env(EXACT_PAGED_ATTENTION_ENV, LONG_RESPONSE_STAGED_ENV),
    benchmark_ids=(
        "gdn8-flappy-nofan-directhttp-seed42-cli-fix-20260501",
        "phase0j-gdn8-speed4-target-layer-sched-state-root-mlx-vector-paged-flappy-uncapped-modelswap-20260501",
    ),
    caveats=(
        "Lower peak throughput than the fan-backed Burst lane.",
        "Selected for repeatable long replies while the v0.2 decay work continues.",
    ),
)

PERFORMANCE_COLD_PROFILE = RuntimeProfile(
    name="performance-cold",
    runtime_profile="native_mtp_60_cold",
    summary=(
        "Burst engine: native-MTP performance-cold path for the old max-fan "
        "headline lane. Use only for short contexts, recommended max 8K."
    ),
    env=_items(NATIVE_MTP_60_FAST_PATH_ENV),
    benchmark_ids=(
        "mtp-depth-d3-gdn8-speed4-cyankiwi-mtp-draftlmhead4b-gs64-linear-gdn-from-conv-tape-mlx-qmv-unroll4-clean-preflight-batchedtargetarrays-lazymtphistory-dropevents-skipsnapshot-v4-20260429-143701",
    ),
    caveats=(
        "Best fan-backed burst throughput; not recommended for long context.",
        "Requires the MLX-MTPLX fork for the native QMV/QMM fast path.",
    ),
    required_mlx_fork_commit=NATIVE_MTP_60_MLX_FORK_COMMIT,
    required_mlx_fork_fragment=NATIVE_MTP_60_MLX_FORK_FRAGMENT,
    draft_lm_head=DraftLMHeadRequirement(bits=4, group_size=64, mode="affine"),
)

SUSTAINED_PROFILE = RuntimeProfile(
    name="sustained",
    runtime_profile="native_mtp_sustained",
    summary=(
        "Sustained Mode: explicit long-context native-MTP path with chunked "
        "prefill, final-token logits, and request-sized paged KV."
    ),
    env=_items(SUSTAINED_PREFILL_ENV),
    caveats=(
        "User-selected; no automatic profile switching.",
        "Targets long-context memory safety while preserving most Burst TPS.",
        "Does not include v0.2 decode-state eval scheduling flags.",
    ),
    required_mlx_fork_commit=NATIVE_MTP_60_MLX_FORK_COMMIT,
    required_mlx_fork_fragment=NATIVE_MTP_60_MLX_FORK_FRAGMENT,
    draft_lm_head=DraftLMHeadRequirement(bits=4, group_size=64, mode="affine"),
)

EXACT_PROFILE = RuntimeProfile(
    name="exact",
    runtime_profile="exact",
    summary="QA-only exact paged verifier profile with release gates enabled.",
    env=_items(EXACT_PAGED_ATTENTION_ENV),
    qa_only=True,
    product_claim_eligible=False,
)

MAX_DIAGNOSTIC_PROFILE = RuntimeProfile(
    name="max-diagnostic",
    runtime_profile="max_diagnostic",
    summary=(
        "Diagnostic fan-control profile for QA-only experiments."
    ),
    env=_merge_env(EXACT_PAGED_ATTENTION_ENV, LONG_RESPONSE_STAGED_ENV),
    caveats=(
        "Requires explicit --max before fan control is allowed.",
        "Clock-anchor behavior is separate and experimental.",
    ),
    fan_control_allowed=True,
    clock_anchor_allowed=True,
    product_claim_eligible=False,
)

PROFILES: dict[ProfileName, RuntimeProfile] = {
    STABLE_PROFILE.name: STABLE_PROFILE,
    PERFORMANCE_COLD_PROFILE.name: PERFORMANCE_COLD_PROFILE,
    SUSTAINED_PROFILE.name: SUSTAINED_PROFILE,
    EXACT_PROFILE.name: EXACT_PROFILE,
    MAX_DIAGNOSTIC_PROFILE.name: MAX_DIAGNOSTIC_PROFILE,
}

PROFILE_ALIASES = {
    "default": "performance-cold",
    "safe": "stable",
    "native-mtp-60": "performance-cold",
    "native_mtp_60": "performance-cold",
    "native_mtp_60_cold": "performance-cold",
    "long_response_exact_staged": "stable",
    "max": "max-diagnostic",
}


def resolve_profile_name(name: str | None) -> ProfileName:
    raw = (name or DEFAULT_PROFILE_NAME).strip()
    resolved = PROFILE_ALIASES.get(raw, raw)
    if resolved not in PROFILES:
        choices = ", ".join(PROFILE_CHOICES)
        raise ValueError(f"unknown MTPLX profile {raw!r}; expected one of: {choices}")
    return resolved


def get_profile(name: str | None = None) -> RuntimeProfile:
    return PROFILES[resolve_profile_name(name)]


def list_profiles() -> list[dict[str, object]]:
    return [PROFILES[name].to_dict() for name in PROFILE_CHOICES]


def apply_profile_env(
    name: str | None,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str | None]:
    target = os.environ if environ is None else environ
    profile = get_profile(name)
    previous = {key: target.get(key) for key in profile.env_dict()}
    for key, value in profile.env:
        target[key] = value
    return previous


def restore_profile_env(
    previous: Mapping[str, str | None],
    *,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    target = os.environ if environ is None else environ
    for key, value in previous.items():
        if value is None:
            target.pop(key, None)
        else:
            target[key] = value


def profile_env_status(
    name: str | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, dict[str, object]]:
    target = os.environ if environ is None else environ
    profile = get_profile(name)
    return {
        key: {
            "expected": expected,
            "observed": target.get(key),
            "ok": target.get(key) == expected,
        }
        for key, expected in profile.env
    }
