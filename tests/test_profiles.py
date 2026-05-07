from __future__ import annotations

from mtplx.profiles import (
    DEFAULT_PROFILE_NAME,
    NATIVE_MTP_60_FAST_PATH_ENV,
    SUSTAINED_PREFILL_ENV,
    apply_profile_env,
    get_profile,
    list_profiles,
    profile_env_status,
    restore_profile_env,
    resolve_long_context_mtp_depth,
)


def test_profile_registry_default_is_sustained() -> None:
    profile = get_profile()

    assert DEFAULT_PROFILE_NAME == "sustained"
    assert profile.name == "sustained"
    assert profile.runtime_profile == "native_mtp_sustained"
    assert profile.product_claim_eligible is True


def test_performance_cold_is_explicit_fast_path() -> None:
    profile = get_profile("performance-cold")

    assert profile.runtime_profile == "native_mtp_60_cold"
    assert profile.required_mlx_fork_commit == "2377a99f"
    assert profile.draft_lm_head is not None
    assert profile.env_dict() == NATIVE_MTP_60_FAST_PATH_ENV
    assert "MTPLX_SUSTAINED_PREFILL_LAYOUT" not in profile.env_dict()


def test_legacy_native_mtp_60_alias_resolves_to_performance_cold() -> None:
    assert get_profile("native-mtp-60").name == "performance-cold"
    assert get_profile("default").name == "sustained"


def test_apply_and_restore_profile_env() -> None:
    environ: dict[str, str] = {}

    previous = apply_profile_env("performance-cold", environ=environ)
    assert previous == {key: None for key in NATIVE_MTP_60_FAST_PATH_ENV}
    assert profile_env_status("performance-cold", environ=environ)[
        "MTPLX_LAZY_VERIFY_LOGITS"
    ]["ok"] is True

    restore_profile_env(previous, environ=environ)
    assert environ == {}


def test_list_profiles_includes_all_public_modes() -> None:
    names = [profile["name"] for profile in list_profiles()]

    assert names == ["stable", "performance-cold", "sustained", "exact", "max-diagnostic"]


def test_sustained_profile_is_native_mtp_long_context_path() -> None:
    profile = get_profile("sustained")

    assert profile.runtime_profile == "native_mtp_sustained"
    assert profile.draft_lm_head is not None
    assert profile.env_dict() == SUSTAINED_PREFILL_ENV
    assert profile.env_dict()["MTPLX_SUSTAINED_PREFILL_LAYOUT"] == "auto"
    assert profile.env_dict()["MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT"] == "131072"
    assert profile.env_dict()["MTPLX_PREFILL_CHUNK_SIZE"] == "auto"
    assert profile.env_dict()["MTPLX_PREFILL_CHUNK_SIZE_DENSE"] == "2048"
    assert profile.env_dict()["MTPLX_PREFILL_CHUNK_SIZE_REPAGE"] == "2048"
    assert profile.env_dict()["MTPLX_PREFILL_CHUNK_CACHE_CLEANUP"] == "1"
    assert profile.env_dict()["MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY"] == "auto"
    assert profile.env_dict()["MTPLX_PREFILL_OMLX_EXTERNAL"] == "1"
    assert profile.env_dict()["MTPLX_PREFILL_EXTERNAL_EMIT_LOGITS"] == "0"
    assert profile.env_dict()["MTPLX_CLEAR_CACHE_EVERY"] == "auto"
    assert profile.env_dict()["MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD"] == "98304"
    assert profile.env_dict()["MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT"] == "16"
    assert profile.env_dict()["MTPLX_LAZY_VERIFY_LOGITS"] == "1"
    assert profile.env_dict()["MTPLX_BATCH_TARGET_ARRAYS"] == "1"
    assert profile.env_dict()["MTPLX_DEFER_VERIFY_HIDDEN_EVAL"] == "1"
    assert profile.env_dict()["MTPLX_VERIFY_HIDDEN_MODE"] == "logits_first_committed_slice"
    assert profile.env_dict()["MTPLX_LONG_CONTEXT_MTP_DEPTH_POLICY"] == "auto"
    assert profile.env_dict()["MTPLX_LONG_CONTEXT_MTP_DEPTH_THRESHOLD"] == "98304"
    assert profile.env_dict()["MTPLX_LONG_CONTEXT_MTP_DEPTH"] == "2"
    assert profile.env_dict()["MTPLX_LAZY_MTP_HISTORY_APPEND"] == "1"
    assert profile.env_dict()["MTPLX_DROP_EVENTS"] == "1"
    assert profile.env_dict()["MTPLX_SKIP_VERIFY_SNAPSHOT"] == "1"
    assert profile.env_dict()["MTPLX_VLLM_METAL_PAGED_TURBOQUANT"] == "0"
    assert "MTPLX_TRUNK_CACHE_MATERIALIZE_EVERY" not in profile.env_dict()
    assert "MTPLX_EVAL_STATE_ROOTS_ON_COMMIT" not in profile.env_dict()


def test_sustained_long_context_depth_policy_caps_only_128k_class() -> None:
    env = SUSTAINED_PREFILL_ENV

    depth, details = resolve_long_context_mtp_depth(
        prompt_tokens=65536,
        requested_depth=3,
        env=env,
    )
    assert depth == 3
    assert details["active"] is False
    assert details["reason"] == "below_threshold"

    depth, details = resolve_long_context_mtp_depth(
        prompt_tokens=131072,
        requested_depth=3,
        env=env,
    )
    assert depth == 2
    assert details["active"] is True
    assert details["reason"] == "long_context_depth_cap"
    assert details["requested_depth"] == 3
    assert details["effective_depth"] == 2

    depth, details = resolve_long_context_mtp_depth(
        prompt_tokens=131072,
        requested_depth=1,
        env=env,
    )
    assert depth == 1
    assert details["active"] is False
    assert details["reason"] == "already_within_cap"
