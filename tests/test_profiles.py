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
)


def test_profile_registry_default_is_stable() -> None:
    profile = get_profile()

    assert DEFAULT_PROFILE_NAME == "performance-cold"
    assert profile.name == "performance-cold"
    assert profile.runtime_profile == "native_mtp_60_cold"
    assert profile.product_claim_eligible is True


def test_performance_cold_is_explicit_fast_path() -> None:
    profile = get_profile("performance-cold")

    assert profile.runtime_profile == "native_mtp_60_cold"
    assert profile.required_mlx_fork_commit == "2377a99f"
    assert profile.draft_lm_head is not None
    assert profile.env_dict() == NATIVE_MTP_60_FAST_PATH_ENV


def test_legacy_native_mtp_60_alias_resolves_to_performance_cold() -> None:
    assert get_profile("native-mtp-60").name == "performance-cold"


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
    assert "MTPLX_TRUNK_CACHE_MATERIALIZE_EVERY" not in profile.env_dict()
    assert "MTPLX_EVAL_STATE_ROOTS_ON_COMMIT" not in profile.env_dict()
