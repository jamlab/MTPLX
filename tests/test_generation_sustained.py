from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.generation import (
    _clear_cache_every,
    _defer_verify_hidden_eval_enabled,
    _make_target_prefill_cache,
    _maybe_repage_target_prefill_cache,
    _prefill,
    _prefill_chunk_cache_cleanup_every,
    _prefill_chunk_size,
    _prefill_committed_mtp_history_streaming,
    _sustained_prefill_layout,
    generate_ar,
    restore_or_prefill_prompt_state,
)
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig


class TinyTokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(str(int(token)) for token in tokens)


class TinyModel:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def mtp_update_cache(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        concat_order=None,
        position_offset=None,
    ):
        return hidden_states

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
    ):
        self.calls.append(
            {
                "tokens": int(input_ids.shape[1]),
                "return_hidden": bool(return_hidden),
                "emit_logits": bool(emit_logits),
                "logits_keep": logits_keep,
            }
        )
        length = int(input_ids.shape[1])
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        if not emit_logits:
            if return_hidden:
                return None, hidden
            return None
        keep = length if logits_keep is None else min(length, max(1, int(logits_keep)))
        logits = mx.zeros((1, keep, 4), dtype=mx.float32)
        logits = logits + mx.array([0.0, 1.0, 0.0, 0.0], dtype=mx.float32)
        if return_hidden:
            return logits, hidden
        return logits


class KwargsOnlyTinyModel(TinyModel):
    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        **kwargs,
    ):
        return super().__call__(
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
            hidden_variant=hidden_variant,
            **kwargs,
        )


def _runtime(model: TinyModel, *, mtp_enabled: bool = True) -> MTPLXRuntime:
    return MTPLXRuntime(
        model=model,
        tokenizer=TinyTokenizer(),
        model_path=Path("tiny"),
        mtp_enabled=mtp_enabled,
        contract=MTPContract(),
    )


def test_contiguous_then_repage_cache_layout_restores_paged_env(monkeypatch):
    cache: list[object] = []
    events: list[tuple[str, str | None]] = []

    class Runtime:
        def make_cache(self):
            events.append(("make_cache", os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN")))
            return cache

    def configure(received_cache):
        events.append(("repage", os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN")))
        assert received_cache is cache
        return {"enabled": 1, "entries": 0, "skipped": 0}

    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "contiguous_then_repage")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_OWNED_ATTN_KV", "1")
    monkeypatch.setenv("MTPLX_BLOCK_OWNED_ATTN_KV", "1")
    monkeypatch.setattr(
        "mtplx.cache_state.configure_tail_owned_attention_kv_cache",
        configure,
    )

    made_cache = _make_target_prefill_cache(Runtime())
    elapsed = _maybe_repage_target_prefill_cache(made_cache)

    assert elapsed >= 0.0
    assert events == [("make_cache", "0"), ("repage", "1")]
    assert os.environ["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert os.environ["MTPLX_OWNED_ATTN_KV"] == "1"
    assert os.environ["MTPLX_BLOCK_OWNED_ATTN_KV"] == "1"


def test_contiguous_dense_decode_cache_layout_does_not_repage(monkeypatch):
    cache: list[object] = []
    events: list[tuple[str, str | None]] = []

    class Runtime:
        def make_cache(self):
            events.append(("make_cache", os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN")))
            return cache

    def configure(_received_cache):
        raise AssertionError("dense decode layout must not repage after prefill")

    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "contiguous_dense_decode")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_OWNED_ATTN_KV", "1")
    monkeypatch.setenv("MTPLX_BLOCK_OWNED_ATTN_KV", "1")
    monkeypatch.setattr(
        "mtplx.cache_state.configure_tail_owned_attention_kv_cache",
        configure,
    )

    made_cache = _make_target_prefill_cache(Runtime())
    elapsed = _maybe_repage_target_prefill_cache(made_cache)

    assert elapsed == 0.0
    assert events == [("make_cache", "0")]
    assert os.environ["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert os.environ["MTPLX_OWNED_ATTN_KV"] == "1"
    assert os.environ["MTPLX_BLOCK_OWNED_ATTN_KV"] == "1"


def test_auto_sustained_prefill_policy_keeps_dense_decode_through_128k(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "131072")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "2048")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_REPAGE", "2048")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY", "auto")
    monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "auto")
    monkeypatch.setenv("MTPLX_CLEAR_CACHE_EVERY", "auto")
    monkeypatch.setenv("MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD", "16384")
    monkeypatch.setenv("MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT", "256")

    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "65536")
    assert _sustained_prefill_layout() == "contiguous_dense_decode"
    assert _prefill_chunk_size() == 2048
    assert _prefill_chunk_cache_cleanup_every() == 1
    assert _defer_verify_hidden_eval_enabled() is True
    assert _clear_cache_every() == 256

    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "131072")
    assert _sustained_prefill_layout() == "contiguous_dense_decode"
    assert _prefill_chunk_size() == 2048
    assert _prefill_chunk_cache_cleanup_every() == 1
    assert _defer_verify_hidden_eval_enabled() is True
    assert _clear_cache_every() == 256

    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "196608")
    assert _sustained_prefill_layout() == "contiguous_then_repage"
    assert _prefill_chunk_cache_cleanup_every() == 2
    assert _clear_cache_every() == 0


def test_non_sustained_long_context_prefill_is_blocked_before_full_hidden_eval(monkeypatch):
    monkeypatch.delenv("MTPLX_SUSTAINED_PREFILL", raising=False)
    monkeypatch.delenv("MTPLX_ALLOW_UNSAFE_LONG_CONTEXT_PREFILL", raising=False)
    monkeypatch.setenv("MTPLX_UNSAFE_LONG_CONTEXT_PREFILL_GUARD_TOKENS", "8")
    model = TinyModel()

    with pytest.raises(RuntimeError, match="Blocked unsafe long-context MTP prefill path"):
        restore_or_prefill_prompt_state(
            _runtime(model, mtp_enabled=True),
            list(range(8)),
            mtp_history_policy="committed",
        )

    assert model.calls == []


def test_non_sustained_long_context_prefill_guard_has_explicit_escape_hatch(monkeypatch):
    monkeypatch.delenv("MTPLX_SUSTAINED_PREFILL", raising=False)
    monkeypatch.setenv("MTPLX_ALLOW_UNSAFE_LONG_CONTEXT_PREFILL", "1")
    monkeypatch.setenv("MTPLX_UNSAFE_LONG_CONTEXT_PREFILL_GUARD_TOKENS", "8")
    model = TinyModel()

    restore_or_prefill_prompt_state(
        _runtime(model, mtp_enabled=True),
        list(range(8)),
        mtp_history_policy="committed",
    )

    assert model.calls


def test_generate_ar_does_not_request_hidden_by_default(monkeypatch):
    monkeypatch.delenv("MTPLX_AR_RETURN_HIDDEN", raising=False)
    monkeypatch.delenv("MTPLX_DIAGNOSTIC_AR_RETURN_HIDDEN", raising=False)
    model = TinyModel()

    out = generate_ar(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=2,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        stop_token_ids=set(),
    )

    assert out.stats.ar_return_hidden is False
    assert out.stats.forward_ar_hidden_calls == 0
    assert out.stats.forward_ar_plain_calls >= 1
    assert out.stats.prompt_target_prefill_time_s == out.stats.prompt_eval_time_s
    assert out.stats.prompt_mtp_history_time_s == 0.0
    assert out.stats.prompt_target_prefill_tok_s > 0.0
    assert all(call["return_hidden"] is False for call in model.calls)


def test_sustained_prefill_chunks_without_full_prompt_logits(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert [call["tokens"] for call in model.calls] == [2, 2, 1]
    assert [call["emit_logits"] for call in model.calls] == [False, False, True]
    assert model.calls[-1]["logits_keep"] == 1
    assert rt.diagnostic_counters["prefill_chunks"] == 2
    assert rt.diagnostic_counters.get("full_logits_tokens_emitted", 0) == 0
    assert rt.diagnostic_counters["final_logits_tokens_emitted"] == 1


def test_warm_restored_suffix_prefill_is_chunked_and_typed_for_abort(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    appended: list[list[int]] = []

    class Bank:
        last_miss_reason = None

        def restore(self, *_args, **_kwargs):
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=3),
                cache=[],
                logits=mx.zeros((1, 4), dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                mtp_history_cache=[],
                restore_mode="clone",
            )

    def append_history(
        _rt,
        _mtp_cache,
        hidden_states,
        token_ids,
        *,
        mtp_hidden_variant,
        position_offset=None,
        force_eval=False,
    ):
        assert hidden_states.shape[1] == len(token_ids)
        assert force_eval is True
        appended.append(list(token_ids))
        return 0.0

    monkeypatch.setattr("mtplx.generation._append_mtp_history", append_history)

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        [0, 1, 2, 3, 4, 5, 6],
        mtp_history_policy="committed",
        session_bank=Bank(),
    )

    assert prompt_state.cache_hit is True
    assert prompt_state.cached_tokens == 3
    assert prompt_state.suffix_tokens == 4
    assert [call["tokens"] for call in model.calls] == [2, 1, 1]
    assert [call["return_hidden"] for call in model.calls] == [True, True, True]
    assert [call["emit_logits"] for call in model.calls] == [False, False, True]
    assert appended == [[3], [4, 5], [6]]
    assert rt.diagnostic_counters["restored_suffix_prefill_chunks"] == 2


def test_sustained_prefill_chunk_cache_cleanup_is_explicit(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_CACHE_CLEANUP", "1")
    calls: list[str] = []
    monkeypatch.setattr("mtplx.generation.mx.synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr("mtplx.generation.mx.clear_cache", lambda: calls.append("clear"))
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert calls == ["sync", "clear", "sync", "clear"]
    assert rt.diagnostic_counters["prefill_chunk_cache_cleanup_events"] == 2


def test_sustained_prefill_stock_cache_only_requires_unsafe_allow(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    monkeypatch.setenv("MTPLX_PREFILL_STOCK_CACHE_ONLY", "1")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert [call["tokens"] for call in model.calls] == [2, 2, 1]
    assert [call["return_hidden"] for call in model.calls] == [False, False, True]
    assert [call["emit_logits"] for call in model.calls] == [False, False, True]
    assert rt.diagnostic_counters.get("prefill_stock_cache_only_calls", 0) == 0


def test_sustained_prefill_stock_cache_only_is_explicit_unsafe(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    monkeypatch.setenv("MTPLX_PREFILL_STOCK_CACHE_ONLY", "1")
    monkeypatch.setenv("MTPLX_ALLOW_UNSAFE_PREFILL_STOCK_CACHE_ONLY", "1")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert [call["tokens"] for call in model.calls] == [2, 2, 1]
    assert [call["return_hidden"] for call in model.calls] == [False, False, True]
    assert [call["emit_logits"] for call in model.calls] == [True, True, True]
    assert rt.diagnostic_counters["prefill_external_cache_only_calls"] == 2
    assert rt.diagnostic_counters["prefill_stock_cache_only_calls"] == 2


def test_sustained_prefill_omlx_external_is_safe_profile_path(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    monkeypatch.setenv("MTPLX_PREFILL_OMLX_EXTERNAL", "1")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert [call["tokens"] for call in model.calls] == [2, 2, 1]
    assert [call["return_hidden"] for call in model.calls] == [False, False, True]
    assert [call["emit_logits"] for call in model.calls] == [True, True, True]
    assert rt.diagnostic_counters["prefill_external_cache_only_calls"] == 2
    assert rt.diagnostic_counters["prefill_omlx_external_calls"] == 2
    assert rt.diagnostic_counters.get("prefill_stock_cache_only_calls", 0) == 0


def test_sustained_prefill_forwards_logits_controls_through_patched_kwargs_wrapper(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = KwargsOnlyTinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert [call["emit_logits"] for call in model.calls] == [False, False, True]
    assert rt.diagnostic_counters.get("full_logits_tokens_emitted", 0) == 0


def test_last_window_mtp_history_skips_discarded_chunk_hidden(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    appended: list[tuple[list[int], int | None]] = []

    def append_history(
        _rt,
        _mtp_cache,
        hidden_states,
        token_ids,
        *,
        mtp_hidden_variant,
        position_offset=None,
        force_eval=False,
    ):
        appended.append((list(token_ids), position_offset))
        return 0.0

    monkeypatch.setattr("mtplx.generation._append_mtp_history", append_history)

    _prefill_committed_mtp_history_streaming(
        rt,
        list(range(9)),
        mtp_hidden_variant="post_norm",
        history_window_tokens=3,
    )

    assert [call["tokens"] for call in model.calls] == [2, 2, 2, 2, 1]
    assert [call["return_hidden"] for call in model.calls] == [
        False,
        False,
        True,
        True,
        True,
    ]
    assert appended == [([6], 5), ([7, 8], 6)]


def test_32k_prefill_peak_memory_bounded():
    """
    Regression guard for the Ivan/Benchand 32K memory balloon.
    Run only on the Apple Silicon long-context QA machine.
    """
    if os.environ.get("MTPLX_RUN_32K_MEMORY_QA") != "1":
        pytest.skip("set MTPLX_RUN_32K_MEMORY_QA=1 on the long-context QA Mac")
    model_path = os.environ.get("MTPLX_32K_QA_MODEL")
    if not model_path:
        pytest.skip("set MTPLX_32K_QA_MODEL to a local runnable MTPLX model")

    from mtplx.runtime import load

    rt = load(model_path, mtp=True)
    text = ("def f(x): return x + 1\n" * 4096)
    prompt_ids = rt.tokenizer.encode(text)[:32768]
    if len(prompt_ids) < 32000:
        pytest.skip("QA prompt did not tokenize to 32K tokens")

    mx.reset_peak_memory()
    os.environ["MTPLX_SUSTAINED_PREFILL"] = "1"
    os.environ["MTPLX_PREFILL_CHUNK_SIZE"] = "2048"
    os.environ["MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS"] = "0"
    _prefill(rt, prompt_ids, return_hidden=True)
    peak_gb = mx.get_peak_memory() / (1024**3)

    assert peak_gb < 35.0, f"32K Sustained prefill peak was {peak_gb:.1f} GB"
