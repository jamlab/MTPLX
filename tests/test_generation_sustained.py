from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import pytest

from mtplx.generation import _prefill, generate_ar
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


def test_sustained_prefill_forwards_logits_controls_through_patched_kwargs_wrapper(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = KwargsOnlyTinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert [call["emit_logits"] for call in model.calls] == [False, False, True]
    assert rt.diagnostic_counters.get("full_logits_tokens_emitted", 0) == 0


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
