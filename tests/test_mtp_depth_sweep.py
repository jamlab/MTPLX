from __future__ import annotations

from types import SimpleNamespace

from mtplx.benchmarks.runners import mtp_depth_sweep


def test_depth_sweep_uses_packaged_draft_lm_head_helper(monkeypatch, tmp_path) -> None:
    calls = []
    fake_runtime = SimpleNamespace(
        model=object(),
        tokenizer=object(),
        contract=SimpleNamespace(
            mtp_quant_bits=None,
            mtp_quant_group_size=64,
            mtp_quant_mode="affine",
            mtp_quant_policy=None,
        ),
        mtp_adapter_metadata=None,
    )

    monkeypatch.setattr(mtp_depth_sweep, "load", lambda *_args, **_kwargs: fake_runtime)
    monkeypatch.setattr(mtp_depth_sweep, "load_prompt_suite", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "mtplx.draft_lm_head._install_draft_lm_head",
        lambda runtime, **kwargs: calls.append((runtime, kwargs)) or {"installed": True},
    )

    result = mtp_depth_sweep.run_mtp_depth_sweep(
        tmp_path / "model",
        tmp_path / "suite.jsonl",
        depths=[1],
        draft_lm_head_bits=4,
        draft_lm_head_group_size=64,
        draft_lm_head_mode="affine",
    )

    assert result["draft_lm_head"] == {"installed": True}
    assert calls == [
        (
            fake_runtime,
            {"bits": 4, "group_size": 64, "mode": "affine"},
        )
    ]
