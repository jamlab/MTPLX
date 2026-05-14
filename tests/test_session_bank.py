from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mtplx.session_bank import SessionBank


class DenseMaterializingCache:
    @property
    def state(self):
        raise RuntimeError("Paged KV cache attempted to materialize active K/V arrays")


def test_session_bank_skips_single_oversized_snapshot_before_insert():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)

    entry = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3],
        cache=[],
        logits=None,
        hidden=None,
        session_id="session-1",
        nbytes_override=2048,
    )

    assert entry is None
    assert len(bank) == 0
    assert bank.last_put_nbytes == 2048
    assert bank.last_put_skipped_oversized_snapshot is True
    assert bank.eviction_log[-1]["reason"] == "skipped_oversized_snapshot"


def test_session_bank_skips_dense_materializing_snapshot():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)

    entry = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3],
        cache=[DenseMaterializingCache()],
        logits=None,
        hidden=None,
        session_id="session-1",
    )

    assert entry is None
    assert len(bank) == 0
    assert bank.last_put_skipped_oversized_snapshot is True
    assert bank.eviction_log[-1]["reason"] == "skipped_dense_materializing_snapshot"


def test_session_bank_near_prefix_candidates_only_accept_boundary_drift():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(200)),
        cache=[],
        logits=None,
        hidden=None,
        session_id="session-1",
        nbytes_override=128,
    )
    assert entry is not None

    near = list(range(197)) + [10_001, 10_002, 10_003, 10_004]
    far = list(range(120)) + [20_001, 20_002]

    candidates = bank.near_prefix_candidates(
        near,
        max_token_gap=8,
        min_matched_tokens=64,
    )

    assert candidates == [(entry, 197)]
    assert (
        bank.near_prefix_candidates(
            far,
            max_token_gap=8,
            min_matched_tokens=64,
        )
        == []
    )
