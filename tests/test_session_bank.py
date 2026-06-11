from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mtplx.session_bank import SessionBank


class DenseMaterializingCache:
    @property
    def state(self):
        raise RuntimeError("Paged KV cache attempted to materialize active K/V arrays")


class TrimmableLiveCache:
    def __init__(self, offset: int):
        self.offset = offset
        self.trimmed: list[int] = []

    @property
    def state(self):
        raise RuntimeError("Paged KV cache attempted to materialize active K/V arrays")

    def trim(self, n: int) -> int:
        self.trimmed.append(int(n))
        self.offset -= int(n)
        return int(n)


class RuntimeWithCaches:
    model_path = Path("models/example")
    mtp_enabled = True

    def make_cache(self):
        return [TrimmableLiveCache(0)]

    def make_mtp_cache(self):
        return [TrimmableLiveCache(0)]


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


def test_session_bank_oversized_prompt_prefix_can_use_live_reference_lease():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = RuntimeWithCaches()
    cache = [TrimmableLiveCache(offset=11)]
    mtp_cache = [TrimmableLiveCache(offset=11)]

    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(10)),
        cache=cache,
        logits="logits",
        hidden="hidden",
        keep_live_ref=True,
        session_id="session-1",
        mtp_history_policy="committed",
        mtp_history_cache_ref=mtp_cache,
        snapshot_epoch=10,
        mtp_snapshot_epoch=10,
        nbytes_override=2048,
    )

    assert entry is not None
    assert entry.live_ref_only is True
    assert entry.cache_ref is cache
    assert entry.mtp_history_cache_ref is mtp_cache
    assert bank.eviction_log[-1]["fallback"] == "live_reference_lease"

    restored = bank.restore(
        runtime,
        list(range(10)),
        mode="reference",
        session_id="session-1",
        mtp_history_policy="committed",
    )

    assert restored is not None
    assert restored.restore_mode == "reference_lease"
    assert restored.cache is cache
    assert restored.mtp_history_cache is mtp_cache
    assert cache[0].offset == 9
    assert mtp_cache[0].offset == 9
    assert entry.cache_ref is None
    assert entry.mtp_history_cache_ref is None


def test_session_bank_clone_restore_can_use_custom_cache_factory():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    custom_cache = ["prefill-layout-cache"]

    entry = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3],
        cache=[],
        logits="logits",
        hidden="hidden",
        session_id="session-1",
        nbytes_override=128,
    )
    assert entry is not None

    restored = bank.restore(
        runtime,
        [1, 2, 3, 4],
        mode="clone",
        cache_factory=lambda: custom_cache,
    )

    assert restored is not None
    assert restored.cache is custom_cache
    assert restored.restore_mode == "clone"


def test_session_bank_live_reference_can_restore_block_prefix_boundary():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = RuntimeWithCaches()
    cache = [TrimmableLiveCache(offset=1199)]
    mtp_cache = [TrimmableLiveCache(offset=1199)]

    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(1200)),
        cache=cache,
        logits="logits",
        hidden="hidden",
        keep_live_ref=True,
        session_id="session-1",
        mtp_history_policy="committed",
        mtp_history_cache_ref=mtp_cache,
        snapshot_epoch=1200,
        mtp_snapshot_epoch=1200,
        nbytes_override=2048,
    )

    assert entry is not None
    assert entry.live_ref_only is True

    restored = bank.restore_entry_prefix_cache(
        runtime,
        entry,
        1024,
        mode="reference",
    )

    assert restored is not None
    restored_cache, restored_mtp_cache, restore_mode = restored
    assert restored_cache is cache
    assert restored_mtp_cache is mtp_cache
    assert restore_mode == "reference_lease"
    assert cache[0].offset == 1023
    assert mtp_cache[0].offset == 1023
    assert entry.cache_ref is None
    assert entry.mtp_history_cache_ref is None


def test_session_bank_near_prefix_trims_mtp_history_by_gap_not_absolute_offset():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = RuntimeWithCaches()
    cache = [TrimmableLiveCache(offset=1199)]
    mtp_cache = [TrimmableLiveCache(offset=127)]

    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(1200)),
        cache=cache,
        logits="logits",
        hidden="hidden",
        keep_live_ref=True,
        session_id="session-1",
        mtp_history_policy="last_window",
        mtp_history_cache_ref=mtp_cache,
        snapshot_epoch=1200,
        mtp_snapshot_epoch=1200,
        nbytes_override=2048,
    )

    assert entry is not None

    restored = bank.restore_entry_prefix_cache(
        runtime,
        entry,
        1199,
        mode="reference",
    )

    assert restored is not None
    assert cache[0].trimmed == [1]
    assert mtp_cache[0].trimmed == [1]
    assert cache[0].offset == 1198
    assert mtp_cache[0].offset == 126


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
            allow_block_prefix=False,
        )
        == []
    )


def test_session_bank_near_prefix_rejects_prompt_inside_longer_completion():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(70)) + [90_001, 90_002],
        cache=[],
        logits=None,
        hidden=None,
        session_id="session-1",
        nbytes_override=128,
    )
    assert entry is not None

    prompt_only = list(range(70))

    assert (
        bank.near_prefix_candidates(
            prompt_only,
            max_token_gap=8,
            min_matched_tokens=64,
            allow_block_prefix=True,
        )
        == []
    )


def test_session_bank_contained_long_prompt_uses_block_prefix_not_answer_tail():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(1200)),
        cache=[],
        logits=None,
        hidden=None,
        session_id="session-1",
        nbytes_override=128,
    )
    assert entry is not None

    prompt_inside_completion = list(range(1197))
    candidates = bank.near_prefix_candidates(
        prompt_inside_completion,
        max_token_gap=8,
        min_matched_tokens=64,
        block_size=256,
        block_min_matched_tokens=512,
        allow_block_prefix=True,
    )

    assert candidates == [(entry, 1024)]
    assert bank.last_prefix_diagnostic is not None
    assert bank.last_prefix_diagnostic["restore_kind"] == "block_prefix"


def test_session_bank_block_prefix_candidates_restore_large_agent_overlap():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(1200)),
        cache=[],
        logits=None,
        hidden=None,
        session_id="session-1",
        nbytes_override=128,
    )
    assert entry is not None

    followup = list(range(1050)) + [99_001, 99_002, 99_003]
    candidates = bank.near_prefix_candidates(
        followup,
        max_token_gap=8,
        min_matched_tokens=64,
        block_size=256,
        block_min_matched_tokens=512,
        allow_block_prefix=True,
    )

    assert candidates == [(entry, 1024)]
    assert bank.last_prefix_diagnostic is not None
    assert bank.last_prefix_diagnostic["restore_kind"] == "block_prefix"
    assert bank.last_prefix_diagnostic["new_prefill_tokens"] == len(followup) - 1024
