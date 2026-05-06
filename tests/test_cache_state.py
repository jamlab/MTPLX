from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from mtplx.attention_context import attention_phase
from mtplx.cache_state import (
    BlockOwnedKVCache,
    OwnedRecurrentStateCache,
    TailOwnedKVCache,
    TensorOffsetVllmMetalPagedKVCache,
    VllmMetalPagedKVCache,
    configure_owned_recurrent_state_cache,
    configure_tail_owned_attention_kv_cache,
    detach_array_leaf,
    detach_attention_cache_state,
    detach_cache_state,
    detach_recurrent_cache_state,
    install_block_owned_attention_kv_cache,
    install_owned_recurrent_state_cache,
    install_tail_owned_attention_kv_cache,
    install_vllm_metal_paged_attention_kv_cache,
    owned_recurrent_state_stats,
    rollback_after_verify,
    restore_cache,
    snapshot_cache,
    snapshot_untrimmable_cache,
    tail_owned_attention_kv_stats,
)


class DummyCache:
    def __init__(self):
        self.state = [mx.array([1, 2, 3])]
        self.meta_state = ("meta", "3")


def test_restore_cache_rewinds_mutated_array_state():
    cache = [DummyCache()]
    snap = snapshot_cache(cache)
    cache[0].state = [mx.array([9])]
    cache[0].meta_state = ("meta", "1")

    restore_cache(cache, snap)

    assert cache[0].meta_state == ("meta", "3")
    assert cache[0].state[0].tolist() == [1, 2, 3]


def test_restore_cache_preserves_list_state_identity():
    cache = [DummyCache()]
    original_state = cache[0].state
    snap = snapshot_cache(cache)
    cache[0].state[0] = mx.array([9])

    restore_cache(cache, snap)

    assert cache[0].state is original_state
    assert cache[0].state[0].tolist() == [1, 2, 3]


def test_snapshot_cache_does_not_alias_later_mlx_array_mutation():
    from mlx_lm.models.cache import KVCache

    kv = KVCache()
    keys = mx.array([[[[1.0], [2.0]]]])
    values = mx.array([[[[3.0], [4.0]]]])
    kv.update_and_fetch(keys, values)
    snap = snapshot_cache([kv])

    kv.update_and_fetch(mx.array([[[[9.0]]]]), mx.array([[[[10.0]]]]))
    restore_cache([kv], snap)

    assert kv.offset == 2
    assert kv.keys.tolist() == [[[[1.0], [2.0]]]]
    assert kv.values.tolist() == [[[[3.0], [4.0]]]]


class TrimmableDummyCache:
    def __init__(self):
        self.trimmed = 0

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.trimmed += n

    @property
    def state(self):
        return [mx.array([5])]

    @state.setter
    def state(self, value):
        raise AssertionError("trimmable cache should not be restored by state assignment")

    @property
    def meta_state(self):
        return ""


def test_rollback_after_verify_trims_kv_and_restores_recurrent_state():
    recurrent = DummyCache()
    kv = TrimmableDummyCache()
    cache = [recurrent, kv]
    snap = snapshot_untrimmable_cache(cache)

    recurrent.state = [mx.array([9])]
    recurrent.meta_state = ("meta", "advanced")
    rollback_after_verify(cache, snap, verified_tokens=3)

    assert recurrent.meta_state == ("meta", "3")
    assert recurrent.state[0].tolist() == [1, 2, 3]
    assert kv.trimmed == 3


def test_detach_recurrent_cache_state_replaces_requested_list_leaves():
    recurrent = DummyCache()
    recurrent.state = [
        mx.array([1, 2, 3]) + mx.zeros((), dtype=mx.int32),
        mx.array([4, 5, 6]) + mx.zeros((), dtype=mx.int32),
    ]
    original_state = recurrent.state
    original_conv = original_state[0]
    original_gdn = original_state[1]

    stats = detach_recurrent_cache_state(
        [recurrent],
        components={"gdn"},
        mode="contiguous_eval",
    )

    assert recurrent.state is original_state
    assert recurrent.state[0] is original_conv
    assert recurrent.state[1] is not original_gdn
    assert recurrent.state[1].tolist() == [4, 5, 6]
    assert stats["entries"] == 1
    assert stats["arrays"] == 1
    assert stats["bytes"] == recurrent.state[1].nbytes


def test_detach_recurrent_cache_state_skips_trimmable_entries():
    kv = TrimmableDummyCache()

    stats = detach_recurrent_cache_state(
        [kv],
        components={"gdn", "conv"},
        mode="contiguous_eval",
    )

    assert stats == {"entries": 0, "arrays": 0, "bytes": 0}


def test_owned_recurrent_state_cache_reuses_fixed_buffers():
    owned = OwnedRecurrentStateCache(size=2)
    first = mx.array([[1.0, 2.0]])
    second = mx.array([[3.0, 4.0]]) + mx.zeros((), dtype=mx.float32)

    owned.replace_state([None, first])
    first_buffer = owned[1]
    owned.replace_state([None, second])

    assert owned[1] is first_buffer
    assert owned[1].tolist() == [[3.0, 4.0]]
    assert owned.owner_allocations == 1
    assert owned.owner_inplace_updates == 1
    assert owned.owner_updates == 2


def test_owned_recurrent_state_cache_keeps_speculative_writes_out_of_owner_buffer():
    owned = OwnedRecurrentStateCache(size=2)
    owned.replace_state([None, mx.array([[1.0]])])
    owner_buffer = owned[1]

    owned[1] = mx.array([[9.0]])
    assert owned[1].tolist() == [[9.0]]
    assert owned.owner_updates == 1

    owned.replace_state([None, mx.array([[2.0]])])
    assert owned[1] is owner_buffer
    assert owned[1].tolist() == [[2.0]]
    assert owned.owner_updates == 2


def test_owned_recurrent_state_restore_uses_owner_buffers():
    owned = OwnedRecurrentStateCache(size=2)
    owned.replace_state([mx.array([[1.0]]), mx.array([[2.0]])])
    first_conv = owned[0]
    first_gdn = owned[1]
    snap = snapshot_cache([owned])

    owned.replace_state([mx.array([[9.0]]), mx.array([[10.0]])])
    restore_cache([owned], snap)

    assert owned[0] is first_conv
    assert owned[1] is first_gdn
    assert owned[0].tolist() == [[1.0]]
    assert owned[1].tolist() == [[2.0]]


def test_install_owned_recurrent_state_cache_replaces_arrays_cache_only():
    from mlx_lm.models.cache import ArraysCache, KVCache

    recurrent = ArraysCache(size=2)
    kv = KVCache()
    cache = [recurrent, kv]

    stats = install_owned_recurrent_state_cache(cache)

    assert stats["entries"] == 1
    assert stats["skipped"] == 1
    assert isinstance(cache[0], OwnedRecurrentStateCache)
    assert cache[1] is kv


def test_configure_owned_recurrent_state_cache_uses_environment(monkeypatch):
    from mlx_lm.models.cache import ArraysCache

    cache = [ArraysCache(size=2)]
    monkeypatch.setenv("MTPLX_OWNED_RECURRENT_STATE", "1")

    stats = configure_owned_recurrent_state_cache(cache)

    assert stats["enabled"] == 1
    assert stats["entries"] == 1
    assert isinstance(cache[0], OwnedRecurrentStateCache)


def test_owned_recurrent_state_stats_aggregates_entries():
    cache = [OwnedRecurrentStateCache(size=2)]
    cache[0].replace_state([mx.array([[1.0]]), mx.array([[2.0]])])

    stats = owned_recurrent_state_stats(cache)

    assert stats["enabled"] == 1
    assert stats["entries"] == 1
    assert stats["updates"] == 2
    assert stats["arrays"] == 2
    assert stats["allocations"] == 2


class AttentionDummyCache:
    def __init__(self):
        self.keys = mx.array([[[[1.0], [2.0]]]])
        self.values = mx.array([[[[3.0], [4.0]]]])

    def is_trimmable(self):
        return True


def test_detach_attention_cache_state_eval_only_accounts_kv_arrays():
    kv = AttentionDummyCache()

    stats = detach_attention_cache_state([kv], mode="eval_only")

    assert stats["entries"] == 1
    assert stats["arrays"] == 2
    assert stats["bytes"] == kv.keys.nbytes + kv.values.nbytes
    assert kv.keys.tolist() == [[[[1.0], [2.0]]]]


def test_detach_array_leaf_supports_metal_copy_leaf_mode():
    value = mx.array([1.0, 2.0, 3.0]) + mx.zeros((), dtype=mx.float32)

    detached = detach_array_leaf(value, mode="metal_copy_leaf")

    assert detached.tolist() == [1.0, 2.0, 3.0]


def test_detach_cache_state_combines_recurrent_and_attention_groups():
    recurrent = DummyCache()
    recurrent.state = [mx.array([1]), mx.array([2])]
    kv = AttentionDummyCache()

    stats = detach_cache_state(
        [recurrent, kv],
        components={"gdn", "attn"},
        mode="eval_only",
    )

    assert stats["entries"] == 2
    assert stats["arrays"] == 3


def test_tail_owned_kv_cache_matches_stock_kv_cache_updates():
    from mlx_lm.models.cache import KVCache

    stock = KVCache()
    owned = TailOwnedKVCache(mode="contiguous_eval")
    first_k = mx.arange(4, dtype=mx.float32).reshape(1, 1, 4, 1)
    first_v = 10 + first_k
    next_k = 100 + mx.arange(2, dtype=mx.float32).reshape(1, 1, 2, 1)
    next_v = 200 + mx.arange(2, dtype=mx.float32).reshape(1, 1, 2, 1)

    stock_k, stock_v = stock.update_and_fetch(first_k, first_v)
    owned_k, owned_v = owned.update_and_fetch(first_k, first_v)
    stock_k, stock_v = stock.update_and_fetch(next_k, next_v)
    owned_k, owned_v = owned.update_and_fetch(next_k, next_v)
    mx.eval(stock_k, stock_v, owned_k, owned_v)

    assert owned.size() == stock.size() == 6
    assert owned_k.tolist() == stock_k.tolist()
    assert owned_v.tolist() == stock_v.tolist()
    assert owned.tail_owner_updates == 2
    assert owned.tail_owner_arrays == 4
    assert owned.tail_owner_bytes == (
        first_k.nbytes + first_v.nbytes + next_k.nbytes + next_v.nbytes
    )


def test_install_tail_owned_attention_kv_cache_replaces_stock_kv_only():
    from mlx_lm.models.cache import KVCache

    recurrent = DummyCache()
    kv = KVCache()
    cache = [recurrent, kv]

    stats = install_tail_owned_attention_kv_cache(cache, mode="contiguous_eval")

    assert stats["entries"] == 1
    assert stats["skipped"] == 1
    assert cache[0] is recurrent
    assert isinstance(cache[1], TailOwnedKVCache)


def test_configure_tail_owned_attention_kv_cache_uses_environment(monkeypatch):
    from mlx_lm.models.cache import KVCache

    cache = [KVCache()]
    monkeypatch.setenv("MTPLX_OWNED_ATTN_KV", "tail")
    monkeypatch.setenv("MTPLX_OWNED_ATTN_KV_MODE", "eval_only")

    stats = configure_tail_owned_attention_kv_cache(cache)

    assert stats["enabled"] == 1
    assert stats["entries"] == 1
    assert isinstance(cache[0], TailOwnedKVCache)
    assert cache[0].mode == "eval_only"


def test_tail_owned_attention_kv_stats_aggregates_entries():
    cache = [TailOwnedKVCache(mode="eval_only")]
    cache[0].update_and_fetch(
        mx.ones((1, 1, 1, 1)),
        2 * mx.ones((1, 1, 1, 1)),
    )

    stats = tail_owned_attention_kv_stats(cache)

    assert stats["enabled"] == 1
    assert stats["entries"] == 1
    assert stats["mode"] == "eval_only"
    assert stats["updates"] == 1
    assert stats["arrays"] == 2


def test_block_owned_kv_cache_matches_stock_across_block_boundary_and_trim():
    from mlx_lm.models.cache import KVCache

    stock = KVCache()
    block = BlockOwnedKVCache(mode="contiguous_eval", block_size=3)
    chunks = [
        (
            mx.arange(2, dtype=mx.float32).reshape(1, 1, 2, 1),
            10 + mx.arange(2, dtype=mx.float32).reshape(1, 1, 2, 1),
        ),
        (
            100 + mx.arange(3, dtype=mx.float32).reshape(1, 1, 3, 1),
            200 + mx.arange(3, dtype=mx.float32).reshape(1, 1, 3, 1),
        ),
    ]

    for keys, values in chunks:
        stock_k, stock_v = stock.update_and_fetch(keys, values)
        block_k, block_v = block.update_and_fetch(keys, values)
    mx.eval(stock_k, stock_v, block_k, block_v)

    assert block.size() == stock.size() == 5
    assert block_k.tolist() == stock_k.tolist()
    assert block_v.tolist() == stock_v.tolist()
    assert len(block.key_blocks) == 2

    stock.trim(2)
    block.trim(2)
    keys = 300 + mx.ones((1, 1, 1, 1))
    values = 400 + mx.ones((1, 1, 1, 1))
    stock_k, stock_v = stock.update_and_fetch(keys, values)
    block_k, block_v = block.update_and_fetch(keys, values)
    mx.eval(stock_k, stock_v, block_k, block_v)

    assert block.size() == stock.size() == 4
    assert block_k.tolist() == stock_k.tolist()
    assert block_v.tolist() == stock_v.tolist()


def test_install_block_owned_attention_kv_cache_replaces_stock_kv_only():
    from mlx_lm.models.cache import KVCache

    recurrent = DummyCache()
    kv = KVCache()
    cache = [recurrent, kv]

    stats = install_block_owned_attention_kv_cache(
        cache,
        mode="contiguous_eval",
        block_size=512,
    )

    assert stats["entries"] == 1
    assert stats["skipped"] == 1
    assert stats["block_size"] == 512
    assert cache[0] is recurrent
    assert isinstance(cache[1], BlockOwnedKVCache)
    assert cache[1].block_size == 512


def test_vllm_metal_paged_kv_cache_matches_stock_kv_cache_updates_and_trim():
    from mlx_lm.models.cache import KVCache

    stock = KVCache()
    paged = VllmMetalPagedKVCache(block_size=4, num_blocks=4)
    chunks = [
        (
            mx.arange(3, dtype=mx.float32).reshape(1, 1, 3, 1),
            10 + mx.arange(3, dtype=mx.float32).reshape(1, 1, 3, 1),
        ),
        (
            100 + mx.arange(5, dtype=mx.float32).reshape(1, 1, 5, 1),
            200 + mx.arange(5, dtype=mx.float32).reshape(1, 1, 5, 1),
        ),
    ]

    for keys, values in chunks:
        stock_k, stock_v = stock.update_and_fetch(keys, values)
        paged_k, paged_v = paged.update_and_fetch(keys, values)
    mx.eval(stock_k, stock_v, paged_k, paged_v)

    assert paged.size() == stock.size() == 8
    assert paged_k.tolist() == stock_k.tolist()
    assert paged_v.tolist() == stock_v.tolist()
    assert paged.paged_stats()["updates"] == 2
    assert paged.paged_stats()["capacity"] == 16

    stock.trim(3)
    paged.trim(3)
    keys = 300 + mx.ones((1, 1, 2, 1))
    values = 400 + mx.ones((1, 1, 2, 1))
    stock_k, stock_v = stock.update_and_fetch(keys, values)
    paged_k, paged_v = paged.update_and_fetch(keys, values)
    mx.eval(stock_k, stock_v, paged_k, paged_v)

    assert paged.size() == stock.size() == 7
    assert paged_k.tolist() == stock_k.tolist()
    assert paged_v.tolist() == stock_v.tolist()


def test_install_vllm_metal_paged_attention_kv_cache_replaces_stock_kv_only(monkeypatch):
    from mlx_lm.models.cache import KVCache

    monkeypatch.setattr("mtplx.cache_state._load_vllm_metal_ops", lambda: object())
    recurrent = DummyCache()
    kv = KVCache()
    cache = [recurrent, kv]

    stats = install_vllm_metal_paged_attention_kv_cache(
        cache,
        block_size=16,
        num_blocks=64,
    )

    assert stats["entries"] == 1
    assert stats["skipped"] == 1
    assert stats["block_size"] == 16
    assert stats["num_blocks"] == 64
    assert cache[0] is recurrent
    assert isinstance(cache[1], VllmMetalPagedKVCache)


def test_configure_tail_owned_attention_kv_cache_uses_vllm_metal_env(monkeypatch):
    from mlx_lm.models.cache import KVCache

    monkeypatch.setattr("mtplx.cache_state._load_vllm_metal_ops", lambda: object())
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE", "16")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS", "32")
    cache = [KVCache()]

    stats = configure_tail_owned_attention_kv_cache(cache)

    assert stats["mode"] == "vllm_metal_paged"
    assert stats["external_ops_required"] == 1
    assert stats["entries"] == 1
    assert isinstance(cache[0], VllmMetalPagedKVCache)
    assert cache[0].block_size == 16
    assert cache[0].num_blocks == 32


def test_dynamic_paged_kv_sizes_capacity_from_request(monkeypatch):
    from mlx_lm.models.cache import KVCache

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "mlx_vector_paged")
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV_TOKENS", str(32768 + 128 + 3))
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV_MARGIN", "128")
    cache = [KVCache()]

    stats = configure_tail_owned_attention_kv_cache(cache)

    expected_blocks = ((32768 + 128 + 3 + 128) + 16 - 1) // 16
    assert stats["num_blocks"] >= expected_blocks
    assert isinstance(cache[0], VllmMetalPagedKVCache)
    assert cache[0].capacity >= 32768 + 128 + 3 + 128


def test_paged_kv_grows_on_dynamic_overflow(monkeypatch):
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    paged = VllmMetalPagedKVCache(block_size=4, num_blocks=1)
    keys = mx.zeros((1, 2, 6, 3), dtype=mx.float32)
    values = mx.zeros((1, 2, 6, 3), dtype=mx.float32)

    paged.update_without_fetch(keys, values)

    assert paged.capacity >= 6
    assert paged.paged_stats()["grow_events"] == 1


def test_paged_active_array_assertion_guards_dense_fallback(monkeypatch):
    monkeypatch.setenv("MTPLX_ASSERT_NO_PAGED_ACTIVE_ARRAYS", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD", "4")
    paged = VllmMetalPagedKVCache(block_size=4, num_blocks=4)
    keys = mx.zeros((1, 2, 4, 3), dtype=mx.float32)
    values = mx.zeros((1, 2, 4, 3), dtype=mx.float32)
    paged.update_without_fetch(keys, values)

    with pytest.raises(RuntimeError, match="materialize active K/V arrays"):
        _ = paged.state


def test_paged_attention_records_phase_aware_large_q_bailout(monkeypatch):
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "mlx_vector_paged")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "1")

    paged = VllmMetalPagedKVCache(block_size=4, num_blocks=16)
    keys = mx.zeros((1, 2, 32, 8), dtype=mx.float32)
    values = mx.zeros((1, 2, 32, 8), dtype=mx.float32)
    queries = mx.zeros((1, 8, 4, 8), dtype=mx.float32)
    paged.update_without_fetch(keys, values)

    with attention_phase("prefill"):
        assert paged.paged_attention(queries, scale=8**-0.5, mask="causal") is None
        paged.record_dense_fallback()

    stats = paged.paged_stats()
    assert stats["prefill_dense_fallback_calls"] == 1
    assert stats["paged_attention_bailouts_by_phase_reason"] == {
        "prefill:q_len_gt_max": 1
    }


def test_mlx_vector_large_q_routes_to_partitioned_paged(monkeypatch):
    class FakeOps:
        def __init__(self):
            self.calls = 0

        def paged_attention_v2_online_partitioned(self, *args, **kwargs):
            self.calls += 1

    fake_ops = FakeOps()
    monkeypatch.setattr("mtplx.cache_state._load_vllm_metal_ops", lambda: fake_ops)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "mlx_vector_paged")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE", "8")

    paged = VllmMetalPagedKVCache(block_size=4, num_blocks=16)
    keys = mx.zeros((1, 2, 32, 8), dtype=mx.float32)
    values = mx.zeros((1, 2, 32, 8), dtype=mx.float32)
    queries = mx.zeros((1, 8, 4, 8), dtype=mx.float32)
    paged.update_without_fetch(keys, values)

    with attention_phase("prefill"):
        actual = paged.paged_attention(queries, scale=8**-0.5, mask="causal")

    assert actual is not None
    assert fake_ops.calls == 1
    stats = paged.paged_stats()
    assert stats["partitioned_paged_calls"] == 1
    assert stats["paged_attention_large_q_path"] == "partitioned_paged"
    assert stats["dense_fallback_calls"] == 0


def test_large_q_split_fallback_stays_in_paged_storage(monkeypatch):
    from mlx_lm.models.base import scaled_dot_product_attention

    def missing_ops():
        raise RuntimeError("no external ops")

    monkeypatch.setattr("mtplx.cache_state._load_vllm_metal_ops", missing_ops)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "mlx_vector_paged")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_LARGE_Q_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_LARGE_Q_KV_CHUNK_SIZE", "7")

    mx.random.seed(1357)
    paged = VllmMetalPagedKVCache(block_size=4, num_blocks=16)
    queries = mx.random.normal((1, 8, 4, 8), dtype=mx.float32)
    keys = mx.random.normal((1, 2, 32, 8), dtype=mx.float32)
    values = mx.random.normal((1, 2, 32, 8), dtype=mx.float32)
    paged.update_without_fetch(keys, values)

    expected = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=8**-0.5,
        mask="causal",
    )
    actual = paged.paged_attention(queries, scale=8**-0.5, mask="causal")
    mx.eval(expected, actual)

    assert actual is not None
    diff = mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))
    mx.eval(diff)
    assert float(diff.item()) <= 1e-3
    stats = paged.paged_stats()
    assert stats["large_q_split_sdpa_fallback_calls"] == 1
    assert stats["active_array_calls"] == 0
    assert stats["dense_fallback_calls"] == 0


def test_long_context_dense_fallback_guard_accepts_new_override(monkeypatch):
    monkeypatch.setenv("MTPLX_ASSERT_NO_PAGED_ACTIVE_ARRAYS", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD", "4")
    monkeypatch.setenv("MTPLX_ALLOW_LONG_CONTEXT_DENSE_FALLBACK", "1")
    paged = VllmMetalPagedKVCache(block_size=4, num_blocks=4)
    keys = mx.zeros((1, 2, 4, 3), dtype=mx.float32)
    values = mx.zeros((1, 2, 4, 3), dtype=mx.float32)
    paged.update_without_fetch(keys, values)

    assert paged.state[0] is not None


def test_configure_vllm_metal_paged_cache_mlx_vector_is_packaged(monkeypatch):
    from mlx_lm.models.cache import KVCache

    def fail_if_external_ops_loads():
        raise AssertionError("mlx_vector_paged should not require vllm-metal checkout")

    monkeypatch.setattr("mtplx.cache_state._load_vllm_metal_ops", fail_if_external_ops_loads)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "mlx_vector_paged")
    cache = [KVCache()]

    stats = configure_tail_owned_attention_kv_cache(cache)

    assert stats["mode"] == "vllm_metal_paged"
    assert stats["attention_impl"] == "mlx_vector_paged"
    assert stats["external_ops_required"] == 0
    assert stats["entries"] == 1
    assert isinstance(cache[0], VllmMetalPagedKVCache)


def test_configure_vllm_metal_paged_cache_can_enable_turboquant(monkeypatch):
    from mlx_lm.models.cache import KVCache

    monkeypatch.setattr("mtplx.cache_state._load_vllm_metal_ops", lambda: object())
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_TURBOQUANT", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_TURBOQUANT_K_QUANT", "q8_0")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_TURBOQUANT_V_QUANT", "q3_0")
    cache = [KVCache()]

    stats = configure_tail_owned_attention_kv_cache(cache)

    assert stats["mode"] == "vllm_metal_paged_turboquant"
    assert stats["external_ops_required"] == 1
    assert stats["turboquant"] == 1
    assert stats["turboquant_k_quant"] == "q8_0"
    assert stats["turboquant_v_quant"] == "q3_0"
    assert isinstance(cache[0], VllmMetalPagedKVCache)
    assert cache[0].turboquant is True
    assert cache[0].turboquant_config.key_quant == "q8_0"
    assert cache[0].turboquant_config.value_quant == "q3_0"


def test_vllm_metal_paged_attention_matches_stock_attention_with_tolerance():
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    repo = Path(__file__).resolve().parents[1] / "REFERENCES:TOOLS" / "vllm-metal"
    if not repo.exists():
        pytest.skip("vllm-metal reference checkout is unavailable")

    from mlx_lm.models.base import scaled_dot_product_attention

    mx.random.seed(1234)
    q_len = 4
    kv_len = 21
    dim = 128
    queries = mx.random.normal((1, 8, q_len, dim), dtype=mx.float16)
    keys = mx.random.normal((1, 2, kv_len, dim), dtype=mx.float16)
    values = mx.random.normal((1, 2, kv_len, dim), dtype=mx.float16)
    scale = dim**-0.5
    cache = VllmMetalPagedKVCache(block_size=16, num_blocks=4)
    cache.update_without_fetch(keys, values)

    expected = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=scale,
        mask="causal",
    )
    actual = cache.paged_attention(queries, scale=scale)
    assert actual is not None
    mx.eval(expected, actual)

    diff = mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))
    mx.eval(diff)
    assert float(diff.item()) <= 2e-2


def test_vllm_metal_partitioned_paged_attention_matches_stock_attention(monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    repo = Path(__file__).resolve().parents[1] / "REFERENCES:TOOLS" / "vllm-metal"
    if not repo.exists():
        pytest.skip("vllm-metal reference checkout is unavailable")

    from mlx_lm.models.base import scaled_dot_product_attention

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD", "0")

    mx.random.seed(4321)
    q_len = 4
    kv_len = 640
    dim = 128
    queries = mx.random.normal((1, 8, q_len, dim), dtype=mx.float16)
    keys = mx.random.normal((1, 2, kv_len, dim), dtype=mx.float16)
    values = mx.random.normal((1, 2, kv_len, dim), dtype=mx.float16)
    scale = dim**-0.5
    cache = VllmMetalPagedKVCache(block_size=16, num_blocks=64)
    cache.update_without_fetch(keys, values)

    expected = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=scale,
        mask="causal",
    )
    actual = cache.paged_attention(queries, scale=scale)
    assert actual is not None
    mx.eval(expected, actual)

    diff = mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))
    mx.eval(diff)
    assert float(diff.item()) <= 2e-2
    assert cache.paged_stats()["partitioned_attention_calls"] == 1


def test_vllm_metal_paged_attention_exact_gather_matches_stock_attention(monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")

    from mlx_lm.models.base import scaled_dot_product_attention

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "fast_sdpa_gather")

    mx.random.seed(2468)
    q_len = 4
    kv_len = 77
    dim = 128
    queries = mx.random.normal((1, 8, q_len, dim), dtype=mx.float16)
    keys = mx.random.normal((1, 2, kv_len, dim), dtype=mx.float16)
    values = mx.random.normal((1, 2, kv_len, dim), dtype=mx.float16)
    scale = dim**-0.5
    cache = VllmMetalPagedKVCache(block_size=16, num_blocks=8)
    cache.update_without_fetch(keys, values)

    expected = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=scale,
        mask="causal",
    )
    actual = cache.paged_attention(queries, scale=scale, mask="causal")
    assert actual is not None
    mx.eval(expected, actual)

    diff = mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))
    mx.eval(diff)
    assert float(diff.item()) == 0.0


def test_vllm_metal_paged_attention_mlx_vector_paged_matches_stock_attention(monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")

    from mlx_lm.models.base import scaled_dot_product_attention

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "mlx_vector_paged")

    mx.random.seed(9753)
    q_len = 4
    kv_len = 2048
    dim = 128
    queries = mx.random.normal((1, 8, q_len, dim), dtype=mx.bfloat16)
    keys = mx.random.normal((1, 2, kv_len, dim), dtype=mx.bfloat16)
    values = mx.random.normal((1, 2, kv_len, dim), dtype=mx.bfloat16)
    scale = dim**-0.5
    cache = VllmMetalPagedKVCache(block_size=16, num_blocks=128)
    cache.update_without_fetch(keys, values)

    expected = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=scale,
        mask="causal",
    )
    actual = cache.paged_attention(queries, scale=scale, mask="causal")
    assert actual is not None
    mx.eval(expected, actual)

    diff = mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))
    mx.eval(diff)
    assert float(diff.item()) <= 3e-2


def test_vllm_metal_paged_packaged_impl_decline_does_not_load_external_ops(monkeypatch):
    import mtplx.cache_state as cache_state
    import mtplx.kernels.sdpa_2pass_paged as paged_kernel

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "mlx_vector_paged")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "1")
    monkeypatch.setattr(paged_kernel, "sdpa_2pass_paged_tail", lambda **_kwargs: None)

    def fail_external_ops():
        raise AssertionError("packaged paged attention must not load external ops")

    monkeypatch.setattr(cache_state, "_load_vllm_metal_ops", fail_external_ops)

    q_len = 4
    kv_len = 32
    dim = 16
    queries = mx.zeros((1, 8, q_len, dim), dtype=mx.float32)
    keys = mx.zeros((1, 2, kv_len, dim), dtype=mx.float32)
    values = mx.zeros((1, 2, kv_len, dim), dtype=mx.float32)
    cache = VllmMetalPagedKVCache(block_size=16, num_blocks=4)
    cache.update_without_fetch(keys, values)

    assert cache.paged_attention(queries, scale=dim**-0.5, mask="causal") is None


def test_tensor_offset_vllm_metal_paged_attention_matches_stock_attention(monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")

    from mlx_lm.models.base import scaled_dot_product_attention

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q", "8")
    monkeypatch.setenv("MTPLX_GRAPHBANK_PAGED_STATIC_MAX_OFFSET", "32")

    mx.random.seed(8642)
    q_len = 4
    kv_len = 77
    dim = 128
    queries = mx.random.normal((1, 8, q_len, dim), dtype=mx.bfloat16)
    keys = mx.random.normal((1, 2, kv_len, dim), dtype=mx.bfloat16)
    values = mx.random.normal((1, 2, kv_len, dim), dtype=mx.bfloat16)
    scale = dim**-0.5
    paged = VllmMetalPagedKVCache(block_size=16, num_blocks=8)
    paged.update_without_fetch(keys, values)
    cache = TensorOffsetVllmMetalPagedKVCache.from_paged_cache(paged)
    assert cache.paged_stats()["static_max_offset"] == 32

    expected = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=scale,
        mask="causal",
    )
    actual = cache.paged_attention(queries, scale=scale, mask="causal")
    assert actual is not None
    mx.eval(expected, actual)

    diff = mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))
    mx.eval(diff)
    assert float(diff.item()) <= 3e-2


def test_tensor_offset_vllm_metal_paged_cache_updates_offset_inside_compile():
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")

    paged = VllmMetalPagedKVCache(block_size=4, num_blocks=4)
    paged.update_without_fetch(
        mx.ones((1, 1, 2, 1), dtype=mx.float32),
        2 * mx.ones((1, 1, 2, 1), dtype=mx.float32),
    )
    cache = TensorOffsetVllmMetalPagedKVCache.from_paged_cache(paged)

    def update(keys, values):
        cache.update_without_fetch(keys, values)
        return cache.compile_state

    compiled = mx.compile(update, inputs=cache.compile_state, outputs=cache.compile_state)
    compiled(
        3 * mx.ones((1, 1, 2, 1), dtype=mx.float32),
        4 * mx.ones((1, 1, 2, 1), dtype=mx.float32),
    )
    mx.eval(cache.compile_state)

    assert cache.size() == 4
    keys, values = cache.state
    mx.eval(keys, values)
    assert keys[0, 0, :4, 0].tolist() == [1.0, 1.0, 3.0, 3.0]
    assert values[0, 0, :4, 0].tolist() == [2.0, 2.0, 4.0, 4.0]
