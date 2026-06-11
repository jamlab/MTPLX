"""Warm-prefix state reuse for MTPLX target prefill.

SessionBank is deliberately conservative in this first version: it stores
exact token-prefix entries in memory, restores cloned cache state into a fresh
runtime cache, then forwards only the suffix tokens. The benchmark gate compares
the warm result against a cold full prefill before any generation path uses it.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from .cache_state import CacheSnapshot, _clone_tree, restore_cache, snapshot_cache
from .runtime import MTPLXRuntime

GIB = 1024**3
DEFAULT_MAX_ENTRIES = 8
DEFAULT_MAX_BYTES = 24 * GIB
DEFAULT_PER_SESSION_MAX_BYTES = 8 * GIB
DEFAULT_IDLE_TTL_S = 60 * 60
DEFAULT_PREFIX_BLOCK_SIZE = 256
DEFAULT_BLOCK_PREFIX_MIN_MATCH_TOKENS = 512


class CacheMissReason(str, Enum):
    NEW_SESSION = "new_session"
    PREFIX_DIVERGENCE_AT_TOKEN = "prefix_divergence_at_token"
    MODEL_MISMATCH = "model_mismatch"
    TEMPLATE_MISMATCH = "template_mismatch"
    POLICY_MISMATCH = "policy_mismatch"
    EVICTED = "evicted"
    BACKGROUND_BYPASS = "background_bypass"
    SESSION_BUSY = "session_busy"
    SNAPSHOT_DESYNC = "snapshot_desync"
    NO_SNAPSHOT_COVERAGE = "no_snapshot_coverage"


def token_prefix_hash(token_ids: list[int] | tuple[int, ...]) -> str:
    h = hashlib.sha256()
    for token in token_ids:
        h.update(int(token).to_bytes(8, byteorder="little", signed=True))
    return h.hexdigest()


def common_prefix_len(left: list[int] | tuple[int, ...], right: list[int] | tuple[int, ...]) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if int(left[index]) != int(right[index]):
            return index
    return limit


def block_aligned_prefix_len(matched_tokens: int, *, block_size: int) -> int:
    block = max(1, int(block_size))
    matched = max(0, int(matched_tokens))
    return (matched // block) * block


# Policies that share the committed-mtp-cache representation. An entry stored
# under any of these policies can be safely reused for a lookup that requests
# any other policy in this set, because the cache snapshot shape is identical
# (``last_window`` is just a runtime trim of the same committed cache).
_COMMITTED_CACHE_POLICIES = frozenset({"committed", "last_window"})


def _mtp_history_policy_compatible(
    entry_policy: str | None, lookup_policy: str | None
) -> bool:
    """Return True if a bank entry stored under ``entry_policy`` may be reused
    for a lookup that resolved to ``lookup_policy``.

    Equality is always compatible. Beyond that, ``committed`` and
    ``last_window`` are treated as interchangeable because both rely on the
    same committed mtp-history cache shape; the only difference between them
    is a runtime trim that is applied during prefill, which is moot once the
    cache is being restored from a stored snapshot.
    """
    if entry_policy == lookup_policy:
        return True
    if entry_policy is None or lookup_policy is None:
        return False
    return (
        entry_policy in _COMMITTED_CACHE_POLICIES
        and lookup_policy in _COMMITTED_CACHE_POLICIES
    )


def _tree_nbytes(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, CacheSnapshot):
        return _tree_nbytes(value.states) + _tree_nbytes(value.meta_states)
    if isinstance(value, mx.array):
        return int(value.nbytes)
    if isinstance(value, (list, tuple)):
        return sum(_tree_nbytes(item) for item in value)
    if isinstance(value, dict):
        return sum(_tree_nbytes(item) for item in value.values())
    return 0


def _snapshot_nbytes(snapshot: CacheSnapshot) -> int:
    return _tree_nbytes(snapshot.states) + _tree_nbytes(snapshot.meta_states)


@dataclass
class SessionBankEntry:
    token_ids: tuple[int, ...]
    token_hash: str
    model_path: str
    mtp_enabled: bool
    hidden_variant: str | None
    cache_snapshot: CacheSnapshot
    logits: Any
    hidden: Any | None
    cache_ref: list[Any] | None = None
    mtp_history_cache_ref: list[Any] | None = None
    live_ref_only: bool = False
    created_at_s: float = field(default_factory=time.time)
    last_access_s: float = field(default_factory=time.time)
    hits: int = 0
    nbytes: int = 0
    session_id: str | None = None
    template_hash: str | None = None
    mtp_history_policy: str | None = None
    draft_head_identity: str | None = None
    policy_fingerprint: str | None = None
    mtp_history_snapshot: Any | None = None
    snapshot_epoch: int = 0
    mtp_snapshot_epoch: int | None = None
    eviction_reason: str | None = None
    extra_state: dict[str, Any] | None = None

    @property
    def prefix_len(self) -> int:
        return len(self.token_ids)


def _empty_cache_snapshot(cache: list[Any] | None) -> CacheSnapshot:
    size = len(cache or [])
    return CacheSnapshot(states=tuple(None for _ in range(size)), meta_states=tuple(None for _ in range(size)))


def _trim_cache_ref_to_prefix(cache: list[Any] | None, prefix_len: int) -> bool:
    if cache is None:
        return False
    target_offset = max(0, int(prefix_len) - 1)
    for entry in cache:
        current = int(getattr(entry, "offset", target_offset) or 0)
        if current < target_offset:
            return False
        delta = current - target_offset
        if delta <= 0:
            continue
        trim = getattr(entry, "trim", None)
        if not callable(trim):
            return False
        if int(trim(delta)) != delta:
            return False
    return True


def _trim_cache_ref_by_tokens(cache: list[Any] | None, tokens: int) -> bool:
    if cache is None:
        return False
    delta = max(0, int(tokens))
    if delta <= 0:
        return True
    for entry in cache:
        trim = getattr(entry, "trim", None)
        if not callable(trim):
            return False
        if int(trim(delta)) != delta:
            return False
    return True


@dataclass
class SessionBankRestore:
    entry: SessionBankEntry
    cache: list[Any]
    logits: Any
    hidden: Any | None
    restored_nbytes: int
    restore_mode: str = "clone"
    cache_miss_reason: str | None = None
    mtp_history_snapshot: Any | None = None
    mtp_history_cache: list[Any] | None = None
    cache_source: str = "ram"
    ssd_cache_hit: bool = False
    ssd_cached_tokens: int = 0
    ssd_restore_s: float = 0.0
    extra_state: dict[str, Any] | None = None


class SessionBank:
    """In-memory exact prefix table for warm target prefill."""

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_bytes: int = DEFAULT_MAX_BYTES,
        per_session_max_bytes: int = DEFAULT_PER_SESSION_MAX_BYTES,
        idle_ttl_s: float = DEFAULT_IDLE_TTL_S,
        cold_tier: Any | None = None,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")
        if per_session_max_bytes < 1:
            raise ValueError("per_session_max_bytes must be >= 1")
        if idle_ttl_s <= 0:
            raise ValueError("idle_ttl_s must be > 0")
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self.per_session_max_bytes = int(per_session_max_bytes)
        self.idle_ttl_s = float(idle_ttl_s)
        self._entries: dict[tuple[int, ...], SessionBankEntry] = {}
        self.last_miss_reason: str | None = None
        self.last_put_nbytes: int = 0
        self.last_put_skipped_oversized_snapshot: bool = False
        self.eviction_log: list[dict[str, Any]] = []
        self.cold_tier = cold_tier
        self.last_restore_source: str | None = None
        self.last_ssd_restore_s: float = 0.0
        self.last_prefix_diagnostic: dict[str, Any] | None = None

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def total_nbytes(self) -> int:
        return sum(entry.nbytes for entry in self._entries.values())

    def put(
        self,
        *,
        runtime: MTPLXRuntime,
        token_ids: list[int] | tuple[int, ...],
        cache: list[Any],
        logits: Any,
        hidden: Any | None,
        hidden_variant: str | None = None,
        keep_live_ref: bool = False,
        session_id: str | None = None,
        template_hash: str | None = None,
        mtp_history_policy: str | None = None,
        draft_head_identity: str | None = None,
        policy_fingerprint: str | None = None,
        mtp_history_snapshot: Any | None = None,
        mtp_history_cache_ref: list[Any] | None = None,
        snapshot_epoch: int = 0,
        mtp_snapshot_epoch: int | None = None,
        nbytes_override: int | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> SessionBankEntry | None:
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            raise ValueError("cannot store an empty prefix")
        if mtp_snapshot_epoch is not None and int(mtp_snapshot_epoch) != int(snapshot_epoch):
            raise ValueError("trunk and MTP snapshots must share the same commit boundary")
        self.last_put_nbytes = 0
        self.last_put_skipped_oversized_snapshot = False
        def live_ref_entry(reason: str, nbytes: int) -> SessionBankEntry | None:
            if not keep_live_ref or not cache:
                return None
            entry = SessionBankEntry(
                token_ids=tokens,
                token_hash=token_prefix_hash(tokens),
                model_path=str(runtime.model_path),
                mtp_enabled=bool(runtime.mtp_enabled),
                hidden_variant=hidden_variant,
                cache_snapshot=_empty_cache_snapshot(cache),
                logits=_clone_tree(logits),
                hidden=_clone_tree(hidden),
                cache_ref=cache,
                mtp_history_cache_ref=mtp_history_cache_ref,
                live_ref_only=True,
                nbytes=0,
                session_id=session_id,
                template_hash=template_hash,
                mtp_history_policy=mtp_history_policy,
                draft_head_identity=draft_head_identity,
                policy_fingerprint=policy_fingerprint,
                mtp_history_snapshot=None,
                snapshot_epoch=int(snapshot_epoch),
                mtp_snapshot_epoch=(
                    int(mtp_snapshot_epoch)
                    if mtp_snapshot_epoch is not None
                    else (
                        int(snapshot_epoch)
                        if mtp_history_cache_ref is not None
                        else None
                    )
                ),
                extra_state=_clone_tree(extra_state),
            )
            self.eviction_log.append(
                {
                    "reason": reason,
                    "session_id": session_id,
                    "prefix_len": len(tokens),
                    "token_hash": entry.token_hash,
                    "nbytes": int(nbytes),
                    "budget": int(self.per_session_max_bytes),
                    "fallback": "live_reference_lease",
                }
            )
            self._entries[tokens] = entry
            self._evict_if_needed(protected_tokens=tokens)
            return entry

        if nbytes_override is not None and int(nbytes_override) > self.per_session_max_bytes:
            self.last_put_nbytes = int(nbytes_override)
            self.last_put_skipped_oversized_snapshot = True
            live_entry = live_ref_entry(
                "skipped_oversized_snapshot_live_ref",
                int(nbytes_override),
            )
            if live_entry is not None:
                return live_entry
            self.eviction_log.append(
                {
                    "reason": "skipped_oversized_snapshot",
                    "session_id": session_id,
                    "prefix_len": len(tokens),
                    "token_hash": token_prefix_hash(tokens),
                    "nbytes": int(nbytes_override),
                    "budget": int(self.per_session_max_bytes),
                }
            )
            return None
        try:
            snapshot = snapshot_cache(cache)
        except RuntimeError as exc:
            if "materialize active K/V arrays" not in str(exc):
                raise
            self.last_put_skipped_oversized_snapshot = True
            live_entry = live_ref_entry(
                "skipped_dense_materializing_snapshot_live_ref",
                0,
            )
            if live_entry is not None:
                return live_entry
            self.eviction_log.append(
                {
                    "reason": "skipped_dense_materializing_snapshot",
                    "session_id": session_id,
                    "prefix_len": len(tokens),
                    "token_hash": token_prefix_hash(tokens),
                    "nbytes": 0,
                    "budget": int(self.per_session_max_bytes),
                    "error": str(exc),
                }
            )
            return None
        computed_nbytes = (
            _snapshot_nbytes(snapshot)
            + _tree_nbytes(logits)
            + _tree_nbytes(hidden)
            + _tree_nbytes(mtp_history_snapshot)
        )
        entry_nbytes = int(nbytes_override if nbytes_override is not None else computed_nbytes)
        self.last_put_nbytes = int(entry_nbytes)
        if entry_nbytes > self.per_session_max_bytes:
            self.last_put_skipped_oversized_snapshot = True
            live_entry = live_ref_entry(
                "skipped_oversized_snapshot_live_ref",
                int(entry_nbytes),
            )
            if live_entry is not None:
                return live_entry
            self.eviction_log.append(
                {
                    "reason": "skipped_oversized_snapshot",
                    "session_id": session_id,
                    "prefix_len": len(tokens),
                    "token_hash": token_prefix_hash(tokens),
                    "nbytes": int(entry_nbytes),
                    "budget": int(self.per_session_max_bytes),
                }
            )
            return None
        entry = SessionBankEntry(
            token_ids=tokens,
            token_hash=token_prefix_hash(tokens),
            model_path=str(runtime.model_path),
            mtp_enabled=bool(runtime.mtp_enabled),
            hidden_variant=hidden_variant,
            cache_snapshot=snapshot,
            logits=_clone_tree(logits),
            hidden=_clone_tree(hidden),
            cache_ref=cache if keep_live_ref else None,
            mtp_history_cache_ref=mtp_history_cache_ref if keep_live_ref else None,
            nbytes=int(entry_nbytes),
            session_id=session_id,
            template_hash=template_hash,
            mtp_history_policy=mtp_history_policy,
            draft_head_identity=draft_head_identity,
            policy_fingerprint=policy_fingerprint,
            mtp_history_snapshot=_clone_tree(mtp_history_snapshot),
            snapshot_epoch=int(snapshot_epoch),
            mtp_snapshot_epoch=(
                int(mtp_snapshot_epoch)
                if mtp_snapshot_epoch is not None
                else (int(snapshot_epoch) if mtp_history_snapshot is not None else None)
            ),
            extra_state=_clone_tree(extra_state),
        )
        self._enqueue_cold_entry(entry)
        self._entries[tokens] = entry
        self._evict_if_needed(protected_tokens=tokens)
        return entry

    def put_snapshot(
        self,
        *,
        runtime: MTPLXRuntime,
        token_ids: list[int] | tuple[int, ...],
        cache_snapshot: CacheSnapshot,
        logits: Any = None,
        hidden: Any | None = None,
        hidden_variant: str | None = None,
        keep_live_ref: bool = False,
        cache_ref: list[Any] | None = None,
        session_id: str | None = None,
        template_hash: str | None = None,
        mtp_history_policy: str | None = None,
        draft_head_identity: str | None = None,
        policy_fingerprint: str | None = None,
        mtp_history_snapshot: Any | None = None,
        snapshot_epoch: int = 0,
        mtp_snapshot_epoch: int | None = None,
        nbytes_override: int | None = None,
    ) -> SessionBankEntry | None:
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            raise ValueError("cannot store an empty prefix")
        if mtp_snapshot_epoch is not None and int(mtp_snapshot_epoch) != int(snapshot_epoch):
            raise ValueError("trunk and MTP snapshots must share the same commit boundary")
        self.last_put_nbytes = 0
        self.last_put_skipped_oversized_snapshot = False
        computed_nbytes = (
            _snapshot_nbytes(cache_snapshot)
            + _tree_nbytes(logits)
            + _tree_nbytes(hidden)
            + _tree_nbytes(mtp_history_snapshot)
        )
        entry_nbytes = int(nbytes_override if nbytes_override is not None else computed_nbytes)
        self.last_put_nbytes = int(entry_nbytes)
        if entry_nbytes > self.per_session_max_bytes:
            self.last_put_skipped_oversized_snapshot = True
            self.eviction_log.append(
                {
                    "reason": "skipped_oversized_snapshot",
                    "session_id": session_id,
                    "prefix_len": len(tokens),
                    "token_hash": token_prefix_hash(tokens),
                    "nbytes": int(entry_nbytes),
                    "budget": int(self.per_session_max_bytes),
                }
            )
            return None
        snapshot = CacheSnapshot(
            states=tuple(_clone_tree(item) for item in cache_snapshot.states),
            meta_states=tuple(_clone_tree(item) for item in cache_snapshot.meta_states),
        )
        entry = SessionBankEntry(
            token_ids=tokens,
            token_hash=token_prefix_hash(tokens),
            model_path=str(runtime.model_path),
            mtp_enabled=bool(runtime.mtp_enabled),
            hidden_variant=hidden_variant,
            cache_snapshot=snapshot,
            logits=_clone_tree(logits),
            hidden=_clone_tree(hidden),
            cache_ref=cache_ref if keep_live_ref else None,
            nbytes=int(entry_nbytes),
            session_id=session_id,
            template_hash=template_hash,
            mtp_history_policy=mtp_history_policy,
            draft_head_identity=draft_head_identity,
            policy_fingerprint=policy_fingerprint,
            mtp_history_snapshot=_clone_tree(mtp_history_snapshot),
            snapshot_epoch=int(snapshot_epoch),
            mtp_snapshot_epoch=(
                int(mtp_snapshot_epoch)
                if mtp_snapshot_epoch is not None
                else (int(snapshot_epoch) if mtp_history_snapshot is not None else None)
            ),
        )
        self._enqueue_cold_entry(entry)
        self._entries[tokens] = entry
        self._evict_if_needed(protected_tokens=tokens)
        return entry

    def longest_prefix(self, token_ids: list[int] | tuple[int, ...]) -> SessionBankEntry | None:
        tokens = tuple(int(token) for token in token_ids)
        best: SessionBankEntry | None = None
        for prefix, entry in self._entries.items():
            if len(prefix) > len(tokens):
                continue
            if tokens[: len(prefix)] != prefix:
                continue
            if best is None or len(prefix) > len(best.token_ids):
                best = entry
        return best

    def near_prefix_candidates(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        max_token_gap: int = 8,
        min_matched_tokens: int = 64,
        block_size: int = DEFAULT_PREFIX_BLOCK_SIZE,
        block_min_matched_tokens: int = DEFAULT_BLOCK_PREFIX_MIN_MATCH_TOKENS,
        allow_block_prefix: bool = True,
        model_path: str | None = None,
        mtp_enabled: bool | None = None,
        hidden_variant: str | None = None,
        template_hash: str | None = None,
        mtp_history_policy: str | None = None,
        draft_head_identity: str | None = None,
        policy_fingerprint: str | None = None,
    ) -> list[tuple[SessionBankEntry, int]]:
        """Return entries whose divergence can be restored from a safe boundary.

        The tiny-gap path covers tokenizer-boundary drift at the very end of a
        stored transcript. The block-prefix path covers real agent follow-ups:
        if a new prompt shares a large stable token prefix, restore to the last
        full block and prefill only the changed suffix.
        """
        tokens = tuple(int(token) for token in token_ids)
        gap_limit = max(0, int(max_token_gap))
        min_match = max(1, int(min_matched_tokens))
        block = max(1, int(block_size))
        block_min_match = max(block, int(block_min_matched_tokens))
        matches: list[tuple[SessionBankEntry, int]] = []
        best_diag: dict[str, Any] | None = None
        self._purge_expired()
        for entry in self._entries.values():
            prefix = entry.token_ids
            if not prefix:
                continue
            matched = common_prefix_len(tokens, prefix)
            gap = len(prefix) - matched
            safe_block = min(
                block_aligned_prefix_len(matched, block_size=block),
                len(prefix),
                len(tokens),
            )
            diag = {
                "prompt_len": len(tokens),
                "session_id": entry.session_id,
                "stored_prefix_len": len(prefix),
                "common_prefix_tokens": int(matched),
                "nearest_boundary_tokens": int(safe_block),
                "near_prefix_gap": int(gap),
                "token_hash": entry.token_hash,
            }
            if best_diag is None or (
                int(diag["common_prefix_tokens"]),
                int(diag["nearest_boundary_tokens"]),
                int(diag["stored_prefix_len"]),
            ) > (
                int(best_diag["common_prefix_tokens"]),
                int(best_diag["nearest_boundary_tokens"]),
                int(best_diag["stored_prefix_len"]),
            ):
                best_diag = diag

            required_match = min(min_match, max(1, len(prefix) - gap_limit))
            # A stored prefix may include assistant output after the exact
            # prompt boundary. If the requested prompt is wholly contained in
            # that longer continuation, restoring at `matched` can leave decode
            # sitting on a post-answer/EOS boundary. Treat that as unsafe for
            # the tiny-gap path; long prompts can still use a block-aligned
            # restore below and re-prefill the tail to the real prompt end.
            if (
                gap >= 0
                and gap <= gap_limit
                and matched >= required_match
                and matched < len(tokens)
            ):
                matches.append((entry, matched))
                continue

            if not allow_block_prefix:
                continue
            if safe_block < block_min_match:
                continue
            if safe_block < 2:
                continue
            if safe_block > matched:
                continue
            matches.append((entry, safe_block))

        cold_match = self._cold_near_prefix_candidate(
            tokens,
            max_token_gap=gap_limit,
            min_matched_tokens=min_match,
            block_size=block,
            block_min_matched_tokens=block_min_match,
            allow_block_prefix=allow_block_prefix,
            model_path=model_path,
            mtp_enabled=mtp_enabled,
            hidden_variant=hidden_variant,
            template_hash=template_hash,
            mtp_history_policy=mtp_history_policy,
            draft_head_identity=draft_head_identity,
            policy_fingerprint=policy_fingerprint,
        )
        if cold_match is not None:
            matches.append(cold_match)
        matches.sort(key=lambda item: (item[1], item[0].prefix_len), reverse=True)
        if matches:
            entry, matched = matches[0]
            cache_source = str(getattr(entry, "cache_source", "ram") or "ram")
            self.last_prefix_diagnostic = {
                "prompt_len": len(tokens),
                "session_id": entry.session_id,
                "stored_prefix_len": entry.prefix_len,
                "common_prefix_tokens": int(common_prefix_len(tokens, entry.token_ids)),
                "nearest_boundary_tokens": int(matched),
                "new_prefill_tokens": max(0, len(tokens) - int(matched)),
                "miss_reason": None,
                "restore_kind": (
                    "near_boundary"
                    if entry.prefix_len - int(matched) <= gap_limit
                    else "block_prefix"
                ),
                "cache_source": cache_source,
            }
        else:
            best = best_diag or {
                "prompt_len": len(tokens),
                "session_id": None,
                "stored_prefix_len": 0,
                "common_prefix_tokens": 0,
                "nearest_boundary_tokens": 0,
            }
            best["miss_reason"] = (
                CacheMissReason.PREFIX_DIVERGENCE_AT_TOKEN.value
                if self._entries
                else CacheMissReason.NEW_SESSION.value
            )
            self.last_prefix_diagnostic = best
        return matches

    def _cold_near_prefix_candidate(
        self,
        tokens: tuple[int, ...],
        *,
        max_token_gap: int,
        min_matched_tokens: int,
        block_size: int,
        block_min_matched_tokens: int,
        allow_block_prefix: bool,
        model_path: str | None,
        mtp_enabled: bool | None,
        hidden_variant: str | None,
        template_hash: str | None,
        mtp_history_policy: str | None,
        draft_head_identity: str | None,
        policy_fingerprint: str | None,
    ) -> tuple[SessionBankEntry, int] | None:
        if self.cold_tier is None:
            return None
        if model_path is None or mtp_enabled is None:
            return None
        raw_enabled = os.environ.get("MTPLX_SESSION_BLOCK_PREFIX_RESTORE")
        if raw_enabled is None or str(raw_enabled).strip().lower() in {"0", "false", "no", "off"}:
            return None
        lookup = getattr(self.cold_tier, "lookup_prefix_boundary", None)
        if not callable(lookup):
            return None
        result = lookup(
            tokens,
            model_path=model_path,
            mtp_enabled=bool(mtp_enabled),
            hidden_variant=hidden_variant,
            template_hash=template_hash,
            mtp_history_policy=mtp_history_policy,
            draft_head_identity=draft_head_identity,
            policy_fingerprint=policy_fingerprint,
            max_token_gap=max_token_gap,
            min_matched_tokens=min_matched_tokens,
            block_size=block_size,
            block_min_matched_tokens=block_min_matched_tokens,
            allow_block_prefix=allow_block_prefix,
        )
        if result is None:
            return None
        record = getattr(result, "record", None)
        matched = int(getattr(result, "matched_tokens", 0) or 0)
        if record is None or matched <= 0:
            return None
        metadata = dict(getattr(record, "metadata", {}) or {})
        entry = SessionBankEntry(
            token_ids=tuple(int(token) for token in record.token_ids),
            token_hash=metadata.get("token_hash") or token_prefix_hash(record.token_ids),
            model_path=str(metadata.get("model_path") or model_path),
            mtp_enabled=bool(metadata.get("mtp_enabled", mtp_enabled)),
            hidden_variant=metadata.get("hidden_variant"),
            cache_snapshot=record.cache_snapshot,
            logits=_clone_tree(record.logits),
            hidden=_clone_tree(record.hidden),
            nbytes=int(getattr(record, "nbytes", 0) or 0),
            session_id=metadata.get("session_id"),
            template_hash=metadata.get("template_hash"),
            mtp_history_policy=metadata.get("mtp_history_policy"),
            draft_head_identity=metadata.get("draft_head_identity"),
            policy_fingerprint=metadata.get("policy_fingerprint"),
            mtp_history_snapshot=_clone_tree(record.mtp_history_snapshot),
            snapshot_epoch=int(metadata.get("snapshot_epoch") or len(record.token_ids)),
            mtp_snapshot_epoch=(
                int(metadata["mtp_snapshot_epoch"])
                if metadata.get("mtp_snapshot_epoch") is not None
                else (
                    int(metadata.get("snapshot_epoch") or len(record.token_ids))
                    if record.mtp_history_snapshot is not None
                    else None
                )
            ),
        )
        setattr(entry, "cache_source", "ssd")
        setattr(entry, "ssd_cache_hit", True)
        setattr(entry, "ssd_cached_tokens", matched)
        setattr(entry, "ssd_restore_s", float(getattr(record, "restore_s", 0.0) or 0.0))
        return entry, matched

    def restore(
        self,
        runtime: MTPLXRuntime,
        token_ids: list[int] | tuple[int, ...],
        *,
        mode: str = "clone",
        session_id: str | None = None,
        hidden_variant: str | None = None,
        template_hash: str | None = None,
        mtp_history_policy: str | None = None,
        draft_head_identity: str | None = None,
        policy_fingerprint: str | None = None,
        cache_factory: Callable[[], list[Any]] | None = None,
        mtp_cache_factory: Callable[[], list[Any]] | None = None,
    ) -> SessionBankRestore | None:
        mode = str(mode).replace("-", "_")
        if mode == "reference_lease":
            mode = "reference"
        if mode not in {"clone", "reference"}:
            raise ValueError("mode must be 'clone', 'reference', or 'reference_lease'")
        self.last_miss_reason = None
        self._purge_expired()

        def cold_fallback() -> SessionBankRestore | None:
            return self._restore_cold(
                runtime,
                token_ids,
                session_id=session_id,
                hidden_variant=hidden_variant,
                template_hash=template_hash,
                mtp_history_policy=mtp_history_policy,
                draft_head_identity=draft_head_identity,
                policy_fingerprint=policy_fingerprint,
            )

        entry = self.longest_prefix(token_ids)
        if entry is None:
            self.last_miss_reason = (
                CacheMissReason.PREFIX_DIVERGENCE_AT_TOKEN.value
                if self._entries
                else CacheMissReason.NEW_SESSION.value
            )
            if self.last_prefix_diagnostic is not None:
                self.last_prefix_diagnostic["miss_reason"] = self.last_miss_reason
            return cold_fallback()
        if entry.model_path != str(runtime.model_path):
            self.last_miss_reason = CacheMissReason.MODEL_MISMATCH.value
            return cold_fallback()
        if hidden_variant is not None and entry.hidden_variant != hidden_variant:
            self.last_miss_reason = CacheMissReason.POLICY_MISMATCH.value
            return cold_fallback()
        if template_hash is not None and entry.template_hash != template_hash:
            self.last_miss_reason = CacheMissReason.TEMPLATE_MISMATCH.value
            return cold_fallback()
        if mtp_history_policy is not None and not _mtp_history_policy_compatible(
            entry.mtp_history_policy, mtp_history_policy
        ):
            self.last_miss_reason = CacheMissReason.POLICY_MISMATCH.value
            return cold_fallback()
        if draft_head_identity is not None and entry.draft_head_identity != draft_head_identity:
            self.last_miss_reason = CacheMissReason.POLICY_MISMATCH.value
            return cold_fallback()
        if policy_fingerprint is not None and entry.policy_fingerprint != policy_fingerprint:
            self.last_miss_reason = CacheMissReason.POLICY_MISMATCH.value
            return cold_fallback()
        if (
            entry.mtp_snapshot_epoch is not None
            and int(entry.mtp_snapshot_epoch) != int(entry.snapshot_epoch)
        ):
            self.last_miss_reason = CacheMissReason.SNAPSHOT_DESYNC.value
            return cold_fallback()
        actual_restore_mode = "clone"
        if mode == "reference" and entry.cache_ref is not None:
            cache = entry.cache_ref
            entry.cache_ref = None
            if not _trim_cache_ref_to_prefix(cache, entry.prefix_len):
                self.last_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
                return cold_fallback()
            actual_restore_mode = "reference_lease"
        else:
            if entry.live_ref_only:
                self.last_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
                return cold_fallback()
            cache = cache_factory() if cache_factory is not None else runtime.make_cache()
            restore_cache(
                cache,
                entry.cache_snapshot,
                restore_meta_state=cache_factory is None,
            )
        mtp_history_cache = None
        if mode == "reference" and entry.mtp_history_cache_ref is not None:
            mtp_history_cache = entry.mtp_history_cache_ref
            entry.mtp_history_cache_ref = None
            if not _trim_cache_ref_to_prefix(mtp_history_cache, entry.prefix_len):
                self.last_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
                return cold_fallback()
        elif entry.mtp_history_snapshot is not None:
            mtp_history_cache = (
                mtp_cache_factory()
                if mtp_cache_factory is not None
                else runtime.make_mtp_cache()
            )
            restore_cache(mtp_history_cache, entry.mtp_history_snapshot)
        elif entry.live_ref_only and entry.mtp_snapshot_epoch is not None:
            self.last_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
            return cold_fallback()
        entry.hits += 1
        entry.last_access_s = time.time()
        self.last_restore_source = "ram"
        self.last_ssd_restore_s = 0.0
        lookup_len = len(tuple(int(token) for token in token_ids))
        self.last_prefix_diagnostic = {
            "prompt_len": lookup_len,
            "session_id": entry.session_id,
            "stored_prefix_len": entry.prefix_len,
            "common_prefix_tokens": entry.prefix_len,
            "nearest_boundary_tokens": entry.prefix_len,
            "new_prefill_tokens": max(0, lookup_len - entry.prefix_len),
            "miss_reason": None,
            "restore_kind": "exact_prefix",
        }
        return SessionBankRestore(
            entry=entry,
            cache=cache,
            logits=_clone_tree(entry.logits),
            hidden=_clone_tree(entry.hidden),
            restored_nbytes=entry.nbytes,
            restore_mode=actual_restore_mode,
            mtp_history_snapshot=_clone_tree(entry.mtp_history_snapshot),
            mtp_history_cache=mtp_history_cache,
            cache_source="ram",
            extra_state=_clone_tree(entry.extra_state),
        )

    def restore_entry_prefix_cache(
        self,
        runtime: MTPLXRuntime,
        entry: SessionBankEntry,
        prefix_len: int,
        *,
        mode: str = "clone",
        cache_factory: Callable[[], list[Any]] | None = None,
        mtp_cache_factory: Callable[[], list[Any]] | None = None,
    ) -> tuple[list[Any], list[Any] | None, str] | None:
        """Restore a cached entry to an earlier safe prefix boundary.

        Exact ``restore()`` only works when the stored token prefix is a
        literal prefix of the next prompt. Real agent transcripts often diverge
        at the assistant-generation marker while sharing almost the entire long
        user/workspace prefix. This helper lets the generation layer restore a
        block-aligned boundary from the same entry and prefill only the suffix.
        """

        mode = str(mode).replace("-", "_")
        if mode == "reference_lease":
            mode = "reference"
        if mode not in {"clone", "reference"}:
            raise ValueError("mode must be 'clone', 'reference', or 'reference_lease'")
        matched = int(prefix_len)
        if matched < 1 or matched > int(entry.prefix_len):
            return None

        actual_restore_mode = "clone"
        mtp_history_trim_tokens = max(0, int(entry.prefix_len) - matched)
        if mode == "reference" and entry.cache_ref is not None:
            cache = entry.cache_ref
            entry.cache_ref = None
            if not _trim_cache_ref_to_prefix(cache, matched):
                self.last_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
                return None
            actual_restore_mode = "reference_lease"
        else:
            if entry.live_ref_only:
                self.last_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
                return None
            cache = cache_factory() if cache_factory is not None else runtime.make_cache()
            restore_cache(
                cache,
                entry.cache_snapshot,
                restore_meta_state=cache_factory is None,
            )
            if not _trim_cache_ref_to_prefix(cache, matched):
                self.last_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
                return None

        mtp_history_cache = None
        if mode == "reference" and entry.mtp_history_cache_ref is not None:
            mtp_history_cache = entry.mtp_history_cache_ref
            entry.mtp_history_cache_ref = None
            if not _trim_cache_ref_by_tokens(
                mtp_history_cache,
                mtp_history_trim_tokens,
            ):
                self.last_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
                return None
        elif entry.mtp_history_snapshot is not None:
            mtp_history_cache = (
                mtp_cache_factory()
                if mtp_cache_factory is not None
                else runtime.make_mtp_cache()
            )
            restore_cache(mtp_history_cache, entry.mtp_history_snapshot)
            if not _trim_cache_ref_by_tokens(
                mtp_history_cache,
                mtp_history_trim_tokens,
            ):
                self.last_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
                return None
        elif entry.live_ref_only and entry.mtp_snapshot_epoch is not None:
            self.last_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
            return None

        return cache, mtp_history_cache, actual_restore_mode

    def clear(self, *, session_id: str | None = None) -> int:
        if session_id is None:
            count = len(self._entries)
            self._entries.clear()
            return count
        victims = [
            tokens
            for tokens, entry in self._entries.items()
            if entry.session_id == session_id
        ]
        for tokens in victims:
            self._entries.pop(tokens, None)
        return len(victims)

    def archive_cold_tier(self) -> dict[str, Any]:
        if self.cold_tier is None:
            return {"archived": False, "reason": "ssd_cache_disabled"}
        archive = getattr(self.cold_tier, "archive", None)
        if not callable(archive):
            return {"archived": False, "reason": "ssd_cache_archive_unavailable"}
        return archive()

    def flush_cold_tier(self, *, timeout_s: float = 30.0) -> bool:
        if self.cold_tier is None:
            return True
        flush = getattr(self.cold_tier, "flush", None)
        if not callable(flush):
            return True
        return bool(flush(timeout_s=timeout_s))

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes,
            "per_session_max_bytes": self.per_session_max_bytes,
            "idle_ttl_s": self.idle_ttl_s,
            "entries": len(self._entries),
            "total_nbytes": self.total_nbytes,
            "last_miss_reason": self.last_miss_reason,
            "last_restore_source": self.last_restore_source,
            "last_ssd_restore_s": self.last_ssd_restore_s,
            "last_prefix_diagnostic": self.last_prefix_diagnostic,
            "cold_tier": (
                self.cold_tier.stats()
                if self.cold_tier is not None and hasattr(self.cold_tier, "stats")
                else {"enabled": False}
            ),
            "prefixes": [
                {
                    "session_id": entry.session_id,
                    "prefix_len": entry.prefix_len,
                    "token_hash": entry.token_hash,
                    "model_path": entry.model_path,
                    "mtp_enabled": entry.mtp_enabled,
                    "hidden_variant": entry.hidden_variant,
                    "template_hash": entry.template_hash,
                    "mtp_history_policy": entry.mtp_history_policy,
                    "draft_head_identity": entry.draft_head_identity,
                    "policy_fingerprint": entry.policy_fingerprint,
                    "hits": entry.hits,
                    "nbytes": entry.nbytes,
                    "created_at_s": entry.created_at_s,
                    "last_access_s": entry.last_access_s,
                    "has_live_ref": entry.cache_ref is not None,
                    "has_mtp_history_live_ref": entry.mtp_history_cache_ref is not None,
                    "live_ref_only": bool(entry.live_ref_only),
                    "snapshot_epoch": entry.snapshot_epoch,
                    "mtp_snapshot_epoch": entry.mtp_snapshot_epoch,
                }
                for entry in sorted(self._entries.values(), key=lambda item: item.prefix_len)
            ],
            "eviction_log": list(self.eviction_log[-16:]),
        }

    def _enqueue_cold_entry(self, entry: SessionBankEntry) -> None:
        if entry.live_ref_only:
            return
        if self.cold_tier is None:
            return
        put_entry = getattr(self.cold_tier, "put_entry", None)
        if not callable(put_entry):
            return
        capabilities = ["ar_insert"]
        if entry.logits is not None and entry.hidden is not None:
            capabilities.append("mtp_full")
        try:
            put_entry(entry, capabilities=capabilities)
        except Exception as exc:
            self.eviction_log.append(
                {
                    "reason": "ssd_enqueue_error",
                    "session_id": entry.session_id,
                    "prefix_len": entry.prefix_len,
                    "token_hash": entry.token_hash,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    def _restore_cold(
        self,
        runtime: MTPLXRuntime,
        token_ids: list[int] | tuple[int, ...],
        *,
        session_id: str | None,
        hidden_variant: str | None,
        template_hash: str | None,
        mtp_history_policy: str | None,
        draft_head_identity: str | None,
        policy_fingerprint: str | None,
    ) -> SessionBankRestore | None:
        if self.cold_tier is None:
            return None
        lookup = getattr(self.cold_tier, "lookup", None)
        if not callable(lookup):
            return None
        record = lookup(
            token_ids,
            model_path=str(runtime.model_path),
            mtp_enabled=bool(runtime.mtp_enabled),
            hidden_variant=hidden_variant,
            template_hash=template_hash,
            mtp_history_policy=mtp_history_policy,
            draft_head_identity=draft_head_identity,
            policy_fingerprint=policy_fingerprint,
        )
        if record is None:
            if hasattr(self.cold_tier, "stats"):
                cold_stats = self.cold_tier.stats()
                cold_miss = cold_stats.get("last_miss_reason")
                if cold_miss:
                    self.last_miss_reason = str(cold_miss)
                    if self.last_prefix_diagnostic is not None:
                        self.last_prefix_diagnostic["miss_reason"] = self.last_miss_reason
            return None
        if hidden_variant is not None and (
            getattr(record, "logits", None) is None
            or getattr(record, "hidden", None) is None
        ):
            self.last_miss_reason = "ssd_missing_mtp_generation_state"
            return None
        if (
            mtp_history_policy in _COMMITTED_CACHE_POLICIES
            and getattr(record, "mtp_history_snapshot", None) is None
        ):
            self.last_miss_reason = "ssd_missing_mtp_history"
            return None
        metadata = dict(getattr(record, "metadata", {}) or {})
        entry = SessionBankEntry(
            token_ids=tuple(int(token) for token in record.token_ids),
            token_hash=metadata.get("token_hash") or token_prefix_hash(record.token_ids),
            model_path=str(metadata.get("model_path") or runtime.model_path),
            mtp_enabled=bool(metadata.get("mtp_enabled", runtime.mtp_enabled)),
            hidden_variant=metadata.get("hidden_variant"),
            cache_snapshot=record.cache_snapshot,
            logits=_clone_tree(record.logits),
            hidden=_clone_tree(record.hidden),
            nbytes=int(getattr(record, "nbytes", 0) or 0),
            session_id=session_id or metadata.get("session_id"),
            template_hash=metadata.get("template_hash"),
            mtp_history_policy=metadata.get("mtp_history_policy"),
            draft_head_identity=metadata.get("draft_head_identity"),
            policy_fingerprint=metadata.get("policy_fingerprint"),
            mtp_history_snapshot=_clone_tree(record.mtp_history_snapshot),
            snapshot_epoch=int(metadata.get("snapshot_epoch") or len(record.token_ids)),
            mtp_snapshot_epoch=(
                int(metadata["mtp_snapshot_epoch"])
                if metadata.get("mtp_snapshot_epoch") is not None
                else (
                    int(metadata.get("snapshot_epoch") or len(record.token_ids))
                    if record.mtp_history_snapshot is not None
                    else None
                )
            ),
        )
        if (
            entry.mtp_snapshot_epoch is not None
            and int(entry.mtp_snapshot_epoch) != int(entry.snapshot_epoch)
        ):
            self.last_miss_reason = CacheMissReason.SNAPSHOT_DESYNC.value
            return None
        cache = runtime.make_cache()
        restore_cache(cache, entry.cache_snapshot)
        mtp_history_cache = None
        if entry.mtp_history_snapshot is not None:
            mtp_history_cache = runtime.make_mtp_cache()
            restore_cache(mtp_history_cache, entry.mtp_history_snapshot)
        entry.hits += 1
        entry.last_access_s = time.time()
        self._entries[entry.token_ids] = entry
        self._evict_if_needed(protected_tokens=entry.token_ids)
        self.last_restore_source = "ssd"
        self.last_ssd_restore_s = float(getattr(record, "restore_s", 0.0) or 0.0)
        self.last_miss_reason = None
        lookup_len = len(tuple(int(token) for token in token_ids))
        self.last_prefix_diagnostic = {
            "prompt_len": lookup_len,
            "session_id": entry.session_id,
            "stored_prefix_len": entry.prefix_len,
            "common_prefix_tokens": entry.prefix_len,
            "nearest_boundary_tokens": entry.prefix_len,
            "new_prefill_tokens": max(0, lookup_len - entry.prefix_len),
            "miss_reason": None,
            "restore_kind": "ssd_prefix",
        }
        return SessionBankRestore(
            entry=entry,
            cache=cache,
            logits=_clone_tree(entry.logits),
            hidden=_clone_tree(entry.hidden),
            restored_nbytes=entry.nbytes,
            restore_mode="ssd_clone",
            mtp_history_snapshot=_clone_tree(entry.mtp_history_snapshot),
            mtp_history_cache=mtp_history_cache,
            cache_source="ssd",
            ssd_cache_hit=True,
            ssd_cached_tokens=entry.prefix_len,
            ssd_restore_s=self.last_ssd_restore_s,
            extra_state=_clone_tree(entry.extra_state),
        )

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [
            entry
            for entry in self._entries.values()
            if now - float(entry.last_access_s) > self.idle_ttl_s
        ]
        for entry in expired:
            self._evict_entry(entry, reason=CacheMissReason.EVICTED.value)

    def _session_nbytes(self, session_id: str | None) -> int:
        return sum(
            entry.nbytes
            for entry in self._entries.values()
            if entry.session_id == session_id
        )

    def _evict_if_needed(self, *, protected_tokens: tuple[int, ...] | None = None) -> None:
        while True:
            if not self._entries:
                return
            session_over_budget = {
                entry.session_id
                for entry in self._entries.values()
                if self._session_nbytes(entry.session_id) > self.per_session_max_bytes
            }
            reason: str | None = None
            candidates = list(self._entries.values())
            if len(self._entries) > self.max_entries:
                reason = CacheMissReason.EVICTED.value
            elif self.total_nbytes > self.max_bytes:
                reason = CacheMissReason.EVICTED.value
            elif session_over_budget:
                reason = CacheMissReason.EVICTED.value
                candidates = [
                    entry
                    for entry in candidates
                    if entry.session_id in session_over_budget
                ]
            else:
                return

            unprotected = [
                entry
                for entry in candidates
                if protected_tokens is None or entry.token_ids != protected_tokens
            ]
            if unprotected:
                candidates = unprotected
            elif len(candidates) == 1:
                entry = candidates[0]
                if (
                    entry.nbytes > self.per_session_max_bytes
                    or entry.nbytes > self.max_bytes
                ):
                    self._evict_entry(entry, reason=reason or CacheMissReason.EVICTED.value)
                    continue
                return
            victim = min(
                candidates,
                key=lambda entry: (entry.last_access_s, -entry.nbytes, entry.created_at_s),
            )
            self._evict_entry(victim, reason=reason)

    def _evict_entry(self, entry: SessionBankEntry, *, reason: str) -> None:
        entry.eviction_reason = reason
        self._entries.pop(entry.token_ids, None)
        self.eviction_log.append(
            {
                "reason": reason,
                "session_id": entry.session_id,
                "prefix_len": entry.prefix_len,
                "token_hash": entry.token_hash,
                "nbytes": entry.nbytes,
                "last_access_s": entry.last_access_s,
            }
        )


def prefill_target(
    runtime: MTPLXRuntime,
    token_ids: list[int],
    *,
    return_hidden: bool = True,
) -> tuple[list[Any], Any, Any | None, float]:
    """Prefill using the same all-but-last/last-token split as generation."""
    if not token_ids:
        raise ValueError("token_ids must not be empty")
    cache = runtime.make_cache()
    elapsed = 0.0
    if len(token_ids) > 1:
        started = time.perf_counter()
        prefill = runtime.forward_ar(
            mx.array([token_ids[:-1]]),
            cache=cache,
            return_hidden=False,
        )
        mx.eval(prefill)
        elapsed += time.perf_counter() - started

    started = time.perf_counter()
    result = runtime.forward_ar(
        mx.array([[token_ids[-1]]]),
        cache=cache,
        return_hidden=return_hidden,
    )
    if return_hidden:
        logits, hidden_seq = result
        mx.eval(logits, hidden_seq)
        hidden = hidden_seq[:, -1:, :]
    else:
        logits = result
        hidden = None
        mx.eval(logits)
    elapsed += time.perf_counter() - started
    return cache, logits[:, -1, :], hidden, elapsed


def prefill_target_with_session_bank(
    runtime: MTPLXRuntime,
    token_ids: list[int],
    bank: SessionBank,
    *,
    return_hidden: bool = True,
    restore_mode: str = "clone",
) -> tuple[list[Any], Any, Any | None, float, dict[str, Any]]:
    started_total = time.perf_counter()
    restored = bank.restore(runtime, token_ids, mode=restore_mode)
    if restored is None:
        cache, logits, hidden, elapsed = prefill_target(
            runtime,
            token_ids,
            return_hidden=return_hidden,
        )
        return cache, logits, hidden, elapsed, {
            "hit": False,
            "prefix_len": 0,
            "suffix_len": len(token_ids),
        }

    suffix = list(token_ids[restored.entry.prefix_len :])
    if not suffix:
        elapsed = time.perf_counter() - started_total
        return restored.cache, restored.logits, restored.hidden, elapsed, {
            "hit": True,
            "prefix_len": restored.entry.prefix_len,
            "suffix_len": 0,
            "restored_nbytes": restored.restored_nbytes,
            "restore_included_s": elapsed,
            "restore_mode": restore_mode,
        }

    elapsed_suffix = 0.0
    if len(suffix) > 1:
        started = time.perf_counter()
        prefill = runtime.forward_ar(
            mx.array([suffix[:-1]]),
            cache=restored.cache,
            return_hidden=False,
        )
        mx.eval(prefill)
        elapsed_suffix += time.perf_counter() - started

    started = time.perf_counter()
    result = runtime.forward_ar(
        mx.array([[suffix[-1]]]),
        cache=restored.cache,
        return_hidden=return_hidden,
    )
    if return_hidden:
        logits, hidden_seq = result
        mx.eval(logits, hidden_seq)
        hidden = hidden_seq[:, -1:, :]
    else:
        logits = result
        hidden = None
        mx.eval(logits)
    elapsed_suffix += time.perf_counter() - started
    elapsed_total = time.perf_counter() - started_total
    return restored.cache, logits[:, -1, :], hidden, elapsed_total, {
        "hit": True,
        "prefix_len": restored.entry.prefix_len,
        "suffix_len": len(suffix),
        "restored_nbytes": restored.restored_nbytes,
        "suffix_forward_s": elapsed_suffix,
        "restore_and_suffix_s": elapsed_total,
        "restore_mode": restore_mode,
    }


def max_abs_diff(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    diff = mx.abs(left.astype(mx.float32) - right.astype(mx.float32))
    mx.eval(diff)
    return float(np.max(np.asarray(diff)))
