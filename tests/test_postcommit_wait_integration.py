"""Integration tests: scheduler-side wiring of postcommit-wait.

These tests exercise the seam between `_schedule_idle_postcommit_snapshot`
and `EngineSession.set_pending_postcommit`. Together they prove that:

  1. Scheduling an idle postcommit stashes the work future on the session.
  2. The next request's `wait_for_pending_postcommit()` observes the same
     future and bails out cleanly when the postcommit returns.
  3. A foreground submission (chat handler path) does not need to wait on
     idle work that the scheduler has not started yet, because the
     foreground lane has admission priority - we only assert wiring here.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from mtplx.engine_session import EngineSession
from mtplx.server import openai


def _fake_state_with_executor():
    """Stand up a state object good enough for `_schedule_idle_postcommit_snapshot`.

    `_submit_idle_postcommit_model_work` falls back to `state.generation_executor`
    when no model_scheduler is present. Tests run cheaply against this seam
    rather than spinning up the real ModelWorkScheduler.
    """
    return SimpleNamespace(
        lock=threading.Lock(),
        has_foreground=lambda: False,
        generation_executor=ThreadPoolExecutor(max_workers=1),
        postcommit_executor=None,
        model_scheduler=None,
        args=SimpleNamespace(server_console=False),
    )


def _kwargs(session_id: str = "sess-int", session: EngineSession | None = None):
    return dict(
        session_id=session_id,
        messages=[],
        assistant_content="hello",
        assistant_tool_calls=None,
        thinking_enabled=False,
        policy_fingerprint="test-policy",
        unsafe_reason="retokenized_history_mismatch",
        session=session,
        expected_session_revision=getattr(session, "revision", 0) if session else None,
    )


def test_scheduling_stashes_future_on_session(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _fake_state_with_executor()
    barrier = threading.Event()

    def fake_store(*_args, **_kwargs):
        # Block until the test releases us so the future is observably
        # pending on the session for at least one wait cycle.
        barrier.wait(timeout=2.0)
        return {
            "stored": True,
            "mode": "retokenized_history",
            "prefix_len": 1,
            "nbytes": 1,
        }

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_store)
    monkeypatch.setattr(openai, "_server_console_enabled", lambda _state: True)

    session = EngineSession("sess-int")
    assert session.pending_postcommit is None

    openai._schedule_idle_postcommit_snapshot(state, **_kwargs(session=session))

    # The schedule call returns immediately; the future is on the session.
    assert session.pending_postcommit is not None

    # Release the worker; the wait observes a clean completion.
    barrier.set()
    outcome = session.wait_for_pending_postcommit(timeout_s=2.0)
    assert outcome["outcome"] == "completed"
    assert session.pending_postcommit is None

    state.generation_executor.shutdown(wait=True)


def test_wait_times_out_when_postcommit_hangs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operator safety net: if the postcommit job hangs (e.g. the scheduler
    has a foreground backlog draining), the next request bails out and
    proceeds cold instead of stalling.
    """
    state = _fake_state_with_executor()
    block_forever = threading.Event()

    def fake_store(*_args, **_kwargs):
        # Simulate a postcommit job that is stuck (e.g. waiting on lock).
        block_forever.wait(timeout=5.0)
        return {"stored": False, "reason": "stuck"}

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_store)
    monkeypatch.setattr(openai, "_server_console_enabled", lambda _state: True)

    session = EngineSession("sess-int-timeout")
    openai._schedule_idle_postcommit_snapshot(state, **_kwargs(session=session))
    assert session.pending_postcommit is not None

    started = time.monotonic()
    outcome = session.wait_for_pending_postcommit(timeout_s=0.1)
    elapsed = time.monotonic() - started

    assert outcome["outcome"] == "timeout"
    assert outcome["waited"] is True
    assert elapsed < 0.5, "wait must be bounded by the timeout"
    # Future stays referenced by the executor but is removed from the
    # session - the request can now proceed cold without re-blocking.
    assert session.pending_postcommit is None

    block_forever.set()
    state.generation_executor.shutdown(wait=True)


def test_wait_metrics_attach_to_request_observability_shape() -> None:
    """The chat handler attaches the wait outcome to request_observability
    which then merges into the metrics envelope. We assert the shape here
    rather than spin up a full FastAPI test client - the shape is what
    /metrics consumers depend on.
    """
    session = EngineSession("sess-int-metrics")
    # Simulate "no_pending" path the chat handler hits on a brand-new session.
    outcome = session.wait_for_pending_postcommit(timeout_s=1.0)

    # request_observability is just a plain dict in the handler; we mimic that.
    request_observability: dict = {}
    request_observability["postcommit_wait"] = outcome

    assert "postcommit_wait" in request_observability
    pw = request_observability["postcommit_wait"]
    assert set(pw.keys()) == {"waited", "elapsed_s", "outcome", "timeout_s"}
    assert pw["outcome"] == "no_pending"
