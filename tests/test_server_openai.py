from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import time
from threading import Lock
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.profiles import get_profile
from mtplx.server import openai
from mtplx.server.openai import _RateLimiter, create_app, parse_args


def test_runtime_mode_label_distinguishes_sustained_max_and_burst():
    assert (
        openai._health_runtime_mode_label(
            "sustained", "mtp", fan_boost_active=False
        )
        == "Sustained MTP"
    )
    assert (
        openai._health_runtime_mode_label(
            "sustained", "mtp", fan_boost_active=True
        )
        == "Sustained Max MTP"
    )
    assert (
        openai._health_runtime_mode_label(
            "performance-cold", "mtp", fan_boost_active=True
        )
        == "Burst MTP"
    )
    assert (
        openai._health_runtime_mode_label(
            "sustained", "ar", fan_boost_active=False
        )
        == "Sustained AR"
    )


def test_startup_urls_distinguish_wildcard_bind_from_local_url():
    args = SimpleNamespace(host="0.0.0.0", port=8000)

    assert openai._startup_bind_label(args) == "0.0.0.0:8000 (all interfaces)"
    assert openai._startup_server_url(args) == "http://127.0.0.1:8000"
    assert openai._startup_openai_base_url(args) == "http://127.0.0.1:8000/v1"


def test_dynamic_paged_kv_reservation_caps_oversized_response_budget(monkeypatch):
    monkeypatch.delenv("MTPLX_DYNAMIC_PAGED_KV_MAX_INITIAL_NEW_TOKENS", raising=False)

    reservation = openai._dynamic_paged_kv_reservation(
        prompt_tokens=181,
        max_new_tokens=65536,
        mtp_depth=3,
    )

    assert reservation["env"]["MTPLX_DYNAMIC_PAGED_KV_TOKENS"] == str(181 + 16384 + 3)
    assert reservation["requested_new_tokens"] == 65536
    assert reservation["reserved_new_tokens"] == 16384
    assert reservation["initial_new_token_cap"] == 16384
    assert reservation["reservation_capped"] is True


def test_dynamic_paged_kv_reservation_can_disable_initial_cap(monkeypatch):
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV_MAX_INITIAL_NEW_TOKENS", "off")

    reservation = openai._dynamic_paged_kv_reservation(
        prompt_tokens=10,
        max_new_tokens=65536,
        mtp_depth=3,
    )

    assert reservation["env"]["MTPLX_DYNAMIC_PAGED_KV_TOKENS"] == str(10 + 65536 + 3)
    assert reservation["reserved_new_tokens"] == 65536
    assert reservation["initial_new_token_cap"] is None
    assert reservation["reservation_capped"] is False


def test_implicit_anonymous_sessions_do_not_keep_live_refs(monkeypatch):
    monkeypatch.delenv(
        "MTPLX_SESSIONBANK_LIVE_REFS_FOR_IMPLICIT_SESSIONS",
        raising=False,
    )

    assert (
        openai._session_keep_live_refs_for_request(
            session_source="implicit_hash", session_id="anon-bench"
        )
        is False
    )


def test_anonymous_coding_agent_tool_sessions_keep_live_refs(monkeypatch):
    monkeypatch.delenv(
        "MTPLX_SESSIONBANK_LIVE_REFS_FOR_IMPLICIT_SESSIONS",
        raising=False,
    )

    assert (
        openai._session_keep_live_refs_for_request(
            session_source="new",
            session_id="anon-opencode",
            tool_names=["bash", "read", "write"],
        )
        is True
    )


def test_anonymous_non_tool_benchmark_sessions_stay_cold(monkeypatch):
    monkeypatch.delenv(
        "MTPLX_SESSIONBANK_LIVE_REFS_FOR_IMPLICIT_SESSIONS",
        raising=False,
    )

    assert (
        openai._session_keep_live_refs_for_request(
            session_source="new",
            session_id="anon-aime",
            tool_names=[],
        )
        is False
    )


def test_explicit_sessions_keep_live_refs(monkeypatch):
    monkeypatch.delenv(
        "MTPLX_SESSIONBANK_LIVE_REFS_FOR_IMPLICIT_SESSIONS",
        raising=False,
    )

    assert (
        openai._session_keep_live_refs_for_request(
            session_source="metadata.chat_id", session_id="chat-123"
        )
        is True
    )


def test_implicit_session_live_refs_have_diagnostic_override(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSIONBANK_LIVE_REFS_FOR_IMPLICIT_SESSIONS", "1")

    assert (
        openai._session_keep_live_refs_for_request(
            session_source="implicit_hash", session_id="anon-bench"
        )
        is True
    )


class FakeExecutor:
    def submit(self, fn, *args, **kwargs):
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # pragma: no cover - surfaced by caller
            future.set_exception(exc)
        return future

    def shutdown(self, **_kwargs):
        return None


class StreamingTokenizer:
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, **_kwargs
    ):
        assert tokenize is True
        text = "\n".join(
            f"{message['role']}:{message.get('content') or ''}" for message in messages
        )
        if add_generation_prompt:
            text = f"{text}\nassistant:" if text else "assistant:"
        return [ord(char) for char in text]

    def encode(self, text):
        return [ord(char) for char in str(text)]

    def decode(self, tokens, **_kwargs):
        return "".join(chr(int(token)) for token in tokens)


class RecordingBank:
    def __init__(self):
        self.puts: list[dict] = []

    def put(self, **kwargs):
        self.puts.append(kwargs)
        return SimpleNamespace(
            prefix_len=len(kwargs["token_ids"]),
            nbytes=123,
            token_hash="test-token-hash",
        )


class ForegroundState:
    def __init__(self) -> None:
        self.lock = Lock()
        self.foreground_active = 0
        self.last_request_started_at = 0.0

    def begin_foreground(self) -> None:
        self.foreground_active += 1

    def end_foreground(self) -> None:
        self.foreground_active = max(0, self.foreground_active - 1)

    def has_foreground(self) -> bool:
        return self.foreground_active > 0

    def foreground_count(self) -> int:
        return self.foreground_active


def _fake_state(*, api_key: str | None = None, rate_limit: int = 0):
    argv = ["--warmup-tokens", "0", "--rate-limit", str(rate_limit)]
    if api_key:
        argv.extend(["--api-key", api_key])
    args = parse_args(argv)
    return SimpleNamespace(
        args=args,
        model_id="mtplx-test-model",
        runtime=SimpleNamespace(
            model_path=Path("models/example"),
            mtp_enabled=True,
            tokenizer=SimpleNamespace(
                decode=lambda tokens, **_kwargs: "".join(
                    chr(int(token)) for token in tokens
                ),
            ),
        ),
        profile=get_profile(args.profile),
        context_window=4096,
        load_time_s=0.25,
        draft_lm_head={"installed": False, "reason": "test"},
        draft_head_identity="test-head",
        template_hash="test-template",
        main_system_prompt_hash=None,
        fast_path_env_status={},
        profile_env_status={},
        mlx_cache_limit_status={"configured": False},
        metal_memory_caps={"applied": False, "reason": "test"},
        mlx_fork_status={"ok": False},
        warmup_status={"enabled": False, "ran": False, "tokens": 0},
        last_metrics=[{"tok_s": 12.5, "accept_rate": 0.75}],
        rate_limiter=_RateLimiter(rate_limit),
        sessions=SimpleNamespace(
            list_sessions=lambda: {"sessions": []},
            clear_session=lambda session_id: {"cleared": session_id},
            clear_all=lambda: {"cleared": True},
        ),
        generation_executor=FakeExecutor(),
    )


def _fake_streaming_session_state():
    state = _fake_state()
    foreground = ForegroundState()
    state.lock = foreground.lock
    state.begin_foreground = foreground.begin_foreground
    state.end_foreground = foreground.end_foreground
    state.has_foreground = foreground.has_foreground
    state.foreground_count = foreground.foreground_count
    state.runtime.tokenizer = StreamingTokenizer()
    state.sessions = openai.EngineSessionManager(bank=RecordingBank())
    state.generation_executor = ThreadPoolExecutor(max_workers=1)
    state.postcommit_executor = FakeExecutor()
    state.args.stats_footer = False
    return state


def _fake_final_state(tokens):
    return SimpleNamespace(
        final_trunk_cache=["cache"],
        final_logits="logits",
        final_hidden="hidden",
        final_committed_mtp_cache=None,
        generated_token_ids=tuple(tokens),
        safe_to_commit=True,
        finish_reason="stop",
    )


def test_mtplx_settings_endpoint_controls_server_reasoning():
    state = _fake_state(api_key="mtplx-local")
    client = TestClient(create_app(state))

    initial = client.get(
        "/v1/mtplx/settings", headers={"Authorization": "Bearer mtplx-local"}
    )
    assert initial.status_code == 200
    assert initial.json()["reasoning"] == "on"
    assert initial.json()["metal_memory_caps"] == {"applied": False, "reason": "test"}

    off = client.post(
        "/v1/mtplx/settings",
        json={"reasoning": "off"},
        headers={"Authorization": "Bearer mtplx-local"},
    )
    assert off.status_code == 200
    assert off.json()["reasoning"] == "off"
    assert off.json()["enable_thinking"] is False
    assert state.args.enable_thinking is False

    on = client.post(
        "/v1/mtplx/settings",
        json={"reasoning": "on"},
        headers={"Authorization": "Bearer mtplx-local"},
    )
    assert on.status_code == 200
    assert on.json()["reasoning"] == "on"
    assert on.json()["enable_thinking"] is True
    assert state.args.enable_thinking is True


def test_server_console_controls_reasoning_and_mtp_defaults():
    state = _fake_state()

    assert "Reasoning: off" in openai._server_console_handle_command(
        state, "/reasoning off"
    )
    assert state.args.reasoning == "off"
    assert state.args.enable_thinking is False

    assert "Reasoning: auto" in openai._server_console_handle_command(
        state, "/reasoning auto"
    )
    assert state.args.reasoning == "auto"
    assert state.args.enable_thinking is True

    assert "MTP: off" in openai._server_console_handle_command(state, "/mtp off")
    assert state.args.generation_mode == "ar"

    assert "MTP: on" in openai._server_console_handle_command(state, "/mtp on")
    assert state.args.generation_mode == "mtp"


def test_openai_server_health_metrics_and_models_fake_state():
    client = TestClient(create_app(_fake_state()))

    root_head = client.head("/")
    assert root_head.status_code == 200

    root = client.get("/")
    assert root.status_code == 200
    assert "text/html" in root.headers["content-type"]
    # Brand and chat scaffold
    assert "<title>MTPLX</title>" in root.text
    assert "MTPLX" in root.text
    assert 'id="messages"' in root.text
    assert 'id="prompt"' in root.text
    assert "Message MTPLX" in root.text
    assert "/v1/chat/completions" in root.text
    assert "reasoning_content" in root.text
    # Inference settings sidebar — all sliders now
    assert 'id="ctl-temp"' in root.text
    assert 'id="ctl-top-p"' in root.text
    assert 'id="ctl-top-k" type="range"' in root.text
    assert 'id="ctl-mtp" type="checkbox"' in root.text
    assert 'id="ctl-depth" type="range"' in root.text
    assert 'id="ctl-max-tokens" type="range"' in root.text
    assert 'id="ctl-system"' in root.text
    assert 'id="reset-defaults"' in root.text
    # New layout: avatar circles + reasoning-as-its-own-block + turn-* classes
    assert "turn turn-assistant" in root.text
    assert 'class="avatar"' in root.text
    assert "reasoning-block" in root.text
    # Auto-scroll, stop, new-chat, persistence
    assert 'id="jump-pill"' not in root.text
    assert 'id="messages-bottom"' in root.text
    assert "ResizeObserver" in root.text
    assert "scrollIntoView" in root.text
    assert "forceAutoScroll" in root.text
    assert "SCROLL_PIN_THRESHOLD = 160" in root.text
    assert 'id="new-chat-btn"' in root.text
    assert "AbortController" in root.text
    # SETTINGS_KEY bumped to v4 when MTP on/off settings landed; bumping the
    # version invalidates saved sidebar settings without a generation-mode bit.
    assert "mtplx.chat.settings.v4" in root.text
    # Auto-detect of context length must be hooked up so the slider isn't
    # capped at a stale 32k for a 256k-context model.
    assert "discoverServerLimits" in root.text
    assert "/health" in root.text
    # Transport watchdog plus heartbeat handling keep long active generations
    # from looking like crashed servers.
    assert "armStallWatchdog" in root.text
    assert "mtplx_progress.heartbeat" in root.text
    assert "Still working" in root.text
    assert "stream connection went quiet" in root.text
    assert "server has crashed" not in root.text
    # Markdown via marked.js
    assert "marked.min.js" in root.text
    # Live tps element
    assert 'id="live-stats"' in root.text
    assert "tok/s" in root.text
    assert '"mtp_enabled": true' in root.text
    assert 'generation_mode: settingsNow.mtp_enabled ? "mtp" : "ar"' in root.text
    assert '"depth": 3' in root.text
    assert 'id="ctl-depth" type="range" min="1" max="3" step="1" value="3"' in root.text
    # Updated max-tokens default and cap.
    assert 'value="8192"' in root.text
    assert 'min="256" max="32768"' in root.text

    v1 = client.get("/v1")
    assert v1.status_code == 200
    assert v1.json()["openwebui"]["base_url"].endswith("/v1")
    assert "Paste this URL into Open WebUI" in v1.json()["message"]

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model"] == "mtplx-test-model"
    assert health.json()["api_key_required"] is False
    assert health.json()["warmup"]["ran"] is False
    assert health.json()["foreground_active"] == 0
    assert health.json()["active_requests"] == 0
    assert health.json()["last_request_started_at"] == 0.0

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["latest"]["tok_s"] == 12.5

    models = client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "mtplx-test-model"


def test_openai_server_auth_and_rate_limit_fake_state():
    client = TestClient(create_app(_fake_state(api_key="test-key", rate_limit=1)))

    assert client.get("/v1/models").status_code == 401
    assert (
        client.get(
            "/v1/models", headers={"Authorization": "Bearer test-key"}
        ).status_code
        == 200
    )

    limited = client.get("/v1/models", headers={"Authorization": "Bearer test-key"})
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) > 0


def test_anthropic_messages_rejects_empty_request_before_generation():
    client = TestClient(create_app(_fake_state()))

    response = client.post(
        "/v1/messages",
        json={
            "model": "mtplx-test-model",
            "max_tokens": 8,
            "stream": True,
            "messages": [],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "messages must not be empty"


def test_chat_ui_uses_server_depth_default():
    state = _fake_state()
    state.args.depth = 2
    client = TestClient(create_app(state))

    root = client.get("/")

    assert root.status_code == 200
    assert '"depth": 2' in root.text
    assert '"mtp_enabled": true' in root.text
    assert 'id="ctl-depth" type="range" min="1" max="2" step="1" value="2"' in root.text


def test_chat_generation_mode_request_override_routes_ar(monkeypatch):
    captured: dict[str, object] = {}
    state = _fake_state()
    client = TestClient(create_app(state))

    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])

    def fake_run_generation(_state, prompt_ids, **kwargs):
        captured["prompt_ids"] = prompt_ids
        captured["generation_mode"] = kwargs["generation_mode"]
        captured["depth"] = kwargs["depth"]
        return {
            "text": "ok",
            "tokens": [4],
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 1,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 1,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Say READY"}],
            "max_tokens": 4,
            "generation_mode": "ar",
            "depth": 3,
        },
    )

    assert response.status_code == 200
    assert captured["generation_mode"] == "ar"
    assert captured["depth"] == 0
    assert response.json()["mtplx_stats"]["generation_mode"] == "ar"
    assert response.json()["mtplx_stats"]["mtp_depth"] == 0


def test_generation_truth_stats_distinguish_stock_ar_from_target_ar():
    target_state = _fake_state()
    target_state.args.load_mtp = True
    target_state.runtime.mtp_enabled = True
    target_state.draft_lm_head = {"draft_only": {"bits": 4}}

    stock_state = _fake_state()
    stock_state.args.load_mtp = False
    stock_state.runtime.mtp_enabled = False

    target = openai._generation_truth_stats(target_state, "ar")
    assert target["benchmark_mode"] == "mtplx_mtp_loaded_target_ar"
    assert target["draft_head_installed"] is True
    stock = openai._generation_truth_stats(stock_state, "ar")
    assert stock["benchmark_mode"] == "mtplx_stock_ar_unloaded"
    assert stock["load_mtp"] is False
    assert stock["runtime_mtp_enabled"] is False


def test_chat_generation_mode_request_override_routes_mtp_depth(monkeypatch):
    captured: dict[str, object] = {}
    client = TestClient(create_app(_fake_state()))

    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        captured["generation_mode"] = kwargs["generation_mode"]
        captured["depth"] = kwargs["depth"]
        return {
            "text": "ok",
            "tokens": [4],
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 1,
            },
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Say READY"}],
            "max_tokens": 4,
            "generation_mode": "mtp",
            "depth": 1,
        },
    )

    assert response.status_code == 200
    assert captured == {"generation_mode": "mtp", "depth": 1}
    assert response.json()["mtplx_stats"]["generation_mode"] == "mtp"
    assert response.json()["mtplx_stats"]["mtp_depth"] == 1


def test_streaming_session_uses_generation_final_postcommit_without_retokenized_tail(
    monkeypatch,
):
    state = _fake_streaming_session_state()
    captured: dict[str, object] = {}

    def fail_retokenized(*_args, **_kwargs):
        raise AssertionError(
            "streaming fast path must not retokenize/prefill postcommit"
        )

    def fake_run_generation(_state, prompt_ids, **kwargs):
        captured.setdefault(
            "commit_final_state_to_bank",
            kwargs.get("commit_final_state_to_bank"),
        )
        token_callback = kwargs.get("token_callback")
        tokens = [ord("O"), ord("K")]
        if token_callback is not None:
            token_callback(tokens[:1])
            token_callback(tokens[1:])
        return {
            "text": "OK",
            "tokens": tokens,
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 2,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 2,
            "finish_reason": "stop",
            "_final_state": _fake_final_state(tokens),
        }

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fail_retokenized)
    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    with TestClient(create_app(state)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "stream-session"},
            json={
                "messages": [{"role": "user", "content": "Say OK"}],
                "enable_thinking": False,
                "stream": True,
                "max_tokens": 4,
            },
        )
        second = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "stream-session"},
            json={
                "messages": [
                    {"role": "user", "content": "Say OK"},
                    {"role": "assistant", "content": "OK"},
                    {"role": "user", "content": "Again"},
                ],
                "enable_thinking": False,
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert '"content": "OK"' in response.text or (
        '"content": "O"' in response.text and '"content": "K"' in response.text
    )
    assert '"mode": "generation_final_exact"' in response.text
    assert captured["commit_final_state_to_bank"] is False
    assert second.status_code == 200
    assert "already in flight" not in second.text


def test_streaming_unsafe_postcommit_releases_without_blocking_second_request(
    monkeypatch,
):
    state = _fake_streaming_session_state()
    scheduled: list[dict] = []

    def fake_store_generation_final(*_args, **_kwargs):
        return {
            "stored": False,
            "mode": "unsafe",
            "reason": "retokenized_history_mismatch",
        }

    def fake_schedule(*_args, **kwargs):
        scheduled.append(kwargs)
        return {
            "stored": False,
            "mode": "async_pending",
            "reason": kwargs["unsafe_reason"],
        }

    def fake_run_generation(_state, prompt_ids, **kwargs):
        token_callback = kwargs.get("token_callback")
        tokens = [ord("O"), ord("K")]
        if token_callback is not None:
            token_callback(tokens)
        return {
            "text": "OK",
            "tokens": tokens,
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 2,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 2,
            "finish_reason": "stop",
            "_final_state": _fake_final_state(tokens),
        }

    monkeypatch.setattr(
        openai, "_store_generation_final_history_snapshot", fake_store_generation_final
    )
    monkeypatch.setattr(openai, "_schedule_idle_postcommit_snapshot", fake_schedule)
    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    with TestClient(create_app(state)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "unsafe-session"},
            json={
                "messages": [{"role": "user", "content": "Say OK"}],
                "enable_thinking": False,
                "stream": True,
                "max_tokens": 4,
            },
        )
        second = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "unsafe-session"},
            json={
                "messages": [{"role": "user", "content": "Say OK again"}],
                "enable_thinking": False,
                "stream": True,
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    assert '"mode": "async_pending"' in response.text
    assert '"reason": "retokenized_history_mismatch"' in response.text
    assert '"session_prompt_prefix_commit"' in response.text
    assert '"postcommit_prompt_prefix"' in response.text
    assert scheduled
    assert second.status_code == 200
    assert "already in flight" not in second.text


def test_streaming_stop_boundary_mismatch_skips_idle_retokenized_postcommit(
    monkeypatch,
):
    state = _fake_streaming_session_state()
    cleanup_calls: list[str] = []

    def fake_store_generation_final(*_args, **_kwargs):
        return {
            "stored": False,
            "mode": "unsafe",
            "reason": "stop_token_boundary_mismatch",
        }

    def fail_schedule(*_args, **_kwargs):
        raise AssertionError("stop-token mismatches should not schedule GPU postcommit")

    def fake_cleanup(_state, *, reason):
        cleanup_calls.append(reason)
        return {"cleared": True, "reason": reason}

    def fake_run_generation(_state, prompt_ids, **kwargs):
        token_callback = kwargs.get("token_callback")
        tokens = [ord("O"), ord("K")]
        if token_callback is not None:
            token_callback(tokens)
        return {
            "text": "OK",
            "tokens": tokens,
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 2,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 2,
            "finish_reason": "stop",
            "_final_state": _fake_final_state(tokens),
        }

    monkeypatch.setattr(
        openai, "_store_generation_final_history_snapshot", fake_store_generation_final
    )
    monkeypatch.setattr(openai, "_schedule_idle_postcommit_snapshot", fail_schedule)
    monkeypatch.setattr(openai, "_clear_mlx_cache_after_request", fake_cleanup)
    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    with TestClient(create_app(state)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "stop-boundary-session"},
            json={
                "messages": [{"role": "user", "content": "Say OK"}],
                "enable_thinking": False,
                "stream": True,
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    assert '"mode": "async_skipped"' in response.text
    assert '"reason": "stop_token_boundary_mismatch"' in response.text
    assert '"mlx_cache_cleanup"' in response.text
    assert '"postcommit_prompt_prefix"' in response.text
    assert cleanup_calls == ["stop_token_boundary_mismatch_postcommit_skipped"]


def test_streaming_ar_schedules_async_postcommit_in_default_mode(monkeypatch):
    state = _fake_streaming_session_state()
    scheduled: list[dict] = []

    def fail_retokenized(*_args, **_kwargs):
        raise AssertionError("AR streaming must not retokenize inline by default")

    def fake_schedule(*_args, **kwargs):
        scheduled.append(kwargs)
        return {
            "stored": False,
            "mode": "async_pending",
            "reason": kwargs["unsafe_reason"],
        }

    def fake_run_generation(_state, prompt_ids, **kwargs):
        token_callback = kwargs.get("token_callback")
        tokens = [ord("A"), ord("R")]
        if token_callback is not None:
            token_callback(tokens)
        return {
            "text": "AR",
            "tokens": tokens,
            "stats": {
                "generation_mode": "ar",
                "mtp_depth": 0,
                "completion_tokens": 2,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 2,
            "finish_reason": "stop",
            "_final_state": None,
        }

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fail_retokenized)
    monkeypatch.setattr(openai, "_schedule_idle_postcommit_snapshot", fake_schedule)
    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    with TestClient(create_app(state)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "ar-session"},
            json={
                "messages": [{"role": "user", "content": "Say AR"}],
                "enable_thinking": False,
                "generation_mode": "ar",
                "stream": True,
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    assert scheduled
    assert scheduled[0]["unsafe_reason"] == "missing_generation_final_state"
    assert '"mode": "async_pending"' in response.text
    assert '"reason": "missing_generation_final_state"' in response.text
    assert '"generation_mode": "ar"' in response.text


def test_streaming_ar_honors_explicit_inline_postcommit_mode(monkeypatch):
    state = _fake_streaming_session_state()
    state.args.session_postcommit_mode = "inline"
    retokenized_calls: list[dict] = []

    def fake_retokenized(*_args, **kwargs):
        retokenized_calls.append(kwargs)
        return {
            "stored": True,
            "mode": "retokenized_history",
            "prefix_len": 32,
            "nbytes": 99,
        }

    def fake_run_generation(_state, prompt_ids, **kwargs):
        token_callback = kwargs.get("token_callback")
        tokens = [ord("A"), ord("R")]
        if token_callback is not None:
            token_callback(tokens)
        return {
            "text": "AR",
            "tokens": tokens,
            "stats": {
                "generation_mode": "ar",
                "mtp_depth": 0,
                "completion_tokens": 2,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 2,
            "finish_reason": "stop",
            "_final_state": None,
        }

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_retokenized)
    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    with TestClient(create_app(state)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "ar-inline-session"},
            json={
                "messages": [{"role": "user", "content": "Say AR"}],
                "enable_thinking": False,
                "generation_mode": "ar",
                "stream": True,
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    assert retokenized_calls
    assert '"mode": "retokenized_history"' in response.text
    assert '"generation_mode": "ar"' in response.text


def test_invalid_generation_mode_returns_400():
    client = TestClient(create_app(_fake_state()))

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Say READY"}],
            "generation_mode": "off",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "generation_mode must be 'mtp' or 'ar'"
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_chat_accepts_max_completion_tokens_alias_and_benign_extras(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    client = TestClient(create_app(state))
    seen_max_tokens: list[int | None] = []

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        seen_max_tokens.append(kwargs.get("max_tokens"))
        return _fake_generation("ok")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "user-agent": "AndroidStudio"},
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 7,
            "stream_options": {"include_usage": True},
            "response_format": {"type": "text"},
            "parallel_tool_calls": False,
            "metadata": {"client": "android-studio"},
            "user": "local-user",
        },
    )

    assert response.status_code == 200
    assert seen_max_tokens == [7]


def test_android_studio_issue58_replay_fixture_is_accepted(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    client = TestClient(create_app(state))
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "android_studio_issue58_chat.json").read_text(
            encoding="utf-8"
        )
    )
    fixture["stream"] = False
    seen_max_tokens: list[int | None] = []

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        seen_max_tokens.append(kwargs.get("max_tokens"))
        return _fake_generation("ok")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "user-agent": "AndroidStudio"},
        json=fixture,
    )

    assert response.status_code == 200
    assert seen_max_tokens == [64]


class CaptureTokenizer:
    def __init__(self):
        self.calls: list[tuple[list[dict[str, object]], dict[str, object]]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [1, 2, 3]

    def encode(self, text):
        return [ord(char) for char in str(text)]

    def decode(self, tokens, **_kwargs):
        return "".join(chr(int(token)) for token in tokens)


class ToolSchemaRejectingTokenizer(CaptureTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if "tools" in kwargs:
            raise RuntimeError("rich tool schemas are unsupported")
        return [1, 2, 3]


class QwenToolHistoryBoundaryTokenizer:
    """Tiny Qwen-like tokenizer that exposes the OpenCode cache-boundary bug.

    It deliberately merges ``\n\n</think>`` when a full rendered transcript is
    tokenized at once. Segmenting after ``<think>\n`` preserves the same token
    boundary produced by the previous generation prompt.
    """

    merged = 900_001

    def apply_chat_template(self, messages, **kwargs):
        rendered = self._render(messages, add_generation_prompt=bool(kwargs.get("add_generation_prompt")))
        if kwargs.get("tokenize"):
            return self.encode(rendered, add_special_tokens=False)
        return rendered

    def encode(self, text, **_kwargs):
        text = str(text)
        needle = "\n\n</think>"
        ids: list[int] = []
        i = 0
        while i < len(text):
            if text.startswith(needle, i):
                ids.append(self.merged)
                i += len(needle)
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def decode(self, tokens, **_kwargs):
        parts: list[str] = []
        for token in tokens:
            parts.append("\n\n</think>" if int(token) == self.merged else chr(int(token)))
        return "".join(parts)

    def _render(self, messages, *, add_generation_prompt: bool) -> str:
        chunks: list[str] = []
        for message in messages:
            role = message["role"]
            content = str(message.get("content") or "")
            if role == "system":
                chunks.append(f"<|im_start|>system\n{content}<|im_end|>\n")
            elif role == "user":
                chunks.append(f"<|im_start|>user\n{content}<|im_end|>\n")
            elif role == "assistant":
                chunks.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
                if content:
                    chunks.append(content + "\n\n")
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call["function"]
                    chunks.append(f"<tool_call>\n<function={function['name']}>\n")
                    for key, value in function.get("arguments", {}).items():
                        chunks.append(f"<parameter={key}>\n{value}\n</parameter>\n")
                    chunks.append("</function>\n</tool_call>")
                chunks.append("<|im_end|>\n")
            elif role == "tool":
                chunks.append(
                    f"<|im_start|>user\n<tool_response>\n{content}\n"
                    "</tool_response><|im_end|>\n"
                )
        if add_generation_prompt:
            chunks.append("<|im_start|>assistant\n<think>\n")
        return "".join(chunks)


def _tool_schema():
    return {
        "type": "function",
        "function": {
            "name": "session_status",
            "description": "Show the current agent session status.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }


def _add_tool_schema():
    return {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }


def _write_tool_schema():
    return {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "content": {"type": "string"},
                    "createDirs": {"type": "boolean"},
                },
                "required": ["filePath", "content"],
                "additionalProperties": False,
            },
        },
    }


def _bash_tool_schema():
    return {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "description": {"type": "string"},
                    "timeout": {"type": "number"},
                },
                "required": ["command", "description"],
                "additionalProperties": False,
            },
        },
    }


def _question_tool_schema():
    return {
        "type": "function",
        "function": {
            "name": "question",
            "description": "Ask the user one or more questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {"type": "array"},
                },
                "required": ["questions"],
                "additionalProperties": False,
            },
        },
    }


def _fake_generation(text: str):
    return {
        "text": text,
        "tokens": [4],
        "stats": {
            "generation_mode": "ar",
            "mtp_depth": 0,
            "completion_tokens": 1,
        },
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "finish_reason": "stop",
    }


def _fake_streaming_generation(text: str):
    tokens = [ord(char) for char in text]

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        token_callback = kwargs.get("token_callback")
        if token_callback is not None:
            for token in tokens:
                token_callback([token])
        return {
            "text": text,
            "tokens": tokens,
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
            },
            "prompt_tokens": 3,
            "completion_tokens": len(tokens),
            "finish_reason": "stop",
        }

    return fake_run_generation


def _stream_payloads(response_text: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in response_text.splitlines()
        if line.startswith("data: {")
    ]


def test_chat_tools_are_passed_to_qwen_template_and_inherit_default_thinking(
    monkeypatch,
):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(
        openai, "_run_generation", lambda *_args, **_kwargs: _fake_generation("ok")
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Use the tool."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    messages, kwargs = state.runtime.tokenizer.calls[0]
    assert messages[0]["role"] == "system"
    assert "MTPLX tool contract:" in messages[0]["content"]
    assert "session_status" in messages[0]["content"]
    assert kwargs["tools"] == [_tool_schema()]
    assert kwargs["enable_thinking"] is True


def test_tool_contract_includes_exact_schema_keys_for_opencode_write(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(
        openai, "_run_generation", lambda *_args, **_kwargs: _fake_generation("ok")
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Write a file."}],
            "tools": [_write_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    messages, _kwargs = state.runtime.tokenizer.calls[0]
    contract = messages[0]["content"]
    assert "write(filePath:string, content:string, createDirs?:boolean)" in contract
    assert '"filePath":"<string>"' in contract
    assert '"content":"<string>"' in contract
    assert "exact argument keys/case" in contract
    assert "Include every required key" in contract


def test_opencode_title_request_fast_path_does_not_enter_model(monkeypatch):
    state = _fake_state()
    client = TestClient(create_app(state))

    def fail_generation(*_args, **_kwargs):
        raise AssertionError("OpenCode title request should not enter generation")

    monkeypatch.setattr(openai, "_run_generation", fail_generation)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mtplx-test-model",
            "stream": True,
            "max_tokens": 32000,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a title generator. You output ONLY a thread title. "
                        "Generate a brief title. Never use tools."
                    ),
                },
                {"role": "user", "content": "Generate a title for this conversation:"},
                {"role": "user", "content": "hi"},
            ],
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    assert any(
        choice["delta"].get("content") == "Greeting"
        for payload in payloads
        for choice in payload["choices"]
    )
    assert state.last_metrics[-1]["opencode_title_fast_path"] is True
    assert state.last_metrics[-1]["request_max_tokens"] == 32000


def test_tool_requests_enable_prompt_prefix_bank_commit(monkeypatch):
    state = _fake_streaming_session_state()
    state.draft_sampler = None
    state.requests_completed = 0
    captured: dict[str, object] = {}

    def fake_generate_mtpk(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            tokens=[],
            text="",
            stats=SimpleNamespace(
                to_dict=lambda: {
                    "prompt_eval_time_s": 0.0,
                    "generated_tokens": 0,
                    "elapsed_s": 0.0,
                    "tok_s": 0.0,
                }
            ),
            final_state=None,
        )

    monkeypatch.setattr(openai, "generate_mtpk", fake_generate_mtpk)

    openai._run_generation(
        state,
        [1, 2, 3],
        max_tokens=1,
        temperature=None,
        top_p=None,
        top_k=None,
        seed=None,
        generation_mode="mtp",
        depth=3,
        session_id="sess-tool",
        session_bank=state.sessions.bank,
        session_template_hash=state.template_hash,
        session_draft_head_identity=state.draft_head_identity,
        session_policy_fingerprint="policy",
        commit_prompt_prefix_to_bank=True,
    )

    assert captured["commit_prompt_state_to_bank"] is True
    assert captured["commit_prompt_state_keep_live_ref"] is True


def test_run_generation_can_store_final_state_without_live_cache_ref(monkeypatch):
    state = _fake_streaming_session_state()
    state.draft_sampler = None
    state.requests_completed = 0

    def fake_generate_mtpk(*_args, **_kwargs):
        tokens = [ord("O"), ord("K")]
        return SimpleNamespace(
            tokens=tokens,
            text="OK",
            stats=SimpleNamespace(
                to_dict=lambda: {
                    "prompt_eval_time_s": 0.0,
                    "generated_tokens": 2,
                    "elapsed_s": 0.1,
                    "tok_s": 20.0,
                }
            ),
            final_state=_fake_final_state(tokens),
        )

    monkeypatch.setattr(openai, "generate_mtpk", fake_generate_mtpk)

    openai._run_generation(
        state,
        [1, 2, 3],
        max_tokens=65536,
        temperature=None,
        top_p=None,
        top_k=None,
        seed=42,
        generation_mode="mtp",
        depth=3,
        session_id="anon-bench",
        session_bank=state.sessions.bank,
        session_template_hash=state.template_hash,
        session_draft_head_identity=state.draft_head_identity,
        session_policy_fingerprint="policy",
        session_keep_live_ref=False,
    )

    assert state.sessions.bank.puts
    assert state.sessions.bank.puts[-1]["keep_live_ref"] is False


def test_tool_template_schema_failure_retries_with_compact_contract(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = ToolSchemaRejectingTokenizer()
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(
        openai, "_run_generation", lambda *_args, **_kwargs: _fake_generation("ok")
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Write a file."}],
            "tools": [_write_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    first_messages, first_kwargs = state.runtime.tokenizer.calls[0]
    second_messages, second_kwargs = state.runtime.tokenizer.calls[-1]
    assert "tools" in first_kwargs
    assert "tools" not in second_kwargs
    assert first_messages == second_messages
    assert "MTPLX tool contract:" in second_messages[0]["content"]
    assert "write(filePath:string, content:string, createDirs?:boolean)" in second_messages[0]["content"]


def test_chat_tools_honor_explicit_disable_thinking(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    client = TestClient(create_app(state))
    monkeypatch.setattr(
        openai, "_run_generation", lambda *_args, **_kwargs: _fake_generation("ok")
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Use the tool."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "enable_thinking": False,
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    _messages, kwargs = state.runtime.tokenizer.calls[0]
    assert kwargs["enable_thinking"] is False


def test_streaming_tool_request_emits_raw_reasoning_by_default(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation("<think>raw tool reasoning</think>\nanswer"),
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Use the tool."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 32,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    payloads = _stream_payloads(body)
    reasoning = "".join(
        choice.get("delta", {}).get("reasoning_content", "")
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    content = "".join(
        choice.get("delta", {}).get("content", "")
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    assert reasoning == "raw tool reasoning"
    assert content.strip() == "answer"


def test_chat_template_preserves_assistant_tool_calls_and_tool_results():
    tokenizer = CaptureTokenizer()

    openai._encode_messages(
        tokenizer,
        [
            openai.ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "call_test",
                        "type": "function",
                        "function": {
                            "name": "session_status",
                            "arguments": "{}",
                        },
                    }
                ],
            ),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_test",
                content='{"status":"ok"}',
            ),
        ],
        enable_thinking=False,
        add_generation_prompt=False,
    )

    messages, _kwargs = tokenizer.calls[0]
    assert messages[0]["role"] == "assistant"
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {}
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "call_test"


def test_tool_result_history_encoding_keeps_generation_prompt_prefix():
    tokenizer = QwenToolHistoryBoundaryTokenizer()
    tools = [_tool_schema()]
    initial_messages = [
        openai.ChatMessage(role="system", content="You are opencode."),
        openai.ChatMessage(role="user", content="write hello.py"),
    ]
    tool_call = {
        "id": "call_write",
        "type": "function",
        "function": {
            "name": "session_status",
            "arguments": "{}",
        },
    }
    resumed_messages = [
        *initial_messages,
        openai.ChatMessage(role="assistant", content="", tool_calls=[tool_call]),
        openai.ChatMessage(
            role="tool",
            tool_call_id="call_write",
            content='{"status":"ok"}',
        ),
    ]

    first_prompt = openai._encode_messages(
        tokenizer,
        initial_messages,
        enable_thinking=True,
        tools=tools,
    )
    resumed_prompt = openai._encode_messages(
        tokenizer,
        resumed_messages,
        enable_thinking=True,
        tools=tools,
    )

    assert resumed_prompt[: len(first_prompt)] == first_prompt
    assert tokenizer.merged not in resumed_prompt[: len(first_prompt)]
    rendered = tokenizer.decode(resumed_prompt)
    assert "<function=session_status>" in rendered
    assert "<tool_response>" in rendered


def test_postcommit_tool_history_encoding_matches_next_tool_turn_prefix():
    tokenizer = QwenToolHistoryBoundaryTokenizer()
    tools = [_tool_schema()]
    initial_messages = [
        openai.ChatMessage(role="system", content="You are opencode."),
        openai.ChatMessage(role="user", content="write hello.py"),
    ]
    tool_call = {
        "id": "call_write",
        "type": "function",
        "function": {
            "name": "session_status",
            "arguments": "{}",
        },
    }
    history_messages = [
        *initial_messages,
        openai.ChatMessage(role="assistant", content="", tool_calls=[tool_call]),
    ]
    resumed_messages = [
        *history_messages,
        openai.ChatMessage(
            role="tool",
            tool_call_id="call_write",
            content='{"status":"ok"}',
        ),
    ]

    postcommit_prefix = openai._postcommit_next_turn_prefix_ids(
        tokenizer,
        history_messages,
        enable_thinking=True,
        strip_assistant_reasoning_history=False,
        tools=tools,
        assistant_tool_calls=[tool_call],
    )
    resumed_prompt = openai._encode_messages(
        tokenizer,
        resumed_messages,
        enable_thinking=True,
        tools=tools,
    )

    assert postcommit_prefix is not None
    assert resumed_prompt[: len(postcommit_prefix)] == postcommit_prefix
    assert tokenizer.merged not in postcommit_prefix


def test_chat_tool_xml_returns_openai_tool_calls_nonstream(monkeypatch):
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda *_args, **_kwargs: _fake_generation(
            "<tool_call>\n<function=session_status>\n</function>\n</tool_call>"
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"] == {
        "name": "session_status",
        "arguments": "{}",
    }
    assert "<tool_call>" not in json.dumps(payload)


def test_chat_tool_json_returns_openai_tool_calls_nonstream(monkeypatch):
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda *_args, **_kwargs: _fake_generation(
            '<tool_call>{"name":"session_status","arguments":{}}</tool_call>'
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_chat_stream_tool_calls_emit_delta_tool_calls(monkeypatch):
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(
            "<tool_call>\n<function=session_status>\n</function>\n</tool_call>"
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    assert '"tool_calls"' in response.text
    assert '"finish_reason": "tool_calls"' in response.text
    assert "<tool_call>" not in response.text


def test_chat_stream_consecutive_qwen_xml_tool_calls_emit_ordered_deltas(monkeypatch):
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    generated = (
        "<tool_call>\n<function=write>\n<parameter=filePath>\n"
        "one.py\n</parameter>\n<parameter=content>\nprint(1)\n</parameter>\n"
        "</function>\n</tool_call>\n\n"
        "<tool_call>\n<function=write>\n<parameter=filePath>\n"
        "two.py\n</parameter>\n<parameter=content>\nprint(2)\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=write>\n<parameter=filePath>\n"
        "three.py\n</parameter>\n<parameter=content>\nprint(3)\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    monkeypatch.setattr(openai, "_run_generation", _fake_streaming_generation(generated))

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Write files."}],
            "tools": [_write_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 256,
        },
    )

    assert response.status_code == 200
    assert "<tool_call" not in response.text
    assert "_call>" not in response.text
    assert "<function=" not in response.text
    payloads = _stream_payloads(response.text)
    tool_deltas = [
        item
        for payload in payloads
        for item in payload["choices"][0]["delta"].get("tool_calls", [])
    ]
    name_deltas = [
        item
        for item in tool_deltas
        if item.get("function", {}).get("name") == "write"
    ]
    assert [item["index"] for item in name_deltas] == [0, 1, 2]
    assert any(
        payload["choices"][0].get("finish_reason") == "tool_calls"
        for payload in payloads
    )


def test_chat_stream_qwen_xml_shell_arguments_are_typed(monkeypatch):
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    generated = (
        "<tool_call>\n<function=bash>\n"
        "<parameter=command>\nnpm run build\n</parameter>\n"
        "<parameter=description>\nBuild the project\n</parameter>\n"
        "<parameter=timeout>\n60000\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    monkeypatch.setattr(openai, "_run_generation", _fake_streaming_generation(generated))

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Build."}],
            "tools": [_bash_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 256,
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    args_text = "".join(
        item.get("function", {}).get("arguments", "")
        for payload in payloads
        for item in payload["choices"][0]["delta"].get("tool_calls", [])
    )
    args = json.loads(args_text)
    assert args == {
        "command": "npm run build",
        "description": "Build the project",
        "timeout": 60000,
    }
    assert isinstance(args["timeout"], int)
    assert '"timeout":"60000"' not in args_text
    assert any(
        payload["choices"][0].get("finish_reason") == "tool_calls"
        for payload in payloads
    )


def test_chat_stream_qwen_xml_questions_arguments_are_arrays(monkeypatch):
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    questions = (
        '[{"header":"Scope","id":"scope","question":"Pick one",'
        '"options":[{"label":"A","description":"first"},'
        '{"label":"B","description":"second"}]}]'
    )
    generated = (
        "<tool_call>\n<function=question>\n"
        f"<parameter=questions>\n{questions}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    monkeypatch.setattr(openai, "_run_generation", _fake_streaming_generation(generated))

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Ask."}],
            "tools": [_question_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 256,
        },
    )

    assert response.status_code == 200
    args_text = "".join(
        item.get("function", {}).get("arguments", "")
        for payload in _stream_payloads(response.text)
        for item in payload["choices"][0]["delta"].get("tool_calls", [])
    )
    args = json.loads(args_text)
    assert isinstance(args["questions"], list)
    assert args["questions"][0]["options"][1]["description"] == "second"


def test_chat_stream_tool_call_preamble_is_stored_for_postcommit(monkeypatch):
    state = _fake_streaming_session_state()
    state.args.stream_interval = 1
    captured_generation_final: list[dict] = []
    scheduled: list[dict] = []

    def fake_store_generation_final(*_args, **kwargs):
        captured_generation_final.append(kwargs)
        return {
            "stored": False,
            "mode": "unsafe",
            "reason": "tool_call_history_rewrite",
        }

    def fake_schedule(*_args, **kwargs):
        scheduled.append(kwargs)
        return {
            "stored": False,
            "mode": "async_pending",
            "reason": kwargs["unsafe_reason"],
        }

    monkeypatch.setattr(
        openai, "_store_generation_final_history_snapshot", fake_store_generation_final
    )
    monkeypatch.setattr(openai, "_schedule_idle_postcommit_snapshot", fake_schedule)
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(
            "Let me research first.\n\n"
            "<tool_call>\n<function=session_status>\n</function>\n</tool_call>"
        ),
    )

    with TestClient(create_app(state)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "stream-tool-preamble"},
            json={
                "messages": [{"role": "user", "content": "Status."}],
                "tools": [_tool_schema()],
                "tool_choice": "auto",
                "stream": True,
                "max_tokens": 64,
                "enable_thinking": False,
            },
        )

    assert response.status_code == 200
    assert '"tool_calls"' in response.text
    assert '"finish_reason": "tool_calls"' in response.text
    streamed_content = "".join(
        payload["choices"][0]["delta"]["content"]
        for payload in _stream_payloads(response.text)
        if payload["choices"][0]["delta"].get("content")
    )
    assert streamed_content == "Let me research first.\n\n"
    assert "<tool_call>" not in response.text

    assert captured_generation_final
    assert scheduled
    for call in (captured_generation_final[0], scheduled[0]):
        assert call["assistant_content"] == "Let me research first."
        assert "<tool_call>" not in call["assistant_content"]
        assert call["assistant_tool_calls"][0]["function"] == {
            "name": "session_status",
            "arguments": "{}",
        }


def test_chat_stream_tool_call_postcommit_strips_reasoning_content(monkeypatch):
    state = _fake_streaming_session_state()
    state.args.stream_interval = 1
    captured_generation_final: list[dict] = []
    scheduled: list[dict] = []

    def fake_store_generation_final(*_args, **kwargs):
        captured_generation_final.append(kwargs)
        return {
            "stored": False,
            "mode": "unsafe",
            "reason": "tool_call_history_rewrite",
        }

    def fake_schedule(*_args, **kwargs):
        scheduled.append(kwargs)
        return {
            "stored": False,
            "mode": "async_pending",
            "reason": kwargs["unsafe_reason"],
        }

    monkeypatch.setattr(
        openai, "_store_generation_final_history_snapshot", fake_store_generation_final
    )
    monkeypatch.setattr(openai, "_schedule_idle_postcommit_snapshot", fake_schedule)
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(
            "<think>raw private plan</think>\n"
            "Let me check.\n"
            "<tool_call>\n<function=session_status>\n</function>\n</tool_call>"
        ),
    )

    with TestClient(create_app(state)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "stream-tool-reasoning"},
            json={
                "messages": [{"role": "user", "content": "Status."}],
                "tools": [_tool_schema()],
                "tool_choice": "auto",
                "stream": True,
                "max_tokens": 64,
            },
        )

    assert response.status_code == 200
    reasoning = "".join(
        payload["choices"][0]["delta"].get("reasoning_content", "")
        for payload in _stream_payloads(response.text)
        for _choice in [payload["choices"][0]]
    )
    assert reasoning == "raw private plan"
    assert captured_generation_final
    assert scheduled
    for call in (captured_generation_final[0], scheduled[0]):
        assert call["assistant_content"] == "Let me check."
        assert "raw private plan" not in call["assistant_content"]
        assert "<think>" not in call["assistant_content"]


def test_chat_stream_tools_plain_content_stays_incremental(monkeypatch):
    state = _fake_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation("Count: 1, 2, 3.\n"),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Count."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 16,
            "enable_thinking": False,
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    content_deltas = [
        payload["choices"][0]["delta"]["content"]
        for payload in payloads
        if payload["choices"][0]["delta"].get("content")
    ]
    assert len(content_deltas) > 1
    assert "".join(content_deltas) == "Count: 1, 2, 3.\n"
    assert not any(
        payload["choices"][0]["delta"].get("tool_calls") for payload in payloads
    )
    assert any(
        payload["choices"][0].get("finish_reason") == "stop" for payload in payloads
    )


def test_chat_stream_tool_call_arguments_are_incremental(monkeypatch):
    state = _fake_state()
    state.args.stream_interval = 1
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(
            '<tool_call>{"name":"add","arguments":{"a":25,"b":17}}</tool_call>'
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Use add."}],
            "tools": [_add_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    tool_deltas = [
        payload["choices"][0]["delta"]["tool_calls"][0]
        for payload in payloads
        if payload["choices"][0]["delta"].get("tool_calls")
    ]
    assert tool_deltas[0]["id"].startswith("call_")
    assert tool_deltas[0]["type"] == "function"
    assert tool_deltas[0]["function"] == {"name": "add", "arguments": ""}

    argument_fragments = [
        delta["function"]["arguments"]
        for delta in tool_deltas
        if delta.get("function", {}).get("arguments") is not None
    ]
    assert "".join(argument_fragments) == '{"a":25,"b":17}'
    assert len([fragment for fragment in argument_fragments if fragment]) > 1
    assert '{"a":25,"b":17}' not in argument_fragments


def test_chat_stream_emits_heartbeat_during_alive_silence(monkeypatch):
    state = _fake_state()
    state.args.stats_footer = False
    state.generation_executor = ThreadPoolExecutor(max_workers=1)
    client = TestClient(create_app(state))
    tokens = [ord("o"), ord("k"), ord("\n")]

    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(openai, "STREAM_HEARTBEAT_INTERVAL_S", 0.0)
    monkeypatch.setattr(openai, "STREAM_SILENCE_WARN_S", 0.01)
    monkeypatch.setattr(openai, "STREAM_SILENCE_WARN_INTERVAL_S", 60.0)

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        # Leave enough wall time for the ASGI stream loop to observe an empty
        # token queue even on slower GitHub macOS runners.
        time.sleep(1.25)
        kwargs["token_callback"](tokens)
        return {
            "text": "ok\n",
            "tokens": tokens,
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
            },
            "prompt_tokens": 3,
            "completion_tokens": len(tokens),
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    try:
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-cache-mode": "bypass"},
            json={
                "messages": [{"role": "user", "content": "Say ok."}],
                "stream": True,
                "max_tokens": 16,
                "enable_thinking": False,
            },
        )
    finally:
        state.generation_executor.shutdown(wait=True)

    assert response.status_code == 200
    payloads = []
    for event in response.text.split("\n\n"):
        if not event.startswith("data: {"):
            continue
        payloads.append(json.loads(event.removeprefix("data: ")))

    heartbeats = [
        payload
        for payload in payloads
        if payload.get("mtplx_progress", {}).get("heartbeat") is True
    ]
    content = "".join(
        payload["choices"][0].get("delta", {}).get("content", "")
        for payload in payloads
    )
    final_chunks = [
        payload
        for payload in payloads
        if payload["choices"][0].get("finish_reason") == "stop"
    ]

    assert heartbeats
    assert heartbeats[0]["mtplx_progress"]["phase"] == "generating"
    assert heartbeats[0]["mtplx_progress"]["completion_tokens"] == 0
    assert content == "ok\n"
    assert final_chunks
    assert "data: [DONE]" in response.text


def test_chat_tools_malformed_tool_call_falls_back_to_content(monkeypatch):
    state = _fake_state()
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda *_args, **_kwargs: _fake_generation("<tool_call>not json</tool_call>"),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "<tool_call>not json</tool_call>"
    assert "tool_calls" not in choice["message"]
    stats = response.json()["mtplx_stats"]
    assert stats["tool_parse_fallback"] is True
    assert stats["tool_parse_fallback_kind"] == "malformed_tool_call"
    assert state.tool_parse_counters["malformed_tool_call"] == 1
    assert state.tool_parse_counters["tool_parse_fallback"] == 1


def test_chat_tools_unknown_generated_tool_falls_back_to_content(monkeypatch):
    state = _fake_state()
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    text = (
        "<tool_call>\n<function=Agent>\n"
        "<parameter=description>\nList files\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda *_args, **_kwargs: _fake_generation(text),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == text
    assert "tool_calls" not in choice["message"]
    stats = response.json()["mtplx_stats"]
    assert stats["tool_parse_fallback"] is True
    assert stats["tool_parse_fallback_kind"] == "unknown_tool_name"
    assert "unknown tool 'Agent'" in stats["tool_parse_fallback_reason"]


def test_chat_stream_unknown_generated_tool_falls_back_to_content(monkeypatch):
    state = _fake_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    text = (
        "<tool_call>\n<function=task>\n"
        "<parameter=description>\nList files\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    monkeypatch.setattr(openai, "_run_generation", _fake_streaming_generation(text))

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    streamed_content = "".join(
        payload["choices"][0]["delta"].get("content", "") for payload in payloads
    )
    assert streamed_content == text
    assert not any(
        payload["choices"][0]["delta"].get("tool_calls") for payload in payloads
    )
    final = [payload for payload in payloads if payload["choices"][0]["finish_reason"]]
    assert final[-1]["choices"][0]["finish_reason"] == "stop"
    stats = final[-1]["mtplx_stats"]
    assert stats["tool_parse_fallback"] is True
    assert stats["tool_parse_fallback_kind"] == "unknown_tool_name"
    assert "data: [DONE]" in response.text


def test_chat_stream_unclosed_tool_call_falls_back_and_finishes(monkeypatch):
    state = _fake_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    text = "<tool_call>\n<function=session_status>\n"
    monkeypatch.setattr(openai, "_run_generation", _fake_streaming_generation(text))

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    streamed_content = "".join(
        payload["choices"][0]["delta"].get("content", "") for payload in payloads
    )
    assert streamed_content == text
    assert not any(
        payload["choices"][0]["delta"].get("tool_calls") for payload in payloads
    )
    final = [payload for payload in payloads if payload["choices"][0]["finish_reason"]]
    assert final[-1]["choices"][0]["finish_reason"] == "stop"
    assert final[-1]["mtplx_stats"]["tool_parse_fallback_kind"] == "unclosed_tool_call"
    assert "data: [DONE]" in response.text


def test_server_state_emits_startup_progress(monkeypatch, capsys):
    monkeypatch.setattr(openai, "apply_profile_env", lambda _profile: None)
    monkeypatch.setattr(openai, "profile_env_status", lambda _profile: {})
    monkeypatch.setattr(openai, "_fast_path_env_status", lambda: {})
    monkeypatch.setattr(openai, "_mlx_fork_status", lambda: {"ok": True})
    monkeypatch.setattr(
        openai, "_configure_mlx_cache_limit", lambda _args: {"configured": False}
    )
    monkeypatch.setattr(
        openai,
        "load",
        lambda model, mtp, contract: SimpleNamespace(
            model_path=Path(model),
            mtp_enabled=mtp,
            tokenizer=SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        openai, "_install_draft_lm_head", lambda *_args, **_kwargs: {"installed": True}
    )
    monkeypatch.setattr(openai, "_draft_head_identity", lambda _runtime: "draft-head")
    monkeypatch.setattr(openai, "_template_hash", lambda _tokenizer: "template")
    monkeypatch.setattr(
        openai, "_resolve_context_window", lambda _tokenizer, _model: 32768
    )
    monkeypatch.setattr(openai, "EngineSessionManager", lambda: SimpleNamespace())

    args = parse_args(["--model", "models/example", "--warmup-tokens", "0"])
    state = openai.ServerState(args)

    captured = capsys.readouterr().out
    assert "[4/6] Preparing Sustained MTP runtime" in captured
    assert "[5/6] Loading model weights: models/example" in captured
    assert "This is the long step" in captured
    assert "Model load in progress (this may take a minute)" in captured
    assert "[5/6] Model loaded" in captured
    assert "[6/6] Warmup skipped" in captured
    assert state.context_window == 32768


def test_server_state_reports_model_load_failure(monkeypatch, capsys):
    monkeypatch.setattr(openai, "apply_profile_env", lambda _profile: None)
    monkeypatch.setattr(openai, "profile_env_status", lambda _profile: {})
    monkeypatch.setattr(openai, "_fast_path_env_status", lambda: {})
    monkeypatch.setattr(openai, "_mlx_fork_status", lambda: {"ok": True})
    monkeypatch.setattr(
        openai, "_configure_mlx_cache_limit", lambda _args: {"configured": False}
    )

    def fail_load(model, mtp, contract):
        assert model == "models/example"
        assert mtp is True
        assert contract is not None
        raise RuntimeError("boom")

    monkeypatch.setattr(openai, "load", fail_load)

    args = parse_args(["--model", "models/example", "--warmup-tokens", "0"])
    with pytest.raises(RuntimeError, match="boom"):
        openai.ServerState(args)

    captured = capsys.readouterr().out
    assert "[5/6] Model load failed" in captured
    assert "RuntimeError: boom" in captured
