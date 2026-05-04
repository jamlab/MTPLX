from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.profiles import get_profile
from mtplx.server import openai
from mtplx.server.openai import _RateLimiter, create_app, parse_args


class FakeExecutor:
    def shutdown(self, **_kwargs):
        return None


def _fake_state(*, api_key: str | None = None, rate_limit: int = 0):
    argv = ["--warmup-tokens", "0", "--rate-limit", str(rate_limit)]
    if api_key:
        argv.extend(["--api-key", api_key])
    args = parse_args(argv)
    return SimpleNamespace(
        args=args,
        model_id="mtplx-test-model",
        runtime=SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True),
        profile=get_profile(args.profile),
        context_window=4096,
        load_time_s=0.25,
        draft_lm_head={"installed": False, "reason": "test"},
        draft_head_identity="test-head",
        template_hash="test-template",
        fast_path_env_status={},
        profile_env_status={},
        mlx_cache_limit_status={"configured": False},
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
    assert 'id="ctl-depth" type="range"' in root.text
    assert 'id="ctl-max-tokens" type="range"' in root.text
    assert 'id="ctl-system"' in root.text
    assert 'id="reset-defaults"' in root.text
    # New layout: avatar circles + reasoning-as-its-own-block + turn-* classes
    assert "turn turn-assistant" in root.text
    assert 'class="avatar"' in root.text
    assert "reasoning-block" in root.text
    # Auto-scroll, stop, new-chat, persistence
    assert 'id="jump-pill"' in root.text
    assert 'id="new-chat-btn"' in root.text
    assert "AbortController" in root.text
    # SETTINGS_KEY bumped to v3 when context-window auto-detect landed; bumping
    # the version invalidates stale saved settings that could be wedged in
    # corrupted state (one of the causes of the user-reported "stuck on
    # Thinking" repro after a max_tokens slider change).
    assert "mtplx.chat.settings.v3" in root.text
    # Auto-detect of context length must be hooked up so the slider isn't
    # capped at a stale 32k for a 256k-context model.
    assert "discoverServerLimits" in root.text
    assert "/health" in root.text
    # Stall watchdog so the UI surfaces a real error instead of parking on
    # "Thinking" forever when the server hangs (also user-reported).
    assert "armStallWatchdog" in root.text
    assert "no response from server" in root.text
    # Markdown via marked.js
    assert "marked.min.js" in root.text
    # Live tps element
    assert 'id="live-stats"' in root.text
    assert "tok/s" in root.text
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
    assert client.get("/v1/models", headers={"Authorization": "Bearer test-key"}).status_code == 200

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
    assert response.json()["detail"] == "messages must not be empty"


def test_server_state_emits_startup_progress(monkeypatch, capsys):
    monkeypatch.setattr(openai, "apply_profile_env", lambda _profile: None)
    monkeypatch.setattr(openai, "profile_env_status", lambda _profile: {})
    monkeypatch.setattr(openai, "_fast_path_env_status", lambda: {})
    monkeypatch.setattr(openai, "_mlx_fork_status", lambda: {"ok": True})
    monkeypatch.setattr(openai, "_configure_mlx_cache_limit", lambda _args: {"configured": False})
    monkeypatch.setattr(
        openai,
        "load",
        lambda model, mtp, contract: SimpleNamespace(
            model_path=Path(model),
            mtp_enabled=mtp,
            tokenizer=SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(openai, "_install_draft_lm_head", lambda *_args, **_kwargs: {"installed": True})
    monkeypatch.setattr(openai, "_draft_head_identity", lambda _runtime: "draft-head")
    monkeypatch.setattr(openai, "_template_hash", lambda _tokenizer: "template")
    monkeypatch.setattr(openai, "_resolve_context_window", lambda _tokenizer, _model: 32768)
    monkeypatch.setattr(openai, "EngineSessionManager", lambda: SimpleNamespace())

    args = parse_args(["--model", "models/example", "--warmup-tokens", "0"])
    state = openai.ServerState(args)

    captured = capsys.readouterr().out
    assert "[4/6] Preparing Medium MTP runtime" in captured
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
    monkeypatch.setattr(openai, "_configure_mlx_cache_limit", lambda _args: {"configured": False})

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
