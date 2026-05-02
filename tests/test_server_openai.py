from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.profiles import get_profile
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

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model"] == "mtplx-test-model"
    assert health.json()["api_key_required"] is False
    assert health.json()["warmup"]["ran"] is False

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
