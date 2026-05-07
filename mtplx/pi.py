"""Pi coding-agent integration helpers.

The public CLI uses this module to make ``mtplx start pi`` a real connection
flow: merge an MTPLX provider into Pi's ``models.json`` and then start the
OpenAI-compatible MTPLX server with matching settings.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any

PI_PROVIDER_ID = "mtplx"
PI_LOCAL_API_KEY = "mtplx-local"
PI_DEFAULT_CONTEXT_WINDOW = 131_072
PI_DEFAULT_MAX_TOKENS = 4096


def pi_models_json_path(path: str | Path | None = None) -> Path:
    """Return Pi's custom models config path.

    ``MTPLX_PI_MODELS_JSON`` exists only for tests and power-user overrides.
    Normal users get Pi's documented ``~/.pi/agent/models.json`` path.
    """

    if path is not None:
        return Path(path).expanduser()
    env = os.environ.get("MTPLX_PI_MODELS_JSON")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".pi" / "agent" / "models.json"


def pi_model_ref(model_id: str, *, provider_id: str = PI_PROVIDER_ID) -> str:
    return f"{provider_id}/{model_id}"


def pi_launch_command(model_id: str, *, provider_id: str = PI_PROVIDER_ID) -> str:
    return f"pi --model {pi_model_ref(model_id, provider_id=provider_id)}"


def build_pi_provider_config(
    *,
    base_url: str,
    model_id: str,
    model_name: str | None = None,
    api_key: str = PI_LOCAL_API_KEY,
    context_window: int = PI_DEFAULT_CONTEXT_WINDOW,
    max_tokens: int = PI_DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Build the Pi provider block MTPLX needs.

    Pi's OpenAI-compatible transport currently needs the Chat Completions API
    name, a dummy-or-real API key, and compatibility flags so it sends
    ``system`` instead of ``developer`` and ``max_tokens`` instead of the newer
    OpenAI field.
    """

    return {
        "baseUrl": str(base_url).rstrip("/"),
        "api": "openai-completions",
        "apiKey": str(api_key),
        "authHeader": True,
        "compat": {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
            "maxTokensField": "max_tokens",
        },
        "models": [
            {
                "id": str(model_id),
                "name": model_name or f"MTPLX {model_id}",
                "reasoning": False,
                "input": ["text"],
                "contextWindow": int(context_window),
                "maxTokens": int(max_tokens),
                "cost": {
                    "input": 0,
                    "output": 0,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                },
            }
        ],
    }


def _backup_invalid_config(path: Path) -> Path:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.invalid-{stamp}.bak")
    counter = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.invalid-{stamp}-{counter}.bak")
        counter += 1
    path.replace(backup)
    return backup


def merge_pi_models_config(
    existing: dict[str, Any] | None,
    *,
    provider_config: dict[str, Any],
    provider_id: str = PI_PROVIDER_ID,
) -> dict[str, Any]:
    """Merge or create a Pi ``models.json`` payload.

    MTPLX owns only the ``providers.mtplx`` block. Existing user providers are
    preserved byte-for-byte at the JSON object level.
    """

    payload = dict(existing or {})
    providers = payload.get("providers")
    if not isinstance(providers, dict):
        providers = {}
    else:
        providers = dict(providers)
    providers[str(provider_id)] = provider_config
    payload["providers"] = providers
    return payload


def write_pi_models_config(
    *,
    base_url: str,
    model_id: str,
    model_name: str | None = None,
    api_key: str = PI_LOCAL_API_KEY,
    path: str | Path | None = None,
    provider_id: str = PI_PROVIDER_ID,
    context_window: int = PI_DEFAULT_CONTEXT_WINDOW,
    max_tokens: int = PI_DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Write the MTPLX provider into Pi's config and return a handoff payload."""

    config_path = pi_models_json_path(path)
    backup_path: Path | None = None
    existing: dict[str, Any] | None = None
    if config_path.exists():
        try:
            parsed = json.loads(config_path.read_text(encoding="utf-8"))
            existing = parsed if isinstance(parsed, dict) else {}
        except (OSError, json.JSONDecodeError):
            backup_path = _backup_invalid_config(config_path)
            existing = {}

    provider_config = build_pi_provider_config(
        base_url=base_url,
        model_id=model_id,
        model_name=model_name,
        api_key=api_key,
        context_window=context_window,
        max_tokens=max_tokens,
    )
    merged = merge_pi_models_config(
        existing,
        provider_config=provider_config,
        provider_id=provider_id,
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    try:
        config_path.chmod(0o600)
    except OSError:
        pass
    return {
        "config_path": str(config_path),
        "backup_path": str(backup_path) if backup_path is not None else None,
        "provider_id": provider_id,
        "base_url": provider_config["baseUrl"],
        "model_id": model_id,
        "model_ref": pi_model_ref(model_id, provider_id=provider_id),
        "launch_command": pi_launch_command(model_id, provider_id=provider_id),
        "api_key": api_key,
        "context_window": int(context_window),
        "max_tokens": int(max_tokens),
        "written": True,
    }
