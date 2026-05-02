#!/usr/bin/env python3
"""Serve MTPLX through a minimal OpenAI-compatible API.

This is intentionally small: it exists so local UI clients such as Open WebUI
can exercise the native-MTP runtime without turning MTPLX into a deployment
server yet. The generation path is serialized because the current cache and
GraphBank machinery has not been audited for concurrent requests.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import os
import re
import secrets
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from queue import Queue
from threading import Event, Lock
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from mtplx.adaptive import AdaptiveDepthPolicy, ExpectedValueDepthPolicy
from mtplx.cache_state import snapshot_cache
from mtplx.mtp_patch import MTPContract
from mtplx.sampling import SamplerConfig
from mtplx.profiles import (
    DEFAULT_PROFILE_NAME,
    PROFILE_CHOICES,
    apply_profile_env,
    get_profile,
    profile_env_status,
)
from mtplx.draft_lm_head import _install_draft_lm_head

try:
    from mtplx.generation import generate_ar, generate_mtpk, restore_or_prefill_prompt_state
    from mtplx.native_mlp import native_mlp_stats
    from mtplx.engine_session import (
        EngineSessionBusy,
        EngineSessionManager,
        hash_text,
        is_background_request,
        system_prompt_hash,
    )
    from mtplx.runtime import load
    from mtplx.session_bank import CacheMissReason
    _RUNTIME_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    _RUNTIME_IMPORT_ERROR = exc

    def _missing_runtime(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"MTPLX runtime dependencies are unavailable: {_RUNTIME_IMPORT_ERROR}") from _RUNTIME_IMPORT_ERROR

    generate_ar = _missing_runtime
    generate_mtpk = _missing_runtime
    restore_or_prefill_prompt_state = _missing_runtime
    load = _missing_runtime

    def native_mlp_stats() -> dict[str, Any]:
        return {"available": False, "error": repr(_RUNTIME_IMPORT_ERROR)}

    class EngineSessionBusy(RuntimeError):
        pass

    class EngineSessionManager:
        def __init__(self) -> None:
            _missing_runtime()

    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def is_background_request(*_args: Any, **_kwargs: Any) -> bool:
        return False

    def system_prompt_hash(messages: list[Any]) -> str | None:
        for message in messages:
            role = getattr(message, "role", None)
            if role == "system":
                return hash_text(_content_to_text(getattr(message, "content", "")))
        return None

    class CacheMissReason(Enum):
        NEW_SESSION = "new_session"
        SESSION_BUSY = "session_busy"
        BACKGROUND_BYPASS = "background_bypass"


FAST_PATH_ENV = {
    "MTPLX_LAZY_VERIFY_LOGITS": "1",
    "MTPLX_BATCH_TARGET_ARRAYS": "1",
    "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
    "MTPLX_DROP_EVENTS": "1",
    "MTPLX_SKIP_VERIFY_SNAPSHOT": "1",
}
EXPECTED_MLX_QMV_FORK_COMMIT = "2377a99f"
EXPECTED_MLX_QMV_FORK_FRAGMENT = "mlx-mtplx-0.31.2-qmm"
STATS_FOOTER_MARKER = "\n---\n⚡ **MTPLX TPS:**"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
_REASONING_DETAILS_RE = re.compile(
    r"<details\b(?=[^>]*\btype=[\"']reasoning[\"'])[^>]*>.*?</details>",
    re.IGNORECASE | re.DOTALL,
)
_REASONING_DETAILS_UNCLOSED_RE = re.compile(
    r"<details\b(?=[^>]*\btype=[\"']reasoning[\"'])[^>]*>.*\Z",
    re.IGNORECASE | re.DOTALL,
)
_REASONING_TAG_RE = re.compile(
    r"<(think|thinking|reason|reasoning|thought)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)


def _comma_floats(value: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
_BEGIN_END_THOUGHT_RE = re.compile(
    r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>",
    re.IGNORECASE | re.DOTALL,
)
_STATS_FOOTER_RE = re.compile(
    r"\n---\s*\n⚡\s*(?:\*\*)?MTPLX TPS:(?:\*\*)?.*\Z",
    re.IGNORECASE | re.DOTALL,
)


def _fast_path_env_status() -> dict[str, dict[str, Any]]:
    return {
        key: {
            "expected": expected,
            "observed": os.environ.get(key),
            "ok": os.environ.get(key) == expected,
        }
        for key, expected in FAST_PATH_ENV.items()
    }


def _assert_fast_path_env() -> dict[str, dict[str, Any]]:
    status = _fast_path_env_status()
    bad = {key: value for key, value in status.items() if not value["ok"]}
    if bad:
        raise RuntimeError(
            "MTPLX fast-path env is incomplete: "
            + json.dumps(bad, sort_keys=True)
        )
    return status


def _template_hash(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        template = repr(type(tokenizer))
    return hash_text(str(template))


def _array_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    try:
        return np.asarray(value).tobytes()
    except Exception:
        return repr(value).encode("utf-8")


def _draft_head_identity(runtime: Any) -> str | None:
    text_model = getattr(runtime.model, "language_model", runtime.model)
    draft_head = getattr(text_model, "_mtplx_draft_lm_head", None)
    if draft_head is None:
        return None
    h = hashlib.sha256()
    for name in ("weight", "scales", "biases", "bias"):
        if hasattr(draft_head, name):
            h.update(_array_bytes(getattr(draft_head, name)))
    return h.hexdigest()[:16]


def _mlx_fork_status() -> dict[str, Any]:
    try:
        import mlx.core as mx

        path = Path(mx.__file__).resolve()
        version = getattr(mx, "__version__", None)
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}
    commit = None
    for parent in [path.parent, *path.parents]:
        if EXPECTED_MLX_QMV_FORK_FRAGMENT in parent.name or EXPECTED_MLX_QMV_FORK_FRAGMENT in str(parent):
            try:
                commit = subprocess.check_output(
                    ["git", "-C", str(parent), "rev-parse", "--short", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                commit = None
            break
    ok = EXPECTED_MLX_QMV_FORK_FRAGMENT in str(path) and (
        commit in {None, EXPECTED_MLX_QMV_FORK_COMMIT}
    )
    return {
        "ok": ok,
        "path": str(path),
        "version": version,
        "expected_path_fragment": EXPECTED_MLX_QMV_FORK_FRAGMENT,
        "expected_commit": EXPECTED_MLX_QMV_FORK_COMMIT,
        "observed_commit": commit,
    }


def _parse_byte_limit(value: str | int | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", "")
    if not text:
        return None
    multipliers = {
        "b": 1,
        "kb": 1024,
        "kib": 1024,
        "mb": 1024**2,
        "mib": 1024**2,
        "gb": 1024**3,
        "gib": 1024**3,
    }
    for suffix, multiplier in sorted(multipliers.items(), key=lambda item: -len(item[0])):
        if text.endswith(suffix):
            number = text[: -len(suffix)]
            return int(float(number) * multiplier)
    return int(float(text))


def _configure_mlx_cache_limit(args: argparse.Namespace) -> dict[str, Any]:
    raw = args.mlx_cache_limit or os.environ.get("MTPLX_MLX_CACHE_LIMIT")
    requested = _parse_byte_limit(raw)
    if requested is None:
        return {"requested": raw, "configured": False}
    import mlx.core as mx

    old_limit = int(mx.set_cache_limit(int(requested)))
    return {
        "requested": raw,
        "configured": True,
        "limit_bytes": int(requested),
        "previous_limit_bytes": old_limit,
    }


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    enable_thinking: bool | None = None
    stream: bool = False


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    prompt: str | list[int] | list[str] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    stream: bool = False


class AnthropicMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = ""


class AnthropicMessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    max_tokens: int | None = None
    messages: list[AnthropicMessage] = Field(default_factory=list)
    system: Any | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False


class ServerState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model_id = args.model_id
        self.lock = Lock()
        self.foreground_lock = Lock()
        self.foreground_active = 0
        self.rate_limiter = _RateLimiter(args.rate_limit)
        self.profile = get_profile(args.profile)
        if args.generation_mode == "mtp" and not args.load_mtp:
            raise ValueError("--generation-mode mtp requires --load-mtp")
        if args.diagnostic_env_ablation:
            self.profile_env_status = profile_env_status(self.profile.name)
            self.fast_path_env_status = _fast_path_env_status()
        else:
            apply_profile_env(self.profile.name)
            self.profile_env_status = profile_env_status(self.profile.name)
            if args.strict_startup_asserts:
                bad_profile_env = {
                    key: value
                    for key, value in self.profile_env_status.items()
                    if not value["ok"]
                }
                if bad_profile_env:
                    raise RuntimeError(
                        "MTPLX profile env is incomplete: "
                        + json.dumps(bad_profile_env, sort_keys=True)
                    )
            self.fast_path_env_status = _fast_path_env_status()
        self.mlx_fork_status = _mlx_fork_status()
        if (
            args.strict_mlx_fork_assert
            and self.profile.required_mlx_fork_commit
            and not self.mlx_fork_status.get("ok")
        ):
            raise RuntimeError(
                "Patched MLX qmv fork is not active: "
                + json.dumps(self.mlx_fork_status, sort_keys=True)
            )
        self.mlx_cache_limit_status = _configure_mlx_cache_limit(args)
        started = time.perf_counter()
        self.runtime = load(args.model, mtp=bool(args.load_mtp), contract=MTPContract())
        self.draft_lm_head = (
            _install_draft_lm_head(
                self.runtime,
                bits=args.draft_lm_head_bits,
                group_size=args.draft_lm_head_group_size,
                mode=args.draft_lm_head_mode,
            )
            if self.runtime.mtp_enabled
            else {"installed": False, "reason": "mtp_disabled"}
        )
        self.draft_head_identity = _draft_head_identity(self.runtime)
        self.template_hash = _template_hash(self.runtime.tokenizer)
        self.load_time_s = time.perf_counter() - started
        self.context_window = (
            int(args.context_window)
            if int(args.context_window) > 0
            else _resolve_context_window(self.runtime.tokenizer, args.model)
        )
        self.sessions = EngineSessionManager()
        self.last_metrics: list[dict[str, Any]] = []
        self.main_system_prompt_hash: str | None = None
        self.generation_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mtplx-generation",
        )
        self.warmup_status = _run_startup_warmup(self)

    def begin_foreground(self) -> None:
        with self.foreground_lock:
            self.foreground_active += 1

    def end_foreground(self) -> None:
        with self.foreground_lock:
            self.foreground_active = max(0, self.foreground_active - 1)

    def has_foreground(self) -> bool:
        with self.foreground_lock:
            return self.foreground_active > 0


LOCALHOST_BINDS = {"", "127.0.0.1", "::1", "localhost"}


def _is_localhost_bind(host: str | None) -> bool:
    return str(host or "").strip().lower().strip("[]") in LOCALHOST_BINDS


def validate_server_security_args(args: argparse.Namespace) -> None:
    if not _is_localhost_bind(getattr(args, "host", None)) and not getattr(args, "api_key", None):
        raise SystemExit("--api-key is required when --host is not localhost")
    if int(getattr(args, "stream_interval", 1)) < 1:
        raise SystemExit("--stream-interval must be >= 1")
    if int(getattr(args, "rate_limit", 0)) < 0:
        raise SystemExit("--rate-limit must be >= 0")
    if int(getattr(args, "warmup_tokens", 0)) < 0:
        raise SystemExit("--warmup-tokens must be >= 0")


def _request_api_key(request: Request) -> str | None:
    auth_header = request.headers.get("authorization", "")
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() == "bearer" and token:
        return token
    api_key = request.headers.get("x-api-key")
    return api_key or None


def _request_is_authorized(request: Request, configured_api_key: str | None) -> bool:
    if not configured_api_key:
        return True
    candidate = _request_api_key(request)
    return bool(candidate and secrets.compare_digest(candidate, configured_api_key))


def _rate_limit_key(request: Request) -> str:
    api_key = _request_api_key(request)
    if api_key:
        return "key:" + hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    client = request.client.host if request.client is not None else "unknown"
    return f"client:{client}"


class _RateLimiter:
    def __init__(self, requests_per_minute: int | None) -> None:
        self.requests_per_minute = max(0, int(requests_per_minute or 0))
        self._lock = Lock()
        self._events: dict[str, list[float]] = {}

    @property
    def enabled(self) -> bool:
        return self.requests_per_minute > 0

    def check(self, key: str, *, now: float | None = None) -> tuple[bool, int]:
        if not self.enabled:
            return True, 0
        timestamp = time.monotonic() if now is None else float(now)
        window_start = timestamp - 60.0
        with self._lock:
            events = [item for item in self._events.get(key, []) if item > window_start]
            if len(events) >= self.requests_per_minute:
                retry_after = max(1, int(60.0 - (timestamp - events[0])))
                self._events[key] = events
                return False, retry_after
            events.append(timestamp)
            self._events[key] = events
            return True, 0


def _resolve_context_window(tokenizer: Any, model_path: str) -> int:
    candidates: list[int] = []
    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max, int):
        candidates.append(tokenizer_max)

    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except Exception:
            config = {}
        for section in (config, config.get("text_config", {})):
            if not isinstance(section, dict):
                continue
            for key in (
                "max_position_embeddings",
                "model_max_length",
                "max_sequence_length",
                "seq_length",
                "context_length",
            ):
                value = section.get(key)
                if isinstance(value, int):
                    candidates.append(value)

    sane = [value for value in candidates if 0 < value <= 1_000_000]
    return max(sane) if sane else 262_144


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _anthropic_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                block_type = str(item.get("type") or "")
                if block_type == "text" or "text" in item:
                    parts.append(str(item.get("text", "")))
                elif block_type == "tool_result":
                    parts.append(_anthropic_content_to_text(item.get("content")))
                else:
                    parts.append(json.dumps(item, sort_keys=True))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(content, dict):
        if content.get("type") == "text" or "text" in content:
            return str(content.get("text", ""))
        return json.dumps(content, sort_keys=True)
    return str(content)


def _anthropic_to_chat_request(request: AnthropicMessagesRequest) -> ChatCompletionRequest:
    messages: list[ChatMessage] = []
    system_text = _anthropic_content_to_text(request.system).strip()
    if system_text:
        messages.append(ChatMessage(role="system", content=system_text))
    for message in request.messages:
        role = "assistant" if message.role == "assistant" else "user"
        messages.append(
            ChatMessage(
                role=role,
                content=_anthropic_content_to_text(message.content),
            )
        )
    return ChatCompletionRequest(
        model=request.model,
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stream=False,
    )


def _anthropic_payload_from_openai(openai_payload: dict[str, Any]) -> dict[str, Any]:
    choices = openai_payload.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    text = str(message.get("content") or choice.get("text") or "")
    usage = openai_payload.get("usage") or {}
    finish_reason = choice.get("finish_reason") or "stop"
    stop_reason = "max_tokens" if finish_reason == "length" else "end_turn"
    return {
        "id": "msg_" + uuid.uuid4().hex,
        "type": "message",
        "role": "assistant",
        "model": openai_payload.get("model"),
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
        "mtplx_stats": openai_payload.get("mtplx_stats"),
    }


def _strip_stats_footer(text: str) -> str:
    marker_index = text.rfind(STATS_FOOTER_MARKER)
    if marker_index < 0:
        return _STATS_FOOTER_RE.sub("", text).rstrip()
    return text[:marker_index].rstrip()


def _details_block_to_think(match: re.Match[str]) -> str:
    block = match.group(0)
    block = re.sub(r"<summary\b[^>]*>.*?</summary>", "", block, flags=re.IGNORECASE | re.DOTALL)
    block = re.sub(r"</?details\b[^>]*>", "", block, flags=re.IGNORECASE | re.DOTALL)
    block = re.sub(r"<br\s*/?>", "\n", block, flags=re.IGNORECASE)
    block = re.sub(r"<[^>]+>", "", block)
    lines = []
    for line in html.unescape(block).splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            stripped = stripped[1:].lstrip()
        if stripped:
            lines.append(stripped)
        elif lines and lines[-1]:
            lines.append("")
    reasoning = "\n".join(lines).strip()
    if not reasoning:
        return ""
    return f"{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}"


def _normalize_openwebui_reasoning_details(text: str) -> str:
    return _REASONING_DETAILS_RE.sub(_details_block_to_think, text)


def _strip_assistant_history_baggage(text: str) -> str:
    """Remove UI-only metadata before feeding assistant turns back to Qwen."""
    text = _strip_stats_footer(text)
    previous = None
    while previous != text:
        previous = text
        text = _REASONING_DETAILS_RE.sub("", text)
        text = _REASONING_TAG_RE.sub("", text)
        text = _BEGIN_END_THOUGHT_RE.sub("", text)
    text = _REASONING_DETAILS_UNCLOSED_RE.sub("", text)
    return text.strip()


def _encode_messages(
    tokenizer: Any,
    messages: list[ChatMessage],
    *,
    enable_thinking: bool,
    strip_assistant_reasoning_history: bool = False,
    add_generation_prompt: bool = True,
) -> list[int]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not message.role:
            continue
        content = _content_to_text(message.content)
        if message.role == "assistant":
            content = (
                _strip_assistant_history_baggage(content)
                if strip_assistant_reasoning_history
                else _normalize_openwebui_reasoning_details(_strip_stats_footer(content)).strip()
            )
        if content:
            normalized.append({"role": message.role, "content": content})
    if not normalized:
        normalized = [{"role": "user", "content": ""}]
    try:
        return list(
                tokenizer.apply_chat_template(
                    normalized,
                    tokenize=True,
                    add_generation_prompt=add_generation_prompt,
                    enable_thinking=enable_thinking,
                    preserve_thinking=not strip_assistant_reasoning_history,
                )
            )
    except TypeError:
        try:
            return list(
                tokenizer.apply_chat_template(
                    normalized,
                    tokenize=True,
                    add_generation_prompt=add_generation_prompt,
                )
            )
        except Exception:
            pass
    except Exception:
        pass
    prompt = "\n".join(f"{item['role']}: {item['content']}" for item in normalized)
    if add_generation_prompt:
        prompt += "\nassistant:"
    return list(tokenizer.encode(prompt))


def _encode_prompt(tokenizer: Any, prompt: str | list[int] | list[str] | None) -> list[int]:
    if prompt is None:
        return []
    if isinstance(prompt, str):
        return list(tokenizer.encode(prompt))
    if isinstance(prompt, list) and all(isinstance(item, int) for item in prompt):
        return [int(item) for item in prompt]
    if isinstance(prompt, list):
        return list(tokenizer.encode("\n".join(str(item) for item in prompt)))
    return list(tokenizer.encode(str(prompt)))


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return int(value)
    except Exception:
        return str(value)


def _request_extra(model: BaseModel, key: str, default: Any = None) -> Any:
    extra = getattr(model, "model_extra", None) or {}
    return extra.get(key, default)


def _request_metadata(model: BaseModel) -> dict[str, Any]:
    metadata = _request_extra(model, "metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _token_window_rate(token_times: list[float], window: int) -> float | None:
    if len(token_times) < 2:
        return None
    subset = token_times[-window:]
    if len(subset) < 2:
        return None
    elapsed = subset[-1] - subset[0]
    if elapsed <= 0:
        return None
    return (len(subset) - 1) / elapsed


def _token_window_rate_first(token_times: list[float], window: int) -> float | None:
    if len(token_times) < 2:
        return None
    subset = token_times[:window]
    if len(subset) < 2:
        return None
    elapsed = subset[-1] - subset[0]
    if elapsed <= 0:
        return None
    return (len(subset) - 1) / elapsed


def _metrics_envelope(
    *,
    stats: dict[str, Any],
    prompt_tokens: int,
    completion_tokens: int,
    request_elapsed_s: float,
    token_times: list[float],
    request_started_s: float,
    lock_wait_time_s: float,
    session_id: str | None,
    session_cache_hit: bool,
    cache_miss_reason: str | None,
    session_restore_mode: str,
    mtp_depth: int,
    generation_limits: dict[str, Any],
) -> dict[str, Any]:
    decode_tok_s, decode_elapsed_s = _decode_timing(stats)
    prompt_eval_time_s = float(stats.get("prompt_eval_time_s") or 0.0)
    ttft_s = (
        max(0.0, token_times[0] - request_started_s)
        if token_times
        else None
    )
    cached_tokens = int(stats.get("cached_tokens") or 0)
    new_prefill_tokens = int(stats.get("new_prefill_tokens") or (prompt_tokens - cached_tokens))
    return {
        "prompt_tokens": int(prompt_tokens),
        "cached_tokens": cached_tokens,
        "new_prefill_tokens": max(0, new_prefill_tokens),
        "completion_tokens": int(completion_tokens),
        "prompt_eval_time_s": prompt_eval_time_s,
        "ttft_s": ttft_s,
        "decode_elapsed_s": decode_elapsed_s,
        "request_elapsed_s": request_elapsed_s,
        "request_tok_s": completion_tokens / request_elapsed_s if request_elapsed_s > 0 else 0.0,
        "decode_tok_s": decode_tok_s,
        "sliding_decode_tok_s_first_32": _token_window_rate_first(token_times, 32),
        "sliding_decode_tok_s_first_64": _token_window_rate_first(token_times, 64),
        "sliding_decode_tok_s_last_32": _token_window_rate(token_times, 32),
        "sliding_decode_tok_s_last_64": _token_window_rate(token_times, 64),
        "mtp_depth": int(mtp_depth),
        "verify_calls": int(stats.get("verify_calls") or 0),
        "accepted_by_depth": stats.get("accepted_by_depth") or [],
        "correction_tokens": int(stats.get("correction_tokens") or 0),
        "bonus_tokens": int(stats.get("bonus_tokens") or 0),
        "verify_time_s": float(stats.get("verify_time_s") or 0.0),
        "draft_time_s": float(stats.get("draft_time_s") or 0.0),
        "accept_time_s": float(stats.get("accept_time_s") or 0.0),
        "repair_time_s": float(stats.get("repair_time_s") or 0.0),
        "session_cache_hit": bool(session_cache_hit),
        "cache_miss_reason": cache_miss_reason,
        "session_restore_mode": session_restore_mode,
        "context_len": int(prompt_tokens + completion_tokens),
        "lock_wait_time_s": lock_wait_time_s,
        "session_id": session_id,
        **generation_limits,
    }


def _effective_completion_tokens(
    *,
    generated_tokens: list[int],
    streamed_token_times: list[float],
) -> int:
    return max(len(generated_tokens), len(streamed_token_times))


def _repair_streamed_generation_stats(
    stats: dict[str, Any],
    *,
    completion_tokens: int,
    elapsed_s: float,
) -> dict[str, Any]:
    repaired = dict(stats)
    raw_generated = int(repaired.get("generated_tokens") or 0)
    if completion_tokens > raw_generated:
        repaired["generated_tokens_raw"] = raw_generated
        repaired["generated_tokens_recovered_from_stream"] = True
        repaired["generated_tokens"] = int(completion_tokens)
        repaired["tok_s"] = (
            float(completion_tokens) / elapsed_s if elapsed_s > 0 else 0.0
        )
    return repaired


def _request_observability(
    request: ChatCompletionRequest,
    *,
    headers: dict[str, str],
    metadata: dict[str, Any],
    session_source: str | None,
) -> dict[str, Any]:
    user_texts = [
        _content_to_text(message.content)
        for message in request.messages
        if message.role == "user"
    ]
    candidate_headers = {
        key: value
        for key, value in headers.items()
        if key.lower()
        in {
            "x-mtplx-session-id",
            "x-openwebui-chat-id",
            "x-openwebui-user-id",
            "x-openwebui-task",
        }
    }
    return {
        "request_message_count": len(request.messages),
        "request_message_roles": [message.role for message in request.messages],
        "request_message_chars": [
            len(_content_to_text(message.content)) for message in request.messages
        ],
        "request_extra_keys": sorted((getattr(request, "model_extra", None) or {}).keys()),
        "request_metadata_keys": sorted(metadata.keys()),
        "request_session_source": session_source,
        "request_session_candidate_headers": candidate_headers,
        "request_last_user_preview": user_texts[-1][:180] if user_texts else None,
        "request_last_user_chars": len(user_texts[-1]) if user_texts else 0,
    }


def _policy_fingerprint(
    state: ServerState,
    *,
    thinking_enabled: bool,
) -> str:
    adaptive = _adaptive_config(state.args)
    proposal_cache = _proposal_cache_config(state.args)
    online_hidden = _online_hidden_config(state.args)
    return ";".join(
        [
            f"template={state.template_hash}",
            f"thinking={int(bool(thinking_enabled))}",
            f"strip_reasoning={int(bool(state.args.strip_assistant_reasoning_history))}",
            f"generation_mode={getattr(state.args, 'generation_mode', 'mtp')}",
            "hidden_variant=post_norm",
            "mtp_history_policy=committed",
            f"draft_head={state.draft_head_identity}",
            f"adaptive={json.dumps(adaptive, sort_keys=True, separators=(',', ':'))}",
            f"proposal_cache={json.dumps(proposal_cache, sort_keys=True, separators=(',', ':'))}",
            f"online_hidden={json.dumps(online_hidden, sort_keys=True, separators=(',', ':'))}",
        ]
    )


def _proposal_cache_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "online_correction_cache": bool(args.online_correction_cache),
        "online_correction_cache_min_depth": int(args.online_correction_cache_min_depth),
        "online_correction_cache_key": str(args.online_correction_cache_key),
        "prompt_correction_cache": bool(args.prompt_correction_cache),
        "prompt_correction_cache_min_depth": int(args.prompt_correction_cache_min_depth),
    }


def _online_hidden_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "alpha": float(getattr(args, "online_hidden_corrector_alpha", 0.0)),
        "decay": float(getattr(args, "online_hidden_corrector_decay", 0.8)),
        "warmup": int(getattr(args, "online_hidden_corrector_warmup", 1)),
        "max_feed_depth": getattr(args, "online_hidden_corrector_max_feed_depth", None),
        "key": str(getattr(args, "online_hidden_corrector_key", "global")),
    }


def _adaptive_config(args: argparse.Namespace) -> dict[str, Any]:
    policy = str(getattr(args, "adaptive_policy", "none") or "none")
    if policy == "none":
        return {"policy": "none"}
    config: dict[str, Any] = {
        "policy": policy,
        "max_depth": int(args.depth),
        "min_depth": int(args.adaptive_min_depth),
    }
    if policy == "streak":
        config.update(
            {
                "start_depth": int(args.adaptive_start_depth),
                "increase_after": int(args.adaptive_increase_after),
                "decrease_after": int(args.adaptive_decrease_after),
            }
        )
    elif policy == "expected_value":
        config.update(
            {
                "base_depth": int(args.adaptive_ev_base_depth),
                "accept_priors": [float(v) for v in args.adaptive_ev_accept_priors],
                "draft_cost_s": float(args.adaptive_ev_draft_cost_s),
                "extra_verify_cost_s": float(args.adaptive_ev_extra_verify_cost_s),
                "baseline_tok_s": float(args.adaptive_ev_baseline_tok_s),
                "safety_margin": float(args.adaptive_ev_safety_margin),
                "margin_center": float(args.adaptive_ev_margin_center),
                "margin_scale": float(args.adaptive_ev_margin_scale),
                "confidence_weight": float(args.adaptive_ev_confidence_weight),
                "min_extra_accept_probability": float(
                    args.adaptive_ev_min_extra_accept_probability
                ),
            }
        )
    return config


def _make_adaptive_policy(
    args: argparse.Namespace,
) -> AdaptiveDepthPolicy | ExpectedValueDepthPolicy | None:
    policy = str(getattr(args, "adaptive_policy", "none") or "none")
    if policy == "none":
        return None
    if policy == "streak":
        return AdaptiveDepthPolicy(
            max_depth=int(args.depth),
            min_depth=int(args.adaptive_min_depth),
            start_depth=int(args.adaptive_start_depth),
            increase_after=int(args.adaptive_increase_after),
            decrease_after=int(args.adaptive_decrease_after),
        )
    if policy == "expected_value":
        return ExpectedValueDepthPolicy(
            max_depth=int(args.depth),
            min_depth=int(args.adaptive_min_depth),
            base_depth=int(args.adaptive_ev_base_depth),
            accept_priors=tuple(float(v) for v in args.adaptive_ev_accept_priors),
            draft_cost_s=float(args.adaptive_ev_draft_cost_s),
            extra_verify_cost_s=float(args.adaptive_ev_extra_verify_cost_s),
            baseline_tok_s=float(args.adaptive_ev_baseline_tok_s),
            safety_margin=float(args.adaptive_ev_safety_margin),
            margin_center=float(args.adaptive_ev_margin_center),
            margin_scale=float(args.adaptive_ev_margin_scale),
            confidence_weight=float(args.adaptive_ev_confidence_weight),
            min_extra_accept_probability=float(
                args.adaptive_ev_min_extra_accept_probability
            ),
        )
    raise ValueError(f"unknown adaptive policy: {policy}")


def _store_retokenized_history_snapshot(
    state: ServerState,
    *,
    session_id: str | None,
    messages: list[ChatMessage],
    assistant_content: str,
    thinking_enabled: bool,
    policy_fingerprint: str,
) -> dict[str, Any]:
    if session_id is None:
        return {"stored": False, "reason": "no_session_id"}
    history_messages = list(messages) + [
        ChatMessage(role="assistant", content=assistant_content),
    ]
    encoded_with_sentinel = _encode_messages(
        state.runtime.tokenizer,
        history_messages,
        enable_thinking=thinking_enabled,
        strip_assistant_reasoning_history=state.args.strip_assistant_reasoning_history,
        add_generation_prompt=False,
    )
    history_ids = encoded_with_sentinel
    if not history_ids:
        return {"stored": False, "reason": "empty_boundary_prefix"}
    started = time.perf_counter()
    state.begin_foreground()
    state.lock.acquire()
    try:
        prompt_state = restore_or_prefill_prompt_state(
            state.runtime,
            history_ids,
            mtp_hidden_variant="post_norm",
            mtp_history_policy="committed",
        )
        mtp_snapshot = (
            snapshot_cache(prompt_state.committed_mtp_cache)
            if prompt_state.committed_mtp_cache is not None
            else None
        )
        entry = state.sessions.bank.put(
            runtime=state.runtime,
            token_ids=history_ids,
            cache=prompt_state.trunk_cache,
            logits=prompt_state.logits,
            hidden=prompt_state.hidden,
            hidden_variant="post_norm",
            keep_live_ref=True,
            session_id=session_id,
            template_hash=state.template_hash,
            mtp_history_policy="committed",
            draft_head_identity=state.draft_head_identity,
            policy_fingerprint=policy_fingerprint,
            mtp_history_snapshot=mtp_snapshot,
            snapshot_epoch=len(history_ids),
            mtp_snapshot_epoch=len(history_ids) if mtp_snapshot is not None else None,
        )
    finally:
        state.lock.release()
        state.end_foreground()
    return {
        "stored": True,
        "prefix_len": entry.prefix_len,
        "nbytes": entry.nbytes,
        "elapsed_s": time.perf_counter() - started,
        "token_hash": entry.token_hash,
    }


def _generation_params(
    state: ServerState,
    *,
    prompt_token_count: int,
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
) -> tuple[int, SamplerConfig, dict[str, Any]]:
    remaining_context = max(1, int(state.context_window) - int(prompt_token_count))
    request_max_tokens = None if max_tokens is None else int(max_tokens)
    requested_max = remaining_context if request_max_tokens is None else request_max_tokens
    before_server_cap = requested_max
    server_max_response_tokens = state.args.max_response_tokens
    if state.args.max_response_tokens is not None:
        requested_max = min(requested_max, int(state.args.max_response_tokens))
    after_server_cap = requested_max
    requested_max = max(1, min(after_server_cap, remaining_context))
    sampler = SamplerConfig(
        temperature=state.args.temperature if temperature is None else float(temperature),
        top_p=state.args.top_p if top_p is None else float(top_p),
        top_k=state.args.top_k if top_k is None else int(top_k),
    )
    return requested_max, sampler, {
        "request_max_tokens": request_max_tokens,
        "server_max_response_tokens": (
            None if server_max_response_tokens is None else int(server_max_response_tokens)
        ),
        "effective_max_tokens": int(requested_max),
        "remaining_context_tokens": int(remaining_context),
        "server_cap_applied": bool(
            server_max_response_tokens is not None and after_server_cap < before_server_cap
        ),
        "context_cap_applied": bool(requested_max < after_server_cap),
    }


def _fresh_seed() -> int:
    # Keep within signed 31-bit range for downstream RNG compatibility.
    return secrets.randbelow(2**31 - 1)


def _session_bank_restore_mode(mode: str | None) -> str:
    normalized = str(mode or "clone").replace("-", "_")
    if normalized in {"reference", "reference_lease"}:
        return "reference"
    if normalized == "clone":
        return "clone"
    return "clone"


def _resolve_seed(state: ServerState, request_seed: int | None) -> tuple[int, bool]:
    if request_seed is not None:
        return int(request_seed), True
    if state.args.seed is not None:
        return int(state.args.seed), True
    return _fresh_seed(), False


def _run_generation(
    state: ServerState,
    prompt_ids: list[int],
    *,
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
    token_callback: Callable[[list[int]], None] | None = None,
    session_id: str | None = None,
    session_cache_hit: bool = False,
    cache_miss_reason: str | None = CacheMissReason.NEW_SESSION.value,
    session_restore_mode: str = "cold",
    session_bank: Any | None = None,
    session_template_hash: str | None = None,
    session_draft_head_identity: str | None = None,
    session_policy_fingerprint: str | None = None,
    background_request: bool = False,
    commit_final_state_to_bank: bool = True,
    request_observability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response_max, sampler, generation_limits = _generation_params(
        state,
        prompt_token_count=len(prompt_ids),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    started = time.perf_counter()
    token_times: list[float] = []
    lock_wait_time_s = 0.0

    def record_tokens(new_tokens: list[int]) -> None:
        now = time.perf_counter()
        token_times.extend([now for _token in new_tokens])
        if token_callback is not None:
            token_callback(new_tokens)

    blank_retry_budget = max(0, int(state.args.blank_retry_attempts))
    streaming_response = token_callback is not None
    max_attempts = 1 if streaming_response else 1 + blank_retry_budget
    last: dict[str, Any] | None = None
    trace_preview = (
        str((request_observability or {}).get("request_last_user_preview") or "")
        .replace("\n", " ")
        .strip()
    )
    trace_label = f"{session_id or 'stateless'}:{trace_preview[:64]}" if trace_preview else session_id
    trace_metadata = {
        "session_id": session_id,
        "session_restore_mode": session_restore_mode,
        "background_request": bool(background_request),
        "cache_bypass": session_bank is None,
        **(request_observability or {}),
    }
    for attempt in range(max_attempts):
        generation_seed, seed_is_explicit = _resolve_seed(state, seed)
        lock_started = time.perf_counter()
        if background_request:
            if state.has_foreground() or state.lock.locked():
                raise HTTPException(
                    status_code=503,
                    detail="background request bypassed while foreground generation is busy",
                    headers={"Retry-After": "1"},
                )
            acquired = state.lock.acquire(blocking=False)
            if not acquired:
                raise HTTPException(
                    status_code=503,
                    detail="background request bypassed while model lock is busy",
                    headers={"Retry-After": "1"},
                )
        else:
            state.begin_foreground()
            state.lock.acquire()
        lock_wait_time_s += time.perf_counter() - lock_started
        try:
            if state.args.generation_mode == "ar":
                out = generate_ar(
                    state.runtime,
                    prompt_ids,
                    max_tokens=response_max,
                    sampler=sampler,
                    seed=generation_seed,
                    token_callback=record_tokens,
                    trace_label=trace_label,
                    trace_metadata=trace_metadata,
                )
            else:
                adaptive_policy = _make_adaptive_policy(state.args)
                out = generate_mtpk(
                    state.runtime,
                    prompt_ids,
                    max_tokens=response_max,
                    sampler=sampler,
                    speculative_depth=state.args.depth,
                    seed=generation_seed,
                    mtp_hidden_variant="post_norm",
                    mtp_cache_policy="persistent",
                    mtp_history_policy="committed",
                    verify_strategy=state.args.verify_strategy,
                    verify_core=state.args.verify_core,
                    token_callback=record_tokens,
                    session_bank=session_bank,
                    session_restore_mode=_session_bank_restore_mode(session_restore_mode),
                    session_template_hash=session_template_hash,
                    session_draft_head_identity=session_draft_head_identity,
                    session_policy_fingerprint=session_policy_fingerprint,
                    capture_final_state=session_bank is not None,
                    trace_label=trace_label,
                    trace_metadata=trace_metadata,
                    adaptive_policy=adaptive_policy,
                    online_correction_cache=bool(state.args.online_correction_cache),
                    online_correction_cache_min_depth=int(
                        state.args.online_correction_cache_min_depth
                    ),
                    online_correction_cache_key=str(state.args.online_correction_cache_key),
                    prompt_correction_cache=bool(state.args.prompt_correction_cache),
                    prompt_correction_cache_min_depth=int(
                        state.args.prompt_correction_cache_min_depth
                    ),
                    online_hidden_corrector_alpha=float(
                        state.args.online_hidden_corrector_alpha
                    ),
                    online_hidden_corrector_decay=float(
                        state.args.online_hidden_corrector_decay
                    ),
                    online_hidden_corrector_warmup=int(
                        state.args.online_hidden_corrector_warmup
                    ),
                    online_hidden_corrector_max_feed_depth=(
                        state.args.online_hidden_corrector_max_feed_depth
                    ),
                    online_hidden_corrector_key=str(state.args.online_hidden_corrector_key),
                )
        finally:
            state.lock.release()
            if not background_request:
                state.end_foreground()
        elapsed_s = time.perf_counter() - started
        completion_tokens = _effective_completion_tokens(
            generated_tokens=list(out.tokens),
            streamed_token_times=token_times,
        )
        tok_s = completion_tokens / elapsed_s if elapsed_s > 0 else 0.0
        stats = _repair_streamed_generation_stats(
            out.stats.to_dict(),
            completion_tokens=completion_tokens,
            elapsed_s=elapsed_s,
        )
        if session_bank is not None:
            cache_miss_reason = stats.get("cache_miss_reason") or cache_miss_reason
            session_cache_hit = bool(stats.get("session_cache_hit") or False)
            if session_cache_hit:
                cache_miss_reason = None
            session_restore_mode = str(stats.get("session_restore_mode") or session_restore_mode)
        final_state = out.final_state
        if (
            commit_final_state_to_bank
            and
            session_bank is not None
            and session_id is not None
            and final_state is not None
            and final_state.safe_to_commit
        ):
            final_token_ids = list(prompt_ids) + list(out.tokens)
            mtp_snapshot = (
                snapshot_cache(final_state.final_committed_mtp_cache)
                if final_state.final_committed_mtp_cache is not None
                else None
            )
            session_bank.put(
                runtime=state.runtime,
                token_ids=final_token_ids,
                cache=final_state.final_trunk_cache,
                logits=final_state.final_logits,
                hidden=final_state.final_hidden,
                hidden_variant="post_norm",
                keep_live_ref=True,
                session_id=session_id,
                template_hash=session_template_hash,
                mtp_history_policy="committed",
                draft_head_identity=session_draft_head_identity,
                policy_fingerprint=session_policy_fingerprint,
                mtp_history_snapshot=mtp_snapshot,
                snapshot_epoch=len(final_token_ids),
                mtp_snapshot_epoch=len(final_token_ids) if mtp_snapshot is not None else None,
            )
        envelope = _metrics_envelope(
            stats=stats,
            prompt_tokens=len(prompt_ids),
            completion_tokens=completion_tokens,
            request_elapsed_s=elapsed_s,
            token_times=token_times,
            request_started_s=started,
            lock_wait_time_s=lock_wait_time_s,
            session_id=session_id,
            session_cache_hit=session_cache_hit,
            cache_miss_reason=cache_miss_reason,
            session_restore_mode=session_restore_mode,
            mtp_depth=state.args.depth if state.args.generation_mode == "mtp" else 0,
            generation_limits=generation_limits,
        )
        if request_observability:
            envelope.update(request_observability)
        stats["generation_mode"] = state.args.generation_mode
        stats.update(envelope)
        stats["server_elapsed_s"] = elapsed_s
        stats["server_tok_s"] = tok_s
        stats["server_seed"] = generation_seed
        stats["server_attempts"] = attempt + 1
        stats["server_blank_retries"] = attempt
        stats["server_blank_retry_suppressed"] = bool(streaming_response and blank_retry_budget)
        state.last_metrics.append(dict(envelope))
        state.last_metrics = state.last_metrics[-100:]
        last = {
            "text": out.text,
            "tokens": out.tokens,
            "stats": _json_safe(stats),
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": completion_tokens,
            "elapsed_s": elapsed_s,
            "tok_s": tok_s,
            "finish_reason": (
                out.final_state.finish_reason
                if out.final_state is not None
                else "stop"
            ),
        }
        if seed_is_explicit or out.text.strip():
            break
    assert last is not None
    print(
        json.dumps(
            {
                "event": "mtplx_openai_generation",
                "prompt_tokens": last["prompt_tokens"],
                "completion_tokens": last["completion_tokens"],
                "elapsed_s": round(float(last["elapsed_s"]), 6),
                "tok_s": round(float(last["tok_s"]), 6),
                "seed": last["stats"].get("server_seed"),
                "attempts": last["stats"].get("server_attempts"),
                "blank_retries": last["stats"].get("server_blank_retries"),
                "text_preview": str(last["text"])[:120],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return last


def _run_startup_warmup(state: ServerState) -> dict[str, Any]:
    warmup_tokens = int(getattr(state.args, "warmup_tokens", 0) or 0)
    status: dict[str, Any] = {
        "enabled": warmup_tokens > 0,
        "ran": False,
        "tokens": warmup_tokens,
        "elapsed_s": 0.0,
        "error": None,
    }
    if warmup_tokens <= 0:
        return status
    started = time.perf_counter()
    try:
        prompt_ids = _encode_prompt(state.runtime.tokenizer, "MTPLX warmup.")
        generated = _run_generation(
            state,
            prompt_ids,
            max_tokens=warmup_tokens,
            temperature=state.args.temperature,
            top_p=state.args.top_p,
            top_k=state.args.top_k,
            seed=0,
            request_observability={"warmup": True},
        )
    except BaseException as exc:
        status.update(
            {
                "elapsed_s": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        if getattr(state.args, "strict_warmup", False):
            raise
        return status
    status.update(
        {
            "ran": True,
            "elapsed_s": time.perf_counter() - started,
            "completion_tokens": generated.get("completion_tokens"),
            "tok_s": generated.get("tok_s"),
        }
    )
    return status


def _chunk_text(text: str, chunk_chars: int = 24) -> list[str]:
    if not text:
        return [""]
    return [text[index : index + chunk_chars] for index in range(0, len(text), chunk_chars)]


def _decode_timing(stats: dict[str, Any]) -> tuple[float, float]:
    generated_tokens = int(stats.get("generated_tokens") or 0)
    elapsed_s = float(stats.get("elapsed_s") or 0.0)
    if "prompt_eval_time_s" in stats:
        prompt_eval_time_s = float(stats.get("prompt_eval_time_s") or 0.0)
    else:
        target_forward_time_s = float(stats.get("target_forward_time_s") or 0.0)
        verify_time_s = float(stats.get("verify_time_s") or 0.0)
        repair_time_s = float(stats.get("repair_time_s") or 0.0)
        prompt_eval_time_s = max(0.0, target_forward_time_s - verify_time_s - repair_time_s)
    decode_elapsed_s = max(0.0, elapsed_s - prompt_eval_time_s)
    if decode_elapsed_s <= 0.0:
        return 0.0, 0.0
    return generated_tokens / decode_elapsed_s, decode_elapsed_s


def _stats_footer_text(state: ServerState, generated: dict[str, Any]) -> str:
    if not state.args.stats_footer:
        return ""
    stats = generated["stats"]
    tok_s, decode_elapsed_s = _decode_timing(stats)
    completion_tokens = int(generated.get("completion_tokens") or 0)
    footer = f"**{tok_s:.1f} tok/s** · {completion_tokens} tokens · {decode_elapsed_s:.2f}s decode"
    return f"{STATS_FOOTER_MARKER} {footer}"


def _usage_payload(generated: dict[str, Any]) -> dict[str, int]:
    prompt_tokens = int(generated.get("prompt_tokens") or 0)
    completion_tokens = int(generated.get("completion_tokens") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _split_thinking_segments(text: str, *, thinking_enabled: bool) -> tuple[str, str]:
    if not thinking_enabled:
        return "", text
    reasoning_parts: list[str] = []
    content_parts: list[str] = []

    def append_reasoning(segment: str) -> None:
        if not segment:
            return
        if reasoning_parts and not reasoning_parts[-1].endswith(("\n", " ")) and not segment.startswith(("\n", " ")):
            reasoning_parts.append("\n")
        reasoning_parts.append(segment)

    position = 0
    inside_thinking = True
    while position < len(text):
        if inside_thinking:
            close_index = text.find(THINK_CLOSE, position)
            if close_index < 0:
                segment = text[position:].replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
                append_reasoning(segment)
                break
            segment = text[position:close_index].replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
            append_reasoning(segment)
            position = close_index + len(THINK_CLOSE)
            inside_thinking = False
            continue

        open_index = text.find(THINK_OPEN, position)
        if open_index < 0:
            segment = text[position:].replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
            if segment:
                content_parts.append(segment)
            break
        segment = text[position:open_index].replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
        if segment:
            content_parts.append(segment)
        position = open_index + len(THINK_OPEN)
        inside_thinking = True
    return "".join(reasoning_parts).strip(), "".join(content_parts).strip()


def _normalize_thinking_tags(text: str, *, thinking_enabled: bool) -> str:
    """Make Qwen thinking output parseable by Open WebUI's native tag handler.

    Qwen's chat template can put the opening <think> tag in the prompt, so the
    generated text may begin inside the reasoning block and only emit </think>.
    Open WebUI's built-in parser needs a start tag in the streamed content.
    """
    reasoning, content = _split_thinking_segments(text, thinking_enabled=thinking_enabled)
    if not thinking_enabled:
        return content
    pieces: list[str] = []
    if reasoning:
        pieces.append(f"{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}")
    if content:
        pieces.append(content)
    return "\n\n".join(pieces)


class _IncrementalTokenDecoder:
    """Small TextStreamer-style decoder for committed-token SSE streaming.

    The previous bridge decoded the entire generated token buffer after every
    callback. That is prefix-stable, but it becomes O(n^2) tokenizer work during
    long reasoning streams. This keeps only the current partial word and flushes
    finalized text as soon as whitespace or CJK boundaries make it safe.
    """

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._token_cache: list[int] = []
        self._print_len = 0

    def _decode(self, tokens: list[int]) -> str:
        try:
            return self._tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
        except TypeError:
            return self._tokenizer.decode(tokens)

    def feed(self, tokens: list[int]) -> str:
        if not tokens:
            return ""
        self._token_cache.extend(int(token) for token in tokens)
        text = self._decode(self._token_cache)
        if text.endswith("\n"):
            printable = text[self._print_len :]
            self._token_cache = []
            self._print_len = 0
            return printable
        if text and self._is_cjk_char(ord(text[-1])):
            printable = text[self._print_len :]
            self._print_len += len(printable)
            return printable
        close_index = text.find(THINK_CLOSE, self._print_len)
        if close_index >= 0:
            boundary = close_index + len(THINK_CLOSE)
            printable = text[self._print_len : boundary]
            self._print_len = boundary
            return printable

        boundary = -1
        for index in range(len(text) - 1, -1, -1):
            if text[index].isspace():
                boundary = index + 1
                break
        if boundary <= self._print_len:
            return ""
        printable = text[self._print_len : boundary]
        self._print_len = boundary
        return printable

    def finish(self) -> str:
        if not self._token_cache:
            return ""
        text = self._decode(self._token_cache)
        printable = text[self._print_len :]
        self._token_cache = []
        self._print_len = 0
        return printable

    @staticmethod
    def _is_cjk_char(cp: int) -> bool:
        if (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3400 <= cp <= 0x4DBF)
            or (0x20000 <= cp <= 0x2A6DF)
            or (0x2A700 <= cp <= 0x2B73F)
            or (0x2B740 <= cp <= 0x2B81F)
            or (0x2B820 <= cp <= 0x2CEAF)
            or (0xF900 <= cp <= 0xFAFF)
            or (0x2F800 <= cp <= 0x2FA1F)
        ):
            return True
        return False


class _NonDuplicatingTokenDecoder(_IncrementalTokenDecoder):
    """Compatibility alias for old bridge tests/imports."""

    def feed(self, tokens: list[int]) -> str:
        text = super().feed(tokens)
        if text:
            return text
        if not self._token_cache:
            return ""
        joined = self._decode(self._token_cache)
        if THINK_CLOSE in joined:
            return self.finish()
        return ""


class _ThinkingContentStreamSplitter:
    def __init__(self, *, thinking_enabled: bool) -> None:
        self._thinking_enabled = thinking_enabled
        self._inside_thinking = thinking_enabled
        self._pending = ""
        self._reentry_count = 0

    @property
    def reentry_count(self) -> int:
        return self._reentry_count

    def start(self) -> list[tuple[str, str]]:
        return []

    def feed(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return []
        if not self._thinking_enabled:
            return [("content", text)]
        self._pending += text
        return self._drain(final=False)

    def finish(self) -> list[tuple[str, str]]:
        chunks = self._drain(final=True)
        self._inside_thinking = False
        return chunks

    def _append_chunk(
        self,
        chunks: list[tuple[str, str]],
        field: str,
        text: str,
    ) -> None:
        cleaned = text.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
        if cleaned:
            chunks.append((field, cleaned))

    def _drain(self, *, final: bool) -> list[tuple[str, str]]:
        chunks: list[tuple[str, str]] = []
        keep = max(len(THINK_OPEN), len(THINK_CLOSE)) - 1
        while self._pending:
            if self._inside_thinking:
                close_index = self._pending.find(THINK_CLOSE)
                if close_index < 0:
                    emit_len = len(self._pending) if final else max(0, len(self._pending) - keep)
                    if emit_len <= 0:
                        break
                    self._append_chunk(chunks, "reasoning_content", self._pending[:emit_len])
                    self._pending = self._pending[emit_len:]
                    break
                self._append_chunk(chunks, "reasoning_content", self._pending[:close_index])
                self._pending = self._pending[close_index + len(THINK_CLOSE) :].lstrip()
                self._inside_thinking = False
                continue

            open_index = self._pending.find(THINK_OPEN)
            if open_index < 0:
                emit_len = len(self._pending) if final else max(0, len(self._pending) - keep)
                if emit_len <= 0:
                    break
                self._append_chunk(chunks, "content", self._pending[:emit_len])
                self._pending = self._pending[emit_len:]
                break
            self._append_chunk(chunks, "content", self._pending[:open_index])
            self._pending = self._pending[open_index + len(THINK_OPEN) :]
            self._inside_thinking = True
            self._reentry_count += 1
        return chunks


class _ThinkingContentStreamNormalizer(_ThinkingContentStreamSplitter):
    """Compatibility wrapper returning only text chunks."""

    def start(self) -> list[str]:
        return [text for _, text in super().start()]

    def feed(self, text: str) -> list[str]:
        return [chunk for _, chunk in super().feed(text)]

    def finish(self) -> list[str]:
        return [chunk for _, chunk in super().finish()]


def _display_text(
    state: ServerState,
    generated: dict[str, Any],
    *,
    thinking_enabled: bool = False,
) -> str:
    raw_text = str(generated["text"])
    text = (
        _normalize_thinking_tags(
            raw_text,
            thinking_enabled=thinking_enabled,
        )
        if state.args.normalize_thinking_tags
        else raw_text
    )
    if not state.args.stats_footer:
        return text
    footer = _stats_footer_text(state, generated)
    separator = "\n\n" if text.endswith("\n") else "\n\n"
    return f"{text}{separator}{footer}"


def _thinking_enabled_for_request(
    state: ServerState,
    request: ChatCompletionRequest,
) -> bool:
    if state.args.reasoning_parser == "none":
        return False
    return state.args.enable_thinking if request.enable_thinking is None else bool(request.enable_thinking)


def create_app(state: ServerState) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            state.generation_executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="MTPLX OpenAI-compatible server", lifespan=lifespan)
    app.state.mtplx = state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def api_key_and_rate_limit(request: Request, call_next: Callable[[Request], Any]) -> Any:
        if not _request_is_authorized(request, state.args.api_key):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "missing or invalid API key",
                        "type": "authentication_error",
                    }
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
        allowed, retry_after = state.rate_limiter.check(_rate_limit_key(request))
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "rate limit exceeded",
                        "type": "rate_limit_error",
                    }
                },
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "name": "MTPLX OpenAI-compatible server",
            "model": state.model_id,
            "base_url": "/v1",
            "chat_completions": "/v1/chat/completions",
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "model": state.model_id,
            "model_path": str(state.runtime.model_path),
            "generation_mode": state.args.generation_mode,
            "load_mtp": bool(state.args.load_mtp),
            "mtp_enabled": bool(state.runtime.mtp_enabled),
            "depth": state.args.depth,
            "profile": state.profile.to_dict(),
            "adaptive": _adaptive_config(state.args),
            "proposal_cache": _proposal_cache_config(state.args),
            "online_hidden": _online_hidden_config(state.args),
            "verify_core": state.args.verify_core,
            "verify_strategy": state.args.verify_strategy,
            "enable_thinking": state.args.enable_thinking,
            "context_window": state.context_window,
            "max_response_tokens": state.args.max_response_tokens,
            "api_key_required": bool(state.args.api_key),
            "rate_limit_per_minute": int(state.args.rate_limit),
            "stream_interval": int(state.args.stream_interval),
            "warmup": state.warmup_status,
            "reasoning_parser": state.args.reasoning_parser,
            "load_time_s": state.load_time_s,
            "draft_lm_head": state.draft_lm_head,
            "draft_head_identity": state.draft_head_identity,
            "tokenizer_template_hash": state.template_hash,
            "fast_path_env": state.fast_path_env_status,
            "profile_env": state.profile_env_status,
            "diagnostic_env_ablation": bool(state.args.diagnostic_env_ablation),
            "mtp_history_materialize_every": (
                os.environ.get("MTPLX_MTP_HISTORY_MATERIALIZE_EVERY")
            ),
            "clear_cache_every": os.environ.get("MTPLX_CLEAR_CACHE_EVERY"),
            "trunk_cache_materialize_every": (
                os.environ.get("MTPLX_TRUNK_CACHE_MATERIALIZE_EVERY")
            ),
            "state_rebase_every": os.environ.get("MTPLX_STATE_REBASE_EVERY"),
            "state_root_eval": os.environ.get("MTPLX_EVAL_STATE_ROOTS_ON_COMMIT"),
            "state_root_eval_include_mtp": os.environ.get(
                "MTPLX_EVAL_STATE_ROOTS_INCLUDE_MTP"
            ),
            "state_root_eval_include_live": os.environ.get(
                "MTPLX_EVAL_STATE_ROOTS_INCLUDE_LIVE"
            ),
            "target_layer_eval_every": os.environ.get("MTPLX_TARGET_LAYER_EVAL_EVERY"),
            "target_layer_eval_schedule": os.environ.get(
                "MTPLX_TARGET_LAYER_EVAL_SCHEDULE"
            ),
            "target_layer_eval_context_threshold": os.environ.get(
                "MTPLX_TARGET_LAYER_EVAL_CONTEXT_THRESHOLD"
            ),
            "target_layer_eval_max_q": os.environ.get("MTPLX_TARGET_LAYER_EVAL_MAX_Q"),
            "defer_verify_hidden_eval": os.environ.get("MTPLX_DEFER_VERIFY_HIDDEN_EVAL"),
            "late_depth_switch_after_tokens": os.environ.get(
                "MTPLX_LATE_DEPTH_SWITCH_AFTER_TOKENS"
            ),
            "late_depth_before": os.environ.get("MTPLX_LATE_DEPTH_BEFORE"),
            "late_depth_after": os.environ.get("MTPLX_LATE_DEPTH_AFTER"),
            "mtp_position_mode": os.environ.get("MTPLX_MTP_POSITION_MODE"),
            "mtp_position_cap": os.environ.get("MTPLX_MTP_POSITION_CAP"),
            "mtp_position_period": os.environ.get("MTPLX_MTP_POSITION_PERIOD"),
            "mtp_position_base": os.environ.get("MTPLX_MTP_POSITION_BASE"),
            "split_verify_eval": os.environ.get("MTPLX_SPLIT_VERIFY_EVAL"),
            "split_full_attn": os.environ.get("MTPLX_SPLIT_FULL_ATTN"),
            "split_full_attn_chunk_size": os.environ.get(
                "MTPLX_SPLIT_FULL_ATTN_CHUNK_SIZE"
            ),
            "split_full_attn_threshold": os.environ.get(
                "MTPLX_SPLIT_FULL_ATTN_THRESHOLD"
            ),
            "sdpa_2pass": os.environ.get("MTPLX_SDPA_2PASS"),
            "sdpa_2pass_threshold": os.environ.get("MTPLX_SDPA_2PASS_THRESHOLD"),
            "sdpa_2pass_max_q": os.environ.get("MTPLX_SDPA_2PASS_MAX_Q"),
            "blockwise_attn": os.environ.get("MTPLX_BLOCKWISE_ATTN"),
            "blockwise_attn_threshold": os.environ.get(
                "MTPLX_BLOCKWISE_ATTN_THRESHOLD"
            ),
            "dirty_detach_components": os.environ.get("MTPLX_DETACH_COMPONENTS"),
            "dirty_detach_mode": os.environ.get("MTPLX_DETACH_MODE"),
            "dirty_detach_gdn_every": os.environ.get("MTPLX_DETACH_GDN_EVERY"),
            "dirty_detach_conv_every": os.environ.get("MTPLX_DETACH_CONV_EVERY"),
            "dirty_detach_attn_every": os.environ.get("MTPLX_DETACH_ATTN_EVERY"),
            "capture_commit_detach_components": os.environ.get(
                "MTPLX_CAPTURE_COMMIT_DETACH_COMPONENTS"
            ),
            "capture_commit_detach_mode": os.environ.get(
                "MTPLX_CAPTURE_COMMIT_DETACH_MODE"
            ),
            "capture_commit_detach_gdn_every": os.environ.get(
                "MTPLX_CAPTURE_COMMIT_DETACH_GDN_EVERY"
            ),
            "capture_commit_detach_conv_every": os.environ.get(
                "MTPLX_CAPTURE_COMMIT_DETACH_CONV_EVERY"
            ),
            "owned_recurrent_state": os.environ.get("MTPLX_OWNED_RECURRENT_STATE"),
            "owned_recurrent_state_mode": os.environ.get(
                "MTPLX_OWNED_RECURRENT_STATE_MODE"
            ),
            "owned_attn_kv": os.environ.get("MTPLX_OWNED_ATTN_KV"),
            "owned_attn_kv_mode": os.environ.get("MTPLX_OWNED_ATTN_KV_MODE"),
            "owned_attn_kv_step": os.environ.get("MTPLX_OWNED_ATTN_KV_STEP"),
            "owned_attn_kv_block_size": os.environ.get(
                "MTPLX_OWNED_ATTN_KV_BLOCK_SIZE"
            ),
            "vllm_metal_paged_attn": os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN"),
            "vllm_metal_paged_block_size": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE"
            ),
            "vllm_metal_paged_num_blocks": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS"
            ),
            "vllm_metal_paged_sliding_window": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_SLIDING_WINDOW"
            ),
            "vllm_metal_paged_partitioned_attn": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN"
            ),
            "vllm_metal_paged_partition_threshold": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD"
            ),
            "vllm_metal_paged_partition_size": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE"
            ),
            "vllm_metal_paged_mtp_attn": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_MTP_ATTN"
            ),
            "vllm_metal_paged_mtp_block_size": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_MTP_BLOCK_SIZE"
            ),
            "vllm_metal_paged_mtp_num_blocks": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_MTP_NUM_BLOCKS"
            ),
            "vllm_metal_paged_turboquant": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_TURBOQUANT"
            ),
            "vllm_metal_paged_turboquant_k_quant": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_TURBOQUANT_K_QUANT"
            ),
            "vllm_metal_paged_turboquant_v_quant": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_TURBOQUANT_V_QUANT"
            ),
            "native_mlp_rowwise": os.environ.get("MTPLX_NATIVE_MLP_ROWWISE"),
            "native_mlp_min_m": os.environ.get("MTPLX_NATIVE_MLP_MIN_M"),
            "native_mlp_max_m": os.environ.get("MTPLX_NATIVE_MLP_MAX_M"),
            "native_mlp_context_threshold": os.environ.get(
                "MTPLX_NATIVE_MLP_CONTEXT_THRESHOLD"
            ),
            "mlp_call_variant": os.environ.get("MTPLX_MLP_CALL_VARIANT"),
            "mlp_variant_stats": native_mlp_stats(),
            "native_gdn_tail": os.environ.get("MTPLX_NATIVE_GDN_TAIL"),
            "native_gdn_tail_simdgroups": os.environ.get(
                "MTPLX_NATIVE_GDN_TAIL_SIMDGROUPS"
            ),
            "live_output_detach": os.environ.get("MTPLX_DETACH_LIVE_OUTPUTS"),
            "live_output_detach_mode": os.environ.get(
                "MTPLX_DETACH_LIVE_OUTPUTS_MODE"
            ),
            "mlx_cache_limit": state.mlx_cache_limit_status,
            "mlx_fork": state.mlx_fork_status,
        }

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return {
            "latest": state.last_metrics[-1] if state.last_metrics else None,
            "recent": state.last_metrics[-32:],
        }

    @app.get("/admin/sessions")
    def admin_sessions() -> dict[str, Any]:
        return state.sessions.list_sessions()

    @app.post("/admin/sessions/{session_id}/clear")
    def admin_clear_session(session_id: str) -> dict[str, Any]:
        return state.sessions.clear_session(session_id)

    @app.post("/admin/cache/clear")
    def admin_clear_cache() -> dict[str, Any]:
        return state.sessions.clear_all()

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": state.model_id,
                    "object": "model",
                    "created": now,
                    "owned_by": "mtplx",
                    "context_length": state.context_window,
                    "max_context_length": state.context_window,
                    "max_model_len": state.context_window,
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(raw_request: Request, request: ChatCompletionRequest) -> Any:
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        headers = dict(raw_request.headers)
        metadata = _request_metadata(request)
        cache_bypass = (
            headers.get("x-mtplx-cache-mode", "").lower() in {"bypass", "stateless", "off"}
            or str(metadata.get("cache_mode", "")).lower() in {"bypass", "stateless", "off"}
        )
        background = is_background_request(
            messages=request.messages,
            max_tokens=request.max_tokens,
            headers=headers,
            metadata=metadata,
            main_system_hash=state.main_system_prompt_hash,
        )
        if background and (state.has_foreground() or state.lock.locked()):
            return JSONResponse(
                status_code=503,
                headers={"Retry-After": "1"},
                content={
                    "error": {
                        "message": "background Open WebUI task bypassed while foreground generation is busy",
                        "type": CacheMissReason.SESSION_BUSY.value,
                    },
                    "mtplx_stats": {
                        "session_cache_hit": False,
                        "cache_miss_reason": CacheMissReason.SESSION_BUSY.value,
                        "session_restore_mode": "background_bypass",
                    },
                },
            )
        thinking_enabled = _thinking_enabled_for_request(state, request)
        prompt_ids = _encode_messages(
            state.runtime.tokenizer,
            request.messages,
            enable_thinking=thinking_enabled,
            strip_assistant_reasoning_history=state.args.strip_assistant_reasoning_history,
        )
        current_system_hash = system_prompt_hash(request.messages)
        if current_system_hash is not None and not background:
            state.main_system_prompt_hash = current_system_hash
        session_id: str | None = None
        session_source: str | None = None
        session = None
        cache_miss_reason = (
            CacheMissReason.BACKGROUND_BYPASS.value
            if background
            else CacheMissReason.NEW_SESSION.value
        )
        session_restore_mode = "background_bypass" if background else "cold"
        policy_fingerprint = _policy_fingerprint(
            state,
            thinking_enabled=thinking_enabled,
        )
        if not background and not cache_bypass:
            requested_restore_mode = headers.get("x-mtplx-restore-mode", "reference_lease")
            requested_restore_mode = requested_restore_mode.replace("-", "_")
            session_restore_mode = (
                "clone"
                if requested_restore_mode == "clone"
                else "reference_lease"
            )
            session_id, session_source = state.sessions.resolve_session_id(
                headers=headers,
                metadata=metadata,
                user=_request_extra(request, "user"),
                chat_id=_request_extra(request, "chat_id"),
                conversation_id=_request_extra(request, "conversation_id"),
                prompt_ids=prompt_ids,
            )
            session = state.sessions.get_or_create(session_id)
            session.last_cache_miss_reason = cache_miss_reason
            session.last_restore_mode = session_restore_mode
        request_observability = _request_observability(
            request,
            headers=headers,
            metadata=metadata,
            session_source=session_source,
        )
        model = request.model or state.model_id
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if request.stream:
            async def event_stream():
                first = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(first)}\n\n"

                queue: Queue[tuple[str, Any]] = Queue()
                decoder = _IncrementalTokenDecoder(state.runtime.tokenizer)
                splitter = _ThinkingContentStreamSplitter(thinking_enabled=thinking_enabled)
                stream_interval = max(1, int(state.args.stream_interval))
                pending_stream_tokens: list[int] = []
                commit_event = Event()
                commit_state = {"commit": False, "assistant_history_content": None}

                def on_tokens(new_tokens: list[int]) -> None:
                    queue.put(("tokens", list(new_tokens)))

                def worker() -> None:
                    try:
                        if session is None:
                            generated = _run_generation(
                                state,
                                prompt_ids,
                                max_tokens=request.max_tokens,
                                temperature=request.temperature,
                                top_p=request.top_p,
                                top_k=request.top_k,
                                seed=request.seed,
                                token_callback=on_tokens,
                                session_id=session_id,
                                cache_miss_reason=cache_miss_reason,
                                session_restore_mode=session_restore_mode,
                                session_bank=None if background or cache_bypass else state.sessions.bank,
                                session_template_hash=state.template_hash,
                                session_draft_head_identity=state.draft_head_identity,
                                session_policy_fingerprint=policy_fingerprint,
                                background_request=background,
                                request_observability=request_observability,
                            )
                        else:
                            with session.in_flight_generation():
                                generated = _run_generation(
                                    state,
                                    prompt_ids,
                                    max_tokens=request.max_tokens,
                                    temperature=request.temperature,
                                    top_p=request.top_p,
                                    top_k=request.top_k,
                                    seed=request.seed,
                                    token_callback=on_tokens,
                                    session_id=session_id,
                                    cache_miss_reason=cache_miss_reason,
                                    session_restore_mode=session_restore_mode,
                                    session_bank=state.sessions.bank,
                                    session_template_hash=state.template_hash,
                                    session_draft_head_identity=state.draft_head_identity,
                                    session_policy_fingerprint=policy_fingerprint,
                                    commit_final_state_to_bank=False,
                                    request_observability=request_observability,
                                )
                                queue.put(("done", generated))
                                commit_event.wait()
                                if commit_state["commit"]:
                                    assistant_history_content = (
                                        str(commit_state.get("assistant_history_content") or "")
                                        or (
                                            _normalize_thinking_tags(
                                                str(generated["text"]),
                                                thinking_enabled=thinking_enabled,
                                            )
                                            if state.args.normalize_thinking_tags
                                            else str(generated["text"])
                                        )
                                    )
                                    postcommit = _store_retokenized_history_snapshot(
                                        state,
                                        session_id=session_id,
                                        messages=request.messages,
                                        assistant_content=assistant_history_content,
                                        thinking_enabled=thinking_enabled,
                                        policy_fingerprint=policy_fingerprint,
                                    )
                                    generated["stats"]["session_postcommit_snapshot"] = postcommit
                                    session.commit(
                                        prompt_ids=prompt_ids,
                                        generated_ids=generated["tokens"],
                                        finish_reason=generated.get("finish_reason", "stop"),
                                    )
                                    queue.put(("committed", generated))
                                else:
                                    queue.put(("released", None))
                                return
                    except EngineSessionBusy as exc:
                        queue.put(("error", HTTPException(status_code=409, detail=str(exc))))
                    except BaseException as exc:
                        if (
                            session is not None
                            and commit_event.is_set()
                            and state.args.session_postcommit_mode == "async"
                        ):
                            print(
                                f"[mtplx] async session postcommit failed: {exc!r}",
                                flush=True,
                            )
                        queue.put(("error", exc))
                    else:
                        queue.put(("done", generated))

                state.generation_executor.submit(worker)

                def delta_chunk(field: str, text: str) -> str:
                    payload = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                                {
                                    "index": 0,
                                    "delta": {field: text},
                                    "finish_reason": None,
                                }
                        ],
                    }
                    return f"data: {json.dumps(payload)}\n\n"

                def error_chunk(exc: BaseException) -> str:
                    if isinstance(exc, HTTPException):
                        message = str(exc.detail)
                        error_type = str(exc.status_code)
                        status_code = exc.status_code
                    else:
                        message = str(exc)
                        error_type = type(exc).__name__
                        status_code = 500
                    payload = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                        "error": {
                            "message": message,
                            "type": error_type,
                            "status_code": status_code,
                        },
                    }
                    return f"data: {json.dumps(payload)}\n\n"

                generated: dict[str, Any] | None = None
                history_reasoning_chunks: list[str] = []
                history_content_chunks: list[str] = []

                def remember_stream_chunk(field: str, text: str) -> None:
                    if field == "reasoning_content":
                        history_reasoning_chunks.append(text)
                    elif field == "content":
                        history_content_chunks.append(text)

                def drain_stream_tokens(
                    tokens: list[int],
                    *,
                    force: bool = False,
                ) -> list[tuple[str, str]]:
                    pending_stream_tokens.extend(tokens)
                    chunks: list[tuple[str, str]] = []
                    while pending_stream_tokens and (
                        force or len(pending_stream_tokens) >= stream_interval
                    ):
                        batch_size = len(pending_stream_tokens) if force else stream_interval
                        batch = pending_stream_tokens[:batch_size]
                        del pending_stream_tokens[:batch_size]
                        delta = decoder.feed(batch)
                        chunks.extend(
                            (field, text)
                            for field, text in splitter.feed(delta)
                            if text
                        )
                    return chunks

                def streamed_history_content() -> str:
                    reasoning = (
                        "".join(history_reasoning_chunks)
                        .replace(THINK_OPEN, "")
                        .replace(THINK_CLOSE, "")
                        .strip()
                    )
                    content = (
                        "".join(history_content_chunks)
                        .replace(THINK_OPEN, "")
                        .replace(THINK_CLOSE, "")
                        .strip()
                    )
                    pieces: list[str] = []
                    if reasoning:
                        pieces.append(f"{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}")
                    if content:
                        pieces.append(content)
                    return "\n\n".join(pieces)

                for field, text in splitter.start():
                    if text:
                        remember_stream_chunk(field, text)
                        yield delta_chunk(field, text)

                try:
                    while True:
                        kind, item = await asyncio.to_thread(queue.get)
                        if kind == "tokens":
                            for field, text in drain_stream_tokens(item):
                                remember_stream_chunk(field, text)
                                yield delta_chunk(field, text)
                        elif kind == "done":
                            generated = item
                            for field, text in drain_stream_tokens([], force=True):
                                remember_stream_chunk(field, text)
                                yield delta_chunk(field, text)
                            tail = decoder.finish()
                            if tail:
                                for field, text in splitter.feed(tail):
                                    if text:
                                        remember_stream_chunk(field, text)
                                        yield delta_chunk(field, text)
                            for field, text in splitter.finish():
                                if text:
                                    remember_stream_chunk(field, text)
                                    yield delta_chunk(field, text)
                            if session is not None:
                                commit_state["assistant_history_content"] = streamed_history_content()
                                commit_state["commit"] = True
                                commit_event.set()
                                if state.args.session_postcommit_mode == "async":
                                    generated["stats"]["session_postcommit_snapshot"] = {
                                        "stored": False,
                                        "mode": "async_pending",
                                    }
                                else:
                                    commit_kind, commit_item = await asyncio.to_thread(queue.get)
                                    if commit_kind == "committed":
                                        generated = commit_item
                                    elif commit_kind == "error":
                                        yield error_chunk(commit_item)
                                        yield "data: [DONE]\n\n"
                                        return
                                    else:
                                        yield error_chunk(
                                            RuntimeError(f"unexpected commit event: {commit_kind}")
                                        )
                                        yield "data: [DONE]\n\n"
                                        return
                            generated["stats"]["reasoning_reentries"] = splitter.reentry_count
                            if state.last_metrics:
                                state.last_metrics[-1]["reasoning_reentries"] = splitter.reentry_count
                            footer = _stats_footer_text(state, generated)
                            if footer:
                                yield delta_chunk("content", f"\n\n{footer}")
                            break
                        elif kind == "error":
                            yield error_chunk(item)
                            yield "data: [DONE]\n\n"
                            return
                        else:
                            yield error_chunk(RuntimeError(f"unexpected stream event: {kind}"))
                            yield "data: [DONE]\n\n"
                            return
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    yield error_chunk(exc)
                    yield "data: [DONE]\n\n"
                    return
                finally:
                    if session is not None and not commit_event.is_set():
                        commit_state["commit"] = False
                        commit_event.set()

                if generated is None:
                    yield error_chunk(RuntimeError("generation ended without a result"))
                    yield "data: [DONE]\n\n"
                    return
                done = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": generated.get("finish_reason", "stop"),
                        }
                    ],
                    "usage": _usage_payload(generated),
                    "mtplx_stats": generated["stats"],
                }
                yield f"data: {json.dumps(done)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        def run_nonstream_generation() -> dict[str, Any]:
            if session is None:
                return _run_generation(
                    state,
                    prompt_ids,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    seed=request.seed,
                    session_id=session_id,
                    cache_miss_reason=cache_miss_reason,
                    session_restore_mode=session_restore_mode,
                    session_bank=None if background or cache_bypass else state.sessions.bank,
                    session_template_hash=state.template_hash,
                    session_draft_head_identity=state.draft_head_identity,
                    session_policy_fingerprint=policy_fingerprint,
                    background_request=background,
                    request_observability=request_observability,
                )
            with session.in_flight_generation():
                generated_result = _run_generation(
                    state,
                    prompt_ids,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    seed=request.seed,
                    session_id=session_id,
                    cache_miss_reason=cache_miss_reason,
                    session_restore_mode=session_restore_mode,
                    session_bank=state.sessions.bank,
                    session_template_hash=state.template_hash,
                    session_draft_head_identity=state.draft_head_identity,
                    session_policy_fingerprint=policy_fingerprint,
                    request_observability=request_observability,
                )
                session.commit(
                    prompt_ids=prompt_ids,
                    generated_ids=generated_result["tokens"],
                    finish_reason=generated_result.get("finish_reason", "stop"),
                )
                return generated_result

        loop = asyncio.get_running_loop()
        try:
            generated = await loop.run_in_executor(
                state.generation_executor,
                run_nonstream_generation,
            )
        except EngineSessionBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        display_text = _display_text(
            state,
            generated,
            thinking_enabled=thinking_enabled,
        )
        if session is not None:
            if state.args.session_postcommit_mode == "async":
                generated["stats"]["session_postcommit_snapshot"] = {
                    "stored": False,
                    "mode": "async_pending",
                }

                def async_nonstream_postcommit() -> None:
                    try:
                        _store_retokenized_history_snapshot(
                            state,
                            session_id=session_id,
                            messages=request.messages,
                            assistant_content=display_text,
                            thinking_enabled=thinking_enabled,
                            policy_fingerprint=policy_fingerprint,
                        )
                    except BaseException as exc:
                        print(
                            f"[mtplx] async session postcommit failed: {exc!r}",
                            flush=True,
                        )

                state.generation_executor.submit(async_nonstream_postcommit)
            else:
                postcommit = await loop.run_in_executor(
                    state.generation_executor,
                    lambda: _store_retokenized_history_snapshot(
                        state,
                        session_id=session_id,
                        messages=request.messages,
                        assistant_content=display_text,
                        thinking_enabled=thinking_enabled,
                        policy_fingerprint=policy_fingerprint,
                    ),
                )
                generated["stats"]["session_postcommit_snapshot"] = postcommit
        message: dict[str, str] = {"role": "assistant", "content": display_text}
        return JSONResponse(
            {
                "id": response_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "stop",
                    }
                ],
                "usage": _usage_payload(generated),
                "mtplx_stats": generated["stats"],
            }
        )

    @app.post("/v1/messages")
    async def anthropic_messages(raw_request: Request, request: AnthropicMessagesRequest) -> Any:
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        if request.stream:
            raise HTTPException(
                status_code=501,
                detail="Anthropic streaming is not implemented yet; send stream=false.",
            )
        chat_request = _anthropic_to_chat_request(request)
        response = await chat_completions(raw_request, chat_request)
        if not isinstance(response, JSONResponse):
            return response
        try:
            openai_payload = json.loads(response.body)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to translate response: {exc}") from exc
        if response.status_code >= 400:
            return response
        payload = _anthropic_payload_from_openai(openai_payload)
        return JSONResponse(payload, status_code=response.status_code)

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest) -> Any:
        prompt_ids = _encode_prompt(state.runtime.tokenizer, request.prompt)
        generated = _run_generation(
            state,
            prompt_ids,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            seed=request.seed,
        )
        model = request.model or state.model_id
        response_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        display_text = _display_text(state, generated)
        if request.stream:
            async def event_stream():
                chunk_chars = 24 * max(1, int(state.args.stream_interval))
                for chunk in _chunk_text(display_text, chunk_chars=chunk_chars):
                    payload = {
                        "id": response_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "text": chunk, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")
        return JSONResponse(
            {
                "id": response_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "text": display_text, "finish_reason": "stop"}],
                "usage": _usage_payload(generated),
                "mtplx_stats": generated["stats"],
            }
        )

    @app.exception_handler(Exception)
    async def unhandled_exception(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": type(exc).__name__}},
        )

    return app


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    postcommit_default = os.environ.get("MTPLX_SESSION_POSTCOMMIT_MODE", "inline")
    if postcommit_default not in {"inline", "async"}:
        postcommit_default = "inline"
    parser.add_argument("--model", default="models/Qwen3.6-27B-MTPLX-GDN8-Speed4-CyanKiwiMTP")
    parser.add_argument("--model-id", default="mtplx-qwen36-27b-native-mtp")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default=DEFAULT_PROFILE_NAME)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MTPLX_AUTH"),
        help="Require Bearer or X-API-Key auth. Required for non-localhost binds.",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=_env_int("MTPLX_RATE_LIMIT_PER_MINUTE", 0),
        help="Requests per minute per client/API key. Use 0 to disable.",
    )
    parser.add_argument(
        "--stream-interval",
        type=int,
        default=_env_int("MTPLX_STREAM_INTERVAL", 1),
        help="Committed-token batch size per chat SSE chunk. Default: 1.",
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=_env_int("MTPLX_WARMUP_COUNT", 16),
        help="Startup warmup generation length. Use 0 to disable.",
    )
    parser.add_argument(
        "--strict-warmup",
        action="store_true",
        help="Fail server startup if the warmup pass fails.",
    )
    parser.add_argument(
        "--generation-mode",
        choices=["mtp", "ar"],
        default="mtp",
        help="Diagnostic generation mode. 'ar' disables native-MTP drafting but keeps the same target/runtime stack.",
    )
    parser.add_argument(
        "--load-mtp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load and inject the native MTP sidecar. Disable only for stock AR diagnostics.",
    )
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument(
        "--max-response-tokens",
        "--max-tokens",
        dest="max_response_tokens",
        type=int,
        default=None,
        help="Optional emergency generation cap. Unset means no bridge cap; requests are limited only by remaining context.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=0,
        help="Override context window. Default reads the model/tokenizer config.",
    )
    parser.add_argument("--temperature", "--default-temperature", dest="temperature", type=float, default=0.6)
    parser.add_argument("--top-p", "--default-top-p", dest="top_p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--adaptive-policy",
        choices=["none", "streak", "expected_value"],
        default="none",
        help="Optional per-request native-MTP depth policy. Exact sampler semantics remain unchanged.",
    )
    parser.add_argument("--adaptive-min-depth", type=int, default=1)
    parser.add_argument("--adaptive-start-depth", type=int, default=1)
    parser.add_argument("--adaptive-increase-after", type=int, default=4)
    parser.add_argument("--adaptive-decrease-after", type=int, default=1)
    parser.add_argument("--adaptive-ev-base-depth", type=int, default=2)
    parser.add_argument(
        "--adaptive-ev-accept-priors",
        type=_comma_floats,
        default=(0.92, 0.64, 0.32),
    )
    parser.add_argument("--adaptive-ev-draft-cost-s", type=float, default=0.0048)
    parser.add_argument("--adaptive-ev-extra-verify-cost-s", type=float, default=0.0060)
    parser.add_argument("--adaptive-ev-baseline-tok-s", type=float, default=40.0)
    parser.add_argument("--adaptive-ev-safety-margin", type=float, default=0.10)
    parser.add_argument("--adaptive-ev-margin-center", type=float, default=1.0)
    parser.add_argument("--adaptive-ev-margin-scale", type=float, default=2.0)
    parser.add_argument("--adaptive-ev-confidence-weight", type=float, default=0.35)
    parser.add_argument(
        "--adaptive-ev-min-extra-accept-probability",
        type=float,
        default=0.18,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic seed. Omit for fresh per-request interactive sampling.",
    )
    parser.add_argument(
        "--blank-retry-attempts",
        type=int,
        default=3,
        help="Retry empty unseeded generations this many times with fresh seeds.",
    )
    parser.add_argument(
        "--no-stats-footer",
        action="store_false",
        dest="stats_footer",
        help="Do not append the visible MTPLX TPS footer to returned text.",
    )
    parser.add_argument(
        "--session-postcommit-mode",
        choices=["inline", "async"],
        default=postcommit_default,
        help=(
            "SessionBank postcommit policy. 'inline' preserves existing behavior; "
            "'async' returns the stream/response before the retokenized history "
            "snapshot finishes, while the single generation executor completes "
            "the exact snapshot in the background."
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass enable_thinking to the Qwen chat template for visible <think> reasoning blocks.",
    )
    parser.add_argument(
        "--reasoning-parser",
        choices=["qwen3", "none"],
        default="qwen3",
        help="Parser for streamed reasoning tags. Use 'none' to stream all text as content.",
    )
    parser.add_argument(
        "--strip-assistant-reasoning-history",
        action="store_true",
        help=(
            "Opt-in speed/debug mode: remove prior assistant <think>/reasoning blocks "
            "before re-encoding chat history. Default preserves reasoning context."
        ),
    )
    parser.add_argument(
        "--normalize-thinking-tags",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Normalize non-stream text with explicit <think> tags. Default keeps "
            "raw generated text so assistant history remains byte/token stable."
        ),
    )
    parser.add_argument("--verify-strategy", default="capture_commit")
    parser.add_argument("--verify-core", default="linear-gdn-from-conv-tape")
    parser.add_argument(
        "--online-correction-cache",
        action="store_true",
        help=(
            "Enable the exact target-feedback proposal cache. Diagnostic: changes "
            "draft q only; target verification and residual correction remain exact."
        ),
    )
    parser.add_argument("--online-correction-cache-min-depth", type=int, default=1)
    parser.add_argument(
        "--online-correction-cache-key",
        choices=["local_prefix", "source_token", "primary_source"],
        default="local_prefix",
    )
    parser.add_argument(
        "--prompt-correction-cache",
        action="store_true",
        help=(
            "Seed the exact proposal cache from prompt-local n-grams. Diagnostic "
            "only; historically not additive on short-code screens."
        ),
    )
    parser.add_argument("--prompt-correction-cache-min-depth", type=int, default=2)
    parser.add_argument(
        "--online-hidden-corrector-alpha",
        type=float,
        default=0.0,
        help=(
            "Apply an online hidden residual to MTP draft hidden states. "
            "Diagnostic: changes proposal q only; target verification and "
            "residual correction remain authoritative."
        ),
    )
    parser.add_argument("--online-hidden-corrector-decay", type=float, default=0.8)
    parser.add_argument("--online-hidden-corrector-warmup", type=int, default=1)
    parser.add_argument("--online-hidden-corrector-max-feed-depth", type=int)
    parser.add_argument(
        "--online-hidden-corrector-key",
        choices=["global", "token"],
        default="global",
    )
    parser.add_argument("--draft-lm-head-bits", type=int, default=4)
    parser.add_argument("--draft-lm-head-group-size", type=int, default=64)
    parser.add_argument("--draft-lm-head-mode", default="affine")
    parser.add_argument(
        "--mlx-cache-limit",
        help=(
            "Optional MLX allocator cache limit, e.g. 0, 512MB, 1GB. "
            "Defaults to MTPLX_MLX_CACHE_LIMIT when set."
        ),
    )
    parser.add_argument(
        "--strict-startup-asserts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refuse startup unless the selected MTPLX profile env is active after profile setup.",
    )
    parser.add_argument(
        "--diagnostic-env-ablation",
        action="store_true",
        help=(
            "Diagnostic mode: report observed MTPLX profile/fast-path env exactly as launched "
            "without asserting or filling missing flags."
        ),
    )
    parser.add_argument(
        "--strict-mlx-fork-assert",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refuse startup unless the selected profile's required MLX fork is active.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_server_security_args(args)
    state = ServerState(args)
    app = create_app(state)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
