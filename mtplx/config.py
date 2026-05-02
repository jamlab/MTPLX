"""No-MLX user configuration helpers."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mtplx.constants import DEFAULT_RUNTIME_MODEL_DIR
from mtplx.profiles import DEFAULT_PROFILE_NAME, resolve_profile_name


DEFAULT_CONFIG_PATH = Path("~/.mtplx/config.toml").expanduser()
RUNTIME_MODEL_COMMANDS = {"run", "chat", "serve"}
CACHE_COMMANDS = {"pull", "list", "remove"}


@dataclass(frozen=True)
class UserConfig:
    path: Path
    exists: bool
    model: str | None = None
    model_dir: str | None = None
    profile: str | None = None
    thermal_control: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "model": self.model,
            "model_dir": self.model_dir,
            "profile": self.profile,
            "thermal_control": self.thermal_control,
        }


def user_config_path(value: str | Path | None = None) -> Path:
    if value:
        return Path(value).expanduser()
    env = os.environ.get("MTPLX_CONFIG")
    if env:
        return Path(env).expanduser()
    return DEFAULT_CONFIG_PATH


def load_user_config(path: str | Path | None = None) -> UserConfig:
    resolved = user_config_path(path)
    if not resolved.exists():
        return UserConfig(path=resolved, exists=False)
    with resolved.open("rb") as handle:
        data = tomllib.load(handle)
    model = data.get("model")
    model_dir = data.get("model_dir")
    profile = data.get("profile")
    thermal_control = data.get("thermal_control")
    if profile is not None:
        profile = resolve_profile_name(str(profile))
    return UserConfig(
        path=resolved,
        exists=True,
        model=str(model) if model else None,
        model_dir=str(model_dir) if model_dir else None,
        profile=str(profile) if profile else None,
        thermal_control=str(thermal_control) if thermal_control else None,
    )


def apply_user_config(args: Any, *, config_path: str | Path | None = None) -> UserConfig:
    config = load_user_config(config_path)
    setattr(args, "mtplx_config", config.to_dict())
    if not config.exists:
        return config

    command = getattr(args, "command", None)
    if command in RUNTIME_MODEL_COMMANDS:
        _apply_model_default(args, config)
        _apply_cache_default(args, config)
        _apply_profile_default(args, config)
    elif command == "bench" and getattr(args, "bench_action", None) == "run":
        _apply_model_default(args, config)
        _apply_cache_default(args, config)
        _apply_profile_default(args, config)
    elif command in CACHE_COMMANDS:
        _apply_cache_default(args, config)
    elif command == "doctor" and getattr(args, "model_cache", None) is None and config.model_dir:
        args.model_cache = config.model_dir
    return config


def _apply_model_default(args: Any, config: UserConfig) -> None:
    current = getattr(args, "model", None)
    if config.model and current == str(DEFAULT_RUNTIME_MODEL_DIR):
        args.model = config.model


def _apply_cache_default(args: Any, config: UserConfig) -> None:
    if hasattr(args, "cache_dir") and getattr(args, "cache_dir", None) is None and config.model_dir:
        args.cache_dir = config.model_dir


def _apply_profile_default(args: Any, config: UserConfig) -> None:
    current = getattr(args, "profile", None)
    if config.profile and current == DEFAULT_PROFILE_NAME:
        args.profile = config.profile
