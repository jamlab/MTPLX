"""Opt-in thermal-control helpers for MTPLX.

The public contract is deliberately conservative: detect known fan-control
tools, run their documented/profile-style CLI commands when available, and
otherwise report clear installation instructions. This module never falls back
to spin loops or clock-anchor hacks.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator


PROFILE_LABELS = {
    "performance": "Performance",
    "max": "Max",
    "silent": "Silent",
}

INSTALL_INSTRUCTIONS = {
    "thermalforge": "Install ThermalForge and ensure the thermalforge CLI is on PATH.",
    "tgpro": "Install TG Pro and ensure tgpro or tgpro-cli is on PATH.",
    "none": (
        "Install ThermalForge, or install TG Pro with a CLI, then rerun the "
        "same command. MTPLX will continue without fan control when --max is "
        "requested and no supported tool is present."
    ),
}


def _run_probe(command: list[str], *, timeout_s: float = 3.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except Exception as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "ok": False,
        }
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "ok": proc.returncode == 0,
    }


def _version(path: str) -> dict[str, Any]:
    for args in (["--version"], ["version"]):
        result = _run_probe([path, *args])
        if result["ok"]:
            return result
    return _run_probe([path, "--help"])


@lru_cache(maxsize=1)
def detect_thermal_control() -> dict[str, Any]:
    thermalforge = shutil.which("thermalforge")
    tgpro = shutil.which("tgpro") or shutil.which("tgpro-cli")
    tools: list[dict[str, Any]] = []
    if thermalforge:
        tools.append(
            {
                "kind": "thermalforge",
                "path": thermalforge,
                "version": _version(thermalforge),
            }
        )
    if tgpro:
        tools.append(
            {
                "kind": "tgpro",
                "path": tgpro,
                "version": _version(tgpro),
            }
        )
    selected = tools[0] if tools else None
    selected_kind = selected["kind"] if selected else "none"
    return {
        "available": selected is not None,
        "selected": selected,
        "tools": tools,
        "instructions": INSTALL_INSTRUCTIONS[selected_kind],
        "clock_anchor_enabled": os.environ.get("MTPLX_GPU_CLOCK_ANCHOR") == "1",
        "clock_anchor_policy": "explicit experimental only; never used for product claims",
    }


def _profile_command_candidates(tool: dict[str, Any], profile: str) -> list[list[str]]:
    label = PROFILE_LABELS[profile]
    path = str(tool["path"])
    kind = str(tool["kind"])
    if kind == "thermalforge":
        return [
            [path, "profile", label],
            [path, "set-profile", label],
            [path, "set", label],
            [path, "--profile", label],
        ]
    if kind == "tgpro":
        return [
            [path, "profile", label],
            [path, "set-profile", label],
            [path, "--profile", label],
        ]
    return []


def _status_command_candidates(tool: dict[str, Any]) -> list[list[str]]:
    path = str(tool["path"])
    return [
        [path, "status", "--json"],
        [path, "status"],
        [path, "sensors", "--json"],
        [path, "sensors"],
    ]


def thermal_status() -> dict[str, Any]:
    detection = detect_thermal_control()
    selected = detection.get("selected")
    if not selected:
        return {"ok": False, "detection": detection, "status": None}
    attempts = [_run_probe(command) for command in _status_command_candidates(selected)]
    first_ok = next((attempt for attempt in attempts if attempt["ok"]), None)
    return {
        "ok": first_ok is not None,
        "detection": detection,
        "status": first_ok,
        "attempts": attempts,
    }


def set_thermal_profile(profile: str, *, dry_run: bool = False) -> dict[str, Any]:
    if profile not in PROFILE_LABELS:
        raise ValueError(f"unknown thermal profile: {profile}")
    detection = detect_thermal_control()
    selected = detection.get("selected")
    if not selected:
        return {
            "ok": False,
            "profile": profile,
            "dry_run": dry_run,
            "detection": detection,
            "attempts": [],
            "message": detection["instructions"],
        }
    commands = _profile_command_candidates(selected, profile)
    if dry_run:
        return {
            "ok": True,
            "profile": profile,
            "dry_run": True,
            "detection": detection,
            "command": commands[0] if commands else None,
            "attempts": [],
        }
    attempts = []
    for command in commands:
        result = _run_probe(command, timeout_s=15.0)
        attempts.append(result)
        if result["ok"]:
            return {
                "ok": True,
                "profile": profile,
                "dry_run": False,
                "detection": detection,
                "command": command,
                "attempts": attempts,
            }
    return {
        "ok": False,
        "profile": profile,
        "dry_run": False,
        "detection": detection,
        "attempts": attempts,
        "message": (
            "Thermal tool was detected, but MTPLX could not switch profiles "
            "through its CLI. Check the tool's CLI syntax or run mtplx max --status."
        ),
    }


@contextmanager
def thermal_profile(profile: str, *, enabled: bool) -> Iterator[dict[str, Any]]:
    state: dict[str, Any] = {"enabled": bool(enabled), "start": None, "restore": None}
    if enabled:
        state["start"] = set_thermal_profile(profile)
    try:
        yield state
    finally:
        if enabled and state.get("start", {}).get("detection", {}).get("available"):
            state["restore"] = set_thermal_profile("silent")


def run_command_with_profile(
    command: list[str],
    *,
    profile: str,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    with thermal_profile(profile, enabled=True) as thermal:
        proc = subprocess.run(command, env=env, cwd=cwd, check=False)
    return {"returncode": proc.returncode, "thermal": thermal, "command": command}
