"""Interactive onboarding flow for ``mtplx start``.

Three screens for first-time users (model -> mode -> interface), and a
"same as last time?" prompt for returning users. Choices persist to
``~/.mtplx/quickstart.json`` so the next run can offer the same defaults.

The module gracefully degrades to plain stdlib ``print`` and ``input`` when
``rich`` is not available, so the onboarding works even in minimal venvs.
"""

from __future__ import annotations

import datetime
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mtplx.profiles import DEFAULT_HF_MODEL_ID

DEFAULT_HF_MODEL = DEFAULT_HF_MODEL_ID
STATE_PATH = Path("~/.mtplx/quickstart.json").expanduser()
LOCAL_SCAN_TIMEOUT_S = 3.0
LOCAL_SCAN_CLASSIFY_TIMEOUT_S = 4.0
_TIER_RANK: dict[str, int] = {
    "verified": 0,
    "arch-compatible": 1,
    "needs-verification": 2,
    "mtp-invalid": 3,
    "mtp-missing": 4,
    "backend-pending": 5,
    "no-mtp": 6,
    "incompatible": 7,
    "unknown": 8,
}


def _pretty_path(value: str | Path | None) -> str:
    """Render a filesystem path with the user's home directory collapsed."""

    if value is None:
        return ""
    text = str(value)
    if not text:
        return ""
    if "://" in text:
        return text
    try:
        path = Path(text).expanduser()
        home = Path.home()
        return "~/" + str(path.relative_to(home)) if path.is_absolute() else text
    except Exception:
        return text


# ---------- state file ------------------------------------------------------
def _state_path() -> Path:
    env = os.environ.get("MTPLX_QUICKSTART_STATE")
    return Path(env).expanduser() if env else STATE_PATH


def load_state() -> dict | None:
    """Return the last saved quickstart state, or ``None`` if absent / invalid."""

    path = _state_path()
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def save_state(state: dict) -> None:
    """Persist the chosen configuration. Adds a UTC timestamp."""

    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["saved_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


# ---------- local-folder scanning ------------------------------------------
@dataclass(frozen=True)
class ScannedModel:
    path: Path
    tier: str
    arch_id: str | None
    architecture: str | None
    error: str | None = None


def _is_model_dir(path: Path) -> bool:
    return (path / "config.json").is_file()


_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".cache",
        "__pycache__",
        "node_modules",
        "blobs",
        "refs",
        "DerivedData",
    }
)


def _scan_for_models(
    root: Path,
    *,
    max_depth: int = 4,
    cap: int = 200,
    timeout_s: float = LOCAL_SCAN_TIMEOUT_S,
) -> list[Path]:
    if _is_model_dir(root):
        return [root]

    results: list[Path] = []
    deadline = time.monotonic() + max(0.1, float(timeout_s)) if timeout_s else None

    def _expired() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    def _walk(p: Path, depth: int) -> None:
        if depth > max_depth or len(results) >= cap or _expired():
            return
        try:
            entries = sorted(p.iterdir(), key=lambda x: x.name.lower())
        except (PermissionError, OSError):
            return
        for child in entries:
            if len(results) >= cap or _expired():
                return
            if not child.is_dir():
                continue
            name = child.name
            if name in _SKIP_DIRS:
                continue
            if name.startswith(".") and depth >= 0:
                continue
            if _is_model_dir(child):
                results.append(child)
                continue
            _walk(child, depth + 1)

    _walk(root, 0)
    return results


def _scan_mtp_sidecar_exists(model_dir: Path, config: dict[str, Any]) -> bool:
    candidates: list[str] = []
    extra = config.get("mlx_lm_extra_tensors", {})
    if isinstance(extra, dict) and extra.get("mtp_file"):
        candidates.append(str(extra["mtp_file"]))
    candidates.extend(("mtp.safetensors", "mtp/weights.safetensors", "model-mtp.safetensors"))
    for rel in candidates:
        try:
            if (model_dir / rel).is_file():
                return True
        except OSError:
            continue
    return False


def _scan_embedded_mtp_keys(model_dir: Path) -> tuple[str, ...]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return ()
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return ()
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict):
        return ()
    return tuple(sorted(str(key) for key in weight_map if str(key).startswith("mtp.")))


def _classify_scanned_model(model_dir: Path) -> ScannedModel:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return ScannedModel(model_dir, "incompatible", None, None, "missing config.json")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return ScannedModel(model_dir, "unknown", None, None, str(exc)[:80])

    tcfg = config.get("text_config", config) if isinstance(config, dict) else {}
    archs = (
        (config.get("architectures") if isinstance(config, dict) else None)
        or tcfg.get("architectures")
        or []
    )
    architecture = archs[0] if archs else None
    model_type = tcfg.get("model_type") or (
        config.get("model_type") if isinstance(config, dict) else None
    )

    def _maybe_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    mtp_num_hidden_layers = max(
        _maybe_int(tcfg.get("mtp_num_hidden_layers")),
        _maybe_int(tcfg.get("num_nextn_predict_layers")),
        _maybe_int(tcfg.get("num_mtp_modules")),
        _maybe_int(config.get("num_nextn_predict_layers")) if isinstance(config, dict) else 0,
        _maybe_int(config.get("num_mtp_modules")) if isinstance(config, dict) else 0,
    )

    contract_path = model_dir / "mtplx_runtime.json"
    runtime_contract_data: dict | None = None
    runtime_contract_error: str | None = None
    if contract_path.is_file():
        try:
            runtime_contract_data = json.loads(contract_path.read_text(encoding="utf-8"))
        except Exception as exc:
            runtime_contract_error = str(exc)[:80]

    try:
        model_files = tuple(sorted(p.name for p in model_dir.glob("model*.safetensors")))
    except OSError:
        model_files = ()
    sidecar_exists = _scan_mtp_sidecar_exists(model_dir, config if isinstance(config, dict) else {})
    embedded_mtp_keys = _scan_embedded_mtp_keys(model_dir)
    mtp_artifact_exists = sidecar_exists or bool(embedded_mtp_keys)

    class _StubMTP:
        # Picker-level evidence only. The runtime still validates the sidecar
        # header and exact key layout before loading.
        exists = mtp_artifact_exists
        passes_tensor_gate = mtp_artifact_exists

    class _StubInspection:
        pass

    stub = _StubInspection()
    stub.model_dir = str(model_dir)
    stub.architecture = architecture
    stub.model_type = model_type
    stub.mtp_num_hidden_layers = mtp_num_hidden_layers
    stub.mtp = _StubMTP()
    stub.model_files = model_files
    stub.runtime_contract_data = runtime_contract_data
    stub.runtime_contract_error = runtime_contract_error
    stub.runtime_contract_path = str(contract_path) if contract_path.is_file() else None

    try:
        from mtplx.backends.registry import SUPPORTED_ARCH_IDS, compatibility_for_inspection

        verdict = compatibility_for_inspection(stub)
    except Exception as exc:
        return ScannedModel(model_dir, "unknown", None, architecture, str(exc)[:80])

    raw_tier = verdict.tier
    runtime_status = verdict.runtime_compatibility
    artifact_missing = mtp_num_hidden_layers > 0 and not mtp_artifact_exists
    if raw_tier == "verified":
        tier = "verified"
    elif verdict.can_run or raw_tier == "family-compatible-unverified":
        tier = "arch-compatible"
    elif raw_tier == "architecture-compatible-but-unverified":
        if (
            verdict.runtime_contract is not None
            and verdict.arch_id is not None
            and verdict.arch_id in SUPPORTED_ARCH_IDS
            and runtime_contract_error is None
        ):
            tier = "verified"
        elif verdict.runtime_compatibility == "missing-mtp-weights":
            tier = "mtp-missing"
        elif artifact_missing:
            tier = "mtp-missing"
        elif verdict.runtime_compatibility == "invalid-mtp-tensor-layout":
            tier = "mtp-invalid"
        elif verdict.runtime_compatibility in {"needs-contract", "needs-grafting"}:
            tier = "needs-verification"
        elif verdict.runtime_compatibility == "recognized-backend-pending":
            tier = "backend-pending"
        else:
            tier = "needs-verification"
    elif raw_tier == "no-MTP":
        tier = "no-mtp"
    elif raw_tier == "incompatible-architecture":
        tier = "backend-pending" if runtime_status == "recognized-backend-pending" else "incompatible"
    else:
        tier = "unknown"

    return ScannedModel(model_dir, tier, verdict.arch_id, architecture)


def _tier_badge(tier: str) -> tuple[str, str]:
    if tier == "verified":
        return ("Verified", "bold green")
    if tier == "arch-compatible":
        return ("Runnable (unverified)", "yellow")
    if tier == "needs-verification":
        return ("Needs MTPLX verification", "yellow")
    if tier == "mtp-invalid":
        return ("MTP weights invalid", "yellow")
    if tier == "mtp-missing":
        return ("MTP weights missing", "yellow")
    if tier == "backend-pending":
        return ("Backend not runnable yet", "dim")
    if tier == "no-mtp":
        return ("No MTP head", "dim")
    if tier == "incompatible":
        return ("Unsupported architecture", "red")
    return ("Unknown", "dim")


def _scanned_model_options(
    models: list[ScannedModel],
    root: Path,
) -> list[tuple[str, str, str]]:
    options: list[tuple[str, str, str]] = []
    for index, model in enumerate(models, start=1):
        try:
            display_name = str(model.path.relative_to(root))
        except ValueError:
            display_name = _pretty_path(model.path)
        badge, _ = _tier_badge(model.tier)
        arch = model.architecture or model.arch_id or "unknown architecture"
        subline = f"{badge}  ·  {model.error[:80]}" if model.error else f"{badge}  ·  {arch}"
        options.append((str(index), display_name, subline))
    return options


# ---------- rich helpers (with stdlib fallback) -----------------------------
def _console() -> Any | None:
    try:
        from rich.console import Console
    except ImportError:
        return None
    try:
        return Console()
    except Exception:
        return None


def _step_panel(
    *,
    step: int,
    total: int,
    title: str,
    options: list[tuple[str, str, str]],
) -> None:
    """Render a numbered onboarding step (Step N of M)."""

    _choice_panel(
        heading=f"Step {step} of {total}  ·  {title}",
        options=options,
        border_style="cyan",
    )


def _choice_panel(
    *,
    heading: str,
    options: list[tuple[str, str, str]],
    intro: str | None = None,
    border_style: str = "cyan",
) -> None:
    """Render a panel of numbered choices; falls back to plain text."""

    try:
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        print()
        print(f"  {heading}")
        print()
        if intro:
            print(f"  {intro}")
            print()
        for key, headline, subline in options:
            print(f"  {key}.  {headline}")
            if subline:
                print(f"      {subline}")
            print()
        return

    console = _console()
    if console is None:
        print()
        print(f"  {heading}")
        print()
        if intro:
            print(f"  {intro}")
            print()
        for key, headline, subline in options:
            print(f"  {key}.  {headline}")
            if subline:
                print(f"      {subline}")
            print()
        return

    body = Text()
    if intro:
        body.append(intro, style="")
        body.append("\n\n")
    for index, (key, headline, subline) in enumerate(options):
        if index > 0:
            body.append("\n\n")
        body.append(f"{key}.  ", style="bold cyan")
        body.append(headline, style="bold")
        if subline:
            body.append("\n    ")
            body.append(subline, style="dim")

    panel = Panel(
        body,
        title=Text(heading, style="bold"),
        title_align="left",
        border_style=border_style,
        padding=(1, 2),
        expand=False,
    )
    console.print()
    console.print(panel)
    console.print()


def _prompt_choice(prompt: str, choices: list[str], default: str | None = None) -> str:
    """Read a digit choice from stdin, looping until valid. Raises on Ctrl-C.

    The prompt is rendered as ``Type 1-N and press Enter [1]:`` so users with
    no prior CLI familiarity know exactly what action to take.
    """

    if choices:
        try:
            low = min(int(c) for c in choices)
            high = max(int(c) for c in choices)
            if low == high:
                hint = f"Type {low} and press Enter"
            else:
                hint = f"Type {low}-{high} and press Enter"
        except ValueError:
            hint = f"Type one of {', '.join(choices)} and press Enter"
    else:
        hint = "Type your choice and press Enter"

    suffix = f" [default {default}]" if default else ""
    while True:
        answer = input(f"  {hint}{suffix}: ").strip()
        if not answer and default:
            return default
        if answer in choices:
            return answer
        print(f"  please type one of: {', '.join(choices)}")


def _prompt_text(prompt: str, *, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"  {prompt}{suffix}: ").strip()
    if not answer and default:
        return default
    return answer


def _print_welcome() -> None:
    try:
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        print()
        print("Welcome to MTPLX. Three quick questions to get you set up.")
        print("For each step, type the number of your choice and press Enter.")
        print()
        return
    console = _console()
    if console is None:
        print()
        print("Welcome to MTPLX. Three quick questions to get you set up.")
        print("For each step, type the number of your choice and press Enter.")
        print()
        return
    body = Text()
    body.append("Welcome to MTPLX.\n\n", style="bold")
    body.append("Three quick questions to get you set up:\n", style="")
    body.append("  1. Which model?\n", style="dim")
    body.append("  2. Which runtime mode?\n", style="dim")
    body.append("  3. Browser chat or terminal chat?\n\n", style="dim")
    body.append("For each step, type the number of your choice and press Enter.", style="italic")
    panel = Panel(
        body,
        title=Text("First-time setup", style="bold cyan"),
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
        expand=False,
    )
    console.print()
    console.print(panel)
    console.print()


def _print_summary(state: dict) -> None:
    try:
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        print()
        print("  Your quickstart configuration:")
        print(f"    Model:     {state.get('model', '?')}")
        print(f"    Mode:      {mode_label(state)}")
        print(f"    Interface: {interface_label(state.get('target'))}")
        print()
        return
    console = _console()
    if console is None:
        return
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right", no_wrap=True)
    table.add_column(no_wrap=False)
    table.add_row("Model", str(state.get("model", "?")))
    table.add_row("Mode", mode_label(state))
    table.add_row("Interface", interface_label(state.get("target")))
    panel = Panel(
        table,
        title=Text("Ready to go", style="bold green"),
        title_align="left",
        border_style="green",
        padding=(1, 2),
        expand=False,
    )
    console.print()
    console.print(panel)
    console.print()


# ---------- screens ---------------------------------------------------------
def screen_model(*, configured: str | None = None) -> str:
    """Render screen 1 and return the user's choice.

    If ``configured`` is set and differs from the canonical default, it is
    surfaced as the first option (so accepting the default reuses the user's
    already-resolved local path instead of forcing a re-download).
    """

    show_configured = bool(configured) and configured != DEFAULT_HF_MODEL

    options: list[tuple[str, str, str]] = []
    if show_configured:
        options.append(
            (
                "1",
                "Use your configured model",
                str(configured),
            )
        )
        options.append(
            (
                "2",
                "Verified default",
                f"{DEFAULT_HF_MODEL} · cold-speed champion",
            )
        )
        options.append(
            (
                "3",
                "Custom Hugging Face repo",
                "e.g. Qwen/Qwen3-Next-80B-A3B-Instruct",
            )
        )
        options.append(
            (
                "4",
                "Local folder",
                "e.g. ~/models/your-model  ·  or a parent like ~/.lmstudio/models",
            )
        )
    else:
        options.append(
            (
                "1",
                "Verified default",
                f"{DEFAULT_HF_MODEL} · cold-speed champion",
            )
        )
        options.append(
            (
                "2",
                "Custom Hugging Face repo",
                "e.g. Qwen/Qwen3-Next-80B-A3B-Instruct",
            )
        )
        options.append(
            (
                "3",
                "Local folder",
                "e.g. ~/models/your-model  ·  or a parent like ~/.lmstudio/models",
            )
        )

    _step_panel(step=1, total=3, title="Choose your model", options=options)
    valid_choices = [opt[0] for opt in options]
    choice = _prompt_choice("Select", valid_choices, default="1")

    if show_configured:
        if choice == "1":
            return str(configured)
        if choice == "2":
            return DEFAULT_HF_MODEL
        if choice == "3":
            entered = _prompt_text("Hugging Face repo id (namespace/name)", default=DEFAULT_HF_MODEL)
            return entered or DEFAULT_HF_MODEL
        # choice == "4"
        return _pick_local_model(default=str(configured))

    if choice == "1":
        return DEFAULT_HF_MODEL
    if choice == "2":
        entered = _prompt_text("Hugging Face repo id (namespace/name)", default=DEFAULT_HF_MODEL)
        return entered or DEFAULT_HF_MODEL
    # choice == "3"
    return _pick_local_model(default=None)


def _pick_local_model(*, default: str | None) -> str:
    pretty_default = _pretty_path(default) if default else None
    while True:
        entered = _prompt_text("Local folder path", default=pretty_default)
        if not entered:
            print("  Path is required. Falling back to verified default.")
            return DEFAULT_HF_MODEL

        root = Path(entered).expanduser().resolve()
        if not root.exists():
            print(f"  Path does not exist: {_pretty_path(root)}")
            print()
            continue
        if not root.is_dir():
            print(f"  Not a directory: {_pretty_path(root)}")
            print()
            continue

        if _is_model_dir(root):
            return str(root)

        chosen = _scan_and_pick(root)
        if chosen is None:
            continue
        return chosen


def _scan_and_pick(root: Path) -> str | None:
    print(f"  Scanning {_pretty_path(root)} for model folders...")
    found = _scan_for_models(root)
    if found:
        print(f"  Found {len(found)} candidate model folder(s). Checking configs...")
    classified: list[ScannedModel] = []
    classify_deadline = time.monotonic() + LOCAL_SCAN_CLASSIFY_TIMEOUT_S
    for path in found:
        if classified and time.monotonic() >= classify_deadline:
            print(
                "  Compatibility scan is taking longer than expected; "
                f"showing {len(classified)} result(s)."
            )
            break
        classified.append(_classify_scanned_model(path))

    if not classified:
        print(f"  No models found under {_pretty_path(root)}.")
        print(
            "  (a model directory has a config.json at its root — "
            "for LM Studio that's <publisher>/<repo-name>/)"
        )
        print()
        return None

    classified.sort(key=lambda m: (_TIER_RANK.get(m.tier, 99), str(m.path).lower()))

    cap = 24
    visible = classified[:cap]
    hidden = len(classified) - cap
    options = _scanned_model_options(visible, root)
    options.append(
        (
            str(len(visible) + 1),
            "Type a different folder path",
            "Re-enter a model directory or a different parent.",
        )
    )

    verified = sum(1 for m in classified if m.tier == "verified")
    runnable = sum(1 for m in classified if m.tier == "arch-compatible")
    needs = sum(1 for m in classified if m.tier == "needs-verification")
    missing = sum(1 for m in classified if m.tier == "mtp-missing")
    intro = (
        f"Found {len(classified)} model(s) under {_pretty_path(root)}  ·  "
        f"{verified} verified, {runnable} runnable unverified"
    )
    if needs:
        intro += f", {needs} need verification"
    if missing:
        intro += f", {missing} missing MTP weights"
    if hidden > 0:
        intro += f"  (showing first {cap}; {hidden} not shown)"

    _choice_panel(
        heading="Pick a model",
        intro=intro,
        options=options,
        border_style="cyan",
    )

    valid = [opt[0] for opt in options]
    choice = _prompt_choice("Select", valid, default="1")
    idx = int(choice) - 1
    if idx == len(visible):
        return None
    return str(visible[idx].path)


def screen_mode() -> tuple[str, bool]:
    """Return (profile_name, max_mode_flag).

    The consumer onboarding is speed-first:

      Medium : native-MTP speed path, Apple fan curve, burst not sustained
      Max    : same path, ThermalForge pins fans at 100% for sustained speed

    Stable remains available through ``--profile stable`` / ``--profile safe``.
    """

    _step_panel(
        step=2,
        total=3,
        title="Choose a runtime mode",
        options=[
            (
                "1",
                "Medium  ·  native-MTP speed path, about 2.2x burst (not sustained)",
                "Fast speculative path with Apple's default fan curve; snappy on short replies, slower on long hot runs.",
            ),
            (
                "2",
                "Max  ·  Medium path plus fans pinned at 100%, about 2.24x (loud)",
                "Same runtime path as Medium, plus ThermalForge fan control to reduce long-reply slowdown.",
            ),
        ],
    )
    choice = _prompt_choice("Select", ["1", "2"], default="1")
    if choice == "1":
        return "performance-cold", False
    return "performance-cold", True


def screen_interface() -> str:
    """Return the target string (``openwebui`` or ``terminal``)."""

    _step_panel(
        step=3,
        total=3,
        title="Where do you want to chat?",
        options=[
            (
                "1",
                "Web UI [browser at http://127.0.0.1:8000/]",
                "Markdown rendering · live tokens-per-second · inference settings sidebar.",
            ),
            (
                "2",
                "CLI [this terminal]",
                "Streamed answers with rich styling and a stats footer.",
            ),
        ],
    )
    choice = _prompt_choice("Select", ["1", "2"], default="1")
    return "openwebui" if choice == "1" else "terminal"


def run_onboarding_screens(*, configured_model: str | None = None) -> dict:
    """Walk all three screens and return the chosen state dict.

    When the user picks Max mode but no fan controller is detected, this
    function offers to install ThermalForge automatically. If install is
    declined or fails, ``max`` is left as ``False`` so the rest of the
    pipeline doesn't promise fan-boosted speeds it can't deliver.
    """

    model = screen_model(configured=configured_model)
    profile, max_mode = screen_mode()
    if max_mode:
        max_mode = ensure_thermal_control_installed()
    target = screen_interface()
    return {
        "model": model,
        "profile": profile,
        "max": max_mode,
        "target": target,
    }


def ensure_thermal_control_installed() -> bool:
    """Detect a fan controller; offer to install ThermalForge if absent.

    Returns ``True`` when fan control is available after this call (so Max
    mode is honest), ``False`` otherwise. The caller is expected to drop
    ``args.max`` when this returns ``False``.

    The chooser screen mirrors the rest of the onboarding (numbered options,
    ``Type 1-N and press Enter`` prompt) so the UX is consistent — no more
    Y/N prompt sandwiched between numbered screens.
    """

    from mtplx.thermal import detect_thermal_control, install_thermal_control

    detection = detect_thermal_control()
    if detection.get("available"):
        return True

    _choice_panel(
        heading="Max mode setup  ·  ThermalForge",
        intro=(
            "Max mode needs a fan controller. ThermalForge (free, open "
            "source) is not installed on this Mac yet."
        ),
        options=[
            (
                "1",
                "Install ThermalForge now (recommended)",
                "Builds from source via git + Xcode CLI tools. One sudo password prompt.",
            ),
            (
                "2",
                "Skip — Max falls back to Medium",
                "No fan boost, but everything else still works.",
            ),
        ],
        border_style="yellow",
    )
    choice = _prompt_choice("Select", ["1", "2"], default="1")
    if choice == "2":
        _print_install_skipped()
        return False

    try:
        from rich.console import Console

        console = Console()
        console.rule("Installing ThermalForge", style="cyan")
        console.print(
            "[dim]Streaming git + swift + sudo output. "
            "Enter your password if prompted.[/dim]"
        )
    except Exception:
        print("\n--- Installing ThermalForge ---")
        print("Streaming git + swift + sudo output. Enter your password if prompted.")

    result = install_thermal_control()
    _print_install_result(result)
    return bool(result.get("ok") and result.get("daemon_ok") is not False)


def _print_install_skipped() -> None:
    try:
        from rich.console import Console

        console = Console()
        console.print(
            "[yellow]  Skipped. Continuing without fan boost — "
            "Max behaves like Medium.[/yellow]\n"
        )
    except Exception:
        print("  Skipped. Continuing without fan boost.\n")


def _print_install_result(result: dict) -> None:
    ok = bool(result.get("ok"))
    daemon_ok = result.get("daemon_ok")
    message = result.get("message") or ("ThermalForge installed." if ok else "Install failed.")

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        prefix = "[ok]" if ok and daemon_ok is not False else "[partial]" if ok else "[fail]"
        print(f"\n  {prefix} {message}\n")
        return

    console = _console()
    if console is None:
        prefix = "[ok]" if ok and daemon_ok is not False else "[partial]" if ok else "[fail]"
        print(f"\n  {prefix} {message}\n")
        return

    if ok and daemon_ok is not False:
        title = Text("ThermalForge ready", style="bold green")
        border = "green"
    elif ok:
        title = Text("ThermalForge installed (daemon pending)", style="bold yellow")
        border = "yellow"
    else:
        title = Text("ThermalForge install did not complete", style="bold red")
        border = "red"
    panel = Panel(
        Text(message, style=""),
        title=title,
        title_align="left",
        border_style=border,
        padding=(1, 2),
        expand=False,
    )
    console.print()
    console.print(panel)
    console.print()


# ---------- top-level flow --------------------------------------------------
def confirm_same_as_last(last: dict) -> bool:
    """Ask the user whether to reuse the last configuration."""

    try:
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        print()
        print("  Last time you used:")
        print(f"    Model:     {last.get('model', '?')}")
        print(f"    Mode:      {mode_label(last)}")
        print(f"    Interface: {interface_label(last.get('target'))}")
        print()
        answer = input("  Use the same configuration? [Y/n] ").strip().lower()
        return answer in {"", "y", "yes", "same"}

    console = _console()
    if console is None:
        print()
        print("  Last time you used:")
        print(f"    Model:     {last.get('model', '?')}")
        print(f"    Mode:      {mode_label(last)}")
        print(f"    Interface: {interface_label(last.get('target'))}")
        print()
        answer = input("  Use the same configuration? [Y/n] ").strip().lower()
        return answer in {"", "y", "yes", "same"}

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right", no_wrap=True)
    table.add_column(no_wrap=False)
    table.add_row("Model", str(last.get("model", "?")))
    table.add_row("Mode", mode_label(last))
    table.add_row("Interface", interface_label(last.get("target")))
    panel = Panel(
        table,
        title=Text("Welcome back", style="bold cyan"),
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
        expand=False,
    )
    console.print()
    console.print(panel)
    console.print()
    answer = input("  Use the same configuration? [Y/n] ").strip().lower()
    return answer in {"", "y", "yes", "same"}


def _quickstart_state_is_reusable(last: dict | None) -> bool:
    if not isinstance(last, dict):
        return False
    # Stable/safe remains available by explicit flag, but old saved states
    # should not keep showing it in the speed-first onboarding path.
    return str(last.get("profile") or "") == "performance-cold"


def run_quickstart_flow(
    *,
    fresh: bool = False,
    configured_model: str | None = None,
) -> dict | None:
    """Decide between 'same as last time' or fresh onboarding.

    Returns the chosen config dict (with keys ``model``, ``profile``, ``max``,
    ``target``), or ``None`` if the user aborted (Ctrl-C / EOF).

    ``configured_model`` is the model already resolved from
    ``~/.mtplx/config.toml`` (if any). When set and not the canonical default,
    screen 1 surfaces it as the top choice so accepting the default does not
    force a re-download of an HF mirror that's already on disk elsewhere.
    """

    last = None if fresh else load_state()
    if not _quickstart_state_is_reusable(last):
        last = None
    try:
        if last and not fresh:
            if confirm_same_as_last(last):
                # If the saved state says Max but ThermalForge has gone away
                # (uninstalled, new machine) we must re-offer the install
                # rather than silently dump a JSON warning at runtime.
                if last.get("max") and not ensure_thermal_control_installed():
                    last = dict(last)
                    last["max"] = False
                    save_state(last)
                return last
        else:
            _print_welcome()
        choice = run_onboarding_screens(configured_model=configured_model)
        save_state(choice)
        _print_summary(choice)
        return choice
    except (KeyboardInterrupt, EOFError):
        try:
            print()
        except Exception:  # pragma: no cover - stdout is closed
            pass
        return None


# ---------- label helpers ---------------------------------------------------
def mode_label(state: dict) -> str:
    profile = state.get("profile", "performance-cold")
    if state.get("max"):
        return "Max  ·  Medium path plus fans pinned at 100%, ~2.24x"
    if profile == "performance-cold":
        return "Medium  ·  native-MTP speed path, ~2.2x burst (not sustained)"
    if profile == "stable":
        return "Stable  ·  exact/staged long-reply path"
    return str(profile)


def interface_label(target: str | None) -> str:
    if target in ("openwebui", "open-webui", "web"):
        return "Web UI  ·  browser"
    if target in ("cli", "terminal"):
        return "CLI  ·  this terminal"
    return target or "?"
