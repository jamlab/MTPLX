from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from mtplx.cli import build_parser, main
from mtplx.commands import public
from mtplx.profiles import (
    DEFAULT_FP16_HF_MODEL_ID,
    DEFAULT_PUBLIC_MODEL_ID,
    LEGACY_OPTIMIZED_PUBLIC_MODEL_ID,
    QUALITY_PUBLIC_MODEL_ID,
)
from mtplx.version import DISPLAY_VERSION, __version__


def test_version_metadata_matches_package_metadata():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]

    assert __version__ == project["version"]
    assert DISPLAY_VERSION == __version__


def test_version_command_without_subcommand(capsys):
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0

    captured = capsys.readouterr().out
    assert f"mtplx {DISPLAY_VERSION} ({__version__})" in captured


def test_runtime_mode_display_respects_ar_mode():
    assert public._runtime_mode_display("sustained") == "Sustained MTP"
    assert (
        public._runtime_mode_display("sustained", generation_mode="ar")
        == "Sustained AR"
    )
    assert (
        public._runtime_mode_display(
            "sustained",
            max_mode=True,
            generation_mode="ar",
        )
        == "Sustained Max AR"
    )


def test_empty_cli_shows_friendly_consumer_help(capsys):
    code = main([])

    captured = capsys.readouterr().out
    assert code == 0
    # Compact help: ASCII banner + version pill + Commands + Examples + footer.
    assert f"v{DISPLAY_VERSION}" in captured
    assert "Native MTP speculative decoding" in captured
    assert "mtplx quickstart" in captured
    assert "Prepare config and the model cache" in captured
    assert "mtplx start" in captured
    assert "mtplx help advanced" in captured
    assert "runtime-smoke" not in captured
    assert "capture-commit-equivalence" not in captured


def test_hardware_inspect_json(monkeypatch, capsys):
    import mtplx.hardware as hardware

    monkeypatch.setattr(
        hardware,
        "inspect_hardware",
        lambda: {
            "chip": "Apple M5 Max",
            "macos_version": "26.2",
            "mlx_version": "0.31.0",
            "hardware_acceleration_eligible": True,
            "hardware_acceleration_confirmed": False,
            "warnings": ["Eligibility is not proof."],
        },
    )

    code = main(["hardware", "inspect", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["chip"] == "Apple M5 Max"
    assert payload["hardware_acceleration_eligible"] is True
    assert payload["hardware_acceleration_confirmed"] is False


def test_bench_prefill_ladder_dry_run_json(monkeypatch, capsys):
    import mtplx.prefill_bench as prefill_bench

    monkeypatch.setattr(
        prefill_bench,
        "inspect_hardware",
        lambda: {
            "chip": "Apple M5 Max",
            "hardware_acceleration_eligible": True,
            "hardware_acceleration_confirmed": False,
        },
    )

    code = main(
        [
            "bench",
            "prefill-ladder",
            "--contexts",
            "512,1k",
            "--max-tokens",
            "8",
            "--defer-verify-hidden-eval",
            "--verify-hidden-mode",
            "logits-first-committed-slice",
            "--dry-run",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["kind"] == "prefill_ladder"
    assert payload["dry_run"] is True
    assert payload["seed"] == 0
    assert payload["vary_seed_by_context"] is False
    assert payload["inter_context_cache_cleanup"]["enabled"] is True
    assert payload["inter_context_cache_cleanup"]["events"] == 0
    assert payload["contexts"] == [512, 1024]
    assert payload["rows"] == []
    assert payload["prompt"]["style"] == "coding-agent"
    assert payload["prompt"]["format"] == "chat"
    assert payload["prompt"]["enable_thinking"] is False
    assert payload["prompt"]["policy"] == "coding_agent_tail_v2"
    assert payload["prompt"]["tail_sha256"]
    assert payload["prompt"]["release_valid"] is True
    assert payload["prefill_layout"]["requested"] == "profile"
    assert payload["prefill_layout"]["env_value"] is None
    assert payload["defer_verify_hidden_eval_override"] is True
    assert payload["verify_hidden_mode_override"] == "logits_first_committed_slice"
    assert payload["env"]["MTPLX_DEFER_VERIFY_HIDDEN_EVAL"] == "1"
    assert payload["env"]["MTPLX_VERIFY_HIDDEN_MODE"] == "logits_first_committed_slice"
    assert payload["recommended_plugged_in_commands"]
    assert "--prompt-format chat" in payload["recommended_plugged_in_commands"][0]
    assert "--disable-thinking" in payload["recommended_plugged_in_commands"][0]
    assert payload["profile"]["env"]["MTPLX_LAZY_VERIFY_LOGITS"] == "1"
    assert payload["profile"]["env"]["MTPLX_BATCH_TARGET_ARRAYS"] == "1"
    assert payload["profile"]["env"]["MTPLX_PREFILL_CHUNK_CACHE_CLEANUP"] == "1"
    assert payload["profile"]["env"]["MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY"] == "auto"
    assert payload["profile"]["env"]["MTPLX_PREFILL_OMLX_EXTERNAL"] == "1"
    assert payload["profile"]["env"]["MTPLX_DEFER_VERIFY_HIDDEN_EVAL"] == "1"
    assert payload["profile"]["env"]["MTPLX_VERIFY_HIDDEN_MODE"] == "logits_first_committed_slice"
    assert payload["profile"]["env"]["MTPLX_LONG_CONTEXT_MTP_DEPTH_POLICY"] == "auto"
    assert payload["profile"]["env"]["MTPLX_LONG_CONTEXT_MTP_DEPTH_THRESHOLD"] == "98304"
    assert payload["profile"]["env"]["MTPLX_LONG_CONTEXT_MTP_DEPTH"] == "2"
    assert payload["profile"]["env"]["MTPLX_VLLM_METAL_PAGED_TURBOQUANT"] == "0"


def test_server_cli_surfaces_default_to_sustained_profile():
    parser = build_parser()

    quickstart_args = parser.parse_args(["quickstart"])
    serve_args = parser.parse_args(["serve", "--yes"])

    assert quickstart_args.profile == "sustained"
    assert serve_args.profile == "sustained"


def test_shell_banner_env_suppresses_compact_help_ascii(monkeypatch, capsys):
    monkeypatch.setenv("MTPLX_SHELL_BANNER_SHOWN", "1")

    code = main([])

    captured = capsys.readouterr().out
    assert code == 0
    assert "███╗" not in captured
    assert "Commands" in captured
    assert "mtplx start" in captured


def test_render_banner_respects_shell_banner_env(monkeypatch, capsys):
    from mtplx.ui.banner import render_banner

    monkeypatch.setenv("MTPLX_SHELL_BANNER_SHOWN", "1")
    render_banner(no_color=True)

    assert capsys.readouterr().out == ""


def test_top_level_help_is_friendly(capsys):
    code = main(["--help"])

    captured = capsys.readouterr().out
    assert code == 0
    assert "Commands" in captured
    assert "Examples" in captured
    assert "mtplx quickstart" in captured
    assert "mtplx help <command>" in captured
    assert "positional arguments" not in captured


def test_advanced_help_keeps_lab_tools_discoverable(capsys):
    code = main(["help", "advanced"])

    captured = capsys.readouterr().out
    assert code == 0
    assert "MTPLX advanced tools" in captured
    assert "mtplx bench" in captured
    assert "profile *" in captured
    assert "runtime-smoke" in captured


def test_help_with_no_topic_is_verbose_not_compact(capsys):
    """Bare `mtplx` shows compact help; `mtplx help` must show the verbose view."""
    capsys.readouterr()
    main([])
    compact = capsys.readouterr().out
    code = main(["help"])
    verbose = capsys.readouterr().out
    assert code == 0
    assert len(verbose) > len(compact), "`mtplx help` must be more verbose than bare `mtplx`"
    # Verbose-only sections.
    assert "Overview" in verbose
    assert "Help subtopics" in verbose
    assert "mtplx help commands" in verbose
    assert "mtplx help flags" in verbose
    assert "mtplx help advanced" in verbose
    # Bare `mtplx` does not list the subtopics inline.
    assert "Help subtopics" not in compact


def test_help_commands_lists_consumer_and_advanced_surfaces(capsys):
    code = main(["help", "commands"])

    captured = capsys.readouterr().out
    assert code == 0
    assert "MTPLX commands" in captured
    assert "Consumer commands" in captured
    # Consumer rows
    assert "quickstart" in captured
    assert "setup" in captured
    assert "models" in captured
    # Advanced rows
    assert "bench *" in captured
    assert "runtime-smoke" in captured


def test_help_flags_lists_every_command_flag(capsys):
    code = main(["help", "flags"])

    captured = capsys.readouterr().out
    assert code == 0
    assert "MTPLX flags" in captured
    assert "Top-level options" in captured
    # Several known flags across multiple commands must appear.
    assert "--temperature" in captured
    assert "--top-p" in captured
    assert "--top-k" in captured
    assert "--port" in captured
    assert "--profile" in captured
    assert "--max-tokens" in captured


def test_main_menu_advertises_help_command(capsys):
    """`help` must appear in the bare `mtplx` command list as one of the first commands."""
    code = main([])

    captured = capsys.readouterr().out
    assert code == 0
    # `help` appears in the Commands section, not just as part of the footer.
    assert "  help " in captured or "help         " in captured
    # Listed near the top: must come before the lab-only commands.
    quickstart_pos = captured.find("start")
    help_pos = captured.find("\n  help")
    inspect_pos = captured.find("inspect")
    assert quickstart_pos != -1 and help_pos != -1 and inspect_pos != -1
    assert quickstart_pos < help_pos < inspect_pos


def test_start_help_is_a_user_journey(capsys):
    code = main(["help", "start"])

    captured = capsys.readouterr().out
    assert code == 0
    assert "MTPLX start" in captured
    # Help leads with the interactive onboarding (model/mode/where) instead
    # of the old browser/CLI bifurcation.
    assert "Interactive end-to-end setup" in captured
    assert "What gets asked" in captured
    assert "Sustained" in captured
    assert "Sustained Max" in captured
    assert "Burst" in captured
    assert "ThermalForge" in captured  # fan-backed modes advertise fan control
    # Power-user shortcuts still showcased.
    assert "mtplx start --fresh" in captured
    assert "mtplx start --max" in captured
    assert "mtplx start cli" in captured
    assert "mtplx start pi" in captured
    assert "mtplx start opencode" in captured
    assert "mtplx start --download" in captured
    assert "/speed" in captured
    assert "Aliases:" in captured
    assert "openwebui" in captured
    assert "usage: mtplx" not in captured


def test_unknown_command_is_targeted_not_argparse_dump(capsys):
    code = main(["wut"])

    captured = capsys.readouterr().out
    assert code == 2
    assert "Unknown command: wut" in captured
    assert "mtplx setup" in captured
    assert "usage: mtplx" not in captured


def test_start_dry_run_is_consumer_friendly(monkeypatch, tmp_path, capsys):
    """The default start opens the browser chat; the terminal flow lives behind `cli`."""
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))

    code = main(
        [
            "start",
            "cli",
            "--dry-run",
            "--model",
            "models/example",
            "--yes",
        ]
    )

    captured = capsys.readouterr().out
    assert code == 0
    assert "MTPLX start" in captured
    assert "model: models/example" in captured
    assert "profile: sustained" in captured
    assert "then: load once -> chat in this terminal -> stream output -> show speed stats" in captured


def test_start_auto_default_can_route_to_fp16(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setenv("MTPLX_DEFAULT_MODEL_VARIANT", "fp16")

    code = main(["start", "cli", "--dry-run", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["model"] == DEFAULT_FP16_HF_MODEL_ID
    assert payload["default_model_selection"]["variant"] == "fp16"
    assert payload["default_model_selection"]["precision"] == "FP16"


def test_start_explicit_model_bypasses_auto_default(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setenv("MTPLX_DEFAULT_MODEL_VARIANT", "fp16")

    code = main(["start", "cli", "--dry-run", "--json", "--model", "local/custom"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["model"] == "local/custom"
    assert payload["default_model_selection"] is None


def test_quality_model_ref_uses_quality_public_model_id():
    value = public._public_model_id_for_ref(
        "/tmp/Qwen3.6-27B-MTPLX-Optimized-Quality",
        default_model_id="mtplx-qwen36-27b-optimized-speed",
    )

    assert value == QUALITY_PUBLIC_MODEL_ID


def test_legacy_optimized_model_ref_uses_neutral_public_model_id():
    value = public._public_model_id_for_ref(
        "/tmp/Qwen3.6-27B-MTPLX-Optimized",
        default_model_id=DEFAULT_PUBLIC_MODEL_ID,
    )

    assert value == LEGACY_OPTIMIZED_PUBLIC_MODEL_ID


def test_explicit_model_id_wins_over_loaded_artifact_identity():
    args = SimpleNamespace(
        model="/tmp/Qwen3.6-27B-MTPLX-Optimized-Quality",
        model_id="custom-served-id",
        _cli_flags={"model-id"},
    )

    assert public._public_model_id_for_args(args, args.model) == "custom-served-id"


def test_start_default_target_is_browser(monkeypatch, tmp_path):
    """`mtplx start` (no target) must dry-run as the openwebui browser path."""
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    args = SimpleNamespace(
        target=None,
        model="models/example",
        cache_dir=None,
        download=False,
        dry_run=True,
        json=True,
        yes=True,
        prompt=None,
        profile="performance-cold",
        show_stats=True,
        unsafe_force_unverified=False,
        host="127.0.0.1",
        port=8000,
        model_id=None,
    )
    code = public.cmd_quickstart_public(args)
    assert code == 0


def test_start_target_aliases_route_correctly(monkeypatch, tmp_path, capsys):
    """Target aliases normalize to the surface MTPLX actually starts."""
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))

    def run(target: str | None) -> dict:
        argv = ["start"]
        if target:
            argv.append(target)
        argv += ["--dry-run", "--json", "--model", "models/example", "--yes"]
        capsys.readouterr()
        code = main(argv)
        captured = capsys.readouterr().out
        assert code == 0, captured
        return json.loads(captured)

    assert run(None)["target"] == "openwebui"
    assert run("web")["target"] == "openwebui"
    assert run("openwebui")["target"] == "openwebui"
    assert run("open-webui")["target"] == "openwebui"
    assert run("cli")["target"] == "terminal"
    assert run("terminal")["target"] == "terminal"
    assert run("pi")["target"] == "pi"
    assert run("pie")["target"] == "pi"
    assert run("opencode")["target"] == "opencode"
    assert run("open-code")["target"] == "opencode"
    assert run("oc")["target"] == "opencode"


def test_start_opencode_dry_run_json_writes_no_hidden_cap(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(tmp_path / "opencode.json"))

    code = main(
        [
            "start",
            "opencode",
            "--dry-run",
            "--json",
            "--model",
            "models/example",
            "--yes",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["target"] == "opencode"
    assert payload["opencode"]["api_base_url"] == "http://127.0.0.1:18083/v1"
    assert payload["opencode"]["model_ref"].startswith("mtplx/")
    assert payload["opencode"]["no_hidden_max_tokens"] is True
    assert "maxTokens" not in json.dumps(payload["opencode"]["config"])
    assert payload["opencode"]["provider"]["models"]


def test_terminal_quickstart_max_uses_verified_max_session(monkeypatch):
    calls: list[str] = []

    class FakeMaxSession:
        def __init__(self, **_kwargs):
            self.thermal = {"enabled": True}

        def start(self):
            calls.append("start")
            return True

        def stop(self):
            calls.append("stop")
            self.thermal["restore"] = {"ok": True}
            return {"ok": True}

    fake_runtime = ModuleType("mtplx.runtime")
    fake_runtime.load = lambda *a, **kw: SimpleNamespace(tokenizer=object())
    fake_draft = ModuleType("mtplx.draft_lm_head")
    fake_draft._install_draft_lm_head = lambda *a, **kw: {"installed": True}

    monkeypatch.setitem(sys.modules, "mtplx.runtime", fake_runtime)
    monkeypatch.setitem(sys.modules, "mtplx.draft_lm_head", fake_draft)
    monkeypatch.setattr("mtplx.thermal.MaxSession", FakeMaxSession)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr(
        public,
        "_quickstart_generate",
        lambda **kw: {"text": "ok", "streamed": True, "validations": [], "stats": {}},
    )

    args = SimpleNamespace(
        profile="performance-cold",
        max=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        depth=3,
        prompt="hello",
        show_stats=False,
    )

    rc = public._quickstart_run_terminal_chat(args, runtime_model="/tmp/model", inspection={})

    assert rc == 0
    assert calls == ["start", "stop"]


def test_one_shot_max_uses_verified_max_session(monkeypatch):
    calls: list[str] = []

    class FakeMaxSession:
        def __init__(self, **_kwargs):
            self.thermal = {"enabled": True}

        def start(self):
            calls.append("start")
            return True

        def stop(self):
            calls.append("stop")
            self.thermal["restore"] = {"ok": True}
            return {"ok": True}

    fake_runtime = ModuleType("mtplx.runtime")
    fake_runtime.load = lambda *a, **kw: SimpleNamespace(tokenizer=object())
    fake_schema = ModuleType("mtplx.benchmarks.schema")
    fake_schema.PromptCase = lambda **kw: SimpleNamespace(**kw)
    fake_schema.encode_prompt_case = lambda *a, **kw: [1, 2, 3]
    fake_generation = ModuleType("mtplx.generation")
    fake_generation.generate_mtpk = lambda *a, **kw: SimpleNamespace(
        text="ok",
        tokens=[1],
        stats=SimpleNamespace(generated_tokens=1, tok_s=1.0, verify_time_s=0.0, verify_calls=0),
    )
    fake_generation.generate_ar = fake_generation.generate_mtpk
    fake_sampling = ModuleType("mtplx.sampling")
    fake_sampling.SamplerConfig = lambda **kw: SimpleNamespace(**kw)

    monkeypatch.setitem(sys.modules, "mtplx.runtime", fake_runtime)
    monkeypatch.setitem(sys.modules, "mtplx.benchmarks.schema", fake_schema)
    monkeypatch.setitem(sys.modules, "mtplx.generation", fake_generation)
    monkeypatch.setitem(sys.modules, "mtplx.sampling", fake_sampling)
    monkeypatch.setattr("mtplx.thermal.MaxSession", FakeMaxSession)
    monkeypatch.setattr(public, "_resolve_runtime_model_path", lambda model, cache_dir=None: ("/tmp/model", None))
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda runtime_model, *, unsafe_force_unverified, yes: ({}, None),
    )

    args = SimpleNamespace(
        prompt="hello",
        prompt_arg=None,
        model="/tmp/model",
        cache_dir=None,
        unsafe_force_unverified=False,
        yes=True,
        profile="performance-cold",
        max=True,
        system=None,
        max_tokens=8,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        depth=3,
        seed=0,
        expect_python=False,
    )

    code, payload, _validations = public._generate_one_shot_public(args, command="run")

    assert code == 0
    assert payload["thermal"]["restore"]["ok"] is True
    assert calls == ["start", "stop"]


def test_start_parser_accepts_target_choices():
    """Parser must accept the new `web` and `cli` target literals."""
    parser = build_parser()
    for target in (
        "web",
        "cli",
        "openwebui",
        "open-webui",
        "terminal",
        "pi",
        "pie",
        "opencode",
        "open-code",
        "oc",
    ):
        args = parser.parse_args(["start", target, "--dry-run"])
        assert args.target == target
    # Default target (no positional) is now None — the absence of an explicit
    # target is what tells the handler to run the interactive onboarding flow
    # (or fall back to "web" when non-interactive). The `--fresh` flag is also
    # accepted for forcing the full onboarding.
    args_default = parser.parse_args(["start", "--dry-run"])
    assert args_default.target is None
    args_fresh = parser.parse_args(["start", "--fresh", "--dry-run"])
    assert args_fresh.fresh is True
    args_no_mtp = parser.parse_args(["start", "cli", "--no-mtp", "--dry-run"])
    assert args_no_mtp.no_mtp is True
    args_mtp = parser.parse_args(["start", "cli", "--mtp", "--dry-run"])
    assert args_mtp.no_mtp is False


def test_start_help_mentions_target_only_ar_mtp_controls(capsys):
    code = main(["help", "start"])

    captured = capsys.readouterr().out
    assert code == 0
    assert "--no-mtp" in captured
    assert "target-only AR generation" in captured
    assert "/mtp off" in captured
    assert "/stats" in captured


def test_mtp_toggle_flags_parse_on_public_generation_surfaces():
    parser = build_parser()

    cases = [
        ["start", "cli", "--no-mtp", "--dry-run"],
        ["quickstart", "--no-mtp"],
        ["serve", "--no-mtp"],
        ["ask", "hello", "--no-mtp"],
        ["run", "hello", "--no-mtp"],
        ["chat", "--prompt", "hello", "--no-mtp"],
    ]
    for argv in cases:
        assert parser.parse_args(argv).no_mtp is True

    mtp_cases = [
        ["start", "cli", "--mtp", "--dry-run"],
        ["quickstart", "--mtp"],
        ["serve", "--mtp"],
        ["ask", "hello", "--mtp"],
        ["run", "hello", "--mtp"],
        ["chat", "--prompt", "hello", "--mtp"],
    ]
    for argv in mtp_cases:
        assert parser.parse_args(argv).no_mtp is False


def test_cli_response_cap_defaults_to_remaining_context():
    parser = build_parser()

    quickstart = parser.parse_args(["start", "cli", "--dry-run"])
    ask = parser.parse_args(["ask", "hello"])
    run = parser.parse_args(["run", "hello"])
    chat = parser.parse_args(["chat", "--prompt", "hello"])

    assert quickstart.max_tokens is None
    assert ask.max_tokens is None
    assert run.max_tokens is None
    assert chat.max_tokens is None


def test_cli_reasoning_flags_parse_without_being_chat_text():
    parser = build_parser()

    quickstart = parser.parse_args(["start", "cli", "--reasoning", "on"])
    run = parser.parse_args(["run", "hello", "--reasoning", "off"])
    serve = parser.parse_args(["serve", "--reasoning", "auto"])

    assert quickstart.reasoning == "on"
    assert run.reasoning == "off"
    assert serve.reasoning == "auto"


def test_start_missing_model_suggests_download(monkeypatch, capsys):
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (None, {"detail": "not cached"}),
    )
    args = SimpleNamespace(
        command="start",
        model="mtplx/example",
        cache_dir="/tmp/mtplx-models",
        download=False,
        dry_run=False,
        json=False,
        yes=True,
        prompt=None,
        profile="stable",
        show_stats=True,
        unsafe_force_unverified=False,
    )

    code = public.cmd_quickstart_public(args)

    captured = capsys.readouterr().out
    assert code == 1
    assert "MTPLX start" in captured
    assert "model is not available locally" in captured
    assert "detail: not cached" in captured
    assert "try: mtplx start --download" in captured
    assert "try: mtplx start --model /path/to/model" in captured


def test_quickstart_short_reply_reports_decode_tps():
    line = public._quickstart_stats_line(
        {
            "profile": {"name": "performance-cold"},
            "stats": {
                "generated_tokens": 10,
                "end_to_end_tok_s": 28.48,
                "elapsed_s": 0.351,
                "prompt_eval_time_s": 0.0,
                "verify_ms_per_call": 54.8,
            },
        }
    )

    assert "10 tokens in 0.35s decode" in line
    assert "28.49 tok/s" in line
    assert "54.8 ms/verify" in line


def test_quickstart_short_reply_prefers_decode_tps_and_labels_live_window():
    line = public._quickstart_stats_line(
        {
            "profile": {"name": "performance-cold"},
            "stats": {
                "generated_tokens": 10,
                "stream_tok_s": 42.0,
                "decode_tok_s": 28.48,
                "end_to_end_tok_s": 25.0,
                "decode_elapsed_s": 0.351,
                "verify_ms_per_call": 54.8,
                "verify_calls": 5,
                "accepted_by_depth": [4, 3, 2],
                "ttft_s": 0.08,
            },
        }
    )

    assert "28.48 tok/s" in line
    assert "live_window=42.00" in line
    assert "total=25.00" in line
    assert "5 verify calls" in line
    assert "accept=[4, 3, 2]" in line
    assert "ttft=0.08s" in line
    assert "short sample" not in line


def test_quickstart_tiny_reply_prefers_decode_over_noisy_stream_window():
    line = public._quickstart_stats_line(
        {
            "profile": {"name": "performance-cold"},
            "stats": {
                "generated_tokens": 3,
                "stream_tok_s": 12.24,
                "decode_tok_s": 34.68,
                "end_to_end_tok_s": 8.73,
                "decode_elapsed_s": 0.09,
                "verify_ms_per_call": 55.8,
                "verify_calls": 1,
            },
        }
    )

    assert "34.68 tok/s" in line
    assert "live_window=12.24" in line
    assert "total=8.73" in line


def test_quickstart_long_reply_uses_decode_tps():
    line = public._quickstart_stats_line(
        {
            "profile": {"name": "performance-cold"},
            "stats": {
                "generated_tokens": 192,
                "end_to_end_tok_s": 48.0,
                "elapsed_s": 4.0,
                "prompt_eval_time_s": 0.2,
                "verify_ms_per_call": 60.2,
            },
        }
    )

    assert "192 tokens" in line
    assert "50.53 tok/s" in line
    assert "total=48.00" in line
    assert "60.2 ms/verify" in line


def test_quickstart_incremental_decoder_streams_word_boundaries():
    class TinyTokenizer:
        def decode(self, tokens, **_kwargs):
            return "".join(chr(token) for token in tokens)

    decoder = public._QuickstartIncrementalTokenDecoder(TinyTokenizer())

    assert decoder.feed([104, 101, 108, 108, 111]) == ""
    assert decoder.feed([32]) == "hello "
    assert decoder.feed([119, 111, 114, 108, 100]) == ""
    assert decoder.finish() == "world"


def test_quickstart_generation_default_uses_remaining_model_context(monkeypatch, tmp_path):
    captured: dict[str, int] = {}

    class TinyTokenizer:
        model_max_length = 100

        def apply_chat_template(self, *_args, **kwargs):
            captured["enable_thinking"] = kwargs.get("enable_thinking")
            return list(range(12))

    fake_generation = ModuleType("mtplx.generation")

    def fake_generate_mtpk(*_args, **kwargs):
        captured["max_tokens"] = kwargs["max_tokens"]
        return SimpleNamespace(
            text="ok",
            stats=SimpleNamespace(
                generated_tokens=1,
                tok_s=1.0,
                elapsed_s=1.0,
                prompt_eval_time_s=0.0,
                verify_time_s=0.0,
                target_forward_time_s=1.0,
                repair_time_s=0.0,
                draft_time_s=0.0,
                verify_calls=0,
                accepted_by_depth=[],
                drafted_by_depth=[],
                correction_tokens=0,
                bonus_tokens=0,
            ),
        )

    fake_generation.generate_mtpk = fake_generate_mtpk
    fake_generation.generate_ar = fake_generate_mtpk
    fake_sampling = ModuleType("mtplx.sampling")
    fake_sampling.SamplerConfig = lambda **kw: SimpleNamespace(**kw)

    monkeypatch.setitem(sys.modules, "mtplx.generation", fake_generation)
    monkeypatch.setitem(sys.modules, "mtplx.sampling", fake_sampling)

    rt = SimpleNamespace(tokenizer=TinyTokenizer(), model_path=tmp_path)
    payload = public._quickstart_generate(
        rt=rt,
        inspection={},
        profile=SimpleNamespace(to_dict=lambda: {"name": "stable"}),
        args=SimpleNamespace(
            system=None,
            max_tokens=None,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            depth=3,
            seed=0,
        ),
        prompt="hello",
        history=[],
        turn_index=0,
    )

    assert captured["max_tokens"] == 88
    assert captured["enable_thinking"] is True
    assert payload["stats"]["max_tokens"] == 88
    assert payload["stats"]["remaining_context_tokens"] == 88
    assert payload["stats"]["reasoning"] == "on"


def test_quickstart_generation_no_mtp_uses_ar(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class TinyTokenizer:
        model_max_length = 100

        def apply_chat_template(self, *_args, **_kwargs):
            return list(range(12))

    def stats() -> SimpleNamespace:
        return SimpleNamespace(
            generated_tokens=1,
            tok_s=1.0,
            elapsed_s=1.0,
            prompt_eval_time_s=0.0,
            verify_time_s=0.0,
            target_forward_time_s=1.0,
            repair_time_s=0.0,
            draft_time_s=0.0,
            verify_calls=0,
            accepted_by_depth=[],
            drafted_by_depth=[],
            correction_tokens=0,
            bonus_tokens=0,
        )

    fake_generation = ModuleType("mtplx.generation")

    def fake_generate_ar(*_args, **kwargs):
        captured["mode"] = "ar"
        captured["max_tokens"] = kwargs["max_tokens"]
        return SimpleNamespace(text="ok", stats=stats())

    def fake_generate_mtpk(*_args, **_kwargs):  # pragma: no cover - must not be used
        captured["mode"] = "mtp"
        return SimpleNamespace(text="wrong", stats=stats())

    fake_generation.generate_ar = fake_generate_ar
    fake_generation.generate_mtpk = fake_generate_mtpk
    fake_sampling = ModuleType("mtplx.sampling")
    fake_sampling.SamplerConfig = lambda **kw: SimpleNamespace(**kw)

    monkeypatch.setitem(sys.modules, "mtplx.generation", fake_generation)
    monkeypatch.setitem(sys.modules, "mtplx.sampling", fake_sampling)

    payload = public._quickstart_generate(
        rt=SimpleNamespace(tokenizer=TinyTokenizer(), model_path=tmp_path),
        inspection={},
        profile=SimpleNamespace(to_dict=lambda: {"name": "stable"}),
        args=SimpleNamespace(
            system=None,
            max_tokens=None,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            depth=3,
            seed=0,
            no_mtp=True,
        ),
        prompt="hello",
        history=[],
        turn_index=0,
    )

    assert captured == {"mode": "ar", "max_tokens": 88}
    assert payload["stats"]["generation_mode"] == "ar"
    assert payload["stats"]["mtp_depth"] == 0
    assert payload["stats"]["verify_calls"] == 0
    assert payload["stats"]["accepted_by_depth"] == []
    assert payload["stats"]["drafted_by_depth"] == []


def test_quickstart_mtp_slash_command_toggles_next_turn(capsys):
    args = SimpleNamespace(no_mtp=False)
    runtime = SimpleNamespace(mtp_enabled=True)

    assert public._handle_quickstart_mtp_command(args, "/mtp status", runtime=runtime) is True
    assert public._handle_quickstart_mtp_command(args, "/mtp off", runtime=runtime) is True
    assert args.no_mtp is True
    assert public._handle_quickstart_mtp_command(args, "/mtp on", runtime=runtime) is True
    assert args.no_mtp is False

    captured = capsys.readouterr().out
    assert "MTP: on" in captured
    assert "MTP: off for the next turn" in captured
    assert "MTP: on for the next turn" in captured


def test_quickstart_generation_reasoning_on_passes_enable_thinking(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class TinyTokenizer:
        model_max_length = 64

        def apply_chat_template(self, *_args, **kwargs):
            captured["enable_thinking"] = kwargs.get("enable_thinking")
            return [1, 2, 3]

    fake_generation = ModuleType("mtplx.generation")
    fake_generation.generate_mtpk = lambda *_args, **_kwargs: SimpleNamespace(
        text="ok",
        stats=SimpleNamespace(
            generated_tokens=1,
            tok_s=1.0,
            elapsed_s=1.0,
            prompt_eval_time_s=0.0,
            verify_time_s=0.0,
            target_forward_time_s=1.0,
            repair_time_s=0.0,
            draft_time_s=0.0,
            verify_calls=0,
            accepted_by_depth=[],
            drafted_by_depth=[],
            correction_tokens=0,
            bonus_tokens=0,
        ),
    )
    fake_generation.generate_ar = fake_generation.generate_mtpk
    fake_sampling = ModuleType("mtplx.sampling")
    fake_sampling.SamplerConfig = lambda **kw: SimpleNamespace(**kw)

    monkeypatch.setitem(sys.modules, "mtplx.generation", fake_generation)
    monkeypatch.setitem(sys.modules, "mtplx.sampling", fake_sampling)

    public._quickstart_generate(
        rt=SimpleNamespace(tokenizer=TinyTokenizer(), model_path=tmp_path),
        inspection={},
        profile=SimpleNamespace(to_dict=lambda: {"name": "stable"}),
        args=SimpleNamespace(
            system=None,
            max_tokens=8,
            reasoning="on",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            depth=3,
            seed=0,
        ),
        prompt="hello",
        history=[],
        turn_index=0,
    )

    assert captured["enable_thinking"] is True


def test_terminal_reasoning_command_updates_local_mode(capsys):
    args = SimpleNamespace(reasoning=None)

    assert public._handle_quickstart_reasoning_command(args, "--reasoning on") is True

    assert args.reasoning == "on"
    assert "Reasoning: on" in capsys.readouterr().out


def test_quickstart_openwebui_dry_run_json(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))

    code = main(
        [
            "start",
            "openwebui",
            "--dry-run",
            "--json",
            "--model",
            "models/example",
            "--port",
            "18012",
            "--yes",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["target"] == "openwebui"
    assert payload["terminal_chat"] is False
    assert payload["openwebui"]["server_url"] == "http://127.0.0.1:18012"
    assert payload["openwebui"]["base_url"] == "http://127.0.0.1:18012/v1"
    assert payload["openwebui"]["api_base_url"] == "http://127.0.0.1:18012/v1"
    assert payload["openwebui"]["chat_url"] == "http://127.0.0.1:18012/"
    assert "Open chat UI: http://127.0.0.1:18012/" in payload["openwebui"]["openwebui_steps"]
    assert "OpenAI-compatible API base URL: http://127.0.0.1:18012/v1" in payload["openwebui"]["openwebui_steps"]
    assert "--profile sustained" in payload["openwebui"]["server_command"]
    assert "--no-stats-footer" in payload["openwebui"]["server_command"]
    assert "--open-browser" in payload["openwebui"]["server_command"]


def test_quickstart_pi_dry_run_json(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setenv("MTPLX_PI_MODELS_JSON", str(tmp_path / "pi" / "models.json"))

    code = main(
        [
            "start",
            "pi",
            "--dry-run",
            "--json",
            "--model",
            "models/example",
            "--port",
            "18012",
            "--yes",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["target"] == "pi"
    assert payload["terminal_chat"] is False
    assert payload["pi"]["base_url"] == "http://127.0.0.1:18012/v1"
    assert payload["pi"]["model_ref"] == "mtplx/example"
    assert payload["pi"]["launch_command"] == "pi --model mtplx/example"
    assert payload["pi"]["provider"]["api"] == "openai-completions"
    assert payload["pi"]["provider"]["authHeader"] is True
    assert payload["pi"]["provider"]["compat"]["supportsDeveloperRole"] is False
    assert payload["pi"]["provider"]["compat"]["supportsReasoningEffort"] is False
    assert payload["pi"]["provider"]["compat"]["maxTokensField"] == "max_tokens"
    assert "--api-key mtplx-local" in payload["pi"]["server_command"]


def test_start_pi_missing_cli_stops_before_model_check(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))

    class NonInteractiveStdin:
        def isatty(self):
            return False

    monkeypatch.setattr(public.sys, "stdin", NonInteractiveStdin())
    monkeypatch.setattr(public.shutil, "which", lambda name: None if name == "pi" else "/usr/bin/npm")

    def fail_resolve(*_args, **_kwargs):
        raise AssertionError("Pi preflight must run before model resolution")

    monkeypatch.setattr(public, "_quickstart_resolve_model", fail_resolve)

    code = main(["start", "pi", "--model", "models/example", "--yes"])

    captured = capsys.readouterr().out
    assert code == 2
    assert "Pi is not installed" in captured
    assert "MTPLX has not loaded the model yet" in captured
    assert "npm install -g @earendil-works/pi-coding-agent" in captured
    assert "Then re-run: mtplx start pi" in captured
    assert "[1/4] Checking model" not in captured


def test_pi_models_config_merge_preserves_other_providers(tmp_path):
    from mtplx.pi import write_pi_models_config

    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "providers": {
                    "other": {
                        "baseUrl": "https://example.invalid/v1",
                        "api": "openai-completions",
                        "apiKey": "OTHER_KEY",
                        "models": [{"id": "other-model"}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = write_pi_models_config(
        base_url="http://127.0.0.1:18012/v1",
        model_id="mtplx-test-model",
        path=config_path,
    )

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert result["config_path"] == str(config_path)
    assert payload["providers"]["other"]["models"][0]["id"] == "other-model"
    assert payload["providers"]["mtplx"]["baseUrl"] == "http://127.0.0.1:18012/v1"
    assert payload["providers"]["mtplx"]["models"][0]["id"] == "mtplx-test-model"


def test_start_pi_handoff_writes_config_and_starts_authenticated_server(
    monkeypatch,
    tmp_path,
):
    config_path = tmp_path / "pi" / "models.json"
    monkeypatch.setenv("MTPLX_PI_MODELS_JSON", str(config_path))
    monkeypatch.setattr(public.shutil, "which", lambda _name: "/usr/local/bin/pi")
    captured: dict[str, object] = {}

    def fake_serve(serve_args):
        captured["api_key"] = serve_args.api_key
        captured["quickstart_pi"] = serve_args.quickstart_pi
        captured["open_browser"] = serve_args.open_browser
        captured["stats_footer"] = serve_args.stats_footer
        return 0

    monkeypatch.setattr(public, "cmd_serve_public", fake_serve)
    args = SimpleNamespace(
        host="127.0.0.1",
        port=18012,
        model="/models/qwen",
        model_id="mtplx-test-model",
        profile="sustained",
        max=False,
        api_key=None,
        cache_dir=None,
        unsafe_force_unverified=False,
        depth=3,
        no_mtp=False,
        rate_limit=0,
        stream_interval=1,
        warmup_tokens=16,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning=None,
        reasoning_parser="qwen3",
        strict_warmup=False,
        strict_fast_path=False,
        max_idle_min=15,
    )

    rc = public._quickstart_run_pi(args, runtime_model="/models/qwen", inspection={})

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert captured == {
        "api_key": "mtplx-local",
        "quickstart_pi": True,
        "open_browser": False,
        "stats_footer": False,
    }
    assert payload["providers"]["mtplx"]["baseUrl"] == "http://127.0.0.1:18012/v1"
    assert payload["providers"]["mtplx"]["models"][0]["id"] == "mtplx-test-model"


def test_run_json_model_summary_excludes_heavy_inspect_fields():
    summary = public._compact_model_summary(
        {
            "source": "local",
            "model_dir": "/models/champion",
            "architecture": "Qwen3_5ForConditionalGeneration",
            "model_type": "qwen3_5_text",
            "mtp_arch": "qwen3-next-mtp",
            "mtp_supported": "yes",
            "recommended_backend": "qwen3_next",
            "runtime_compatibility": "native",
            "runtime_contract_path": None,
            "mtp": {"tensors": [{"key": "large"}]},
            "quantization": {"language_model.model.layers.0": {"bits": 4}},
            "compatibility": {
                "tier": "verified",
                "can_run": True,
                "supported": True,
                "exit_code": 0,
                "message": "Verified MTPLX runtime contract found.",
                "arch_id": "qwen3-next-mtp",
                "recommended_profile": "stable",
                "runtime_contract_path": "/models/champion/mtplx_runtime.json",
            },
        }
    )

    assert summary["model_dir"] == "/models/champion"
    assert summary["compatibility"]["tier"] == "verified"
    assert summary["runtime_contract_path"] == "/models/champion/mtplx_runtime.json"
    assert "mtp" not in summary
    assert "quantization" not in summary


def test_inspect_human_uses_compatibility_runtime_contract(capsys):
    public._print_inspect_human(
        {
            "model_dir": "/models/champion",
            "source": "local",
            "architecture": "Qwen3_5ForConditionalGeneration",
            "mtp_num_hidden_layers": 1,
            "mtp": {"tensor_count": 29},
            "runtime_contract_path": None,
            "compatibility": {
                "tier": "verified",
                "can_run": True,
                "recognized": True,
                "runtime_contract_path": "/models/champion/mtplx_runtime.json",
                "runtime_compatibility": "native",
                "support_level": "verified-native",
                "recommended_profile": "stable",
                "message": "Verified MTPLX runtime contract found.",
            },
        }
    )

    captured = capsys.readouterr().out
    assert "runtime_contract: true" in captured


def test_public_bench_run_dry_run(capsys):
    code = main(
        [
            "bench",
            "run",
            "--model",
            "models/not-loaded-in-dry-run",
            "--suite",
            "flappy",
            "--max-tokens",
            "10000",
            "--no-fanmax",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["action"] == "bench run"
    assert payload["exactness_smoke"]["automatic"] is True
    assert payload["profile"]["name"] == "sustained"
    assert payload["harness"] == "direct-http"
    assert payload["runtime_profile"] == "native_mtp_sustained"
    assert payload["runtime_env"]["MTPLX_SUSTAINED_PREFILL"] == "1"
    assert payload["direct_http_command"] is not None
    assert "--profiles" in payload["direct_http_command"]
    assert (
        payload["direct_http_command"][payload["direct_http_command"].index("--profiles") + 1]
        == "sustained"
    )


def test_public_bench_long_context_default_is_sustained(capsys):
    code = main(
        [
            "bench",
            "run",
            "--model",
            "models/not-loaded-in-dry-run",
            "--suite",
            "long_code_uncapped",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["profile"]["name"] == "sustained"
    assert payload["harness"] == "direct-http"
    assert payload["runtime_profile"] == "native_mtp_sustained"
    assert (
        payload["direct_http_command"][payload["direct_http_command"].index("--profiles") + 1]
        == "sustained"
    )


def test_public_bench_long_code_dash_alias_uses_sustained_direct_test(capsys):
    code = main(
        [
            "bench",
            "run",
            "--model",
            "models/not-loaded-in-dry-run",
            "--suite",
            "long-code",
            "--max-tokens",
            "1024",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    command = payload["direct_http_command"]
    assert code == 0
    assert payload["profile"]["name"] == "sustained"
    assert command[command.index("--profiles") + 1] == "sustained"
    assert command[command.index("--tests") + 1] == "long_code"


def test_public_bench_run_dry_run_records_external_kernel_env(monkeypatch, capsys):
    monkeypatch.setenv("MTPLX_VERIFY_OUTPUT_DEPENDS", "recurrent")
    monkeypatch.setenv("MTPLX_VERIFY_OUTPUT_DEPENDS_AFTER_TOKENS", "1024")
    monkeypatch.setenv("MTPLX_SDPA_2PASS_BLOCKS", "64")
    monkeypatch.setenv("MTPLX_SDPA_DYNAMIC_OFFSET_ACTIVE_BLOCKS", "1")
    monkeypatch.setenv("MTPLX_EXPORT_VERIFY_DOT_DIR", "outputs/dot-probe")
    monkeypatch.setenv("MTPLX_EXPORT_VERIFY_DOT_CYCLES", "1,128")
    monkeypatch.setenv("MTPLX_EVAL_STATE_ROOTS_INCLUDE_LIVE", "0")

    code = main(
        [
            "bench",
            "run",
            "--model",
            "models/not-loaded-in-dry-run",
            "--suite",
            "flappy",
            "--max-tokens",
            "2048",
            "--no-fanmax",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["runtime_env"]["MTPLX_VERIFY_OUTPUT_DEPENDS"] == "recurrent"
    assert payload["runtime_env"]["MTPLX_VERIFY_OUTPUT_DEPENDS_AFTER_TOKENS"] == "1024"
    assert payload["runtime_env"]["MTPLX_SDPA_2PASS_BLOCKS"] == "64"
    assert payload["runtime_env"]["MTPLX_SDPA_DYNAMIC_OFFSET_ACTIVE_BLOCKS"] == "1"
    assert payload["runtime_env"]["MTPLX_EXPORT_VERIFY_DOT_DIR"] == "outputs/dot-probe"
    assert payload["runtime_env"]["MTPLX_EXPORT_VERIFY_DOT_CYCLES"] == "1,128"
    assert payload["runtime_env"]["MTPLX_EVAL_STATE_ROOTS_INCLUDE_LIVE"] == "0"


def test_public_bench_cold_run_defaults_to_sustained_mode(capsys):
    code = main(
        [
            "bench",
            "run",
            "--model",
            "models/not-loaded-in-dry-run",
            "--suite",
            "cold-long-code-192",
            "--max-tokens",
            "192",
            "--strict-cold",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert code == 0
    assert payload["profile"]["name"] == "sustained"
    assert payload["harness"] == "direct-http"
    assert payload["seed"] == 42
    assert payload["runtime_profile"] == "native_mtp_sustained"
    assert payload["runtime_env"]["MTPLX_SUSTAINED_PREFILL"] == "1"
    assert payload["runtime_env"]["MTPLX_PREFILL_OMLX_EXTERNAL"] == "1"
    assert payload["runtime_env"]["MTPLX_VLLM_METAL_PAGED_TURBOQUANT"] == "0"
    assert payload["direct_http_command"] is not None


def test_public_bench_performance_cold_is_explicit(capsys):
    code = main(
        [
            "bench",
            "run",
            "--model",
            "models/not-loaded-in-dry-run",
            "--suite",
            "cold-long-code-192",
            "--max-tokens",
            "192",
            "--profile",
            "performance-cold",
            "--strict-cold",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert code == 0
    assert payload["profile"]["name"] == "performance-cold"
    assert payload["harness"] == "depth-sweep"
    assert payload["seed"] == 0
    assert payload["runtime_profile"] == "native_mtp_60_cold"
    assert payload["runtime_env"]["MTPLX_LAZY_VERIFY_LOGITS"] == "1"
    assert "MTPLX_TARGET_LAYER_EVAL_SCHEDULE" not in payload["runtime_env"]


def test_public_bench_explicit_performance_cold_overrides_long_context_default(capsys):
    code = main(
        [
            "bench",
            "run",
            "--model",
            "models/not-loaded-in-dry-run",
            "--suite",
            "long_code_uncapped",
            "--profile",
            "performance-cold",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["profile"]["name"] == "performance-cold"
    assert payload["harness"] == "depth-sweep"
    assert payload["runtime_profile"] == "native_mtp_60_cold"
    assert payload["direct_http_command"] is None


def test_public_qa_distribution_parser_dry_shape():
    parser = build_parser()
    args = parser.parse_args(
        [
            "qa",
            "distribution",
            "--model",
            "models/example",
            "--suite",
            "distribution-smoke",
        ]
    )

    assert args.command == "qa"
    assert args.qa_action == "distribution"


def test_public_profile_dispatch_without_trace_is_actionable(capsys):
    code = main(
        [
            "profile",
            "dispatch",
            "--model",
            "models/example",
            "--suite",
            "flappy",
            "--max-tokens",
            "2048",
        ]
    )

    captured = capsys.readouterr().out
    assert code == 0
    assert '"implemented_capture": false' in captured
    assert "--trace PATH" in captured


def test_reference_vllm_dry_run_includes_ssh_capture_command(capsys):
    code = main(
        [
            "bench",
            "reference-vllm",
            "--suite",
            "flappy",
            "--max-tokens",
            "6000",
            "--capture-dispatch",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr().out
    assert code == 0
    assert '"action": "bench reference-vllm"' in captured
    assert '"remote_capture_kind": "offline"' in captured
    assert "--capture-range=cudaProfilerApi" in captured
    assert "--cuda-graph-trace=graph" in captured
    assert "cuda_api_sum" in captured
    assert "mtplx-3090" in captured
    assert '"remote_prompt_override"' in captured
    assert '"max_tokens": 6000' in captured


def test_champion_bakeoff_compare_dry_run_lists_required_tasks(capsys):
    code = main(
        [
            "bench",
            "compare",
            "--models",
            "models/a",
            "models/b",
            "--suite",
            "champion-bakeoff",
            "--no-fanmax",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr().out
    assert code == 0
    assert '"action": "bench compare"' in captured
    assert '"label": "flappy-10k"' in captured
    assert '"max_tokens": 10000' in captured
    assert '"label": "python-modules-long"' in captured
    assert '"max_tokens": 6000' in captured
    assert '"label": "cold-long-code-192"' in captured
    assert '"strict_cold": true' in captured


def test_public_bench_parser_has_seed_for_live_child_runs():
    parser = build_parser()
    args = parser.parse_args(
        [
            "bench",
            "compare",
            "--models",
            "models/a",
            "models/b",
            "--suite",
            "champion-bakeoff",
        ]
    )

    assert args.seed is None


def test_bench_nightly_dry_run_lists_kernel_gate_tasks(capsys):
    code = main(
        [
            "bench",
            "nightly",
            "--model",
            "models/not-loaded-in-dry-run",
            "--run-id",
            "nightly-test",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["action"] == "bench nightly"
    assert [task["label"] for task in payload["tasks"]] == [
        "cold-long-code-192",
        "flappy-6k",
        "flappy-10k",
        "python-modules-6k",
    ]
    assert payload["policy"]["fanmax_counts_for_product_gate"] is False
    assert payload["full_exactness_command"][0:2] == ["qa", "exactness"]
    assert [task["profile"] for task in payload["tasks"]] == [
        "performance-cold",
        "sustained",
        "sustained",
        "sustained",
    ]
    flappy_6k_command = payload["tasks"][1]["direct_http_command"]
    assert "--python-bin" in flappy_6k_command
    assert "--max-tokens" in flappy_6k_command
    assert flappy_6k_command[flappy_6k_command.index("--max-tokens") + 1] == "6000"


def test_bench_compare_envelopes_detects_cold_regression(tmp_path, capsys):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(
        json.dumps(
            {
                "suite": "cold-long-code-192",
                "runtime": {"tok_s": 60.0},
                "quality": {"passed": True},
                "correctness": {"exactness_smoke": {"passed": True}},
            }
        ),
        encoding="utf-8",
    )
    after.write_text(
        json.dumps(
            {
                "suite": "cold-long-code-192",
                "runtime": {"tok_s": 58.0},
                "quality": {"passed": True},
                "correctness": {"exactness_smoke": {"passed": True}},
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "bench",
            "compare",
            "--before",
            str(before),
            "--after",
            str(after),
            "--strict-cold",
            "--strict-exactness",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 5
    assert payload["action"] == "bench compare envelopes"
    assert payload["passed"] is False
    assert payload["comparisons"][0]["gates"]["cold_floor_ge_59"] is False


def test_chat_and_serve_default_to_sustained_mode():
    parser = build_parser()

    run_args = parser.parse_args(["run", "hello", "--cache-dir", "/tmp/mtplx-models"])
    chat_args = parser.parse_args(["chat", "--prompt", "hello"])
    serve_args = parser.parse_args(["serve"])
    serve_max_args = parser.parse_args(["serve", "--max"])
    serve_no_footer_args = parser.parse_args(["serve", "--no-stats-footer"])

    assert run_args.profile == "sustained"
    assert run_args.prompt_arg == "hello"
    assert run_args.cache_dir == "/tmp/mtplx-models"
    assert run_args.max_tokens is None
    assert run_args.reasoning is None
    assert chat_args.profile == "sustained"
    assert chat_args.max_tokens is None
    assert chat_args.reasoning is None
    assert serve_args.profile == "sustained"
    assert serve_args.reasoning is None
    assert serve_args.stock_ar is False
    assert serve_max_args.max is True
    assert serve_args.stream_interval == 1
    assert serve_args.rate_limit == 0
    assert serve_args.reasoning_parser == "qwen3"
    assert serve_args.stats_footer is True
    assert serve_no_footer_args.stats_footer is False


def test_stock_ar_is_diagnostic_serve_and_bench_only():
    parser = build_parser()

    serve = parser.parse_args(["serve", "--stock-ar"])
    bench = parser.parse_args(["bench", "context", "--stock-ar", "--dry-run"])

    assert serve.stock_ar is True
    assert bench.stock_ar is True
    assert bench.bench_action == "context"
    for argv in (
        ["quickstart", "--stock-ar"],
        ["chat", "--stock-ar"],
        ["run", "hello", "--stock-ar"],
        ["start", "cli", "--stock-ar", "--dry-run"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_product_helper_commands_parse():
    parser = build_parser()

    start = parser.parse_args(
        ["start", "--prompt", "hello", "--max-tokens", "16", "--no-stats"]
    )
    start_openwebui = parser.parse_args(["start", "openwebui", "--port", "18012"])
    start_opencode = parser.parse_args(["start", "opencode", "--port", "18083"])
    start_openwebui_strict = parser.parse_args(["start", "openwebui", "--strict-fast-path"])
    quickstart = parser.parse_args(["quickstart", "--port", "18012"])
    quickstart_alias = parser.parse_args(["quick-start", "--port", "18013"])
    setup = parser.parse_args(["setup", "--dry-run"])
    pull_default = parser.parse_args(["pull"])
    ask = parser.parse_args(["ask", "hello"])
    ask_stats = parser.parse_args(["ask", "hello", "--stats"])
    serve_start = parser.parse_args(["serve", "--port", "18012"])
    status = parser.parse_args(["status", "--deep"])
    doctor_opencode = parser.parse_args(["doctor", "opencode", "--json"])
    connect = parser.parse_args(["connect", "openwebui", "--port", "18012"])
    connect_opencode = parser.parse_args(["connect", "opencode", "--port", "18012"])
    models = parser.parse_args(["models", "--json"])
    report = parser.parse_args(["report", "--output-dir", "reports"])
    nightly = parser.parse_args(["bench", "nightly", "--out", "out.json"])
    nightly_json = parser.parse_args(["bench", "nightly", "--json", "--dry-run"])
    debug = parser.parse_args(["debug", "bundle", "--run-id", "debug-test"])
    hotpath = parser.parse_args(["debug", "hotpath"])
    metrics = parser.parse_args(["metrics", "watch", "--count", "1", "--json"])
    openwebui = parser.parse_args(["integrate", "openwebui", "--port", "18012", "--json"])
    openwebui_docker = parser.parse_args(["openwebui", "docker-command", "--mtplx-port", "18012"])
    claude = parser.parse_args(["integrate", "claude-code", "--port", "18012"])
    opencode = parser.parse_args(["integrate", "opencode", "--port", "18012"])
    architectures = parser.parse_args(["model", "architectures", "--json"])
    qa_architectures = parser.parse_args(["model", "qa-architectures", "--json"])
    publish = parser.parse_args(["model", "publish-check", "--repo-id", "mtplx/example"])
    config = parser.parse_args(["config", "set", "profile", "exact", "--dry-run"])
    attribution = parser.parse_args(["profile", "eval-attribution", "--dry-run"])

    assert start.command == "start"
    assert start.prompt == "hello"
    assert start.max_tokens == 16
    assert start.show_stats is False
    assert start_openwebui.target == "openwebui"
    assert start_openwebui.port == 18012
    assert start_opencode.target == "opencode"
    assert start_opencode.port == 18083
    assert start_openwebui.strict_fast_path is False
    assert start_openwebui_strict.strict_fast_path is True
    assert quickstart.command == "quickstart"
    assert quickstart.port == 18012
    assert quickstart.profile == "sustained"
    assert quickstart_alias.command == "quick-start"
    assert quickstart_alias.port == 18013
    assert quickstart_alias.profile == "sustained"
    assert setup.command == "setup"
    assert setup.dry_run is True
    assert pull_default.command == "pull"
    assert pull_default.model == "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
    assert ask.command == "ask"
    assert ask.prompt_arg == "hello"
    assert ask.quiet is True
    assert ask_stats.quiet is False
    assert serve_start.command == "serve"
    assert serve_start.port == 18012
    assert serve_start.stats_footer is True
    assert status.command == "status"
    assert status.deep is True
    assert doctor_opencode.command == "doctor"
    assert doctor_opencode.topic == "opencode"
    assert connect.command == "connect"
    assert connect.integration == "openwebui"
    assert connect_opencode.integration == "opencode"
    assert models.command == "models"
    assert report.command == "report"
    assert report.bundle is True
    assert report.deep is True
    assert nightly.bench_action == "nightly"
    assert nightly.output == "out.json"
    assert nightly_json.json is True
    assert debug.debug_action == "bundle"
    assert hotpath.debug_action == "hotpath"
    assert metrics.metrics_action == "watch"
    assert openwebui.integration == "openwebui"
    assert openwebui_docker.openwebui_action == "docker-command"
    assert openwebui_docker.mtplx_port == 18012
    assert claude.integration == "claude-code"
    assert opencode.integration == "opencode"
    assert architectures.model_action == "architectures"
    assert qa_architectures.model_action == "qa-architectures"
    assert publish.model_action == "publish-check"
    assert config.config_action == "set"
    assert attribution.profile_action == "eval-attribution"


def test_model_architectures_json_lists_verified_and_pending(capsys):
    code = main(["model", "architectures", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    ids = {row["arch_id"] for row in payload["architectures"]}
    assert "qwen3-next-mtp" in payload["verified_runtime_arch_ids"]
    assert "deepseek-v3-mtp" in payload["verified_runtime_arch_ids"]
    assert "glm-moe-dsa-mtp" in payload["verified_runtime_arch_ids"]
    assert "glm4-moe-mtp" in payload["verified_runtime_arch_ids"]
    assert "glm4-moe-lite-mtp" in payload["verified_runtime_arch_ids"]
    assert "mimo-mtp" in payload["verified_runtime_arch_ids"]
    assert "glm4-moe-mtp" in ids
    assert "gemma-mtp" in ids


def test_model_qa_architectures_runs_contract_fixture_gates(capsys):
    code = main(["model", "qa-architectures", "--json", "--runtime-import-smoke"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["action"] == "model qa-architectures"
    assert payload["passed"] is True
    assert payload["gates"]["catalog_has_main_families"] is True
    assert payload["gates"]["fixture_inspections_passed"] is True
    labels = {row["label"]: row for row in payload["fixtures"]}
    assert labels["deepseek-v3-contract-gated"]["observed"]["tier"] == "verified"
    assert labels["glm4-moe-lite-contract-gated"]["observed"]["recommended_backend"] == "glm_mtp"
    assert labels["minimax-m2-num-mtp-modules-recognized-pending"]["observed"]["runtime_compatibility"] == "recognized-backend-pending"
    assert labels["gemma4-without-mtp-stays-no-mtp"]["observed"]["tier"] == "no-MTP"
    assert all(row["passed"] for row in payload["runtime_import_smokes"])


def test_integrate_openwebui_json(capsys):
    code = main(["integrate", "openwebui", "--port", "18012", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["integration"] == "openwebui"
    assert payload["server_url"] == "http://127.0.0.1:18012"
    assert payload["base_url"] == "http://127.0.0.1:18012/v1"
    assert payload["docker_api_base_url"] == "http://host.docker.internal:18012/v1"
    assert "host.docker.internal:18012/v1" in payload["docker_command"]
    assert "--no-stats-footer" in payload["server_command"]


def test_integrate_opencode_json_uses_raw_reasoning_contract(capsys):
    code = main(["integrate", "opencode", "--port", "18012", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["integration"] == "opencode"
    assert payload["api_base_url"] == "http://127.0.0.1:18012/v1"
    assert "--reasoning on" in payload["server_command"]
    model = payload["config"]["provider"]["mtplx"]["models"][payload["model_id"]]
    assert model["reasoning"] is True
    assert model["interleaved"] == {"field": "reasoning_content"}
    assert model["options"]["enable_thinking"] is True
    assert "reasoningSummary" not in model["options"]


def test_config_set_dry_run_uses_selected_path(tmp_path, capsys):
    config = tmp_path / "config.toml"

    code = main(["config", "set", "profile", "exact", "--config", str(config), "--dry-run"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["path"] == str(config)
    assert payload["updated"] == {"profile": "exact"}
    assert not config.exists()


def test_debug_hotpath_reports_next_kernel_boundary(capsys):
    code = main(["debug", "hotpath"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["action"] == "debug hotpath"
    names = {row["name"] for row in payload["boundaries"]}
    assert "verify_output_eval" in names
    assert "native_rowwise_mlp" in names
    assert "native_residual_mlp" in names
    assert "fused_logits_topk_distribution" in names
    assert "external_vllm_partitioned_fallback" in names
    assert payload["raw_sync_markers"]["native_mlp_is_mlx_primitive"] is True
    assert "native residual MLP layer-boundary fusion" in payload["verdict"]["do_not_loop"]
    assert "standalone dense-logit top-k distribution kernels" in payload["verdict"]["do_not_loop"]
    assert "larger owned verify-layer or verify-cycle primitive" in payload["verdict"]["highest_upside_next"]


def test_serve_dispatches_packaged_openai_server(monkeypatch, capsys):
    calls = {}

    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(executable, cmd, env):
        calls["executable"] = executable
        calls["cmd"] = cmd
        calls["env"] = env
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        model="models/example",
        cache_dir=None,
        profile="stable",
        unsafe_force_unverified=False,
        yes=False,
        host="127.0.0.1",
        port=8000,
        depth=3,
        api_key="test-key",
        rate_limit=120,
        stream_interval=4,
        max_response_tokens=512,
        temperature=0.4,
        top_p=0.9,
        reasoning="off",
        reasoning_parser="qwen3",
        stats_footer=False,
        warmup_tokens=8,
        strict_warmup=True,
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    captured = capsys.readouterr().out
    # Banner + framed status panel from the new mtplx.ui module.
    assert "MTPLX" in captured
    assert DISPLAY_VERSION in captured
    assert "127.0.0.1:8000/v1" in captured
    assert "127.0.0.1:8000/" in captured
    # Numbered handoff lines that still print before the model load.
    assert "[1/6] Server config ready" in captured
    assert "[2/6] Model resolved: models/example" in captured
    assert "[3/6] Runtime contract verified" in captured
    assert "Loading the model can take about a minute" in captured
    assert calls["cmd"][1:3] == ["-m", "mtplx.server.openai"]
    assert "--model" in calls["cmd"]
    assert calls["cmd"][calls["cmd"].index("--api-key") + 1] == "test-key"
    assert calls["cmd"][calls["cmd"].index("--rate-limit") + 1] == "120"
    assert calls["cmd"][calls["cmd"].index("--stream-interval") + 1] == "4"
    assert calls["cmd"][calls["cmd"].index("--max-response-tokens") + 1] == "512"
    assert calls["cmd"][calls["cmd"].index("--model-id") + 1] == "example"
    assert calls["cmd"][calls["cmd"].index("--generation-mode") + 1] == "mtp"
    assert "--no-enable-thinking" in calls["cmd"]
    assert "--no-stats-footer" in calls["cmd"]
    assert "--strict-warmup" in calls["cmd"]


def test_serve_wildcard_host_displays_bind_and_forwards_host(monkeypatch, capsys):
    calls = {}

    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(executable, cmd, env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        model="models/example",
        cache_dir=None,
        profile="sustained",
        unsafe_force_unverified=False,
        yes=True,
        host="0.0.0.0",
        port=8000,
        depth=3,
        api_key="test-key",
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning=None,
        reasoning_parser="qwen3",
        stats_footer=False,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
        open_browser=False,
        stock_ar=False,
        _cli_flags={"model", "host", "api_key"},
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    captured = capsys.readouterr().out
    assert "Listening  0.0.0.0:8000 (all interfaces)" in captured
    assert (
        "[1/6] Server config ready: listening on 0.0.0.0:8000 (all interfaces)"
        in captured
    )
    assert "Local API Base URL: http://127.0.0.1:8000/v1" in captured
    assert calls["cmd"][calls["cmd"].index("--host") + 1] == "0.0.0.0"


def test_serve_uses_quality_public_model_id_for_quality_local_path(monkeypatch):
    calls = {}
    quality_path = "/tmp/Qwen3.6-27B-MTPLX-Optimized-Quality"

    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(_executable, cmd, _env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        model=quality_path,
        model_id="mtplx-qwen36-27b-optimized-speed",
        cache_dir=None,
        profile="sustained",
        unsafe_force_unverified=False,
        yes=True,
        host="127.0.0.1",
        port=8000,
        depth=3,
        no_mtp=False,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning="off",
        reasoning_parser="qwen3",
        stats_footer=False,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
    )

    with pytest.raises(SystemExit) as exc:
        public.cmd_serve_public(args)

    assert exc.value.code == 0
    assert calls["cmd"][calls["cmd"].index("--model-id") + 1] == QUALITY_PUBLIC_MODEL_ID


def test_serve_uses_legacy_public_model_id_for_legacy_optimized_local_path(monkeypatch):
    calls = {}
    legacy_path = "/tmp/Qwen3.6-27B-MTPLX-Optimized"

    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(_executable, cmd, _env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        model=legacy_path,
        model_id=DEFAULT_PUBLIC_MODEL_ID,
        cache_dir=None,
        profile="sustained",
        unsafe_force_unverified=False,
        yes=True,
        host="127.0.0.1",
        port=8000,
        depth=3,
        no_mtp=False,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning="off",
        preserve_thinking="auto",
        reasoning_parser="qwen3",
        stats_footer=False,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
    )

    with pytest.raises(SystemExit) as exc:
        public.cmd_serve_public(args)

    assert exc.value.code == 0
    assert (
        calls["cmd"][calls["cmd"].index("--model-id") + 1]
        == LEGACY_OPTIMIZED_PUBLIC_MODEL_ID
    )


def test_serve_uses_model_contract_depth_when_depth_not_explicit(monkeypatch):
    calls = {}

    monkeypatch.setattr(public, "_resolve_runtime_model_path", lambda model, cache_dir=None: (model, None))
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {
                "compatibility": {
                    "tier": "verified",
                    "can_run": True,
                    "exit_code": 0,
                    "runtime_contract": {"mtp_depth_max": 2},
                }
            },
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(executable, cmd, env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        model="models/example",
        cache_dir=None,
        profile="performance-cold",
        unsafe_force_unverified=False,
        yes=True,
        host="127.0.0.1",
        port=8000,
        depth=3,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=True,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
        _cli_flags=set(),
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    assert calls["cmd"][calls["cmd"].index("--depth") + 1] == "2"


def test_serve_no_mtp_dispatches_ar_generation_mode(monkeypatch):
    calls = {}

    monkeypatch.setattr(public, "_resolve_runtime_model_path", lambda model, cache_dir=None: (model, None))
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(_executable, cmd, _env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        command="serve",
        model="models/example",
        cache_dir=None,
        profile="performance-cold",
        unsafe_force_unverified=False,
        yes=True,
        host="127.0.0.1",
        port=8000,
        depth=3,
        no_mtp=True,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=True,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
        _cli_flags=set(),
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    assert calls["cmd"][calls["cmd"].index("--generation-mode") + 1] == "ar"


def test_serve_stock_ar_dispatches_unloaded_ar(monkeypatch):
    calls = {}

    monkeypatch.setattr(public, "_resolve_runtime_model_path", lambda model, cache_dir=None: (model, None))
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(_executable, cmd, _env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        command="serve",
        model="models/example",
        cache_dir=None,
        profile="performance-cold",
        unsafe_force_unverified=False,
        yes=True,
        host="127.0.0.1",
        port=8000,
        depth=3,
        no_mtp=False,
        stock_ar=True,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=True,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
        _cli_flags=set(),
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    assert calls["cmd"][calls["cmd"].index("--generation-mode") + 1] == "ar"
    assert "--stock-ar" in calls["cmd"]


def test_quickstart_pi_passes_launch_command_to_server(monkeypatch):
    calls = {}

    monkeypatch.setattr(public, "_resolve_runtime_model_path", lambda model, cache_dir=None: (model, None))
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(_executable, cmd, _env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        command="serve",
        model="models/example",
        model_id="mtplx-qwen36-27b-optimized-speed",
        cache_dir=None,
        profile="sustained",
        unsafe_force_unverified=False,
        yes=True,
        host="127.0.0.1",
        port=8000,
        depth=3,
        no_mtp=False,
        stock_ar=False,
        api_key="mtplx-local",
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=False,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        quickstart_pi=True,
        max=False,
        _cli_flags=set(),
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    assert "--launch-pi" in calls["cmd"]
    assert "--server-console" in calls["cmd"]
    command = calls["cmd"][calls["cmd"].index("--pi-launch-command") + 1]
    assert command == "pi --model mtplx/example"


def test_launch_pi_in_terminal_does_not_false_positive_on_server_command(monkeypatch):
    from mtplx import pi

    calls = {}
    monkeypatch.setattr(pi.sys, "platform", "darwin")

    def fake_popen(cmd, stdout=None, stderr=None):
        calls["cmd"] = cmd
        calls["stdout"] = stdout
        calls["stderr"] = stderr

        class Proc:
            pass

        return Proc()

    monkeypatch.setattr(pi.subprocess, "Popen", fake_popen)

    result = pi.launch_pi_in_terminal(
        "pi --model mtplx/mtplx-qwen36-27b-optimized-speed",
        model_ref="mtplx/mtplx-qwen36-27b-optimized-speed",
    )

    assert result["status"] == "launched"
    assert calls["cmd"][0] == "osascript"
    script = calls["cmd"][2]
    assert "do script" in script
    assert "pi --model mtplx/mtplx-qwen36-27b-optimized-speed" in script


def test_bare_serve_invokes_server_onboarding_in_tty(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)
    monkeypatch.setattr(public, "_resolve_runtime_model_path", lambda model, cache_dir=None: (model, None))
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    invocations: list[dict] = []

    def fake_flow(**kwargs):
        invocations.append(kwargs)
        return {
            "model": "models/onboarded",
            "profile": "performance-cold",
            "max": False,
            "target": "server",
            "open_browser": False,
        }

    monkeypatch.setattr("mtplx.ui.onboarding.run_serve_flow", fake_flow)
    calls = {}

    def fake_execvpe(executable, cmd, env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        command="serve",
        model="models/configured",
        cache_dir=None,
        download=False,
        profile="stable",
        unsafe_force_unverified=False,
        yes=False,
        host="127.0.0.1",
        port=8765,
        depth=3,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=True,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
        open_browser=False,
        _cli_flags=set(),
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    assert len(invocations) == 1
    assert invocations[0]["configured_model"] == "models/configured"
    assert invocations[0]["port"] == 8765
    assert args._onboarded is True
    assert args.model == "models/onboarded"
    assert calls["cmd"][calls["cmd"].index("--model") + 1] == "models/onboarded"
    assert "--open-browser" not in calls["cmd"]


def test_bare_serve_hf_choice_enables_download_and_browser(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )

    def fake_flow(**_kwargs):
        return {
            "model": "owner/repo",
            "profile": "performance-cold",
            "max": False,
            "target": "openwebui",
            "open_browser": True,
        }

    resolution_calls: list[dict] = []

    def fake_quickstart_resolve(model, *, cache_dir, download):
        resolution_calls.append({"model": model, "download": download})
        return "/tmp/runtime-model", {
            "model": model,
            "runtime_model": "/tmp/runtime-model",
            "downloaded": True,
            "download_ref": model,
        }

    monkeypatch.setattr("mtplx.ui.onboarding.run_serve_flow", fake_flow)
    monkeypatch.setattr(public, "_quickstart_resolve_model", fake_quickstart_resolve)
    calls = {}

    def fake_execvpe(executable, cmd, env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        command="serve",
        model="models/configured",
        cache_dir=None,
        download=False,
        profile="stable",
        unsafe_force_unverified=False,
        yes=False,
        host="127.0.0.1",
        port=8765,
        depth=3,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=True,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
        open_browser=False,
        _cli_flags=set(),
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    assert resolution_calls == [{"model": "owner/repo", "download": True}]
    assert calls["cmd"][calls["cmd"].index("--model") + 1] == "/tmp/runtime-model"
    assert "--open-browser" in calls["cmd"]


def test_serve_skips_onboarding_with_explicit_model(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)
    monkeypatch.setattr(public, "_resolve_runtime_model_path", lambda model, cache_dir=None: (model, None))
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )

    def fail_flow(**_kwargs):
        raise AssertionError("explicit --model should skip server onboarding")

    monkeypatch.setattr("mtplx.ui.onboarding.run_serve_flow", fail_flow)
    calls = {}

    def fake_execvpe(executable, cmd, env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        command="serve",
        model="models/explicit",
        cache_dir=None,
        download=False,
        profile="performance-cold",
        unsafe_force_unverified=False,
        yes=False,
        host="127.0.0.1",
        port=8000,
        depth=3,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=True,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
        open_browser=False,
        _cli_flags={"model"},
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    assert calls["cmd"][calls["cmd"].index("--model") + 1] == "models/explicit"


def test_serve_relaxes_missing_fast_mlx_fork_for_product_start(monkeypatch, capsys):
    calls = {}

    monkeypatch.setattr(public, "_resolve_runtime_model_path", lambda model, cache_dir=None: (model, None))
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)
    monkeypatch.setattr(
        public,
        "_active_mlx_fork_status",
        lambda **_kwargs: {
            "ok": False,
            "path": "/venv/site-packages/mlx/core.cpython-313-darwin.so",
            "version": "0.31.2",
        },
    )

    def fake_execvpe(executable, cmd, env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        model="models/example",
        cache_dir=None,
        profile="performance-cold",
        unsafe_force_unverified=False,
        yes=True,
        host="127.0.0.1",
        port=8000,
        depth=3,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=False,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    captured = capsys.readouterr().out
    assert "Fast MLX fork not active" not in captured
    assert "stock-MLX compatibility" not in captured
    assert "--no-strict-mlx-fork-assert" in calls["cmd"]


def test_serve_strict_fast_path_fails_cleanly_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(public, "_resolve_runtime_model_path", lambda model, cache_dir=None: (model, None))
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)
    monkeypatch.setattr(
        public,
        "_active_mlx_fork_status",
        lambda **_kwargs: {"ok": False, "error": "mlx.core is not installed"},
    )

    args = SimpleNamespace(
        model="models/example",
        cache_dir=None,
        profile="performance-cold",
        unsafe_force_unverified=False,
        yes=True,
        host="127.0.0.1",
        port=8000,
        depth=3,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=False,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=True,
        max=False,
    )

    assert public.cmd_serve_public(args) == 2
    captured = capsys.readouterr().out
    assert "Fast MLX fork is required but not active" in captured
    assert "Traceback" not in captured


def test_serve_reports_busy_port_before_model_resolution(monkeypatch, capsys):
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: True)
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (_ for _ in ()).throw(AssertionError("should not resolve")),
    )
    args = SimpleNamespace(
        model="models/example",
        cache_dir=None,
        profile="stable",
        unsafe_force_unverified=False,
        yes=False,
        host="127.0.0.1",
        port=8000,
        depth=3,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=False,
        warmup_tokens=16,
        strict_warmup=False,
    )

    assert public.cmd_serve_public(args) == 2
    captured = capsys.readouterr().out
    # Banner still prints before the busy-port error path.
    assert "MTPLX" in captured
    assert DISPLAY_VERSION in captured
    assert "error: port 8000 is already in use" in captured
    assert "stop the old mtplx quickstart terminal with Ctrl-C" in captured
    assert "try: mtplx quickstart --profile stable --port 8001" in captured


def test_quickstart_openwebui_reuses_existing_server(monkeypatch, capsys):
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: True)
    monkeypatch.setattr(
        public,
        "_http_json",
        lambda url, timeout=15.0: {"ok": True, "model": "mtplx-qwen36-27b-optimized-speed"},
    )
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (_ for _ in ()).throw(AssertionError("should not resolve")),
    )
    opened = {}
    monkeypatch.setattr(public, "_open_browser_url", lambda url: opened.setdefault("url", url))
    args = SimpleNamespace(
        model="models/example",
        model_id="mtplx-qwen36-27b-optimized-speed",
        cache_dir=None,
        profile="performance-cold",
        unsafe_force_unverified=False,
        yes=True,
        host="127.0.0.1",
        port=8000,
        depth=3,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=False,
        warmup_tokens=16,
        strict_warmup=False,
        strict_fast_path=False,
        quickstart_openwebui=True,
        max=False,
    )

    assert public.cmd_serve_public(args) == 0
    captured = capsys.readouterr().out
    assert "MTPLX is already running." in captured
    assert "Chat URL: http://127.0.0.1:8000/" in captured
    assert "OpenAI API Base URL: http://127.0.0.1:8000/v1" in captured
    assert "API key: leave blank" in captured
    assert "Opening chat UI in your browser" in captured
    assert opened["url"] == "http://127.0.0.1:8000/"


def test_serve_rejects_non_localhost_without_api_key(monkeypatch, capsys):
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (_ for _ in ()).throw(AssertionError("should not resolve")),
    )
    args = SimpleNamespace(
        model="models/example",
        cache_dir=None,
        profile="stable",
        unsafe_force_unverified=False,
        yes=False,
        host="0.0.0.0",
        port=8000,
        depth=3,
        api_key=None,
    )

    assert public.cmd_serve_public(args) == 2
    captured = capsys.readouterr().out
    assert "--api-key is required" in captured


def test_max_status_command_is_no_mlx_stable(monkeypatch, capsys):
    from mtplx import thermal

    thermal.detect_thermal_control.cache_clear()
    # Must mock both lookups: detect_thermal_control checks
    # ``~/.mtplx/bin/thermalforge`` first via ``_find_thermalforge`` so a
    # real install on the dev machine would otherwise leak in.
    monkeypatch.setattr(thermal, "_find_thermalforge", lambda: None)
    monkeypatch.setattr(thermal.shutil, "which", lambda _name: None)

    code = main(["max", "--status", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["detection"]["available"] is False
    thermal.detect_thermal_control.cache_clear()


def test_inspect_accepts_direct_model_argument():
    parser = build_parser()

    args = parser.parse_args(["inspect", "models/example", "--json"])

    assert args.command == "inspect"
    assert args.model_args == ["models/example"]
    assert args.strict_exit_code is True


def test_inspect_accepts_legacy_model_subword_form():
    parser = build_parser()

    args = parser.parse_args(["inspect", "model", "models/example", "--json"])

    assert args.command == "inspect"
    assert args.model_args == ["model", "models/example"]


def test_inspect_human_output_is_default(tmp_path, capsys):
    model = tmp_path / "plain-model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "llama"}\n', encoding="utf-8")

    code = main(["inspect", str(model)])

    captured = capsys.readouterr().out
    assert code == 2
    assert "MTPLX inspect" in captured
    assert f"model: {model}" in captured
    assert "tier: no-MTP" in captured
    assert "can_run: false" in captured
    assert "message: Model has no MTP head." in captured


def test_start_gate_failure_is_human_readable_for_config_only_qwen(tmp_path, capsys):
    model = tmp_path / "qwen-config-only"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "mtp_num_hidden_layers": 1,
                "num_hidden_layers": 64,
                "hidden_size": 5120,
            }
        ),
        encoding="utf-8",
    )

    code = main(["start", "cli", "--model", str(model), "--yes"])

    captured = capsys.readouterr().out
    assert code == 3
    assert "error: model cannot run with MTPLX" in captured
    assert "runtime: missing-mtp-weights" in captured
    assert "mtplx_runtime.json is optional metadata" in captured
    assert "fix: choose a model with real MTP weights" in captured
    assert "\"model_files\"" not in captured


def test_start_gate_failure_is_human_readable_for_config_only_glm(tmp_path, capsys):
    model = tmp_path / "glm-config-only"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Glm4MoeLiteForCausalLM"],
                "model_type": "glm4_moe_lite",
                "num_hidden_layers": 47,
                "num_nextn_predict_layers": 1,
            }
        ),
        encoding="utf-8",
    )

    code = main(["start", "cli", "--model", str(model), "--yes"])

    captured = capsys.readouterr().out
    assert code == 3
    assert "error: model cannot run with MTPLX" in captured
    assert "runtime: missing-mtp-weights" in captured
    assert "mtplx_runtime.json is optional metadata" in captured
    assert "fix: choose a model with real MTP weights" in captured
    assert "MTP MTP markers" not in captured
    assert "\"model_files\"" not in captured


def test_profiles_command_lists_default_without_mlx(capsys):
    code = main(["profiles", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["default"] == "sustained"
    assert [profile["name"] for profile in payload["profiles"]] == [
        "stable",
        "performance-cold",
        "sustained",
        "exact",
        "max-diagnostic",
    ]


def test_model_cache_commands_parse():
    parser = build_parser()

    pull_args = parser.parse_args(["pull", "mtplx/example", "--revision", "main"])
    list_args = parser.parse_args(["list", "--cache-dir", "/tmp/mtplx-models"])
    remove_args = parser.parse_args(["remove", "mtplx/example", "--missing-ok"])

    assert pull_args.command == "pull"
    assert pull_args.model == "mtplx/example"
    assert pull_args.revision == "main"
    assert list_args.command == "list"
    assert list_args.cache_dir == "/tmp/mtplx-models"
    assert remove_args.command == "remove"
    assert remove_args.missing_ok is True


def test_init_parser_exposes_model_cache_and_profile_options():
    parser = build_parser()

    args = parser.parse_args(
        [
            "init",
            "--model",
            "mtplx/example",
            "--model-dir",
            "/tmp/mtplx-models",
            "--profile",
            "exact",
            "--thermal-control",
            "none",
            "--download",
            "--write",
        ]
    )

    assert args.command == "init"
    assert args.model == "mtplx/example"
    assert args.model_dir == "/tmp/mtplx-models"
    assert args.profile == "exact"
    assert args.thermal_control == "none"
    assert args.download is True
    assert args.write is True


def test_compile_audit_dry_run_is_real_command(capsys):
    code = main(
        [
            "profile",
            "compile-audit",
            "--prefill-chunks",
            "128,256",
            "--skip-verify",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr().out
    assert code == 0
    assert '"action": "profile compile-audit"' in captured
    assert "probe_mx_compile_buckets.py" in captured
    assert "--prefill-chunks" in captured
    assert '"MTPLX_VLLM_METAL_PAGED_ATTN_IMPL": "mlx_vector_paged"' in captured
    assert '"attention_impl": "mlx_vector_paged"' in captured


def test_eval_attribution_dry_run_is_real_command(capsys):
    code = main(
        [
            "profile",
            "eval-attribution",
            "--prefix-tokens",
            "64",
            "--orders",
            "outputs,recurrent;recurrent,outputs",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["action"] == "profile eval-attribution"
    assert "probe_eval_attribution.py" in " ".join(payload["command"])
    assert "--prefix-tokens" in payload["command"]
    assert "outputs,recurrent;recurrent,outputs" in payload["command"]
    assert "larger owned kernel boundary" in payload["purpose"]
