from __future__ import annotations

import json

from mtplx.cli import build_parser, main


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

    captured = capsys.readouterr().out
    assert code == 0
    assert '"action": "bench run"' in captured
    assert '"automatic": true' in captured
    assert '"harness": "direct-http"' in captured
    assert '"seed": 42' in captured
    assert "run_context_degradation_diagnostics.py" in captured
    assert '"runtime_profile": "long_response_exact_staged"' in captured
    assert '"MTPLX_EVAL_STATE_ROOTS_ON_COMMIT": "1"' in captured
    assert '"MTPLX_TARGET_LAYER_EVAL_SCHEDULE": "2048:16,8192:8"' in captured


def test_public_bench_cold_run_defaults_to_stable_profile(capsys):
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
    assert payload["profile"]["name"] == "stable"
    assert payload["harness"] == "direct-http"
    assert payload["seed"] == 42
    assert payload["runtime_profile"] == "long_response_exact_staged"
    assert payload["runtime_env"]["MTPLX_TARGET_LAYER_EVAL_SCHEDULE"] == "2048:16,8192:8"


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


def test_chat_and_serve_default_to_stable_profile():
    parser = build_parser()

    run_args = parser.parse_args(["run", "hello", "--cache-dir", "/tmp/mtplx-models"])
    chat_args = parser.parse_args(["chat", "--prompt", "hello"])
    serve_args = parser.parse_args(["serve"])

    assert run_args.profile == "stable"
    assert run_args.prompt_arg == "hello"
    assert run_args.cache_dir == "/tmp/mtplx-models"
    assert chat_args.profile == "stable"
    assert serve_args.profile == "stable"


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


def test_profiles_command_lists_default_without_mlx(capsys):
    code = main(["profiles", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["default"] == "stable"
    assert [profile["name"] for profile in payload["profiles"]] == [
        "stable",
        "performance-cold",
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
