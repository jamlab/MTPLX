"""Handlers for the product-facing MTPLX CLI surface."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mtplx.artifacts import inspect_model
from mtplx.benchmarks.validators.basic import (
    validate_balanced_delimiters,
    validate_no_degenerate_loop,
    validate_python_syntax,
)
from mtplx.constants import DEFAULT_RUNTIME_MODEL_DIR
from mtplx.env import collect_environment
from mtplx.kpi import (
    EXIT_EXACTNESS,
    EXIT_QUALITY,
    EXIT_STRICT_GATE,
    EXIT_UNSUPPORTED_MODEL,
    build_benchmark_envelope,
    default_output_path,
    exact_paged_attention_env,
    prompt_suite_path,
    run_exactness_smoke,
    summarize_vllm_reference,
    write_json,
)
from mtplx.kpi.runtime_kpis import (
    distribution_suite_names,
    repo_root,
)
from mtplx.backends.registry import (
    TIER_ARCH_COMPATIBLE_UNVERIFIED,
)
from mtplx.profiles import DEFAULT_PROFILE_NAME, apply_profile_env, get_profile


DEFAULT_CHAMPION = "models/Qwen3.6-27B-MTPLX-GDN8-Speed4-CyanKiwiMTP"
LONG_RESPONSE_DIRECT_PROFILE = (
    "vllm_metal_paged_attn_partitioned_block_16_blocks_1024_"
    "partition_threshold_2048_impl_mlx_vector_paged"
)


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _benchmark_seed(args: Any, *, runtime_profile: str, harness: str) -> int:
    explicit = getattr(args, "seed", None)
    if explicit is not None:
        return int(explicit)
    if runtime_profile == "native_mtp_60_cold" or harness == "depth-sweep":
        return 0
    return 42


def _bench_run_console_summary(envelope: dict[str, Any]) -> dict[str, Any]:
    runtime = envelope.get("runtime") or {}
    trace = envelope.get("decode_trace") or {}
    quality = envelope.get("quality") or {}
    correctness = envelope.get("correctness") or {}
    smoke = correctness.get("exactness_smoke") or {}
    return {
        "run_id": envelope.get("run_id"),
        "harness": envelope.get("harness") or "depth-sweep",
        "runtime_profile": envelope.get("runtime_profile"),
        "tok_s": runtime.get("tok_s") or runtime.get("mean_tok_s"),
        "generated_tokens": runtime.get("generated_tokens") or trace.get("generated_tokens"),
        "first64_tok_s": trace.get("first64_tok_s"),
        "last64_tok_s": trace.get("last64_tok_s"),
        "last64_over_first64": trace.get("last64_over_first64"),
        "last10_over_first10": trace.get("last10_over_first10"),
        "late_verify_ms": trace.get("late_verify_ms") or runtime.get("late_verify_ms"),
        "quality_passed": quality.get("passed"),
        "quality_failures": quality.get("failures") or [],
        "exactness_smoke_passed": smoke.get("passed"),
        "strict_passed": envelope.get("strict_passed"),
        "artifacts": envelope.get("artifacts") or {},
    }


def _model_gate(
    model: str,
    *,
    unsafe_force_unverified: bool = False,
    yes: bool = False,
) -> tuple[dict[str, Any], int | None]:
    try:
        inspection = inspect_model(model).to_dict()
    except Exception as exc:
        return {"error": "inspect failed", "model": model, "detail": str(exc)}, 1
    compatibility = inspection.get("compatibility") or {}
    tier = compatibility.get("tier")
    exit_code = int(compatibility.get("exit_code", EXIT_UNSUPPORTED_MODEL))
    if exit_code == 0 and compatibility.get("can_run"):
        return inspection, None
    if unsafe_force_unverified and yes and tier == TIER_ARCH_COMPATIBLE_UNVERIFIED:
        print(
            "WARNING: running an architecture-compatible but unverified MTPLX model; "
            "exactness and performance are not guaranteed.",
            file=sys.stderr,
        )
        return inspection, None
    return inspection, exit_code


def _resolve_runtime_model_path(model: str, *, cache_dir: str | None = None) -> tuple[str, dict[str, Any] | None]:
    from mtplx.hf_loader import resolve_model_path

    try:
        return str(resolve_model_path(model, cache_dir=cache_dir)), None
    except Exception as exc:
        return model, {
            "error": "model is not available locally",
            "model": model,
            "detail": str(exc),
        }


def _exactness_profile_kwargs(args: Any) -> dict[str, Any]:
    return {
        "attention_impl": getattr(args, "exactness_attention_impl", "mlx_vector_paged"),
        "block_size": int(getattr(args, "exactness_block_size", 16)),
        "num_blocks": int(getattr(args, "exactness_num_blocks", 1024)),
        "partitioned": not bool(getattr(args, "exactness_no_partitioned", False)),
        "partition_threshold": int(getattr(args, "exactness_partition_threshold", 2048)),
        "partition_size": int(getattr(args, "exactness_partition_size", 512)),
    }


def _exact_paged_env_from_args(args: Any) -> dict[str, str]:
    return exact_paged_attention_env(**_exactness_profile_kwargs(args))


def _depth_sweep_native60(
    *,
    model: str,
    prompt_suite: str,
    max_tokens: int,
    limit: int | None,
    seed: int,
) -> dict[str, Any]:
    from mtplx.benchmarks.runners.mtp_depth_sweep import run_mtp_depth_sweep

    apply_profile_env("performance-cold")
    return run_mtp_depth_sweep(
        model,
        prompt_suite,
        depths="3",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=max_tokens,
        seed=seed,
        limit=limit,
        enable_thinking=False,
        compare_ar=False,
        mtp_hidden_variant="post_norm",
        mtp_cache_policy="persistent",
        mtp_history_policy="committed",
        min_speculative_depth=1,
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
        draft_lm_head_bits=4,
        draft_lm_head_group_size=64,
        draft_lm_head_mode="affine",
    )


class _temporary_env:
    def __init__(self, updates: dict[str, str]) -> None:
        self.updates = updates
        self.previous: dict[str, str | None] = {}

    def __enter__(self) -> None:
        self.previous = {key: os.environ.get(key) for key in self.updates}
        os.environ.update(self.updates)

    def __exit__(self, *_exc: object) -> None:
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def cmd_doctor(args: Any) -> int:
    env = collect_environment(args.project_root).to_dict()
    from mtplx.hf_loader import hf_cache_report

    smc_path = Path(args.smc_path)
    report = {
        "environment": env,
        "huggingface": hf_cache_report(cache_dir=getattr(args, "model_cache", None)),
        "tools": {
            "python": sys.executable,
            "powermetrics": shutil.which("powermetrics"),
            "sudo": shutil.which("sudo"),
            "smc_atlas": str(smc_path),
            "smc_atlas_exists": smc_path.exists(),
            "sovereign": str(args.sovereign_path),
            "sovereign_exists": Path(args.sovereign_path).exists(),
        },
        "policy": {
            "fanmax_counts_for_product_gate": False,
            "benchmark_exactness_smoke_context": 2048,
        },
    }
    _print(report)
    return 0


def cmd_inspect_model_public(args: Any) -> int:
    model_args = list(getattr(args, "model_args", []) or [])
    if model_args and model_args[0] == "model":
        model_args = model_args[1:]
    model = args.model or getattr(args, "model_arg", None) or (model_args[0] if model_args else None)
    if len(model_args) > 1:
        raise SystemExit("inspect accepts exactly one model path/repo id")
    if not model:
        raise SystemExit("inspect requires MODEL or --model MODEL")
    try:
        inspection = inspect_model(model).to_dict()
    except Exception as exc:
        _print({"error": "inspect failed", "model": model, "detail": str(exc)})
        return 1
    if getattr(args, "json", False):
        _print(inspection)
    else:
        _print_inspect_human(inspection)
    compatibility = inspection.get("compatibility") or {}
    exit_code = int(compatibility.get("exit_code", 0))
    if args.require_mtp or getattr(args, "strict_exit_code", True):
        return exit_code
    return 0


def _print_inspect_human(inspection: dict[str, Any]) -> None:
    compatibility = inspection.get("compatibility") or {}
    mtp = inspection.get("mtp") or {}
    architecture = inspection.get("architecture") or inspection.get("model_type") or "unknown"
    tensor_count = mtp.get("tensor_count")
    mtp_layers = inspection.get("mtp_num_hidden_layers")
    runtime_contract_present = bool(inspection.get("runtime_contract_path"))
    print("MTPLX inspect")
    print(f"model: {inspection.get('model_dir')}")
    print(f"source: {inspection.get('source') or 'local'}")
    print(f"architecture: {architecture}")
    print(f"tier: {compatibility.get('tier', 'unknown')}")
    print(f"can_run: {str(bool(compatibility.get('can_run'))).lower()}")
    if mtp_layers is not None:
        print(f"mtp_layers: {mtp_layers}")
    if tensor_count is not None:
        print(f"mtp_tensors_present: {tensor_count}")
    print(f"runtime_contract: {str(runtime_contract_present).lower()}")
    recommended = compatibility.get("recommended_profile") or inspection.get("recommended_profile")
    if recommended:
        print(f"recommended_profile: {recommended}")
    message = compatibility.get("message")
    if message:
        print(f"message: {message}")


def cmd_bench_public(args: Any) -> int:
    action = args.bench_action
    if action == "run":
        return _cmd_bench_run(args)
    if action == "compare":
        return _cmd_bench_compare(args)
    if action == "serve":
        return _cmd_bench_serve(args)
    if action == "reference":
        return _cmd_bench_reference(args)
    if action == "reference-vllm":
        return _cmd_bench_reference_vllm(args)
    raise SystemExit(f"unknown bench action: {action}")


def cmd_pull_public(args: Any) -> int:
    from mtplx.hf_loader import pull_model

    try:
        result = pull_model(args.model, cache_dir=args.cache_dir, revision=args.revision)
    except Exception as exc:
        _print({"error": "pull failed", "model": args.model, "detail": str(exc)})
        return 1
    _print(result)
    return 0


def cmd_list_public(args: Any) -> int:
    from mtplx.hf_loader import list_cached_models, model_cache_dir

    models = [row.to_dict() for row in list_cached_models(cache_dir=args.cache_dir)]
    _print({"cache_dir": str(model_cache_dir(args.cache_dir)), "models": models})
    return 0


def cmd_remove_public(args: Any) -> int:
    from mtplx.hf_loader import remove_cached_model

    result = remove_cached_model(args.model, cache_dir=args.cache_dir)
    _print(result)
    return 0 if result["removed"] or args.missing_ok else 1


def _cmd_bench_run(args: Any) -> int:
    model = args.model or DEFAULT_CHAMPION
    suite = args.suite or "default"
    selected_profile = get_profile(getattr(args, "profile", None) or DEFAULT_PROFILE_NAME)
    prompt_suite = prompt_suite_path(suite)
    run_id = args.run_id or f"cli-bench-{suite}-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir = Path(args.output_dir or "outputs/cli/bench") / run_id
    output = Path(args.output) if args.output else output_dir / "depth-sweep.json"
    envelope_output = output_dir / "envelope.json"
    decode_trace = output_dir / "decode-trace.jsonl"
    exact_paged_env = _exact_paged_env_from_args(args)
    runtime_profile = selected_profile.runtime_profile
    runtime_env = selected_profile.env_dict()
    if selected_profile.name in {"stable", "exact", "max-diagnostic"}:
        runtime_env.update(exact_paged_env)
    harness = getattr(args, "harness", "auto")
    if harness == "auto":
        harness = "depth-sweep" if selected_profile.name == "performance-cold" else "direct-http"
    benchmark_seed = _benchmark_seed(args, runtime_profile=runtime_profile, harness=harness)

    if args.dry_run:
        direct_command = _direct_http_bench_command(
            args,
            model=model,
            suite=suite,
            run_id=run_id,
            output_dir=output_dir,
            seed=benchmark_seed,
        )
        _print(
            {
                "dry_run": True,
                "action": "bench run",
                "model": model,
                "suite": suite,
                "prompt_suite": prompt_suite,
                "exactness_smoke": {
                    "context": 2048,
                    "automatic": True,
                    "profile": _exactness_profile_kwargs(args),
                },
                "harness": harness,
                "seed": benchmark_seed,
                "profile": selected_profile.to_dict(),
                "runtime_profile": runtime_profile,
                "runtime_env": runtime_env,
                "direct_http_command": direct_command if harness == "direct-http" else None,
                "exact_paged_env": exact_paged_env,
                "output": str(output),
                "envelope": str(envelope_output),
                "decode_trace": str(decode_trace),
            }
        )
        return 0
    runtime_model, resolve_error = _resolve_runtime_model_path(
        model,
        cache_dir=getattr(args, "cache_dir", None),
    )
    if resolve_error is not None:
        _print(resolve_error)
        return 1
    inspection, gate_exit = _model_gate(
        runtime_model,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=bool(getattr(args, "yes", False)),
    )
    if gate_exit is not None:
        _print({"error": "model failed MTP primary gate", "model": inspection})
        return gate_exit

    from mtplx.benchmarks.runners.preflight import run_preflight

    preflight = run_preflight(".")
    smoke = run_exactness_smoke(
        runtime_model,
        context=2048,
        prompt_suite=prompt_suite_path("flappy"),
        output=output_dir / "exactness-smoke.json",
        **_exactness_profile_kwargs(args),
    )
    if not smoke["passed"]:
        write_json(envelope_output, {"run_id": run_id, "correctness": {"exactness_smoke": smoke}})
        _print({"error": "Phase 0H exactness smoke failed", "exactness_smoke": smoke})
        return EXIT_EXACTNESS

    if harness == "direct-http":
        return _cmd_bench_run_direct_http(
            args,
            model=runtime_model,
            suite=suite,
            run_id=run_id,
            output_dir=output_dir,
            envelope_output=envelope_output,
            exactness_smoke=smoke,
            runtime_profile=runtime_profile,
            runtime_env=runtime_env,
            preflight=preflight,
            model_inspection=inspection,
            seed=benchmark_seed,
        )

    from mtplx.benchmarks.runners.mtp_depth_sweep import write_depth_sweep

    decode_trace.parent.mkdir(parents=True, exist_ok=True)
    with _temporary_env(
        {
            "MTPLX_DECODE_TRACE_JSONL": str(decode_trace),
            "MTPLX_DECODE_TRACE_INTERVAL_S": str(args.trace_interval_s),
            "MTPLX_DECODE_TRACE_LABEL": run_id,
            **runtime_env,
        }
    ):
        result = _depth_sweep_native60(
            model=runtime_model,
            prompt_suite=prompt_suite,
            max_tokens=args.max_tokens,
            limit=args.limit,
            seed=benchmark_seed,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_depth_sweep(output, result)
    envelope = build_benchmark_envelope(
        result=result,
        model_inspection=inspection,
        run_id=run_id,
        suite=suite,
        runtime_profile=runtime_profile,
        runtime_env=runtime_env,
        exactness_smoke=smoke,
        fan_controlled=bool(args.fanmax),
        strict=bool(args.strict),
        strict_cold=bool(args.strict_cold),
        telemetry={
            "telemetry_unavailable": False,
            "source": "bench-preflight",
            "power": preflight.get("power"),
            "issues": preflight.get("issues"),
            "deep_smc_trace_attached": False,
        },
        decode_trace_path=decode_trace,
    )
    envelope["artifacts"] = {
        "depth_sweep": str(output),
        "envelope": str(envelope_output),
        "decode_trace": str(decode_trace),
    }
    write_json(envelope_output, envelope)
    _print(_bench_run_console_summary(envelope))

    if not envelope["quality"]["passed"]:
        return EXIT_QUALITY
    if args.strict or args.strict_cold:
        if envelope.get("strict_passed") is False:
            return EXIT_STRICT_GATE
    return 0


def _direct_http_bench_command(
    args: Any,
    *,
    model: str,
    suite: str,
    run_id: str,
    output_dir: Path,
    seed: int,
) -> list[str]:
    test_name = "python_modules_long" if suite in {"python_modules_long", "python-modules-long"} else suite
    if test_name == "long_code_uncapped":
        test_name = "long_code_uncapped"
    if test_name not in {"flappy", "python_modules_long", "long_code_uncapped", "long_code"}:
        test_name = "flappy"
    return [
        sys.executable,
        str(repo_root() / "scripts" / "run_context_degradation_diagnostics.py"),
        "local-ablation",
        "--label",
        run_id,
        "--run-id",
        run_id,
        "--output-dir",
        str(output_dir / "direct-http"),
        "--port",
        str(getattr(args, "port", 8041) or 8041),
        "--model",
        model,
        "--model-id",
        Path(model).name,
        "--generation-mode",
        "mtp",
        "--load-mtp",
        "--depth",
        "3",
        "--verify-strategy",
        "capture_commit",
        "--verify-core",
        "linear-gdn-from-conv-tape",
        "--profiles",
        LONG_RESPONSE_DIRECT_PROFILE,
        "--tests",
        test_name,
        "--temperature",
        "0.6",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--seed",
        str(seed),
        "--trace-interval-s",
        str(getattr(args, "trace_interval_s", 1.0)),
        "--request-timeout-s",
        "2400",
        "--startup-timeout-s",
        "600",
    ]


def _validation_dicts_for_text(text: str, *, suite: str) -> list[dict[str, Any]]:
    validations = [validate_no_degenerate_loop(text), validate_balanced_delimiters(text)]
    if suite in {"python_modules_long", "python-modules-long", "long_code", "long-code", "cold-long-code-192"}:
        validations.append(validate_python_syntax(text))
    return [validation.__dict__ for validation in validations]


def _cmd_bench_run_direct_http(
    args: Any,
    *,
    model: str,
    suite: str,
    run_id: str,
    output_dir: Path,
    envelope_output: Path,
    exactness_smoke: dict[str, Any],
    runtime_profile: str,
    runtime_env: dict[str, str],
    preflight: dict[str, Any],
    model_inspection: dict[str, Any],
    seed: int,
) -> int:
    command = _direct_http_bench_command(
        args,
        model=model,
        suite=suite,
        run_id=run_id,
        output_dir=output_dir,
        seed=seed,
    )
    proc = subprocess.run(
        command,
        cwd=repo_root(),
        env={**os.environ, **runtime_env},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    context_summary = output_dir / "direct-http" / run_id / "ablation-summary.json"
    row: dict[str, Any] = {}
    if context_summary.exists():
        summary = json.loads(context_summary.read_text(encoding="utf-8"))
        rows = summary.get("rows") or []
        row = dict(rows[0]) if rows else {}
    text = ""
    content_path = row.get("content_path")
    if content_path and Path(content_path).exists():
        text = Path(content_path).read_text(encoding="utf-8", errors="replace")
    validations = _validation_dicts_for_text(text, suite=suite) if text else []
    quality_failures = [
        {
            "prompt_id": row.get("test") or row.get("suite") or suite,
            "validation": validation.get("name"),
            "detail": validation.get("detail", ""),
        }
        for validation in validations
        if not validation.get("passed")
    ]
    trace_summary = row.get("trace_summary") or {}
    runtime = {
        "generated_tokens": row.get("completion_tokens"),
        "elapsed_s": row.get("request_elapsed_s"),
        "tok_s": row.get("decode_tok_s"),
        "mean_tok_s": row.get("decode_tok_s"),
        "verify_ms_per_call": None,
        "late_verify_ms": row.get("late_verify_ms"),
        "accepted_by_depth": None,
        "drafted_by_depth": None,
        "mean_accept_probability_by_depth": None,
        "acceptance_by_depth": None,
        "correction_tokens": None,
        "bonus_tokens": None,
    }
    trace = {
        "path": trace_summary.get("trace_path"),
        "available": bool(trace_summary),
        "buckets": trace_summary.get("trace_rows"),
        "positive_buckets": trace_summary.get("trace_usable_rows"),
        "generated_tokens": trace_summary.get("trace_generated_tokens") or row.get("completion_tokens"),
        "first64_tok_s": row.get("first64"),
        "last64_tok_s": row.get("last64"),
        "last64_over_first64": row.get("last64_over_first64"),
        "first10_tok_s": row.get("first10_tok_s"),
        "last10_tok_s": row.get("last10_tok_s"),
        "last10_over_first10": row.get("last10_over_first10"),
        "late_verify_ms": row.get("late_verify_ms"),
        "cache_gib_last": row.get("last_cache_gib"),
    }
    strict_gates: dict[str, bool] = {}
    if args.strict:
        strict_gates = {
            "flappy_tok_s_ge_50": bool(runtime["tok_s"] is not None and float(runtime["tok_s"]) >= 50.0),
            "last64_over_first64_ge_0_90": bool(trace["last64_over_first64"] is not None and float(trace["last64_over_first64"]) >= 0.90),
            "last10_over_first10_ge_0_85": bool(trace["last10_over_first10"] is not None and float(trace["last10_over_first10"]) >= 0.85),
            "late_verify_le_75ms": bool(trace["late_verify_ms"] is not None and float(trace["late_verify_ms"]) <= 75.0),
            "telemetry_available": False,
            "no_fan_control": not bool(args.fanmax),
        }
    envelope = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "suite": suite,
        "model": model_inspection,
        "fan_controlled": bool(args.fanmax),
        "harness": "direct-http",
        "runtime_profile": runtime_profile,
        "runtime_env": runtime_env,
        "fast_path_env": row.get("fast_path_env"),
        "runtime": runtime,
        "decode_trace": trace,
        "thermal": {
            "telemetry_unavailable": False,
            "source": "bench-preflight",
            "power": preflight.get("power"),
            "issues": preflight.get("issues"),
            "deep_smc_trace_attached": False,
        },
        "dispatch": {"dispatch_trace_attached": False, "command_buffers_per_token": None},
        "quality": {
            "passed": not quality_failures,
            "failures": quality_failures,
            "validations": validations,
            "acceptance_smells": [],
        },
        "correctness": {
            "exactness_smoke": exactness_smoke,
            "full_exactness": None,
            "distribution_exactness": None,
        },
        "strict_gates": strict_gates,
        "strict_passed": all(strict_gates.values()) if strict_gates else None,
        "artifacts": {
            "context_summary": str(context_summary),
            "envelope": str(envelope_output),
            "server_stdout": str(output_dir / "direct-http-command.log"),
            "content": content_path,
            "events": row.get("events_path"),
        },
        "direct_returncode": proc.returncode,
    }
    (output_dir / "direct-http-command.log").write_text(proc.stdout, encoding="utf-8")
    write_json(envelope_output, envelope)
    _print(_bench_run_console_summary(envelope))
    if proc.returncode != 0:
        return EXIT_STRICT_GATE
    if not envelope["quality"]["passed"]:
        return EXIT_QUALITY
    if args.strict and envelope.get("strict_passed") is False:
        return EXIT_STRICT_GATE
    return 0


def _cmd_bench_compare(args: Any) -> int:
    models = args.models or []
    if not models:
        raise SystemExit("bench compare requires --models PATH_A PATH_B [...]")
    suite = args.suite or "champion-bakeoff"
    run_id = args.run_id or f"cli-compare-{time.strftime('%Y%m%d-%H%M%S')}"
    if suite == "champion-bakeoff":
        tasks = [
            {
                "label": "flappy-10k",
                "suite": "flappy",
                "max_tokens": 10000,
                "strict": bool(args.strict),
                "strict_cold": False,
            },
            {
                "label": "python-modules-long",
                "suite": "python_modules_long",
                "max_tokens": 6000,
                "strict": False,
                "strict_cold": False,
            },
            {
                "label": "cold-long-code-192",
                "suite": "cold-long-code-192",
                "max_tokens": 192,
                "strict": False,
                "strict_cold": True,
            },
        ]
    else:
        tasks = [
            {
                "label": suite,
                "suite": suite,
                "max_tokens": int(args.max_tokens),
                "strict": bool(args.strict),
                "strict_cold": bool(args.strict_cold),
            }
        ]
    if args.dry_run:
        _print(
            {
                "dry_run": True,
                "action": "bench compare",
                "models": models,
                "suite": suite,
                "tasks": tasks,
                "exactness_smoke_per_model": True,
            }
        )
        return 0
    results = []
    worst_exit = 0
    for model in models:
        for task in tasks:
            child = type("BenchArgs", (), vars(args).copy())()
            child.model = model
            child.suite = task["suite"]
            child.max_tokens = task["max_tokens"]
            child.strict = task["strict"]
            child.strict_cold = task["strict_cold"]
            child.run_id = f"{run_id}-{Path(model).name}-{task['label']}"
            child.output_dir = str(Path(args.output_dir or "outputs/cli/compare") / run_id)
            code = _cmd_bench_run(child)
            envelope_path = Path(child.output_dir) / child.run_id / "envelope.json"
            envelope = (
                json.loads(envelope_path.read_text(encoding="utf-8"))
                if envelope_path.exists()
                else {}
            )
            results.append(
                {
                    "model": model,
                    "task": task,
                    "exit_code": code,
                    "envelope": envelope,
                }
            )
            worst_exit = max(worst_exit, code)

    scorecards = []
    for model in models:
        model_rows = [row for row in results if row["model"] == model]
        by_label = {row["task"]["label"]: row for row in model_rows}
        quality_passed = all(
            row.get("envelope", {}).get("quality", {}).get("passed") for row in model_rows
        )
        cold_row = by_label.get("cold-long-code-192", {})
        cold_gate = (
            cold_row.get("envelope", {})
            .get("strict_gates", {})
            .get("cold_tok_s_ge_55")
        )
        metrics = {
            label: row.get("envelope", {}).get("runtime", {})
            for label, row in by_label.items()
        }
        eligible = bool(model_rows) and quality_passed and cold_gate is not False
        scorecards.append(
            {
                "model": model,
                "eligible": eligible,
                "quality_passed": quality_passed,
                "cold_tok_s_ge_55": cold_gate,
                "metrics": metrics,
            }
        )
    eligible_scorecards = [row for row in scorecards if row["eligible"]]
    recommended = None
    if eligible_scorecards:
        recommended = max(
            eligible_scorecards,
            key=lambda row: (
                float(row["metrics"].get("flappy-10k", {}).get("tok_s") or 0.0),
                float(row["metrics"].get("python-modules-long", {}).get("tok_s") or 0.0),
                float(row["metrics"].get("cold-long-code-192", {}).get("tok_s") or 0.0),
            ),
        )
    summary = {
        "run_id": run_id,
        "suite": suite,
        "results": results,
        "scorecards": scorecards,
        "recommended_champion": recommended["model"] if recommended else None,
        "champion_policy": "No-fan sustained and quality outrank fanmax tie-breakers.",
    }
    champion_path = Path("mtplx/champion.json")
    write_json(Path(args.output or Path("outputs/cli/compare") / run_id / "summary.json"), summary)
    if args.record_champion and recommended:
        write_json(
            champion_path,
            {
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "model": recommended["model"],
                "suite": suite,
                "reason": "highest eligible no-fan champion-bakeoff scorecard",
                "run_id": run_id,
                "scorecard": recommended,
            },
        )
        summary["recorded_champion"] = str(champion_path)
    _print(summary)
    return worst_exit


def _http_json(url: str, *, timeout: float = 15.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc), "url": url}


def _cmd_bench_serve(args: Any) -> int:
    base = args.url.rstrip("/")
    health = _http_json(base + "/health")
    metrics = _http_json(base + "/metrics")
    report = {
        "suite": args.suite or "multiturn-flappy",
        "turns": args.turns,
        "health": health,
        "metrics": metrics,
        "required_cache_hit_ratio_turn_2_plus": 0.85,
        "note": "This smoke validates the server surface; full multi-turn generation remains the serving QA expansion point.",
    }
    _print(report)
    return 0 if health.get("ok") else EXIT_STRICT_GATE


def _cmd_bench_reference(args: Any) -> int:
    _print(
        {
            "action": "bench reference",
            "champion": args.champion,
            "references": args.references,
            "suite": args.suite,
            "max_tokens": args.max_tokens,
            "product_gate": False,
            "note": "Reference commands are diagnostic floors and do not promote MTPLX runs.",
        }
    )
    return 0


def _ssh_command(host: str, remote_script: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        host,
        f"bash -lc {shlex.quote(remote_script)}",
    ]


def _run_ssh(host: str, remote_script: str, *, timeout_s: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _ssh_command(host, remote_script),
        cwd=repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout_s,
    )


def _remote_read_text(host: str, path: str, *, timeout_s: int = 30) -> tuple[str | None, str | None]:
    proc = _run_ssh(host, f"cat {shlex.quote(path)}", timeout_s=timeout_s)
    if proc.returncode != 0:
        return None, proc.stdout[-2000:]
    return proc.stdout, None


def _remote_stat(host: str, path: str, *, timeout_s: int = 30) -> dict[str, Any] | None:
    proc = _run_ssh(host, f"stat -c '%Y %s' {shlex.quote(path)}", timeout_s=timeout_s)
    if proc.returncode != 0:
        return None
    parts = proc.stdout.strip().split()
    if len(parts) < 2:
        return None
    try:
        return {"mtime_epoch": int(parts[0]), "size_bytes": int(parts[1])}
    except ValueError:
        return None


def _remote_epoch(host: str) -> int | None:
    proc = _run_ssh(host, "date +%s", timeout_s=10)
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def _remote_probe_script(args: Any) -> str:
    venv_python = str(Path(args.remote_venv) / "bin" / "python")
    return "\n".join(
        [
            "set -e",
            "echo host=$(hostname)",
            "date -Is",
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
            f"{shlex.quote(venv_python)} - <<'PY'",
            "import importlib.metadata as md",
            "for pkg in ('vllm', 'torch', 'transformers', 'flashinfer-python'):",
            "    try:",
            "        print(f'{pkg}=' + md.version(pkg))",
            "    except Exception as exc:",
            "        print(f'{pkg}=unavailable:{exc}')",
            "PY",
            "command -v nsys",
            f"test -d {shlex.quote(args.remote_phase_dir)}",
            f"test -f {shlex.quote(str(Path(args.remote_phase_dir) / args.remote_run_script))}",
        ]
    )


def _write_remote_artifact(local_dir: Path, name: str, text: str | None) -> str | None:
    if text is None:
        return None
    path = local_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _remote_reference_prompt_line(args: Any) -> str:
    prompt_path = Path(prompt_suite_path(args.suite or "flappy"))
    with prompt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                break
        else:
            raise SystemExit(f"empty prompt suite: {prompt_path}")
    row["suite"] = args.suite or row.get("suite") or "reference"
    row["max_tokens"] = int(args.max_tokens)
    row.setdefault("id", f"reference_{args.suite or 'prompt'}")
    if "prompt" not in row:
        raise SystemExit(f"remote vLLM reference requires prompt field in {prompt_path}")
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def _remote_capture_script(args: Any, *, remote_run_cmd: str) -> str:
    prompt_line = _remote_reference_prompt_line(args)
    prompt_file = str(Path(args.remote_phase_dir) / "profile_prompt.jsonl")
    backup_file = str(Path(args.remote_phase_dir) / ".profile_prompt.jsonl.mtplx_backup")
    return "\n".join(
        [
            "set -e",
            f"if [ -f {shlex.quote(prompt_file)} ]; then cp {shlex.quote(prompt_file)} {shlex.quote(backup_file)}; fi",
            "restore_prompt() {",
            f"  if [ -f {shlex.quote(backup_file)} ]; then mv {shlex.quote(backup_file)} {shlex.quote(prompt_file)}; fi",
            "}",
            "trap restore_prompt EXIT",
            f"printf '%s\\n' {shlex.quote(prompt_line)} > {shlex.quote(prompt_file)}",
            remote_run_cmd,
        ]
    )


def _remote_offline_capture_script(args: Any, *, remote_out_dir: str) -> str:
    prompt_line = _remote_reference_prompt_line(args)
    offline_python = r'''from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["no-mtp", "mtp5"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.prompt_file.read_text().splitlines() if line.strip()]
    row = rows[0]
    prompt = row["prompt"]
    spec_summary = None
    if args.mode == "mtp5":
        spec_summary = {"method": "mtp", "num_speculative_tokens": 5}
    llm = LLM(
        model=args.model,
        served_model_name="qwen3.6-27b",
        tokenizer=args.model,
        quantization="compressed-tensors",
        tensor_parallel_size=2,
        max_model_len=32768,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        block_size=32,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.90,
        trust_remote_code=False,
        dtype="bfloat16",
        speculative_config=dict(spec_summary) if spec_summary else None,
        profiler_config={"profiler": "cuda"},
        disable_log_stats=True,
    )
    warm = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=min(64, max(1, args.max_tokens)),
        seed=42,
    )
    llm.generate([prompt], warm, use_tqdm=False)
    sampling = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=args.max_tokens,
        seed=42,
    )
    llm.start_profile()
    started = time.perf_counter()
    outputs = llm.generate([prompt], sampling, use_tqdm=False)
    elapsed = time.perf_counter() - started
    completion = outputs[0].outputs[0]
    completion_tokens = len(completion.token_ids)
    result = {
        "summary": {
            "label": f"offline-nsys-{args.mode}",
            "model": "qwen3.6-27b",
            "prompt_count": 1,
            "ok_count": 1,
            "total_completion_tokens": completion_tokens,
            "mean_decode_tok_s": completion_tokens / elapsed if elapsed else None,
            "mean_end_to_end_tok_s": completion_tokens / elapsed if elapsed else None,
        },
        "rows": [
            {
                "id": row.get("id"),
                "suite": row.get("suite"),
                "status": "ok",
                "max_tokens": args.max_tokens,
                "completion_tokens": completion_tokens,
                "wall_s": elapsed,
                "decode_tok_s": completion_tokens / elapsed if elapsed else None,
                "end_to_end_tok_s": completion_tokens / elapsed if elapsed else None,
                "text_prefix": completion.text[:240],
            }
        ],
        "sampler": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 42},
        "speculative_config": spec_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    llm.stop_profile()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    return f"""set -e
OUT_DIR={shlex.quote(remote_out_dir)}
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
cleanup_vllm_children() {{
  for pattern in "[V]LLM::EngineCore" "[V]LLM::Worker" "[v]llm serve" "offline_capture.py" "multiprocessing.resource_tracker"; do
    pids="$(pgrep -f "$pattern" 2>/dev/null | awk -v self="$$" '$1 != self {{print}}' || true)"
    if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null || true; fi
  done
  sleep 2
  for pattern in "[V]LLM::EngineCore" "[V]LLM::Worker" "[v]llm serve" "offline_capture.py" "multiprocessing.resource_tracker"; do
    pids="$(pgrep -f "$pattern" 2>/dev/null | awk -v self="$$" '$1 != self {{print}}' || true)"
    if [ -n "$pids" ]; then kill -KILL $pids 2>/dev/null || true; fi
  done
}}
trap cleanup_vllm_children EXIT
PROMPT_FILE="$OUT_DIR/profile_prompt.jsonl"
OFFLINE_SCRIPT="$OUT_DIR/offline_capture.py"
BENCH_JSON="$OUT_DIR/bench.json"
STDOUT_LOG="$OUT_DIR/stdout.log"
REPORT_BASE="$OUT_DIR/nsys-{shlex.quote(args.remote_mode)}"
printf '%s\\n' {shlex.quote(prompt_line)} > "$PROMPT_FILE"
cat > "$OFFLINE_SCRIPT" <<'PY'
{offline_python}
PY
source {shlex.quote(str(Path(args.remote_venv) / "bin" / "activate"))}
export CUDA_VISIBLE_DEVICES=0,1
export RAY_memory_monitor_refresh_ms=0
export NCCL_CUMEM_ENABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export VLLM_SLEEP_WHEN_IDLE=0
export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_ATTENTION_BACKEND=FLASHINFER
set +e
nsys profile --force-overwrite=true --wait=primary --trace=cuda,nvtx --cuda-graph-trace=graph --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown -o "$REPORT_BASE" \\
  {shlex.quote(str(Path(args.remote_venv) / "bin" / "python"))} "$OFFLINE_SCRIPT" --mode {shlex.quote(args.remote_mode)} --model /home/youssof/ai/models/Qwen3.6-27B-AWQ-BF16-INT4 --prompt-file "$PROMPT_FILE" --output "$BENCH_JSON" --max-tokens {int(args.max_tokens)} > "$STDOUT_LOG" 2>&1
NSYS_STATUS=$?
cleanup_vllm_children
set -e
REP="${{REPORT_BASE}}.nsys-rep"
if [ -f "$REP" ]; then
  nsys stats --force-overwrite=true --force-export=true --report cuda_gpu_kern_sum "$REP" > "$OUT_DIR/cuda_gpu_kern_sum.txt" 2>&1 || true
  nsys stats --force-overwrite=true --force-export=true --report cuda_api_sum "$REP" > "$OUT_DIR/cuda_api_sum.txt" 2>&1 || true
  nsys stats --force-overwrite=true --force-export=true --report cuda_gpu_trace "$REP" > "$OUT_DIR/cuda_gpu_trace.txt" 2>&1 || true
  nsys stats --force-overwrite=true --force-export=true --report nvtx_sum "$REP" > "$OUT_DIR/nvtx_sum.txt" 2>&1 || true
fi
find "$OUT_DIR" -maxdepth 1 -type f -printf '%f %s\\n' | sort
exit "$NSYS_STATUS"
"""


def _cmd_bench_reference_vllm(args: Any) -> int:
    run_id = args.run_id or f"vllm-reference-{args.remote_mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    local_dir = Path(args.remote_output_dir or "outputs/cli/reference-vllm") / run_id
    remote_kind = getattr(args, "remote_capture_kind", "offline")
    remote_out_dir = str(Path(args.remote_phase_dir) / f"nsys-v4-{remote_kind}-{args.remote_mode}")
    remote_script = str(Path(args.remote_phase_dir) / args.remote_run_script)
    remote_run_cmd = (
        f"cd {shlex.quote(args.remote_phase_dir)} && "
        f"bash {shlex.quote(remote_script)} {shlex.quote(args.remote_mode)} {int(args.remote_port)}"
    )
    capture_script = (
        _remote_offline_capture_script(args, remote_out_dir=remote_out_dir)
        if remote_kind == "offline"
        else _remote_capture_script(args, remote_run_cmd=remote_run_cmd)
    )
    if args.dry_run:
        _print(
            {
                "dry_run": True,
                "action": "bench reference-vllm",
                "ssh_host": args.ssh_host,
                "remote_capture_kind": remote_kind,
                "probe_command": _ssh_command(args.ssh_host, _remote_probe_script(args)),
                "capture_command": _ssh_command(args.ssh_host, capture_script),
                "remote_prompt_override": json.loads(_remote_reference_prompt_line(args)),
                "local_output_dir": str(local_dir),
                "product_gate": False,
            }
        )
        return 0

    local_dir.mkdir(parents=True, exist_ok=True)
    probe = _run_ssh(args.ssh_host, _remote_probe_script(args), timeout_s=60)
    _write_remote_artifact(local_dir, "remote-probe.txt", probe.stdout)
    if probe.returncode != 0:
        report = {
            "run_id": run_id,
            "action": "bench reference-vllm",
            "passed": False,
            "error": "remote 3090 probe failed",
            "returncode": probe.returncode,
            "stdout_tail": probe.stdout[-4000:],
        }
        write_json(local_dir / "summary.json", report)
        _print(report)
        return EXIT_STRICT_GATE

    capture_error = None
    capture_started_epoch = None
    if args.capture_dispatch:
        capture_started_epoch = _remote_epoch(args.ssh_host)
        try:
            capture = _run_ssh(args.ssh_host, capture_script, timeout_s=int(args.remote_timeout_s))
        except subprocess.TimeoutExpired as exc:
            capture_error = {
                "error": "remote capture timed out",
                "timeout_s": args.remote_timeout_s,
                "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else None,
            }
        else:
            _write_remote_artifact(local_dir, "remote-capture.txt", capture.stdout)
            if capture.returncode != 0:
                capture_error = {
                    "error": "remote Nsight capture failed",
                    "returncode": capture.returncode,
                    "stdout_tail": capture.stdout[-4000:],
                }

    kernel_text, kernel_error = _remote_read_text(
        args.ssh_host,
        str(Path(remote_out_dir) / "cuda_gpu_kern_sum.txt"),
    )
    cuda_api_text, cuda_api_error = _remote_read_text(
        args.ssh_host,
        str(Path(remote_out_dir) / "cuda_api_sum.txt"),
    )
    bench_text, bench_error = _remote_read_text(
        args.ssh_host,
        str(Path(remote_out_dir) / "bench.json"),
    )
    client_log, _client_error = _remote_read_text(
        args.ssh_host,
        str(Path(remote_out_dir) / "client.log"),
    )
    stdout_log, _stdout_error = _remote_read_text(
        args.ssh_host,
        str(Path(remote_out_dir) / "stdout.log"),
    )
    kernel_stat = _remote_stat(args.ssh_host, str(Path(remote_out_dir) / "cuda_gpu_kern_sum.txt"))
    cuda_api_stat = _remote_stat(args.ssh_host, str(Path(remote_out_dir) / "cuda_api_sum.txt"))
    bench_stat = _remote_stat(args.ssh_host, str(Path(remote_out_dir) / "bench.json"))
    kernel_stale = bool(
        args.capture_dispatch
        and capture_started_epoch is not None
        and kernel_stat
        and int(kernel_stat["mtime_epoch"]) < int(capture_started_epoch)
    )
    cuda_api_stale = bool(
        args.capture_dispatch
        and capture_started_epoch is not None
        and cuda_api_stat
        and int(cuda_api_stat["mtime_epoch"]) < int(capture_started_epoch)
    )
    _write_remote_artifact(local_dir, "cuda_gpu_kern_sum.txt", kernel_text)
    _write_remote_artifact(local_dir, "cuda_api_sum.txt", cuda_api_text)
    _write_remote_artifact(local_dir, "bench.json", bench_text)
    _write_remote_artifact(local_dir, "client.log", client_log)
    _write_remote_artifact(local_dir, "stdout.log", stdout_log)

    reference = summarize_vllm_reference(
        cuda_kernel_summary_text=kernel_text,
        cuda_api_summary_text=cuda_api_text,
        bench_json_text=bench_text,
    )
    capture_artifacts_valid = bool(
        kernel_text
        and bench_text
        and not kernel_stale
        and (not cuda_api_text or not cuda_api_stale)
    )
    capture_warning = None
    if capture_error and capture_artifacts_valid:
        capture_warning = {
            "warning": "remote Nsight command returned nonzero but fresh artifacts were recovered",
            "capture_error": capture_error,
        }
    report = {
        "run_id": run_id,
        "action": "bench reference-vllm",
        "ssh_host": args.ssh_host,
        "remote_mode": args.remote_mode,
        "remote_capture_kind": remote_kind,
        "remote_phase_dir": args.remote_phase_dir,
        "remote_out_dir": remote_out_dir,
        "suite": args.suite,
        "max_tokens": args.max_tokens,
        "capture_dispatch": bool(args.capture_dispatch),
        "capture_started_epoch": capture_started_epoch,
        "product_gate": False,
        "remote_prompt_override": json.loads(_remote_reference_prompt_line(args)),
        "capture_error": None if capture_warning else capture_error,
        "capture_warning": capture_warning,
        "probe_stdout_tail": probe.stdout[-4000:],
        "artifact_errors": {
            "cuda_gpu_kern_sum": kernel_error,
            "cuda_api_sum": cuda_api_error,
            "bench_json": bench_error,
        },
        "artifact_stats": {
            "cuda_gpu_kern_sum": kernel_stat,
            "cuda_api_sum": cuda_api_stat,
            "bench_json": bench_stat,
            "cuda_gpu_kern_sum_stale_for_capture": kernel_stale,
            "cuda_api_sum_stale_for_capture": cuda_api_stale,
        },
        "reference": reference,
        "local_output_dir": str(local_dir),
    }
    write_json(local_dir / "summary.json", report)
    _print(report)
    if args.capture_dispatch:
        return 0 if capture_artifacts_valid else EXIT_STRICT_GATE
    return 0 if bench_text else EXIT_STRICT_GATE


def cmd_qa_public(args: Any) -> int:
    if args.qa_action == "exactness":
        return _cmd_qa_exactness(args)
    if args.qa_action == "distribution":
        return _cmd_qa_distribution(args)
    raise SystemExit(f"unknown qa action: {args.qa_action}")


def _run_exactness_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(repo_root() / "scripts" / "phase0h_paged_verifier_exactness.py"), *args],
        cwd=repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _cmd_qa_exactness(args: Any) -> int:
    output = Path(args.output) if args.output else default_output_path("qa-exactness")
    cmd = [
        "--model",
        args.model,
        "--contexts",
        args.contexts,
        "--attention-impl",
        str(args.exactness_attention_impl),
        "--block-size",
        str(args.exactness_block_size),
        "--num-blocks",
        str(args.exactness_num_blocks),
        "--partition-threshold",
        str(args.exactness_partition_threshold),
        "--partition-size",
        str(args.exactness_partition_size),
        "--output",
        str(output),
    ]
    if args.exactness_no_partitioned:
        cmd.append("--no-partitioned")
    else:
        cmd.append("--partitioned")
    if args.prompt_suite:
        cmd.extend(["--prompt-suite", args.prompt_suite])
    proc = _run_exactness_command(cmd)
    print(proc.stdout, end="")
    return 0 if proc.returncode == 0 else EXIT_EXACTNESS


def _cmd_qa_distribution(args: Any) -> int:
    suite_names = distribution_suite_names(args.suite)
    rows = []
    worst = 0
    for suite_name in suite_names:
        output = Path(args.output_dir or "outputs/cli/qa-distribution") / f"{suite_name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
        proc = _run_exactness_command(
            [
                "--model",
                args.model,
                "--contexts",
                args.contexts,
                "--attention-impl",
                str(args.exactness_attention_impl),
                "--block-size",
                str(args.exactness_block_size),
                "--num-blocks",
                str(args.exactness_num_blocks),
                "--partition-threshold",
                str(args.exactness_partition_threshold),
                "--partition-size",
                str(args.exactness_partition_size),
                "--prompt-suite",
                prompt_suite_path(suite_name),
                "--output",
                str(output),
                *(
                    ["--no-partitioned"]
                    if args.exactness_no_partitioned
                    else ["--partitioned"]
                ),
            ]
        )
        rows.append(
            {
                "suite": suite_name,
                "output": str(output),
                "returncode": proc.returncode,
                "passed": proc.returncode == 0,
                "stdout_tail": proc.stdout[-2000:],
            }
        )
        worst = max(worst, proc.returncode)
    report = {
        "model": args.model,
        "reference_stack": args.reference_stack,
        "contexts": args.contexts,
        "tolerance": args.tolerance,
        "rows": rows,
        "passed": worst == 0,
    }
    _print(report)
    return 0 if worst == 0 else EXIT_EXACTNESS


def cmd_profile_public(args: Any) -> int:
    if args.profile_action == "dispatch":
        return _cmd_profile_dispatch(args)
    if args.profile_action == "thermal":
        return _cmd_profile_thermal(args)
    if args.profile_action == "compile-audit":
        return _cmd_profile_compile_audit(args)
    raise SystemExit(f"unknown profile action: {args.profile_action}")


def _cmd_profile_dispatch(args: Any) -> int:
    if args.trace:
        out_dir = Path(args.output_dir or "outputs/cli/dispatch") / time.strftime("%Y%m%d-%H%M%S")
        proc = subprocess.run(
            [
                sys.executable,
                str(repo_root() / "scripts" / "analyze_metal_command_trace.py"),
                args.trace,
                "--out-dir",
                str(out_dir),
            ],
            cwd=repo_root(),
            text=True,
            check=False,
        )
        return proc.returncode
    _print(
        {
            "action": "profile dispatch",
            "model": args.model,
            "suite": args.suite,
            "max_tokens": args.max_tokens,
            "implemented_capture": False,
            "next": "Run with --trace PATH to analyze an existing MLX Metal command trace.",
        }
    )
    return 0


def _cmd_profile_thermal(args: Any) -> int:
    cmd = [
        sys.executable,
        str(repo_root() / "scripts" / "run_flappy_smc_thermal_diagnostics.py"),
        "--model",
        args.model,
        "--run-id",
        args.run_id or f"cli-thermal-{time.strftime('%Y%m%d-%H%M%S')}",
        "--output-dir",
        args.output_dir or "outputs/cli/thermal",
    ]
    if args.dry_run:
        _print({"dry_run": True, "command": cmd})
        return 0
    return subprocess.call(cmd, cwd=repo_root())


def _cmd_profile_compile_audit(args: Any) -> int:
    inspection, gate_exit = _model_gate(args.model)
    output = Path(args.output) if args.output else (
        Path(args.output_dir or "outputs/cli/compile-audit")
        / f"compile-audit-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    cmd = [
        sys.executable,
        str(repo_root() / "scripts" / "probe_mx_compile_buckets.py"),
        "--model",
        args.model,
        "--prompts",
        args.prompts,
        "--prompt-index",
        str(args.prompt_index),
        "--prefill-chunks",
        args.prefill_chunks,
        "--depths",
        args.depths,
        "--max-tokens",
        str(args.max_tokens),
        "--repeats",
        str(args.repeats),
        "--warmup",
        str(args.warmup),
        "--verify-core",
        args.verify_core,
        "--output",
        str(output),
    ]
    if args.disable_thinking:
        cmd.append("--disable-thinking")
    if args.skip_prefill:
        cmd.append("--skip-prefill")
    if args.skip_verify:
        cmd.append("--skip-verify")
    if args.dry_run:
        _print(
            {
                "dry_run": True,
                "action": "profile compile-audit",
                "model": inspection,
                "exactness_smoke": {
                    "automatic": not args.skip_exactness_smoke,
                    "context": 2048,
                    "profile": _exactness_profile_kwargs(args),
                },
                "exact_paged_env": _exact_paged_env_from_args(args),
                "command": cmd,
                "output": str(output),
            }
        )
        return 0
    if gate_exit is not None:
        _print({"error": "model failed MTP primary gate", "model": inspection})
        return gate_exit
    if not args.skip_exactness_smoke:
        smoke = run_exactness_smoke(
            args.model,
            context=2048,
            prompt_suite=prompt_suite_path("flappy"),
            output=output.with_name(output.stem + "-exactness-smoke.json"),
            **_exactness_profile_kwargs(args),
        )
        if not smoke["passed"]:
            write_json(
                output.with_name(output.stem + "-failed.json"),
                {"error": "Phase 0H exactness smoke failed", "exactness_smoke": smoke},
            )
            _print({"error": "Phase 0H exactness smoke failed", "exactness_smoke": smoke})
            return EXIT_EXACTNESS
    proc = subprocess.run(
        cmd,
        cwd=repo_root(),
        env={**os.environ, **_exact_paged_env_from_args(args)},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout, end="")
    return proc.returncode


def cmd_thermal_public(args: Any) -> int:
    if args.thermal_action != "fanmax-run":
        raise SystemExit(f"unknown thermal action: {args.thermal_action}")
    run_id = args.run_id or f"cli-fanmax-{time.strftime('%Y%m%d-%H%M%S')}"
    child = [
        sys.executable,
        "-m",
        "mtplx.cli",
        "bench",
        "run",
        "--model",
        args.model,
        "--suite",
        args.suite,
        "--max-tokens",
        str(args.max_tokens),
        "--run-id",
        run_id,
        "--fanmax",
    ]
    cmd = [
        sys.executable,
        str(repo_root() / "scripts" / "run_fanmax_command.py"),
        "--output-dir",
        args.output_dir or "outputs/cli/fanmax",
        "--",
        *child,
    ]
    if args.dry_run:
        _print({"dry_run": True, "command": cmd})
        return 0
    return subprocess.call(cmd, cwd=repo_root())


def cmd_serve_public(args: Any) -> int:
    profile = get_profile(getattr(args, "profile", None) or DEFAULT_PROFILE_NAME)
    runtime_model, resolve_error = _resolve_runtime_model_path(
        args.model,
        cache_dir=getattr(args, "cache_dir", None),
    )
    if resolve_error is not None:
        _print(resolve_error)
        return 1
    inspection, gate_exit = _model_gate(
        runtime_model,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=bool(getattr(args, "yes", False)),
    )
    if gate_exit is not None:
        _print({"error": "model failed MTPLX compatibility gate", "model": inspection})
        return gate_exit
    cmd = [
        sys.executable,
        str(repo_root() / "scripts" / "serve_openai_mtplx.py"),
        "--model",
        runtime_model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--depth",
        str(args.depth),
        "--profile",
        profile.name,
        "--verify-strategy",
        "capture_commit",
        "--verify-core",
        "linear-gdn-from-conv-tape",
        "--draft-lm-head-bits",
        "4",
        "--draft-lm-head-group-size",
        "64",
        "--draft-lm-head-mode",
        "affine",
    ]
    os.execvpe(sys.executable, cmd, os.environ.copy())
    return 0


def _generate_one_shot_public(args: Any, *, command: str) -> tuple[int, dict[str, Any], list[Any]]:
    prompt = getattr(args, "prompt", None) or getattr(args, "prompt_arg", None)
    if not prompt:
        raise SystemExit(f"mtplx {command} requires a prompt")
    runtime_model, resolve_error = _resolve_runtime_model_path(
        args.model,
        cache_dir=getattr(args, "cache_dir", None),
    )
    if resolve_error is not None:
        return 1, resolve_error, []
    inspection, gate_exit = _model_gate(
        runtime_model,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=bool(getattr(args, "yes", False)),
    )
    if gate_exit is not None:
        return gate_exit, {"error": "model failed MTP primary gate", "model": inspection}, []
    profile = get_profile(getattr(args, "profile", None) or DEFAULT_PROFILE_NAME)
    apply_profile_env(profile.name)

    from mtplx.benchmarks.schema import PromptCase, encode_prompt_case
    from mtplx.generation import generate_mtpk
    from mtplx.runtime import load
    from mtplx.sampling import SamplerConfig

    rt = load(runtime_model, mtp=True)
    messages = None
    system = getattr(args, "system", None)
    if system:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    case = PromptCase(
        id=f"cli_{command}",
        category=command,
        prompt=prompt,
        max_tokens=args.max_tokens,
        messages=messages,
    )
    prompt_ids = encode_prompt_case(rt.tokenizer, case, chat_template=True, enable_thinking=False)
    out = generate_mtpk(
        rt,
        prompt_ids,
        max_tokens=args.max_tokens,
        sampler=SamplerConfig(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k),
        speculative_depth=args.depth,
        seed=args.seed,
        mtp_hidden_variant="post_norm",
        mtp_cache_policy="persistent",
        mtp_history_policy="committed",
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
    )
    validations = [
        validate_no_degenerate_loop(out.text),
        validate_balanced_delimiters(out.text),
    ]
    if args.expect_python:
        validations.append(validate_python_syntax(out.text))
    payload = {
        "text": out.text,
        "model": inspection,
        "profile": profile.to_dict(),
        "stats": {
            "generated_tokens": out.stats.generated_tokens,
            "tok_s": out.stats.tok_s,
            "verify_ms_per_call": (
                1000.0 * out.stats.verify_time_s / out.stats.verify_calls
                if out.stats.verify_calls
                else None
            ),
        },
        "validations": [v.__dict__ for v in validations],
    }
    return 0 if all(v.passed for v in validations) else EXIT_QUALITY, payload, validations


def cmd_run_public(args: Any) -> int:
    code, payload, _validations = _generate_one_shot_public(args, command="run")
    if "error" in payload:
        _print(payload)
        return code
    if getattr(args, "json", False):
        _print(payload)
    else:
        text = payload["text"]
        print(text, end="" if text.endswith("\n") else "\n")
        if not getattr(args, "quiet", False):
            stats = payload["stats"]
            tok_s = stats.get("tok_s")
            tok_s_text = f"{tok_s:.2f}" if isinstance(tok_s, (int, float)) else "n/a"
            print(
                f"\n[mtplx] profile={payload['profile']['name']} "
                f"tokens={stats.get('generated_tokens')} tok_s={tok_s_text}"
            )
    return code


def cmd_chat_public(args: Any) -> int:
    # Still a one-shot smoke path until the interactive REPL lands in Phase 5.
    code, payload, _validations = _generate_one_shot_public(args, command="chat")
    _print(payload)
    return code
