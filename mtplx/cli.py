"""MTPLX command line interface."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
from pathlib import Path

from .constants import DEFAULT_RUNTIME_MODEL_DIR
from .profiles import (
    DEFAULT_HF_MODEL_ID,
    DEFAULT_MODEL_ID,
    DEFAULT_PROFILE_NAME,
    PROFILE_CHOICES,
    get_profile,
    list_profiles,
)


DEFAULT_TRUTH_MODES = (
    "ar",
    "mtp1_batched",
    "mtp1_graphbank",
    "d2_batched",
    "d2_graphbank_capture_commit",
    "d2_graphbank_capture_commit_linear_gdn",
    "d2_graphbank_capture_commit_linear_gdn_committed",
    "d2_correction_cache_d2only",
    "d2_c3_blend015",
    "d3_c3_blend015",
)
DEFAULT_C3_CORRECTOR = Path(
    "outputs/correctors/logit-corrector-20260428-012607-c3-logit-r16.npz"
)


VERIFY_CORE_CHOICES = [
    "stock",
    "linear-gdn",
    "linear-gdn-len5",
    "linear-gdn-from-conv",
    "linear-gdn-from-conv-len5",
    "linear-gdn-from-conv-stream",
    "linear-gdn-from-conv-stream-len5",
    "linear-gdn-from-conv-stream-skip0",
    "linear-gdn-from-conv-stream-skip0-len5",
    "linear-gdn-from-conv-tape",
    "linear-gdn-from-conv-tape-len5",
    "linear-gdn-from-conv-inline-g",
    "linear-gdn-from-conv-inline-g-len5",
    "linear-gdn-final",
]

NATIVE_MTP_60_MODEL = DEFAULT_MODEL_ID


def _comma_floats(value: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def cmd_bench_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_bench_public as handler

    return handler(args)


def cmd_chat_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_chat_public as handler

    return handler(args)


def cmd_doctor(args: argparse.Namespace) -> int:
    from .commands.public import cmd_doctor as handler

    return handler(args)


def cmd_inspect_model_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_inspect_model_public as handler

    return handler(args)


def cmd_profile_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_profile_public as handler

    return handler(args)


def cmd_pull_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_pull_public as handler

    return handler(args)


def cmd_list_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_list_public as handler

    return handler(args)


def cmd_remove_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_remove_public as handler

    return handler(args)


def cmd_run_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_run_public as handler

    return handler(args)


def cmd_qa_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_qa_public as handler

    return handler(args)


def cmd_serve_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_serve_public as handler

    return handler(args)


def cmd_thermal_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_thermal_public as handler

    return handler(args)


def cmd_max_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_max_public as handler

    return handler(args)


def _cmd_env(args: argparse.Namespace) -> int:
    from .env import collect_environment

    snapshot = collect_environment(args.project_root)
    print(snapshot.to_json())
    return 0


def _cmd_bench_preflight(args: argparse.Namespace) -> int:
    from .benchmarks.runners.preflight import run_preflight, write_preflight

    result = run_preflight(
        args.project_root,
        top_limit=args.top_limit,
        cpu_threshold=args.cpu_threshold,
        min_free_gib=args.min_free_gib,
    )
    if args.output:
        write_preflight(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["clean"] or not args.strict else 2


def _cmd_inspect_model(args: argparse.Namespace) -> int:
    from .artifacts import inspect_model

    try:
        inspection = inspect_model(args.model)
    except Exception as exc:
        print(
            json.dumps(
                {"error": "inspect failed", "model": args.model, "detail": str(exc)},
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(inspection.to_json())
    compatibility = inspection.compatibility or {}
    if args.require_mtp or getattr(args, "strict_exit_code", True):
        return int(compatibility.get("exit_code", 0))
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    from .hf_loader import model_cache_dir, pull_model
    from .thermal import detect_thermal_control

    config_path = Path(args.config).expanduser()
    model_dir = model_cache_dir(args.model_dir)
    thermal_detection = detect_thermal_control()
    selected_thermal = thermal_detection.get("selected") or {}
    thermal_tool = selected_thermal.get("kind", "none")
    hardware = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "is_macos": platform.system() == "Darwin",
        "is_apple_silicon": platform.system() == "Darwin" and platform.machine() == "arm64",
    }
    profile = get_profile(args.profile)
    commands = {
        "doctor": "mtplx doctor --json",
        "pull": f"mtplx pull {args.model}",
        "inspect": f"mtplx inspect {args.model} --json",
        "run": f"mtplx run \"hello\" --model {args.model}",
        "serve": f"mtplx serve --model {args.model}",
    }
    report = {
        "status": "ready_for_init",
        "config_path": str(config_path),
        "dry_run": bool(args.dry_run),
        "model": args.model,
        "model_dir": str(model_dir),
        "profile": profile.to_dict(),
        "hardware": hardware,
        "thermal_control": {
            "requested": args.thermal_control,
            "detected": thermal_tool,
            "details": thermal_detection,
        },
        "download_requested": bool(args.download),
        "downloaded": False,
        "wrote_config": False,
        "commands": commands,
        "next_steps": list(commands.values()),
    }
    if args.write and not args.dry_run:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            "# MTPLX user configuration\n"
            f"model = {json.dumps(args.model)}\n"
            f"model_dir = {json.dumps(str(model_dir))}\n"
            f"profile = {json.dumps(profile.name)}\n"
            f"thermal_control = {json.dumps(args.thermal_control)}\n",
            encoding="utf-8",
        )
        report["wrote_config"] = True
    if args.download and not args.dry_run:
        try:
            report["download_result"] = pull_model(args.model, cache_dir=model_dir)
        except Exception as exc:
            report["download_error"] = str(exc)
            if args.json:
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                print("MTPLX init")
                print(f"download failed: {exc}")
            return 1
        report["downloaded"] = True
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("MTPLX init")
        print(f"config: {config_path}")
        print(f"model: {args.model}")
        print(f"profile: {profile.name}")
        print(f"model cache: {model_dir}")
        print(
            "hardware: "
            f"{hardware['system']} {hardware['release']} {hardware['machine']} "
            f"(apple_silicon={str(hardware['is_apple_silicon']).lower()})"
        )
        print(f"thermal control: {thermal_tool}")
        if args.write and not args.dry_run:
            print("wrote config")
        else:
            print("dry run: no files written")
        if args.download and not args.dry_run:
            print("downloaded model")
        print(f"next: {commands['doctor']}")
        print(f"next: {commands['pull']}")
    return 0


def _cmd_profiles(args: argparse.Namespace) -> int:
    payload = {"default": DEFAULT_PROFILE_NAME, "profiles": list_profiles()}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"default: {DEFAULT_PROFILE_NAME}")
    for profile in payload["profiles"]:
        print(f"{profile['name']}: {profile['summary']}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    if getattr(args, "bench_action", None):
        return cmd_bench_public(args)
    if args.profile:
        return _cmd_bench_profile(args)
    from .benchmarks.runners.harness import run_manifest_only
    from .benchmarks.schema import BenchmarkConfig, now_run_id

    out = Path(args.output) if args.output else Path("outputs") / f"{now_run_id(args.backend)}.jsonl"
    config = BenchmarkConfig(
        backend=args.backend,
        model_path=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        draft_temperature=args.draft_temperature,
        draft_top_p=args.draft_top_p,
        draft_top_k=args.draft_top_k,
        max_tokens=args.max_tokens,
        speculative_depth=args.speculative_depth,
        adaptive=args.adaptive,
    )
    if args.backend != "manifest":
        raise SystemExit("Only backend=manifest is implemented in this scaffold gate")
    records = run_manifest_only(args.prompts, config, out)
    print(json.dumps({"records": len(records), "output": str(out)}, indent=2))
    return 0


def _suite_to_prompts(suite: str | None, fallback: str) -> str:
    if suite is None:
        return fallback
    suites = {
        "default": "mtplx/benchmarks/prompts/default.jsonl",
        "long_code": "mtplx/benchmarks/prompts/long_code.jsonl",
        "calibration_coding": "mtplx/benchmarks/prompts/calibration_coding.jsonl",
    }
    if suite not in suites:
        raise SystemExit(f"unknown benchmark suite: {suite}")
    return suites[suite]


def _cmd_bench_profile(args: argparse.Namespace) -> int:
    from .benchmarks.runners.mtp_depth_sweep import run_mtp_depth_sweep, write_depth_sweep
    from .benchmarks.runners.preflight import run_preflight
    from .benchmarks.schema import now_run_id

    profile = get_profile(args.profile)
    if profile.name != "performance-cold":
        raise SystemExit(f"unknown benchmark profile: {args.profile}")
    for key, value in profile.env_dict().items():
        os.environ[key] = value
    preflight = None
    if args.strict:
        preflight = run_preflight(
            ".",
            cpu_threshold=args.cpu_threshold,
            min_free_gib=args.min_free_gib,
        )
        if not preflight["clean"]:
            print(json.dumps({"profile": profile.name, "preflight": preflight}, indent=2, sort_keys=True))
            return 2
    prompts = _suite_to_prompts(args.suite, args.prompts)
    out = Path(args.output) if args.output else Path("outputs") / f"{now_run_id(profile.name)}.json"
    result = run_mtp_depth_sweep(
        NATIVE_MTP_60_MODEL if args.model == str(DEFAULT_RUNTIME_MODEL_DIR) else args.model,
        prompts,
        depths="3",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=192 if args.max_tokens == 128 else args.max_tokens,
        seed=0,
        limit=args.limit,
        enable_thinking=False if args.disable_thinking else None,
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
    result["profile"] = {
        **profile.to_dict(),
        "fast_path_env": profile.env_dict(),
        "model": NATIVE_MTP_60_MODEL,
        "depth": 3,
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
        "draft_lm_head": {"bits": 4, "group_size": 64, "mode": "affine"},
        "expected_mlx_qmv_fork_commit": profile.required_mlx_fork_commit,
        "strict_preflight": bool(args.strict),
        "preflight": preflight,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    write_depth_sweep(out, result)
    print(json.dumps({"profile": profile.name, "output": str(out)}, indent=2, sort_keys=True))
    return 0


def _cmd_runtime_smoke(args: argparse.Namespace) -> int:
    from .benchmarks.runners.runtime_smoke import run_runtime_smoke

    result = run_runtime_smoke(args.model, args.prompt)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["mtp_enabled"] and result["mtp_valid"] else 2


def _cmd_probe_contract(args: argparse.Namespace) -> int:
    from .benchmarks.runners.contract_probe import run_contract_probe

    result = run_contract_probe(
        args.model,
        args.prompts,
        max_prompt_tokens=args.max_prompt_tokens,
        chat_template=not args.raw_prompts,
        enable_thinking=False if args.disable_thinking else None,
    )
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_verify_ratio(args: argparse.Namespace) -> int:
    from .benchmarks.runners.verify_ratio import run_verify_ratio

    result = run_verify_ratio(
        args.model,
        args.prompt,
        max_k=args.max_k,
        repeats=args.repeats,
    )
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_verify_profile(args: argparse.Namespace) -> int:
    from .benchmarks.runners.verify_profile import run_verify_profile, write_verify_profile

    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    result = run_verify_profile(
        args.model,
        args.prompts,
        lengths=lengths,
        repeats=args.repeats,
        warmup=args.warmup,
        prompt_index=args.prompt_index,
        enable_thinking=False if args.disable_thinking else None,
    )
    if args.output:
        write_verify_profile(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_verify_qmm_probe(args: argparse.Namespace) -> int:
    from .benchmarks.runners.verify_qmm_probe import run_verify_qmm_probe, write_verify_qmm_probe

    result = run_verify_qmm_probe(
        args.model,
        m_values=args.m_values,
        repeats=args.repeats,
        warmup=args.warmup,
        include=args.include,
        dtype=args.dtype,
        mtp=not args.no_mtp,
        max_groups=args.max_groups,
        seed=args.seed,
        dense_mirror=args.dense_mirror,
    )
    if args.output:
        write_verify_qmm_probe(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_multi_qmv_probe(args: argparse.Namespace) -> int:
    from .benchmarks.runners.multi_qmv_probe import run_multi_qmv_probe, write_multi_qmv_probe

    result = run_multi_qmv_probe(
        args.model,
        include=args.include,
        repeats=args.repeats,
        warmup=args.warmup,
        dtype=args.dtype,
        seed=args.seed,
        mtp=not args.no_mtp,
    )
    if args.output:
        write_multi_qmv_probe(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_batch_equivalence(args: argparse.Namespace) -> int:
    from .benchmarks.runners.batch_equivalence import run_batch_equivalence, write_batch_equivalence

    result = run_batch_equivalence(
        args.model,
        args.prompts,
        suffix_len=args.suffix_len,
        limit=args.limit,
        expand_to=args.expand_to,
        enable_thinking=False if args.disable_thinking else None,
        tolerance=args.tolerance,
    )
    if args.output:
        write_batch_equivalence(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


def _cmd_capture_commit_equivalence(args: argparse.Namespace) -> int:
    from .benchmarks.runners.capture_commit_equivalence import (
        run_capture_commit_equivalence,
        write_capture_commit_equivalence,
    )

    result = run_capture_commit_equivalence(
        args.model,
        args.prompts,
        suffix_len=args.suffix_len,
        min_keep_tokens=args.min_keep_tokens,
        limit=args.limit,
        expand_to=args.expand_to,
        enable_thinking=False if args.disable_thinking else None,
        tolerance=args.tolerance,
        verify_backend=args.verify_backend,
        verify_core=args.verify_core,
    )
    if args.output:
        write_capture_commit_equivalence(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


def _cmd_mtp1_greedy_gate(args: argparse.Namespace) -> int:
    from .benchmarks.runners.mtp1_gate import run_mtp1_greedy_gate, write_gate_result

    result = run_mtp1_greedy_gate(
        args.model,
        args.prompts,
        max_tokens=args.max_tokens,
        seed=args.seed,
        limit=args.limit,
        expand_to=args.expand_to,
        enable_thinking=False if args.disable_thinking else None,
        verify_strategy=args.verify_strategy,
        verify_core=args.verify_core,
        draft_margin_threshold=args.draft_margin_threshold,
        mtp_quant_bits=args.mtp_quant_bits,
        mtp_quant_group_size=args.mtp_quant_group_size,
        mtp_quant_mode=args.mtp_quant_mode,
    )
    if args.output:
        write_gate_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


def _cmd_mtp1_sampler_smoke(args: argparse.Namespace) -> int:
    from .benchmarks.runners.mtp1_sampler_smoke import run_mtp1_sampler_smoke, write_sampler_smoke

    result = run_mtp1_sampler_smoke(
        args.model,
        args.prompts,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        draft_temperature=args.draft_temperature,
        draft_top_p=args.draft_top_p,
        draft_top_k=args.draft_top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        limit=args.limit,
        enable_thinking=False if args.disable_thinking else None,
        compare_ar=args.compare_ar,
        verify_strategy=args.verify_strategy,
        verify_core=args.verify_core,
        draft_margin_threshold=args.draft_margin_threshold,
        mtp_quant_bits=args.mtp_quant_bits,
        mtp_quant_group_size=args.mtp_quant_group_size,
        mtp_quant_mode=args.mtp_quant_mode,
    )
    if args.output:
        write_sampler_smoke(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    failures = [
        v
        for row in result["rows"]
        for v in row["validations"]
        if not v["passed"]
    ]
    return 0 if not failures else 2


def _cmd_mtp_depth_sweep(args: argparse.Namespace) -> int:
    from .benchmarks.runners.mtp_depth_sweep import run_mtp_depth_sweep, write_depth_sweep

    result = run_mtp_depth_sweep(
        args.model,
        args.prompts,
        depths=args.depths,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        draft_temperature=args.draft_temperature,
        draft_top_p=args.draft_top_p,
        draft_top_k=args.draft_top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        limit=args.limit,
        enable_thinking=False if args.disable_thinking else None,
        compare_ar=args.compare_ar,
        mtp_hidden_variant=args.mtp_hidden_variant,
        mtp_cache_policy=args.mtp_cache_policy,
        mtp_history_policy=args.mtp_history_policy,
        draft_margin_threshold=args.draft_margin_threshold,
        min_speculative_depth=args.min_speculative_depth,
        verify_strategy=args.verify_strategy,
        verify_core=args.verify_core,
        draft_core=args.draft_core,
        mtp_quant_bits=args.mtp_quant_bits,
        mtp_quant_group_size=args.mtp_quant_group_size,
        mtp_quant_mode=args.mtp_quant_mode,
        mtp_adapter_path=args.mtp_adapter,
        mtp_corrector_path=args.mtp_corrector,
        mtp_corrector_blend=args.mtp_corrector_blend,
        online_hidden_corrector_alpha=args.online_hidden_corrector_alpha,
        online_hidden_corrector_decay=args.online_hidden_corrector_decay,
        online_hidden_corrector_warmup=args.online_hidden_corrector_warmup,
        online_hidden_corrector_max_feed_depth=args.online_hidden_corrector_max_feed_depth,
        online_hidden_corrector_key=args.online_hidden_corrector_key,
        online_correction_cache=args.online_correction_cache,
        online_correction_cache_min_depth=args.online_correction_cache_min_depth,
        online_correction_cache_key=args.online_correction_cache_key,
        prompt_correction_cache=args.prompt_correction_cache,
        prompt_correction_cache_min_depth=args.prompt_correction_cache_min_depth,
        adapter_ensemble_q=args.adapter_ensemble_q,
        adapter_ensemble_epsilon=args.adapter_ensemble_epsilon,
        adapter_ensemble_min_depth=args.adapter_ensemble_min_depth,
        mtp_topk_reranker_calib=args.mtp_topk_reranker_calib,
        mtp_topk_reranker_depths=args.mtp_topk_reranker_depths,
        mtp_topk_reranker_topk=args.mtp_topk_reranker_topk,
        mtp_topk_reranker_q_weight=args.mtp_topk_reranker_q_weight,
        mtp_topk_reranker_token_weight=args.mtp_topk_reranker_token_weight,
        mtp_topk_reranker_rank_weight=args.mtp_topk_reranker_rank_weight,
        mtp_topk_reranker_prefix_active_only=not args.mtp_topk_reranker_all_rows,
    )
    if args.output:
        write_depth_sweep(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    failures = [
        v
        for depth in result["depths"]
        for row in depth["rows"]
        for v in row["validations"]
        if not v["passed"]
    ]
    return 0 if not failures else 2


def _cmd_mtp_chain_probe(args: argparse.Namespace) -> int:
    from .benchmarks.runners.mtp_chain_probe import run_mtp_chain_probe, write_mtp_chain_probe

    result = run_mtp_chain_probe(
        args.model,
        args.prompts,
        depth=args.depth,
        limit=args.limit,
        max_prompt_tokens=args.max_prompt_tokens,
        chat_template=not args.raw_prompts,
        enable_thinking=False if args.disable_thinking else None,
        windows=args.windows,
        stride=args.stride,
        top_ranks=args.top_ranks,
        mtp_quant_bits=args.mtp_quant_bits,
        mtp_quant_group_size=args.mtp_quant_group_size,
        mtp_quant_mode=args.mtp_quant_mode,
        base_hidden_variants=args.base_hidden_variants,
        mtp_hidden_variants=args.mtp_hidden_variants,
        cache_policies=args.cache_policies,
        concat_orders=args.concat_orders,
        mtp_position_modes=args.mtp_position_modes,
        history_modes=args.history_modes,
        anchors=args.anchors,
    )
    if args.output:
        write_mtp_chain_probe(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_mtp_tree_probe(args: argparse.Namespace) -> int:
    from .benchmarks.runners.mtp_tree_probe import run_mtp_tree_probe, write_mtp_tree_probe

    result = run_mtp_tree_probe(
        args.model,
        args.prompts,
        depth=args.depth,
        budgets=args.budgets,
        branch_factor=args.branch_factor,
        limit=args.limit,
        windows=args.windows,
        stride=args.stride,
        max_prompt_tokens=args.max_prompt_tokens,
        chat_template=not args.raw_prompts,
        enable_thinking=False if args.disable_thinking else None,
        mtp_quant_bits=args.mtp_quant_bits,
        mtp_quant_group_size=args.mtp_quant_group_size,
        mtp_quant_mode=args.mtp_quant_mode,
        base_hidden_variant=args.base_hidden_variant,
        mtp_hidden_variant=args.mtp_hidden_variant,
        mtp_cache_policy=args.mtp_cache_policy,
        anchor=args.anchor,
    )
    if args.output:
        write_mtp_tree_probe(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_mtp_depth_grid(args: argparse.Namespace) -> int:
    from .benchmarks.runners.mtp_depth_grid import run_mtp_depth_policy_grid, write_depth_grid

    result = run_mtp_depth_policy_grid(
        args.model,
        args.prompts,
        depth=args.depth,
        thresholds=args.thresholds,
        min_depths=args.min_depths,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        draft_temperature=args.draft_temperature,
        draft_top_p=args.draft_top_p,
        draft_top_k=args.draft_top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        limit=args.limit,
        enable_thinking=False if args.disable_thinking else None,
        compare_ar=args.compare_ar,
        mtp_hidden_variant=args.mtp_hidden_variant,
        mtp_cache_policy=args.mtp_cache_policy,
        mtp_history_policy=args.mtp_history_policy,
        verify_strategy=args.verify_strategy,
        mtp_corrector_path=args.mtp_corrector,
        mtp_corrector_blend=args.mtp_corrector_blend,
        store_events=args.store_events,
    )
    if args.output:
        write_depth_grid(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    failures = [
        v
        for cell in result["grid"]
        for row in cell["rows"]
        for v in row["validations"]
        if not v["passed"]
    ]
    return 0 if not failures else 2


def _cmd_mtp_adaptive(args: argparse.Namespace) -> int:
    from .benchmarks.runners.mtp_adaptive import run_mtp_adaptive, write_adaptive

    result = run_mtp_adaptive(
        args.model,
        args.prompts,
        max_depth=args.max_depth,
        min_depth=args.min_depth,
        start_depth=args.start_depth,
        increase_after=args.increase_after,
        decrease_after=args.decrease_after,
        policy_kind=args.policy,
        ev_base_depth=args.ev_base_depth,
        ev_accept_priors=args.ev_accept_priors,
        ev_draft_cost_s=args.ev_draft_cost_s,
        ev_extra_verify_cost_s=args.ev_extra_verify_cost_s,
        ev_baseline_tok_s=args.ev_baseline_tok_s,
        ev_safety_margin=args.ev_safety_margin,
        ev_margin_center=args.ev_margin_center,
        ev_margin_scale=args.ev_margin_scale,
        ev_confidence_weight=args.ev_confidence_weight,
        ev_min_extra_accept_probability=args.ev_min_extra_accept_probability,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        draft_temperature=args.draft_temperature,
        draft_top_p=args.draft_top_p,
        draft_top_k=args.draft_top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        limit=args.limit,
        enable_thinking=False if args.disable_thinking else None,
        compare_ar=args.compare_ar,
        mtp_hidden_variant=args.mtp_hidden_variant,
        mtp_cache_policy=args.mtp_cache_policy,
        mtp_history_policy=args.mtp_history_policy,
        verify_strategy=args.verify_strategy,
        verify_core=args.verify_core,
        mtp_quant_bits=args.mtp_quant_bits,
        mtp_quant_group_size=args.mtp_quant_group_size,
        mtp_quant_mode=args.mtp_quant_mode,
    )
    if args.output:
        write_adaptive(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    failures = [
        v
        for row in result["rows"]
        for v in row["validations"]
        if not v["passed"]
    ]
    return 0 if not failures else 2


def _cmd_dflash_mlx_baseline(args: argparse.Namespace) -> int:
    from .benchmarks.runners.competitor_baselines import (
        run_dflash_mlx_baseline,
        write_competitor_result,
    )

    result = run_dflash_mlx_baseline(
        args.model,
        args.draft_model,
        args.prompts,
        dflash_source=args.dflash_source,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        block_size=args.block_size,
        seed=args.seed,
        limit=args.limit,
        enable_thinking=False if args.disable_thinking else None,
        draft_sliding_window_size=args.draft_sliding_window_size,
    )
    if args.output:
        write_competitor_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("error"):
        return 2
    failures = [
        v
        for row in result["rows"]
        for v in row.get("validations", [])
        if not v["passed"]
    ]
    return 0 if not failures else 2


def _cmd_ddtree_mlx_baseline(args: argparse.Namespace) -> int:
    from .benchmarks.runners.competitor_baselines import (
        run_ddtree_mlx_baseline,
        write_competitor_result,
    )

    result = run_ddtree_mlx_baseline(
        args.model,
        args.draft_model,
        args.prompts,
        ddtree_source=args.ddtree_source,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        tree_budget=args.tree_budget,
        limit=args.limit,
        enable_thinking=False if args.disable_thinking else None,
    )
    if args.output:
        write_competitor_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("error"):
        return 2
    failures = [
        v
        for row in result["rows"]
        for v in row.get("validations", [])
        if not v["passed"]
    ]
    return 0 if not failures else 2


def _cmd_truth_report(args: argparse.Namespace) -> int:
    from .benchmarks.runners.truth import run_truth_report, write_truth_report

    result = run_truth_report(
        model_path=args.model,
        prompt_suite=args.prompts,
        modes=args.modes,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        draft_temperature=args.draft_temperature,
        draft_top_p=args.draft_top_p,
        draft_top_k=args.draft_top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        limit=args.limit,
        enable_thinking=False if args.disable_thinking else None,
        mtp_hidden_variant=args.mtp_hidden_variant,
        mtp_cache_policy=args.mtp_cache_policy,
        mtp_history_policy=args.mtp_history_policy,
        c3_corrector=args.c3_corrector,
        c3_blend=args.c3_blend,
        project_root=args.project_root,
        min_free_gib=args.min_free_gib,
        cpu_threshold=args.cpu_threshold,
        keep_going=not args.fail_fast,
    )
    output_dir = Path(args.output_dir)
    output_json = Path(args.output_json) if args.output_json else output_dir / f"{result['run_id']}.json"
    output_md = Path(args.output_md) if args.output_md else output_dir / f"{result['run_id']}.md"
    write_truth_report(output_json, output_md, result)
    print(json.dumps({"json": str(output_json), "markdown": str(output_md), "passed": result["passed"], "claim_label": result["claim_label"]}, indent=2, sort_keys=True))
    if args.strict_preflight and not result["preflight"].get("clean"):
        return 2
    return 0 if result["passed"] else 2


def _cmd_session_bank(args: argparse.Namespace) -> int:
    from .benchmarks.runners.session_bank import (
        run_session_bank_benchmark,
        write_session_bank_report,
    )

    result = run_session_bank_benchmark(
        args.model,
        args.prompts,
        prompt_index=args.prompt_index,
        suffix_text=args.suffix_text,
        max_prompt_tokens=args.max_prompt_tokens,
        chat_template=not args.raw_prompts,
        enable_thinking=False if args.disable_thinking else None,
        max_entries=args.max_entries,
        tolerance=args.tolerance,
        restore_mode=args.restore_mode,
    )
    if args.output:
        write_session_bank_report(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["exact"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mtplx")
    sub = parser.add_subparsers(dest="command", required=True)
    default_model = str(DEFAULT_RUNTIME_MODEL_DIR)

    env_p = sub.add_parser("env", help="Print reproducible environment snapshot")
    env_p.add_argument("--project-root", default=".")
    env_p.set_defaults(func=_cmd_env)

    doctor_p = sub.add_parser("doctor", help="Check MTPLX CLI, model, thermal, and tool environment")
    doctor_p.add_argument("--project-root", default=".")
    doctor_p.add_argument("--smc-path", default="/Users/youssof/Documents/Domain Expansion-Infinite watts/smc-atlas/smc")
    doctor_p.add_argument("--sovereign-path", default="/Users/youssof/Documents/Sovereign/build/sovereign")
    doctor_p.add_argument("--model-cache")
    doctor_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    doctor_p.set_defaults(func=cmd_doctor)

    inspect_public_p = sub.add_parser("inspect", help="Inspect a model and auto-check MTP support")
    inspect_public_p.add_argument(
        "model_args",
        nargs="*",
        metavar="MODEL",
        help="Model path/repo id. Legacy form 'inspect model MODEL' is also accepted.",
    )
    inspect_public_p.add_argument("--model")
    inspect_public_p.add_argument("--require-mtp", action="store_true")
    inspect_public_p.add_argument(
        "--no-strict-exit-code",
        action="store_false",
        dest="strict_exit_code",
        help="Always exit 0 after printing the compatibility verdict.",
    )
    inspect_public_p.set_defaults(strict_exit_code=True)
    inspect_public_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    inspect_public_p.set_defaults(func=cmd_inspect_model_public)

    init_p = sub.add_parser("init", help="Initialize MTPLX user config without importing MLX")
    init_p.add_argument("--config", default="~/.mtplx/config.toml")
    init_p.add_argument("--model", default=DEFAULT_HF_MODEL_ID, help="Default verified model repo id or path")
    init_p.add_argument("--model-dir", help="Model cache directory; defaults to MTPLX_MODEL_DIR or ~/.mtplx/models")
    init_p.add_argument("--profile", choices=PROFILE_CHOICES, default=DEFAULT_PROFILE_NAME)
    init_p.add_argument("--thermal-control", choices=("auto", "none"), default="auto")
    init_p.add_argument("--download", action="store_true", help="Download the selected model into the cache")
    init_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    init_p.add_argument("--dry-run", action="store_true", help="Show init actions without writing files")
    init_p.add_argument("--write", action="store_true", help="Write the initial config file")
    init_p.set_defaults(func=_cmd_init)

    profiles_p = sub.add_parser("profiles", help="List MTPLX runtime profiles without importing MLX")
    profiles_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    profiles_p.set_defaults(func=_cmd_profiles)

    pull_p = sub.add_parser("pull", help="Download a Hugging Face model into the MTPLX cache")
    pull_p.add_argument("model", help="Hugging Face repo id or URL")
    pull_p.add_argument("--cache-dir")
    pull_p.add_argument("--revision")
    pull_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    pull_p.set_defaults(func=cmd_pull_public)

    list_p = sub.add_parser("list", help="List locally cached MTPLX models")
    list_p.add_argument("--cache-dir")
    list_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    list_p.set_defaults(func=cmd_list_public)

    remove_p = sub.add_parser("remove", help="Remove a locally cached MTPLX model")
    remove_p.add_argument("model", help="Hugging Face repo id, URL, or cached safe name")
    remove_p.add_argument("--cache-dir")
    remove_p.add_argument("--missing-ok", action="store_true")
    remove_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    remove_p.set_defaults(func=cmd_remove_public)

    run_p = sub.add_parser("run", help="Run a one-shot verified MTPLX completion")
    run_p.add_argument("prompt_arg", nargs="?", help="Prompt text")
    run_p.add_argument("--model", default=default_model)
    run_p.add_argument("--cache-dir")
    run_p.add_argument("--profile", choices=PROFILE_CHOICES, default=DEFAULT_PROFILE_NAME)
    run_p.add_argument("--unsafe-force-unverified", action="store_true")
    run_p.add_argument("--yes", action="store_true", help="Confirm unsafe non-interactive actions")
    run_p.add_argument("--prompt", help="Prompt text, as an alternative to the positional prompt")
    run_p.add_argument("--system", help="Optional system prompt")
    run_p.add_argument("--max-tokens", type=int, default=192)
    run_p.add_argument("--temperature", type=float, default=0.6)
    run_p.add_argument("--top-p", type=float, default=0.95)
    run_p.add_argument("--top-k", type=int, default=20)
    run_p.add_argument("--depth", type=int, default=3)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--quiet", action="store_true", help="Hide the stats footer")
    run_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    run_p.add_argument("--expect-python", action="store_true")
    run_p.add_argument("--max", action="store_true", help="Opt into ThermalForge/TG Pro performance fan profile for this run")
    run_p.set_defaults(func=cmd_run_public)

    chat_p = sub.add_parser("chat", help="Run one native-MTP chat smoke generation")
    chat_p.add_argument("--model", default=default_model)
    chat_p.add_argument("--cache-dir")
    chat_p.add_argument("--profile", choices=PROFILE_CHOICES, default=DEFAULT_PROFILE_NAME)
    chat_p.add_argument("--unsafe-force-unverified", action="store_true")
    chat_p.add_argument("--yes", action="store_true", help="Confirm unsafe non-interactive actions")
    chat_p.add_argument("--prompt", required=True)
    chat_p.add_argument("--max-tokens", type=int, default=192)
    chat_p.add_argument("--temperature", type=float, default=0.6)
    chat_p.add_argument("--top-p", type=float, default=0.95)
    chat_p.add_argument("--top-k", type=int, default=20)
    chat_p.add_argument("--depth", type=int, default=3)
    chat_p.add_argument("--seed", type=int, default=0)
    chat_p.add_argument("--expect-python", action="store_true")
    chat_p.add_argument("--max", action="store_true", help="Opt into ThermalForge/TG Pro performance fan profile for this run")
    chat_p.set_defaults(func=cmd_chat_public)

    serve_p = sub.add_parser("serve", help="Start the OpenAI-compatible MTPLX server")
    serve_p.add_argument("--model", default=default_model)
    serve_p.add_argument("--cache-dir")
    serve_p.add_argument("--profile", choices=PROFILE_CHOICES, default=DEFAULT_PROFILE_NAME)
    serve_p.add_argument("--unsafe-force-unverified", action="store_true")
    serve_p.add_argument("--yes", action="store_true", help="Confirm unsafe non-interactive actions")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--depth", type=int, default=3)
    serve_p.add_argument(
        "--api-key",
        default=os.environ.get("MTPLX_AUTH"),
        help="Require Bearer or X-API-Key auth. Required for non-localhost binds.",
    )
    serve_p.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Requests per minute per client/API key. Use 0 to disable.",
    )
    serve_p.add_argument(
        "--stream-interval",
        type=int,
        default=1,
        help="Committed-token batch size per chat SSE chunk.",
    )
    serve_p.add_argument(
        "--max-tokens",
        dest="max_response_tokens",
        type=int,
        help="Default server-side response-token ceiling.",
    )
    serve_p.add_argument("--default-temperature", dest="temperature", type=float, default=0.6)
    serve_p.add_argument("--default-top-p", dest="top_p", type=float, default=0.95)
    serve_p.add_argument("--reasoning-parser", choices=["qwen3", "none"], default="qwen3")
    serve_p.add_argument("--max", action="store_true", help="Opt into ThermalForge/TG Pro performance fan profile for the server lifetime")
    serve_p.add_argument(
        "--warmup-tokens",
        type=int,
        default=16,
        help="Startup warmup generation length. Use 0 to disable.",
    )
    serve_p.add_argument(
        "--strict-warmup",
        action="store_true",
        help="Fail server startup if the warmup pass fails.",
    )
    serve_p.set_defaults(func=cmd_serve_public)

    preflight_p = sub.add_parser("bench-preflight", help="Check benchmark contamination before speed runs")
    preflight_p.add_argument("--project-root", default=".")
    preflight_p.add_argument("--top-limit", type=int, default=12)
    preflight_p.add_argument("--cpu-threshold", type=float, default=25.0)
    preflight_p.add_argument("--min-free-gib", type=float, default=25.0)
    preflight_p.add_argument("--strict", action="store_true")
    preflight_p.add_argument("--output")
    preflight_p.set_defaults(func=_cmd_bench_preflight)

    inspect_p = sub.add_parser("inspect-model", help="Inspect Qwen/MLX model artifacts")
    inspect_p.add_argument("model")
    inspect_p.add_argument("--require-mtp", action="store_true")
    inspect_p.add_argument(
        "--no-strict-exit-code",
        action="store_false",
        dest="strict_exit_code",
        help="Always exit 0 after printing the compatibility verdict.",
    )
    inspect_p.set_defaults(strict_exit_code=True)
    inspect_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    inspect_p.set_defaults(func=_cmd_inspect_model)

    bench_p = sub.add_parser("bench", help="Run benchmark harness")
    bench_p.add_argument(
        "bench_action",
        nargs="?",
        choices=["run", "compare", "serve", "reference", "reference-vllm"],
        help="Public benchmark action. Omit for legacy benchmark flags.",
    )
    bench_p.add_argument("--backend", default="manifest")
    bench_p.add_argument(
        "--profile",
        choices=(*PROFILE_CHOICES, "native-mtp-60"),
        help=(
            "Runtime profile for product benchmark actions. Defaults to stable; "
            "native-mtp-60 is a legacy alias for performance-cold."
        ),
    )
    bench_p.add_argument(
        "--suite",
        choices=[
            "default",
            "long_code",
            "long-code",
            "long_code_uncapped",
            "long-code-uncapped",
            "calibration_coding",
            "calibration-coding",
            "flappy",
            "python_modules_long",
            "python-modules-long",
            "cold-long-code-192",
            "champion-bakeoff",
            "distribution-smoke",
            "multiturn-flappy",
        ],
    )
    bench_p.add_argument("--strict", action="store_true", help="Run clean-preflight before profile benchmarks")
    bench_p.add_argument("--strict-cold", action="store_true", help="Enforce cold 55 tok/s regression gate")
    bench_p.add_argument("--no-fanmax", action="store_true", help="Mark run as no-fan product candidate")
    bench_p.add_argument("--fanmax", action="store_true", help="Mark run as fan-controlled diagnostic")
    bench_p.add_argument("--max", action="store_true", dest="fanmax", help="Alias for --fanmax")
    bench_p.add_argument("--unsafe-force-unverified", action="store_true")
    bench_p.add_argument("--yes", action="store_true", help="Confirm unsafe non-interactive actions")
    bench_p.add_argument("--dry-run", action="store_true")
    bench_p.add_argument(
        "--harness",
        choices=["auto", "direct-http", "depth-sweep"],
        default="auto",
        help="Benchmark execution harness. auto uses the selected profile's safest harness.",
    )
    bench_p.add_argument("--run-id")
    bench_p.add_argument("--output-dir")
    bench_p.add_argument("--trace-interval-s", type=float, default=1.0)
    bench_p.add_argument("--exactness-attention-impl", default="mlx_vector_paged")
    bench_p.add_argument("--exactness-block-size", type=int, default=16)
    bench_p.add_argument("--exactness-num-blocks", type=int, default=1024)
    bench_p.add_argument("--exactness-no-partitioned", action="store_true")
    bench_p.add_argument("--exactness-partition-threshold", type=int, default=2048)
    bench_p.add_argument("--exactness-partition-size", type=int, default=512)
    bench_p.add_argument("--models", nargs="+")
    bench_p.add_argument("--record-champion", action="store_true")
    bench_p.add_argument("--champion", default="models/Qwen3.6-27B-MTPLX-GDN8-Speed4-CyanKiwiMTP")
    bench_p.add_argument("--references", nargs="+", default=["stock_mlx_lm", "llama_cpp"])
    bench_p.add_argument("--url", default="http://127.0.0.1:8000")
    bench_p.add_argument("--port", type=int, default=8041)
    bench_p.add_argument("--turns", type=int, default=5)
    bench_p.add_argument("--capture-dispatch", action="store_true")
    bench_p.add_argument("--ssh-host", default="mtplx-3090")
    bench_p.add_argument("--remote-phase-dir", default="/home/youssof/ai/mtplx-phase1-v4-20260429-012151")
    bench_p.add_argument("--remote-venv", default="/home/youssof/ai/vllm-venv")
    bench_p.add_argument("--remote-run-script", default="run_nsys_server_capture.sh")
    bench_p.add_argument("--remote-mode", choices=["no-mtp", "mtp5"], default="mtp5")
    bench_p.add_argument("--remote-capture-kind", choices=["offline", "server"], default="offline")
    bench_p.add_argument("--remote-port", type=int, default=8065)
    bench_p.add_argument("--remote-timeout-s", type=int, default=3600)
    bench_p.add_argument("--remote-output-dir")
    bench_p.add_argument("--cpu-threshold", type=float, default=25.0)
    bench_p.add_argument("--min-free-gib", type=float, default=25.0)
    bench_p.add_argument("--model", default=default_model)
    bench_p.add_argument("--cache-dir")
    bench_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    bench_p.add_argument("--output")
    bench_p.add_argument("--temperature", type=float, default=0.6)
    bench_p.add_argument("--top-p", type=float, default=0.95)
    bench_p.add_argument("--top-k", type=int, default=20)
    bench_p.add_argument("--draft-temperature", type=float)
    bench_p.add_argument("--draft-top-p", type=float)
    bench_p.add_argument("--draft-top-k", type=int)
    bench_p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampler seed. Defaults are harness-aware: 42 for long direct-HTTP runs, 0 for cold depth-sweep runs.",
    )
    bench_p.add_argument("--max-tokens", type=int, default=128)
    bench_p.add_argument("--limit", type=int)
    bench_p.add_argument("--disable-thinking", action="store_true")
    bench_p.add_argument("--speculative-depth", type=int, default=0)
    bench_p.add_argument("--adaptive", action="store_true")
    bench_p.set_defaults(func=_cmd_bench)

    qa_p = sub.add_parser("qa", help="Run MTPLX correctness gates")
    qa_sub = qa_p.add_subparsers(dest="qa_action", required=True)
    qa_exact_p = qa_sub.add_parser("exactness", help="Run full Phase 0H paged-verifier exactness")
    qa_exact_p.add_argument("--model", default=default_model)
    qa_exact_p.add_argument("--contexts", default="64,2048,6144,10240")
    qa_exact_p.add_argument("--prompt-suite")
    qa_exact_p.add_argument("--exactness-attention-impl", default="mlx_vector_paged")
    qa_exact_p.add_argument("--exactness-block-size", type=int, default=16)
    qa_exact_p.add_argument("--exactness-num-blocks", type=int, default=1024)
    qa_exact_p.add_argument("--exactness-no-partitioned", action="store_true")
    qa_exact_p.add_argument("--exactness-partition-threshold", type=int, default=2048)
    qa_exact_p.add_argument("--exactness-partition-size", type=int, default=512)
    qa_exact_p.add_argument("--output")
    qa_exact_p.set_defaults(func=cmd_qa_public)
    qa_dist_p = qa_sub.add_parser("distribution", help="Run distribution-level exactness smoke across suites")
    qa_dist_p.add_argument("--model", default=default_model)
    qa_dist_p.add_argument("--reference-stack", default="stock_mlx_lm_ar")
    qa_dist_p.add_argument("--suite", default="distribution-smoke")
    qa_dist_p.add_argument("--contexts", default="2048")
    qa_dist_p.add_argument("--tolerance", default="kl=0.01,chi2_p=0.01")
    qa_dist_p.add_argument("--exactness-attention-impl", default="mlx_vector_paged")
    qa_dist_p.add_argument("--exactness-block-size", type=int, default=16)
    qa_dist_p.add_argument("--exactness-num-blocks", type=int, default=1024)
    qa_dist_p.add_argument("--exactness-no-partitioned", action="store_true")
    qa_dist_p.add_argument("--exactness-partition-threshold", type=int, default=2048)
    qa_dist_p.add_argument("--exactness-partition-size", type=int, default=512)
    qa_dist_p.add_argument("--output-dir")
    qa_dist_p.set_defaults(func=cmd_qa_public)

    profile_public_p = sub.add_parser("profile", help="Profile dispatch, thermal, and compile behavior")
    profile_sub = profile_public_p.add_subparsers(dest="profile_action", required=True)
    profile_dispatch_p = profile_sub.add_parser("dispatch", help="Analyze or prepare dispatch-count profiling")
    profile_dispatch_p.add_argument("--model", default=default_model)
    profile_dispatch_p.add_argument("--suite", default="flappy")
    profile_dispatch_p.add_argument("--max-tokens", type=int, default=2048)
    profile_dispatch_p.add_argument("--trace")
    profile_dispatch_p.add_argument("--output-dir")
    profile_dispatch_p.set_defaults(func=cmd_profile_public)
    profile_thermal_p = profile_sub.add_parser("thermal", help="Run SMC Atlas / powermetrics thermal profile")
    profile_thermal_p.add_argument("--model", default=default_model)
    profile_thermal_p.add_argument("--suite", default="flappy")
    profile_thermal_p.add_argument("--max-tokens", type=int, default=10000)
    profile_thermal_p.add_argument("--no-fanmax", action="store_true")
    profile_thermal_p.add_argument("--run-id")
    profile_thermal_p.add_argument("--output-dir")
    profile_thermal_p.add_argument("--dry-run", action="store_true")
    profile_thermal_p.set_defaults(func=cmd_profile_public)
    profile_compile_p = profile_sub.add_parser("compile-audit", help="Audit mx.compile as a measured lever")
    profile_compile_p.add_argument("--model", default=default_model)
    profile_compile_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/long_code.jsonl")
    profile_compile_p.add_argument("--prompt-index", type=int, default=0)
    profile_compile_p.add_argument("--prefill-chunks", default="128,256,512,1024")
    profile_compile_p.add_argument("--depths", default="3,4")
    profile_compile_p.add_argument("--max-tokens", type=int, default=64)
    profile_compile_p.add_argument("--repeats", type=int, default=2)
    profile_compile_p.add_argument("--warmup", type=int, default=1)
    profile_compile_p.add_argument("--verify-core", default="linear-gdn-from-conv-tape")
    profile_compile_p.add_argument("--exactness-attention-impl", default="mlx_vector_paged")
    profile_compile_p.add_argument("--exactness-block-size", type=int, default=16)
    profile_compile_p.add_argument("--exactness-num-blocks", type=int, default=1024)
    profile_compile_p.add_argument("--exactness-no-partitioned", action="store_true")
    profile_compile_p.add_argument("--exactness-partition-threshold", type=int, default=2048)
    profile_compile_p.add_argument("--exactness-partition-size", type=int, default=512)
    profile_compile_p.add_argument("--skip-prefill", action="store_true")
    profile_compile_p.add_argument("--skip-verify", action="store_true")
    profile_compile_p.add_argument("--skip-exactness-smoke", action="store_true")
    profile_compile_p.add_argument("--disable-thinking", action="store_true")
    profile_compile_p.add_argument("--output")
    profile_compile_p.add_argument("--output-dir")
    profile_compile_p.add_argument("--dry-run", action="store_true")
    profile_compile_p.set_defaults(func=cmd_profile_public)

    thermal_p = sub.add_parser("thermal", help="Thermal diagnostic helpers")
    thermal_sub = thermal_p.add_subparsers(dest="thermal_action", required=True)
    fanmax_p = thermal_sub.add_parser("fanmax-run", help="Run a diagnostic with both fans pinned to max")
    fanmax_p.add_argument("--model", default=default_model)
    fanmax_p.add_argument("--suite", default="flappy")
    fanmax_p.add_argument("--max-tokens", type=int, default=10000)
    fanmax_p.add_argument("--run-id")
    fanmax_p.add_argument("--output-dir")
    fanmax_p.add_argument("--dry-run", action="store_true")
    fanmax_p.set_defaults(func=cmd_thermal_public)

    max_p = sub.add_parser("max", help="Opt-in fan profile control via ThermalForge or TG Pro")
    max_group = max_p.add_mutually_exclusive_group(required=True)
    max_group.add_argument("--on", dest="max_action", action="store_const", const="performance", help="Set the Performance fan profile")
    max_group.add_argument("--max", dest="max_action", action="store_const", const="max", help="Set the Max fan profile")
    max_group.add_argument("--off", dest="max_action", action="store_const", const="silent", help="Restore the Silent fan profile")
    max_group.add_argument("--status", dest="max_action", action="store_const", const="status", help="Show thermal-control status")
    max_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    max_p.add_argument("--dry-run", action="store_true", help="Show the command without changing fan state")
    max_p.set_defaults(func=cmd_max_public)

    smoke_p = sub.add_parser("runtime-smoke", help="Load model, inject MTP, and run one AR/MTP forward")
    smoke_p.add_argument("--model", default=default_model)
    smoke_p.add_argument(
        "--prompt",
        default="def add(a: int, b: int) -> int:\\n    return",
    )
    smoke_p.set_defaults(func=_cmd_runtime_smoke)

    probe_p = sub.add_parser("probe-contract", help="Probe MTP hidden-state and concat-order contracts")
    probe_p.add_argument("--model", default=default_model)
    probe_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    probe_p.add_argument("--max-prompt-tokens", type=int, default=256)
    probe_p.add_argument("--raw-prompts", action="store_true")
    probe_p.add_argument("--disable-thinking", action="store_true")
    probe_p.add_argument("--output")
    probe_p.set_defaults(func=_cmd_probe_contract)

    ratio_p = sub.add_parser("verify-ratio", help="Measure cached forward(k+1) / forward(1)")
    ratio_p.add_argument("--model", default=default_model)
    ratio_p.add_argument(
        "--prompt",
        default="def add(a: int, b: int) -> int:\\n    return",
    )
    ratio_p.add_argument("--max-k", type=int, default=8)
    ratio_p.add_argument("--repeats", type=int, default=3)
    ratio_p.add_argument("--output")
    ratio_p.set_defaults(func=_cmd_verify_ratio)

    profile_p = sub.add_parser("verify-profile", help="Synchronously profile target verify sections")
    profile_p.add_argument("--model", default=default_model)
    profile_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    profile_p.add_argument("--lengths", default="1,2,3,6")
    profile_p.add_argument("--repeats", type=int, default=2)
    profile_p.add_argument("--warmup", type=int, default=1)
    profile_p.add_argument("--prompt-index", type=int, default=0)
    profile_p.add_argument("--disable-thinking", action="store_true")
    profile_p.add_argument("--output")
    profile_p.set_defaults(func=_cmd_verify_profile)

    qmm_probe_p = sub.add_parser(
        "verify-qmm-probe",
        help="Rank isolated QuantizedLinear small-M costs for VerifyCore qmm targets",
    )
    qmm_probe_p.add_argument("--model", default=default_model)
    qmm_probe_p.add_argument("--m-values", default="1,3,4,5,16")
    qmm_probe_p.add_argument("--repeats", type=int, default=5)
    qmm_probe_p.add_argument("--warmup", type=int, default=2)
    qmm_probe_p.add_argument("--include", default="mlp,gdn,attn,lm_head,mtp")
    qmm_probe_p.add_argument("--dtype", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"], default="bf16")
    qmm_probe_p.add_argument("--max-groups", type=int)
    qmm_probe_p.add_argument("--seed", type=int, default=0)
    qmm_probe_p.add_argument("--no-mtp", action="store_true")
    qmm_probe_p.add_argument(
        "--dense-mirror",
        action="store_true",
        help="Also time a BF16/FP16 dequantized dense-weight mirror for each sampled QuantizedLinear.",
    )
    qmm_probe_p.add_argument("--output")
    qmm_probe_p.set_defaults(func=_cmd_verify_qmm_probe)

    qmv_probe_p = sub.add_parser(
        "multi-qmv-probe",
        help="Probe the experimental M=3 multi-vector qmv VerifyCore primitive",
    )
    qmv_probe_p.add_argument("--model", default=default_model)
    qmv_probe_p.add_argument("--include", default="mlp")
    qmv_probe_p.add_argument("--repeats", type=int, default=10)
    qmv_probe_p.add_argument("--warmup", type=int, default=3)
    qmv_probe_p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    qmv_probe_p.add_argument("--seed", type=int, default=0)
    qmv_probe_p.add_argument("--no-mtp", action="store_true")
    qmv_probe_p.add_argument("--output")
    qmv_probe_p.set_defaults(func=_cmd_multi_qmv_probe)

    batch_eq_p = sub.add_parser(
        "batch-equivalence",
        help="Compare batched target forward against sequential one-token forward",
    )
    batch_eq_p.add_argument("--model", default=default_model)
    batch_eq_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    batch_eq_p.add_argument("--suffix-len", type=int, default=2)
    batch_eq_p.add_argument("--limit", type=int)
    batch_eq_p.add_argument("--expand-to", type=int)
    batch_eq_p.add_argument("--disable-thinking", action="store_true")
    batch_eq_p.add_argument("--tolerance", type=float, default=1e-3)
    batch_eq_p.add_argument("--output")
    batch_eq_p.set_defaults(func=_cmd_batch_equivalence)

    capture_eq_p = sub.add_parser(
        "capture-commit-equivalence",
        help="Verify captured GDN prefix commit against sequential AR state",
    )
    capture_eq_p.add_argument("--model", default=default_model)
    capture_eq_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    capture_eq_p.add_argument("--suffix-len", type=int, default=6)
    capture_eq_p.add_argument("--min-keep-tokens", type=int, default=1)
    capture_eq_p.add_argument("--limit", type=int)
    capture_eq_p.add_argument("--expand-to", type=int)
    capture_eq_p.add_argument("--disable-thinking", action="store_true")
    capture_eq_p.add_argument("--tolerance", type=float, default=1e-3)
    capture_eq_p.add_argument("--verify-backend", choices=["direct", "graphbank"], default="direct")
    capture_eq_p.add_argument(
        "--verify-core",
        choices=VERIFY_CORE_CHOICES,
        default="stock",
        help="Capture/VerifyCore backend for GDN state capture.",
    )
    capture_eq_p.add_argument("--output")
    capture_eq_p.set_defaults(func=_cmd_capture_commit_equivalence)

    mtp1_p = sub.add_parser("mtp1-greedy-gate", help="Compare MTP-1 greedy output against AR")
    mtp1_p.add_argument("--model", default=default_model)
    mtp1_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    mtp1_p.add_argument("--max-tokens", type=int, default=32)
    mtp1_p.add_argument("--seed", type=int, default=0)
    mtp1_p.add_argument("--limit", type=int)
    mtp1_p.add_argument("--expand-to", type=int)
    mtp1_p.add_argument("--disable-thinking", action="store_true")
    mtp1_p.add_argument(
        "--verify-strategy",
        choices=[
            "batched",
            "sequential",
            "capture",
            "capture_commit",
            "graphbank",
            "graphbank_capture_commit",
        ],
        default="batched",
    )
    mtp1_p.add_argument(
        "--verify-core",
        choices=VERIFY_CORE_CHOICES,
        default="stock",
        help="Capture/VerifyCore backend for capture-commit strategies.",
    )
    mtp1_p.add_argument("--draft-margin-threshold", type=float)
    mtp1_p.add_argument("--mtp-quant-bits", type=int)
    mtp1_p.add_argument("--mtp-quant-group-size", type=int, default=64)
    mtp1_p.add_argument("--mtp-quant-mode", default="affine")
    mtp1_p.add_argument("--output")
    mtp1_p.set_defaults(func=_cmd_mtp1_greedy_gate)

    sampler_p = sub.add_parser("mtp1-sampler-smoke", help="Run MTP-1 at non-greedy sampler settings")
    sampler_p.add_argument("--model", default=default_model)
    sampler_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    sampler_p.add_argument("--temperature", type=float, default=0.6)
    sampler_p.add_argument("--top-p", type=float, default=0.95)
    sampler_p.add_argument("--top-k", type=int, default=20)
    sampler_p.add_argument("--draft-temperature", type=float)
    sampler_p.add_argument("--draft-top-p", type=float)
    sampler_p.add_argument("--draft-top-k", type=int)
    sampler_p.add_argument("--max-tokens", type=int, default=96)
    sampler_p.add_argument("--seed", type=int, default=0)
    sampler_p.add_argument("--limit", type=int)
    sampler_p.add_argument("--disable-thinking", action="store_true")
    sampler_p.add_argument("--compare-ar", action="store_true")
    sampler_p.add_argument(
        "--verify-strategy",
        choices=[
            "batched",
            "sequential",
            "capture",
            "capture_commit",
            "graphbank",
            "graphbank_capture_commit",
        ],
        default="batched",
    )
    sampler_p.add_argument(
        "--verify-core",
        choices=VERIFY_CORE_CHOICES,
        default="stock",
        help="Capture/VerifyCore backend for capture-commit strategies.",
    )
    sampler_p.add_argument("--draft-margin-threshold", type=float)
    sampler_p.add_argument("--mtp-quant-bits", type=int)
    sampler_p.add_argument("--mtp-quant-group-size", type=int, default=64)
    sampler_p.add_argument("--mtp-quant-mode", default="affine")
    sampler_p.add_argument("--output")
    sampler_p.set_defaults(func=_cmd_mtp1_sampler_smoke)

    depth_p = sub.add_parser("mtp-depth-sweep", help="Run fixed-depth native MTP sweep")
    depth_p.add_argument("--model", default=default_model)
    depth_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    depth_p.add_argument("--depths", default="1,2,3")
    depth_p.add_argument("--temperature", type=float, default=0.6)
    depth_p.add_argument("--top-p", type=float, default=0.95)
    depth_p.add_argument("--top-k", type=int, default=20)
    depth_p.add_argument("--draft-temperature", type=float)
    depth_p.add_argument("--draft-top-p", type=float)
    depth_p.add_argument("--draft-top-k", type=int)
    depth_p.add_argument("--max-tokens", type=int, default=96)
    depth_p.add_argument("--seed", type=int, default=0)
    depth_p.add_argument("--limit", type=int)
    depth_p.add_argument("--disable-thinking", action="store_true")
    depth_p.add_argument("--compare-ar", action="store_true")
    depth_p.add_argument("--mtp-hidden-variant", default="post_norm")
    depth_p.add_argument("--mtp-cache-policy", choices=["persistent", "fresh"], default="persistent")
    depth_p.add_argument("--mtp-history-policy", choices=["cycle", "committed"], default="cycle")
    depth_p.add_argument("--draft-margin-threshold", type=float)
    depth_p.add_argument(
        "--min-speculative-depth",
        type=int,
        default=1,
        help=(
            "Number of draft depths to always attempt before margin gating can "
            "skip a candidate. Default 1 keeps D1 live and gates D2+."
        ),
    )
    depth_p.add_argument(
        "--verify-strategy",
        choices=["batched", "capture_commit", "graphbank", "graphbank_capture_commit"],
        default="batched",
    )
    depth_p.add_argument(
        "--verify-core",
        choices=VERIFY_CORE_CHOICES,
        default="stock",
        help="Capture/VerifyCore backend for capture-commit strategies.",
    )
    depth_p.add_argument(
        "--draft-core",
        choices=["stock", "device-d2"],
        default="stock",
        help=(
            "Experimental DraftCore backend. device-d2 compiles the greedy D2 "
            "native-MTP argmax chain for the exact draft-temperature 0 path."
        ),
    )
    depth_p.add_argument("--mtp-quant-bits", type=int)
    depth_p.add_argument("--mtp-quant-group-size", type=int, default=64)
    depth_p.add_argument("--mtp-quant-mode", default="affine")
    depth_p.add_argument("--mtp-adapter", type=Path)
    depth_p.add_argument("--mtp-corrector", type=Path)
    depth_p.add_argument("--mtp-corrector-blend", type=float)
    depth_p.add_argument(
        "--online-hidden-corrector-alpha",
        type=float,
        default=0.0,
        help=(
            "Experimental session-local EWMA residual applied to MTP hidden states "
            "before the next draft depth. Default 0 disables it."
        ),
    )
    depth_p.add_argument("--online-hidden-corrector-decay", type=float, default=0.8)
    depth_p.add_argument("--online-hidden-corrector-warmup", type=int, default=1)
    depth_p.add_argument("--online-hidden-corrector-max-feed-depth", type=int)
    depth_p.add_argument(
        "--online-hidden-corrector-key",
        choices=["global", "token"],
        default="global",
    )
    depth_p.add_argument(
        "--online-correction-cache",
        action="store_true",
        help=(
            "Experimental exact proposal override cache keyed by the local "
            "speculative prefix. Stores target top tokens after rejections."
        ),
    )
    depth_p.add_argument("--online-correction-cache-min-depth", type=int, default=1)
    depth_p.add_argument(
        "--online-correction-cache-key",
        choices=["local_prefix", "source_token", "primary_source"],
        default="local_prefix",
        help=(
            "Experimental correction-cache key policy. local_prefix preserves "
            "the original behavior; source_token and primary_source trade "
            "more hits for broader proposal reuse."
        ),
    )
    depth_p.add_argument(
        "--prompt-correction-cache",
        action="store_true",
        help=(
            "Experimental exact proposal cache seeded from prompt-local "
            "n-gram continuations. Uses the same one-hot q acceptance path."
        ),
    )
    depth_p.add_argument("--prompt-correction-cache-min-depth", type=int, default=2)
    depth_p.add_argument(
        "--adapter-ensemble-q",
        action="store_true",
        help=(
            "Experimental exact sparse-q proposal over base-vs-adapter MTP "
            "argmax tokens. Requires --mtp-adapter and greedy draft sampling."
        ),
    )
    depth_p.add_argument("--adapter-ensemble-epsilon", type=float, default=0.5)
    depth_p.add_argument("--adapter-ensemble-min-depth", type=int, default=2)
    depth_p.add_argument(
        "--mtp-topk-reranker-calib",
        type=Path,
        help=(
            "Experimental exact one-hot proposal selector fit from a hidden "
            "calibration shard. Diagnostic only."
        ),
    )
    depth_p.add_argument("--mtp-topk-reranker-depths", default="4")
    depth_p.add_argument("--mtp-topk-reranker-topk", type=int, default=32)
    depth_p.add_argument("--mtp-topk-reranker-q-weight", type=float, default=0.5)
    depth_p.add_argument("--mtp-topk-reranker-token-weight", type=float, default=1.0)
    depth_p.add_argument("--mtp-topk-reranker-rank-weight", type=float, default=0.0)
    depth_p.add_argument(
        "--mtp-topk-reranker-all-rows",
        action="store_true",
        help="Fit top-k proposal priors on all calibration rows, not just prefix-active rows.",
    )
    depth_p.add_argument("--output")
    depth_p.set_defaults(func=_cmd_mtp_depth_sweep)

    chain_p = sub.add_parser("mtp-chain-probe", help="Probe recursive MTP agreement by history/cache contract")
    chain_p.add_argument("--model", default=default_model)
    chain_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    chain_p.add_argument("--depth", type=int, default=5)
    chain_p.add_argument("--limit", type=int)
    chain_p.add_argument("--max-prompt-tokens", type=int, default=256)
    chain_p.add_argument("--windows", type=int, default=1)
    chain_p.add_argument("--stride", type=int, default=1)
    chain_p.add_argument("--top-ranks", default="1,2,4,8")
    chain_p.add_argument("--mtp-quant-bits", type=int)
    chain_p.add_argument("--mtp-quant-group-size", type=int, default=64)
    chain_p.add_argument("--mtp-quant-mode", default="affine")
    chain_p.add_argument("--raw-prompts", action="store_true")
    chain_p.add_argument("--disable-thinking", action="store_true")
    chain_p.add_argument("--base-hidden-variants", default="post_norm")
    chain_p.add_argument("--mtp-hidden-variants", default="post_norm,pre_norm,fc")
    chain_p.add_argument("--cache-policies", default="fresh,persistent")
    chain_p.add_argument("--concat-orders", default="embedding_hidden")
    chain_p.add_argument(
        "--mtp-position-modes",
        default="local",
        help=(
            "MTP RoPE position contract to probe. 'local' preserves current "
            "MLX cache-offset behavior; 'absolute' applies prompt/window "
            "absolute positions before MTP cache update."
        ),
    )
    chain_p.add_argument(
        "--history-modes",
        default="recursive,target_forced,target_token_recursive_hidden",
    )
    chain_p.add_argument("--anchors", default="prompt_boundary,after_one_target")
    chain_p.add_argument("--output")
    chain_p.set_defaults(func=_cmd_mtp_chain_probe)

    tree_probe_p = sub.add_parser("mtp-tree-probe", help="Probe native-MTP tree coverage without target verify")
    tree_probe_p.add_argument("--model", default=default_model)
    tree_probe_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    tree_probe_p.add_argument("--depth", type=int, default=5)
    tree_probe_p.add_argument("--budgets", default="1,2,4,8,16")
    tree_probe_p.add_argument("--branch-factor", type=int, default=4)
    tree_probe_p.add_argument("--limit", type=int)
    tree_probe_p.add_argument("--windows", type=int, default=32)
    tree_probe_p.add_argument("--stride", type=int, default=1)
    tree_probe_p.add_argument("--max-prompt-tokens", type=int, default=256)
    tree_probe_p.add_argument("--raw-prompts", action="store_true")
    tree_probe_p.add_argument("--disable-thinking", action="store_true")
    tree_probe_p.add_argument("--mtp-quant-bits", type=int)
    tree_probe_p.add_argument("--mtp-quant-group-size", type=int, default=64)
    tree_probe_p.add_argument("--mtp-quant-mode", default="affine")
    tree_probe_p.add_argument("--base-hidden-variant", choices=["post_norm", "pre_norm"], default="post_norm")
    tree_probe_p.add_argument("--mtp-hidden-variant", default="pre_norm")
    tree_probe_p.add_argument(
        "--mtp-cache-policy",
        choices=["fresh", "persistent_path"],
        default="fresh",
        help=(
            "MTP cache contract for branch expansion. 'persistent_path' "
            "replays each branch path into one MTP cache before expanding it."
        ),
    )
    tree_probe_p.add_argument("--anchor", choices=["prompt_boundary", "after_one_target"], default="prompt_boundary")
    tree_probe_p.add_argument("--output")
    tree_probe_p.set_defaults(func=_cmd_mtp_tree_probe)

    grid_p = sub.add_parser("mtp-depth-grid", help="Run a sequential fixed-depth policy grid")
    grid_p.add_argument("--model", default=default_model)
    grid_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    grid_p.add_argument("--depth", type=int, default=5)
    grid_p.add_argument("--thresholds", default="0.5,0.75,1.0,1.25,1.5,2.0")
    grid_p.add_argument("--min-depths", default="0,1,2,3")
    grid_p.add_argument("--temperature", type=float, default=0.6)
    grid_p.add_argument("--top-p", type=float, default=0.95)
    grid_p.add_argument("--top-k", type=int, default=20)
    grid_p.add_argument("--draft-temperature", type=float, default=0.0)
    grid_p.add_argument("--draft-top-p", type=float)
    grid_p.add_argument("--draft-top-k", type=int)
    grid_p.add_argument("--max-tokens", type=int, default=96)
    grid_p.add_argument("--seed", type=int, default=0)
    grid_p.add_argument("--limit", type=int)
    grid_p.add_argument("--disable-thinking", action="store_true")
    grid_p.add_argument("--compare-ar", action="store_true")
    grid_p.add_argument("--mtp-hidden-variant", default="pre_norm")
    grid_p.add_argument("--mtp-cache-policy", choices=["persistent", "fresh"], default="fresh")
    grid_p.add_argument("--mtp-history-policy", choices=["cycle", "committed"], default="cycle")
    grid_p.add_argument(
        "--verify-strategy",
        choices=["batched", "capture_commit", "graphbank", "graphbank_capture_commit"],
        default="batched",
    )
    grid_p.add_argument("--mtp-corrector", type=Path)
    grid_p.add_argument("--mtp-corrector-blend", type=float)
    grid_p.add_argument("--store-events", action="store_true")
    grid_p.add_argument("--output")
    grid_p.set_defaults(func=_cmd_mtp_depth_grid)

    adaptive_p = sub.add_parser("mtp-adaptive", help="Run adaptive-depth native MTP")
    adaptive_p.add_argument("--model", default=default_model)
    adaptive_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    adaptive_p.add_argument("--max-depth", type=int, default=5)
    adaptive_p.add_argument("--min-depth", type=int, default=1)
    adaptive_p.add_argument("--start-depth", type=int, default=1)
    adaptive_p.add_argument("--increase-after", type=int, default=4)
    adaptive_p.add_argument("--decrease-after", type=int, default=1)
    adaptive_p.add_argument("--policy", choices=["streak", "expected_value"], default="streak")
    adaptive_p.add_argument("--ev-base-depth", type=int, default=2)
    adaptive_p.add_argument("--ev-accept-priors", type=_comma_floats, default=(0.92, 0.64, 0.32))
    adaptive_p.add_argument("--ev-draft-cost-s", type=float, default=0.0048)
    adaptive_p.add_argument("--ev-extra-verify-cost-s", type=float, default=0.0060)
    adaptive_p.add_argument("--ev-baseline-tok-s", type=float, default=40.0)
    adaptive_p.add_argument("--ev-safety-margin", type=float, default=0.10)
    adaptive_p.add_argument("--ev-margin-center", type=float, default=1.0)
    adaptive_p.add_argument("--ev-margin-scale", type=float, default=2.0)
    adaptive_p.add_argument("--ev-confidence-weight", type=float, default=0.35)
    adaptive_p.add_argument("--ev-min-extra-accept-probability", type=float, default=0.18)
    adaptive_p.add_argument("--temperature", type=float, default=0.6)
    adaptive_p.add_argument("--top-p", type=float, default=0.95)
    adaptive_p.add_argument("--top-k", type=int, default=20)
    adaptive_p.add_argument("--draft-temperature", type=float)
    adaptive_p.add_argument("--draft-top-p", type=float)
    adaptive_p.add_argument("--draft-top-k", type=int)
    adaptive_p.add_argument("--max-tokens", type=int, default=96)
    adaptive_p.add_argument("--seed", type=int, default=0)
    adaptive_p.add_argument("--limit", type=int)
    adaptive_p.add_argument("--disable-thinking", action="store_true")
    adaptive_p.add_argument("--compare-ar", action="store_true")
    adaptive_p.add_argument("--mtp-hidden-variant", default="post_norm")
    adaptive_p.add_argument("--mtp-cache-policy", choices=["persistent", "fresh"], default="persistent")
    adaptive_p.add_argument("--mtp-history-policy", choices=["cycle", "committed"], default="cycle")
    adaptive_p.add_argument(
        "--verify-strategy",
        choices=["batched", "capture_commit", "graphbank", "graphbank_capture_commit"],
        default="batched",
    )
    adaptive_p.add_argument(
        "--verify-core",
        choices=VERIFY_CORE_CHOICES,
        default="stock",
    )
    adaptive_p.add_argument("--mtp-quant-bits", type=int)
    adaptive_p.add_argument("--mtp-quant-group-size", type=int, default=64)
    adaptive_p.add_argument("--mtp-quant-mode", default="affine")
    adaptive_p.add_argument("--output")
    adaptive_p.set_defaults(func=_cmd_mtp_adaptive)

    dflash_p = sub.add_parser("dflash-mlx-baseline", help="Run official DFlash MLX baseline")
    dflash_p.add_argument("--model", default=default_model)
    dflash_p.add_argument("--draft-model", default="z-lab/Qwen3.6-27B-DFlash")
    dflash_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    dflash_p.add_argument("--dflash-source", default="REFERENCES:TOOLS/dflash")
    dflash_p.add_argument("--temperature", type=float, default=0.6)
    dflash_p.add_argument("--top-p", type=float, default=0.95)
    dflash_p.add_argument("--top-k", type=int, default=20)
    dflash_p.add_argument("--max-tokens", type=int, default=96)
    dflash_p.add_argument("--block-size", type=int)
    dflash_p.add_argument("--seed", type=int, default=0)
    dflash_p.add_argument("--limit", type=int)
    dflash_p.add_argument("--disable-thinking", action="store_true")
    dflash_p.add_argument("--draft-sliding-window-size", type=int)
    dflash_p.add_argument("--output")
    dflash_p.set_defaults(func=_cmd_dflash_mlx_baseline)

    ddtree_p = sub.add_parser("ddtree-mlx-baseline", help="Run DDTree MLX baseline")
    ddtree_p.add_argument("--model", default=default_model)
    ddtree_p.add_argument("--draft-model", default="z-lab/Qwen3.6-27B-DFlash")
    ddtree_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    ddtree_p.add_argument("--ddtree-source", default="REFERENCES:TOOLS/ddtree-mlx")
    ddtree_p.add_argument("--temperature", type=float, default=0.6)
    ddtree_p.add_argument("--top-p", type=float, default=0.95)
    ddtree_p.add_argument("--top-k", type=int, default=20)
    ddtree_p.add_argument("--max-tokens", type=int, default=96)
    ddtree_p.add_argument("--tree-budget", type=int, default=4)
    ddtree_p.add_argument("--limit", type=int)
    ddtree_p.add_argument("--disable-thinking", action="store_true")
    ddtree_p.add_argument("--output")
    ddtree_p.set_defaults(func=_cmd_ddtree_mlx_baseline)

    truth_p = sub.add_parser("truth-report", help="Run the Phase 0 evidence-grade MTPLX truth harness")
    truth_p.add_argument("--model", default=default_model)
    truth_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    truth_p.add_argument(
        "--modes",
        default=",".join(DEFAULT_TRUTH_MODES),
        help="Comma-separated truth modes to run",
    )
    truth_p.add_argument("--temperature", type=float, default=0.6)
    truth_p.add_argument("--top-p", type=float, default=0.95)
    truth_p.add_argument("--top-k", type=int, default=20)
    truth_p.add_argument("--draft-temperature", type=float, default=0.0)
    truth_p.add_argument("--draft-top-p", type=float)
    truth_p.add_argument("--draft-top-k", type=int, default=1)
    truth_p.add_argument("--max-tokens", type=int, default=96)
    truth_p.add_argument("--seed", type=int, default=0)
    truth_p.add_argument("--limit", type=int, default=1)
    truth_p.add_argument("--disable-thinking", action="store_true")
    truth_p.add_argument("--mtp-hidden-variant", default="pre_norm")
    truth_p.add_argument("--mtp-cache-policy", choices=["persistent", "fresh"], default="persistent")
    truth_p.add_argument("--mtp-history-policy", choices=["cycle", "committed"], default="cycle")
    truth_p.add_argument("--c3-corrector", type=Path, default=DEFAULT_C3_CORRECTOR)
    truth_p.add_argument("--c3-blend", type=float, default=0.15)
    truth_p.add_argument("--project-root", default=".")
    truth_p.add_argument("--min-free-gib", type=float, default=120.0)
    truth_p.add_argument("--cpu-threshold", type=float, default=25.0)
    truth_p.add_argument("--output-dir", default="outputs/reports/truth")
    truth_p.add_argument("--output-json")
    truth_p.add_argument("--output-md")
    truth_p.add_argument("--strict-preflight", action="store_true")
    truth_p.add_argument("--fail-fast", action="store_true")
    truth_p.set_defaults(func=_cmd_truth_report)

    session_p = sub.add_parser("session-bank", help="Benchmark exact warm-prefix SessionBank prefill reuse")
    session_p.add_argument("--model", default=default_model)
    session_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    session_p.add_argument("--prompt-index", type=int, default=0)
    session_p.add_argument("--suffix-text", default="\n\n# Follow-up request:\nRefactor this into a cleaner implementation.\n")
    session_p.add_argument("--max-prompt-tokens", type=int, default=512)
    session_p.add_argument("--raw-prompts", action="store_true")
    session_p.add_argument("--disable-thinking", action="store_true")
    session_p.add_argument("--max-entries", type=int, default=4)
    session_p.add_argument("--tolerance", type=float, default=1e-3)
    session_p.add_argument("--restore-mode", choices=["clone", "reference"], default="clone")
    session_p.add_argument("--output")
    session_p.set_defaults(func=_cmd_session_bank)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    from .config import apply_user_config

    apply_user_config(args)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
