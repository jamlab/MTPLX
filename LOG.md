# MTPLX Release Log

## 2026-05-17 05:21 BST - Release Readiness Evidence For Issues #67/#68/#70/#71

Scope:

```text
worktree=/Users/youssof/Documents/MTPLX-release/mtplx
branch=main
base_head=fc0efb2 Eliminate stale relative model defaults
release_issues=#67 default installed model; #68 honest speed metrics; #70 Claude Code tools; #71 tune stale paths/failure reporting
local_fix=mtplx start dry-run now stores the resolved verified default on args.model before building OpenWebUI handoff payloads
```

Validation:

```text
source_neutral_dry_runs=from /tmp with fresh HOME/MTPLX_CONFIG/cache, `start --dry-run --json`, `tune --dry-run --json`, and `bench tune --dry-run --json --no-telemetry` resolve to Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed or the verified local absolute artifact, never stale models/Qwen3.6-27B-MTPLX-Optimized-Speed
start_handoff_fix=`mtplx start --dry-run --json` now emits openwebui.model_id=mtplx-qwen36-27b-optimized-speed and server_command contains the resolved model, not `--model None`
fresh_package=rebuilt /tmp/mtplx-release-readiness-final-pkg/dist/mtplx-0.3.6.{tar.gz,py3-none-any.whl}, installed wheel into /tmp/mtplx-release-readiness-final-pkg/venv, and passed `mtplx --version`, `mtplx help`, `mtplx start --dry-run --json`, `mtplx tune --dry-run --json`, `mtplx bench tune --dry-run --json --no-telemetry`, `python -m mtplx.server.openai --help`, and `mtplx connect claude-code --json`
real_server_67=source quickstart from /tmp on port 18901 with /Users/youssof/Documents/MTPLX/models/Qwen3.6-27B-MTPLX-Optimized-Speed reached Runtime contract verified, /health generation_mode=mtp mtp_enabled=true runtime_mode=Sustained MTP, and an OpenAI chat completion returned separate tok_s/decode_tok_s and end_to_end_tok_s fields
real_tune_71_68=max-fan `bench tune --model /Users/youssof/Documents/MTPLX/models/Qwen3.6-27B-MTPLX-Optimized-Speed --depths 1,2,3 --max-tokens 192 --limit 1 --seed 0 --no-save --retune --no-telemetry` wrote AR/D1/D2/D3 artifacts under /tmp/mtplx-release-readiness-tune-max/tune-20260517-050926; AR=22.58 decode tok/s, D1=35.07, D2=38.31, D3=39.73 best 1.76x; output explicitly said speed shown is decode tok/s and prefill is tracked separately
bad_model_71=final bad-model rerun under /tmp/mtplx-release-readiness-bad-tune-final/tune-20260517-051932 reports AR and D1 failed with `model failed MTP primary gate: Model has no MTP head. MTPLX requires an MTP-equipped model.`, not a missing-artifact placeholder
mlx_lm_comparison_68=same local Optimized Speed family, 181-token prompt and 192-token generation in mlx_lm reported prefill 281.57 tokens/s and generation 31.11 tokens/s; MTPLX AR in the max Tune run was lower at 22.58 decode tok/s while MTPLX D3 was higher at 39.73 decode tok/s, so the discrepancy is a real runtime/mode comparison issue rather than a prefill-inclusive headline
claude_code_70=count_tokens with a Bash tool returned input_tokens=351; headless Claude Code v2.1.143 against MTPLX on port 18902 created a Snake game over 6 turns with 5 tool_use and 5 tool_result blocks, ran Bash, produced 36/36 passing tests, and transcript parsing found no assistant-text leaks for Bash(, Skill(, <tool_call>, or <function=
claude_code_interactive_70=interactive PTY Claude Code against the same MTPLX server visibly executed Bash(./test.sh), displayed the test output, and summarized all 36 tests passing
focused_tests=/Users/youssof/Documents/MTPLX-release/mtplx/.venv/bin/python -m pytest tests/test_public_cli.py tests/test_server_openai.py tests/test_openai_bridge.py tests/test_generation_sustained.py tests/test_hf_loader.py tests/test_default_models.py tests/test_config.py -q -> pass, one existing skip, deprecation warnings only
```

Release status:

```text
ready_for_commit_after_docs=true
not_yet_pushed_at_entry_time=true
required_after_push=wait for GitHub Actions on pushed main SHA before closing issues
```

## 2026-05-15 21:01 BST - Post-v0.3.6 Model Reference Hotfixes, No Public Release

Scope:

```text
worktree=/Users/youssof/Documents/MTPLX-release/mtplx-v0.3.6
branch=codex/model-alias-resolution-fix
base_release=v0.3.6 public release, tag/main commit 1d6a7b7
hotfix_commits=7aa8624 Fix public model alias resolution; 16155bf Fix direct CLI default model path
user_gate=no new GitHub release, PyPI publish, or Homebrew update yet; local global venv hotpatched only for user testing
trigger_1=external screenshot showed `mtplx quickstart --max --model Qwen3.6-27B-MTPLX-Optimized-Quality` failing as no-MTP
trigger_2=GitHub issue #67 reported `mtplx quickstart --profile sustained` failing after `mtplx start` installed and ran the default model successfully
```

Release-artifact investigation:

```text
verified_actual_release=downloaded PyPI mtplx==0.3.6 sdist and inspected v0.3.6/main state before final diagnosis
public_0_3_6_repro_1=bare Quality artifact name was treated as a local folder; missing/nonexistent local path then produced a false no-MTP error
public_0_3_6_repro_2=direct `mtplx quickstart --profile sustained` defaulted to `models/Qwen3.6-27B-MTPLX-Optimized-Speed`, not the installed Hugging Face cache
public_0_3_6_repro_3=standalone `mtplx tune --dry-run --json` also emitted candidate commands with `--model models/Qwen3.6-27B-MTPLX-Optimized-Speed`
issue_67_doctor_context=reporter cache was valid at `/Users/matt/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed` with config/tokenizer/index/mtp/runtime contract present
correct_diagnosis=the model choice was right, but direct CLI/tune default references were stale repo-relative paths; the no-MTP message was a false diagnosis from inspecting the wrong path
```

Fix:

```text
public_aliases=known public artifact names now canonicalize to `Youssofal/...` repo ids for Speed, Speed-FP16, legacy Optimized, and Quality
missing_local_path=resolve_model_path now rejects unavailable local paths instead of returning a nonexistent folder to the MTP compatibility gate
direct_cli_default=parser default for quickstart/serve/tune now uses `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed` instead of `models/Qwen3.6-27B-MTPLX-Optimized-Speed`
standalone_tune_default=DEFAULT_CHAMPION now points at the public verified model id, so `mtplx tune` and `mtplx-tune` do not fall back to the old repo-local model path
legacy_config=stale config value `models/Qwen3.6-27B-MTPLX-Optimized-Speed` is treated as a legacy default and ignored, preserving real custom model paths
```

Validation:

```text
uv run --extra dev python -m pytest tests/test_hf_loader.py tests/test_public_cli.py::test_quickstart_public_quality_alias_missing_cache_is_not_no_mtp tests/test_public_cli.py::test_start_missing_model_suggests_download -q -> pass
uv run --extra dev python -m pytest tests/test_public_cli.py::test_product_helper_commands_parse tests/test_public_cli.py::test_quickstart_default_missing_cache_is_not_legacy_models_path tests/test_public_cli.py::test_tune_default_dry_run_is_not_legacy_models_path tests/test_public_cli.py::test_quickstart_public_quality_alias_missing_cache_is_not_no_mtp tests/test_config.py::test_apply_user_config_ignores_legacy_optimized_speed_default -q -> pass
uv run --extra dev python -m pytest tests/test_public_cli.py tests/test_config.py tests/test_default_models.py tests/test_hf_loader.py tests/test_no_mlx_imports.py -q -> pass
python3 -m compileall -q mtplx tests -> pass
uv run --extra dev python -m ruff check mtplx/cli.py mtplx/config.py mtplx/commands/public.py mtplx/artifacts.py mtplx/hf_loader.py tests/test_public_cli.py tests/test_config.py tests/test_hf_loader.py -> pass
git diff --check -> pass
global_hotpatch=/opt/homebrew/var/mtplx/venv-0.3.6/bin/python -m pip install --force-reinstall --no-deps . -> pass
global_quickstart_probe=MTPLX_CONFIG=/tmp/mtplx-issue67-global-no-config.toml mtplx quickstart --profile sustained --cache-dir /tmp/mtplx-empty-cache-issue67-global --yes --warmup-tokens 0 --port 18113 -> loads/checks `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`, exits with honest not-cached message, no no-MTP claim
global_tune_probe=MTPLX_CONFIG=/tmp/mtplx-issue67-global-no-config.toml mtplx tune --dry-run --json --cache-dir /tmp/mtplx-empty-cache-issue67-global -> candidate command uses `--model Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`
```

Public status:

```text
not_published=PyPI/Homebrew/GitHub Releases remain public v0.3.6 until user approves a hotfix release
known_release_risk=public v0.3.6 still has issue #67 behavior until a new release is cut
next_release_candidate=hotfix should be versioned as a new patch release, not silently mutate v0.3.6
```

## 2026-05-14 23:47 BST - v0.3.6 Bench Tune Hardware Telemetry

Scope:

```text
worktree=/Users/youssof/Documents/MTPLX-release/mtplx-v0.3.6
branch=codex/release-v0.3.6
trigger=user reported mtplx bench tune was not diagnostic enough for M3/M4/M5 chip behavior; wanted MX Power Gadget-style power, temperature, frequency, and utilization during AR/D1/D2/D3 tests
public_release_done=false
```

Implementation:

```text
bench_tune_telemetry=enabled only for mtplx bench tune, not normal mtplx tune
candidate_scope=each isolated candidate subprocess gets its own telemetry sampler and row-level JSON summary
powermetrics_fields=package/cpu/gpu/ane watts, P/M/GPU GHz, P/M/GPU utilization, thermal pressure
thermalforge_fields=fan RPM plus CPU/core and GPU temperature summaries with latest raw sensor values
human_output=per-candidate progress and verbose results now include power/frequency/temp/utilization/fan samples
fallback_policy=if powermetrics is unavailable or sudo -n is not configured, bench tune still reports thermalforge temps/fans and records the reason
```

Validation:

```text
python3 -m py_compile mtplx/commands/public.py tests/test_public_cli.py -> pass
uv run --extra dev python -m ruff check mtplx/commands/public.py tests/test_public_cli.py -> pass
uv run --extra dev python -m pytest tests/test_public_cli.py -q -> pass
uv run --extra dev python -m pytest tests/test_default_models.py -q -> pass
uv run --extra dev python -m pytest tests/test_onboarding.py::test_quickstart_applies_saved_tuned_depth tests/test_onboarding.py::test_quickstart_tuning_prompt_can_save_and_apply tests/test_onboarding.py::test_quickstart_explicit_depth_never_uses_tuning -q -> pass
git diff --check -> pass
non_model_probe=thermalforge and powermetrics parsers returned watts/GHz/C/utilization/fan fields on this Mac
real_short_bench_tune=MTPLX_TUNE_STATE=/tmp/mtplx-bench-tune-telemetry-state.json uv run --extra dev python -m mtplx.cli bench tune --model /Users/youssof/.mtplx/hf-upload/Qwen3.6-27B-MTPLX-Optimized --limit 1 --max-tokens 16 --depths 1 --retune --no-save --run-id telemetry-smoke-20260514 --output-dir /tmp/mtplx-bench-tune-telemetry --yes
real_short_result=AR 16.98 tok/s; D1 24.30 tok/s; output printed package/cpu/ane/gpu watts, P/M/GPU GHz, core/GPU temps, P/M/GPU utilization, fan RPM, and sample counts for both candidates
thermal_restore=post-run thermalforge status fan_modes auto/auto
uv run --extra dev python -m build -> pass; rebuilt dist/mtplx-0.3.6.tar.gz and dist/mtplx-0.3.6-py3-none-any.whl
uv run --extra dev python -m twine check dist/* -> pass
/opt/homebrew/opt/python@3.14/bin/python3.14 -m pip install --force-reinstall --no-deps dist/mtplx-0.3.6-py3-none-any.whl -> pass
global_mtplx_version=mtplx 0.3.6
global_real_short_bench_tune=MTPLX_TUNE_STATE=/tmp/mtplx-bench-tune-global-telemetry-state.json mtplx bench tune --model /Users/youssof/.mtplx/hf-upload/Qwen3.6-27B-MTPLX-Optimized --limit 1 --max-tokens 8 --depths 1 --retune --no-save --run-id global-telemetry-smoke-20260514 --output-dir /tmp/mtplx-bench-tune-global-telemetry --yes
global_real_short_result=AR 13.42 tok/s; D1 17.41 tok/s; global PATH output printed power/frequency/temp/utilization/fan telemetry for both candidates
global_thermal_restore=post-run thermalforge status fan_modes auto/auto
```

## 2026-05-14 23:31 BST - v0.3.6 Onboarding Regression Fix Before User Test

Scope:

```text
worktree=/Users/youssof/Documents/MTPLX-release/mtplx-v0.3.6
branch=codex/release-v0.3.6
trigger=user-tested global mtplx start wizard and found the verified default labeled BF16, prompting a download despite the local Optimized Speed artifact, and missing the CLI Tune offer
public_release_done=false
```

Root cause:

```text
default_model_metadata=stale v0.3.0 BF16 wording still labeled the current Optimized Speed default as BF16 on M3/M4/M5-class Macs
default_model_resolution=select_default_model returned the Hugging Face repo id even when /Users/youssof/.mtplx/hf-upload/Qwen3.6-27B-MTPLX-Optimized was already installed and complete
cli_tune_prompt=_quickstart_apply_tuned_depth returned immediately for target=terminal, so first-run Tune was offered only for Web UI
```

Fix:

```text
speed_default_label=Q4 target with Q4 MTP sidecar
local_speed_preference=prefer complete local Optimized Speed artifacts from ~/.mtplx/hf-upload, ~/.mtplx/models, and repo-local models before the HF mirror
legacy_bf16_env_alias=MTPLX_DEFAULT_MODEL_VARIANT=bf16 remains accepted but maps to optimized speed and no longer prints BF16 to users
cli_tune_prompt=first-run Tune offer now applies to CLI/terminal as well as Web UI
tests_added_or_updated=default speed local preference, no BF16 label, legacy alias behavior, verified legacy local path, terminal Tune application
```

Validation:

```text
python3 -m py_compile mtplx/default_models.py mtplx/commands/public.py mtplx/ui/onboarding.py tests/test_default_models.py tests/test_onboarding.py -> pass
uv run --extra dev python -m ruff check mtplx/default_models.py mtplx/commands/public.py mtplx/ui/onboarding.py tests/test_default_models.py tests/test_onboarding.py -> pass
uv run --extra dev python -m pytest tests/test_default_models.py tests/test_onboarding.py -q -> pass
uv run --extra dev python -m build -> pass; rebuilt dist/mtplx-0.3.6.tar.gz and dist/mtplx-0.3.6-py3-none-any.whl
uv run --extra dev python -m twine check dist/* -> pass
/opt/homebrew/opt/python@3.14/bin/python3.14 -m pip install --force-reinstall --no-deps dist/mtplx-0.3.6-py3-none-any.whl -> pass
mtplx --version -> mtplx 0.3.6
mtplx start cli --dry-run --json -> model=/Users/youssof/.mtplx/hf-upload/Qwen3.6-27B-MTPLX-Optimized; download_if_missing=false; precision=Q4 target with Q4 MTP sidecar
global pseudo-tty mtplx start --fresh -> Step 1 shows Q4 target + Q4 MTP, no BF16, no missing/download prompt, CLI selected, Step 4 Tune prompt appears before model chat starts
```

## 2026-05-14 23:55 BST - v0.3.6 Tune UX And Served Model-Id Regression Fix

Scope:

```text
worktree=/Users/youssof/Documents/MTPLX-release/mtplx-v0.3.6
branch=codex/release-v0.3.6
trigger=user-tested mtplx start from home directory; Tune appeared to hang silently after fan ramp, then printed n/a for every candidate and "tuning did not finish"; server banner mislabeled the installed speed artifact as mtplx-qwen36-27b-optimized-quality
public_release_done=false
```

Root cause:

```text
tune_artifacts=_cmd_tune built output_root as a relative path from the caller cwd, but _run_tune_candidates launched subprocesses with cwd=repo_root; child candidate outputs were therefore addressed relative to a different directory, so the parent could not see the artifacts
tune_ux=_run_tune_candidates captured child stdout and printed no per-candidate progress, making isolated model-load/timing work look frozen after fan ramp
tune_error_copy=all-missing candidate artifacts were summarized as "No MTP depth beat AR" instead of an actual Tune failure with log paths
served_model_id=_public_model_id_from_metadata treated any 8-bit layer inside the quantization config as Optimized Quality; the installed speed artifact is mixed Q4/Q8, so the server banner incorrectly said optimized-quality
```

Fix:

```text
tune_paths=normalize Tune output_dir/output/candidate paths to absolute user paths before spawning candidate subprocesses
tune_progress=print artifacts path and AR/D1/D2/D3 per-candidate start/finish/fail lines to stderr during Tune
tune_errors=report candidate artifact failures as Tune failures with log paths; quickstart now says "tuning failed; using default depth"
served_model_id=classify mixed Q4/Q8 metadata as Optimized Speed and all-INT8/Flat8 metadata as Optimized Quality; verified_on.model speed metadata also maps to Optimized Speed
```

Validation:

```text
python3 -m py_compile mtplx/default_models.py mtplx/commands/public.py tests/test_default_models.py tests/test_public_cli.py -> pass
uv run --extra dev python -m ruff check mtplx/default_models.py mtplx/commands/public.py tests/test_default_models.py tests/test_public_cli.py -> pass
uv run --extra dev python -m pytest tests/test_default_models.py tests/test_public_cli.py::test_tune_candidate_outputs_are_absolute_from_non_repo_cwd tests/test_public_cli.py::test_tune_human_reports_candidate_errors_instead_of_false_no_win tests/test_public_cli.py::test_serve_uses_quality_public_model_id_for_quality_local_path tests/test_public_cli.py::test_serve_uses_legacy_public_model_id_for_legacy_optimized_local_path -q -> pass
uv run --extra dev python -m pytest tests/test_onboarding.py tests/test_public_cli.py -q -> pass
uv run python -m mtplx.cli start web --dry-run --json --model /Users/youssof/.mtplx/hf-upload/Qwen3.6-27B-MTPLX-Optimized --yes -> openwebui.model_id=mtplx-qwen36-27b-optimized-speed
uv run --extra dev python -m build -> pass; rebuilt dist/mtplx-0.3.6.tar.gz and dist/mtplx-0.3.6-py3-none-any.whl
uv run --extra dev python -m twine check dist/* -> pass
/opt/homebrew/opt/python@3.14/bin/python3.14 -m pip install --force-reinstall --no-deps dist/mtplx-0.3.6-py3-none-any.whl -> pass
from_home_global_dry_run=mtplx tune --model /Users/youssof/.mtplx/hf-upload/Qwen3.6-27B-MTPLX-Optimized --dry-run --run-id smoke-path --output-dir /tmp/mtplx-tune-from-home-check -> candidate --_candidate-output paths are absolute under /tmp/mtplx-tune-from-home-check/smoke-path
from_home_real_short_tune=MTPLX_TUNE_STATE=/tmp/mtplx-tune-home-real-state.json mtplx tune --model /Users/youssof/.mtplx/hf-upload/Qwen3.6-27B-MTPLX-Optimized --limit 1 --max-tokens 32 --depths 1 --seed 0 --retune --no-save --run-id home-real-smoke --output-dir /tmp/mtplx-tune-home-real --yes -> AR=20.25 tok/s, D1=30.31 tok/s, best=D1 1.50x, artifacts written under /tmp/mtplx-tune-home-real/home-real-smoke, no n/a rows
thermalforge post-run status -> fans mode=auto actual=0 target=0
```

## 2026-05-15 00:10 BST - v0.3.6 Tune Results Copy Fix

Scope:

```text
worktree=/Users/youssof/Documents/MTPLX-release/mtplx-v0.3.6
branch=codex/release-v0.3.6
trigger=user noted Tune printed "Close heavy apps for cleaner results" after the benchmark had already completed
public_release_done=false
```

Fix:

```text
tune_pre_run_copy=print close-heavy-apps/fans-may-get-loud warnings before fan ramp and candidate measurements start
tune_results_copy=final Tune output now shows results/artifact path only; no stale pre-run advice after measurements are over
```

Validation:

```text
python3 -m py_compile mtplx/commands/public.py tests/test_public_cli.py -> pass
uv run --extra dev python -m ruff check mtplx/commands/public.py tests/test_public_cli.py -> pass
uv run --extra dev python -m pytest tests/test_public_cli.py::test_tune_human_reports_candidate_errors_instead_of_false_no_win tests/test_public_cli.py::test_tune_human_results_do_not_give_pre_run_advice_afterward tests/test_public_cli.py::test_tune_candidate_outputs_are_absolute_from_non_repo_cwd -q -> pass
uv run --extra dev python -m build -> pass; rebuilt dist/mtplx-0.3.6.tar.gz and dist/mtplx-0.3.6-py3-none-any.whl
uv run --extra dev python -m twine check dist/* -> pass
/opt/homebrew/opt/python@3.14/bin/python3.14 -m pip install --force-reinstall --no-deps dist/mtplx-0.3.6-py3-none-any.whl -> pass
global packaged Tune result-render smoke -> final output says "Results written to ..." and contains no "Close heavy apps" or "Fans may get loud"
```

## 2026-05-14 22:05 BST - v0.3.6 Release Candidate Assembly

Scope:

```text
branch=codex/release-v0.3.6
base=origin/main 253e7ebd50ddc79e684a775ab85402c43b4702e2
target_version=0.3.6
release_contract=protect decode TPS, prefill/TTFT, memory, and CLI UX together
```

Integrated slices:

```text
memory=dynamic initial paged-KV new-token reserve for huge max_tokens; anonymous no-reuse sessions do not keep live full-capacity cache refs; high-RAM MLX Metal caps remain a safety rail
opencode=tool-result turns reuse stable SessionBank prefixes; Qwen XML tool-call arguments are emitted as schema-typed OpenAI tool-call JSON
tune=mtplx tune, mtplx-tune, bench tune; AR remains the 1.00x baseline; D1/D2/D3 run in isolated subprocesses; losing depths are not saved
cli_ux=Tune added to public help/parser, first-run Web UI can offer tuning, start/pi/opencode/swival surfaces preserved
```

Focused validation completed before full release QA:

```text
python3 -m py_compile mtplx/cli.py mtplx/commands/public.py mtplx/ui/onboarding.py mtplx/config.py mtplx/thermal.py mtplx/benchmarks/runners/mtp_depth_sweep.py tests/test_public_cli.py tests/test_onboarding.py tests/test_config.py tests/test_thermal.py
uv run --extra dev python -m ruff check mtplx/cli.py mtplx/commands/public.py mtplx/ui/onboarding.py mtplx/config.py mtplx/thermal.py mtplx/benchmarks/runners/mtp_depth_sweep.py tests/test_public_cli.py tests/test_onboarding.py tests/test_config.py tests/test_thermal.py
uv run --extra dev python -m pytest tests/test_config.py tests/test_thermal.py tests/test_public_cli.py tests/test_onboarding.py -q
uv run python -m mtplx.cli tune --model models/not-loaded-in-dry-run --dry-run --yes
uv run python -m mtplx.cli bench tune --model models/not-loaded-in-dry-run --dry-run --json --yes
```

Open items before publish:

```text
full_static_package_cli_qa=pending
real_tune_gate=pending
aime20_memory_gate=pending
coding_control_gate=pending
opencode_cli_gate=pending
m3_ultra_512gb_target_gate=pending
pypi_homebrew_publish=pending
```

## 2026-05-14 22:44 BST - v0.3.6 Candidate QA, No Public Release

Scope:

```text
worktree=/Users/youssof/Documents/MTPLX-release/mtplx-v0.3.6
branch=codex/release-v0.3.6
base_commit=253e7ebd50ddc79e684a775ab85402c43b4702e2
user_gate=no merge, tag, GitHub release, PyPI, or Homebrew publish until user tests and approves
machine=Apple M5 Max, 128 GB unified memory
model=/Users/youssof/.mtplx/hf-upload/Qwen3.6-27B-MTPLX-Optimized
server_profile=sustained max native-MTP
```

Static/package gates:

```text
python3 -m compileall -q mtplx tests scripts -> pass
uv run --extra dev python -m ruff check -> pass
uv run --extra dev python -m pytest -q -> pass
git diff --check -> pass
uv run --extra dev python -m build -> built dist/mtplx-0.3.6.tar.gz and dist/mtplx-0.3.6-py3-none-any.whl
uv run --extra dev python -m twine check dist/* -> pass
scripts/fresh_venv_smoke.sh -> pass
fresh wheel no-deps CLI smoke -> mtplx --version 0.3.6, mtplx-tune dry-run pass, bench tune dry-run JSON pass
```

Tune regression found and fixed:

```text
first_real_tune_gate=failed
failure=mtplx.benchmarks.runners.mtp_depth_sweep imported scripts.probe_draft_lm_head_requant, which is not packaged in the release checkout
fix=use mtplx.draft_lm_head._install_draft_lm_head from package code; update scripts/probe_mx_compile_buckets.py compatibility import
test_added=tests/test_mtp_depth_sweep.py::test_depth_sweep_uses_packaged_draft_lm_head_helper
focused_gate=py_compile + ruff + pytest tests/test_mtp_depth_sweep.py -> pass
```

Real AIME-shaped memory gate on this M5 Max:

```text
command=scripts/aime_shape_memory_bench.py run --suite aime10 --repeat 2 --limit 20 --max-tokens 65536 --temperature 0 --disable-thinking --prompt-mode answer-only
artifact=outputs/release-v0.3.6/253e7eb-dirty-m5max/after/aime20/after-aime10-summary.json
requests=20
decode_tok_s_mean=39.6237
ttft_s_mean=0.1758
prefill_tok_s_mean=346.6311
completion_tokens_total=64
process_rss_bytes_max=21357150208
process_rss_bytes_slope_per_request=753664
peak_memory_bytes_max=22779758600
dynamic_requested_new_tokens_max=65536
dynamic_reserved_new_tokens_max=16384
dynamic_reservation_capped_count=20
session_keep_live_ref_values=False
result=bounded locally; no hundreds-of-GB growth on this M5 Max run
```

Real coding-control gate on this M5 Max:

```text
command=scripts/aime_shape_memory_bench.py run --suite coding3 --limit 3 --max-tokens 4096 --temperature 0 --disable-thinking
artifact=outputs/release-v0.3.6/253e7eb-dirty-m5max/after/coding3/after-coding3-summary.json
requests=3
decode_tok_s_mean=44.7730
ttft_s_mean=0.2173
prefill_tok_s_mean=271.2125
completion_tokens_total=2445
dynamic_reserved_new_tokens_max=4096
dynamic_reservation_capped_count=0
process_rss_bytes_max=21357871104
process_rss_bytes_slope_per_request=188416
session_keep_live_ref_values=False
result=normal max_tokens path did not get capped and memory slope stayed flat locally
```

Real OpenCode CLI gate:

```text
opencode_binary=/opt/homebrew/bin/opencode
project=/tmp/mtplx-v036-opencode-project
config_home=/tmp/mtplx-v036-opencode-home
command=opencode run --model mtplx/mtplx-qwen36-27b-optimized --format json --dangerously-skip-permissions "Create a file named hello_mtplx.txt..."
artifact=outputs/release-v0.3.6/253e7eb-dirty-m5max/after/opencode-run.jsonl
file_created=/tmp/mtplx-v036-opencode-project/hello_mtplx.txt
file_contents="MTPLX OpenCode QA v0.3.6"
tool_result_turn=session_cache_hit true; session_restore_mode reference_lease; request_session_source pending_postcommit_near_prefix; postcommit_wait completed
result=OpenCode tool-result turn reused SessionBank instead of cold-prefilling the 10.7k-token history
```

Real Tune gate:

```text
command=MTPLX_TUNE_STATE=outputs/release-v0.3.6/253e7eb-dirty-m5max/after/tune/tuning-state.json mtplx tune --limit 1 --max-tokens 192 --depths 1,2,3 --seed 0 --retune
artifact=outputs/release-v0.3.6/253e7eb-dirty-m5max/after/tune/tune-summary-rerun.json
thermal=max-fan verified before child model loads; restore ok; post-run max status auto with no marker
AR=24.5833 tok/s, 1.00x
D1=41.5 tok/s, 1.69x
D2=46.2 tok/s, 1.88x
D3=49.8756 tok/s, 2.03x, best
saved=true
cache_check=second run without --retune reused saved tuning cleanly
```

Important non-claims / blockers:

```text
public_release_done=false
m3_ultra_512gb_target_gate=not run on this machine; still required before public memory claim against Ivan's failure class
before_v0.3.5_comparison=not rerun locally in this pass; candidate evidence is local after/fix evidence plus production-path behavior
mlx_fast_fork=not active in this venv (stock MLX 0.31.2 observed), so local speed numbers are QA evidence, not public headline-speed proof
```

## 2026-05-15 00:20 BST - Bench Tune Model/Telemetry UX Fix, No Public Release

Scope:

```text
worktree=/Users/youssof/Documents/MTPLX-release/mtplx-v0.3.6
branch=codex/release-v0.3.6
user_gate=no merge, tag, GitHub release, PyPI, or Homebrew publish until user approves
problem=mtplx bench tune looked less efficient than mtplx start/tune and reported a near-tie D2 win over D3
```

Diagnosis:

```text
no_args_bench_tune_model=/Users/youssof/Documents/MTPLX/models/Qwen3.6-27B-MTPLX-Optimized-Speed
source=~/.mtplx/config.toml
start_verified_default_model=/Users/youssof/.mtplx/hf-upload/Qwen3.6-27B-MTPLX-Optimized
result=the confusing run was not an explicit same-artifact comparison with the start path
user_reported_d2_vs_d3_gap=54.67 vs 54.34 tok/s, about 0.6 percent
interpretation=within short-run noise, not a meaningful architectural D2 win
overlap_check=candidates still run as blocking isolated subprocesses; no candidate overlap evidence
```

Fix:

```text
mtplx bench tune now prints the exact model path before model load
no-arg bench tune warns when ~/.mtplx/config.toml selects a path different from the verified default shown by start
mtplx bench tune --no-telemetry added for clean speed comparison without power sampler diagnostics
bench tune dry-run JSON includes telemetry mode and model-source notes
5.0s settle delay added between AR/D1/D2/D3 candidate subprocesses
near-tie policy added: if a deeper MTP depth is within 2.0 percent of raw fastest, prefer the deeper depth and report the tie-break
```

Real same-artifact checks on `/Users/youssof/.mtplx/hf-upload/Qwen3.6-27B-MTPLX-Optimized`:

```text
clean_tune_run=outputs/cli/tune-comparison/clean-tune-20260515-001540
clean_tune=D3 best, AR 23.53 tok/s, D1 41.83, D2 48.42, D3 54.55, 2.32x AR

bench_no_telemetry_run=outputs/cli/tune-comparison/bench-no-telemetry-20260515-001633
bench_no_telemetry=D3 best, AR 24.62 tok/s, D1 41.61, D2 48.58, D3 54.52, 2.21x AR

bench_telemetry_run=outputs/cli/tune-comparison/bench-telemetry-20260515-001729
bench_telemetry=D3 best, AR 24.44 tok/s, D1 41.85, D2 48.25, D3 54.48, 2.23x AR

conclusion=bench wrapper is not materially slower when telemetry is disabled; telemetry run also stayed effectively identical on this short check
fan_restore=thermalforge status after runs showed both fans in auto mode
```

Validation:

```text
python3 -m compileall -q mtplx tests -> pass
uv run --extra dev python -m pytest tests/test_public_cli.py -q -> pass
uv run --extra dev python -m ruff check mtplx/commands/public.py mtplx/cli.py tests/test_public_cli.py -> pass
git diff --check -> pass
uv run --extra dev python -m build -> pass
uv run --extra dev python -m twine check dist/* -> pass
global install=/opt/homebrew/opt/python@3.14/bin/python3.14 -m pip install --force-reinstall --no-deps dist/mtplx-0.3.6-py3-none-any.whl
global smoke=mtplx --version -> mtplx 0.3.6 (0.3.6)
global dry-run=mtplx bench tune --dry-run --no-telemetry --json shows configured model path, telemetry-disabled description, and verified-default mismatch note
```

## 2026-05-15 00:34 BST - Bench Tune Generation-Window Telemetry Fix, No Public Release

Scope:

```text
worktree=/Users/youssof/Documents/MTPLX-release/mtplx-v0.3.6
branch=codex/release-v0.3.6
user_gate=no merge, tag, GitHub release, PyPI, or Homebrew publish until user approves
problem=bench tune telemetry showed misleading low GPU utilization because parent-side samples covered subprocess setup/teardown and one tail-idle powermetrics sample could land inside the old broad window
```

Fix:

```text
mtp_depth_sweep rows now include generation_started_at, generation_ended_at, and generation_window_s
bench tune telemetry now reports scope=generation when powermetrics/thermalforge samples land inside the actual generation window
scope=candidate is still shown when no generation-window sample is available, so the output does not pretend broad telemetry is generation telemetry
powermetrics samples are timestamped at the midpoint of the actual telemetry capture, preventing end-of-generation idle samples from dragging D3 GPU utilization down
```

Real verification on `/Users/youssof/.mtplx/hf-upload/Qwen3.6-27B-MTPLX-Optimized`:

```text
command=uv run --extra dev python -m mtplx.cli bench tune --model /Users/youssof/.mtplx/hf-upload/Qwen3.6-27B-MTPLX-Optimized --limit 1 --max-tokens 192 --depths 1,2,3 --seed 0 --retune --no-save --yes --output-dir outputs/cli/tune-comparison --run-id bench-generation-midpoint-20260515-003100
artifact=outputs/cli/tune-comparison/bench-generation-midpoint-20260515-003100/tune.json
AR=24.52 tok/s, telemetry scope=generation, gpu=15.7W, GPU=98.9%, samples=5, window=7.8s
D1=41.95 tok/s, telemetry scope=generation, gpu=53.1W, GPU=97.8%, samples=3, window=4.6s
D2=48.51 tok/s, telemetry scope=generation, gpu=53.5W, GPU=81.3%, samples=3, window=4.0s
D3=54.51 tok/s, telemetry scope=generation, gpu=72.0W, GPU=98.4%, samples=2, window=3.5s
fan_restore=thermalforge status after run showed both fans in auto mode
```

Validation:

```text
python3 -m compileall -q mtplx tests -> pass
uv run --extra dev python -m pytest tests/test_public_cli.py tests/test_mtp_depth_sweep.py -q -> pass
uv run --extra dev python -m ruff check mtplx/benchmarks/runners/mtp_depth_sweep.py mtplx/commands/public.py tests/test_public_cli.py -> pass
git diff --check -> pass
```

## 2026-05-15 00:45 BST - Release Copy Honesty Pass

Scope:

```text
worktree=/Users/youssof/Documents/MTPLX-release/mtplx-v0.3.6
branch=codex/release-v0.3.6
user_gate=user approved continuing to public release, with production gates still required before publish
public_state=GitHub/PyPI/Homebrew latest all still v0.3.5 / 0.3.5
target_version=0.3.6
```

Changes:

```text
README no longer calls the 2.24x multiplier hardware-independent; it now says paired same-machine multiplier and tells users to run Tune on their own Mac
README now documents bench tune generation-window telemetry scope
CHANGELOG v0.3.6 now includes bench tune model-source warning, --no-telemetry, and generation-window telemetry fixes
```

Verification:

```text
python3 -m compileall -q mtplx tests scripts -> pass
uv run --extra dev python -m ruff check -> pass
uv run --extra dev python -m pytest -q -> pass
uv run --extra dev python -m twine check dist/* -> pass
scripts/fresh_venv_smoke.sh -> pass
git diff --check -> pass
mtplx --version -> mtplx 0.3.6 (0.3.6)
mtplx start opencode --dry-run --json --model models/example --yes -> pass
mtplx start pi --dry-run --json --model models/example --yes -> pass
mtplx-tune --model models/not-loaded-in-dry-run --dry-run --yes -> pass
mtplx bench tune --model models/not-loaded-in-dry-run --dry-run --json --yes --no-telemetry -> pass
```
