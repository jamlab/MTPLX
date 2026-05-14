# MTPLX Release Log

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
