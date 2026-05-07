# MTPLX v0.1.7 Prefill/Decode Review Packet - 2026-05-07

This packet is for an external pro-agent review of the current dirty worktree:

```text
path=/Users/youssof/Documents/MTPLX-release/mtplx-prefill-fix
branch=codex/sustained-prefill-fix
base_commit=09b49d5f7d3a7647e72251779c5636246f69f329
describe=v0.1.6-1-g09b49d5-dirty
status=not release-candidate
```

## User Outcome Goal

The user wants v0.1.7 to make long-context coding-agent prompts usable on M5 Max and M3 Ultra:

- restore fast prompt prefill
- preserve decode TPS / MTP speed advantage
- keep memory bounded, but do not over-optimize memory at the cost of speed
- keep exact speculative sampling at temp 0.6
- do not claim M5 Neural Accelerator usage without xctrace/Metal proof

## What Changed In This Worktree

Important touched files:

```text
mtplx/generation.py
mtplx/prefill_bench.py
mtplx/cli.py
mtplx/commands/public.py
mtplx/profiles.py
mtplx/kernels/sdpa_2pass.py
tests/test_generation_sustained.py
tests/test_profiles.py
tests/test_public_cli.py
```

Implemented / staged ideas:

```text
1. Sustained prefill uses final-token logits and bounded chunked prefill.
2. MTP history policy defaults to last_window at >=16k contexts.
3. last_window MTP-history prefill no longer asks for hidden states on discarded early chunks.
4. Sustained pins MTPLX_VLLM_METAL_PAGED_TURBOQUANT=0 to prevent inherited KV quantization env.
5. bench prefill-ladder now records richer verify/prefill timing and route metadata.
6. OMLX-style cache cleanup is diagnostic only; measured slightly worse at 32k.
7. OMLX-style stock-cache-only diagnostic is quarantined behind MTPLX_ALLOW_UNSAFE_PREFILL_STOCK_CACHE_ONLY=1 after a 64k M5 Max watchdog panic during that diagnostic.
8. Proposed next patch now staged in profile env: MTPLX_DEFER_VERIFY_HIDDEN_EVAL=1 for sustained only.
```

## OMLX Findings

OMLX clone:

```text
path=/Users/youssof/Documents/MTPLX/REFERENCES:TOOLS/omlx
head=2c1f4c677a795f249cb9ac0d34258e6669314c51
```

Relevant OMLX behavior:

```text
prefill=stock/dense MLX external prefill over tokens[:-1], eval cache roots, hand last token into BatchGenerator
MTP=shallow single-sequence patch, 2-token verify, simpler serving path but lower decode ceiling than MTPLX native MTP
decision=borrow prefill hygiene concepts, do not replace MTPLX native-MTP decode with OMLX MTP
```

## Current Benchmark Evidence

### Full Five-State Comparison

This section is the main pro-agent context table. It compares the states the user is asking about:

```text
dash=not measured in the available artifact/screenshot, not zero
v0.1.5_old_memory_bloat_ref=user-provided M5 Max image with good PP/decode but 99.8GB at 32k
v0.1.6_ivan=user-provided Ivan M5 Max v0.1.6 image, memory fixed but PP collapsed
first_prefill_branch=local pre-OMLX-hygiene MTPLX profile-layout artifacts
mtplx_omlx_copy=local MTPLX safe OMLX-hygiene no-cleanup artifacts, not actual OMLX
actual_omlx=user-provided OMLX M5 Max screenshot, no local artifact
```

Prompt prefill TPS:

| Context | v0.1.5 old memory-bloat ref | v0.1.6 Ivan memory-fixed | First prefill branch | MTPLX OMLX-copy hygiene | Actual OMLX screenshot |
|---:|---:|---:|---:|---:|---:|
| 0.5k | 928.0 | 815.4 | 705.1 | - | - |
| 1k | 779.4 | 699.0 | 547.1 | - | 791.3 |
| 2k | 675.3 | 639.4 | 588.1 | - | - |
| 4k | 680.4 | 649.8 | 703.8 | - | 902.7 |
| 8k | 656.3 | 604.2 | 656.5 | - | 850.7 |
| 16k | 598.0 | 512.1 | 635.0 | - | 733.0 |
| 32k | 472.2 | 370.1 | 591.0 | 623.6 | 662.7 |
| 64k | - | 259.9 | 401.1 | 388.9 | 574.9 |
| 128k | - | 187.5 | 242.0 | - | 421.3 |

Decode / generation TPS:

| Context | v0.1.5 old memory-bloat ref | v0.1.6 Ivan memory-fixed | First prefill branch | MTPLX OMLX-copy hygiene | Actual OMLX screenshot |
|---:|---:|---:|---:|---:|---:|
| 0.5k | 61.1 | 51.6 | 43.4 | - | - |
| 1k | 54.2 | 47.8 | 44.5 | - | 31.8 |
| 2k | 50.5 | 48.6 | 43.2 | - | - |
| 4k | 48.8 | 41.0 | 49.6 | - | 31.1 |
| 8k | 43.6 | 45.2 | 46.7 | - | 30.5 |
| 16k | 43.8 | 39.2 | 40.7 | - | 29.1 |
| 32k | 35.6 | 38.0 | 38.9 | 38.2 | 27.6 |
| 64k | - | 25.4 | 23.1 | 18.8 | 23.6 |
| 128k | - | 22.2 | 13.8 | - | 18.6 |

Peak memory:

| Context | v0.1.5 old memory-bloat ref | v0.1.6 Ivan memory-fixed | First prefill branch | MTPLX OMLX-copy hygiene | Actual OMLX screenshot |
|---:|---:|---:|---:|---:|---:|
| 0.5k | 17.3 GB | 18.4 GB | 15.7 GB | - | - |
| 1k | 17.3 GB | 18.7 GB | 16.2 GB | - | 16.5 GB |
| 2k | 19.2 GB | 20.3 GB | 17.3 GB | - | - |
| 4k | 22.5 GB | 20.8 GB | 17.7 GB | - | 17.9 GB |
| 8k | 29.0 GB | 21.2 GB | 18.4 GB | - | 18.9 GB |
| 16k | 47.2 GB | 21.2 GB | 19.6 GB | - | 20.4 GB |
| 32k | 99.8 GB | 27.5 GB | 22.1 GB | 22.1 GB | 23.4 GB |
| 64k | - | 38.2 GB | 27.2 GB | 27.2 GB | 29.5 GB |
| 128k | - | 61.0 GB | 37.5 GB | - | 41.8 GB |

Key deltas versus Ivan's v0.1.6:

| Context | First prefill PP delta | MTPLX OMLX-copy PP delta | Actual OMLX PP delta | First prefill decode delta | MTPLX OMLX-copy decode delta | Actual OMLX decode delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5k | -13.5% | - | - | -15.9% | - | - |
| 1k | -21.7% | - | +13.2% | -6.9% | - | -33.5% |
| 2k | -8.0% | - | - | -11.1% | - | - |
| 4k | +8.3% | - | +38.9% | +21.0% | - | -24.1% |
| 8k | +8.7% | - | +40.8% | +3.3% | - | -32.5% |
| 16k | +24.0% | - | +43.1% | +3.8% | - | -25.8% |
| 32k | +59.7% | +68.5% | +79.1% | +2.4% | +0.5% | -27.4% |
| 64k | +54.3% | +49.6% | +121.2% | -9.1% | -26.0% | -7.1% |
| 128k | +29.1% | - | +124.7% | -37.8% | - | -16.2% |

Interpretation:

```text
v0.1.5 old state had the good user feel but unacceptable 32k memory.
v0.1.6 fixed memory but long-context PP collapsed.
first prefill branch fixed PP locally and over-reduced memory, but 128k decode collapsed to 13.8 tok/s.
MTPLX OMLX-copy hygiene is not enough: 32k PP improved, but 64k decode worsened to 18.8 tok/s.
Actual OMLX has better PP and moderate memory, but lower decode ceiling than MTPLX should have with native MTP.
```

Normal safe OMLX-hygiene runs:

```text
artifact_32k=benchmarks/results/prefill-omlx-hygiene-m5max-32k-no-cleanup.json
artifact_64k=benchmarks/results/prefill-omlx-hygiene-m5max-64k-no-cleanup.json
```

| Context | PP TPS | Decode TPS | TTFT | Memory | Acceptance | Verify | Draft | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 32k | 623.6 | 38.2 | 52.6s | 22.1 GB | 90/112 = 80.4% | 2.89s | 0.59s | safe normal path |
| 64k | 388.9 | 18.8 | 168.5s | 27.2 GB | 89/116 = 76.7% | 6.31s | 1.78s | PP gate pass, decode fail |

Older current artifact before OMLX-hygiene fold-in:

```text
artifact_64k=benchmarks/results/prefill-fixed-m5max-local-64k-optimized-speed-sustained-auto-mtp.json
64k=384.6 pp, 26.5 decode, 27.2GB, acceptance 89/116=76.7%, verify 4.35s
```

Dense-decode diagnostic:

```text
artifact_64k_dense=benchmarks/results/prefill-fixed-m5max-local-64k-optimized-speed-sustained-auto-dense-decode-mtp.json
64k_dense=392.6 pp, 25.6 decode, 27.2GB, acceptance 89/116=76.7%, verify 4.48s

artifact_128k_dense=benchmarks/results/prefill-fixed-m5max-local-128k-optimized-speed-sustained-auto-dense-decode-mtp.json
128k_dense=243.3 pp, 11.6 decode, 37.5GB, acceptance 89/114=78.1%, verify 10.46s
```

Stock-cache-only diagnostic:

```text
artifact_32k_stock=benchmarks/results/prefill-omlx-hygiene-m5max-32k-stock-cache-only.json
32k_stock=627.5 pp, 37.9 decode, 22.1GB
64k_stock=no completed artifact; machine watchdog-panic occurred during this diagnostic
decision=rejected as default, now requires explicit unsafe env
```

## Current Diagnosis

Prompt prefill is improved enough to pass local M5 PP gates in the available artifacts, but the branch is not release-ready because long-context decode regressed.

The 64k decode drop is not mainly acceptance:

```text
32k acceptance=90/112 = 80.4%
64k acceptance=89/116 = 76.7%
```

The measured 64k regression is verify/eval time:

```text
32k verify=2.89s, draft=0.59s, decode=38.2 tok/s
64k normal verify=6.31s, draft=1.78s, decode=18.8 tok/s
64k dense diagnostic verify=4.48s, draft=1.66s, decode=25.6 tok/s
```

The worst 64k normal verify component is hidden eval:

```text
64k verify_forward_time_s=1.80
64k verify_eval_time_s=4.52
64k verify_hidden_eval_time_s=4.52
```

Interpretation: the verifier is eagerly materializing too much target hidden state before it knows which draft prefix actually commits. That is user-visible as slow token streaming after a long prompt.

## Proposed Next Ideas For Pro-Agent Review

1. **Primary proposal: deferred verify hidden eval.**
   - Current staged patch: Sustained adds `MTPLX_DEFER_VERIFY_HIDDEN_EVAL=1`.
   - Intended outcome: exact target probabilities/logits are evaluated first; hidden state materialization is delayed until the committed prefix is known.
   - Expected user-visible win: recover 64k decode toward the 25 tok/s band while preserving PP around 389 tok/s and exact acceptance.
   - Risk: MLX lazy graph slicing may still materialize more hidden than intended; must benchmark 64k, then 128k.

2. **Backup: hybrid/dense decode verifier route.**
   - Evidence: 64k dense-decode diagnostic has same acceptance and memory but 25.6 decode versus 18.8 normal.
   - Caveat: 128k dense diagnostic was worse, so this may need a context threshold or a verifier-only hybrid rather than a blanket default.

3. **Do not promote stock-cache-only prefill.**
   - It produced only a tiny 32k PP win and was the run correlated with the OS watchdog panic at 64k.
   - It is now quarantined behind `MTPLX_ALLOW_UNSAFE_PREFILL_STOCK_CACHE_ONLY=1`.

4. **Do not copy OMLX MTP wholesale.**
   - OMLX gets strong PP but lower tg TPS. Its shallow 2-token MTP is simpler for serving but would likely throw away MTPLX's native-MTP decode advantage.
   - Borrow scheduling/prefill hygiene; preserve MTPLX exact native-MTP verifier.

5. **If deferred hidden eval fails, profile target long-KV decode attention directly.**
   - AR 128k diagnostic showed MTP is helping, not hurting: MTP 128k decode beat target-only AR.
   - Remaining bottleneck is target long-KV verify/decode attention, not draft acceptance.

## Validation Completed After Last Patch

No heavy model-load benchmark has been run after enabling `MTPLX_DEFER_VERIFY_HIDDEN_EVAL=1`.

Light validation:

```text
python3 -m py_compile mtplx/profiles.py tests/test_profiles.py tests/test_public_cli.py
result=passed

uv run --extra dev python -m ruff check mtplx/profiles.py tests/test_profiles.py tests/test_public_cli.py
result=passed

uv run --extra dev python -m pytest tests/test_profiles.py tests/test_public_cli.py::test_bench_prefill_ladder_dry_run_json -q
result=7 passed
```

## Release Status

```text
not_release_candidate=true
PP_gates=mostly pass locally on M5 artifacts
memory_gates=pass locally on M5 artifacts
fallback_gate=pass in artifacts, zero large_q_split_sdpa_fallback_calls
decode_gate=fail until 64k/128k TPS preservation is recovered
M3_Ultra=not rerun yet
xctrace_M5_Neural_Accelerator=not done; no public accelerator claim
```
