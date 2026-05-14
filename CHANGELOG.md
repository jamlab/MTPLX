# Changelog

All notable user-facing changes are recorded here.

## v0.3.6

### Added

- Added `mtplx tune`, `mtplx-tune`, and `mtplx bench tune` to compare AR
  against D1/D2/D3 on a short coding prompt, keep AR as the `1.00x` baseline,
  and save only a depth that actually beats AR for the current model, Mac, MLX
  stack, and settings.
- Added first-run Web UI and CLI tuning affordance so new installs can opt into
  the measured best depth without hiding the default startup path.
- Added hardware diagnostics to `mtplx bench tune`: each AR/D1/D2/D3 candidate
  now records MX Power Gadget-style power, frequency, temperature, utilization,
  thermal-pressure, and fan samples where macOS exposes them.
- `mtplx bench tune` now labels telemetry scope and prefers generation-window
  power/GPU utilization when samples land inside the actual token generation
  span; `--no-telemetry` keeps the same candidate machinery for cleaner speed
  comparisons.

### Fixed

- Fixed Qwen XML streaming tool calls so arguments are emitted as one complete,
  schema-typed JSON object instead of string-fragment XML parameters. This fixes
  OpenCode `bash.timeout` arriving as `"60000"` and `question.questions`
  arriving as a stringified array.
- Fixed restored-prefix suffix extension so warm SessionBank hits advance the
  suffix in prefill chunks, with abort checks between chunks, instead of one
  long hidden/logits AR call that could leave GPU work running after stop.
- Fixed OpenCode stop-boundary postcommit churn: after the foreground prompt
  prefix is committed, tiny final assistant turns no longer schedule a full
  retokenized-history GPU prefill just to resolve a stop-token mismatch.
- Fixed cold benchmark / one-off request memory growth when clients request
  very large response budgets such as `max_tokens=65536`: dynamic paged KV now
  reserves a bounded initial decode window and grows only if the model actually
  reaches it.
- Fixed anonymous SessionBank entries retaining full-capacity live paged-KV
  buffers for no-reuse workloads, and tightened high-RAM default MLX Metal caps
  so 512 GB Apple Silicon systems do not drift into hundreds of GiB of wired
  allocator memory by default.
- Fixed verified-default onboarding copy and resolution for Optimized Speed:
  the current speed artifact is labeled as Q4 target + Q4 MTP, and an already
  installed local verified artifact is preferred over prompting to download the
  Hugging Face mirror again.
- Fixed first-run Tune from `mtplx start` when launched outside the repo:
  candidate artifacts now use absolute paths, Tune prints per-candidate
  progress, and candidate failures are reported as failures instead of a false
  "no MTP depth beat AR" result.
- Moved Tune's "close heavy apps / fans may get loud" warning to the pre-run
  progress output so the final results screen does not give stale advice after
  measurement is already over.
- Fixed served model-id detection for the installed Optimized Speed artifact so
  mixed Q4/Q8 speed configs do not get mislabeled as Optimized Quality.
- Fixed no-argument `mtplx bench tune` confusion by printing the exact model path
  and warning when `~/.mtplx/config.toml` selects a different artifact than the
  verified default shown by `mtplx start`.

## v0.3.5

### Fixed

- Fixed OpenCode tool-result turns cold-prefilling the full conversation
  history. Stable tool-turn prefixes now commit into SessionBank and follow-up
  OpenCode requests can reuse cached tokens instead of sitting at `Thinking...`
  for minutes.
- Fixed unsafe stream postcommit prefix anchoring so streamed assistant/tool
  histories do not poison the next cache boundary.
- Added regression coverage for real-world consecutive Qwen XML tool calls so
  back-to-back tool calls remain structured and do not leak raw XML.

## v0.3.4

### Added

- Added first-class Swival handoff support through `mtplx start swival`,
  `mtplx start sv`, and `mtplx integrate swival`, including generic-provider
  command generation with the served model id, base URL, and context window.

### Changed

- Made idle SessionBank postcommit cooperative and preemptible. Foreground
  Pi/OpenCode/agent requests can now interrupt long background cache-building
  work instead of sitting silently behind it, while successful idle periods
  still keep the cache-reuse path.
- Reduced the default postcommit wait budget so coding-agent turns prioritize
  first-token latency over background maintenance perfection.

### Fixed

- Fixed consecutive Qwen XML tool-call streaming so multiple back-to-back tool
  calls remain structured OpenAI `delta.tool_calls` instead of collapsing into
  raw assistant text or blocking until the end of generation.
- Improved postcommit/admin observability for preempted maintenance work,
  including clearer `foreground_preempted_postcommit` outcomes.

### Security

- Updated the locked indirect `urllib3` dependency from 2.6.3 to 2.7.0 for
  upstream high-severity streaming/decompression and proxy redirect security
  fixes.

## v0.3.3

### Added

- Added `mtplx doctor android-studio` diagnostics for OpenAI-compatible model
  discovery, nonstream chat, streaming chat, and tool-bearing request smoke.
- Added Android Studio/OpenAI-compatible request-shape tolerance for
  `max_completion_tokens`, `stream_options`, `response_format`, `metadata`,
  `parallel_tool_calls`, and other benign client metadata fields.

### Changed

- Qwen XML tool calls now stream OpenAI `delta.tool_calls` incrementally, so
  long file-write/edit tool calls can mount in compatible clients before the
  full argument body has finished generating.
- Pi handoff no longer writes a hidden model-level `maxTokens` cap; the client
  metadata still advertises the correct `max_tokens` field without silently
  limiting responses.

### Fixed

- Fixed Android Studio issue #58 where `/v1/chat/completions` could fail as
  `500: null` instead of returning either a valid response or a useful
  OpenAI-shaped error envelope.
- Hardened malformed, unknown, unclosed, or schema-invalid tool-call output so
  it falls back safely instead of hanging, 500ing, or storing raw XML as a
  successful assistant tool history item.

## v0.3.2

### Added

- Added first-class OpenCode Desktop integration to `mtplx start`, including
  OpenAI-compatible provider config generation, the `reasoning_content`
  interleaved reasoning field, tool-call support metadata, long chunk timeout,
  and no hidden server-side max-token cap.
- Added `mtplx doctor opencode` diagnostics for the OpenCode config, provider
  payload, server health, and reasoning/tool-call compatibility surface.

### Changed

- Forced raw reasoning streaming for the OpenCode target so reasoning tokens are
  emitted through `reasoning_content` instead of only summarized assistant text.
- Restored the Sustained prefill default to 2048-token chunks for both dense and
  repage paths. Local release QA showed this is the safer user-facing default
  for OpenCode/Pi-style long-context sessions, with bounded memory and better
  representative TTFT behavior than the 4096 split.

### Fixed

- Fixed OpenCode serving ergonomics around schema-aware tool prompting,
  malformed tool-call fallback, and SessionBank/OpenCode prefix reuse telemetry.

## v0.3.1

### Fixed

- Fixed `--no-mtp` / AR OpenAI streaming so `[DONE]` and final stats are sent
  immediately after the last visible token in the default async SessionBank
  postcommit mode. Retokenized SessionBank postcommit now runs as idle async
  maintenance work instead of holding the client stream open.
- Fixed startup/runtime labeling for AR mode so `mtplx quickstart --no-mtp`
  displays `Sustained AR` instead of `Sustained MTP`.

## v0.3.0

### Added

- Added hardware-aware verified default model routing. Fresh M1/M2 Apple
  Silicon setups select the FP16 Optimized Speed variant by default, while
  M3/M4/M5 and unknown hardware stay on the BF16 Optimized Speed default.
  `MTPLX_DEFAULT_MODEL_VARIANT=auto|bf16|fp16` remains available for support
  overrides, and explicit `--model` always wins.
- Added the Optimized Quality model option to the start wizard. It points at
  the Flat8 Optimized Quality artifact when present locally, otherwise the
  verified Hugging Face repo.
- Added SessionBank/model-owner scheduler integration so foreground generation
  has priority over idle postcommit/cache-build work.
- Added postcommit cache-reuse observability including cache hit/miss reason,
  cached token counts, suffix token counts, and SessionBank max-entry tuning.

### Fixed

- Fixed returning-user verified-default configs so saved local Optimized Speed
  paths are re-resolved through the current hardware-aware default instead of
  pinning older Macs to the BF16 default.
- Fixed explicit `--model` handling so saved config can never override a model
  path or repo id typed on the command line.
- Fixed SessionBank reuse above the 16K-token policy threshold, postcommit
  prefix reachability, and streamed tool-call preamble retention.
- Fixed TurboQuant startup so missing optional vLLM Metal external ops fall
  back cleanly instead of 500ing.
- Fixed `mtplx serve --host 0.0.0.0` presentation so the CLI distinguishes the
  actual wildcard bind from the local browser/API URL.

## v0.2.1

### Added

- Added two operator-tunable environment variables for SessionBank capacity:
  `MTPLX_SESSION_BANK_MAX_BYTES` (overrides `DEFAULT_MAX_BYTES`, default 24 GiB)
  and `MTPLX_SESSION_BANK_PER_SESSION_BYTES` (overrides
  `DEFAULT_PER_SESSION_MAX_BYTES`, default 8 GiB). Both accept plain integers
  as bytes and `K`, `M`, `G`, `T` suffixes as powers of 1024, for example
  `MTPLX_SESSION_BANK_PER_SESSION_BYTES=16G`. Invalid or nonpositive values
  fall back to the defaults, so existing deployments see no behavior
  difference.

### Fixed

- Fixed `mtplx quickstart --max`, `mtplx serve --max`, and `mtplx start --max`
  so a stale `~/.mtplx/config.toml` value like `profile = "performance-cold"`
  cannot silently override the v0.2 Sustained Max default. Users who really
  want the short-context Burst lane can still opt in explicitly with
  `--profile performance-cold --max`.
- Added a long-context runtime guard that blocks non-Sustained MTP prefill
  above 16K prompt tokens before it can materialize full prompt hidden/logits
  tensors. This turns the old catastrophic memory path into a clear error that
  tells the user to start MTPLX with `--profile sustained` or run
  `mtplx config set profile sustained`.
- Fixed `_ToolAwareContentStreamTranslator` so it correctly handles assistant
  responses that emit preamble text *before* a `<tool_call>` block. Previously
  the splitter locked into `mode="content"` on the first non-marker byte and
  never re-checked, so a response shaped like `"Let me investigate. <tool_call>...`
  would emit the entire payload (including the tool_call markup) as
  `delta.content` text. OpenAI-compatible clients then saw zero
  `delta.tool_calls`, no tools were dispatched, and the agent loop exited
  with no work done. The translator now also scans incoming text in `content`
  mode for `<tool_call>` markers (with proper handling of partial markers
  spanning multiple stream chunks). Tool-only responses and pure-text
  responses are unchanged. Closes #20.

## v0.2.0

### Added

- Added the `mtplx bench prefill-ladder` release-QA command for measuring
  prompt-prefill TPS, decode TPS, TTFT, memory, acceptance, and fallback
  counters across long-context ladders.
- Added `mtplx hardware inspect --json` for Apple Silicon / MLX acceleration
  eligibility reporting. This release does not claim direct M5 Neural
  Accelerator use without profiling evidence.
- Added `mtplx start pi`, plus the onboarding option "Connect to Pi", so users
  can configure Pi and start the MTPLX OpenAI-compatible server from the normal
  start wizard.
- Added live server-console controls for Pi mode: `/reasoning`, `/mtp`,
  `/stats`, and `/help` remain available in the original MTPLX terminal while
  Pi runs as the client.

### Changed

- Made Sustained the default public long-context path for `mtplx start`,
  `quickstart`, `serve`, and benchmark commands unless users explicitly choose
  Burst / `performance-cold`.
- Packaged the local Metal paged-attention support used by the long-context
  Sustained route so installs no longer depend on a private reference checkout.

### Fixed

- Fixed long-context Sustained prompt prefill so 32K/64K/128K prompts use the
  bounded fast-prefill route without returning to the old 32K memory bloat.
- Fixed OpenAI chat streaming when `tools` are present so normal assistant text
  still streams incrementally instead of buffering until the request completes.
  The unified streaming path now translates generated `<tool_call>` blocks into
  OpenAI `delta.tool_calls` chunks only when the response is actually a tool
  call, while preserving normal `delta.content` streaming for Pi, OpenWebUI,
  Zed, and coding-agent clients that include a `tools` array on every request.
- Fixed `_schedule_idle_postcommit_snapshot` to actually run the retokenized
  SessionBank commit when the foreground goes idle. Previously the function was
  a no-op for unsafe compatibility cases such as tool-call responses, so
  tool-using OpenAI-compatible clients paid full cold prefill on later turns
  even with a stable session id. The async path now commits after the stream
  completes while preserving the "do not block foreground latency" contract.

### Release Notes

- v0.2.0 is the fast-prefill and agent-client release: PP/TPS Sustained QA,
  Pi onboarding, and OpenAI tools streaming are the user-visible themes.
- Issues #9, #13, and #15 are the target issue closeouts for this release.
- No Gemma assistant-pair runtime claim, broad continuous-batching claim, or
  direct M5 Neural Accelerator claim is included in this release.

## v0.1.6

### Fixed

- Fixed streamed tool-call responses so OpenAI-compatible agent clients receive structured tool calls instead of raw model markup.
- Fixed paged-tail routing for streamed server responses.
- Fixed public long-context benchmark defaults so `mtplx bench run` uses the Sustained direct-HTTP lane for long/product suites, while keeping `cold-long-code-192` and explicit `--profile performance-cold` on the Burst lane.

### Release Notes

- This is a small production hotfix over v0.1.5.
- No Gemma assistant-pair runtime, model-weight, sampler, or benchmark-result claims are included in this release.
- This release does not claim the future no-fan long-response decay target or a proven 200K-token production ceiling.

## v0.1.5

### Added

- Added explicit Sustained, Sustained Max, and Burst mode semantics across the CLI, onboarding wizard, quickstart paths, docs, and browser UI.
- Added long-context Sustained runtime telemetry for paged-attention routing, dense-fallback avoidance, large-query fallback behavior, and phase-aware prefill/decode diagnostics.
- Added opt-in Apple Silicon long-context QA coverage for Sustained memory and fallback regression checks.

### Fixed

- Fixed the 16K-32K long-context memory balloon by using chunked prefill, final-token logits, request-sized paged KV, dynamic paged-cache growth, and oversized SessionBank snapshot protection.
- Fixed normal Sustained long-context prefill from silently materializing dense full K/V state after the paged threshold.
- Fixed the 16K Sustained TPS regression by routing large-query paged attention through bounded paths instead of the old dense fallback.
- Fixed stale mode wording in help/docs and surfaced the selected runtime mode in `/health` and the browser chat UI.

### Release Notes

- Sustained is the default long-context native-MTP user path. Sustained Max adds explicit fan boost. Burst remains the old performance-cold max-fan headline lane for short prompts and benchmarks.
- Real QA showed the 32K Sustained path staying below the 35 GiB hard guard and the 16K Sustained Max decode gap recovering to within the release budget.
- This release does not claim the future no-fan long-response decay target or a proven 200K-token production ceiling.

## v0.1.4

### Fixed

- Fixed streaming completions hanging after visible generation finished by committing token-safe SessionBank state from the generation final state, and by moving unsafe postcommit work to an idle-only fallback that does not block the client.
- Fixed false web UI stall aborts during long active generations by adding server-side SSE progress heartbeats and heartbeat-aware browser status handling.
- Fixed the local install/release naming problem by moving from the preview/rc package line to stable `0.1.4` / `v0.1.4` naming.

### Release Notes

- Issue #7 and issue #8 are the user-visible fixes in this release.
- No sampler, decode-loop, MTP acceptance, kernel, or model-weight behavior changed for this release.
- Sustained no-fan long-context throughput remains a future performance track; v0.1.4 fixes serving liveness and release packaging, not the thermal/decay target.

## v0.1.0-preview.3

### Fixed

- Corrected the package and CLI version constants so fresh installs report `mtplx 0.1.0-preview.3 (0.1.0rc3)`. Preview 3 supersedes Preview 2, whose artifacts contained the OpenClaw and WebUI fixes but still printed the Preview 1 version string.

## v0.1.0-preview.2

### Added

- Added OpenAI-compatible tool-call support for agent clients such as OpenClaw: MTPLX now accepts `tools` / `tool_choice`, feeds tool schemas into the Qwen chat template, returns structured `message.tool_calls`, streams `delta.tool_calls`, and preserves tool-result history across turns.
- Added target-only AR switching without unloading the runtime: use `--no-mtp`, `/mtp off` in terminal chat, `"generation_mode":"ar"` in API requests, or the browser chat MTP toggle to compare against native-MTP generation.

### Fixed

- Fixed agent clients printing raw Qwen `<tool_call>` markup instead of executing tools.
- Fixed malformed generated tool-call markup leaking to clients; MTPLX now returns an explicit protocol error.

## v0.1.0-preview.1

### Added

- Added `install_preview_global.sh` to the private GitHub release path so the preview wheel installs into a durable `~/.mtplx/preview-venv` and exposes a normal global `mtplx` launcher.

### Fixed

- Added `mtplx help` as a first-class alias for `mtplx --help`.
- Added nested help aliases such as `mtplx help run` and `mtplx help qa exactness`.

## v0.1.0-preview

### Added

- Lazy package imports so `import mtplx` does not import MLX.
- No-MLX-safe `mtplx --help`, `doctor`, `inspect`, and `init` surface.
- Fresh-venv wheel smoke script for the Phase 0 install gate.
- Public benchmark dry-run paths that do not import heavy runtime modules.
- Packaged OpenAI server entrypoint with API-key guard, rate-limit knob, stream interval, warmup metadata, `/health`, `/metrics`, and `/v1/models` fake-state tests.
- No-MLX-safe `mtplx max` thermal-control surface with explicit ThermalForge/TG Pro detection and opt-in `--max` wiring.
- Baseline Anthropic `/v1/messages` translator, including non-stream responses and `stream=true` SSE events.

### Known Caveats

- Sustained no-fan long-context throughput is below the 50+ tok/s target.
- `performance-cold` is opt-in and may require the MTPLX MLX fork.
- The curated release repository is private-first until QA passes.

### Roadmap

- v0.2: kernel ladder for sustained no-fan throughput.
- v0.3: additional MTP architectures and broader serving polish.
