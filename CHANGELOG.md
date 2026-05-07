# Changelog

All notable user-facing changes are recorded here.

## Unreleased

### Fixed

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

- Next: kernel ladder for sustained no-fan throughput.
- Later: additional MTP architectures and broader serving polish.
