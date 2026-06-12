# Changelog

All notable user-facing changes to MTPLX. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [1.0.4] - 2026-06-12

Same-day hotfix: 1.0.3 broke coding agents on their first tool turn,
and pasting a GGUF repo got a wrong answer.

### Fixed

- Coding agents crashed on 1.0.3. Any tool-using client (Pi, Hermes,
  OpenCode, or anything speaking the OpenAI tools protocol) hit
  "unexpected keyword argument 'vision_splice'" on its first tool
  turn after a cache miss: non-streaming callers got a 500, streaming
  agents lost the stream, and the app reported "Stream offline." One
  stray argument left behind by the vision work, in a diagnostics
  call only agent tool turns reach. Removed, with tests that drive
  the exact path and an audit test that fails if any call ever passes
  an argument its target does not accept (#99, #100).
- Pasting a GGUF repo into "Add a model from Hugging Face" claimed
  the repository did not exist. The app now says what is actually
  going on: GGUF is llama.cpp's format and MTPLX runs MLX models. It
  names the source repo the GGUF was made from so Forge can convert
  it, and genuine typos get "check the name" instead.
- The "Add a model" repo check now follows the configured Hugging
  Face download mirror instead of always probing huggingface.co, so
  it works on networks where huggingface.co is blocked.

## [1.0.3] - 2026-06-12

The app can see. Vision lands across the Qwen models, and the
compatibility gate stops blocking models that run fine.

### Added

- Vision support in chat and the API. Attach PNG, JPEG, or WebP images
  in the app composer, or send OpenAI image_url content parts to
  /v1/chat/completions, and the model describes what it sees with MTP
  speculative decoding still running on top. Works on Qwen 3.6 27B
  (Speed and Quality), Qwen 3.6 35B, and Qwen 3.5 9B. The 9B repo on
  Hugging Face regained its vision weights; an explicit mtplx pull now
  syncs such repo updates into existing local copies automatically.
- /health reports whether the loaded model supports vision, and the
  composer adapts to it.

### Fixed

- Models that run fine are no longer refused for paperwork. The
  compatibility gate treated unverified runtime contracts (including
  the official Optimized Quality build) and even "slower than AR"
  speed evidence as reasons not to load. Verification is now a label:
  unverified models load with an honest note, and refusals are
  reserved for models that genuinely cannot execute (#98).
- The gate's explanation message crashed with a traceback instead of
  printing since 1.0.0. It prints again, including the hint that was
  supposed to unblock you (#98).
- Image attachments preview their actual pixels in the composer and
  the transcript instead of a "Could not read" placeholder.

## [1.0.2] - 2026-06-11

Bug-fix release with one small feature.

### Fixed

- Choosing the Auto or Sustained Max profile in the app's Settings left
  the engine unable to start, showing Degraded on every launch until
  the profile was changed back. Both values now resolve to real
  profiles (Sustained Max keeps its pinned-fans intent as the fan mode
  setting), existing configurations heal themselves on load, and the
  picker only offers values the engine accepts. `mtplx serve --profile
  auto` works from the command line too.
- Parallel requests from agent tools that do not send session ids could
  fail with "session anon-... is already in flight" when they shared a
  prompt prefix. Busy sessions now fork to a fresh session instead of
  erroring, and anonymous session ids are random rather than clock
  derived. Reported and fixed by Frank Denis (@jedisct1) in #95.
- A daemon launch that lost its port (another server bound it between
  checks, or a listener invisible to the local probe held it) now
  remediates and retries once before reporting a failure, and the
  failure message names the occupant when it can.

### Added

- Optional Hugging Face download mirror for networks where
  huggingface.co is blocked (requested from mainland China in #96). Set
  it inline in the onboarding download step or later in Settings under
  Advanced; downloads and the engine then use the mirror endpoint. The
  stored HF token is never sent to a mirror, so gated repos stay on the
  official endpoint.

## [1.0.1] - 2026-06-11

Bug-fix release.

### Fixed

- First-run tuning no longer fails on Macs where fan control cannot
  verify a max ramp (for example when the passwordless helper grant is
  not in place yet). Tuning now runs with fans on automatic, the
  results are labeled accordingly, and `--require-max-fans` keeps the
  strict behavior for benchmarking.
- The `mtplx` CLI accepts the official Gemma 4 assistant-pair repos
  directly from Hugging Face. The app already ran them; the CLI's
  preflight now reaches the same verdict.

## [1.0.0] - 2026-06-10

The first full release: the native macOS app and the `mtplx` command line
working as one product. Full notes:
[mtplx.com/releases/notes/v1.0.0](https://mtplx.com/releases/notes/v1.0.0.html).

### Added

- Native macOS app with onboarding (hardware check, model pick, guided
  setup, tuning), a live speed dashboard (decode gauge, acceptance by
  depth, verify waterfall, activity), native chat with attachments and
  web search, Forge, the AIME benchmark, and agent launchers for
  OpenCode, Pi, Hermes, and Open WebUI.
- New models: Gemma 4 (assistant-pair drafting tuned by draft block
  size) and Qwen 3.6 MoE 35B-A3B (prequantized expert sidecars,
  normalized expert layouts, hard blocks on unrunnable layouts),
  alongside Qwen 3.5 4B and 9B for smaller machines.
- KV cache reuse on two layers: warm-prefix reuse in RAM across turns
  and requests (multi-turn chats and agents like OpenCode hit the cache
  instead of re-processing the conversation), and an SSD session cache
  that persists KV state to disk with enforced size caps and restores
  near-instantly across server restarts.
- Concurrency: continuous batching with presets, a scheduler mode, and
  explicit caps (`--max-active-requests`, `--decode-batch-max`,
  `--batch-wait-ms`).
- Smart fan mode across the app, CLI, and server API: ramps while the
  model works, restores on idle, survives client handoffs, and keeps the
  crash-safe restore watchdog.
- Forge: convert any Hugging Face repo to MLX (AWQ, compressed-tensors,
  NVFP4, BF16 sources), calibrate and train the MTP adapter, verify with
  quality gates that reject speed wins that degrade output, and publish
  with provenance. Vision towers are preserved through conversion. In
  the app and as `mtplx forge`.
- Agent-grade serving: hardened tool contracts and dedicated lanes for
  OpenCode, Pi, and Hermes; long-context depth policy; client identity
  tagging; a live server-sent metrics stream plus snapshot, thermal, and
  prefill-history endpoints; honest cancellation that stops decode.
- Automatic runtime setup during onboarding: the app installs its own
  Python engine, fan control (ThermalForge), and the `mtplx` terminal
  command without requiring Homebrew. Release builds bundle a pinned
  CPython interpreter, the engine environment ignores user pip
  configuration, and the interpreter is signed so installed packages
  load on macOS 14 and 15. A stale `mtplx` on PATH is updated
  automatically; a newer one is left alone.
- Official Apple Silicon model catalog (Qwen 3.5/3.6, Gemma 4 in speed,
  balance, and quality builds) with device-aware defaults shared by the
  app and the CLI: chip generation picks precision and machines under
  32 GiB route to the 9B model automatically.
- App-aware `mtplx start`: detects a running MTPLX server and attaches
  instead of loading a second copy, lists installed models first, and
  adds a "Same as the MTPLX app" option. `mtplx stop` knows the app's
  persisted port.
- New commands: `mtplx stop`, `mtplx settings get/set`, and
  `mtplx bench aime` for running the app's AIME benchmark from the
  terminal.
- Sparkle automatic app updates with signed appcasts; the app verifies
  the installed engine against the shipped wheel and refreshes it after
  each update.

### Changed

- Busy ports are now handled gracefully everywhere: the app moves to the
  next free port with a banner (and persists it), and the CLI explains
  exactly who owns a busy port and how to stop it.
- The OpenAI-compatible server honors `stop` sequences (chat,
  completions, and Anthropic `stop_sequences`) and `/v1/completions`
  streams tokens as they are generated with real finish reasons.

[1.0.0]: https://github.com/youssofal/mtplx/releases/tag/v1.0.0
