# Changelog

All notable user-facing changes to MTPLX. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-06-10

The first full release: the native macOS app and the `mtplx` command line
working as one product.

### Added

- Native macOS app with onboarding (hardware check, model pick, guided
  setup, tuning), a live speed dashboard, chat, the AIME benchmark, and
  agent launchers for OpenCode, Pi, Hermes, and Swival.
- Automatic runtime setup during onboarding: the app installs its own
  Python engine, fan control (ThermalForge), and the `mtplx` terminal
  command without requiring Homebrew — release builds bundle a pinned
  CPython interpreter.
- Official Apple Silicon model catalog (Qwen 3.5/3.6, Gemma 4) with
  RAM-aware recommendations shared by the app and the CLI; machines under
  32 GiB are routed to the 9B model automatically.
- App-aware `mtplx start`: detects a running MTPLX server and offers
  "Chat with the running model" instead of loading a second copy, lists
  installed models first, and adds a "Same as the MTPLX app" option for
  returning users.
- New commands: `mtplx stop`, `mtplx settings get/set`, and
  `mtplx bench aime` for running the app's AIME benchmark from the
  terminal.
- Sparkle automatic app updates with signed appcasts; the app refreshes
  its runtime after each update.

### Changed

- Busy ports are now handled gracefully everywhere: the app moves to the
  next free port with a banner (and persists it), and the CLI explains
  exactly who owns a busy port and how to stop it.
- The OpenAI-compatible server honors `stop` sequences (chat,
  completions, and Anthropic `stop_sequences`) and `/v1/completions`
  streams tokens as they are generated with real finish reasons.
- AIME benchmark prompts now carry only the answer-format contract — no
  solution-strategy or style coaching — and every run records the exact
  prompts and rescue policy in its summary for reproducibility.

### Fixed

- Forced final-answer agent turns no longer leak internal rehearsal text
  or drop tools mid-conversation.
- The Qwen 3.6 35B speed preset applies its measured draft sampler unless
  explicitly overridden.
- Skipping the tuning step during onboarding no longer skips runtime
  installation.

[1.0.0]: https://github.com/youssofal/mtplx/releases/tag/v1.0.0
