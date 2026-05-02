# Changelog

All notable user-facing changes are recorded here.

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
