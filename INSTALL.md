# Install MTPLX

MTPLX is preview software for Apple Silicon Macs.

## Requirements

- Apple Silicon Mac
- Python 3.11, 3.12, or 3.13
- macOS with MLX support
- Enough disk for the selected model

## Install

Public install, after PyPI publication:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install mtplx
```

Private preview install, while the GitHub repository is private:

```bash
gh repo clone youssofal/mtplx mtplx-preview
cd mtplx-preview
gh release download v0.1.0-preview.1 --repo youssofal/mtplx --pattern 'mtplx-0.1.0rc1-py3-none-any.whl'
scripts/install_preview_global.sh ./mtplx-0.1.0rc1-py3-none-any.whl
```

The preview installer writes a durable launcher at `~/.local/bin/mtplx`, adds
`~/.local/bin` to zsh startup files when needed, and writes
`/opt/homebrew/bin/mtplx` on Apple Silicon Homebrew installs when that directory
is writable. That means `mtplx help` works from a normal new Terminal tab
without activating a project venv.

For local development:

```bash
python -m pip install -e ".[dev,server]"
```

## Runtime Dependencies

`mtplx --help`, `mtplx doctor`, `mtplx inspect`, and `mtplx init` are designed to work even before MLX is installed. Generation and serving require MLX and a verified model.

The v0.1 default dependency path uses vanilla `mlx`. The opt-in `performance-cold` profile may require the MTPLX MLX fork until the custom-kernel work is upstreamed or extracted.

## Optional Thermal Tools

`--max` is opt-in. It is for users who need sustained throughput and accept fan noise. It is never part of the default quick start and is never used for no-fan product claims.

Check the local thermal-control state:

```bash
mtplx max --status
```

If ThermalForge or TG Pro is not present, MTPLX prints install instructions and continues without fan control for `run`, `chat`, and `serve --max`. It must not silently enable spin-loop or clock-anchor modes.

Supported public commands:

```bash
mtplx max --on       # Performance profile
mtplx max --max      # Max profile
mtplx max --off      # Silent profile
mtplx max --status   # tool/status report
```

`MTPLX_GPU_CLOCK_ANCHOR=1` is an explicit experimental diagnostic only. Do not use it for README, release, or product benchmark claims.
