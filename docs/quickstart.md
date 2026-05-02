# Quickstart

```bash
gh repo clone youssofal/mtplx mtplx-preview
cd mtplx-preview
gh release download v0.1.0-preview.1 --repo youssofal/mtplx --pattern 'mtplx-0.1.0rc1-py3-none-any.whl'
scripts/install_preview_global.sh ./mtplx-0.1.0rc1-py3-none-any.whl

mtplx help
mtplx doctor --json
mtplx init --model /path/to/verified/model --write
mtplx inspect /path/to/verified/model --json
```

Public `pip install mtplx` is the Stage C target after PyPI Trusted Publishing is configured. The current private preview path uses the GitHub release wheel plus `scripts/install_preview_global.sh` so `mtplx` works from a normal Terminal without activating a project venv.

The commands above are no-MLX-safe except generation and serving. A missing MLX runtime should appear in `doctor` as an actionable dependency issue, not a traceback.

After a verified model is available:

```bash
mtplx run "hello"
mtplx chat
mtplx serve --port 8000 --no-stats-footer
```
