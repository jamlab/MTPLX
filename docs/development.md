# Development

```bash
python -m pip install -e ".[dev,server]"
python -m pytest tests/test_no_mlx_imports.py tests/test_public_cli.py tests/test_runtime_kpis.py
python -m build
scripts/fresh_venv_smoke.sh
```

Keep generated artifacts, model weights, and local credentials out of Git. The release repository is a product export, not a research workspace dump.

## Release

Private GitHub preview artifacts are published from a clean tag:

```bash
git tag -a v0.1.0-preview.1 -m "MTPLX v0.1.0-preview.1"
git push origin v0.1.0-preview.1
gh release create v0.1.0-preview.1 dist/* --prerelease --title "MTPLX v0.1.0-preview.1"
```

While the repository is private, direct unauthenticated release-asset URLs return 404. Use GitHub CLI authentication for private artifact smoke tests:

```bash
gh release download v0.1.0-preview.1 --repo youssofal/mtplx --pattern 'mtplx-0.1.0rc1-py3-none-any.whl'
python -m pip install ./mtplx-0.1.0rc1-py3-none-any.whl
```

PyPI publishing is wired through Trusted Publishing, not local long-lived tokens. Before enabling the upload job, configure a pending publisher on PyPI:

```text
project: mtplx
owner: youssofal
repository: mtplx
workflow: release.yml
environment: pypi
```

The `release.yml` workflow always builds and checks artifacts for tags. It uploads to PyPI only when either:

- a maintainer manually runs the workflow with `publish_to_pypi=true`, or
- the repository variable `ENABLE_PYPI_PUBLISH` is set to `true` for tag pushes.

Keep `ENABLE_PYPI_PUBLISH` unset until the PyPI pending publisher and GitHub `pypi` environment approval gate are configured.
