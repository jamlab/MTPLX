# Development

```bash
python -m pip install -e ".[dev,server]"
python -m pytest tests/test_no_mlx_imports.py tests/test_public_cli.py tests/test_runtime_kpis.py
python -m build
scripts/fresh_venv_smoke.sh
```

Keep generated artifacts, model weights, and local credentials out of Git. The release repository is a product export, not a research workspace dump.

## Release

Preview artifacts are published from a clean tag:

```bash
git tag -a v0.1.0-preview.3 -m "MTPLX v0.1.0-preview.3"
git push origin v0.1.0-preview.3
gh release create v0.1.0-preview.3 dist/* scripts/install_macos.sh scripts/install_preview_global.sh --prerelease --title "MTPLX Preview 3"
```

Use GitHub CLI authentication for artifact smoke tests:

```bash
gh release download v0.1.0-preview.3 --repo youssofal/mtplx --pattern 'mtplx-0.1.0rc3-py3-none-any.whl' --pattern 'install_preview_global.sh'
bash install_preview_global.sh ./mtplx-0.1.0rc3-py3-none-any.whl
mtplx help
```

PyPI publishing is wired through Trusted Publishing, not local long-lived tokens. Before enabling the upload job, configure a pending publisher on PyPI:

```text
project: mtplx
owner: youssofal
repository: MTPLX
workflow: release.yml
environment: pypi
```

The `release.yml` workflow always builds and checks artifacts for tags. PyPI
publishing is manual only:

- tag pushes build and check artifacts but do not publish
- a maintainer must run the workflow with `publish_to_pypi=true` after the tag
  exists and the local build/twine gates pass
