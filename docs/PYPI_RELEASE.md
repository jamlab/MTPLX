# PyPI release runbook

MTPLX publishes to PyPI through PyPI Trusted Publishing from GitHub Actions.
This avoids long-lived PyPI API tokens in GitHub secrets.

## One-time PyPI setup

`mtplx` is expected to be created through a pending trusted publisher.

Create the pending publisher at:

```text
https://pypi.org/manage/account/publishing/
```

Use exactly:

```text
PyPI project name: mtplx
Owner: youssofal
Repository name: MTPLX
Workflow filename: release.yml
Environment name: pypi
```

The environment name matters. PyPI checks it against the GitHub OIDC token, so
`pypi` on PyPI must match the `environment: pypi` job in
`.github/workflows/release.yml`.

## Publish Preview 1

After the pending publisher exists, run:

```bash
gh workflow run release.yml \
  --repo youssofal/MTPLX \
  -f ref=v0.1.0-preview.1 \
  -f publish_to_pypi=true
```

Watch the run:

```bash
gh run list --repo youssofal/MTPLX --workflow release --limit 1
gh run watch --repo youssofal/MTPLX --exit-status
```

## Verify the public install

```bash
python3 -m venv /tmp/mtplx-pypi-verify
/tmp/mtplx-pypi-verify/bin/python -m pip install -U pip
/tmp/mtplx-pypi-verify/bin/python -m pip install mtplx
/tmp/mtplx-pypi-verify/bin/mtplx help
```

Preview 1 is packaged as `0.1.0rc1`. If a user's pip is configured to reject
pre-releases even when no stable release exists, this explicit form also works:

```bash
python3 -m pip install --pre mtplx
```

## Release guardrails

- PyPI upload is manual only: tag pushes build artifacts but do not publish.
- Publishing requires `publish_to_pypi=true`.
- Publishing requires the `ref` input to be a version tag beginning with `v`.
- PyPI Trusted Publishing must be configured with the exact repository,
  workflow, and environment above.
