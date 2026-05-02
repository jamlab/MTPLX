# Server

The v0.1 server target is OpenAI-compatible local serving, with Anthropic
Messages compatibility available for coding harness smoke tests.

```bash
mtplx serve --host 127.0.0.1 --port 8000 --no-stats-footer
```

Endpoints:

- `GET /health`
- `GET /metrics`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/messages`
- `GET /admin/sessions`
- `POST /admin/cache/clear`

Binding to a non-localhost host requires an API key:

```bash
mtplx serve --host 0.0.0.0 --port 8000 --api-key "$MTPLX_AUTH"
```

For Open WebUI, set the OpenAI-compatible base URL to:

```text
http://127.0.0.1:8000/v1
```

For Anthropic Messages-compatible clients, point the client base URL at the
same local server root:

```text
http://127.0.0.1:8000/v1
```

Use `--no-stats-footer` for Open WebUI, Claude Code, OpenCode, and other
clients that treat assistant content as the only user-visible answer. Metrics
remain available at `/metrics`.
