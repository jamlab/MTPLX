# API

MTPLX v0.1 targets OpenAI-compatible local serving first.

## `GET /health`

Reports model load state, profile, exactness baseline, MLX/runtime information, fan mode, and warmup status.
The payload includes `api_key_required`, `rate_limit_per_minute`, `stream_interval`, `warmup`, and `reasoning_parser` so client harnesses can confirm the active serving policy.

## `GET /metrics`

Reports runtime KPIs as JSON or Prometheus-style text, depending on server configuration.

## `GET /v1/models`

Lists cached and active models.

## `POST /v1/chat/completions`

OpenAI-compatible chat completions. Streaming uses server-sent events.
Use `--stream-interval N` to batch committed-token SSE chunks when a client prefers less frequent events.

## `POST /v1/completions`

Legacy OpenAI completions.

## `POST /v1/messages`

Anthropic Messages baseline. Requests are translated into the same internal chat path as `/v1/chat/completions` and returned as Anthropic-shaped message payloads.

Supported now:

- `system` as text or text content blocks
- `messages[].content` as text or text/tool-result content blocks
- `max_tokens`, `temperature`, `top_p`, and `top_k`
- `stream=false`
- `stream=true` server-sent events with `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, and `message_stop`

Current streaming note: Qwen reasoning deltas are exposed as text deltas until the Anthropic thinking-block mapping is validated against real Claude Code / OpenCode clients.

Examples:

- [Anthropic Python client](../examples/anthropic-python-client.py)
- [Anthropic Messages curl](../examples/curl-messages.sh)
- [OpenAI Python client](../examples/openai-python-client.py)
- [OpenAI chat completions curl](../examples/curl-chat-completions.sh)

## Server Flags

```bash
mtplx serve --port 8000
mtplx serve --host 0.0.0.0 --api-key "$MTPLX_AUTH"
mtplx serve --rate-limit 120
mtplx serve --stream-interval 4
mtplx serve --warmup-tokens 16
mtplx serve --reasoning-parser qwen3
```

Non-localhost binds require `--api-key`. Requests may authenticate with either:

```text
Authorization: Bearer <key>
X-API-Key: <key>
```

`--warmup-tokens` runs a small startup generation after model load and reports the result in `/health`. `--strict-warmup` makes warmup failure fatal.
