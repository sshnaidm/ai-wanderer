# API Reference

ai-free-swap exposes both an OpenAI-compatible and an Anthropic-compatible API.
Any tool or library that works with either API works with ai-free-swap.

---

## Table of Contents

- [Chat Completions](#chat-completions---post-v1chatcompletions)
- [Responses](#responses---post-v1responses)
- [Anthropic Messages](#anthropic-messages---post-v1messages)
- [List Models](#list-models---get-v1models)
- [Health Check](#health-check---get-health)
- [Authentication](#authentication)
- [Streaming](#streaming)
- [Error Responses](#error-responses)

---

## Chat Completions - `POST /v1/chat/completions`

Standard OpenAI chat completions endpoint.

**Request:**

```json
{
  "model": "aifree",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false
}
```

The `model` field can be:
- `"aifree"` (or whatever you set `model_name` to) -- uses any available provider.
- A specific backend model name (e.g., `"gemini-2.5-flash"`) -- only uses
  providers configured with that model.

**Supported parameters:** `temperature`, `top_p`, `n`, `stop`, `max_tokens`,
`presence_penalty`, `frequency_penalty`, `tools`, `tool_choice`,
`response_format`, `seed`, `user`, and others. Unknown parameters are forwarded
to the provider.

**Response:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gemini-2.5-flash",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "2+2 equals 4."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  },
  "provider_name": "gemini"
}
```

The `model` field in the response shows the actual backend model that handled
the request. The `provider_name` field (when `show_provider: true`) shows which
provider was used.

---

## Responses - `POST /v1/responses`

OpenAI Responses API format.

**Request:**

```json
{
  "model": "aifree",
  "input": "What is the capital of France?",
  "instructions": "Answer concisely.",
  "stream": false
}
```

The `input` field accepts a string, an array of messages, or a message object.

**Response:**

```json
{
  "id": "resp_abc123",
  "object": "response",
  "created_at": 1234567890.123,
  "model": "gemini-2.5-flash",
  "status": "completed",
  "output": [{
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "content": [
      {"type": "output_text", "text": "Paris."}
    ]
  }],
  "output_text": "Paris.",
  "provider_name": "gemini"
}
```

---

## Anthropic Messages - `POST /v1/messages`

Anthropic Messages API endpoint. Allows Anthropic SDK clients to use the proxy
without any code changes -- just set `ANTHROPIC_BASE_URL`.

**Request:**

```json
{
  "model": "aifree",
  "max_tokens": 1024,
  "system": "You are a helpful assistant.",
  "messages": [
    {"role": "user", "content": "What is 2+2?"}
  ],
  "temperature": 0.7,
  "stream": false
}
```

The `model` field works the same way as the OpenAI endpoints: use `"aifree"`
for any available provider, or a specific backend model name.

**Supported parameters:** `max_tokens`, `system` (string or content block
array), `temperature`, `top_p`, `top_k`, `stop_sequences`, `stream`,
`metadata`. Unknown parameters are forwarded to the provider.

**Response:**

```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "content": [
    {"type": "text", "text": "2+2 equals 4."}
  ],
  "model": "gemini-2.5-flash",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0
  },
  "provider_name": "gemini"
}
```

The `model` field in the response shows the actual backend model. The
`provider_name` field (when `show_provider: true`) shows which provider
was used.

**Using with the Anthropic SDK:**

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8000",
    api_key="unused",  # or your proxy api_key if set
)

response = client.messages.create(
    model="aifree",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
```

**Using `ANTHROPIC_BASE_URL`:**

```bash
export ANTHROPIC_BASE_URL="http://localhost:8000"
export ANTHROPIC_API_KEY="your-proxy-key"

# Now any tool using the Anthropic SDK goes through your proxy
claude        # Claude Code
aider         # aider with Anthropic models
```

---

## List Models - `GET /v1/models`

Returns the configured model name.

```json
{
  "object": "list",
  "data": [
    {"id": "aifree", "object": "model", "owned_by": "ai-free-swap"}
  ]
}
```

---

## Health Check - `GET /health`

Returns `{"status": "ok"}` if the server is running. This endpoint does not
require authentication.

---

## Authentication

When the proxy has `server.api_key` configured, clients must authenticate.
Two methods are supported:

- **`Authorization: Bearer <key>`** -- standard OpenAI-style header
- **`x-api-key: <key>`** -- Anthropic-style header

Both headers work on all endpoints. If both are present, `Authorization`
takes precedence. The `/health` endpoint is always public.

---

## Streaming

All three endpoints (`/v1/chat/completions`, `/v1/responses`, and
`/v1/messages`) support streaming by setting `"stream": true`. The proxy
uses Server-Sent Events (SSE).

**Chat completions streaming** sends chunks in standard OpenAI format:

```
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","model":"gemini-2.5-flash","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","model":"gemini-2.5-flash","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data: [DONE]
```

**Responses streaming** sends named events:

```
event: response.created
data: {"type":"response.created","response":{"id":"resp_abc","status":"in_progress",...}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"Hello",...}

event: response.completed
data: {"type":"response.completed","response":{...}}

data: [DONE]
```

**Anthropic messages streaming** sends named events in Anthropic format:

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_abc","type":"message","role":"assistant","content":[],"model":"gemini-2.5-flash",...}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}

event: message_stop
data: {"type":"message_stop"}
```

**Failover during streaming:** if a provider fails *before* sending any data,
the proxy automatically tries the next provider. If a provider fails *after*
sending partial data, the stream ends with an error (no failover mid-stream).

---

## Error Responses

**OpenAI endpoints** (`/v1/chat/completions`, `/v1/responses`):

| HTTP Code | When | Error Code |
|-----------|------|------------|
| 400 | Requested model not configured | `model_not_found` |
| 401 | Invalid or missing API key | `auth_error` |
| 503 | All providers failed | `all_providers_failed` |

```json
{
  "error": {
    "message": "All configured providers failed",
    "type": "server_error",
    "code": "all_providers_failed"
  }
}
```

**Anthropic endpoint** (`/v1/messages`):

| HTTP Code | When | Error Type |
|-----------|------|------------|
| 400 | Requested model not configured | `not_found_error` |
| 401 | Invalid or missing API key | `auth_error` |
| 529 | All providers failed | `overloaded_error` |

```json
{
  "type": "error",
  "error": {
    "type": "overloaded_error",
    "message": "All configured providers failed"
  }
}
```
