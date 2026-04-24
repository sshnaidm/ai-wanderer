# API Reference

ai-free-swap exposes an OpenAI-compatible API. Any tool or library that works
with the OpenAI API works with ai-free-swap.

---

## Table of Contents

- [Chat Completions](#chat-completions---post-v1chatcompletions)
- [Responses](#responses---post-v1responses)
- [List Models](#list-models---get-v1models)
- [Health Check](#health-check---get-health)
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

## Streaming

Both `/v1/chat/completions` and `/v1/responses` support streaming by setting
`"stream": true`. The proxy uses Server-Sent Events (SSE).

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

**Failover during streaming:** if a provider fails *before* sending any data,
the proxy automatically tries the next provider. If a provider fails *after*
sending partial data, the stream ends with an error (no failover mid-stream).

---

## Error Responses

| HTTP Code | When | Error Code |
|-----------|------|------------|
| 400 | Requested model not configured | `model_not_found` |
| 401 | Invalid or missing API key | `auth_error` |
| 503 | All providers failed | `all_providers_failed` |

Error format:

```json
{
  "error": {
    "message": "All configured providers failed",
    "type": "server_error",
    "code": "all_providers_failed"
  }
}
```
