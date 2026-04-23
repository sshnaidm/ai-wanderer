# ai-free-swap

An OpenAI-compatible proxy server that routes your requests through multiple
free-tier AI providers. If one provider is down or rate-limited, the proxy
automatically tries the next one -- so your application keeps working without
any code changes.

You configure a list of AI providers (Google Gemini, Qwen, OpenRouter, xAI Grok,
Anthropic, or any OpenAI-compatible service), assign priorities, and the proxy
handles the rest. Your app talks to one local endpoint, and ai-free-swap finds a
working provider behind the scenes.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
  - [1. Install](#1-install)
  - [2. Get API Keys](#2-get-api-keys)
  - [3. Configure](#3-configure)
  - [4. Run](#4-run)
  - [5. Send a Request](#5-send-a-request)
- [Installation Methods](#installation-methods)
  - [Install with pip](#install-with-pip)
  - [Run with Docker](#run-with-docker)
  - [Run from Source](#run-from-source)
- [Configuration](#configuration)
  - [Configuration File](#configuration-file)
  - [Top-Level Settings](#top-level-settings)
  - [Server Settings](#server-settings)
  - [Provider Settings](#provider-settings)
  - [Using Environment Variables for API Keys](#using-environment-variables-for-api-keys)
  - [Priority and Fallback](#priority-and-fallback)
  - [Custom OpenAI-Compatible Providers](#custom-openai-compatible-providers)
  - [Full Configuration Example](#full-configuration-example)
- [Supported Providers](#supported-providers)
- [API Reference](#api-reference)
  - [Chat Completions](#chat-completions---post-v1chatcompletions)
  - [Responses](#responses---post-v1responses)
  - [List Models](#list-models---get-v1models)
  - [Health Check](#health-check---get-health)
  - [Streaming](#streaming)
  - [Error Responses](#error-responses)
- [Using with Applications](#using-with-applications)
  - [OpenAI Python SDK](#openai-python-sdk)
  - [curl](#curl)
  - [Any OpenAI-Compatible Client](#any-openai-compatible-client)
- [Command-Line Options](#command-line-options)
- [Securing the Proxy](#securing-the-proxy)
- [Troubleshooting](#troubleshooting)
- [Running Tests](#running-tests)

---

## How It Works

```
Your App                ai-free-swap                   Providers
  |                         |                              |
  |-- POST /v1/chat/... -->|                              |
  |                         |-- try Gemini (priority 1) ->|
  |                         |<-- error / rate limited -----|
  |                         |                              |
  |                         |-- try Qwen (priority 2) --->|
  |                         |<-- success ------------------|
  |                         |                              |
  |<-- response -----------|                              |
```

1. Your application sends a standard OpenAI-format request to the proxy.
2. The proxy tries providers in priority order (lowest number = highest priority).
3. Within the same priority level, providers are tried in random order.
4. If a provider fails, the proxy automatically tries the next one.
5. The response is returned in standard OpenAI format -- your app doesn't need
   to know which provider actually handled the request.

---

## Quick Start

### 1. Install

```bash
pip install .
```

### 2. Get API Keys

Sign up for free API keys from one or more providers:

| Provider | Sign Up |
|----------|---------|
| Google Gemini | https://aistudio.google.com/apikey |
| Alibaba Qwen | https://dashscope.console.aliyun.com/ |
| OpenRouter | https://openrouter.ai/keys |
| xAI Grok | https://console.x.ai/ |
| Anthropic | https://console.anthropic.com/ |

### 3. Configure

```bash
cp config.yaml.example config.yaml
```

Open `config.yaml` in any text editor and replace the placeholder API keys
with your actual keys. You can either paste the keys directly or use
environment variables (see [Using Environment Variables](#using-environment-variables-for-api-keys)).

At minimum, you need one provider with an API key. For example, if you only
have a Gemini key:

```yaml
keep_cycles: 1
model_name: "aifree"

server:
  host: "0.0.0.0"
  port: 8000

providers:
  - priority: 1
    backends:
      - provider: gemini
        api_key: "your-gemini-api-key-here"
        model: "gemini-2.5-flash"
```

### 4. Run

```bash
ai-free-swap --config config.yaml
```

The server starts at `http://localhost:8000`.

### 5. Send a Request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "aifree",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

You should get a response like:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "gemini-2.5-flash",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello! How can I help you?"},
    "finish_reason": "stop"
  }],
  "provider_name": "gemini"
}
```

---

## Installation Methods

### Install with pip

Requires **Python 3.11 or later**.

```bash
# Install from the project directory
pip install .

# Or install in development mode (changes to source take effect immediately)
pip install -e .
```

After installation, the `ai-free-swap` command is available system-wide.

### Run with Docker

```bash
# Build the image
docker build -t ai-free-swap .

# Run with your config file mounted in
docker run -p 8000:8000 \
  -v /path/to/your/config.yaml:/app/config.yaml \
  -e GEMINI_API_KEY="your-key" \
  ai-free-swap
```

If your config uses `${ENV_VAR}` syntax for API keys, pass the environment
variables with `-e`:

```bash
docker run -p 8000:8000 \
  -v /path/to/your/config.yaml:/app/config.yaml \
  -e GEMINI_API_KEY_1="key1" \
  -e GEMINI_API_KEY_2="key2" \
  -e DASHSCOPE_API_KEY="key3" \
  ai-free-swap
```

### Run from Source

If you prefer not to install:

```bash
# Install dependencies
pip install -r requirements.txt

# Run directly
python -m ai_free_swap --config config.yaml

# Or use the shell wrapper
./run.sh config.yaml
```

---

## Configuration

### Configuration File

The proxy is configured with a YAML file. Copy the example to get started:

```bash
cp config.yaml.example config.yaml
```

### Top-Level Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `keep_cycles` | `1` | How many times to cycle through all providers before giving up. Set to `2` or `3` if providers have intermittent failures. |
| `model_name` | `"aifree"` | The model name shown in `/v1/models`. Clients can use this name or any backend model name directly. |
| `show_provider` | `true` | When `true`, responses include a `provider_name` field showing which provider handled the request. Set to `false` to hide this. |

### Server Settings

```yaml
server:
  host: "0.0.0.0"    # Listen address ("0.0.0.0" = all interfaces, "127.0.0.1" = local only)
  port: 8000          # Listen port
  api_key: ""         # Optional: require this key from clients (see "Securing the Proxy")
```

### Provider Settings

Providers are organized into priority groups. Each group has a priority number
and a list of backends:

```yaml
providers:
  - priority: 1           # Tried first
    backends:
      - provider: gemini   # Provider type
        api_key: "..."     # API key for this provider
        model: "gemini-2.5-flash"  # Model to use

  - priority: 2           # Tried if all priority 1 backends fail
    backends:
      - provider: qwen
        api_key: "..."
        model: "qwen-flash"
```

Each backend supports these fields:

| Field | Required | Description |
|-------|----------|-------------|
| `provider` | Yes | Provider type (see [Supported Providers](#supported-providers)) or any name if `base_url` is set. |
| `api_key` | Yes | API key. Supports `${ENV_VAR}` syntax. |
| `model` | Yes | Model identifier to use with this provider. |
| `name` | No | Friendly name for this backend. Shown in logs and in the `provider_name` response field. |
| `base_url` | No | Override the provider's API URL. Required for custom/self-hosted providers. |
| `extra` | No | Provider-specific options (e.g., `timeout`, `default_max_tokens`). |

### Using Environment Variables for API Keys

Instead of putting API keys directly in the config file, you can reference
environment variables:

```yaml
backends:
  - provider: gemini
    api_key: "${GEMINI_API_KEY}"
    model: "gemini-2.5-flash"
```

Then set the variable before starting:

```bash
export GEMINI_API_KEY="your-actual-key"
ai-free-swap --config config.yaml
```

This keeps secrets out of your config file.

### Priority and Fallback

The priority number determines the order providers are tried:

- **Lower numbers are tried first** (priority 1 before priority 2).
- **Within the same priority**, backends are tried in **random order** -- this
  distributes load across multiple accounts or keys.
- If all backends in a priority group fail, the proxy moves to the next group.
- After all groups are exhausted, the cycle repeats up to `keep_cycles` times.

**Example: distribute load across three Gemini keys, fall back to Qwen:**

```yaml
keep_cycles: 2  # Try everything twice before giving up

providers:
  - priority: 1
    backends:
      - provider: gemini
        name: "gemini-key-1"
        api_key: "${GEMINI_KEY_1}"
        model: "gemini-2.5-flash"
      - provider: gemini
        name: "gemini-key-2"
        api_key: "${GEMINI_KEY_2}"
        model: "gemini-2.5-flash"
      - provider: gemini
        name: "gemini-key-3"
        api_key: "${GEMINI_KEY_3}"
        model: "gemini-2.5-flash"

  - priority: 2
    backends:
      - provider: qwen
        api_key: "${QWEN_KEY}"
        model: "qwen-flash"
```

With this setup, a request first tries one of the three Gemini backends (random
order). If all three fail, it tries Qwen. If Qwen also fails, it starts over
(because `keep_cycles: 2`) and tries everything again before returning an error.

### Custom OpenAI-Compatible Providers

Any service that speaks the OpenAI API format can be used. Just set `base_url`
and use any name you like for `provider`:

```yaml
providers:
  - priority: 1
    backends:
      # Groq
      - provider: groq
        api_key: "${GROQ_API_KEY}"
        model: "llama-3.3-70b-versatile"
        base_url: "https://api.groq.com/openai/v1"

      # Together AI
      - provider: together
        api_key: "${TOGETHER_API_KEY}"
        model: "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        base_url: "https://api.together.xyz/v1"

      # Local Ollama
      - provider: ollama
        api_key: "unused"
        model: "llama3.2"
        base_url: "http://localhost:11434/v1"
```

The `provider` name here (groq, together, ollama) is just a label used in logs
and responses -- the proxy uses the `base_url` to know where to send requests.

### Full Configuration Example

```yaml
keep_cycles: 1
model_name: "aifree"
show_provider: true

server:
  host: "0.0.0.0"
  port: 8000
  api_key: ""

providers:
  - priority: 1
    backends:
      - provider: gemini
        name: "gemini-flash-1"
        api_key: "${GEMINI_API_KEY_1}"
        model: "gemini-2.5-flash"
      - provider: gemini
        name: "gemini-flash-2"
        api_key: "${GEMINI_API_KEY_2}"
        model: "gemini-2.5-flash"

  - priority: 2
    backends:
      - provider: qwen
        api_key: "${DASHSCOPE_API_KEY}"
        model: "qwen-flash"

  - priority: 3
    backends:
      - provider: openrouter
        api_key: "${OPENROUTER_API_KEY}"
        model: "google/gemini-2.5-flash:free"
      - provider: openrouter
        api_key: "${OPENROUTER_API_KEY}"
        model: "meta-llama/llama-4-scout:free"

  - priority: 4
    backends:
      - provider: grok
        api_key: "${XAI_API_KEY}"
        model: "grok-3-mini"

  - priority: 5
    backends:
      - provider: anthropic
        api_key: "${ANTHROPIC_API_KEY}"
        model: "claude-sonnet-4-6"
```

---

## Supported Providers

These providers have built-in base URLs and work out of the box -- you only
need an API key:

| Provider | `provider` value | Models (examples) |
|----------|-----------------|-------------------|
| Google Gemini | `gemini` | `gemini-2.5-flash`, `gemini-2.5-flash-lite` |
| Alibaba Qwen | `qwen` | `qwen-flash` |
| OpenRouter | `openrouter` | `google/gemini-2.5-flash:free`, `meta-llama/llama-4-scout:free` |
| xAI Grok | `grok` | `grok-3-mini` |
| OpenAI | `openai` | `gpt-4o-mini`, `gpt-4o` |
| Anthropic | `anthropic` | `claude-sonnet-4-6`, `claude-haiku-4-5` |

**Any other OpenAI-compatible service** can be added by setting `base_url` (see
[Custom OpenAI-Compatible Providers](#custom-openai-compatible-providers)).

---

## API Reference

The proxy exposes an OpenAI-compatible API. Any tool or library that works with
the OpenAI API works with ai-free-swap.

### Chat Completions - `POST /v1/chat/completions`

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

### Responses - `POST /v1/responses`

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

### List Models - `GET /v1/models`

Returns the configured model name.

```json
{
  "object": "list",
  "data": [
    {"id": "aifree", "object": "model", "owned_by": "ai-free-swap"}
  ]
}
```

### Health Check - `GET /health`

Returns `{"status": "ok"}` if the server is running. This endpoint does not
require authentication.

### Streaming

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

### Error Responses

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

---

## Using with Applications

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="unused",  # or your proxy api_key if set
)

response = client.chat.completions.create(
    model="aifree",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

**With streaming:**

```python
stream = client.chat.completions.create(
    model="aifree",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### curl

```bash
# Non-streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "aifree", "messages": [{"role": "user", "content": "Hi"}]}'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "aifree", "messages": [{"role": "user", "content": "Hi"}], "stream": true}'

# With proxy authentication
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-key" \
  -d '{"model": "aifree", "messages": [{"role": "user", "content": "Hi"}]}'
```

### Any OpenAI-Compatible Client

ai-free-swap works as a drop-in replacement for the OpenAI API. In most tools,
you just need to change two settings:

- **API Base URL:** `http://localhost:8000/v1`
- **Model:** `aifree` (or any backend model name you configured)

This works with tools like LangChain, LlamaIndex, Continue, Cursor, Open
Interpreter, and any other application that supports custom OpenAI endpoints.

---

## Command-Line Options

```
ai-free-swap [options]

Options:
  --config, -c PATH      Path to config file (default: config.yaml)
  --host HOST            Override the host from config
  --port PORT            Override the port from config
  --log-level LEVEL      Set log verbosity: debug, info, warning, error
                         (default: info)
```

**Examples:**

```bash
# Use a custom config and port
ai-free-swap -c my-config.yaml --port 9000

# Enable debug logging to see provider routing decisions
ai-free-swap --log-level debug

# Run as a Python module
python -m ai_free_swap --config config.yaml
```

---

## Securing the Proxy

If the proxy is accessible on a network (not just localhost), set an API key:

```yaml
server:
  api_key: "your-secret-proxy-key"
```

Clients must then include this key in requests:

```bash
curl http://your-server:8000/v1/chat/completions \
  -H "Authorization: Bearer your-secret-proxy-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "aifree", "messages": [{"role": "user", "content": "Hi"}]}'
```

The `/health` endpoint is always public (no key required).

---

## Troubleshooting

**"Error loading config"**
- Check that `config.yaml` exists and is valid YAML.
- If using `${ENV_VAR}` in API keys, make sure the environment variables are set.

**"All configured providers failed" (503)**
- Check that your API keys are valid and not expired.
- Run with `--log-level debug` to see which providers were tried and why each
  one failed.
- Increase `keep_cycles` to retry more times.

**"Model 'xyz' is not configured" (400)**
- The model name in your request doesn't match any configured backend model.
- Use `"aifree"` (or your `model_name`) to use any available provider.
- Check your config to see which models are configured.

**Server not reachable**
- If running locally, check that the port isn't already in use.
- If running in Docker, make sure you exposed the port with `-p 8000:8000`.
- If `host` is set to `127.0.0.1`, the server only accepts local connections.
  Change to `0.0.0.0` to accept connections from other machines.

**Slow responses**
- The proxy adds minimal overhead. Slow responses usually mean the provider
  is slow or rate-limiting you.
- Add more providers at the same priority level to distribute load.
- Use `--log-level debug` to see timing per provider.

---

## Running Tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_router.py -v
```
