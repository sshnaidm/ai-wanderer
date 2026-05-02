# ai-free-swap

An OpenAI- and Anthropic-compatible proxy server that routes your requests
through multiple free-tier AI providers. If one provider is down or rate-limited,
the proxy automatically tries the next one -- so your application keeps working
without any code changes.

You configure a list of AI providers (Google Gemini, Qwen, OpenRouter, xAI Grok,
Anthropic, or any OpenAI-compatible service), assign priorities, and the proxy
handles the rest. Your app talks to one local endpoint, and ai-free-swap finds a
working provider behind the scenes. Both OpenAI SDK and Anthropic SDK clients
work out of the box.

## One-Click Deploy

Deploy your own instance directly from this repo -- no fork needed:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/sshnaidm/ai-wanderer)

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/template/from?repoUrl=https://github.com/sshnaidm/ai-wanderer&envs=SERVER_API_KEY,GEMINI_API_KEY&optionalEnvs=GEMINI_API_KEY&SERVER_API_KEYDesc=Secret+key+to+protect+your+proxy+from+unauthorized+access&GEMINI_API_KEYDesc=Your+Google+Gemini+API+key+(get+one+at+aistudio.google.com/apikey))

Both platforms auto-generate a `SERVER_API_KEY` to protect your instance.
Add at least one provider API key (e.g., `GEMINI_API_KEY`) and you're live.

See [Cloud Hosting Guide](docs/HOSTING.md) for more options (Fly.io, VPS,
Oracle Cloud free tier).

---

## Table of Contents

- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
  - [Top-Level Settings](#top-level-settings)
  - [Server Settings](#server-settings)
  - [Provider Settings](#provider-settings)
  - [Environment Variables for API Keys](#environment-variables-for-api-keys)
  - [Priority and Fallback](#priority-and-fallback)
  - [Custom OpenAI-Compatible Providers](#custom-openai-compatible-providers)
  - [Full Configuration Example](#full-configuration-example)
- [Supported Providers](#supported-providers)
- [Using with Applications](#using-with-applications)
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
  |-- POST /v1/messages -->|                              |
  |                         |-- try Gemini (priority 1) ->|
  |                         |<-- error / rate limited -----|
  |                         |                              |
  |                         |-- try Qwen (priority 2) --->|
  |                         |<-- success ------------------|
  |                         |                              |
  |<-- response -----------|                              |
```

1. Your application sends a request to the proxy -- either OpenAI format
   (`/v1/chat/completions`) or Anthropic format (`/v1/messages`).
2. The proxy tries providers in priority order (lowest number = highest priority).
3. Within the same priority level, providers are tried in random order.
4. If a provider fails, the proxy automatically tries the next one.
5. The response is returned in the same format the client used -- your app
   doesn't need to know which provider actually handled the request.

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
| Google Gemini | <https://aistudio.google.com/apikey> |
| Alibaba Qwen | <https://modelstudio.console.alibabacloud.com> (International)  <https://dashscope.console.aliyun.com/> (Chinese) |
| OpenRouter | <https://openrouter.ai/keys> |
| xAI Grok | <https://console.x.ai/> |
| Anthropic | <https://platform.claude.com/settings/workspaces/default/keys/> |
| Groq | <https://console.groq.com/keys> |
| Mistral | <https://admin.mistral.ai/organization/api-keys> |
| Cloudflare AI | <https://dash.cloudflare.com/?to=/:account/ai/workers-ai> |
| Hugging Face | <https://huggingface.co/settings/tokens> |
| Cohere | <https://dashboard.cohere.com/welcome/> |
| Deepseek | <https://platform.deepseek.com/api_keys> |
| Z.ai | <https://z.ai/manage-apikey/apikey-list> |
| Moonshot (Kimi) | <https://platform.kimi.ai/console/api-keys> |
| SiliconFlow | <https://cloud.siliconflow.com/account/ak> |
| MiniMax | <https://platform.minimax.io/user-center/payment/token-plan> |

These are just examples -- any service with an OpenAI-compatible API works
(DeepSeek, GLM, Groq, Together, local Ollama, etc.). See
[Custom OpenAI-Compatible Providers](#custom-openai-compatible-providers).

### 3. Configure

```bash
cp config.yaml.example config.yaml
```

Open `config.yaml` and add your API keys. At minimum, you need one provider:

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

---

## Installation

Requires **Python 3.11 or later**.

### pip (recommended)

```bash
pip install .

# Or in development mode
pip install -e .
```

After installation, the `ai-free-swap` command is available system-wide.

### Docker

```bash
docker build -t ai-free-swap .

docker run -p 8000:8000 \
  -v /path/to/your/config.yaml:/app/config.yaml \
  -e GEMINI_API_KEY="your-key" \
  ai-free-swap
```

### From source

```bash
pip install -r requirements.txt
python -m ai_free_swap --config config.yaml
```

---

## Configuration

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
| `model_routing` | `"any"` | How to handle the model name from client requests. `"any"` (default) ignores the client model and uses all providers in priority order -- best for proxy use cases where clients send arbitrary model names. `"match"` routes to backends whose configured model matches the request, falling back to all providers if no match is found -- useful when you configure multiple distinct models and want clients to choose. |

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
      - provider: gemini   # Provider type (or any label if base_url is set)
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
| `provider` | Yes | Provider type (see [Supported Providers](#supported-providers)) or any name you choose if `base_url` is set. |
| `api_key` | Yes | API key. Supports `${ENV_VAR}` syntax. |
| `model` | Yes | Model identifier to use with this provider. |
| `name` | No | Friendly name for this backend. Shown in logs and in the `provider_name` response field. |
| `base_url` | No | Override the provider's API URL. Required for custom/self-hosted providers. |
| `extra` | No | Provider-specific options (e.g., `timeout`, `default_max_tokens`). |

### Environment Variables for API Keys

Instead of putting API keys directly in the config file, you can reference
environment variables with `${VAR_NAME}` syntax:

```yaml
backends:
  - provider: gemini
    api_key: "${GEMINI_API_KEY}"
    model: "gemini-2.5-flash"
```

The variable names are up to you -- use whatever makes sense:

```yaml
backends:
  - provider: deepseek
    api_key: "${MY_DEEPSEEK_KEY}"
    model: "deepseek-chat"
    base_url: "https://api.deepseek.com/v1"
```

Then set the variables before starting:

```bash
export GEMINI_API_KEY="your-actual-key"
export MY_DEEPSEEK_KEY="your-deepseek-key"
ai-free-swap --config config.yaml
```

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

### Custom OpenAI-Compatible Providers

Any service with an OpenAI-compatible API works. Set `base_url` and use
whatever name you want for `provider` -- the name is just a label for logs:

```yaml
providers:
  - priority: 1
    backends:
      # DeepSeek
      - provider: deepseek
        api_key: "${DEEPSEEK_KEY}"
        model: "deepseek-chat"
        base_url: "https://api.deepseek.com/v1"

      # GLM (Zhipu AI)
      - provider: glm
        api_key: "${GLM_KEY}"
        model: "glm-4-flash"
        base_url: "https://open.bigmodel.cn/api/paas/v4"

      # Groq
      - provider: groq
        api_key: "${GROQ_KEY}"
        model: "llama-3.3-70b-versatile"
        base_url: "https://api.groq.com/openai/v1"

      # Together AI
      - provider: together
        api_key: "${TOGETHER_KEY}"
        model: "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        base_url: "https://api.together.xyz/v1"

      # Local Ollama
      - provider: ollama
        api_key: "unused"
        model: "llama3.2"
        base_url: "http://localhost:11434/v1"

      # LM Studio
      - provider: lmstudio
        api_key: "unused"
        model: "local-model"
        base_url: "http://localhost:1234/v1"
```

### Full Configuration Example

```yaml
keep_cycles: 1
model_name: "aifree"
show_provider: true
model_routing: "any"  # "any" = ignore client model, "match" = route by model name

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

These providers have built-in base URLs and work with just an API key:

| Provider | `provider` value | Models (examples) |
|----------|-----------------|-------------------|
| Google Gemini | `gemini` | `gemini-2.5-flash`, `gemini-2.5-flash-lite` |
| Alibaba Qwen | `qwen` | `qwen-flash` |
| OpenRouter | `openrouter` | `google/gemini-2.5-flash:free`, `meta-llama/llama-4-scout:free` |
| xAI Grok | `grok` | `grok-3-mini` |
| OpenAI | `openai` | `gpt-4o-mini`, `gpt-4o` |
| Anthropic | `anthropic` | `claude-sonnet-4-6`, `claude-haiku-4-5` |

**Any other OpenAI-compatible service** works too -- just set `base_url`.
See [Custom OpenAI-Compatible Providers](#custom-openai-compatible-providers)
for examples with DeepSeek, GLM, Groq, Together, Ollama, LM Studio, and more.

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

### Anthropic Python SDK

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
print(response.content[0].text)
```

**With streaming:**

```python
with client.messages.stream(
    model="aifree",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Tell me a story."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="")
```

**Using `ANTHROPIC_BASE_URL` environment variable:**

Many tools that use the Anthropic SDK (Claude Code, aider, etc.) read the
`ANTHROPIC_BASE_URL` environment variable. Set it to point at your proxy:

```bash
export ANTHROPIC_BASE_URL="http://localhost:8000"
export ANTHROPIC_API_KEY="your-proxy-key"  # or any non-empty string if proxy has no api_key

# Now any tool using the Anthropic SDK will go through your proxy
claude   # Claude Code
aider    # aider with --model claude-sonnet-4-6
```

### curl

```bash
# Non-streaming (OpenAI format)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "aifree", "messages": [{"role": "user", "content": "Hi"}]}'

# Streaming (OpenAI format)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "aifree", "messages": [{"role": "user", "content": "Hi"}], "stream": true}'

# Non-streaming (Anthropic format)
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-proxy-key" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model": "aifree", "max_tokens": 1024, "messages": [{"role": "user", "content": "Hi"}]}'

# With proxy authentication (OpenAI style)
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

This works with tools like aider, cline, open-hands, continue, LangChain,
LlamaIndex, Open Interpreter, and any other application that supports custom
OpenAI endpoints.

### Any Anthropic-Compatible Client

ai-free-swap also works as a drop-in replacement for the Anthropic Messages
API. Set the base URL to your proxy:

- **`ANTHROPIC_BASE_URL`:** `http://localhost:8000`
- **`ANTHROPIC_API_KEY`:** your proxy `api_key` (or any non-empty string)

This works with Claude Code, aider (Anthropic mode), and any other tool that
uses the Anthropic SDK with a configurable base URL.

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

Clients must then include this key in requests. Both authentication methods
are supported:

```bash
# OpenAI-style: Authorization header
curl http://your-server:8000/v1/chat/completions \
  -H "Authorization: Bearer your-secret-proxy-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "aifree", "messages": [{"role": "user", "content": "Hi"}]}'

# Anthropic-style: x-api-key header
curl http://your-server:8000/v1/messages \
  -H "x-api-key: your-secret-proxy-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "aifree", "max_tokens": 1024, "messages": [{"role": "user", "content": "Hi"}]}'
```

The `/health` endpoint is always public (no key required).

---

## Troubleshooting

**"Error loading config"** -- Check that `config.yaml` exists and is valid
YAML. If using `${ENV_VAR}` syntax, make sure the environment variables are set.

**"All configured providers failed" (503)** -- Check that your API keys are
valid. Run with `--log-level debug` to see which providers were tried and why
each failed. Increase `keep_cycles` to retry more times.

**"Model 'xyz' is not configured" (400)** -- The model name in your request
doesn't match any configured backend. Use `"aifree"` to use any available
provider, or check your config for the exact model names.

**Server not reachable** -- Check the port isn't already in use. In Docker,
make sure you used `-p 8000:8000`. If `host` is `127.0.0.1`, the server only
accepts local connections -- change to `0.0.0.0`.

---

## Further Reading

- [API Reference](docs/API.md) -- full endpoint documentation, request/response
  formats, streaming protocol, error codes
- [Cloud Hosting Guide](docs/HOSTING.md) -- deploy to Render, Railway, Fly.io,
  or any VPS with step-by-step instructions

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
