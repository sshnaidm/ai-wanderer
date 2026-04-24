# Cloud Hosting

You can host ai-free-swap on a public server so it's accessible from anywhere.
No fork needed -- these platforms deploy directly from the upstream repository.

A `SERVER_API_KEY` is always required for cloud deployments to prevent
unauthorized access to your proxy (and your provider API keys behind it).

---

## Table of Contents

- [Render (Easiest)](#render-easiest)
- [Railway](#railway)
- [Fly.io](#flyio)
- [Any VPS or VM](#any-vps-or-vm)
- [Configuring Providers](#configuring-providers)
- [Custom Config on Cloud Platforms](#custom-config-on-cloud-platforms)
- [Quick Comparison](#quick-comparison)

---

## Render (Easiest)

Render has a free tier. The service sleeps after inactivity (~30 second
cold start on wake).

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/sshnaidm/ai-wanderer)

**Or manually:**

1. Go to [render.com](https://render.com) → **New** → **Web Service** →
   **Public Git Repository**
2. Paste `https://github.com/sshnaidm/ai-wanderer`
3. Render detects the Dockerfile and `render.yaml` automatically
4. `SERVER_API_KEY` is auto-generated for you -- find it in the **Environment** tab
5. Add your provider API keys as environment variables (any names you want --
   they just need to match what's in the config)
6. Click **Deploy**

Your proxy URL will be `https://your-service.onrender.com`.

---

## Railway

Railway gives you $5 free trial credit (enough for months since the proxy
only uses CPU when handling requests).

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/template/from?repoUrl=https://github.com/sshnaidm/ai-wanderer&envs=SERVER_API_KEY,GEMINI_API_KEY&optionalEnvs=GEMINI_API_KEY&SERVER_API_KEYDesc=Secret+key+to+protect+your+proxy+from+unauthorized+access&GEMINI_API_KEYDesc=Your+Google+Gemini+API+key+(get+one+at+aistudio.google.com/apikey))

1. Click the button above
2. Set `SERVER_API_KEY` to any secret string (this becomes your proxy password)
3. Add your provider API keys
4. Click **Deploy**

Your proxy URL will be shown in the Railway dashboard under
**Settings** → **Networking** → **Generate Domain**.

---

## Fly.io

Fly.io gives a free allowance for small apps. Requires a one-time CLI setup.

1. Install flyctl: `curl -L https://fly.io/install.sh | sh`
2. Sign up: `fly auth signup` (or `fly auth login`)
3. Clone and launch:

   ```bash
   git clone https://github.com/sshnaidm/ai-wanderer.git
   cd ai-wanderer
   fly launch
   ```

4. Set your secrets:

   ```bash
   fly secrets set SERVER_API_KEY="pick-a-secret" GEMINI_API_KEY="your-key"
   ```

5. Deploy: `fly deploy`

---

## Any VPS or VM

Works on any Linux server (AWS, GCP, DigitalOcean, Oracle Cloud free tier,
a Raspberry Pi, etc.).

```bash
git clone https://github.com/sshnaidm/ai-wanderer.git
cd ai-wanderer

# Option A: Docker
docker build -t ai-free-swap .
docker run -d --restart=always -p 8000:8000 \
  -e SERVER_API_KEY="pick-a-secret" \
  -e GEMINI_API_KEY="your-key" \
  ai-free-swap

# Option B: Direct
pip install .
export SERVER_API_KEY="pick-a-secret"
export GEMINI_API_KEY="your-key"
ai-free-swap --config config.yaml.cloud
```

**Oracle Cloud** offers an always-free VM (4 ARM cores, 24GB RAM) that's
more than enough to run this proxy permanently at zero cost. Sign up at
[cloud.oracle.com](https://cloud.oracle.com), create a free Compute Instance,
and follow the steps above.

---

## Configuring Providers

The default cloud config (`config.yaml.cloud`) ships with a single Gemini
backend as a starting point. You can use **any provider** and **any
environment variable names** -- the variable names just need to match between
your config and what you set in the hosting platform.

For example, if you want to use DeepSeek and GLM alongside Gemini:

```yaml
providers:
  - priority: 1
    backends:
      - provider: gemini
        api_key: "${MY_GEMINI_KEY}"
        model: "gemini-2.5-flash"
      - provider: deepseek
        api_key: "${DEEPSEEK_KEY}"
        model: "deepseek-chat"
        base_url: "https://api.deepseek.com/v1"

  - priority: 2
    backends:
      - provider: glm
        api_key: "${GLM_KEY}"
        model: "glm-4-flash"
        base_url: "https://open.bigmodel.cn/api/paas/v4"
```

Then set `MY_GEMINI_KEY`, `DEEPSEEK_KEY`, and `GLM_KEY` as environment
variables in your hosting platform. The names are entirely up to you.

Any service with an OpenAI-compatible API works -- just set `base_url`.
The `provider` field is just a label for your own reference in logs.

---

## Custom Config on Cloud Platforms

The Docker image ships with a minimal default config. To use a full custom
config, you have two options:

### Option 1: Mount a config file

If your platform supports volume mounts (Docker, VPS, Fly.io):

```bash
docker run -p 8000:8000 \
  -v /path/to/your/config.yaml:/app/config.yaml \
  -e SERVER_API_KEY="secret" \
  -e MY_GEMINI_KEY="key1" \
  -e DEEPSEEK_KEY="key2" \
  ai-free-swap
```

### Option 2: Build a custom image

Create your own `Dockerfile` that copies your config:

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir .
COPY my-config.yaml /app/config.yaml
EXPOSE 8000
ENTRYPOINT ["ai-free-swap"]
CMD ["--config", "/app/config.yaml"]
```

---

## Quick Comparison

| Platform | Difficulty | Cost | Always On | Setup Time |
|----------|-----------|------|-----------|------------|
| Render | Very easy | Free tier available | No (sleeps) | 2 min |
| Railway | Very easy | Free trial, then ~$5/mo | Yes | 2 min |
| Fly.io | Easy | Free tier | Yes | 10 min |
| VPS (Oracle) | Medium | Free forever | Yes | 30 min |
| VPS (other) | Medium | ~$5/mo | Yes | 30 min |
