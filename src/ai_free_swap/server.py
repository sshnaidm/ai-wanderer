from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from .config import AppConfig
from .models import (
    AnthropicMessagesRequest,
    ChatCompletionRequest,
    ResponsesRequest,
    make_anthropic_error_response,
    make_anthropic_response,
    make_completion_response,
    make_error_response,
    make_responses_response,
    message_to_response_output,
    make_stream_chunk,
)
from .router import (
    AllProvidersFailedError,
    NoMatchingProvidersError,
    PreparedStream,
    Router,
    StreamingProviderError,
)

logger = logging.getLogger(__name__)


def create_app(config: AppConfig, *, state_file: str | None = None) -> FastAPI:
    router = Router(config, state_file=state_file)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            router.save_state()

    app = FastAPI(title="ai-free-swap", lifespan=lifespan)
    server_api_key = config.server.api_key
    model_name = config.model_name
    show_provider = config.show_provider

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if request.url.path in ("/", "/health", "/dashboard", "/dashboard/data", "/favicon.ico"):
            return await call_next(request)
        if server_api_key:
            token = _extract_bearer_token(request.headers.get("authorization", ""))
            if not token:
                token = request.headers.get("x-api-key", "").strip()
            if token != server_api_key:
                return _error_response(401, "Invalid API key", "auth_error")
        return await call_next(request)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        request_id = uuid.uuid4().hex[:8]
        logger.debug(
            "[%s] POST /v1/chat/completions model=%s stream=%s",
            request_id,
            request.model,
            request.stream,
        )
        messages = request.to_messages()
        kwargs = request.to_model_kwargs()

        if request.stream:
            try:
                prepared_stream = await router.prepare_stream(
                    messages,
                    requested_model=request.model,
                    request_id=request_id,
                    **kwargs,
                )
            except NoMatchingProvidersError as e:
                return _error_response(
                    400,
                    f"Model {e.requested_model!r} is not configured",
                    "invalid_request_error",
                    code="model_not_found",
                )
            except AllProvidersFailedError as e:
                logger.warning("[%s] All providers failed before stream start", request_id)
                logger.debug("[%s] Provider failure details: %s", request_id, e.detail_summary)
                return _error_response(
                    503,
                    "All configured providers failed",
                    "server_error",
                    code="all_providers_failed",
                )

            return EventSourceResponse(_stream_response(prepared_stream, show_provider))

        try:
            result = await router.route(
                messages,
                requested_model=request.model,
                request_id=request_id,
                **kwargs,
            )
        except NoMatchingProvidersError as e:
            return _error_response(
                400,
                f"Model {e.requested_model!r} is not configured",
                "invalid_request_error",
                code="model_not_found",
            )
        except AllProvidersFailedError as e:
            logger.warning("[%s] All providers failed", request_id)
            logger.debug("[%s] Provider failure details: %s", request_id, e.detail_summary)
            return _error_response(
                503,
                "All configured providers failed",
                "server_error",
                code="all_providers_failed",
            )

        if result.raw_response is not None:
            raw = result.raw_response
            if show_provider:
                raw = {**raw, "provider_name": result.display_name}
            return raw

        logger.debug(
            "[%s] Response from %s: %s",
            request_id,
            result.provider_name,
            result.content[:200],
        )
        resp = make_completion_response(
            result.content,
            result.model,
            message=result.message,
        )
        if show_provider:
            return {**resp.model_dump(), "provider_name": result.display_name}
        return resp

    @app.post("/v1/responses")
    async def responses(request: ResponsesRequest):
        request_id = uuid.uuid4().hex[:8]
        logger.debug(
            "[%s] POST /v1/responses model=%s stream=%s",
            request_id,
            request.model,
            request.stream,
        )
        messages = request.to_messages()
        kwargs = request.to_model_kwargs()

        if request.stream:
            try:
                prepared_stream = await router.prepare_stream(
                    messages,
                    requested_model=request.model,
                    request_id=request_id,
                    **kwargs,
                )
            except NoMatchingProvidersError as e:
                return _error_response(
                    400,
                    f"Model {e.requested_model!r} is not configured",
                    "invalid_request_error",
                    code="model_not_found",
                )
            except AllProvidersFailedError as e:
                logger.warning("[%s] All providers failed before stream start", request_id)
                logger.debug("[%s] Provider failure details: %s", request_id, e.detail_summary)
                return _error_response(
                    503,
                    "All configured providers failed",
                    "server_error",
                    code="all_providers_failed",
                )

            return EventSourceResponse(_responses_stream(prepared_stream, show_provider))

        try:
            result = await router.route(
                messages,
                requested_model=request.model,
                request_id=request_id,
                **kwargs,
            )
        except NoMatchingProvidersError as e:
            return _error_response(
                400,
                f"Model {e.requested_model!r} is not configured",
                "invalid_request_error",
                code="model_not_found",
            )
        except AllProvidersFailedError as e:
            logger.warning("[%s] All providers failed", request_id)
            logger.debug("[%s] Provider failure details: %s", request_id, e.detail_summary)
            return _error_response(
                503,
                "All configured providers failed",
                "server_error",
                code="all_providers_failed",
            )

        response_id = f"resp_{uuid.uuid4().hex[:24]}"
        resp = make_responses_response(
            result.content,
            result.model,
            response_id,
            message=result.message,
        )
        if show_provider:
            resp["provider_name"] = result.display_name
        return resp

    @app.post("/v1/messages")
    async def anthropic_messages(request: AnthropicMessagesRequest):
        request_id = uuid.uuid4().hex[:8]
        logger.debug(
            "[%s] POST /v1/messages model=%s stream=%s",
            request_id,
            request.model,
            request.stream,
        )
        messages = request.to_messages()
        kwargs = request.to_model_kwargs()

        if request.stream:
            try:
                prepared_stream = await router.prepare_stream(
                    messages,
                    requested_model=request.model,
                    request_id=request_id,
                    **kwargs,
                )
            except NoMatchingProvidersError as e:
                return _anthropic_error_response(
                    400,
                    f"Model {e.requested_model!r} is not configured",
                    "not_found_error",
                )
            except AllProvidersFailedError as e:
                logger.warning("[%s] All providers failed before stream start", request_id)
                logger.debug("[%s] Provider failure details: %s", request_id, e.detail_summary)
                return _anthropic_error_response(
                    529,
                    "All configured providers failed",
                    "overloaded_error",
                )

            return EventSourceResponse(_anthropic_stream_response(prepared_stream, show_provider))

        try:
            result = await router.route(
                messages,
                requested_model=request.model,
                request_id=request_id,
                **kwargs,
            )
        except NoMatchingProvidersError as e:
            return _anthropic_error_response(
                400,
                f"Model {e.requested_model!r} is not configured",
                "not_found_error",
            )
        except AllProvidersFailedError as e:
            logger.warning("[%s] All providers failed", request_id)
            logger.debug("[%s] Provider failure details: %s", request_id, e.detail_summary)
            return _anthropic_error_response(
                529,
                "All configured providers failed",
                "overloaded_error",
            )

        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        resp = make_anthropic_response(
            result.content,
            result.model,
            msg_id,
            message=result.message,
        )
        if show_provider:
            resp["provider_name"] = result.display_name
        return resp

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "ai-free-swap"}],
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        return DASHBOARD_HTML

    @app.get("/dashboard/data")
    async def dashboard_data():
        return router.dashboard_snapshot()

    @app.api_route("/", methods=["GET", "HEAD"])
    async def root():
        return {"status": "ok"}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return JSONResponse(status_code=204, content=None)

    return app


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ai-free-swap dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f7fb;
      --panel: #ffffff;
      --panel-soft: #f8fafc;
      --text: #17202e;
      --muted: #667085;
      --line: #d9e1ea;
      --shadow: 0 10px 30px rgba(16, 24, 40, 0.07);
      --green: #137a46;
      --green-soft: #daf7e6;
      --amber: #9a6200;
      --amber-soft: #fff1c7;
      --red: #b42318;
      --red-soft: #ffe0dc;
      --blue: #235dd8;
      --blue-soft: #dfe9ff;
      --cyan: #087a8f;
      --cyan-soft: #d8f4f8;
      --gray-soft: #edf1f6;
    }
    :root[data-theme="dark"] {
      color-scheme: dark;
      --bg: #10141b;
      --panel: #171d26;
      --panel-soft: #1e2631;
      --text: #eef3f8;
      --muted: #9ba8b7;
      --line: #2b3542;
      --shadow: 0 14px 40px rgba(0, 0, 0, 0.28);
      --green: #62d58d;
      --green-soft: #173523;
      --amber: #f7c45b;
      --amber-soft: #3b2b0f;
      --red: #ff8a80;
      --red-soft: #3e1c1d;
      --blue: #8db2ff;
      --blue-soft: #1c2b4f;
      --cyan: #64d7e6;
      --cyan-soft: #15343b;
      --gray-soft: #273140;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }
    .page { max-width: 1360px; margin: 0 auto; padding: 22px; }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }
    h1 { margin: 0; font-size: 26px; line-height: 1.1; letter-spacing: 0; }
    .sub { color: var(--muted); margin-top: 6px; }
    .top-actions { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; justify-content: flex-end; }
    .status-line { color: var(--muted); text-align: right; min-width: 220px; }
    .theme-switch {
      display: inline-flex;
      padding: 3px;
      gap: 3px;
      background: var(--panel-soft);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .theme-switch button {
      border: 0;
      border-radius: 6px;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      font: inherit;
      min-height: 30px;
      padding: 5px 10px;
    }
    .theme-switch button.active {
      background: var(--panel);
      color: var(--text);
      box-shadow: 0 1px 4px rgba(16, 24, 40, 0.12);
    }
    .kpis {
      display: grid;
      grid-template-columns: repeat(5, minmax(140px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .kpi { padding: 14px; min-height: 88px; }
    .kpi .label { color: var(--muted); font-size: 12px; text-transform: uppercase; }
    .kpi .value { font-size: 28px; font-weight: 750; margin-top: 8px; line-height: 1; }
    .kpi .hint { color: var(--muted); margin-top: 8px; font-size: 12px; }
    .main-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 16px;
      align-items: start;
    }
    .section-title {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      font-weight: 700;
    }
    .priority-groups { display: grid; grid-template-columns: 1fr; gap: 12px; padding: 14px; }
    .priority-group {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: var(--panel-soft);
    }
    .priority-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 11px 12px;
      border-left: 4px solid var(--blue);
      background: var(--panel);
    }
    .priority-title { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
    .priority-title strong { font-size: 15px; }
    .priority-note { color: var(--muted); font-size: 12px; }
    .models-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(245px, 1fr));
      gap: 10px;
      padding: 10px;
    }
    .model-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }
    .model-top { display: flex; justify-content: space-between; gap: 10px; align-items: flex-start; }
    .backend-name { font-weight: 700; }
    .meta { color: var(--muted); font-size: 12px; margin-top: 2px; }
    .model-name { color: var(--text); font-size: 13px; overflow-wrap: anywhere; margin-top: 7px; }
    .stats {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 12px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: var(--panel-soft);
      min-width: 0;
    }
    .stat span { display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; }
    .stat strong { display: block; margin-top: 5px; font-size: 15px; overflow-wrap: anywhere; }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .priority-pill { color: var(--blue); background: var(--blue-soft); }
    .healthy { color: var(--green); background: var(--green-soft); }
    .running { color: var(--blue); background: var(--blue-soft); }
    .failing { color: var(--red); background: var(--red-soft); }
    .limited { color: var(--amber); background: var(--amber-soft); }
    .idle { color: var(--muted); background: var(--gray-soft); }
    .bar {
      width: 100%;
      height: 8px;
      background: var(--gray-soft);
      border-radius: 999px;
      overflow: hidden;
      margin-top: 6px;
    }
    .bar > span { display: block; height: 100%; background: var(--green); border-radius: 999px; }
    .failbar > span { background: var(--red); }
    .tokenbar > span { background: var(--cyan); }
    .spark-row { padding: 12px 14px; border-bottom: 1px solid var(--line); }
    .spark-row:last-child { border-bottom: 0; }
    .spark-head { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 7px; }
    .spark-label { font-weight: 700; overflow-wrap: anywhere; }
    .spark-value { color: var(--muted); white-space: nowrap; }
    .stack {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }
    .error-text { color: var(--red); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-top: 10px; }
    .empty { padding: 18px; color: var(--muted); }
    @media (max-width: 1100px) {
      .kpis { grid-template-columns: repeat(3, minmax(120px, 1fr)); }
      .main-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 720px) {
      .page { padding: 12px; }
      header { align-items: flex-start; flex-direction: column; }
      .top-actions { justify-content: flex-start; }
      .status-line { text-align: left; min-width: 0; }
      .kpis { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .models-grid { grid-template-columns: 1fr; }
      .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="page">
    <header>
      <div>
        <h1>ai-free-swap</h1>
        <div class="sub" id="subtitle">Dashboard loading</div>
      </div>
      <div class="top-actions">
        <div class="theme-switch" aria-label="Theme">
          <button type="button" data-theme-choice="light">Light</button>
          <button type="button" data-theme-choice="dark">Dark</button>
        </div>
        <div class="status-line" id="statusLine">Waiting for data</div>
      </div>
    </header>

    <section class="kpis">
      <div class="panel kpi"><div class="label">Active</div><div class="value" id="kActive">0</div><div class="hint">running provider calls</div></div>
      <div class="panel kpi"><div class="label">Success</div><div class="value" id="kSuccess">-</div><div class="hint" id="kSuccessHint">0 attempts</div></div>
      <div class="panel kpi"><div class="label">Failures</div><div class="value" id="kFailures">0</div><div class="hint">provider failures</div></div>
      <div class="panel kpi"><div class="label">Rate Limited</div><div class="value" id="kLimited">0</div><div class="hint">skipped attempts</div></div>
      <div class="panel kpi"><div class="label">Backends</div><div class="value" id="kBackends">0</div><div class="hint">configured providers</div></div>
    </section>

    <main class="main-grid">
      <section class="panel">
        <div class="section-title">
          <span>Models by Priority</span>
          <span id="modelCount" class="meta"></span>
        </div>
        <div class="priority-groups" id="priorityGroups"></div>
      </section>

      <aside class="stack">
        <section class="panel">
          <div class="section-title">Reliability</div>
          <div id="successBars"></div>
        </section>
        <section class="panel">
          <div class="section-title">Token Share</div>
          <div id="tokenBars"></div>
        </section>
        <section class="panel">
          <div class="section-title">Rate Limit Windows</div>
          <div id="windowBars"></div>
        </section>
      </aside>
    </main>
  </div>

  <script>
    const THEME_KEY = "ai-free-swap-theme";
    const fmt = new Intl.NumberFormat();
    const pct = value => value === null || value === undefined ? "-" : `${value.toFixed(1)}%`;
    const num = value => fmt.format(Math.round(value || 0));
    const ms = value => value === null || value === undefined ? "-" : `${Math.round(value)} ms`;
    const when = seconds => seconds ? new Date(seconds * 1000).toLocaleTimeString() : "-";
    const esc = value => String(value ?? "").replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

    function statusClass(status) {
      return ["healthy", "running", "failing", "limited", "idle"].includes(status) ? status : "idle";
    }

    function setTheme(theme) {
      const selected = theme === "dark" ? "dark" : "light";
      document.documentElement.dataset.theme = selected;
      localStorage.setItem(THEME_KEY, selected);
      document.querySelectorAll("[data-theme-choice]").forEach(button => {
        button.classList.toggle("active", button.dataset.themeChoice === selected);
      });
    }

    function bar(width, cls = "") {
      const safeWidth = Math.max(0, Math.min(100, width || 0));
      return `<div class="bar ${cls}"><span style="width:${safeWidth}%"></span></div>`;
    }

    function renderSpark(container, rows, valueFn, labelFn, cls = "") {
      const max = Math.max(1, ...rows.map(valueFn));
      container.innerHTML = rows.map(item => {
        const value = valueFn(item);
        return `<div class="spark-row">
          <div class="spark-head"><span class="spark-label">${esc(labelFn(item))}</span><span class="spark-value">${num(value)}</span></div>
          ${bar(value / max * 100, cls)}
        </div>`;
      }).join("") || `<div class="empty">No data yet</div>`;
    }

    function groupByPriority(backends) {
      const groups = new Map();
      for (const item of backends) {
        const key = item.priority ?? 999;
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(item);
      }
      return [...groups.entries()].sort((a, b) => a[0] - b[0]);
    }

    function renderModelCard(item) {
      const requestCount = item.successes + item.failures;
      const successWidth = item.success_rate === null ? 0 : item.success_rate;
      return `<article class="model-card">
        <div class="model-top">
          <div>
            <div class="backend-name">${esc(item.label)}</div>
            <div class="meta">${esc(item.provider)}${item.name ? ` | ${esc(item.name)}` : ""}</div>
          </div>
          <span class="pill ${statusClass(item.status)}">${esc(item.status)}</span>
        </div>
        <div class="model-name">${esc(item.model)}</div>
        ${bar(successWidth)}
        <div class="stats">
          <div class="stat"><span>Success</span><strong>${pct(item.success_rate)}</strong></div>
          <div class="stat"><span>Requests</span><strong>${num(requestCount)}</strong></div>
          <div class="stat"><span>Latency</span><strong>${ms(item.avg_latency_ms)}</strong></div>
          <div class="stat"><span>Active</span><strong>${num(item.active)}</strong></div>
          <div class="stat"><span>Tokens</span><strong>${num(item.total_tokens)}</strong></div>
          <div class="stat"><span>Limited</span><strong>${num(item.rate_limited_skips)}</strong></div>
        </div>
        ${item.last_error ? `<div class="error-text" title="${esc(item.last_error)}">${esc(item.last_error)}</div>` : ""}
      </article>`;
    }

    function renderPriorityGroups(backends) {
      const groups = groupByPriority(backends);
      document.getElementById("priorityGroups").innerHTML = groups.map(([priority, items], index) => {
        const active = items.reduce((total, item) => total + item.active, 0);
        const attempts = items.reduce((total, item) => total + item.attempts, 0);
        const successes = items.reduce((total, item) => total + item.successes, 0);
        const rate = attempts ? successes / attempts * 100 : null;
        return `<section class="priority-group">
          <div class="priority-head">
            <div class="priority-title">
              <span class="pill priority-pill">Priority ${priority}</span>
              <strong>${index === 0 ? "Primary route" : "Fallback route"}</strong>
              <span class="priority-note">${num(items.length)} backend${items.length === 1 ? "" : "s"}</span>
            </div>
            <div class="priority-note">${active ? `${num(active)} active | ` : ""}${pct(rate)} success</div>
          </div>
          <div class="models-grid">${items.map(renderModelCard).join("")}</div>
        </section>`;
      }).join("") || `<div class="empty">No backends configured</div>`;
    }

    function render(data) {
      const totals = data.totals;
      const backends = data.backends;
      document.getElementById("subtitle").textContent = `${data.model_name} | routing: ${data.model_routing} | cycles: ${data.keep_cycles}`;
      document.getElementById("statusLine").textContent = `Updated ${when(data.generated_at)} | started ${when(data.started_at)}`;
      document.getElementById("kActive").textContent = num(totals.active);
      document.getElementById("kSuccess").textContent = pct(totals.success_rate);
      document.getElementById("kSuccessHint").textContent = `${num(totals.attempts)} attempts`;
      document.getElementById("kFailures").textContent = num(totals.failures);
      document.getElementById("kLimited").textContent = num(totals.rate_limited_skips);
      document.getElementById("kBackends").textContent = num(totals.backends);
      document.getElementById("modelCount").textContent = `${num(backends.length)} models`;
      renderPriorityGroups(backends);

      const activeFirst = [...backends].sort((a, b) => (b.success_rate || 0) - (a.success_rate || 0));
      renderSpark(document.getElementById("successBars"), activeFirst, item => item.success_rate || 0, item => item.label);
      renderSpark(document.getElementById("tokenBars"), backends, item => item.total_tokens, item => item.label, "tokenbar");
      renderSpark(document.getElementById("windowBars"), backends, item => {
        const counters = item.rate_counters.requests || {};
        return Math.max(0, ...Object.values(counters));
      }, item => item.label, "failbar");
    }

    async function refresh() {
      try {
        const response = await fetch("/dashboard/data", { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        render(await response.json());
      } catch (error) {
        document.getElementById("statusLine").textContent = `Dashboard error: ${error.message}`;
      }
    }

    document.querySelectorAll("[data-theme-choice]").forEach(button => {
      button.addEventListener("click", () => setTheme(button.dataset.themeChoice));
    });
    setTheme(localStorage.getItem(THEME_KEY) || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));

    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>
"""


def _extract_bearer_token(auth_header: str) -> str:
    value = auth_header.strip()
    if not value:
        return ""
    scheme, _, token = value.partition(" ")
    if token and scheme.lower() == "bearer":
        return token.strip()
    return value


def _error_response(
    status_code: int,
    message: str,
    error_type: str,
    *,
    code: str | None = None,
) -> JSONResponse:
    content = make_error_response(message, error_type, code=code)
    logger.debug("Error response %d: %s", status_code, content)
    return JSONResponse(status_code=status_code, content=content)


async def _stream_response(
    prepared_stream: PreparedStream,
    show_provider: bool = True,
) -> AsyncGenerator[dict[str, str], None]:
    chat_request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    provider_fields = {"provider_name": prepared_stream.display_name} if show_provider else {}

    if not prepared_stream.raw_chunks:
        chunk_data = make_stream_chunk(
            None,
            chat_request_id,
            prepared_stream.model,
            role="assistant",
        )
        yield {"data": json.dumps({**chunk_data, **provider_fields})}

    try:
        async for chunk in prepared_stream.chunks:
            if isinstance(chunk, dict):
                payload = dict(chunk)
                payload.setdefault("model", prepared_stream.model)
                payload.update(provider_fields)
                yield {"data": json.dumps(payload)}
                continue
            chunk_data = make_stream_chunk(
                chunk,
                chat_request_id,
                prepared_stream.model,
            )
            yield {"data": json.dumps({**chunk_data, **provider_fields})}
    except StreamingProviderError as e:
        logger.error("[%s] %s", prepared_stream.request_id, e)
        if not prepared_stream.raw_chunks:
            chunk_data = make_stream_chunk(
                None,
                chat_request_id,
                prepared_stream.model,
                finish_reason="error",
            )
            yield {"data": json.dumps({**chunk_data, **provider_fields})}
    else:
        if not prepared_stream.raw_chunks:
            chunk_data = make_stream_chunk(
                None,
                chat_request_id,
                prepared_stream.model,
                finish_reason="stop",
            )
            yield {"data": json.dumps({**chunk_data, **provider_fields})}
    yield {"data": "[DONE]"}


async def _responses_stream(
    prepared_stream: PreparedStream,
    show_provider: bool = True,
) -> AsyncGenerator[dict[str, str], None]:
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    seq = 0

    def _event(event_type: str, payload: dict) -> dict[str, str]:
        nonlocal seq
        seq += 1
        payload["type"] = event_type
        payload.setdefault("sequence_number", seq)
        return {"event": event_type, "data": json.dumps(payload)}

    yield _event(
        "response.created",
        {
            "response": {
                "id": response_id,
                "object": "response",
                "status": "in_progress",
                "model": prepared_stream.model,
                "output": [],
            },
        },
    )

    yield _event(
        "response.output_item.added",
        {
            "output_index": 0,
            "item": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
        },
    )

    yield _event(
        "response.content_part.added",
        {
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": ""},
        },
    )

    full_text: list[str] = []
    status = "completed"
    try:
        async for chunk in prepared_stream.chunks:
            text = _extract_stream_text(chunk)
            if not text:
                continue
            full_text.append(text)
            yield _event(
                "response.output_text.delta",
                {
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": msg_id,
                    "delta": text,
                },
            )
    except StreamingProviderError as e:
        logger.error("[%s] %s", prepared_stream.request_id, e)
        status = "incomplete"

    joined = "".join(full_text)

    yield _event(
        "response.output_text.done",
        {
            "output_index": 0,
            "content_index": 0,
            "item_id": msg_id,
            "text": joined,
        },
    )

    yield _event(
        "response.content_part.done",
        {
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": joined},
        },
    )

    yield _event(
        "response.output_item.done",
        {
            "output_index": 0,
            "item": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "status": status,
                "content": [{"type": "output_text", "text": joined}],
            },
        },
    )

    final_resp = make_responses_response(
        joined,
        prepared_stream.model,
        response_id,
        status=status,
    )
    if show_provider:
        final_resp["provider_name"] = prepared_stream.display_name
    yield _event("response.completed", {"response": final_resp})

    yield {"data": "[DONE]"}


def _anthropic_error_response(
    status_code: int,
    message: str,
    error_type: str,
) -> JSONResponse:
    content = make_anthropic_error_response(message, error_type)
    logger.debug("Anthropic error response %d: %s", status_code, content)
    return JSONResponse(status_code=status_code, content=content)


async def _anthropic_stream_response(
    prepared_stream: PreparedStream,
    show_provider: bool = True,
) -> AsyncGenerator[dict[str, str], None]:
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    provider_fields = {"provider_name": prepared_stream.display_name} if show_provider else {}

    message_obj = {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": prepared_stream.model,
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
        **provider_fields,
    }
    yield {
        "event": "message_start",
        "data": json.dumps({"type": "message_start", "message": message_obj}),
    }

    next_block = 0
    text_block_index: int | None = None
    text_block_closed = False
    tool_block_indices: dict[int, int] = {}
    has_tool_calls = False
    stop_reason = "end_turn"

    try:
        async for chunk in prepared_stream.chunks:
            text, tool_calls = _extract_stream_parts(chunk)

            if text and not text_block_closed:
                if text_block_index is None:
                    text_block_index = next_block
                    next_block += 1
                    yield {
                        "event": "content_block_start",
                        "data": json.dumps(
                            {
                                "type": "content_block_start",
                                "index": text_block_index,
                                "content_block": {"type": "text", "text": ""},
                            }
                        ),
                    }
                yield {
                    "event": "content_block_delta",
                    "data": json.dumps(
                        {
                            "type": "content_block_delta",
                            "index": text_block_index,
                            "delta": {"type": "text_delta", "text": text},
                        }
                    ),
                }

            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_idx = tc.get("index", 0)

                if tc_idx not in tool_block_indices:
                    if text_block_index is not None and not text_block_closed:
                        yield {
                            "event": "content_block_stop",
                            "data": json.dumps(
                                {
                                    "type": "content_block_stop",
                                    "index": text_block_index,
                                }
                            ),
                        }
                        text_block_closed = True

                    func = tc.get("function", {})
                    tool_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                    block_idx = next_block
                    next_block += 1
                    tool_block_indices[tc_idx] = block_idx
                    has_tool_calls = True

                    yield {
                        "event": "content_block_start",
                        "data": json.dumps(
                            {
                                "type": "content_block_start",
                                "index": block_idx,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tool_id,
                                    "name": func.get("name", ""),
                                    "input": {},
                                },
                            }
                        ),
                    }

                    args = func.get("arguments", "")
                    if args:
                        yield {
                            "event": "content_block_delta",
                            "data": json.dumps(
                                {
                                    "type": "content_block_delta",
                                    "index": block_idx,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": args,
                                    },
                                }
                            ),
                        }
                else:
                    block_idx = tool_block_indices[tc_idx]
                    func = tc.get("function", {})
                    args = func.get("arguments", "")
                    if args:
                        yield {
                            "event": "content_block_delta",
                            "data": json.dumps(
                                {
                                    "type": "content_block_delta",
                                    "index": block_idx,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": args,
                                    },
                                }
                            ),
                        }
    except StreamingProviderError as e:
        logger.error("[%s] %s", prepared_stream.request_id, e)
        stop_reason = "error"

    if text_block_index is not None and not text_block_closed:
        yield {
            "event": "content_block_stop",
            "data": json.dumps({"type": "content_block_stop", "index": text_block_index}),
        }

    for tc_idx in sorted(tool_block_indices):
        yield {
            "event": "content_block_stop",
            "data": json.dumps(
                {
                    "type": "content_block_stop",
                    "index": tool_block_indices[tc_idx],
                }
            ),
        }

    if has_tool_calls:
        stop_reason = "tool_use"

    yield {
        "event": "message_delta",
        "data": json.dumps(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 0},
            }
        ),
    }

    yield {
        "event": "message_stop",
        "data": json.dumps({"type": "message_stop"}),
    }


def _extract_stream_text(chunk: str | dict[str, Any]) -> str:
    if isinstance(chunk, str):
        return chunk
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return ""
    text_parts: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            output_item, output_text = message_to_response_output(
                {"role": delta.get("role", "assistant"), "content": content}
            )
            if output_item and output_text:
                text_parts.append(output_text)
    return "".join(text_parts)


def _extract_stream_parts(
    chunk: str | dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(chunk, str):
        return chunk, []
    if not isinstance(chunk, dict):
        return "", []
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return "", []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            _, output_text = message_to_response_output({"role": delta.get("role", "assistant"), "content": content})
            if output_text:
                text_parts.append(output_text)
        tc = delta.get("tool_calls")
        if isinstance(tc, list):
            tool_calls.extend(tc)
    return "".join(text_parts), tool_calls
