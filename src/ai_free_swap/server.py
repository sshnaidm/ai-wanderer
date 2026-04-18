from __future__ import annotations

import json
import logging
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from .config import AppConfig
from .converters import openai_to_langchain
from .models import (
    ChatCompletionRequest,
    make_completion_response,
    make_stream_chunk,
)
from .router import AllProvidersFailedError, Router

logger = logging.getLogger(__name__)


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="ai-free-swap")
    router = Router(config)
    server_api_key = config.server.api_key

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if server_api_key:
            auth = request.headers.get("authorization", "")
            token = auth.removeprefix("Bearer ").strip()
            if token != server_api_key:
                return JSONResponse(
                    status_code=401,
                    content={"error": {"message": "Invalid API key", "type": "auth_error"}},
                )
        return await call_next(request)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        lc_messages = openai_to_langchain(request.messages)

        kwargs = {}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.stop is not None:
            kwargs["stop"] = request.stop

        if request.stream:
            return EventSourceResponse(
                _stream_response(router, lc_messages, kwargs, request.model)
            )

        try:
            content = await router.route(lc_messages, **kwargs)
        except AllProvidersFailedError as e:
            raise HTTPException(status_code=503, detail=str(e))

        return make_completion_response(content, request.model)

    @app.get("/v1/models")
    async def list_models():
        models = set()
        for group in config.providers:
            for backend in group.backends:
                models.add(backend.model)
        return {
            "object": "list",
            "data": [
                {"id": m, "object": "model", "owned_by": "ai-free-swap"}
                for m in sorted(models)
            ],
        }

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


async def _stream_response(router, lc_messages, kwargs, model):
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    yield {
        "event": "message",
        "data": json.dumps(make_stream_chunk(None, request_id, model, role="assistant")),
    }

    try:
        async for text in router.route_stream(lc_messages, **kwargs):
            yield {
                "event": "message",
                "data": json.dumps(make_stream_chunk(text, request_id, model)),
            }
    except AllProvidersFailedError as e:
        logger.error("All providers failed during stream: %s", e)
        yield {
            "event": "message",
            "data": json.dumps(
                make_stream_chunk(f"\n\n[Error: {e}]", request_id, model)
            ),
        }

    yield {
        "event": "message",
        "data": json.dumps(
            make_stream_chunk(None, request_id, model, finish_reason="stop")
        ),
    }
    yield {"data": "[DONE]"}
