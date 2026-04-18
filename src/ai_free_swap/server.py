from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from .config import AppConfig
from .converters import openai_to_langchain
from .models import (
    ChatCompletionRequest,
    make_completion_response,
    make_error_response,
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


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="ai-free-swap")
    router = Router(config)
    server_api_key = config.server.api_key

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        if server_api_key:
            auth = request.headers.get("authorization", "")
            token = auth.removeprefix("Bearer ").strip()
            if token != server_api_key:
                return _error_response(401, "Invalid API key", "auth_error")
        return await call_next(request)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        if request.n not in (None, 1):
            return _error_response(
                400,
                "Only n=1 is currently supported",
                "invalid_request_error",
                code="unsupported_parameter",
            )

        try:
            lc_messages = openai_to_langchain(request.messages)
        except ValueError as e:
            return _error_response(400, str(e), "invalid_request_error")

        kwargs = _build_model_kwargs(request)

        if request.stream:
            try:
                prepared_stream = await router.prepare_stream(
                    lc_messages,
                    requested_model=request.model,
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
                logger.warning("All providers failed before stream start: %s", e.detail_summary)
                return _error_response(
                    503,
                    "All configured providers failed",
                    "server_error",
                    code="all_providers_failed",
                )

            return EventSourceResponse(_stream_response(prepared_stream))

        try:
            result = await router.route(
                lc_messages,
                requested_model=request.model,
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
            logger.warning("All providers failed: %s", e.detail_summary)
            return _error_response(
                503,
                "All configured providers failed",
                "server_error",
                code="all_providers_failed",
            )

        return make_completion_response(result.content, result.model)

    @app.get("/v1/models")
    async def list_models():
        models = set()
        for group in config.providers:
            for backend in group.backends:
                models.add(backend.model)
        return {
            "object": "list",
            "data": [{"id": model, "object": "model", "owned_by": "ai-free-swap"} for model in sorted(models)],
        }

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def _build_model_kwargs(request: ChatCompletionRequest) -> dict:
    kwargs = {}
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if request.stop is not None:
        kwargs["stop"] = request.stop
    if request.presence_penalty is not None:
        kwargs["presence_penalty"] = request.presence_penalty
    if request.frequency_penalty is not None:
        kwargs["frequency_penalty"] = request.frequency_penalty
    if request.user is not None:
        kwargs["user"] = request.user
    return kwargs


def _error_response(
    status_code: int,
    message: str,
    error_type: str,
    *,
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=make_error_response(message, error_type, code=code),
    )


async def _stream_response(
    prepared_stream: PreparedStream,
) -> AsyncGenerator[dict[str, str], None]:
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    yield {
        "data": json.dumps(
            make_stream_chunk(
                None,
                request_id,
                prepared_stream.model,
                role="assistant",
            )
        )
    }

    try:
        async for text in prepared_stream.chunks:
            yield {"data": json.dumps(make_stream_chunk(text, request_id, prepared_stream.model))}
    except StreamingProviderError as e:
        logger.error("%s", e)
        return

    yield {
        "data": json.dumps(
            make_stream_chunk(
                None,
                request_id,
                prepared_stream.model,
                finish_reason="stop",
            )
        )
    }
    yield {"data": "[DONE]"}
