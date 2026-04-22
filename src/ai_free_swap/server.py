from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from .config import AppConfig
from .models import (
    ChatCompletionRequest,
    ResponsesRequest,
    make_completion_response,
    make_error_response,
    make_responses_response,
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
        logger.debug("POST /v1/chat/completions body=%s", request.model_dump(exclude_none=True))
        if request.n not in (None, 1):
            return _error_response(
                400,
                "Only n=1 is currently supported",
                "invalid_request_error",
                code="unsupported_parameter",
            )

        messages = [m.model_dump(exclude_none=True) for m in request.messages]
        kwargs = _build_model_kwargs(request)

        if request.stream:
            try:
                prepared_stream = await router.prepare_stream(
                    messages,
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
                messages,
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

        logger.debug("Response from %s: %s", result.provider_name, result.content[:200])
        return make_completion_response(result.content, result.model)

    @app.post("/v1/responses")
    async def responses(request: ResponsesRequest):
        logger.debug("POST /v1/responses body=%s", request.model_dump(exclude_none=True))
        messages = request.to_messages()
        kwargs = request.to_model_kwargs()

        if request.stream:
            try:
                prepared_stream = await router.prepare_stream(
                    messages,
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

            return EventSourceResponse(
                _responses_stream(prepared_stream)
            )

        try:
            result = await router.route(
                messages,
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

        response_id = f"resp_{uuid.uuid4().hex[:24]}"
        return make_responses_response(result.content, result.model, response_id)

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {"id": "aifree", "object": "model", "owned_by": "ai-free-swap"}
            ],
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
    content = make_error_response(message, error_type, code=code)
    logger.debug("Error response %d: %s", status_code, content)
    return JSONResponse(status_code=status_code, content=content)


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
            yield {
                "data": json.dumps(
                    make_stream_chunk(text, request_id, prepared_stream.model)
                )
            }
    except StreamingProviderError as e:
        logger.error("%s", e)
        finish_reason = "error"
    else:
        finish_reason = "stop"

    yield {
        "data": json.dumps(
            make_stream_chunk(
                None,
                request_id,
                prepared_stream.model,
                finish_reason=finish_reason,
            )
        )
    }
    yield {"data": "[DONE]"}


async def _responses_stream(
    prepared_stream: PreparedStream,
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

    yield _event("response.created", {
        "response": {
            "id": response_id,
            "object": "response",
            "status": "in_progress",
            "model": prepared_stream.model,
            "output": [],
        },
    })

    yield _event("response.output_item.added", {
        "output_index": 0,
        "item": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        },
    })

    yield _event("response.content_part.added", {
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": ""},
    })

    full_text = []
    status = "completed"
    try:
        async for text in prepared_stream.chunks:
            full_text.append(text)
            yield _event("response.output_text.delta", {
                "output_index": 0,
                "content_index": 0,
                "item_id": msg_id,
                "delta": text,
            })
    except StreamingProviderError as e:
        logger.error("%s", e)
        status = "incomplete"

    joined = "".join(full_text)

    yield _event("response.output_text.done", {
        "output_index": 0,
        "content_index": 0,
        "item_id": msg_id,
        "text": joined,
    })

    yield _event("response.content_part.done", {
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": joined},
    })

    yield _event("response.output_item.done", {
        "output_index": 0,
        "item": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": joined}],
        },
    })

    yield _event("response.completed", {
        "response": make_responses_response(joined, prepared_stream.model, response_id),
    })

    yield {"data": "[DONE]"}
