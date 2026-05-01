from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="ai-free-swap")
    router = Router(config)
    server_api_key = config.server.api_key
    model_name = config.model_name
    show_provider = config.show_provider

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if request.url.path == "/health":
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

            return EventSourceResponse(
                _anthropic_stream_response(prepared_stream, show_provider)
            )

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

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


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

    yield {
        "event": "content_block_start",
        "data": json.dumps({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }),
    }

    full_text: list[str] = []
    stop_reason = "end_turn"
    try:
        async for chunk in prepared_stream.chunks:
            text = _extract_stream_text(chunk)
            if not text:
                continue
            full_text.append(text)
            yield {
                "event": "content_block_delta",
                "data": json.dumps({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text},
                }),
            }
    except StreamingProviderError as e:
        logger.error("[%s] %s", prepared_stream.request_id, e)
        stop_reason = "error"

    yield {
        "event": "content_block_stop",
        "data": json.dumps({"type": "content_block_stop", "index": 0}),
    }

    yield {
        "event": "message_delta",
        "data": json.dumps({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 0},
        }),
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
