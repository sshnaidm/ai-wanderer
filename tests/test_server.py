from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from ai_free_swap.router import AllProvidersFailedError, PreparedStream, RoutedResponse, Router, StreamingProviderError
from ai_free_swap.server import create_app

from .conftest import make_config

import ai_free_swap.providers  # noqa: F401


def _chat_payload(**overrides):
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    payload.update(overrides)
    return payload


def _data_lines(body: str) -> list[str]:
    return [line[len("data:") :].strip() for line in body.strip().splitlines() if line.startswith("data:")]


@pytest.fixture
def app():
    return create_app(make_config([[{"model": "test-model", "response": "hello from fake"}]]))


@pytest.fixture
def authed_app():
    return create_app(make_config([[{"model": "test-model", "response": "secured"}]], api_key="secret-key"))


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestModelsEndpoint:
    @pytest.mark.asyncio
    async def test_list_models_returns_single_aifree(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "aifree"
        assert data["data"][0]["owned_by"] == "ai-free-swap"


class TestChatCompletions:
    @pytest.mark.asyncio
    async def test_basic_completion_returns_actual_backend_model(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/v1/chat/completions", json=_chat_payload())
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "test-model"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "hello from fake"
        assert data["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_requested_model_routes_to_matching_backend(self):
        app = create_app(
            make_config(
                [
                    [
                        {"model": "model-a", "response": "from a"},
                        {"model": "model-b", "response": "from b"},
                    ]
                ]
            )
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/chat/completions",
                json=_chat_payload(model="model-b"),
            )
        assert response.status_code == 200
        assert response.json()["model"] == "model-b"
        assert response.json()["choices"][0]["message"]["content"] == "from b"

    @pytest.mark.asyncio
    async def test_unknown_model_returns_400(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/chat/completions",
                json=_chat_payload(model="missing-model"),
            )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_found"

    @pytest.mark.asyncio
    async def test_aifree_model_routes_to_any_backend(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/chat/completions",
                json=_chat_payload(model="aifree"),
            )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "hello from fake"

    @pytest.mark.asyncio
    async def test_n_greater_than_one_is_forwarded(self, app):
        route_mock = AsyncMock(
            return_value=RoutedResponse(
                content="patched",
                model="test-model",
                provider_name="fake(test-model)",
            )
        )
        with patch.object(Router, "route", route_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post("/v1/chat/completions", json=_chat_payload(n=2))
        assert response.status_code == 200
        assert route_mock.await_args.kwargs["n"] == 2

    @pytest.mark.asyncio
    async def test_optional_params_are_forwarded(self, app):
        route_mock = AsyncMock(
            return_value=RoutedResponse(
                content="patched",
                model="test-model",
                provider_name="fake(test-model)",
            )
        )
        with patch.object(Router, "route", route_mock):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as ac:
                response = await ac.post(
                    "/v1/chat/completions",
                    json=_chat_payload(
                        temperature=0.5,
                        top_p=0.8,
                        max_tokens=42,
                        stop=["END"],
                        presence_penalty=0.1,
                        frequency_penalty=0.2,
                        user="user-123",
                    ),
                )
        assert response.status_code == 200
        kwargs = route_mock.await_args.kwargs
        assert kwargs["requested_model"] == "test-model"
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_p"] == 0.8
        assert kwargs["max_tokens"] == 42
        assert kwargs["stop"] == ["END"]
        assert kwargs["presence_penalty"] == 0.1
        assert kwargs["frequency_penalty"] == 0.2
        assert kwargs["user"] == "user-123"

    @pytest.mark.asyncio
    async def test_unknown_and_modern_params_are_forwarded(self, app):
        route_mock = AsyncMock(
            return_value=RoutedResponse(
                content="patched",
                model="test-model",
                provider_name="fake(test-model)",
            )
        )
        with patch.object(Router, "route", route_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post(
                    "/v1/chat/completions",
                    json=_chat_payload(
                        tools=[{"type": "function", "function": {"name": "ping"}}],
                        parallel_tool_calls=True,
                        reasoning={"effort": "medium"},
                    ),
                )
        assert response.status_code == 200
        kwargs = route_mock.await_args.kwargs
        assert kwargs["tools"] == [{"type": "function", "function": {"name": "ping"}}]
        assert kwargs["parallel_tool_calls"] is True
        assert kwargs["reasoning"] == {"effort": "medium"}

    @pytest.mark.asyncio
    async def test_accepts_developer_role_and_content_arrays(self, app):
        route_mock = AsyncMock(
            return_value=RoutedResponse(
                content="patched",
                model="test-model",
                provider_name="fake(test-model)",
            )
        )
        with patch.object(Router, "route", route_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [
                            {
                                "role": "developer",
                                "content": [{"type": "text", "text": "Follow policy"}],
                            }
                        ],
                    },
                )
        assert response.status_code == 200
        assert route_mock.await_args.args[0] == [
            {
                "role": "developer",
                "content": [{"type": "text", "text": "Follow policy"}],
            }
        ]

    @pytest.mark.asyncio
    async def test_all_providers_failed_returns_generic_503(self, app):
        route_mock = AsyncMock(side_effect=AllProvidersFailedError([("fake", RuntimeError("super-secret"))]))
        with patch.object(Router, "route", route_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post("/v1/chat/completions", json=_chat_payload())
        assert response.status_code == 503
        assert response.json()["error"]["message"] == "All configured providers failed"
        assert "super-secret" not in response.text

    @pytest.mark.asyncio
    async def test_missing_model_defaults_to_aifree(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_blank_model_defaults_to_aifree(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/chat/completions",
                json=_chat_payload(model="   "),
            )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_raw_provider_response_is_returned_directly(self, app):
        route_mock = AsyncMock(
            return_value=RoutedResponse(
                content="tool response",
                model="test-model",
                provider_name="fake(test-model)",
                raw_response={
                    "id": "chatcmpl-custom",
                    "object": "chat.completion",
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{"id": "call-1", "type": "function"}],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            )
        )
        with patch.object(Router, "route", route_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post("/v1/chat/completions", json=_chat_payload())
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["tool_calls"] == [{"id": "call-1", "type": "function"}]


class TestStreamingCompletions:
    @pytest.mark.asyncio
    async def test_stream_returns_sse_without_named_events(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/chat/completions",
                json=_chat_payload(stream=True),
            )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        assert "event:" not in response.text
        assert "[DONE]" in response.text

    @pytest.mark.asyncio
    async def test_stream_uses_actual_backend_model(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/chat/completions",
                json=_chat_payload(stream=True),
            )
        data_lines = [line for line in _data_lines(response.text) if line != "[DONE]"]
        first_chunk = json.loads(data_lines[0])
        assert first_chunk["model"] == "test-model"
        assert first_chunk["choices"][0]["delta"]["role"] == "assistant"
        last_chunk = json.loads(data_lines[-1])
        assert last_chunk["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_stream_preflight_failure_returns_json_503(self, app):
        prepare_mock = AsyncMock(side_effect=AllProvidersFailedError([("fake", RuntimeError("super-secret"))]))
        with patch.object(Router, "prepare_stream", prepare_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post(
                    "/v1/chat/completions",
                    json=_chat_payload(stream=True),
                )
        assert response.status_code == 503
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["error"]["code"] == "all_providers_failed"

    @pytest.mark.asyncio
    async def test_stream_can_use_patched_actual_model(self, app):
        async def fake_chunks():
            yield "hello "

        prepare_mock = AsyncMock(
            return_value=PreparedStream(
                model="patched-model",
                provider_name="fake(patched-model)",
                chunks=fake_chunks(),
            )
        )
        with patch.object(Router, "prepare_stream", prepare_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post(
                    "/v1/chat/completions",
                    json=_chat_payload(stream=True),
                )
        data_lines = [line for line in _data_lines(response.text) if line != "[DONE]"]
        assert json.loads(data_lines[0])["model"] == "patched-model"


    @pytest.mark.asyncio
    async def test_stream_mid_generation_error_emits_done(self, app):
        async def failing_chunks():
            yield "partial "
            raise StreamingProviderError("fake(test-model)")

        prepare_mock = AsyncMock(
            return_value=PreparedStream(
                model="test-model",
                provider_name="fake(test-model)",
                chunks=failing_chunks(),
            )
        )
        with patch.object(Router, "prepare_stream", prepare_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post(
                    "/v1/chat/completions",
                    json=_chat_payload(stream=True),
                )
        assert "[DONE]" in response.text
        data_lines = [line for line in _data_lines(response.text) if line != "[DONE]"]
        last_chunk = json.loads(data_lines[-1])
        assert last_chunk["choices"][0]["finish_reason"] == "error"


class TestAuthentication:
    @pytest.mark.asyncio
    async def test_valid_bearer_token(self, authed_app):
        async with AsyncClient(transport=ASGITransport(app=authed_app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/chat/completions",
                json=_chat_payload(),
                headers={"Authorization": "Bearer secret-key"},
            )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_lowercase_bearer_token_is_accepted(self, authed_app):
        async with AsyncClient(transport=ASGITransport(app=authed_app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/chat/completions",
                json=_chat_payload(),
                headers={"Authorization": "bearer secret-key"},
            )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_raw_authorization_token_is_accepted(self, authed_app):
        async with AsyncClient(transport=ASGITransport(app=authed_app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/chat/completions",
                json=_chat_payload(),
                headers={"Authorization": "secret-key"},
            )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_missing_auth_is_rejected(self, authed_app):
        async with AsyncClient(transport=ASGITransport(app=authed_app), base_url="http://test") as ac:
            response = await ac.post("/v1/chat/completions", json=_chat_payload())
        assert response.status_code == 401
        assert response.json()["error"]["type"] == "auth_error"

    @pytest.mark.asyncio
    async def test_health_is_public_even_when_auth_is_enabled(self, authed_app):
        async with AsyncClient(transport=ASGITransport(app=authed_app), base_url="http://test") as ac:
            response = await ac.get("/health")
        assert response.status_code == 200


class TestResponsesAPI:
    @pytest.mark.asyncio
    async def test_basic_response_with_string_input(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/responses",
                json={"model": "test-model", "input": "Hello"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "response"
        assert data["status"] == "completed"
        assert data["output_text"] is not None
        assert data["output"][0]["type"] == "message"
        assert data["output"][0]["role"] == "assistant"
        assert data["output"][0]["content"][0]["type"] == "output_text"
        assert data["id"].startswith("resp_")

    @pytest.mark.asyncio
    async def test_response_with_messages_input(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello"},
                        {"role": "user", "content": "Bye"},
                    ],
                },
            )
        assert response.status_code == 200
        assert response.json()["object"] == "response"

    @pytest.mark.asyncio
    async def test_response_with_instructions(self, app):
        route_mock = AsyncMock(
            return_value=RoutedResponse(
                content="instructed",
                model="test-model",
                provider_name="fake(test-model)",
            )
        )
        with patch.object(Router, "route", route_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post(
                    "/v1/responses",
                    json={
                        "model": "test-model",
                        "input": "Hello",
                        "instructions": "Be brief",
                    },
                )
        assert response.status_code == 200
        messages = route_mock.await_args.args[0]
        assert messages[0] == {"role": "system", "content": "Be brief"}
        assert messages[1] == {"role": "user", "content": "Hello"}

    @pytest.mark.asyncio
    async def test_response_unknown_model_returns_400(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/responses",
                json={"model": "missing-model", "input": "Hello"},
            )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_found"

    @pytest.mark.asyncio
    async def test_response_503_when_all_fail(self, app):
        route_mock = AsyncMock(
            side_effect=AllProvidersFailedError([("fake", RuntimeError("fail"))])
        )
        with patch.object(Router, "route", route_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post(
                    "/v1/responses",
                    json={"model": "test-model", "input": "Hello"},
                )
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_response_streaming(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/responses",
                json={"model": "test-model", "input": "Hello", "stream": True},
            )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        assert "[DONE]" in response.text
        # Check for expected event types
        assert "response.created" in response.text
        assert "response.output_text.delta" in response.text
        assert "response.completed" in response.text

    @pytest.mark.asyncio
    async def test_response_streaming_has_correct_events(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/responses",
                json={"model": "test-model", "input": "Hello", "stream": True},
            )
        events = []
        for line in response.text.strip().splitlines():
            if line.startswith("event:"):
                events.append(line[len("event:"):].strip())
        assert events[0] == "response.created"
        assert "response.output_text.delta" in events
        assert events[-1] == "response.completed"

    @pytest.mark.asyncio
    async def test_response_streaming_marks_midstream_failure_incomplete(self, app):
        async def failing_chunks():
            yield "partial "
            raise StreamingProviderError("fake(test-model)")

        prepare_mock = AsyncMock(
            return_value=PreparedStream(
                model="test-model",
                provider_name="fake(test-model)",
                chunks=failing_chunks(),
                request_id="req-1",
            )
        )
        with patch.object(Router, "prepare_stream", prepare_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post(
                    "/v1/responses",
                    json={"model": "test-model", "input": "Hello", "stream": True},
                )
        assert response.status_code == 200
        assert '"status": "incomplete"' in response.text or '"status":"incomplete"' in response.text

    @pytest.mark.asyncio
    async def test_response_preserves_message_shape_when_available(self, app):
        route_mock = AsyncMock(
            return_value=RoutedResponse(
                content="Hello",
                model="test-model",
                provider_name="fake(test-model)",
                message={
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello"}],
                    "tool_calls": [{"id": "call-1", "type": "function"}],
                },
            )
        )
        with patch.object(Router, "route", route_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post(
                    "/v1/responses",
                    json={"model": "test-model", "input": "Hello"},
                )
        assert response.status_code == 200
        data = response.json()
        assert data["output"][0]["tool_calls"] == [{"id": "call-1", "type": "function"}]
        assert data["output"][0]["content"] == [{"type": "output_text", "text": "Hello"}]

    @pytest.mark.asyncio
    async def test_max_output_tokens_maps_to_max_tokens(self, app):
        route_mock = AsyncMock(
            return_value=RoutedResponse(
                content="ok",
                model="test-model",
                provider_name="fake(test-model)",
            )
        )
        with patch.object(Router, "route", route_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                await ac.post(
                    "/v1/responses",
                    json={
                        "model": "test-model",
                        "input": "Hi",
                        "max_output_tokens": 500,
                    },
                )
        kwargs = route_mock.await_args.kwargs
        assert kwargs["max_tokens"] == 500


class TestProviderRegistry:
    def test_all_providers_registered(self):
        from ai_free_swap.providers.base import PROVIDER_REGISTRY

        expected = {
            "anthropic",
            "fake",
            "gemini",
            "grok",
            "openai",
            "openai_compat",
            "openrouter",
            "qwen",
        }
        assert expected.issubset(set(PROVIDER_REGISTRY))
