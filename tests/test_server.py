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
                ],
                model_routing="match",
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
    async def test_unknown_model_falls_back_to_any_backend(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/chat/completions",
                json=_chat_payload(model="missing-model"),
            )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "hello from fake"

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

    @pytest.mark.asyncio
    async def test_response_includes_provider_name_by_default(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/v1/chat/completions", json=_chat_payload())
        assert response.status_code == 200
        assert response.json()["provider_name"] == "fake"

    @pytest.mark.asyncio
    async def test_response_excludes_provider_name_when_disabled(self):
        cfg = make_config([[{"model": "test-model", "response": "hi"}]])
        cfg.show_provider = False
        app = create_app(cfg)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/v1/chat/completions", json=_chat_payload())
        assert response.status_code == 200
        assert "provider_name" not in response.json()

    @pytest.mark.asyncio
    async def test_response_uses_custom_backend_name(self):
        app = create_app(
            make_config(
                [
                    [
                        {
                            "model": "test-model",
                            "response": "hi",
                            "name": "my-custom",
                        }
                    ]
                ]
            )
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/v1/chat/completions", json=_chat_payload())
        assert response.status_code == 200
        assert response.json()["provider_name"] == "my-custom"


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
                display_name="fake",
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
                display_name="fake",
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
    async def test_unknown_model_falls_back_to_any_backend(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/responses",
                json={"model": "missing-model", "input": "Hello"},
            )
        assert response.status_code == 200
        assert response.json()["output_text"] is not None

    @pytest.mark.asyncio
    async def test_response_503_when_all_fail(self, app):
        route_mock = AsyncMock(side_effect=AllProvidersFailedError([("fake", RuntimeError("fail"))]))
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
                events.append(line[len("event:") :].strip())
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
                display_name="fake",
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


class TestAnthropicMessages:
    @pytest.mark.asyncio
    async def test_basic_completion(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["model"] == "test-model"
        assert data["stop_reason"] == "end_turn"
        assert data["id"].startswith("msg_")
        assert data["content"][0]["type"] == "text"
        assert data["content"][0]["text"] == "hello from fake"

    @pytest.mark.asyncio
    async def test_system_message_forwarded(self, app):
        route_mock = AsyncMock(
            return_value=RoutedResponse(
                content="with system",
                model="test-model",
                provider_name="fake(test-model)",
            )
        )
        with patch.object(Router, "route", route_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                await ac.post(
                    "/v1/messages",
                    json={
                        "model": "test-model",
                        "max_tokens": 100,
                        "system": "Be brief",
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )
        messages = route_mock.await_args.args[0]
        assert messages[0] == {"role": "system", "content": "Be brief"}
        assert messages[1] == {"role": "user", "content": "Hi"}

    @pytest.mark.asyncio
    async def test_system_as_content_blocks(self, app):
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
                    "/v1/messages",
                    json={
                        "model": "test-model",
                        "max_tokens": 100,
                        "system": [
                            {"type": "text", "text": "Rule one"},
                            {"type": "text", "text": "Rule two"},
                        ],
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )
        messages = route_mock.await_args.args[0]
        assert messages[0] == {"role": "system", "content": "Rule one\n\nRule two"}

    @pytest.mark.asyncio
    async def test_stop_sequences_mapped_to_stop(self, app):
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
                    "/v1/messages",
                    json={
                        "model": "test-model",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "stop_sequences": ["END", "STOP"],
                    },
                )
        kwargs = route_mock.await_args.kwargs
        assert kwargs["stop"] == ["END", "STOP"]

    @pytest.mark.asyncio
    async def test_unknown_model_falls_back_to_any_backend(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/messages",
                json={
                    "model": "missing-model",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
        assert response.status_code == 200
        assert response.json()["content"][0]["text"] == "hello from fake"

    @pytest.mark.asyncio
    async def test_all_providers_failed_returns_529(self, app):
        route_mock = AsyncMock(side_effect=AllProvidersFailedError([("fake", RuntimeError("fail"))]))
        with patch.object(Router, "route", route_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post(
                    "/v1/messages",
                    json={
                        "model": "test-model",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )
        assert response.status_code == 529
        assert response.json()["type"] == "error"
        assert response.json()["error"]["type"] == "overloaded_error"

    @pytest.mark.asyncio
    async def test_streaming_returns_correct_events(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        events = []
        for line in response.text.strip().splitlines():
            if line.startswith("event:"):
                events.append(line[len("event:") :].strip())
        assert events[0] == "message_start"
        assert events[1] == "content_block_start"
        assert "content_block_delta" in events
        assert "content_block_stop" in events
        assert "message_delta" in events
        assert events[-1] == "message_stop"

    @pytest.mark.asyncio
    async def test_streaming_message_start_has_model(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )
        for line in response.text.strip().splitlines():
            if line.startswith("data:") and "message_start" in line:
                data = json.loads(line[len("data:") :].strip())
                assert data["message"]["model"] == "test-model"
                assert data["message"]["role"] == "assistant"
                break

    @pytest.mark.asyncio
    async def test_streaming_mid_error_sets_error_stop_reason(self, app):
        async def failing_chunks():
            yield "partial "
            raise StreamingProviderError("fake(test-model)")

        prepare_mock = AsyncMock(
            return_value=PreparedStream(
                model="test-model",
                provider_name="fake(test-model)",
                display_name="fake",
                chunks=failing_chunks(),
                request_id="req-1",
            )
        )
        with patch.object(Router, "prepare_stream", prepare_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post(
                    "/v1/messages",
                    json={
                        "model": "test-model",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "stream": True,
                    },
                )
        assert response.status_code == 200
        for line in response.text.strip().splitlines():
            if line.startswith("data:") and "message_delta" in line:
                data = json.loads(line[len("data:") :].strip())
                assert data["delta"]["stop_reason"] == "error"
                break

    @pytest.mark.asyncio
    async def test_response_includes_provider_name(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
        assert response.status_code == 200
        assert response.json()["provider_name"] == "fake"

    @pytest.mark.asyncio
    async def test_x_api_key_auth_accepted(self, authed_app):
        async with AsyncClient(transport=ASGITransport(app=authed_app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
                headers={"x-api-key": "secret-key"},
            )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_x_api_key_auth_rejected_wrong_key(self, authed_app):
        async with AsyncClient(transport=ASGITransport(app=authed_app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
                headers={"x-api-key": "wrong-key"},
            )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_aifree_model_routes_to_any_backend(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/messages",
                json={
                    "model": "aifree",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
        assert response.status_code == 200
        assert response.json()["content"][0]["text"] == "hello from fake"

    @pytest.mark.asyncio
    async def test_tools_forwarded_as_openai_format(self, app):
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
                    "/v1/messages",
                    json={
                        "model": "test-model",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "tools": [
                            {
                                "name": "get_weather",
                                "description": "Get weather",
                                "input_schema": {
                                    "type": "object",
                                    "properties": {"location": {"type": "string"}},
                                },
                            }
                        ],
                    },
                )
        kwargs = route_mock.await_args.kwargs
        assert kwargs["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

    @pytest.mark.asyncio
    async def test_tool_choice_any_mapped_to_required(self, app):
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
                    "/v1/messages",
                    json={
                        "model": "test-model",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "tools": [{"name": "t", "input_schema": {}}],
                        "tool_choice": {"type": "tool", "name": "t"},
                    },
                )
        kwargs = route_mock.await_args.kwargs
        assert kwargs["tool_choice"] == {
            "type": "function",
            "function": {"name": "t"},
        }

    @pytest.mark.asyncio
    async def test_tool_result_messages_converted(self, app):
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
                    "/v1/messages",
                    json={
                        "model": "test-model",
                        "max_tokens": 100,
                        "messages": [
                            {"role": "user", "content": "What's the weather?"},
                            {
                                "role": "assistant",
                                "content": [
                                    {"type": "text", "text": "Let me check."},
                                    {
                                        "type": "tool_use",
                                        "id": "toolu_1",
                                        "name": "get_weather",
                                        "input": {"location": "SF"},
                                    },
                                ],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": "toolu_1",
                                        "content": "65 degrees",
                                    }
                                ],
                            },
                        ],
                    },
                )
        messages = route_mock.await_args.args[0]
        assert messages[0] == {"role": "user", "content": "What's the weather?"}
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Let me check."
        assert messages[1]["tool_calls"][0]["id"] == "toolu_1"
        assert messages[1]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert messages[2] == {
            "role": "tool",
            "tool_call_id": "toolu_1",
            "content": "65 degrees",
        }

    @pytest.mark.asyncio
    async def test_tool_use_response_format(self, app):
        route_mock = AsyncMock(
            return_value=RoutedResponse(
                content="",
                model="test-model",
                provider_name="fake(test-model)",
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "SF"}',
                            },
                        }
                    ],
                },
            )
        )
        with patch.object(Router, "route", route_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post(
                    "/v1/messages",
                    json={
                        "model": "test-model",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Weather?"}],
                    },
                )
        assert response.status_code == 200
        data = response.json()
        assert data["stop_reason"] == "tool_use"
        assert data["content"][0]["type"] == "tool_use"
        assert data["content"][0]["id"] == "call_1"
        assert data["content"][0]["name"] == "get_weather"
        assert data["content"][0]["input"] == {"location": "SF"}

    @pytest.mark.asyncio
    async def test_tool_use_response_with_text_and_tool(self, app):
        route_mock = AsyncMock(
            return_value=RoutedResponse(
                content="Let me check.",
                model="test-model",
                provider_name="fake(test-model)",
                message={
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "SF"}',
                            },
                        }
                    ],
                },
            )
        )
        with patch.object(Router, "route", route_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post(
                    "/v1/messages",
                    json={
                        "model": "test-model",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Weather?"}],
                    },
                )
        data = response.json()
        assert data["stop_reason"] == "tool_use"
        assert data["content"][0]["type"] == "text"
        assert data["content"][0]["text"] == "Let me check."
        assert data["content"][1]["type"] == "tool_use"

    @pytest.mark.asyncio
    async def test_streaming_tool_use_only(self, app):
        async def tool_chunks():
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": ""},
                                },
                            ]
                        }
                    }
                ],
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"location"'}},
                            ]
                        }
                    }
                ],
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": ': "SF"}'}},
                            ]
                        }
                    }
                ],
            }

        prepare_mock = AsyncMock(
            return_value=PreparedStream(
                model="test-model",
                provider_name="fake(test-model)",
                display_name="fake",
                chunks=tool_chunks(),
                request_id="req-1",
            )
        )
        with patch.object(Router, "prepare_stream", prepare_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post(
                    "/v1/messages",
                    json={
                        "model": "test-model",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Weather?"}],
                        "stream": True,
                    },
                )
        assert response.status_code == 200
        events = []
        for line in response.text.strip().splitlines():
            if line.startswith("event:"):
                events.append(line[len("event:") :].strip())

        assert "content_block_start" in events
        assert "content_block_delta" in events
        assert "content_block_stop" in events

        for line in response.text.strip().splitlines():
            if line.startswith("data:") and "content_block_start" in line:
                data = json.loads(line[len("data:") :].strip())
                if data.get("content_block", {}).get("type") == "tool_use":
                    assert data["content_block"]["id"] == "call_1"
                    assert data["content_block"]["name"] == "get_weather"
                    break
        else:
            pytest.fail("No tool_use content_block_start found")

        for line in response.text.strip().splitlines():
            if line.startswith("data:") and "message_delta" in line:
                data = json.loads(line[len("data:") :].strip())
                assert data["delta"]["stop_reason"] == "tool_use"
                break

    @pytest.mark.asyncio
    async def test_streaming_text_then_tool_use(self, app):
        async def mixed_chunks():
            yield "Let me "
            yield "check. "
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": '{"location": "SF"}'},
                                },
                            ]
                        }
                    }
                ],
            }

        prepare_mock = AsyncMock(
            return_value=PreparedStream(
                model="test-model",
                provider_name="fake(test-model)",
                display_name="fake",
                chunks=mixed_chunks(),
                request_id="req-1",
            )
        )
        with patch.object(Router, "prepare_stream", prepare_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post(
                    "/v1/messages",
                    json={
                        "model": "test-model",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Weather?"}],
                        "stream": True,
                    },
                )
        events = []
        for line in response.text.strip().splitlines():
            if line.startswith("event:"):
                events.append(line[len("event:") :].strip())

        assert events[0] == "message_start"
        assert events.count("content_block_start") == 2
        assert events.count("content_block_stop") == 2

        block_starts = []
        for line in response.text.strip().splitlines():
            if line.startswith("data:") and "content_block_start" in line:
                data = json.loads(line[len("data:") :].strip())
                if data.get("type") == "content_block_start":
                    block_starts.append(data)
        assert block_starts[0]["content_block"]["type"] == "text"
        assert block_starts[0]["index"] == 0
        assert block_starts[1]["content_block"]["type"] == "tool_use"
        assert block_starts[1]["index"] == 1
        assert block_starts[1]["content_block"]["name"] == "get_weather"

        for line in response.text.strip().splitlines():
            if line.startswith("data:") and "message_delta" in line:
                data = json.loads(line[len("data:") :].strip())
                assert data["delta"]["stop_reason"] == "tool_use"
                break

    @pytest.mark.asyncio
    async def test_streaming_multiple_tool_calls(self, app):
        async def multi_tool_chunks():
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": '{"location": "SF"}'},
                                },
                            ]
                        }
                    }
                ],
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {"name": "get_time", "arguments": '{"tz": "PST"}'},
                                },
                            ]
                        }
                    }
                ],
            }

        prepare_mock = AsyncMock(
            return_value=PreparedStream(
                model="test-model",
                provider_name="fake(test-model)",
                display_name="fake",
                chunks=multi_tool_chunks(),
                request_id="req-1",
            )
        )
        with patch.object(Router, "prepare_stream", prepare_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post(
                    "/v1/messages",
                    json={
                        "model": "test-model",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Info?"}],
                        "stream": True,
                    },
                )
        block_starts = []
        for line in response.text.strip().splitlines():
            if line.startswith("data:") and "content_block_start" in line:
                data = json.loads(line[len("data:") :].strip())
                if data.get("type") == "content_block_start":
                    block_starts.append(data)

        tool_starts = [b for b in block_starts if b["content_block"]["type"] == "tool_use"]
        assert len(tool_starts) == 2
        assert tool_starts[0]["content_block"]["name"] == "get_weather"
        assert tool_starts[1]["content_block"]["name"] == "get_time"
        assert tool_starts[0]["index"] == 0
        assert tool_starts[1]["index"] == 1

    @pytest.mark.asyncio
    async def test_streaming_input_json_delta_accumulated(self, app):
        async def incremental_chunks():
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "search", "arguments": ""},
                                },
                            ]
                        }
                    }
                ],
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"q'}},
                            ]
                        }
                    }
                ],
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": 'uery":'}},
                            ]
                        }
                    }
                ],
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": ' "test"}'}},
                            ]
                        }
                    }
                ],
            }

        prepare_mock = AsyncMock(
            return_value=PreparedStream(
                model="test-model",
                provider_name="fake(test-model)",
                display_name="fake",
                chunks=incremental_chunks(),
                request_id="req-1",
            )
        )
        with patch.object(Router, "prepare_stream", prepare_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post(
                    "/v1/messages",
                    json={
                        "model": "test-model",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Search"}],
                        "stream": True,
                    },
                )
        json_parts = []
        for line in response.text.strip().splitlines():
            if line.startswith("data:") and "input_json_delta" in line:
                data = json.loads(line[len("data:") :].strip())
                json_parts.append(data["delta"]["partial_json"])
        assert "".join(json_parts) == '{"query": "test"}'


class TestRootEndpoint:
    @pytest.mark.asyncio
    async def test_head_returns_200(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.head("/")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_root_returns_200(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_head_skips_auth(self, authed_app):
        async with AsyncClient(transport=ASGITransport(app=authed_app), base_url="http://test") as ac:
            response = await ac.head("/")
        assert response.status_code == 200


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
