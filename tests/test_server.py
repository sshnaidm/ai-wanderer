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
    async def test_list_models(self):
        app = create_app(
            make_config(
                [
                    [{"model": "gemini-2.5-flash"}, {"model": "gemini-2.5-flash-lite"}],
                    [{"model": "gpt-4o"}],
                ]
            )
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/v1/models")
        assert response.status_code == 200
        model_ids = [model["id"] for model in response.json()["data"]]
        assert model_ids == ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gpt-4o"]

    @pytest.mark.asyncio
    async def test_models_are_deduplicated(self):
        app = create_app(make_config([[{"model": "same-model"}, {"model": "same-model"}]]))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/v1/models")
        model_ids = [model["id"] for model in response.json()["data"]]
        assert model_ids == ["same-model"]


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
    async def test_rejects_n_greater_than_one(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/v1/chat/completions", json=_chat_payload(n=2))
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "unsupported_parameter"

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
    async def test_all_providers_failed_returns_generic_503(self, app):
        route_mock = AsyncMock(side_effect=AllProvidersFailedError([("fake", RuntimeError("super-secret"))]))
        with patch.object(Router, "route", route_mock):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post("/v1/chat/completions", json=_chat_payload())
        assert response.status_code == 503
        assert response.json()["error"]["message"] == "All configured providers failed"
        assert "super-secret" not in response.text

    @pytest.mark.asyncio
    async def test_missing_model_is_rejected(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_blank_model_is_rejected(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/chat/completions",
                json=_chat_payload(model="   "),
            )
        assert response.status_code == 422


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


class TestProviderRegistry:
    def test_all_providers_registered(self):
        from ai_free_swap.providers.base import PROVIDER_REGISTRY

        expected = {
            "anthropic",
            "fake",
            "gemini",
            "openai",
            "openai_compat",
            "openrouter",
            "qwen",
        }
        assert expected.issubset(set(PROVIDER_REGISTRY))
