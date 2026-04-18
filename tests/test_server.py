from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from ai_free_swap.server import create_app

from .conftest import make_config

import ai_free_swap.providers  # noqa: F401


def _chat_payload(**overrides):
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def app():
    config = make_config([[{}]])
    return create_app(config)


@pytest.fixture
def authed_app():
    config = make_config([[{}]], api_key="secret-key")
    return create_app(config)


@pytest.fixture
def failing_app():
    config = make_config([[{}]])
    a = create_app(config)
    # Hack: make the router's providers fail
    from ai_free_swap.router import Router
    for group in a.state._state.get("router", Router).__dict__.get("priority_groups", []):
        for p in group:
            p.should_fail = True
    return a


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestModelsEndpoint:
    @pytest.mark.asyncio
    async def test_list_models(self):
        config = make_config([
            [{"model": "gemini-2.5-flash"}, {"model": "gemini-2.5-flash-lite"}],
            [{"model": "gpt-4o"}],
        ])
        app = create_app(config)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        assert "gemini-2.5-flash" in model_ids
        assert "gemini-2.5-flash-lite" in model_ids
        assert "gpt-4o" in model_ids

    @pytest.mark.asyncio
    async def test_models_deduplication(self):
        config = make_config([
            [{"model": "same-model"}, {"model": "same-model"}],
        ])
        app = create_app(config)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/v1/models")
        model_ids = [m["id"] for m in resp.json()["data"]]
        assert model_ids.count("same-model") == 1


class TestChatCompletions:
    @pytest.mark.asyncio
    async def test_basic_completion(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/v1/chat/completions", json=_chat_payload())
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["id"].startswith("chatcmpl-")

    @pytest.mark.asyncio
    async def test_model_echoed_back(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/v1/chat/completions", json=_chat_payload(model="my-model"))
        assert resp.json()["model"] == "my-model"

    @pytest.mark.asyncio
    async def test_multi_turn_messages(self, app):
        messages = [
            {"role": "system", "content": "Be brief"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Bye"},
        ]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/v1/chat/completions", json=_chat_payload(messages=messages))
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_optional_params_forwarded(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json=_chat_payload(temperature=0.5, max_tokens=100),
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_503_when_all_providers_fail(self):
        config = make_config([[{}]])
        app = create_app(config)
        # Access the router to mark providers as failing
        # The router is captured in the closure, so we need to trigger failure differently
        # by using a provider that will fail
        config_fail = make_config([[{"provider": "fake"}]])
        app_fail = create_app(config_fail)
        # Mark the fake providers in the router as failing
        # We need to reach into the app's route handler closure
        # Instead, create config with an unknown provider to test error path
        # Actually, let's just test with a properly failing setup
        from ai_free_swap.router import Router
        config2 = make_config([[{}]])
        app2 = create_app(config2)

        # Patch the router inside the app to make all providers fail
        from unittest.mock import patch, AsyncMock
        from ai_free_swap.router import AllProvidersFailedError
        with patch.object(Router, "route", side_effect=AllProvidersFailedError([("fake", RuntimeError("fail"))])):
            async with AsyncClient(transport=ASGITransport(app=app2), base_url="http://test") as ac:
                resp = await ac.post("/v1/chat/completions", json=_chat_payload())
            assert resp.status_code == 503


class TestStreamingCompletions:
    @pytest.mark.asyncio
    async def test_stream_returns_sse(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json=_chat_payload(stream=True),
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_stream_contains_done(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json=_chat_payload(stream=True),
            )
        assert "[DONE]" in resp.text

    @pytest.mark.asyncio
    async def test_stream_chunks_are_valid_json(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json=_chat_payload(stream=True),
            )
        for line in resp.text.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("event:"):
                continue
            if line.startswith("data:"):
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    continue
                parsed = json.loads(data)
                assert parsed["object"] == "chat.completion.chunk"
                assert "choices" in parsed

    @pytest.mark.asyncio
    async def test_stream_has_role_in_first_chunk(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json=_chat_payload(stream=True),
            )
        data_lines = [
            line[len("data:"):].strip()
            for line in resp.text.strip().split("\n")
            if line.strip().startswith("data:") and "[DONE]" not in line
        ]
        first_chunk = json.loads(data_lines[0])
        assert first_chunk["choices"][0]["delta"].get("role") == "assistant"

    @pytest.mark.asyncio
    async def test_stream_has_finish_reason_in_last_chunk(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json=_chat_payload(stream=True),
            )
        data_lines = [
            line[len("data:"):].strip()
            for line in resp.text.strip().split("\n")
            if line.strip().startswith("data:") and "[DONE]" not in line
        ]
        last_chunk = json.loads(data_lines[-1])
        assert last_chunk["choices"][0]["finish_reason"] == "stop"


class TestAuthentication:
    @pytest.mark.asyncio
    async def test_no_auth_required_when_no_key(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/v1/chat/completions", json=_chat_payload())
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_valid_bearer_token(self, authed_app):
        async with AsyncClient(transport=ASGITransport(app=authed_app), base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json=_chat_payload(),
                headers={"Authorization": "Bearer secret-key"},
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_missing_auth_rejected(self, authed_app):
        async with AsyncClient(transport=ASGITransport(app=authed_app), base_url="http://test") as ac:
            resp = await ac.post("/v1/chat/completions", json=_chat_payload())
        assert resp.status_code == 401
        assert resp.json()["error"]["type"] == "auth_error"

    @pytest.mark.asyncio
    async def test_wrong_token_rejected(self, authed_app):
        async with AsyncClient(transport=ASGITransport(app=authed_app), base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json=_chat_payload(),
                headers={"Authorization": "Bearer wrong-key"},
            )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_health_requires_auth_too(self, authed_app):
        async with AsyncClient(transport=ASGITransport(app=authed_app), base_url="http://test") as ac:
            resp = await ac.get("/health")
        assert resp.status_code == 401


class TestProviderRegistry:
    def test_all_providers_registered(self):
        from ai_free_swap.providers.base import PROVIDER_REGISTRY
        expected = {"gemini", "openai", "anthropic", "qwen", "openrouter", "openai_compat", "fake"}
        assert expected.issubset(set(PROVIDER_REGISTRY.keys()))
