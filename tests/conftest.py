from __future__ import annotations

from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk

from ai_free_swap.config import AppConfig, BackendConfig, PriorityGroup, ServerConfig
from ai_free_swap.providers.base import PROVIDER_REGISTRY, BaseProvider, register_provider
from ai_free_swap.server import create_app

import ai_free_swap.providers  # noqa: F401 — ensure real providers register


@register_provider("fake")
class FakeProvider(BaseProvider):
    """Test provider that returns a mock chat model."""

    response: str = "fake response"
    should_fail: bool = False
    fail_error: Exception | None = None

    def create_chat_model(self) -> BaseChatModel:
        mock = MagicMock(spec=BaseChatModel)

        if self.should_fail:
            error = self.fail_error or RuntimeError(f"{self.name} forced failure")
            mock.ainvoke = AsyncMock(side_effect=error)
            mock.astream = AsyncMock(side_effect=error)
        else:
            mock.ainvoke = AsyncMock(return_value=AIMessage(content=self.response))

            async def fake_stream(*args, **kwargs):
                for word in self.response.split():
                    yield AIMessageChunk(content=word + " ")

            mock.astream = MagicMock(side_effect=fake_stream)

        return mock


def make_config(
    groups: list[list[dict]],
    keep_cycles: int = 1,
    api_key: str = "",
) -> AppConfig:
    """Build an AppConfig from a simplified structure.

    groups is a list of priority groups, each group is a list of backend dicts.
    Example: [[{"response": "hi"}], [{"response": "fallback"}]]
    """
    providers = []
    for priority, backends_data in enumerate(groups, start=1):
        backends = []
        for bd in backends_data:
            backends.append(
                BackendConfig(
                    provider=bd.get("provider", "fake"),
                    api_key=bd.get("api_key", "test-key"),
                    model=bd.get("model", "test-model"),
                    base_url=bd.get("base_url"),
                    extra=bd.get("extra", {}),
                )
            )
        providers.append(PriorityGroup(priority=priority, backends=backends))

    return AppConfig(
        keep_cycles=keep_cycles,
        server=ServerConfig(api_key=api_key),
        providers=providers,
    )


@pytest.fixture
def single_backend_config():
    return make_config([[{"response": "hello from fake"}]])


@pytest.fixture
def multi_priority_config():
    return make_config([
        [{"model": "fail-1"}, {"model": "fail-2"}],
        [{"model": "success"}],
    ])


@pytest.fixture
def authed_config():
    return make_config([[{}]], api_key="secret-key")
