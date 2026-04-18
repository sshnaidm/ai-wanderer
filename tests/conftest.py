from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from ai_free_swap.config import AppConfig, BackendConfig, PriorityGroup, ServerConfig
from ai_free_swap.providers.base import BaseProvider, register_provider

import ai_free_swap.providers  # noqa: F401


class FakeChatModel:
    def __init__(self, provider: "FakeProvider"):
        self.provider = provider

    async def ainvoke(self, messages, **kwargs):
        self.provider.invoke_calls.append({"messages": messages, "kwargs": kwargs})
        if self.provider.should_fail:
            raise self.provider.fail_error or RuntimeError(f"{self.provider.name} forced failure")
        return AIMessage(content=self.provider.response)

    async def astream(self, messages, **kwargs):
        self.provider.stream_calls.append({"messages": messages, "kwargs": kwargs})
        if self.provider.should_fail and self.provider.stream_fail_after_chunks is None:
            raise self.provider.fail_error or RuntimeError(f"{self.provider.name} forced failure")

        chunks = self.provider.stream_chunks
        if chunks is None:
            chunks = [word + " " for word in self.provider.response.split()]

        yielded = 0
        for chunk in chunks:
            yield AIMessageChunk(content=chunk)
            yielded += 1
            if self.provider.stream_fail_after_chunks is not None and yielded >= self.provider.stream_fail_after_chunks:
                raise self.provider.fail_error or RuntimeError(f"{self.provider.name} stream interrupted")


@register_provider("fake")
class FakeProvider(BaseProvider):
    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self.response = config.extra.get("response", "fake response")
        self.should_fail = config.extra.get("should_fail", False)
        self.fail_error = config.extra.get("fail_error")
        self.stream_chunks = config.extra.get("stream_chunks")
        self.stream_fail_after_chunks = config.extra.get("stream_fail_after_chunks")
        self.invoke_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    def create_chat_model(self):
        return FakeChatModel(self)


def make_config(
    groups: list[list[dict]],
    *,
    keep_cycles: int = 1,
    api_key: str = "",
    priorities: list[int] | None = None,
) -> AppConfig:
    providers = []
    for index, backends_data in enumerate(groups):
        priority = priorities[index] if priorities is not None else index + 1
        backends = []
        for backend_data in backends_data:
            extra = dict(backend_data.get("extra", {}))
            for key in (
                "response",
                "should_fail",
                "fail_error",
                "stream_chunks",
                "stream_fail_after_chunks",
            ):
                if key in backend_data:
                    extra[key] = backend_data[key]

            backends.append(
                BackendConfig(
                    provider=backend_data.get("provider", "fake"),
                    api_key=backend_data.get("api_key", "test-key"),
                    model=backend_data.get("model", "test-model"),
                    base_url=backend_data.get("base_url"),
                    extra=extra,
                )
            )
        providers.append(PriorityGroup(priority=priority, backends=backends))

    return AppConfig(
        keep_cycles=keep_cycles,
        server=ServerConfig(api_key=api_key),
        providers=providers,
    )


@pytest.fixture
def authed_config():
    return make_config([[{}]], api_key="secret-key")
