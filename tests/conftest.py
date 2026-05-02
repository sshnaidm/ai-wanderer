from __future__ import annotations

from typing import Any

import pytest

from ai_free_swap.config import AppConfig, BackendConfig, PriorityGroup, ServerConfig
from ai_free_swap.providers.base import BaseProvider, register_provider

import ai_free_swap.providers  # noqa: F401


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

    async def complete(self, messages: list[dict], **kwargs) -> str:
        self.invoke_calls.append({"messages": messages, "kwargs": kwargs})
        if self.should_fail:
            raise self.fail_error or RuntimeError(f"{self.name} forced failure")
        return self.response

    async def stream(self, messages: list[dict], **kwargs):
        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        if self.should_fail and self.stream_fail_after_chunks is None:
            raise self.fail_error or RuntimeError(f"{self.name} forced failure")

        chunks = self.stream_chunks
        if chunks is None:
            chunks = [word + " " for word in self.response.split()]

        yielded = 0
        for chunk in chunks:
            yield chunk
            yielded += 1
            if self.stream_fail_after_chunks is not None and yielded >= self.stream_fail_after_chunks:
                raise self.fail_error or RuntimeError(f"{self.name} stream interrupted")


def make_config(
    groups: list[list[dict]],
    *,
    keep_cycles: int = 1,
    api_key: str = "",
    priorities: list[int] | None = None,
    model_routing: str = "any",
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
                    name=backend_data.get("name"),
                    base_url=backend_data.get("base_url"),
                    extra=extra,
                )
            )
        providers.append(PriorityGroup(priority=priority, backends=backends))

    return AppConfig(
        keep_cycles=keep_cycles,
        model_routing=model_routing,
        server=ServerConfig(api_key=api_key),
        providers=providers,
    )


@pytest.fixture
def authed_config():
    return make_config([[{}]], api_key="secret-key")
