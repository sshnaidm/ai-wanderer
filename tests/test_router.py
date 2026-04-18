from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from ai_free_swap.router import (
    AllProvidersFailedError,
    NoMatchingProvidersError,
    Router,
    StreamingProviderError,
)

from .conftest import make_config

import ai_free_swap.providers  # noqa: F401


@pytest.fixture
def messages():
    return [HumanMessage(content="Hello")]


class TestRouterInit:
    def test_single_group(self):
        router = Router(make_config([[{}]]))
        assert len(router.priority_groups) == 1
        assert len(router.priority_groups[0]) == 1

    def test_merges_duplicate_priorities(self):
        router = Router(
            make_config(
                [[{}], [{}], [{}]],
                priorities=[1, 1, 2],
            )
        )
        assert len(router.priority_groups) == 2
        assert len(router.priority_groups[0]) == 2

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider 'nonexistent_xyz'"):
            Router(make_config([[{"provider": "nonexistent_xyz"}]]))


class TestRouterRoute:
    @pytest.mark.asyncio
    async def test_returns_actual_model_for_selected_backend(self, messages):
        router = Router(
            make_config(
                [
                    [
                        {"model": "model-a", "response": "from a"},
                        {"model": "model-b", "response": "from b"},
                    ]
                ]
            )
        )
        result = await router.route(messages, requested_model="model-b")
        assert result.content == "from b"
        assert result.model == "model-b"

    @pytest.mark.asyncio
    async def test_retries_all_same_priority_backends_before_next_priority(self, messages):
        router = Router(
            make_config(
                [
                    [{"should_fail": True}],
                    [{"response": "same-priority success"}],
                    [{"response": "later priority"}],
                ],
                priorities=[1, 1, 2],
            )
        )
        with patch("ai_free_swap.router.random.sample", side_effect=lambda group, _: group):
            result = await router.route(messages)
        assert result.content == "same-priority success"
        assert result.provider_name == "fake(test-model)"

    @pytest.mark.asyncio
    async def test_keep_cycles_retries_until_success(self, messages):
        router = Router(make_config([[{"response": "finally"}]], keep_cycles=3))
        provider = router.priority_groups[0][0]
        attempts = 0
        original_create = provider.create_chat_model

        def counting_create():
            nonlocal attempts
            attempts += 1
            provider.should_fail = attempts < 3
            return original_create()

        provider.create_chat_model = counting_create
        result = await router.route(messages)
        assert result.content == "finally"
        assert attempts == 3

    @pytest.mark.asyncio
    async def test_all_fail_raises(self, messages):
        router = Router(make_config([[{"should_fail": True}], [{"should_fail": True}]]))
        with pytest.raises(AllProvidersFailedError) as exc_info:
            await router.route(messages)
        assert len(exc_info.value.errors) == 2

    @pytest.mark.asyncio
    async def test_unknown_model_raises(self, messages):
        router = Router(make_config([[{"model": "configured-model"}]]))
        with pytest.raises(NoMatchingProvidersError, match="configured"):
            await router.route(messages, requested_model="missing-model")


class TestRouterStream:
    @pytest.mark.asyncio
    async def test_prepare_stream_fails_over_before_first_chunk(self, messages):
        router = Router(
            make_config(
                [
                    [{"model": "broken-model", "should_fail": True}],
                    [{"model": "good-model", "response": "fallback stream"}],
                ]
            )
        )
        prepared = await router.prepare_stream(messages)
        chunks = [chunk async for chunk in prepared.chunks]
        assert "".join(chunks).strip() == "fallback stream"
        assert prepared.model == "good-model"

    @pytest.mark.asyncio
    async def test_prepare_stream_does_not_fail_over_after_partial_output(self, messages):
        router = Router(
            make_config(
                [
                    [
                        {
                            "model": "partial-model",
                            "stream_chunks": ["partial "],
                            "stream_fail_after_chunks": 1,
                        }
                    ],
                    [{"model": "fallback-model", "response": "fallback output"}],
                ]
            )
        )
        prepared = await router.prepare_stream(messages)

        chunks = []
        with pytest.raises(StreamingProviderError):
            async for chunk in prepared.chunks:
                chunks.append(chunk)

        assert chunks == ["partial "]
        assert prepared.model == "partial-model"

    @pytest.mark.asyncio
    async def test_prepare_stream_all_fail_raises(self, messages):
        router = Router(make_config([[{"should_fail": True}]]))
        with pytest.raises(AllProvidersFailedError):
            await router.prepare_stream(messages)
