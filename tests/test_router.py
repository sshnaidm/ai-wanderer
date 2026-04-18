from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from ai_free_swap.config import BackendConfig
from ai_free_swap.router import AllProvidersFailedError, Router

from .conftest import FakeProvider, make_config

import ai_free_swap.providers  # noqa: F401


@pytest.fixture
def messages():
    return [HumanMessage(content="Hello")]


class TestRouterInit:
    def test_single_group(self):
        config = make_config([[{}]])
        router = Router(config)
        assert len(router.priority_groups) == 1
        assert len(router.priority_groups[0]) == 1

    def test_multiple_groups_sorted_by_priority(self):
        config = make_config([[{}], [{}], [{}]])
        router = Router(config)
        assert len(router.priority_groups) == 3

    def test_multiple_backends_in_group(self):
        config = make_config([[{}, {}, {}]])
        router = Router(config)
        assert len(router.priority_groups[0]) == 3

    def test_unknown_provider_raises(self):
        config = make_config([[{"provider": "nonexistent_xyz"}]])
        with pytest.raises(ValueError, match="Unknown provider 'nonexistent_xyz'"):
            Router(config)


class TestRouterRoute:
    @pytest.mark.asyncio
    async def test_success_first_try(self, messages):
        config = make_config([[{}]])
        router = Router(config)
        router.priority_groups[0][0].response = "success!"
        result = await router.route(messages)
        assert result == "success!"

    @pytest.mark.asyncio
    async def test_fallback_within_group(self, messages):
        config = make_config([[{}, {}]])
        router = Router(config)
        router.priority_groups[0][0].should_fail = True
        router.priority_groups[0][1].response = "from second"
        # With random ordering, one of them will succeed
        # Run multiple times to cover both orderings
        results = set()
        for _ in range(20):
            router.priority_groups[0][0].should_fail = True
            router.priority_groups[0][0].response = "from first"
            router.priority_groups[0][1].should_fail = False
            router.priority_groups[0][1].response = "from second"
            result = await router.route(messages)
            results.add(result)
        # Should always get "from second" since first always fails
        # (or "from first" if second is tried first but first doesn't fail...
        # but first always fails, so we always get "from second")
        assert results == {"from second"}

    @pytest.mark.asyncio
    async def test_fallback_to_next_priority(self, messages):
        config = make_config([[{}], [{}]])
        router = Router(config)
        router.priority_groups[0][0].should_fail = True
        router.priority_groups[1][0].response = "from priority 2"
        result = await router.route(messages)
        assert result == "from priority 2"

    @pytest.mark.asyncio
    async def test_all_fail_raises(self, messages):
        config = make_config([[{}], [{}]])
        router = Router(config)
        router.priority_groups[0][0].should_fail = True
        router.priority_groups[1][0].should_fail = True
        with pytest.raises(AllProvidersFailedError) as exc_info:
            await router.route(messages)
        assert len(exc_info.value.errors) == 2

    @pytest.mark.asyncio
    async def test_keep_cycles_retries(self, messages):
        config = make_config([[{}]], keep_cycles=3)
        router = Router(config)
        call_count = 0
        original_create = router.priority_groups[0][0].create_chat_model

        def counting_create():
            nonlocal call_count
            call_count += 1
            mock = MagicMock()
            if call_count < 3:
                mock.ainvoke = AsyncMock(side_effect=RuntimeError("fail"))
            else:
                mock.ainvoke = AsyncMock(return_value=AIMessage(content="finally!"))
            return mock

        router.priority_groups[0][0].create_chat_model = counting_create
        result = await router.route(messages)
        assert result == "finally!"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_keep_cycles_all_fail(self, messages):
        config = make_config([[{}]], keep_cycles=2)
        router = Router(config)
        router.priority_groups[0][0].should_fail = True
        with pytest.raises(AllProvidersFailedError) as exc_info:
            await router.route(messages)
        # 1 backend x 2 cycles = 2 errors
        assert len(exc_info.value.errors) == 2

    @pytest.mark.asyncio
    async def test_tries_all_backends_in_group(self, messages):
        config = make_config([[{}, {}, {}]])
        router = Router(config)
        attempted = []

        for i, provider in enumerate(router.priority_groups[0]):
            provider.should_fail = True
            original_name = provider.name

            def make_create(idx, prov):
                def create():
                    attempted.append(idx)
                    mock = MagicMock()
                    mock.ainvoke = AsyncMock(side_effect=RuntimeError("fail"))
                    return mock

                return create

            provider.create_chat_model = make_create(i, provider)

        with pytest.raises(AllProvidersFailedError):
            await router.route(messages)
        # All 3 backends should have been attempted
        assert sorted(attempted) == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_stops_on_first_success_in_group(self, messages):
        config = make_config([[{}, {}]])
        router = Router(config)
        router.priority_groups[0][0].response = "first wins"
        router.priority_groups[0][1].response = "second wins"

        # Patch random.sample to return deterministic order
        with patch("ai_free_swap.router.random.sample", side_effect=lambda g, n: g):
            result = await router.route(messages)
        assert result == "first wins"


class TestRouterStream:
    @pytest.mark.asyncio
    async def test_stream_success(self, messages):
        config = make_config([[{}]])
        router = Router(config)
        router.priority_groups[0][0].response = "hello world"

        chunks = []
        async for chunk in router.route_stream(messages):
            chunks.append(chunk)
        assert "".join(chunks).strip() == "hello world"

    @pytest.mark.asyncio
    async def test_stream_fallback(self, messages):
        config = make_config([[{}], [{}]])
        router = Router(config)
        router.priority_groups[0][0].should_fail = True
        router.priority_groups[1][0].response = "fallback stream"

        chunks = []
        async for chunk in router.route_stream(messages):
            chunks.append(chunk)
        assert "".join(chunks).strip() == "fallback stream"

    @pytest.mark.asyncio
    async def test_stream_all_fail(self, messages):
        config = make_config([[{}]])
        router = Router(config)
        router.priority_groups[0][0].should_fail = True

        with pytest.raises(AllProvidersFailedError):
            async for _ in router.route_stream(messages):
                pass
