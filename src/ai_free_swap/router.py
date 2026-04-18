from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage

from .config import AppConfig
from .providers.base import PROVIDER_REGISTRY, BaseProvider

logger = logging.getLogger(__name__)


class AllProvidersFailedError(Exception):
    def __init__(self, errors: list[tuple[str, Exception]]):
        self.errors = errors
        super().__init__("All configured providers failed")

    @property
    def detail_summary(self) -> str:
        if not self.errors:
            return "no provider attempts were made"
        return "; ".join(f"{name}: {e}" for name, e in self.errors)


class NoMatchingProvidersError(Exception):
    def __init__(self, requested_model: str):
        self.requested_model = requested_model
        super().__init__(f"Model {requested_model!r} is not configured")


class StreamingProviderError(Exception):
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        super().__init__(f"Streaming interrupted by {provider_name}")


@dataclass(frozen=True)
class RoutedResponse:
    content: str
    model: str
    provider_name: str


@dataclass(frozen=True)
class PreparedStream:
    model: str
    provider_name: str
    chunks: AsyncGenerator[str, None]


class Router:
    def __init__(self, config: AppConfig):
        self.keep_cycles = config.keep_cycles

        priority_map: dict[int, list[BaseProvider]] = defaultdict(list)
        for group in sorted(config.providers, key=lambda g: g.priority):
            for backend in group.backends:
                cls = PROVIDER_REGISTRY.get(backend.provider)
                if cls is None:
                    raise ValueError(
                        f"Unknown provider {backend.provider!r}. " f"Available: {', '.join(sorted(PROVIDER_REGISTRY))}"
                    )
                priority_map[group.priority].append(cls(backend))

        self.priority_groups = [priority_map[priority] for priority in sorted(priority_map)]

        logger.info(
            "Router initialized with %d priority groups, %d total backends",
            len(self.priority_groups),
            sum(len(g) for g in self.priority_groups),
        )

    async def route(
        self,
        messages: list[BaseMessage],
        *,
        requested_model: str | None = None,
        **kwargs,
    ) -> RoutedResponse:
        errors: list[tuple[str, Exception]] = []

        for backend in self._iter_attempts(requested_model):
            try:
                logger.info("Trying %s", backend.name)
                model = backend.create_chat_model()
                result = await model.ainvoke(messages, **kwargs)
                logger.info("Success from %s", backend.name)
                return RoutedResponse(
                    content=self._coerce_text(getattr(result, "content", None)),
                    model=backend.config.model,
                    provider_name=backend.name,
                )
            except Exception as e:
                logger.warning("Failed %s: %s", backend.name, e)
                errors.append((backend.name, e))

        raise AllProvidersFailedError(errors)

    async def prepare_stream(
        self,
        messages: list[BaseMessage],
        *,
        requested_model: str | None = None,
        **kwargs,
    ) -> PreparedStream:
        errors: list[tuple[str, Exception]] = []

        for backend in self._iter_attempts(requested_model):
            try:
                logger.info("Trying %s (stream)", backend.name)
                model = backend.create_chat_model()
                stream = aiter(model.astream(messages, **kwargs))
                buffered: list[str] = []
                while True:
                    try:
                        chunk = await anext(stream)
                    except StopAsyncIteration:
                        break

                    text = self._coerce_text(getattr(chunk, "content", None))
                    if text:
                        buffered.append(text)
                        break

                logger.info("Streaming started from %s", backend.name)
                return PreparedStream(
                    model=backend.config.model,
                    provider_name=backend.name,
                    chunks=self._drain_stream(stream, buffered, backend),
                )
            except Exception as e:
                logger.warning("Failed %s: %s", backend.name, e)
                errors.append((backend.name, e))

        raise AllProvidersFailedError(errors)

    def _iter_attempts(self, requested_model: str | None) -> Iterator[BaseProvider]:
        candidate_groups = self._get_candidate_groups(requested_model)

        for cycle in range(self.keep_cycles):
            if cycle > 0:
                logger.info("Starting cycle %d/%d", cycle + 1, self.keep_cycles)

            for group in candidate_groups:
                yield from random.sample(group, len(group))

    def _get_candidate_groups(self, requested_model: str | None) -> list[list[BaseProvider]]:
        normalized_model = self._normalize_requested_model(requested_model)
        if normalized_model is None:
            return self.priority_groups

        filtered_groups = [
            [provider for provider in group if provider.config.model == normalized_model]
            for group in self.priority_groups
        ]
        filtered_groups = [group for group in filtered_groups if group]
        if not filtered_groups:
            raise NoMatchingProvidersError(normalized_model)
        return filtered_groups

    @staticmethod
    def _normalize_requested_model(requested_model: str | None) -> str | None:
        if requested_model is None:
            return None
        normalized = requested_model.strip()
        if not normalized:
            return None
        return normalized

    @staticmethod
    def _coerce_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts)
        return str(content)

    async def _drain_stream(
        self,
        stream: AsyncIterator[Any],
        buffered: list[str],
        backend: BaseProvider,
    ) -> AsyncGenerator[str, None]:
        for text in buffered:
            yield text

        try:
            async for chunk in stream:
                text = self._coerce_text(getattr(chunk, "content", None))
                if text:
                    yield text
        except Exception as e:
            logger.error("Streaming interrupted from %s: %s", backend.name, e)
            raise StreamingProviderError(backend.name) from e
