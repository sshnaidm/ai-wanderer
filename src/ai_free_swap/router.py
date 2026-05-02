from __future__ import annotations

import logging
import random
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from typing import Any
from dataclasses import dataclass

from .config import AppConfig
from .providers.base import PROVIDER_REGISTRY, BaseProvider, ProviderResponse

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
    display_name: str = ""
    message: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None


@dataclass(frozen=True)
class PreparedStream:
    model: str
    provider_name: str
    display_name: str
    chunks: AsyncGenerator[str | dict[str, Any], None]
    request_id: str = ""
    raw_chunks: bool = False


def _format_error(e: Exception) -> str:
    status = getattr(e, "status_code", None)
    msg = str(e)
    if status is not None:
        return f'{status}, error message="{msg}"'
    return f'error message="{msg}"'


class Router:
    def __init__(self, config: AppConfig):
        self.keep_cycles = config.keep_cycles
        self.model_name = config.model_name
        self.model_routing = config.model_routing

        priority_map: dict[int, list[BaseProvider]] = defaultdict(list)
        for group in sorted(config.providers, key=lambda g: g.priority):
            for backend in group.backends:
                cls = PROVIDER_REGISTRY.get(backend.provider)
                if cls is None:
                    if backend.base_url:
                        cls = PROVIDER_REGISTRY["openai_compat"]
                    else:
                        raise ValueError(
                            f"Unknown provider {backend.provider!r} and no base_url set. "
                            f"Either use a known provider ({', '.join(sorted(PROVIDER_REGISTRY))}) "
                            f"or set base_url for an OpenAI-compatible endpoint."
                        )
                priority_map[group.priority].append(cls(backend))

        self.priority_groups = [priority_map[priority] for priority in sorted(priority_map)]

        model_counter: dict[str, int] = defaultdict(int)
        self._backend_labels: dict[int, str] = {}
        for group in self.priority_groups:
            for backend in group:
                model = backend.config.model
                model_counter[model] += 1
                name = backend.config.name
                if name:
                    label = f"{model}/{name}-{model_counter[model]}"
                else:
                    label = f"{model}-{model_counter[model]}"
                self._backend_labels[id(backend)] = label

        logger.info(
            "Router initialized with %d priority groups, %d total backends",
            len(self.priority_groups),
            sum(len(g) for g in self.priority_groups),
        )

    def _label(self, backend: BaseProvider) -> str:
        return self._backend_labels.get(id(backend), backend.name)

    async def route(
        self,
        messages: list[dict],
        *,
        requested_model: str | None = None,
        request_id: str | None = None,
        **kwargs,
    ) -> RoutedResponse:
        if request_id is None:
            request_id = uuid.uuid4().hex[:8]
        errors: list[tuple[str, Exception]] = []

        for backend in self._iter_attempts(requested_model, request_id):
            label = self._label(backend)
            try:
                logger.debug("[%s] Trying sending request to %s", request_id, label)
                result = await backend.complete(messages, **kwargs)
                if isinstance(result, ProviderResponse):
                    content = result.text
                    message = result.message
                    raw_response = result.raw_response
                else:
                    content = result
                    message = None
                    raw_response = None
                logger.debug("[%s] Success processing with %s", request_id, label)
                return RoutedResponse(
                    content=content,
                    model=backend.config.model,
                    provider_name=backend.name,
                    display_name=backend.config.name or backend.config.provider,
                    message=message,
                    raw_response=raw_response,
                )
            except Exception as e:
                logger.debug(
                    "[%s] Failed to process with %s - %s",
                    request_id,
                    label,
                    _format_error(e),
                )
                errors.append((backend.name, e))

        raise AllProvidersFailedError(errors)

    async def prepare_stream(
        self,
        messages: list[dict],
        *,
        requested_model: str | None = None,
        request_id: str | None = None,
        **kwargs,
    ) -> PreparedStream:
        if request_id is None:
            request_id = uuid.uuid4().hex[:8]
        errors: list[tuple[str, Exception]] = []

        for backend in self._iter_attempts(requested_model, request_id):
            label = self._label(backend)
            try:
                logger.debug("[%s] Trying sending request to %s (stream)", request_id, label)
                stream = aiter(backend.stream(messages, **kwargs))
                buffered: list[str] = []
                while True:
                    try:
                        chunk = await anext(stream)
                    except StopAsyncIteration:
                        break
                    if chunk:
                        buffered.append(chunk)
                        break

                logger.debug("[%s] Success processing with %s (stream)", request_id, label)
                raw_chunks = bool(buffered and isinstance(buffered[0], dict))
                return PreparedStream(
                    model=backend.config.model,
                    provider_name=backend.name,
                    display_name=backend.config.name or backend.config.provider,
                    chunks=self._drain_stream(stream, buffered, backend, request_id),
                    request_id=request_id,
                    raw_chunks=raw_chunks,
                )
            except Exception as e:
                logger.debug(
                    "[%s] Failed to process with %s - %s",
                    request_id,
                    label,
                    _format_error(e),
                )
                errors.append((backend.name, e))

        raise AllProvidersFailedError(errors)

    def _iter_attempts(
        self,
        requested_model: str | None,
        request_id: str,
    ) -> Iterator[BaseProvider]:
        candidate_groups = self._get_candidate_groups(requested_model, request_id)

        for cycle in range(self.keep_cycles):
            if cycle > 0:
                logger.debug(
                    "[%s] Starting cycle %d/%d",
                    request_id,
                    cycle + 1,
                    self.keep_cycles,
                )

            for group in candidate_groups:
                yield from random.sample(group, len(group))

    def _get_candidate_groups(
        self,
        requested_model: str | None,
        request_id: str,
    ) -> list[list[BaseProvider]]:
        if self.model_routing == "any":
            return self.priority_groups

        normalized_model = self._normalize_requested_model(requested_model)
        if normalized_model is None:
            return self.priority_groups

        filtered_groups = [[p for p in group if p.config.model == normalized_model] for group in self.priority_groups]
        filtered_groups = [g for g in filtered_groups if g]
        if not filtered_groups:
            logger.debug(
                "[%s] Model %r not found in backends, falling back to all providers",
                request_id,
                normalized_model,
            )
            return self.priority_groups
        return filtered_groups

    def _normalize_requested_model(self, requested_model: str | None) -> str | None:
        if requested_model is None:
            return None
        normalized = requested_model.strip()
        if not normalized or normalized == self.model_name:
            return None
        return normalized

    async def _drain_stream(
        self,
        stream: AsyncIterator[str | dict[str, Any]],
        buffered: list[str | dict[str, Any]],
        backend: BaseProvider,
        request_id: str,
    ) -> AsyncGenerator[str | dict[str, Any], None]:
        for text in buffered:
            yield text

        try:
            async for chunk in stream:
                if chunk:
                    yield chunk
        except Exception as e:
            label = self._label(backend)
            logger.error(
                "[%s] Streaming interrupted from %s: %s",
                request_id,
                label,
                e,
            )
            raise StreamingProviderError(backend.name) from e
