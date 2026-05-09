from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

from .config import AppConfig, RateLimits
from .providers.base import PROVIDER_REGISTRY, BaseProvider, ProviderResponse
from .rate_limiter import RateLimiter

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
    usage: dict[str, int] | None = None


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
    def __init__(self, config: AppConfig, *, state_file: str | None = None):
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
        rate_key_counter: dict[str, int] = defaultdict(int)
        self._backend_labels: dict[int, str] = {}
        self._backend_rate_keys: dict[int, str] = {}
        self._backend_limits: dict[str, RateLimits] = {}
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
                base_rate_key = self._make_rate_key(backend)
                rate_key_counter[base_rate_key] += 1
                rate_key = f"{base_rate_key}:{rate_key_counter[base_rate_key]}"
                self._backend_rate_keys[id(backend)] = rate_key
                if backend.config.limits:
                    self._backend_limits[rate_key] = backend.config.limits

        has_limits = bool(self._backend_limits)
        self._rate_limiter = RateLimiter(state_file=state_file if has_limits else None)

        logger.info(
            "Router initialized with %d priority groups, %d total backends",
            len(self.priority_groups),
            sum(len(g) for g in self.priority_groups),
        )

    def _label(self, backend: BaseProvider) -> str:
        return self._backend_labels.get(id(backend), backend.name)

    def _rate_key(self, backend: BaseProvider) -> str:
        return self._backend_rate_keys.get(id(backend), self._make_rate_key(backend))

    def save_state(self) -> None:
        self._rate_limiter.save()

    async def _save_state_if_due(self) -> None:
        snapshot = self._rate_limiter.snapshot_if_due()
        if not snapshot:
            return
        state_file, data = snapshot
        try:
            await asyncio.to_thread(self._rate_limiter.write_snapshot, state_file, data)
        except OSError as e:
            self._rate_limiter.mark_dirty()
            logger.warning("Failed to save rate limiter state: %s", e)

    @staticmethod
    def _make_rate_key(backend: BaseProvider) -> str:
        config = backend.config
        identity = {
            "provider": config.provider,
            "model": config.model,
            "base_url": config.base_url or "",
            "name": config.name or "",
            "has_limits": bool(config.limits),
            "api_key_sha256": hashlib.sha256(config.api_key.encode("utf-8")).hexdigest(),
        }
        raw = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        readable = config.name or config.provider
        return f"{config.provider}:{config.model}:{readable}:{digest}"

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
                t0 = time.monotonic()
                result = await backend.complete(messages, **kwargs)
                elapsed = time.monotonic() - t0
                if isinstance(result, ProviderResponse):
                    content = result.text
                    message = result.message
                    raw_response = result.raw_response
                    usage = result.usage
                else:
                    content = result
                    message = None
                    raw_response = None
                    usage = None
                usage_str = ""
                if usage:
                    usage_str = " tokens=%s/%s/%s" % (
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                        usage.get("total_tokens", 0),
                    )
                logger.info(
                    "[%s] Completed with %s in %.2fs%s",
                    request_id,
                    label,
                    elapsed,
                    usage_str,
                )
                rate_key = self._rate_key(backend)
                limits = self._backend_limits.get(rate_key)
                if limits:
                    self._rate_limiter.record_request(rate_key)
                    if usage:
                        total = usage.get("total_tokens", 0)
                        if total:
                            self._rate_limiter.record_tokens(rate_key, total)
                    await self._save_state_if_due()
                return RoutedResponse(
                    content=content,
                    model=backend.config.model,
                    provider_name=backend.name,
                    display_name=backend.config.name or backend.config.provider,
                    message=message,
                    raw_response=raw_response,
                    usage=usage,
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
                t0 = time.monotonic()
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

                ttfb = time.monotonic() - t0
                logger.info(
                    "[%s] Stream started from %s (ttfb=%.2fs)",
                    request_id,
                    label,
                    ttfb,
                )
                rate_key = self._rate_key(backend)
                limits = self._backend_limits.get(rate_key)
                if limits:
                    self._rate_limiter.record_request(rate_key)
                raw_chunks = bool(buffered and isinstance(buffered[0], dict))
                return PreparedStream(
                    model=backend.config.model,
                    provider_name=backend.name,
                    display_name=backend.config.name or backend.config.provider,
                    chunks=self._drain_stream(stream, buffered, backend, request_id, t0),
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
                available = []
                for backend in group:
                    rate_key = self._rate_key(backend)
                    limits = self._backend_limits.get(rate_key)
                    if limits and not self._rate_limiter.is_allowed(rate_key, limits):
                        logger.debug(
                            "[%s] Skipping %s (rate limited)",
                            request_id,
                            self._label(backend),
                        )
                        continue
                    available.append(backend)
                yield from random.sample(available, len(available))

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
        t0: float = 0,
    ) -> AsyncGenerator[str | dict[str, Any], None]:
        label = self._label(backend)
        usage: dict[str, int] | None = None

        for text in buffered:
            if isinstance(text, dict) and text.get("usage"):
                usage = text["usage"]
            yield text

        try:
            async for chunk in stream:
                if chunk:
                    if isinstance(chunk, dict) and chunk.get("usage"):
                        usage = chunk["usage"]
                    yield chunk
        except Exception as e:
            logger.error(
                "[%s] Streaming interrupted from %s: %s",
                request_id,
                label,
                e,
            )
            raise StreamingProviderError(backend.name) from e
        finally:
            elapsed = time.monotonic() - t0 if t0 else 0
            usage_str = ""
            if usage:
                usage_str = " tokens=%s/%s/%s" % (
                    usage.get("prompt_tokens", usage.get("input_tokens", 0)),
                    usage.get("completion_tokens", usage.get("output_tokens", 0)),
                    usage.get(
                        "total_tokens",
                        usage.get("prompt_tokens", usage.get("input_tokens", 0))
                        + usage.get("completion_tokens", usage.get("output_tokens", 0)),
                    ),
                )
            if elapsed:
                logger.info(
                    "[%s] Stream finished from %s in %.2fs%s",
                    request_id,
                    label,
                    elapsed,
                    usage_str,
                )
            rate_key = self._rate_key(backend)
            limits = self._backend_limits.get(rate_key)
            if limits:
                if usage:
                    total = usage.get(
                        "total_tokens",
                        usage.get("prompt_tokens", usage.get("input_tokens", 0))
                        + usage.get("completion_tokens", usage.get("output_tokens", 0)),
                    )
                    if total:
                        self._rate_limiter.record_tokens(rate_key, total)
                await self._save_state_if_due()
