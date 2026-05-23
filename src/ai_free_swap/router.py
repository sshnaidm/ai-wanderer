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


@dataclass
class BackendMetrics:
    key: str
    label: str
    provider: str
    model: str
    name: str
    priority: int
    limits: dict[str, int | None]
    active: int = 0
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    rate_limited_skips: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    last_latency_ms: float | None = None
    last_started_at: float | None = None
    last_finished_at: float | None = None
    last_success_at: float | None = None
    last_failure_at: float | None = None
    last_error: str = ""

    def as_dict(
        self,
        rate_counters: dict[str, dict[str, int]] | None = None,
        *,
        rate_limited: bool = False,
    ) -> dict[str, Any]:
        success_rate = (self.successes / self.attempts * 100) if self.attempts else None
        avg_latency = self.total_latency_ms / self.successes if self.successes else None
        status = "idle"
        if self.active:
            status = "running"
        elif rate_limited:
            status = "limited"
        elif self.last_error and (not self.last_success_at or (self.last_failure_at or 0) >= self.last_success_at):
            status = "failing"
        elif self.successes:
            status = "healthy"
        return {
            "key": self.key,
            "label": self.label,
            "provider": self.provider,
            "model": self.model,
            "name": self.name,
            "priority": self.priority,
            "limits": self.limits,
            "status": status,
            "rate_limited": rate_limited,
            "active": self.active,
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "rate_limited_skips": self.rate_limited_skips,
            "success_rate": success_rate,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "avg_latency_ms": avg_latency,
            "last_latency_ms": self.last_latency_ms,
            "last_started_at": self.last_started_at,
            "last_finished_at": self.last_finished_at,
            "last_success_at": self.last_success_at,
            "last_failure_at": self.last_failure_at,
            "last_error": self.last_error,
            "rate_counters": rate_counters or {},
        }


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
        self._started_at = time.time()

        priority_map: dict[int, list[BaseProvider]] = defaultdict(list)
        backend_priorities: dict[int, int] = {}
        for group in sorted(config.providers, key=lambda g: g.priority):
            for backend in group.backends:
                backend = backend.model_copy(
                    update={"extra": {**backend.extra, "_global_reasoning": config.reasoning}}
                )
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
                instance = cls(backend)
                priority_map[group.priority].append(instance)
                backend_priorities[id(instance)] = group.priority

        self.priority_groups = [priority_map[priority] for priority in sorted(priority_map)]

        model_counter: dict[str, int] = defaultdict(int)
        rate_key_counter: dict[str, int] = defaultdict(int)
        self._backend_labels: dict[int, str] = {}
        self._backend_rate_keys: dict[int, str] = {}
        self._backend_limits: dict[str, RateLimits] = {}
        self._backend_metrics: dict[str, BackendMetrics] = {}
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
                self._backend_metrics[rate_key] = BackendMetrics(
                    key=rate_key,
                    label=label,
                    provider=backend.config.provider,
                    model=backend.config.model,
                    name=backend.config.name or "",
                    priority=backend_priorities[id(backend)],
                    limits=backend.config.limits.model_dump() if backend.config.limits else {},
                )

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

    def dashboard_snapshot(self) -> dict[str, Any]:
        rate_data = self._rate_limiter.counters_snapshot()
        backends = [
            metrics.as_dict(
                rate_data.get(key, {}),
                rate_limited=bool(
                    (limits := self._backend_limits.get(key)) and not self._rate_limiter.is_allowed(key, limits)
                ),
            )
            for key, metrics in sorted(
                self._backend_metrics.items(),
                key=lambda item: (item[1].priority, item[1].model, item[1].label),
            )
        ]
        totals = {
            "backends": len(backends),
            "active": sum(item["active"] for item in backends),
            "attempts": sum(item["attempts"] for item in backends),
            "successes": sum(item["successes"] for item in backends),
            "failures": sum(item["failures"] for item in backends),
            "rate_limited_skips": sum(item["rate_limited_skips"] for item in backends),
            "prompt_tokens": sum(item["prompt_tokens"] for item in backends),
            "completion_tokens": sum(item["completion_tokens"] for item in backends),
            "total_tokens": sum(item["total_tokens"] for item in backends),
        }
        if totals["attempts"]:
            totals["success_rate"] = totals["successes"] / totals["attempts"] * 100
            totals["failure_rate"] = totals["failures"] / totals["attempts"] * 100
        else:
            totals["success_rate"] = None
            totals["failure_rate"] = None
        return {
            "generated_at": time.time(),
            "started_at": self._started_at,
            "model_name": self.model_name,
            "model_routing": self.model_routing,
            "keep_cycles": self.keep_cycles,
            "totals": totals,
            "backends": backends,
        }

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

    def _metrics(self, backend: BaseProvider) -> BackendMetrics:
        return self._backend_metrics[self._rate_key(backend)]

    def _record_backend_start(self, backend: BaseProvider) -> None:
        metrics = self._metrics(backend)
        metrics.active += 1
        metrics.attempts += 1
        metrics.last_started_at = time.time()

    def _record_backend_success(
        self,
        backend: BaseProvider,
        elapsed_seconds: float,
        usage: dict[str, int] | None,
    ) -> None:
        metrics = self._metrics(backend)
        metrics.active = max(0, metrics.active - 1)
        metrics.successes += 1
        metrics.last_error = ""
        metrics.last_latency_ms = elapsed_seconds * 1000
        metrics.total_latency_ms += metrics.last_latency_ms
        now = time.time()
        metrics.last_finished_at = now
        metrics.last_success_at = now
        if usage:
            prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
            completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
            total = usage.get("total_tokens", prompt + completion)
            metrics.prompt_tokens += prompt
            metrics.completion_tokens += completion
            metrics.total_tokens += total

    def _record_backend_failure(self, backend: BaseProvider, elapsed_seconds: float, error: Exception) -> None:
        metrics = self._metrics(backend)
        metrics.active = max(0, metrics.active - 1)
        metrics.failures += 1
        metrics.last_latency_ms = elapsed_seconds * 1000
        now = time.time()
        metrics.last_finished_at = now
        metrics.last_failure_at = now
        metrics.last_error = _format_error(error)

    def _record_rate_limited_skip(self, backend: BaseProvider) -> None:
        self._metrics(backend).rate_limited_skips += 1

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
            t0 = time.monotonic()
            self._record_backend_start(backend)
            try:
                logger.debug("[%s] Trying sending request to %s", request_id, label)
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
                self._record_backend_success(backend, elapsed, usage)
                if limits:
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
                self._record_backend_failure(backend, time.monotonic() - t0, e)
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
            t0 = time.monotonic()
            self._record_backend_start(backend)
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
                self._record_backend_failure(backend, time.monotonic() - t0, e)
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
                        self._record_rate_limited_skip(backend)
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
        stream_error: Exception | None = None

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
            stream_error = e
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
            if stream_error:
                self._record_backend_failure(backend, elapsed, stream_error)
            else:
                self._record_backend_success(backend, elapsed, usage)
            if limits:
                await self._save_state_if_due()
