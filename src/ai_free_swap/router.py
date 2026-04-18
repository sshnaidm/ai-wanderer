from __future__ import annotations

import logging
import random
from collections.abc import AsyncGenerator

from langchain_core.messages import BaseMessage

from .config import AppConfig
from .providers.base import PROVIDER_REGISTRY, BaseProvider

logger = logging.getLogger(__name__)


class AllProvidersFailedError(Exception):
    def __init__(self, errors: list[tuple[str, Exception]]):
        self.errors = errors
        details = "; ".join(f"{name}: {e}" for name, e in errors)
        super().__init__(f"All providers failed: {details}")


class Router:
    def __init__(self, config: AppConfig):
        self.keep_cycles = config.keep_cycles
        self.priority_groups: list[list[BaseProvider]] = []

        groups = sorted(config.providers, key=lambda g: g.priority)
        for group in groups:
            providers = []
            for backend in group.backends:
                cls = PROVIDER_REGISTRY.get(backend.provider)
                if cls is None:
                    raise ValueError(
                        f"Unknown provider {backend.provider!r}. "
                        f"Available: {', '.join(sorted(PROVIDER_REGISTRY))}"
                    )
                providers.append(cls(backend))
            if providers:
                self.priority_groups.append(providers)

        logger.info(
            "Router initialized with %d priority groups, %d total backends",
            len(self.priority_groups),
            sum(len(g) for g in self.priority_groups),
        )

    async def route(self, messages: list[BaseMessage], **kwargs) -> str:
        errors: list[tuple[str, Exception]] = []

        for cycle in range(self.keep_cycles):
            if cycle > 0:
                logger.info("Starting cycle %d/%d", cycle + 1, self.keep_cycles)

            for group in self.priority_groups:
                backends = random.sample(group, len(group))
                for backend in backends:
                    try:
                        logger.info("Trying %s", backend.name)
                        model = backend.create_chat_model()
                        result = await model.ainvoke(messages, **kwargs)
                        logger.info("Success from %s", backend.name)
                        return result.content
                    except Exception as e:
                        logger.warning("Failed %s: %s", backend.name, e)
                        errors.append((backend.name, e))

        raise AllProvidersFailedError(errors)

    async def route_stream(
        self, messages: list[BaseMessage], **kwargs
    ) -> AsyncGenerator[str, None]:
        errors: list[tuple[str, Exception]] = []

        for cycle in range(self.keep_cycles):
            if cycle > 0:
                logger.info("Starting cycle %d/%d", cycle + 1, self.keep_cycles)

            for group in self.priority_groups:
                backends = random.sample(group, len(group))
                for backend in backends:
                    try:
                        logger.info("Trying %s (stream)", backend.name)
                        model = backend.create_chat_model()
                        async for chunk in model.astream(messages, **kwargs):
                            if chunk.content:
                                yield chunk.content
                        logger.info("Stream complete from %s", backend.name)
                        return
                    except Exception as e:
                        logger.warning("Failed %s: %s", backend.name, e)
                        errors.append((backend.name, e))

        raise AllProvidersFailedError(errors)
