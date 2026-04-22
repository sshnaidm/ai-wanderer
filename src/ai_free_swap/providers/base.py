from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from ..config import BackendConfig

PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {}


def register_provider(name: str):
    def decorator(cls: type[BaseProvider]):
        PROVIDER_REGISTRY[name] = cls
        return cls
    return decorator


class BaseProvider(ABC):
    def __init__(self, config: BackendConfig):
        self.config = config

    @abstractmethod
    async def complete(
        self, messages: list[dict], **kwargs
    ) -> str:
        """Return the assistant's text response."""

    @abstractmethod
    async def stream(
        self, messages: list[dict], **kwargs
    ) -> AsyncGenerator[str, None]:
        """Yield text chunks."""

    @property
    def name(self) -> str:
        key_suffix = self.config.api_key[-4:] if len(self.config.api_key) >= 4 else "***"
        return f"{self.config.provider}({self.config.model}@..{key_suffix})"
