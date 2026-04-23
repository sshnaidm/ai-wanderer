from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from ..config import BackendConfig

PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {}


def register_provider(name: str):
    def decorator(cls: type[BaseProvider]):
        PROVIDER_REGISTRY[name] = cls
        return cls

    return decorator


@dataclass(frozen=True)
class ProviderResponse:
    text: str = ""
    message: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None


class BaseProvider(ABC):
    def __init__(self, config: BackendConfig):
        self.config = config

    @abstractmethod
    async def complete(self, messages: list[dict], **kwargs) -> str | ProviderResponse:
        """Return an assistant response or the raw provider payload."""

    @abstractmethod
    async def stream(self, messages: list[dict], **kwargs) -> AsyncGenerator[str | dict[str, Any], None]:
        """Yield text chunks or raw provider stream payloads."""

    @property
    def name(self) -> str:
        key_suffix = self.config.api_key[-4:] if len(self.config.api_key) >= 4 else "***"
        return f"{self.config.provider}({self.config.model}@..{key_suffix})"
