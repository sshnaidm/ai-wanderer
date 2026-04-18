from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from langchain_core.language_models.chat_models import BaseChatModel

from ..config import BackendConfig

PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {}


def register_provider(name: str):
    def decorator(cls: type[BaseProvider]):
        PROVIDER_REGISTRY[name] = cls
        return cls
    return decorator


class BaseProvider(ABC):
    provider_name: ClassVar[str]

    def __init__(self, config: BackendConfig):
        self.config = config

    @abstractmethod
    def create_chat_model(self) -> BaseChatModel: ...

    @property
    def name(self) -> str:
        return f"{self.config.provider}({self.config.model})"
