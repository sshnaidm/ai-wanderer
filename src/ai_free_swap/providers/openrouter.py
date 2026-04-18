from __future__ import annotations

from langchain_openrouter import ChatOpenRouter

from .base import BaseProvider, register_provider


@register_provider("openrouter")
class OpenRouterProvider(BaseProvider):
    def create_chat_model(self):
        return ChatOpenRouter(
            model=self.config.model,
            openrouter_api_key=self.config.api_key,
            **self.config.extra,
        )
