from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from .base import BaseProvider, register_provider


@register_provider("anthropic")
class AnthropicProvider(BaseProvider):
    def create_chat_model(self):
        return ChatAnthropic(
            model=self.config.model,
            api_key=self.config.api_key,
            **self.config.extra,
        )
