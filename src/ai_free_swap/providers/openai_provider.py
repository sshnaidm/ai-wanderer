from __future__ import annotations

from langchain_openai import ChatOpenAI

from .base import BaseProvider, register_provider


@register_provider("openai")
class OpenAIProvider(BaseProvider):
    def create_chat_model(self):
        return ChatOpenAI(
            model=self.config.model,
            api_key=self.config.api_key,
            **self.config.extra,
        )
