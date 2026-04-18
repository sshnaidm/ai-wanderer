from __future__ import annotations

from langchain_openai import ChatOpenAI

from .base import BaseProvider, register_provider


@register_provider("openai_compat")
class OpenAICompatProvider(BaseProvider):
    """Generic provider for any OpenAI-compatible API (Groq, Together, etc.)."""

    def create_chat_model(self):
        if not self.config.base_url:
            raise ValueError(f"openai_compat provider requires base_url, got none for {self.name}")
        return ChatOpenAI(
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            **self.config.extra,
        )
