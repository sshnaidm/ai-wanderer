from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from .base import BaseProvider, register_provider


@register_provider("gemini")
class GeminiProvider(BaseProvider):
    def create_chat_model(self):
        return ChatGoogleGenerativeAI(
            model=self.config.model,
            google_api_key=self.config.api_key,
            **self.config.extra,
        )
