from __future__ import annotations

from langchain_qwq import ChatQwen

from .base import BaseProvider, register_provider


@register_provider("qwen")
class QwenProvider(BaseProvider):
    def create_chat_model(self):
        return ChatQwen(
            model=self.config.model,
            dashscope_api_key=self.config.api_key,
            **self.config.extra,
        )
