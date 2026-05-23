from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI

from .base import BaseProvider, ProviderResponse, register_provider
from .openai_policy import (
    OPENAI_CLIENT_EXTRA_KEYS,
    PROVIDER_BASE_URLS,
    messages_for_provider,
    provider_base_url,
    split_openai_kwargs,
)


def _make_openai_provider(provider_name: str):
    @register_provider(provider_name)
    class _Provider(OpenAICompatProvider):
        pass

    _Provider.__name__ = f"{provider_name.title()}Provider"
    _Provider.__qualname__ = _Provider.__name__
    return _Provider


class OpenAICompatProvider(BaseProvider):
    """Provider for any OpenAI-compatible API."""

    def _messages_for_provider(self, messages: list[dict]) -> list[dict]:
        return messages_for_provider(self.config, messages)

    def _client(self) -> AsyncOpenAI:
        client_kwargs: dict[str, Any] = {
            "api_key": self.config.api_key,
            "base_url": provider_base_url(self.config),
            "max_retries": 0,
        }
        for key in OPENAI_CLIENT_EXTRA_KEYS:
            if key in self.config.extra:
                client_kwargs[key] = self.config.extra[key]
        return AsyncOpenAI(**client_kwargs)

    def _split_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return split_openai_kwargs(self.config, kwargs)

    @staticmethod
    def _extract_text(message: dict[str, Any] | None) -> str:
        if not message:
            return ""
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts)
        return ""

    async def complete(self, messages: list[dict], **kwargs) -> ProviderResponse:
        client = self._client()
        resp = await client.chat.completions.create(
            model=self.config.model,
            messages=self._messages_for_provider(messages),
            stream=False,
            **self._split_kwargs(kwargs),
        )
        raw = resp.model_dump(mode="json", exclude_none=True)
        choices = raw.get("choices") or []
        first_choice = choices[0] if choices else {}
        message = first_choice.get("message") if isinstance(first_choice, dict) else None
        usage = raw.get("usage")
        return ProviderResponse(
            text=self._extract_text(message),
            message=message,
            raw_response=raw,
            usage=usage,
        )

    async def stream(
        self,
        messages: list[dict],
        **kwargs,
    ) -> AsyncGenerator[dict[str, Any], None]:
        client = self._client()
        resp = await client.chat.completions.create(
            model=self.config.model,
            messages=self._messages_for_provider(messages),
            stream=True,
            **self._split_kwargs(kwargs),
        )
        async for chunk in resp:
            yield chunk.model_dump(mode="json", exclude_none=True)


# Register well-known providers with preset base_urls
for _name in PROVIDER_BASE_URLS:
    _make_openai_provider(_name)

# Also register "openai_compat" for custom base_url providers
register_provider("openai_compat")(OpenAICompatProvider)
