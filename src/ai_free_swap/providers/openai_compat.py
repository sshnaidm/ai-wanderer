from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI

from .base import BaseProvider, ProviderResponse, register_provider

PROVIDER_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "grok": "https://api.x.ai/v1",
    "openai": None,
    "openrouter": "https://openrouter.ai/api/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

_OPENAI_CLIENT_EXTRA_KEYS = {"timeout"}
_OPENAI_CHAT_KNOWN_ARGS = {
    "audio",
    "frequency_penalty",
    "function_call",
    "functions",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "metadata",
    "modalities",
    "n",
    "parallel_tool_calls",
    "prediction",
    "presence_penalty",
    "reasoning_effort",
    "response_format",
    "seed",
    "service_tier",
    "stop",
    "store",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "user",
    "web_search_options",
}


def _make_openai_provider(provider_name: str):
    @register_provider(provider_name)
    class _Provider(OpenAICompatProvider):
        pass
    _Provider.__name__ = f"{provider_name.title()}Provider"
    _Provider.__qualname__ = _Provider.__name__
    return _Provider


class OpenAICompatProvider(BaseProvider):
    """Provider for any OpenAI-compatible API."""

    def _client(self) -> AsyncOpenAI:
        base_url = self.config.base_url or PROVIDER_BASE_URLS.get(self.config.provider)
        client_kwargs: dict[str, Any] = {
            "api_key": self.config.api_key,
            "base_url": base_url,
            "max_retries": 0,
        }
        for key in _OPENAI_CLIENT_EXTRA_KEYS:
            if key in self.config.extra:
                client_kwargs[key] = self.config.extra[key]
        return AsyncOpenAI(**client_kwargs)

    def _split_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        known_kwargs: dict[str, Any] = {}
        extra_body: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in _OPENAI_CHAT_KNOWN_ARGS or key.startswith("extra_"):
                known_kwargs[key] = value
            else:
                extra_body[key] = value
        if extra_body:
            known_kwargs["extra_body"] = extra_body
        return known_kwargs

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
            messages=messages,
            stream=False,
            **self._split_kwargs(kwargs),
        )
        raw = resp.model_dump(mode="json", exclude_none=True)
        choices = raw.get("choices") or []
        first_choice = choices[0] if choices else {}
        message = first_choice.get("message") if isinstance(first_choice, dict) else None
        return ProviderResponse(
            text=self._extract_text(message),
            message=message,
            raw_response=raw,
        )

    async def stream(
        self,
        messages: list[dict],
        **kwargs,
    ) -> AsyncGenerator[dict[str, Any], None]:
        client = self._client()
        resp = await client.chat.completions.create(
            model=self.config.model,
            messages=messages,
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
