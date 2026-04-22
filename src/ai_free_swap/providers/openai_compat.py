from __future__ import annotations

from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from .base import BaseProvider, register_provider

PROVIDER_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "grok": "https://api.x.ai/v1",
    "openai": None,
    "openrouter": "https://openrouter.ai/api/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
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
        return AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=base_url,
            max_retries=0,
        )

    async def complete(self, messages: list[dict], **kwargs) -> str:
        client = self._client()
        resp = await client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            stream=False,
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    async def stream(self, messages: list[dict], **kwargs) -> AsyncGenerator[str, None]:
        client = self._client()
        resp = await client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            stream=True,
            **kwargs,
        )
        async for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# Register well-known providers with preset base_urls
for _name in PROVIDER_BASE_URLS:
    _make_openai_provider(_name)

# Also register "openai_compat" for custom base_url providers
register_provider("openai_compat")(OpenAICompatProvider)
