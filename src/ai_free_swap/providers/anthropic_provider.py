from __future__ import annotations

from collections.abc import AsyncGenerator

import anthropic

from .base import BaseProvider, register_provider


def _convert_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Split OpenAI messages into Anthropic system + messages."""
    system = None
    converted = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg.get("content") or ""
        else:
            role = "user" if msg["role"] != "assistant" else "assistant"
            converted.append({"role": role, "content": msg.get("content") or ""})
    return system, converted


_PASSTHROUGH_KWARGS = {"temperature", "top_p", "top_k", "stop", "max_tokens"}
_IGNORED_KWARGS = {"presence_penalty", "frequency_penalty", "user", "n"}


def _filter_kwargs(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if k in _PASSTHROUGH_KWARGS}


def _extract_text(response) -> str:
    parts = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
    return "".join(parts)


@register_provider("anthropic")
class AnthropicProvider(BaseProvider):
    def _client(self) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(api_key=self.config.api_key, max_retries=0)

    @property
    def _default_max_tokens(self) -> int:
        return self.config.extra.get("default_max_tokens", 4096)

    async def complete(self, messages: list[dict], **kwargs) -> str:
        client = self._client()
        system, msgs = _convert_messages(messages)
        filtered = _filter_kwargs(kwargs)
        filtered.setdefault("max_tokens", self._default_max_tokens)
        resp = await client.messages.create(
            model=self.config.model,
            system=system or anthropic.NOT_GIVEN,
            messages=msgs,
            **filtered,
        )
        return _extract_text(resp)

    async def stream(self, messages: list[dict], **kwargs) -> AsyncGenerator[str, None]:
        client = self._client()
        system, msgs = _convert_messages(messages)
        filtered = _filter_kwargs(kwargs)
        filtered.setdefault("max_tokens", self._default_max_tokens)
        async with client.messages.stream(
            model=self.config.model,
            system=system or anthropic.NOT_GIVEN,
            messages=msgs,
            **filtered,
        ) as stream:
            async for text in stream.text_stream:
                yield text
