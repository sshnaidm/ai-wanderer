from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator
from typing import Any

import anthropic

from .base import BaseProvider, ProviderResponse, register_provider

_DATA_URI_RE = re.compile(r"^data:(image/[^;]+);base64,(.+)$", re.DOTALL)


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(json.dumps(item, ensure_ascii=True))
        return "".join(parts)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        return json.dumps(content, ensure_ascii=True)
    return str(content)


def _convert_image_url(part: dict) -> dict:
    """Convert an OpenAI image_url content part to an Anthropic image block."""
    image_url = part.get("image_url", {})
    url = image_url.get("url", "") if isinstance(image_url, dict) else ""
    match = _DATA_URI_RE.match(url)
    if match:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": match.group(1),
                "data": match.group(2),
            },
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _convert_content(content: Any) -> str | list[dict]:
    """Convert OpenAI content to Anthropic format, preserving images."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[dict] = []
        for item in content:
            if isinstance(item, str):
                parts.append({"type": "text", "text": item})
            elif not isinstance(item, dict):
                parts.append({"type": "text", "text": str(item)})
            elif item.get("type") == "image_url":
                parts.append(_convert_image_url(item))
            elif item.get("type") == "text":
                parts.append({"type": "text", "text": item.get("text", "")})
            else:
                parts.append({"type": "text", "text": json.dumps(item, ensure_ascii=True)})
        return parts
    return _stringify_content(content)


def _convert_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Map OpenAI-style messages into Anthropic's system + message format."""
    system_parts: list[str] = []
    converted = []
    for msg in messages:
        role = msg.get("role", "user")
        if role in {"system", "developer"}:
            text = _stringify_content(msg.get("content"))
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            identifier = msg.get("tool_call_id") or msg.get("name") or "tool"
            text = _stringify_content(msg.get("content"))
            content: str | list[dict] = f"[{identifier}] {text}" if text else f"[{identifier}]"
        else:
            content = _convert_content(msg.get("content"))
        converted.append(
            {
                "role": "assistant" if role == "assistant" else "user",
                "content": content,
            }
        )
    system = "\n\n".join(part for part in system_parts if part) or None
    return system, converted


_PASSTHROUGH_KWARGS = {"temperature", "top_p", "top_k", "stop", "max_tokens"}


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
        client_kwargs: dict[str, Any] = {
            "api_key": self.config.api_key,
            "max_retries": 0,
        }
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        if "timeout" in self.config.extra:
            client_kwargs["timeout"] = self.config.extra["timeout"]
        return anthropic.AsyncAnthropic(**client_kwargs)

    @property
    def _default_max_tokens(self) -> int:
        return self.config.extra.get("default_max_tokens", 4096)

    async def complete(self, messages: list[dict], **kwargs) -> ProviderResponse:
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
        text = _extract_text(resp)
        return ProviderResponse(
            text=text,
            message={"role": "assistant", "content": text},
        )

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
