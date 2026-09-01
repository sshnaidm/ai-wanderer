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
            content: str | list[dict] = [
                {
                    "type": "tool_result",
                    "tool_use_id": identifier,
                    "content": text,
                }
            ]
        elif role == "assistant" and msg.get("tool_calls"):
            content = []
            text = _stringify_content(msg.get("content"))
            if text:
                content.append({"type": "text", "text": text})
            for tool_call in msg["tool_calls"]:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function", {})
                arguments = function.get("arguments", {}) if isinstance(function, dict) else {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}
                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.get("id", ""),
                        "name": function.get("name", "") if isinstance(function, dict) else "",
                        "input": arguments,
                    }
                )
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


def _convert_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if "input_schema" in tool and "name" in tool:
            converted.append(dict(tool))
            continue
        function = tool.get("function")
        if tool.get("type") != "function" or not isinstance(function, dict):
            continue
        native: dict[str, Any] = {
            "name": function.get("name", ""),
            "input_schema": function.get("parameters", {"type": "object", "properties": {}}),
        }
        if function.get("description") is not None:
            native["description"] = function["description"]
        if function.get("strict") is not None:
            native["strict"] = function["strict"]
        converted.append(native)
    return converted


def _convert_tool_choice(choice: Any) -> Any:
    if isinstance(choice, str):
        return {"type": {"required": "any"}.get(choice, choice)}
    if not isinstance(choice, dict):
        return choice
    if choice.get("type") == "function":
        function = choice.get("function", {})
        return {
            "type": "tool",
            "name": function.get("name", "") if isinstance(function, dict) else "",
        }
    return choice


def _filter_kwargs(kwargs: dict) -> dict:
    filtered = {k: v for k, v in kwargs.items() if k in _PASSTHROUGH_KWARGS}
    tools = _convert_tools(kwargs.get("tools"))
    if tools:
        filtered["tools"] = tools
    if kwargs.get("tool_choice") is not None:
        filtered["tool_choice"] = _convert_tool_choice(kwargs["tool_choice"])
    return filtered


def _extract_response(response) -> tuple[str, dict[str, Any]]:
    parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input, ensure_ascii=True),
                    },
                }
            )
    text = "".join(parts)
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return text, message


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
        text, message = _extract_response(resp)
        usage = None
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
                "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
            }
        return ProviderResponse(
            text=text,
            message=message,
            usage=usage,
        )

    async def stream(self, messages: list[dict], **kwargs) -> AsyncGenerator[str | dict[str, Any], None]:
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
            prompt_tokens = 0
            async for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "message_start":
                    event_usage = getattr(getattr(event, "message", None), "usage", None)
                    prompt_tokens = getattr(event_usage, "input_tokens", 0) or 0
                    continue
                if event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if getattr(block, "type", "") == "tool_use":
                        yield {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": event.index,
                                                "id": block.id,
                                                "type": "function",
                                                "function": {
                                                    "name": block.name,
                                                    "arguments": "",
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ]
                        }
                    continue
                if event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    delta_type = getattr(delta, "type", "")
                    if delta_type == "text_delta":
                        yield delta.text
                    elif delta_type == "input_json_delta":
                        yield {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": event.index,
                                                "function": {"arguments": delta.partial_json},
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ]
                        }
                    continue
                if event_type == "message_delta":
                    event_usage = getattr(event, "usage", None)
                    output_tokens = getattr(event_usage, "output_tokens", 0) or 0
                    stop_reason = getattr(getattr(event, "delta", None), "stop_reason", None)
                    yield {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "tool_calls" if stop_reason == "tool_use" else stop_reason,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": output_tokens,
                            "total_tokens": prompt_tokens + output_tokens,
                        },
                    }
