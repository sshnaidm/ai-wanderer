from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _normalize_model_name(value: str | None) -> str:
    if value is None:
        return "aifree"
    normalized = value.strip()
    return normalized or "aifree"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str = "user"
    content: Any = None
    name: str | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "aifree"
    messages: list[ChatMessage] = Field(default_factory=list)
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool = False
    stop: list[str] | str | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None

    @field_validator("model", mode="before")
    @classmethod
    def _validate_model(cls, value: Any) -> str:
        if value is None:
            return "aifree"
        return _normalize_model_name(str(value))

    def to_messages(self) -> list[dict[str, Any]]:
        return [message.model_dump(exclude_none=True) for message in self.messages]

    def to_model_kwargs(self) -> dict[str, Any]:
        return self.model_dump(
            exclude_none=True,
            exclude={"model", "messages", "stream"},
        )


class ChoiceMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str = "assistant"
    content: Any = None


class Choice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int = 0
    message: ChoiceMessage
    finish_reason: str | None = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "aifree"
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


def make_completion_response(
    content: Any,
    model: str,
    *,
    message: Mapping[str, Any] | None = None,
    finish_reason: str | None = "stop",
) -> ChatCompletionResponse:
    raw_message = dict(message or {"role": "assistant", "content": content})
    raw_message.setdefault("role", "assistant")
    return ChatCompletionResponse(
        model=model,
        choices=[Choice(message=ChoiceMessage(**raw_message), finish_reason=finish_reason)],
    )


def make_error_response(
    message: str,
    error_type: str,
    *,
    code: str | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"message": message, "type": error_type}
    if code is not None:
        error["code"] = code
    return {"error": error}


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "aifree"
    input: Any = None
    instructions: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    stop: list[str] | str | None = None
    stream: bool = False
    user: str | None = None

    @field_validator("model", mode="before")
    @classmethod
    def _validate_model(cls, value: Any) -> str:
        if value is None:
            return "aifree"
        return _normalize_model_name(str(value))

    def to_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.instructions:
            messages.append({"role": "system", "content": self.instructions})

        if self.input is None:
            return messages
        if isinstance(self.input, str):
            messages.append({"role": "user", "content": self.input})
            return messages
        if isinstance(self.input, list):
            for item in self.input:
                if isinstance(item, dict):
                    messages.append(item)
                else:
                    messages.append({"role": "user", "content": item})
            return messages
        if isinstance(self.input, dict):
            messages.append(self.input)
            return messages

        messages.append({"role": "user", "content": self.input})
        return messages

    def to_model_kwargs(self) -> dict[str, Any]:
        kwargs = self.model_dump(
            exclude_none=True,
            exclude={"model", "input", "instructions", "stream"},
        )
        if "max_output_tokens" in kwargs and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_output_tokens")
        return kwargs


def _response_parts_from_content(content: Any) -> tuple[list[dict[str, Any]], str]:
    if content is None:
        return [], ""
    if isinstance(content, str):
        return [{"type": "output_text", "text": content}], content
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append({"type": "output_text", "text": item})
                text_parts.append(item)
                continue
            if not isinstance(item, dict):
                item_text = str(item)
                parts.append({"type": "output_text", "text": item_text})
                text_parts.append(item_text)
                continue

            part = dict(item)
            item_type = part.get("type")
            if item_type in {"text", "input_text"} and "text" in part:
                part["type"] = "output_text"
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
            parts.append(part)
        return parts, "".join(text_parts)

    coerced = str(content)
    return [{"type": "output_text", "text": coerced}], coerced


def message_to_response_output(message: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    item: dict[str, Any] = {
        "type": "message",
        "role": message.get("role", "assistant"),
        "status": "completed",
    }
    content_parts, output_text = _response_parts_from_content(message.get("content"))
    item["content"] = content_parts
    if "tool_calls" in message:
        item["tool_calls"] = message["tool_calls"]
    if message.get("refusal") is not None:
        item["refusal"] = message["refusal"]
    return item, output_text


def make_responses_response(
    content: Any,
    model: str,
    response_id: str,
    *,
    message: Mapping[str, Any] | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    output_item, output_text = message_to_response_output(message or {"role": "assistant", "content": content})
    output_item["status"] = status
    return {
        "id": response_id,
        "object": "response",
        "created_at": time.time(),
        "model": model,
        "status": status,
        "output": [output_item],
        "output_text": output_text,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }


def make_stream_chunk(
    content: Any,
    request_id: str,
    model: str,
    finish_reason: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


# ---------------------------------------------------------------------------
# Anthropic Messages API models
# ---------------------------------------------------------------------------


class AnthropicMessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "aifree"
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int = 4096
    system: Any = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    metadata: dict[str, Any] | None = None

    @field_validator("model", mode="before")
    @classmethod
    def _validate_model(cls, value: Any) -> str:
        if value is None:
            return "aifree"
        return _normalize_model_name(str(value))

    def to_messages(self) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = []
        if self.system:
            if isinstance(self.system, str):
                msgs.append({"role": "system", "content": self.system})
            elif isinstance(self.system, list):
                text_parts = []
                for block in self.system:
                    if isinstance(block, str):
                        text_parts.append(block)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                msgs.append({"role": "system", "content": "\n\n".join(text_parts)})
        for message in self.messages:
            msgs.append(message.model_dump(exclude_none=True))
        return msgs

    def to_model_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.stop_sequences is not None:
            kwargs["stop"] = self.stop_sequences
        return kwargs


def make_anthropic_response(
    content: Any,
    model: str,
    msg_id: str,
    *,
    stop_reason: str = "end_turn",
) -> dict[str, Any]:
    if isinstance(content, str):
        content_blocks = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        content_blocks = content
    else:
        content_blocks = [{"type": "text", "text": str(content) if content else ""}]
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
        },
    }


def make_anthropic_error_response(
    message: str,
    error_type: str,
) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }
