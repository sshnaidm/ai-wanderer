from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _validate_tool_message(self) -> ChatMessage:
        if self.role == "tool" and not (self.tool_call_id or self.name):
            raise ValueError("tool messages require tool_call_id or name")
        return self


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = Field(default=1, ge=1)
    stream: bool = False
    stop: list[str] | str | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model must not be empty")
        return value


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str | None = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "ai-free-swap"
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


def make_completion_response(content: str, model: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model=model,
        choices=[Choice(message=ChoiceMessage(content=content))],
    )


def make_error_response(
    message: str,
    error_type: str,
    *,
    code: str | None = None,
) -> dict:
    error = {"message": message, "type": error_type}
    if code is not None:
        error["code"] = code
    return {"error": error}


def make_stream_chunk(
    content: str | None,
    request_id: str,
    model: str,
    finish_reason: str | None = None,
    role: str | None = None,
) -> dict:
    delta = {}
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
