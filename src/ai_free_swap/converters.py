from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from .models import ChatMessage

_ROLE_MAP = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}


def openai_to_langchain(messages: list[ChatMessage]) -> list[BaseMessage]:
    result: list[BaseMessage] = []
    for msg in messages:
        if msg.role == "tool":
            result.append(
                ToolMessage(
                    content=msg.content or "",
                    name=msg.name,
                    tool_call_id=msg.tool_call_id or msg.name or "tool",
                )
            )
            continue

        cls = _ROLE_MAP[msg.role]
        kwargs = {}
        if msg.name:
            kwargs["name"] = msg.name
        result.append(cls(content=msg.content or "", **kwargs))
    return result
