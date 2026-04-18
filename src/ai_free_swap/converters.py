from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .models import ChatMessage

_ROLE_MAP = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}


def openai_to_langchain(messages: list[ChatMessage]) -> list[BaseMessage]:
    result: list[BaseMessage] = []
    for msg in messages:
        cls = _ROLE_MAP.get(msg.role, HumanMessage)
        result.append(cls(content=msg.content or ""))
    return result
