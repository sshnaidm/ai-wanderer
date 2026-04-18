from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai_free_swap.converters import openai_to_langchain
from ai_free_swap.models import ChatMessage


class TestOpenaiToLangchain:
    def test_system_message(self):
        msgs = openai_to_langchain([ChatMessage(role="system", content="You are helpful")])
        assert len(msgs) == 1
        assert isinstance(msgs[0], SystemMessage)
        assert msgs[0].content == "You are helpful"

    def test_user_message(self):
        msgs = openai_to_langchain([ChatMessage(role="user", content="Hello")])
        assert len(msgs) == 1
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "Hello"

    def test_assistant_message(self):
        msgs = openai_to_langchain([ChatMessage(role="assistant", content="Hi there")])
        assert len(msgs) == 1
        assert isinstance(msgs[0], AIMessage)

    def test_multi_turn_conversation(self):
        msgs = openai_to_langchain([
            ChatMessage(role="system", content="Be brief"),
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello"),
            ChatMessage(role="user", content="Bye"),
        ])
        assert len(msgs) == 4
        assert isinstance(msgs[0], SystemMessage)
        assert isinstance(msgs[1], HumanMessage)
        assert isinstance(msgs[2], AIMessage)
        assert isinstance(msgs[3], HumanMessage)

    def test_none_content_becomes_empty_string(self):
        msgs = openai_to_langchain([ChatMessage(role="user", content=None)])
        assert msgs[0].content == ""

    def test_unknown_role_defaults_to_human(self):
        msgs = openai_to_langchain([ChatMessage(role="tool", content="result")])
        assert isinstance(msgs[0], HumanMessage)
