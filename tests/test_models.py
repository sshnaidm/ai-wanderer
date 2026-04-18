import pytest

from ai_free_swap.models import (
    ChatCompletionRequest,
    ChatMessage,
    make_completion_response,
    make_error_response,
    make_stream_chunk,
)


class TestMakeCompletionResponse:
    def test_basic_response(self):
        resp = make_completion_response("Hello!", "test-model")
        assert resp.model == "test-model"
        assert resp.object == "chat.completion"
        assert len(resp.choices) == 1
        assert resp.choices[0].message.role == "assistant"
        assert resp.choices[0].message.content == "Hello!"
        assert resp.choices[0].finish_reason == "stop"
        assert resp.id.startswith("chatcmpl-")

    def test_usage_defaults_to_zero(self):
        resp = make_completion_response("Hi", "m")
        assert resp.usage.prompt_tokens == 0
        assert resp.usage.completion_tokens == 0
        assert resp.usage.total_tokens == 0


class TestMakeStreamChunk:
    def test_role_chunk(self):
        chunk = make_stream_chunk(None, "req-1", "model-x", role="assistant")
        assert chunk["id"] == "req-1"
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "model-x"
        assert chunk["choices"][0]["delta"] == {"role": "assistant"}
        assert chunk["choices"][0]["finish_reason"] is None

    def test_content_chunk(self):
        chunk = make_stream_chunk("Hello", "req-1", "model-x")
        assert chunk["choices"][0]["delta"] == {"content": "Hello"}

    def test_finish_chunk(self):
        chunk = make_stream_chunk(None, "req-1", "model-x", finish_reason="stop")
        assert chunk["choices"][0]["delta"] == {}
        assert chunk["choices"][0]["finish_reason"] == "stop"


class TestMakeErrorResponse:
    def test_error_shape(self):
        error = make_error_response("boom", "server_error", code="all_failed")
        assert error == {
            "error": {
                "message": "boom",
                "type": "server_error",
                "code": "all_failed",
            }
        }


class TestChatMessageValidation:
    def test_tool_message_requires_identifier(self):
        with pytest.raises(ValueError, match="tool messages require tool_call_id or name"):
            ChatMessage(role="tool", content="result")

    def test_tool_message_accepts_tool_call_id(self):
        message = ChatMessage(role="tool", content="result", tool_call_id="call-1")
        assert message.tool_call_id == "call-1"


class TestChatCompletionRequestValidation:
    def test_model_must_not_be_empty(self):
        with pytest.raises(ValueError, match="model must not be empty"):
            ChatCompletionRequest(
                model="   ",
                messages=[ChatMessage(role="user", content="Hi")],
            )
