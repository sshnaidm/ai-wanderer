from ai_free_swap.models import (
    ChatCompletionRequest,
    ChatMessage,
    ResponsesRequest,
    make_anthropic_response,
    make_completion_response,
    make_error_response,
    make_responses_response,
    make_stream_chunk,
    message_to_response_output,
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

    def test_preserves_message_shape(self):
        resp = make_completion_response(
            None,
            "test-model",
            message={
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello"}],
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
        )
        assert resp.choices[0].message.content == [{"type": "text", "text": "Hello"}]
        assert resp.choices[0].message.tool_calls == [{"id": "call-1", "type": "function"}]

    def test_propagates_usage_when_available(self):
        resp = make_completion_response(
            "Hello",
            "test-model",
            usage={"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
        )
        assert resp.usage.prompt_tokens == 7
        assert resp.usage.completion_tokens == 5
        assert resp.usage.total_tokens == 12


class TestOtherResponseBuilders:
    def test_responses_builder_propagates_usage(self):
        resp = make_responses_response(
            "Hello",
            "test-model",
            "resp_123",
            usage={"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
        )
        assert resp["usage"] == {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15}

    def test_anthropic_builder_propagates_usage(self):
        resp = make_anthropic_response(
            "Hello",
            "test-model",
            "msg_123",
            usage={"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
        )
        assert resp["usage"] == {"input_tokens": 9, "output_tokens": 3}


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


class TestChatMessageModel:
    def test_allows_developer_role_and_extra_fields(self):
        message = ChatMessage(
            role="developer",
            content=[{"type": "text", "text": "Be brief"}],
            tool_calls=[{"id": "call-1"}],
        )
        dumped = message.model_dump(exclude_none=True)
        assert dumped["role"] == "developer"
        assert dumped["content"] == [{"type": "text", "text": "Be brief"}]
        assert dumped["tool_calls"] == [{"id": "call-1"}]


class TestChatCompletionRequest:
    def test_blank_model_falls_back_to_aifree(self):
        request = ChatCompletionRequest(
            model="   ",
            messages=[ChatMessage(role="user", content="Hi")],
        )
        assert request.model == "aifree"

    def test_to_model_kwargs_preserves_unknown_fields(self):
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
            tools=[{"type": "function", "function": {"name": "ping"}}],
            temperatur=0.7,
        )
        kwargs = request.to_model_kwargs()
        assert kwargs["tools"] == [{"type": "function", "function": {"name": "ping"}}]
        assert kwargs["temperatur"] == 0.7


class TestResponsesRequest:
    def test_to_messages_accepts_non_dict_items(self):
        request = ResponsesRequest(model="test-model", input=["Hello", {"role": "assistant", "content": "Hi"}])
        assert request.to_messages() == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

    def test_structured_content_is_converted_to_chat_shape(self):
        request = ResponsesRequest(
            model="test-model",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this"},
                        {"type": "input_image", "image_url": "https://example.com/image.png"},
                        {"type": "input_file", "file_id": "file-1"},
                    ],
                }
            ],
        )

        assert request.to_messages() == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                    {"type": "file", "file": {"file_id": "file-1"}},
                ],
            }
        ]

    def test_function_items_and_tools_are_converted(self):
        request = ResponsesRequest(
            model="test-model",
            input=[
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "weather",
                    "arguments": '{"city":"Paris"}',
                },
                {"type": "function_call_output", "call_id": "call-1", "output": "sunny"},
            ],
            tools=[
                {
                    "type": "function",
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                }
            ],
            tool_choice={"type": "function", "name": "weather"},
        )

        messages = request.to_messages()
        kwargs = request.to_model_kwargs()

        assert messages[0]["tool_calls"][0]["function"]["name"] == "weather"
        assert messages[1] == {"role": "tool", "tool_call_id": "call-1", "content": "sunny"}
        assert kwargs["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                },
            }
        ]
        assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "weather"}}

    def test_blank_model_falls_back_to_aifree(self):
        request = ResponsesRequest(model="   ", input="Hello")
        assert request.model == "aifree"


class TestResponseOutputMapping:
    def test_message_to_response_output_preserves_tool_calls(self):
        item, output_text = message_to_response_output(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello"}],
                "tool_calls": [{"id": "call-1", "type": "function"}],
            }
        )
        assert item["tool_calls"] == [{"id": "call-1", "type": "function"}]
        assert item["content"] == [{"type": "output_text", "text": "Hello"}]
        assert output_text == "Hello"
