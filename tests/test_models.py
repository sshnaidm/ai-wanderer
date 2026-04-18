from ai_free_swap.models import make_completion_response, make_stream_chunk


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
