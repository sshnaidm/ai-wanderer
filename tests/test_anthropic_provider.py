import json
from types import SimpleNamespace

from ai_free_swap.providers.anthropic_provider import (
    _convert_content,
    _convert_image_url,
    _convert_messages,
    _extract_response,
    _filter_kwargs,
)


class TestConvertImageUrl:
    def test_regular_url(self):
        part = {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}
        result = _convert_image_url(part)
        assert result == {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/cat.png"},
        }

    def test_base64_data_uri(self):
        data_uri = "data:image/png;base64,iVBORw0KGgo="
        part = {"type": "image_url", "image_url": {"url": data_uri}}
        result = _convert_image_url(part)
        assert result == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "iVBORw0KGgo=",
            },
        }

    def test_base64_jpeg(self):
        data_uri = "data:image/jpeg;base64,/9j/4AAQ"
        part = {"type": "image_url", "image_url": {"url": data_uri}}
        result = _convert_image_url(part)
        assert result["source"]["media_type"] == "image/jpeg"
        assert result["source"]["data"] == "/9j/4AAQ"

    def test_base64_webp(self):
        data_uri = "data:image/webp;base64,UklGR"
        result = _convert_image_url({"type": "image_url", "image_url": {"url": data_uri}})
        assert result["source"]["media_type"] == "image/webp"


class TestConvertContent:
    def test_string_passthrough(self):
        assert _convert_content("hello") == "hello"

    def test_none_returns_empty(self):
        assert _convert_content(None) == ""

    def test_text_parts(self):
        content = [{"type": "text", "text": "Hello"}]
        result = _convert_content(content)
        assert result == [{"type": "text", "text": "Hello"}]

    def test_image_url_converted(self):
        content = [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        result = _convert_content(content)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "What is this?"}
        assert result[1] == {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/img.png"},
        }

    def test_mixed_text_and_base64_image(self):
        content = [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
        ]
        result = _convert_content(content)
        assert result[1] == {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "abc123"},
        }

    def test_plain_strings_in_list(self):
        result = _convert_content(["hello", "world"])
        assert result == [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]


class TestConvertMessages:
    def test_image_message_preserved(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            }
        ]
        system, converted = _convert_messages(messages)
        assert system is None
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert len(converted[0]["content"]) == 2
        assert converted[0]["content"][1]["type"] == "image"

    def test_system_message_still_stringified(self):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Be helpful"}]},
            {"role": "user", "content": "Hi"},
        ]
        system, converted = _convert_messages(messages)
        assert system == "Be helpful"
        assert len(converted) == 1
        assert converted[0]["content"] == "Hi"

    def test_tool_message_becomes_native_tool_result(self):
        messages = [
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        ]
        _, converted = _convert_messages(messages)
        assert converted[0]["content"] == [
            {"type": "tool_result", "tool_use_id": "call-1", "content": "result"}
        ]
        assert converted[0]["role"] == "user"

    def test_assistant_tool_calls_become_native_tool_use(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            }
        ]

        _, converted = _convert_messages(messages)

        assert converted[0]["content"] == [
            {
                "type": "tool_use",
                "id": "call-1",
                "name": "weather",
                "input": {"city": "Paris"},
            }
        ]


class TestToolConversion:
    def test_tools_and_choice_are_converted_to_native_shape(self):
        kwargs = _filter_kwargs(
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "description": "Get weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": "weather"}},
            }
        )

        assert kwargs["tools"] == [
            {
                "name": "weather",
                "description": "Get weather",
                "input_schema": {"type": "object"},
            }
        ]
        assert kwargs["tool_choice"] == {"type": "tool", "name": "weather"}

    def test_tool_use_response_is_preserved(self):
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Checking"),
                SimpleNamespace(type="tool_use", id="toolu-1", name="weather", input={"city": "Paris"}),
            ]
        )

        text, message = _extract_response(response)

        assert text == "Checking"
        assert message["tool_calls"][0]["id"] == "toolu-1"
        assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}
