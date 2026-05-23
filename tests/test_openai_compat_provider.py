from __future__ import annotations

from ai_free_swap.config import BackendConfig
from ai_free_swap.providers.openai_compat import OpenAICompatProvider, PROVIDER_BASE_URLS
from ai_free_swap.router import Router

from .conftest import make_config


def _provider(
    *,
    provider: str = "gemini",
    model: str = "gemini-3.5-flash",
    base_url: str | None = None,
    reasoning: bool | None = None,
    extra: dict | None = None,
) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        BackendConfig(
            provider=provider,
            api_key="test-key",
            model=model,
            base_url=(
                base_url
                if base_url is not None
                else "https://generativelanguage.googleapis.com/v1beta/openai/" if provider == "openai_compat" else None
            ),
            reasoning=reasoning,
            extra=extra or {},
        )
    )


def _tool_messages() -> list[dict]:
    return [
        {"role": "user", "content": "Check weather"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city": "SF"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city": "NYC"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "60F"},
        {"role": "tool", "tool_call_id": "call_2", "content": "55F"},
    ]


class TestProviderBaseUrls:
    def test_qwen_uses_international_dashscope_endpoint_by_default(self):
        assert PROVIDER_BASE_URLS["qwen"] == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        assert str(_provider(provider="qwen", model="qwen-flash")._client().base_url) == (
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/"
        )

    def test_qwen_cn_alias_uses_china_dashscope_endpoint(self):
        assert PROVIDER_BASE_URLS["qwen-cn"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        assert str(_provider(provider="qwen-cn", model="qwen-flash")._client().base_url) == (
            "https://dashscope.aliyuncs.com/compatible-mode/v1/"
        )


class TestGeminiThoughtSignatureFallback:
    def test_adds_skip_signature_to_all_missing_gemini_3_tool_calls(self):
        messages = _tool_messages()

        normalized = _provider()._messages_for_provider(messages)

        for tool_call in normalized[1]["tool_calls"]:
            assert tool_call["extra_content"]["google"]["thought_signature"] == "skip_thought_signature_validator"
        assert "extra_content" not in messages[1]["tool_calls"][0]
        assert "extra_content" not in messages[1]["tool_calls"][1]

    def test_preserves_existing_google_thought_signature(self):
        messages = _tool_messages()
        messages[1]["tool_calls"][0]["extra_content"] = {"google": {"thought_signature": "sig_a"}}
        messages[1]["tool_calls"][1]["extra_content"] = {"google": {"thought_signature": "sig_b"}}

        normalized = _provider()._messages_for_provider(messages)

        assert normalized is messages
        assert normalized[1]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "sig_a"
        assert normalized[1]["tool_calls"][1]["extra_content"]["google"]["thought_signature"] == "sig_b"

    def test_patches_later_missing_tool_call_when_first_has_signature(self):
        messages = _tool_messages()
        messages[1]["tool_calls"][0]["extra_content"] = {"google": {"thought_signature": "sig_a"}}

        normalized = _provider()._messages_for_provider(messages)

        assert normalized is not messages
        assert normalized[1]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "sig_a"
        assert (
            normalized[1]["tool_calls"][1]["extra_content"]["google"]["thought_signature"]
            == "skip_thought_signature_validator"
        )

    def test_does_not_modify_non_gemini_3_models(self):
        messages = _tool_messages()

        normalized = _provider(model="gemini-2.5-flash")._messages_for_provider(messages)

        assert normalized is messages
        assert "extra_content" not in normalized[1]["tool_calls"][0]

    def test_handles_google_openai_compat_base_url(self):
        messages = _tool_messages()

        normalized = _provider(provider="openai_compat")._messages_for_provider(messages)

        assert (
            normalized[1]["tool_calls"][0]["extra_content"]["google"]["thought_signature"]
            == "skip_thought_signature_validator"
        )


class TestDeepSeekReasoningExtraBody:
    def test_deepseek_builtin_base_url(self):
        assert PROVIDER_BASE_URLS["deepseek"] == "https://api.deepseek.com"
        assert str(_provider(provider="deepseek", model="deepseek-chat")._client().base_url) == (
            "https://api.deepseek.com"
        )

    def test_adds_reasoning_enabled_for_deepseek_provider(self):
        kwargs = _provider(provider="deepseek", model="deepseek-chat")._split_kwargs({})

        assert kwargs["extra_body"] == {"reasoning": {"enabled": True}}

    def test_deepseek_reasoning_defaults_to_enabled_without_config_setting(self):
        router = Router(make_config([[{"provider": "deepseek", "model": "deepseek-chat"}]]))
        backend = router.priority_groups[0][0]

        assert backend._split_kwargs({})["extra_body"] == {"reasoning": {"enabled": True}}

    def test_adds_reasoning_enabled_for_deepseek_model_on_other_provider(self):
        kwargs = _provider(provider="openrouter", model="deepseek/deepseek-chat-v3.1:free")._split_kwargs({})

        assert kwargs["extra_body"] == {"reasoning": {"enabled": True}}

    def test_global_reasoning_false_controls_deepseek_default(self):
        router = Router(
            make_config(
                [[{"provider": "openrouter", "model": "deepseek/deepseek-chat-v3.1:free"}]],
                reasoning=False,
            )
        )
        backend = router.priority_groups[0][0]

        assert backend._split_kwargs({})["extra_body"] == {"reasoning": {"enabled": False}}
        assert "_global_reasoning" not in backend.config.extra

    def test_backend_reasoning_overrides_global_default(self):
        router = Router(
            make_config(
                [[{"provider": "deepseek", "model": "deepseek-chat", "reasoning": False}]],
                reasoning=True,
            )
        )
        backend = router.priority_groups[0][0]

        assert backend.config.reasoning is False
        assert backend._split_kwargs({})["extra_body"] == {"reasoning": {"enabled": False}}

    def test_preserves_client_reasoning_enabled(self):
        kwargs = _provider(provider="deepseek", model="deepseek-chat")._split_kwargs(
            {"extra_body": {"reasoning": {"enabled": False, "budget": 2048}}}
        )

        assert kwargs["extra_body"]["reasoning"] == {"enabled": False, "budget": 2048}

    def test_merges_enabled_into_client_reasoning_object(self):
        kwargs = _provider(provider="deepseek", model="deepseek-chat")._split_kwargs({"reasoning": {"budget": 2048}})

        assert kwargs["extra_body"]["reasoning"] == {"budget": 2048, "enabled": True}

    def test_configured_extra_body_defaults_support_model_specific_fields(self):
        kwargs = _provider(
            provider="openrouter",
            model="vendor/special-model",
            extra={"extra_body_defaults": {"provider_options": {"foo": True}}},
        )._split_kwargs({})

        assert kwargs["extra_body"] == {"provider_options": {"foo": True}}

    def test_configured_extra_body_defaults_merge_with_built_in_rules(self):
        kwargs = _provider(
            provider="deepseek",
            model="deepseek-chat",
            extra={"extra_body_defaults": {"reasoning": {"budget": 2048}}},
        )._split_kwargs({})

        assert kwargs["extra_body"]["reasoning"] == {"enabled": True, "budget": 2048}

    def test_configured_extra_body_defaults_can_override_built_in_rule_defaults(self):
        kwargs = _provider(
            provider="deepseek",
            model="deepseek-chat",
            extra={"extra_body_defaults": {"reasoning": {"enabled": False}}},
        )._split_kwargs({})

        assert kwargs["extra_body"]["reasoning"] == {"enabled": False}

    def test_client_extra_body_overrides_configured_defaults(self):
        kwargs = _provider(
            provider="openrouter",
            model="vendor/special-model",
            extra={"extra_body_defaults": {"provider_options": {"foo": True, "bar": True}}},
        )._split_kwargs({"extra_body": {"provider_options": {"foo": False}}})

        assert kwargs["extra_body"]["provider_options"] == {"foo": False, "bar": True}

    def test_configured_request_kwargs_defaults_apply_to_known_openai_kwargs(self):
        kwargs = _provider(
            provider="openrouter",
            model="vendor/special-model",
            extra={"request_kwargs_defaults": {"temperature": 0.2, "unknown": "ignored"}},
        )._split_kwargs({})

        assert kwargs == {"temperature": 0.2}

    def test_client_kwargs_override_configured_request_kwargs_defaults(self):
        kwargs = _provider(
            provider="openrouter",
            model="vendor/special-model",
            extra={"request_kwargs_defaults": {"temperature": 0.2}},
        )._split_kwargs({"temperature": 0.8})

        assert kwargs == {"temperature": 0.8}

    def test_does_not_add_reasoning_for_other_providers(self):
        kwargs = _provider(provider="gemini", model="gemini-3.5-flash")._split_kwargs({})

        assert "extra_body" not in kwargs
