from __future__ import annotations

import copy
from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI

from .base import BaseProvider, ProviderResponse, register_provider

PROVIDER_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "grok": "https://api.x.ai/v1",
    "openai": None,
    "openrouter": "https://openrouter.ai/api/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com",
}

_OPENAI_CLIENT_EXTRA_KEYS = {"timeout"}
_OPENAI_CHAT_KNOWN_ARGS = {
    "audio",
    "frequency_penalty",
    "function_call",
    "functions",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "metadata",
    "modalities",
    "n",
    "parallel_tool_calls",
    "prediction",
    "presence_penalty",
    "reasoning_effort",
    "response_format",
    "seed",
    "service_tier",
    "stop",
    "store",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "user",
    "web_search_options",
}
_GEMINI_THOUGHT_SIGNATURE_SKIP = "skip_thought_signature_validator"
_GLOBAL_REASONING_EXTRA_KEY = "_global_reasoning"


def _make_openai_provider(provider_name: str):
    @register_provider(provider_name)
    class _Provider(OpenAICompatProvider):
        pass

    _Provider.__name__ = f"{provider_name.title()}Provider"
    _Provider.__qualname__ = _Provider.__name__
    return _Provider


class OpenAICompatProvider(BaseProvider):
    """Provider for any OpenAI-compatible API."""

    def _needs_gemini_thought_signature_fallback(self) -> bool:
        if self.config.extra.get("gemini_thought_signature_fallback") is False:
            return False

        model = self.config.model.lower().rsplit("/", 1)[-1]
        if not model.startswith("gemini-3"):
            return False

        if self.config.provider == "gemini":
            return True

        base_url = (self.config.base_url or "").lower()
        return "generativelanguage.googleapis.com" in base_url or "aiplatform.googleapis.com" in base_url

    @staticmethod
    def _tool_call_has_google_thought_signature(tool_call: dict[str, Any]) -> bool:
        extra_content = tool_call.get("extra_content")
        if isinstance(extra_content, dict):
            google = extra_content.get("google")
            if isinstance(google, dict) and google.get("thought_signature"):
                return True
        return bool(tool_call.get("thought_signature") or tool_call.get("thoughtSignature"))

    def _messages_for_provider(self, messages: list[dict]) -> list[dict]:
        if not self._needs_gemini_thought_signature_fallback():
            return messages

        normalized: list[dict] = []
        changed = False
        for message in messages:
            tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
            if (
                not isinstance(tool_calls, list)
                or not tool_calls
                or not isinstance(tool_calls[0], dict)
                or self._tool_call_has_google_thought_signature(tool_calls[0])
            ):
                normalized.append(message)
                continue

            patched = copy.deepcopy(message)
            first_tool_call = patched["tool_calls"][0]
            extra_content = first_tool_call.setdefault("extra_content", {})
            if not isinstance(extra_content, dict):
                extra_content = {}
                first_tool_call["extra_content"] = extra_content
            google = extra_content.setdefault("google", {})
            if not isinstance(google, dict):
                google = {}
                extra_content["google"] = google
            google["thought_signature"] = _GEMINI_THOUGHT_SIGNATURE_SKIP
            normalized.append(patched)
            changed = True

        return normalized if changed else messages

    def _client(self) -> AsyncOpenAI:
        base_url = self.config.base_url or PROVIDER_BASE_URLS.get(self.config.provider)
        client_kwargs: dict[str, Any] = {
            "api_key": self.config.api_key,
            "base_url": base_url,
            "max_retries": 0,
        }
        for key in _OPENAI_CLIENT_EXTRA_KEYS:
            if key in self.config.extra:
                client_kwargs[key] = self.config.extra[key]
        return AsyncOpenAI(**client_kwargs)

    def _needs_reasoning_extra_body(self) -> bool:
        model = self.config.model.strip().lower()
        if "deepseek" in model:
            return True
        provider = self.config.provider.strip().lower()
        base_url = (self.config.base_url or PROVIDER_BASE_URLS.get(provider) or "").lower()
        return provider == "deepseek" or "api.deepseek.com" in base_url

    def _provider_extra_body_defaults(self) -> dict[str, Any]:
        if not self._needs_reasoning_extra_body():
            return {}
        return {
            "reasoning": {
                "enabled": bool(self.config.extra.get(_GLOBAL_REASONING_EXTRA_KEY, True)),
            }
        }

    @staticmethod
    def _merge_extra_body_defaults(extra_body: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
        merged = copy.deepcopy(extra_body)
        for key, value in defaults.items():
            if key not in merged:
                merged[key] = copy.deepcopy(value)
                continue
            if isinstance(merged[key], dict) and isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    merged[key].setdefault(nested_key, copy.deepcopy(nested_value))
        return merged

    def _split_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        known_kwargs: dict[str, Any] = {}
        extra_body: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key == "extra_body":
                if isinstance(value, dict):
                    extra_body.update(copy.deepcopy(value))
                else:
                    known_kwargs[key] = value
            elif key in _OPENAI_CHAT_KNOWN_ARGS or key.startswith("extra_"):
                known_kwargs[key] = value
            else:
                extra_body[key] = value
        defaults = self._provider_extra_body_defaults()
        if defaults:
            extra_body = self._merge_extra_body_defaults(extra_body, defaults)
        if extra_body:
            known_kwargs["extra_body"] = extra_body
        return known_kwargs

    @staticmethod
    def _extract_text(message: dict[str, Any] | None) -> str:
        if not message:
            return ""
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts)
        return ""

    async def complete(self, messages: list[dict], **kwargs) -> ProviderResponse:
        client = self._client()
        resp = await client.chat.completions.create(
            model=self.config.model,
            messages=self._messages_for_provider(messages),
            stream=False,
            **self._split_kwargs(kwargs),
        )
        raw = resp.model_dump(mode="json", exclude_none=True)
        choices = raw.get("choices") or []
        first_choice = choices[0] if choices else {}
        message = first_choice.get("message") if isinstance(first_choice, dict) else None
        usage = raw.get("usage")
        return ProviderResponse(
            text=self._extract_text(message),
            message=message,
            raw_response=raw,
            usage=usage,
        )

    async def stream(
        self,
        messages: list[dict],
        **kwargs,
    ) -> AsyncGenerator[dict[str, Any], None]:
        client = self._client()
        resp = await client.chat.completions.create(
            model=self.config.model,
            messages=self._messages_for_provider(messages),
            stream=True,
            **self._split_kwargs(kwargs),
        )
        async for chunk in resp:
            yield chunk.model_dump(mode="json", exclude_none=True)


# Register well-known providers with preset base_urls
for _name in PROVIDER_BASE_URLS:
    _make_openai_provider(_name)

# Also register "openai_compat" for custom base_url providers
register_provider("openai_compat")(OpenAICompatProvider)
