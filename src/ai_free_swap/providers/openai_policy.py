from __future__ import annotations

import copy
from typing import Any

from ..config import BackendConfig

PROVIDER_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "grok": "https://api.x.ai/v1",
    "openai": None,
    "openrouter": "https://openrouter.ai/api/v1",
    "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "qwen-cn": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com",
}

OPENAI_CLIENT_EXTRA_KEYS = {"timeout"}
OPENAI_CHAT_KNOWN_ARGS = {
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

EXTRA_BODY_DEFAULTS_EXTRA_KEY = "extra_body_defaults"
GEMINI_THOUGHT_SIGNATURE_SKIP = "skip_thought_signature_validator"
REQUEST_KWARGS_DEFAULTS_EXTRA_KEY = "request_kwargs_defaults"


def provider_base_url(config: BackendConfig) -> str | None:
    return config.base_url or PROVIDER_BASE_URLS.get(config.provider)


def merge_defaults(
    target: dict[str, Any],
    defaults: dict[str, Any],
    *,
    override: bool = False,
) -> dict[str, Any]:
    merged = copy.deepcopy(target)
    for key, value in defaults.items():
        if key not in merged or override:
            if key not in merged or not (isinstance(merged.get(key), dict) and isinstance(value, dict)):
                merged[key] = copy.deepcopy(value)
                continue
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = merge_defaults(merged[key], value, override=override)
        elif override:
            merged[key] = copy.deepcopy(value)
    return merged


def _needs_deepseek_reasoning(config: BackendConfig) -> bool:
    model = config.model.strip().lower()
    if "deepseek" in model:
        return True
    provider = config.provider.strip().lower()
    base_url = (provider_base_url(config) or "").lower()
    return provider == "deepseek" or "api.deepseek.com" in base_url


def _built_in_extra_body_defaults(config: BackendConfig) -> dict[str, Any]:
    if not _needs_deepseek_reasoning(config):
        return {}
    return {
        "reasoning": {
            "enabled": config.reasoning if config.reasoning is not None else True,
        }
    }


def _configured_request_kwargs_defaults(config: BackendConfig) -> dict[str, Any]:
    defaults = config.extra.get(REQUEST_KWARGS_DEFAULTS_EXTRA_KEY, {})
    return copy.deepcopy(defaults) if isinstance(defaults, dict) else {}


def _configured_extra_body_defaults(config: BackendConfig) -> dict[str, Any]:
    defaults = config.extra.get(EXTRA_BODY_DEFAULTS_EXTRA_KEY, {})
    return copy.deepcopy(defaults) if isinstance(defaults, dict) else {}


def extra_body_defaults(config: BackendConfig) -> dict[str, Any]:
    defaults = _built_in_extra_body_defaults(config)
    configured = _configured_extra_body_defaults(config)
    if configured:
        defaults = merge_defaults(defaults, configured, override=True)
    return defaults


def _apply_request_kwargs_defaults(config: BackendConfig, known_kwargs: dict[str, Any]) -> dict[str, Any]:
    configured = _configured_request_kwargs_defaults(config)
    if not configured:
        return known_kwargs
    filtered = {
        key: value for key, value in configured.items() if key in OPENAI_CHAT_KNOWN_ARGS or key.startswith("extra_")
    }
    return merge_defaults(known_kwargs, filtered)


def split_openai_kwargs(config: BackendConfig, kwargs: dict[str, Any]) -> dict[str, Any]:
    known_kwargs: dict[str, Any] = {}
    extra_body: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key == "extra_body":
            if isinstance(value, dict):
                extra_body = merge_defaults(extra_body, value, override=True)
            else:
                known_kwargs[key] = value
        elif key in OPENAI_CHAT_KNOWN_ARGS or key.startswith("extra_"):
            known_kwargs[key] = value
        else:
            extra_body[key] = value
    known_kwargs = _apply_request_kwargs_defaults(config, known_kwargs)
    defaults = extra_body_defaults(config)
    if defaults:
        extra_body = merge_defaults(extra_body, defaults)
    if extra_body:
        known_kwargs["extra_body"] = extra_body
    return known_kwargs


def _needs_gemini_thought_signature_fallback(config: BackendConfig) -> bool:
    if config.extra.get("gemini_thought_signature_fallback") is False:
        return False

    model = config.model.lower().rsplit("/", 1)[-1]
    if not model.startswith("gemini-3"):
        return False

    if config.provider == "gemini":
        return True

    base_url = (config.base_url or "").lower()
    return "generativelanguage.googleapis.com" in base_url or "aiplatform.googleapis.com" in base_url


def _tool_call_has_google_thought_signature(tool_call: dict[str, Any]) -> bool:
    extra_content = tool_call.get("extra_content")
    if isinstance(extra_content, dict):
        google = extra_content.get("google")
        if isinstance(google, dict) and google.get("thought_signature"):
            return True
    return bool(tool_call.get("thought_signature") or tool_call.get("thoughtSignature"))


def messages_for_provider(config: BackendConfig, messages: list[dict]) -> list[dict]:
    if not _needs_gemini_thought_signature_fallback(config):
        return messages

    normalized: list[dict] = []
    changed = False
    for message in messages:
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(tool_calls, list) or not tool_calls:
            normalized.append(message)
            continue

        missing_signature = [
            index
            for index, tool_call in enumerate(tool_calls)
            if isinstance(tool_call, dict) and not _tool_call_has_google_thought_signature(tool_call)
        ]
        if not missing_signature:
            normalized.append(message)
            continue

        patched = copy.deepcopy(message)
        for index in missing_signature:
            tool_call = patched["tool_calls"][index]
            extra_content = tool_call.setdefault("extra_content", {})
            if not isinstance(extra_content, dict):
                extra_content = {}
                tool_call["extra_content"] = extra_content
            google = extra_content.setdefault("google", {})
            if not isinstance(google, dict):
                google = {}
                extra_content["google"] = google
            google["thought_signature"] = GEMINI_THOUGHT_SIGNATURE_SKIP
        normalized.append(patched)
        changed = True

    return normalized if changed else messages
