from __future__ import annotations

import pytest

from ai_free_swap.config import load_config


def _write_yaml(tmp_path, content: str):
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


class TestLoadConfig:
    def test_minimal_config(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
providers:
  - priority: 1
    backends:
      - provider: gemini
        api_key: "test-key"
        model: "gemini-2.5-flash"
""",
        )
        config = load_config(path)
        assert config.keep_cycles == 1
        assert config.reasoning is True
        assert config.server.port == 8000
        assert len(config.providers) == 1
        assert config.providers[0].backends[0].provider == "gemini"

    def test_full_config(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
keep_cycles: 3
reasoning: false
server:
  host: "127.0.0.1"
  port: 9000
  api_key: "my-secret"
providers:
  - priority: 1
    backends:
      - provider: gemini
        api_key: "key1"
        model: "gemini-2.5-flash"
      - provider: gemini
        api_key: "key2"
        model: "gemini-2.5-flash-lite"
  - priority: 2
    backends:
      - provider: openai
        api_key: "key3"
        model: "gpt-4o"
""",
        )
        config = load_config(path)
        assert config.keep_cycles == 3
        assert config.reasoning is False
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 9000
        assert config.server.api_key == "my-secret"
        assert len(config.providers) == 2

    def test_env_var_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "expanded-key-value")
        path = _write_yaml(
            tmp_path,
            """
providers:
  - priority: 1
    backends:
      - provider: gemini
        api_key: "${TEST_API_KEY}"
        model: "test-model"
""",
        )
        config = load_config(path)
        assert config.providers[0].backends[0].api_key == "expanded-key-value"

    def test_env_var_missing_raises(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
providers:
  - priority: 1
    backends:
      - provider: gemini
        api_key: "${DEFINITELY_NOT_SET_12345}"
        model: "test-model"
""",
        )
        with pytest.raises(ValueError, match="DEFINITELY_NOT_SET_12345"):
            load_config(path)

    def test_openai_compat_allows_safe_extra(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
providers:
  - priority: 1
    backends:
      - provider: openai_compat
        api_key: "key"
        model: "llama-3"
        base_url: "https://api.groq.com/openai/v1"
        extra:
          timeout: 30
""",
        )
        backend = load_config(path).providers[0].backends[0]
        assert backend.base_url == "https://api.groq.com/openai/v1"
        assert backend.extra == {"timeout": 30}

    def test_backend_capabilities_are_loaded(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
providers:
  - priority: 1
    backends:
      - provider: gemini
        api_key: "key"
        model: "gemini-2.5-flash"
        capabilities:
          supports_tools: true
          supports_vision: true
          supports_reasoning: false
          supports_streaming: true
          max_context_tokens: 1048576
          max_output_tokens: 8192
          tags: ["cloud", "fast"]
""",
        )
        backend = load_config(path).providers[0].backends[0]
        assert backend.capabilities is not None
        assert backend.capabilities.supports_tools is True
        assert backend.capabilities.supports_vision is True
        assert backend.capabilities.supports_reasoning is False
        assert backend.capabilities.supports_streaming is True
        assert backend.capabilities.max_context_tokens == 1048576
        assert backend.capabilities.max_output_tokens == 8192
        assert backend.capabilities.tags == ["cloud", "fast"]

    def test_backend_capabilities_reject_invalid_limits(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
providers:
  - priority: 1
    backends:
      - provider: gemini
        api_key: "key"
        model: "gemini-2.5-flash"
        capabilities:
          max_context_tokens: 0
""",
        )
        with pytest.raises(ValueError, match="capability limit must be >= 1"):
            load_config(path)

    def test_backend_capabilities_normalize_tags(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
providers:
  - priority: 1
    backends:
      - provider: gemini
        api_key: "key"
        model: "gemini-2.5-flash"
        capabilities:
          tags: [" cloud ", "fast", "cloud"]
""",
        )
        backend = load_config(path).providers[0].backends[0]
        assert backend.capabilities is not None
        assert backend.capabilities.tags == ["cloud", "fast"]

    def test_state_store_defaults_to_local(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
providers:
  - priority: 1
    backends:
      - provider: gemini
        api_key: "key"
        model: "gemini-2.5-flash"
""",
        )
        config = load_config(path)
        assert config.state_store.type == "local"
        assert config.state_store.redis_url is None

    def test_state_store_redis_requires_url(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
state_store:
  type: redis
providers:
  - priority: 1
    backends:
      - provider: gemini
        api_key: "key"
        model: "gemini-2.5-flash"
""",
        )
        with pytest.raises(ValueError, match="state_store.redis_url is required"):
            load_config(path)

    def test_rejects_empty_api_key(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
providers:
  - priority: 1
    backends:
      - provider: gemini
        api_key: "   "
        model: "gemini-2.5-flash"
""",
        )
        with pytest.raises(ValueError, match="must not be empty"):
            load_config(path)

    def test_rejects_keep_cycles_less_than_one(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
keep_cycles: 0
providers:
  - priority: 1
    backends:
      - provider: gemini
        api_key: "test-key"
        model: "gemini-2.5-flash"
""",
        )
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            load_config(path)

    def test_rejects_restricted_extra_keys(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
providers:
  - priority: 1
    backends:
      - provider: openai
        api_key: "key"
        model: "gpt-4o"
        extra:
          base_url: "https://evil.invalid"
""",
        )
        with pytest.raises(ValueError, match="restricted transport or credential keys"):
            load_config(path)

    def test_allows_max_tokens_and_similar_in_extra(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
providers:
  - priority: 1
    backends:
      - provider: gemini
        api_key: "key"
        model: "gemini-2.5-flash"
        extra:
          max_tokens: 1024
          max_output_tokens: 2048
          token_budget: 500
""",
        )
        backend = load_config(path).providers[0].backends[0]
        assert backend.extra == {"max_tokens": 1024, "max_output_tokens": 2048, "token_budget": 500}

    def test_openai_compat_requires_base_url(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
providers:
  - priority: 1
    backends:
      - provider: openai_compat
        api_key: "key"
        model: "llama-3"
""",
        )
        with pytest.raises(ValueError, match="openai_compat provider requires base_url"):
            load_config(path)

    def test_deepseek_builtin_does_not_require_base_url(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
providers:
  - priority: 1
    backends:
      - provider: deepseek
        api_key: "key"
        model: "deepseek-chat"
""",
        )
        backend = load_config(path).providers[0].backends[0]
        assert backend.provider == "deepseek"
        assert backend.base_url is None

    def test_backend_reasoning_override_is_allowed(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
providers:
  - priority: 1
    backends:
      - provider: deepseek
        api_key: "key"
        model: "deepseek-chat"
        reasoning: false
""",
        )
        backend = load_config(path).providers[0].backends[0]
        assert backend.reasoning is False

    def test_base_url_allowed_on_any_provider(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
providers:
  - priority: 1
    backends:
      - provider: gemini
        api_key: "key"
        model: "gemini-2.5-flash"
        base_url: "https://custom-proxy.example.com/v1"
""",
        )
        backend = load_config(path).providers[0].backends[0]
        assert backend.base_url == "https://custom-proxy.example.com/v1"
