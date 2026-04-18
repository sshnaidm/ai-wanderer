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
        assert config.server.port == 8000
        assert len(config.providers) == 1
        assert config.providers[0].backends[0].provider == "gemini"

    def test_full_config(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
keep_cycles: 3
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
        with pytest.raises(ValueError, match="openai_compat backends require base_url"):
            load_config(path)

    def test_non_openai_compat_rejects_base_url(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
providers:
  - priority: 1
    backends:
      - provider: openai
        api_key: "key"
        model: "gpt-4o"
        base_url: "https://api.example.com"
""",
        )
        with pytest.raises(ValueError, match="base_url is only supported for openai_compat backends"):
            load_config(path)
