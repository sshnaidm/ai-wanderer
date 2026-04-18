from __future__ import annotations

import os
import tempfile

import pytest

from ai_free_swap.config import AppConfig, load_config


def _write_yaml(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.write(content)
    f.close()
    return f.name


class TestLoadConfig:
    def test_minimal_config(self):
        path = _write_yaml("""
providers:
  - priority: 1
    backends:
      - provider: gemini
        api_key: "test-key"
        model: "gemini-2.5-flash"
""")
        config = load_config(path)
        assert config.keep_cycles == 1
        assert config.server.port == 8000
        assert len(config.providers) == 1
        assert config.providers[0].backends[0].provider == "gemini"
        os.unlink(path)

    def test_full_config(self):
        path = _write_yaml("""
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
""")
        config = load_config(path)
        assert config.keep_cycles == 3
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 9000
        assert config.server.api_key == "my-secret"
        assert len(config.providers) == 2
        assert len(config.providers[0].backends) == 2
        assert len(config.providers[1].backends) == 1
        os.unlink(path)

    def test_env_var_expansion(self, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "expanded-key-value")
        path = _write_yaml("""
providers:
  - priority: 1
    backends:
      - provider: gemini
        api_key: "${TEST_API_KEY}"
        model: "test-model"
""")
        config = load_config(path)
        assert config.providers[0].backends[0].api_key == "expanded-key-value"
        os.unlink(path)

    def test_env_var_missing_raises(self):
        path = _write_yaml("""
providers:
  - priority: 1
    backends:
      - provider: gemini
        api_key: "${DEFINITELY_NOT_SET_12345}"
        model: "test-model"
""")
        with pytest.raises(ValueError, match="DEFINITELY_NOT_SET_12345"):
            load_config(path)
        os.unlink(path)

    def test_extra_fields_passed_through(self):
        path = _write_yaml("""
providers:
  - priority: 1
    backends:
      - provider: openai_compat
        api_key: "key"
        model: "llama-3"
        base_url: "https://api.groq.com/openai/v1"
        extra:
          timeout: 30
""")
        config = load_config(path)
        b = config.providers[0].backends[0]
        assert b.base_url == "https://api.groq.com/openai/v1"
        assert b.extra == {"timeout": 30}
        os.unlink(path)
