from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_RESTRICTED_EXTRA_KEYS = frozenset(
    {
        "api_key",
        "secret",
        "password",
        "base_url",
        "api_base",
        "proxy",
        "transport",
    }
)


class RateLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rpm: int | None = None
    rph: int | None = None
    rpd: int | None = None
    tpm: int | None = None
    tph: int | None = None
    tpd: int | None = None

    @field_validator("rpm", "rph", "rpd", "tpm", "tph", "tpd")
    @classmethod
    def _validate_positive(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("limit must be >= 1")
        return value


class BackendCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supports_tools: bool | None = None
    supports_vision: bool | None = None
    supports_reasoning: bool | None = None
    supports_streaming: bool | None = None
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("max_context_tokens", "max_output_tokens")
    @classmethod
    def _validate_positive_limit(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("capability limit must be >= 1")
        return value

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for tag in value:
            stripped = tag.strip()
            if not stripped:
                raise ValueError("capability tags must not be empty")
            if stripped not in seen:
                normalized.append(stripped)
                seen.add(stripped)
        return normalized


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    api_key: str
    model: str
    name: str | None = None
    base_url: str | None = None
    limits: RateLimits | None = None
    capabilities: BackendCapabilities | None = None
    reasoning: bool | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", "api_key", "model")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("base_url must not be empty when provided")
        return value

    @field_validator("extra")
    @classmethod
    def _validate_extra(cls, value: dict[str, Any]) -> dict[str, Any]:
        blocked = sorted(key for key in value if key.strip().lower() in _RESTRICTED_EXTRA_KEYS)
        if blocked:
            blocked_list = ", ".join(blocked)
            raise ValueError("extra contains restricted transport or credential keys: " f"{blocked_list}")
        return value

    @model_validator(mode="after")
    def _validate_provider_specific_fields(self) -> BackendConfig:
        if self.provider == "openai_compat" and not self.base_url:
            raise ValueError("openai_compat provider requires base_url")
        return self


class PriorityGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    priority: int = Field(ge=1)
    backends: list[BackendConfig] = Field(min_length=1)


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    api_key: str = ""

    @field_validator("host", "api_key")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()


_VALID_MODEL_ROUTING = frozenset({"any", "match"})
_VALID_STATE_STORE_TYPES = frozenset({"local", "redis"})


class StateStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = "local"
    redis_url: str | None = None
    redis_prefix: str = "ai_free_swap"

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in _VALID_STATE_STORE_TYPES:
            raise ValueError(f"state_store.type must be one of {sorted(_VALID_STATE_STORE_TYPES)}, got {value!r}")
        return normalized

    @field_validator("redis_url", "redis_prefix")
    @classmethod
    def _strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip()

    @model_validator(mode="after")
    def _validate_backend_requirements(self) -> StateStoreConfig:
        if self.type == "redis" and not self.redis_url:
            raise ValueError("state_store.redis_url is required when state_store.type is 'redis'")
        if self.type == "local" and self.redis_url:
            raise ValueError("state_store.redis_url is only valid when state_store.type is 'redis'")
        if not self.redis_prefix:
            raise ValueError("state_store.redis_prefix must not be empty")
        return self


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keep_cycles: int = Field(default=1, ge=1)
    model_name: str = Field(default="aifree")
    show_provider: bool = Field(default=True)
    model_routing: str = Field(default="any")
    reasoning: bool = True
    server: ServerConfig = Field(default_factory=ServerConfig)
    state_store: StateStoreConfig = Field(default_factory=StateStoreConfig)
    providers: list[PriorityGroup] = Field(min_length=1)

    @field_validator("model_routing")
    @classmethod
    def _validate_model_routing(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in _VALID_MODEL_ROUTING:
            raise ValueError(f"model_routing must be one of {sorted(_VALID_MODEL_ROUTING)}, got {value!r}")
        return value


_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _expand_env_vars(value: str) -> str:
    def _replace(match: re.Match) -> str:
        var = match.group(1)
        result = os.environ.get(var)
        if result is None:
            raise ValueError(f"Environment variable {var!r} is not set")
        return result

    return _ENV_VAR_RE.sub(_replace, value)


def _walk_and_expand(obj):
    if isinstance(obj, str):
        return _expand_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _walk_and_expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_expand(v) for v in obj]
    return obj


def load_config(path: str | Path) -> AppConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError("Config file is empty")
    raw = _walk_and_expand(raw)
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")
    return AppConfig(**raw)
