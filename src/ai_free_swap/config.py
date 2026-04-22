from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_RESTRICTED_EXTRA_KEYS = frozenset({
    "api_key",
    "secret",
    "password",
    "base_url",
    "api_base",
    "proxy",
    "transport",
})


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    api_key: str
    model: str
    base_url: str | None = None
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
        blocked = sorted(
            key for key in value if key.strip().lower() in _RESTRICTED_EXTRA_KEYS
        )
        if blocked:
            blocked_list = ", ".join(blocked)
            raise ValueError("extra contains restricted transport or credential keys: " f"{blocked_list}")
        return value

    @model_validator(mode="after")
    def _validate_provider_specific_fields(self) -> BackendConfig:
        if self.provider == "openai_compat" and not self.base_url:
            raise ValueError("openai_compat backends require base_url")
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


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keep_cycles: int = Field(default=1, ge=1)
    server: ServerConfig = Field(default_factory=ServerConfig)
    providers: list[PriorityGroup] = Field(min_length=1)


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
