from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class BackendConfig(BaseModel):
    provider: str
    api_key: str
    model: str
    base_url: str | None = None
    extra: dict = Field(default_factory=dict)


class PriorityGroup(BaseModel):
    priority: int
    backends: list[BackendConfig]


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""


class AppConfig(BaseModel):
    keep_cycles: int = 1
    server: ServerConfig = Field(default_factory=ServerConfig)
    providers: list[PriorityGroup]


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
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = _walk_and_expand(raw)
    return AppConfig(**raw)
