from __future__ import annotations

import atexit
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import RateLimits, StateStoreConfig

try:
    import redis
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    redis = None

logger = logging.getLogger(__name__)

_CLEANUP_INTERVAL = 500
_REDIS_HASH_TTL_SECONDS = 3 * 24 * 60 * 60


def _window_keys() -> dict[str, str]:
    t = time.gmtime()
    return {
        "m": f"{t.tm_year:04d}{t.tm_mon:02d}{t.tm_mday:02d}T{t.tm_hour:02d}{t.tm_min:02d}",
        "h": f"{t.tm_year:04d}{t.tm_mon:02d}{t.tm_mday:02d}T{t.tm_hour:02d}",
        "d": f"{t.tm_year:04d}{t.tm_mon:02d}{t.tm_mday:02d}",
    }


@dataclass
class _Counters:
    requests: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tokens: dict[str, int] = field(default_factory=lambda: defaultdict(int))


class RateLimiter:
    def __init__(
        self,
        state_file: str | Path | None = None,
        *,
        state_store: StateStoreConfig | None = None,
        save_interval_seconds: float = 60.0,
    ) -> None:
        self._store_config = state_store or StateStoreConfig()
        self._store_type = self._store_config.type
        self._counters: dict[str, _Counters] = {}
        self._ops: int = 0
        self._state_file = Path(state_file) if state_file and self._store_type == "local" else None
        self._save_interval_seconds = save_interval_seconds
        self._last_save_monotonic = time.monotonic()
        self._dirty = False
        self._redis_client: Any | None = None
        self._redis_prefix = self._store_config.redis_prefix

        if self._store_type == "redis":
            self._redis_client = self._build_redis_client(self._store_config)
            return

        if self._state_file:
            self._load()
            atexit.register(self.save)

    @property
    def store_type(self) -> str:
        return self._store_type

    def _get(self, key: str) -> _Counters:
        if key not in self._counters:
            self._counters[key] = _Counters()
        return self._counters[key]

    def is_allowed(self, key: str, limits: RateLimits) -> bool:
        c = self._current_counters(key)
        wk = _window_keys()
        checks = [
            (limits.rpm, c.requests, "m"),
            (limits.rph, c.requests, "h"),
            (limits.rpd, c.requests, "d"),
            (limits.tpm, c.tokens, "m"),
            (limits.tph, c.tokens, "h"),
            (limits.tpd, c.tokens, "d"),
        ]
        for limit_val, counter, period in checks:
            if limit_val is not None and counter[wk[period]] >= limit_val:
                return False
        return True

    def record_request(self, key: str) -> None:
        if self._store_type == "redis":
            self._redis_record(key, "requests", 1)
            return
        c = self._get(key)
        wk = _window_keys()
        for period in ("m", "h", "d"):
            c.requests[wk[period]] += 1
        self._dirty = True
        self._tick_cleanup()

    def record_tokens(self, key: str, tokens: int) -> None:
        if tokens <= 0:
            return
        if self._store_type == "redis":
            self._redis_record(key, "tokens", tokens)
            return
        c = self._get(key)
        wk = _window_keys()
        for period in ("m", "h", "d"):
            c.tokens[wk[period]] += tokens
        self._dirty = True
        self._tick_cleanup()

    def save_if_due(self) -> None:
        snapshot = self.snapshot_if_due()
        if snapshot:
            state_file, data = snapshot
            try:
                self.write_snapshot(state_file, data)
            except OSError as e:
                self.mark_dirty()
                logger.warning("Failed to save rate limiter state: %s", e)

    def save(self) -> None:
        snapshot = self.snapshot()
        if not snapshot:
            return
        state_file, data = snapshot
        try:
            self.write_snapshot(state_file, data)
        except OSError as e:
            self.mark_dirty()
            logger.warning("Failed to save rate limiter state: %s", e)

    def snapshot_if_due(self) -> tuple[Path, dict[str, dict[str, dict[str, int]]]] | None:
        if self._store_type != "local":
            return None
        if not self._state_file or not self._dirty:
            return None
        if time.monotonic() - self._last_save_monotonic < self._save_interval_seconds:
            return None
        return self.snapshot()

    def snapshot(self) -> tuple[Path, dict[str, dict[str, dict[str, int]]]] | None:
        if self._store_type != "local":
            return None
        if not self._state_file or not self._dirty:
            return None
        self._cleanup()
        data = self._to_json_data()
        self._dirty = False
        self._last_save_monotonic = time.monotonic()
        return self._state_file, data

    def mark_dirty(self) -> None:
        if self._store_type == "local":
            self._dirty = True

    def counters_snapshot(self) -> dict[str, dict[str, dict[str, int]]]:
        if self._store_type == "redis":
            return self._redis_snapshot()
        self._cleanup()
        return self._to_json_data()

    def _tick_cleanup(self) -> None:
        self._ops += 1
        if self._ops % _CLEANUP_INTERVAL == 0:
            self._cleanup()

    def _cleanup(self) -> None:
        if self._store_type != "local":
            return
        wk = _window_keys()
        current_keys = set(wk.values())
        for counters in self._counters.values():
            for d in (counters.requests, counters.tokens):
                stale = [k for k in d if k not in current_keys]
                for k in stale:
                    del d[k]

    def _load(self) -> None:
        if self._store_type != "local":
            return
        if not self._state_file or not self._state_file.exists():
            return
        try:
            data = json.loads(self._state_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load rate limiter state: %s", e)
            return
        if not isinstance(data, dict):
            logger.warning("Rate limiter state file has unexpected format, ignoring")
            return
        wk = _window_keys()
        current_keys = set(wk.values())
        loaded = 0
        for key, entry in data.items():
            if not isinstance(key, str) or not isinstance(entry, dict):
                logger.warning("Rate limiter state contains invalid backend entry, skipping")
                continue
            c = self._get(key)
            requests = entry.get("requests", {})
            tokens = entry.get("tokens", {})
            if not isinstance(requests, dict) or not isinstance(tokens, dict):
                logger.warning("Rate limiter state contains invalid counter entry for %s, skipping", key)
                continue
            for wkey, count in requests.items():
                if wkey in current_keys:
                    if not isinstance(count, int):
                        logger.warning("Rate limiter state contains invalid request count for %s, skipping", key)
                        continue
                    c.requests[wkey] = count
                    loaded += 1
            for wkey, count in tokens.items():
                if wkey in current_keys:
                    if not isinstance(count, int):
                        logger.warning("Rate limiter state contains invalid token count for %s, skipping", key)
                        continue
                    c.tokens[wkey] = count
                    loaded += 1
        if loaded:
            logger.info("Loaded rate limiter state: %d counters from %s", loaded, self._state_file)

    def _to_json_data(self) -> dict[str, dict[str, dict[str, int]]]:
        data: dict[str, dict[str, dict[str, int]]] = {}
        for key, c in self._counters.items():
            entry: dict[str, dict[str, int]] = {}
            if c.requests:
                entry["requests"] = dict(c.requests)
            if c.tokens:
                entry["tokens"] = dict(c.tokens)
            if entry:
                data[key] = entry
        return data

    def _current_counters(self, key: str) -> _Counters:
        if self._store_type == "redis":
            return self._redis_get(key)
        return self._get(key)

    @staticmethod
    def write_snapshot(state_file: Path, data: dict[str, dict[str, dict[str, int]]]) -> None:
        tmp = state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(state_file)

    def _save(self) -> None:
        self.save()

    @staticmethod
    def _build_redis_client(state_store: StateStoreConfig) -> Any:
        if redis is None:
            raise RuntimeError(
                "Redis state_store configured but the optional 'redis' package is not installed. "
                "Install the Redis extra first, for example: pip install '.[redis]'."
            )
        assert state_store.redis_url is not None
        return redis.Redis.from_url(state_store.redis_url, decode_responses=True)

    def _redis_backend_key(self, key: str, counter_type: str) -> str:
        return f"{self._redis_prefix}:{counter_type}:{key}"

    def _redis_known_keys_key(self) -> str:
        return f"{self._redis_prefix}:backend_keys"

    def _redis_touch_backend(self, key: str) -> None:
        assert self._redis_client is not None
        self._redis_client.sadd(self._redis_known_keys_key(), key)

    def _redis_record(self, key: str, counter_type: str, amount: int) -> None:
        assert self._redis_client is not None
        wk = _window_keys()
        redis_key = self._redis_backend_key(key, counter_type)
        pipe = self._redis_client.pipeline()
        for period in ("m", "h", "d"):
            pipe.hincrby(redis_key, wk[period], amount)
        pipe.expire(redis_key, _REDIS_HASH_TTL_SECONDS)
        pipe.sadd(self._redis_known_keys_key(), key)
        pipe.execute()

    @staticmethod
    def _coerce_redis_counter(value: Any) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return 0
        return 0

    def _redis_get(self, key: str) -> _Counters:
        assert self._redis_client is not None
        wk = _window_keys()
        current_keys = set(wk.values())
        requests_raw = self._redis_client.hgetall(self._redis_backend_key(key, "requests"))
        tokens_raw = self._redis_client.hgetall(self._redis_backend_key(key, "tokens"))
        counters = _Counters()
        for window_key, value in requests_raw.items():
            if window_key in current_keys:
                counters.requests[window_key] = self._coerce_redis_counter(value)
        for window_key, value in tokens_raw.items():
            if window_key in current_keys:
                counters.tokens[window_key] = self._coerce_redis_counter(value)
        return counters

    def _redis_snapshot(self) -> dict[str, dict[str, dict[str, int]]]:
        assert self._redis_client is not None
        data: dict[str, dict[str, dict[str, int]]] = {}
        for key in self._redis_client.smembers(self._redis_known_keys_key()):
            counters = self._redis_get(key)
            entry: dict[str, dict[str, int]] = {}
            if counters.requests:
                entry["requests"] = dict(counters.requests)
            if counters.tokens:
                entry["tokens"] = dict(counters.tokens)
            if entry:
                data[key] = entry
        return data
