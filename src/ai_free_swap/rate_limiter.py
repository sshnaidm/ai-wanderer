from __future__ import annotations

import atexit
import json
import logging
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import RateLimits, StateStoreConfig

try:
    from redis import asyncio as redis
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
        self._local_lock = threading.Lock()

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

    @staticmethod
    def _is_allowed(c: _Counters, limits: RateLimits) -> bool:
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

    def is_allowed(self, key: str, limits: RateLimits) -> bool:
        if self._store_type == "redis":
            raise RuntimeError("Use is_allowed_async() with a Redis state store")
        with self._local_lock:
            return self._is_allowed(self._get(key), limits)

    async def is_allowed_async(self, key: str, limits: RateLimits) -> bool:
        if self._store_type == "redis":
            return self._is_allowed(await self._redis_get(key), limits)
        return self.is_allowed(key, limits)

    @classmethod
    def is_allowed_snapshot(
        cls,
        counters: dict[str, dict[str, int]],
        limits: RateLimits,
    ) -> bool:
        normalized = _Counters()
        normalized.requests.update(counters.get("requests", {}))
        normalized.tokens.update(counters.get("tokens", {}))
        return cls._is_allowed(normalized, limits)

    async def reserve_request(self, key: str, limits: RateLimits) -> bool:
        """Atomically enforce limits and count a provider request attempt."""
        if self._store_type == "redis":
            return await self._redis_reserve_request(key, limits)
        with self._local_lock:
            counters = self._get(key)
            if not self._is_allowed(counters, limits):
                return False
            self._record_request_local(counters)
            return True

    def _record_request_local(self, counters: _Counters) -> None:
        wk = _window_keys()
        for period in ("m", "h", "d"):
            counters.requests[wk[period]] += 1
        self._dirty = True
        self._tick_cleanup()

    def record_request(self, key: str) -> None:
        if self._store_type == "redis":
            raise RuntimeError("Use record_request_async() with a Redis state store")
        with self._local_lock:
            self._record_request_local(self._get(key))

    async def record_request_async(self, key: str) -> None:
        if self._store_type == "redis":
            await self._redis_record(key, "requests", 1)
            return
        self.record_request(key)

    def record_tokens(self, key: str, tokens: int) -> None:
        if tokens <= 0:
            return
        if self._store_type == "redis":
            raise RuntimeError("Use record_tokens_async() with a Redis state store")
        with self._local_lock:
            c = self._get(key)
            wk = _window_keys()
            for period in ("m", "h", "d"):
                c.tokens[wk[period]] += tokens
            self._dirty = True
            self._tick_cleanup()

    async def record_tokens_async(self, key: str, tokens: int) -> None:
        if tokens <= 0:
            return
        if self._store_type == "redis":
            await self._redis_record(key, "tokens", tokens)
            return
        self.record_tokens(key, tokens)

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
            raise RuntimeError("Use counters_snapshot_async() with a Redis state store")
        with self._local_lock:
            self._cleanup()
            return self._to_json_data()

    async def counters_snapshot_async(self) -> dict[str, dict[str, dict[str, int]]]:
        if self._store_type == "redis":
            return await self._redis_snapshot()
        return self.counters_snapshot()

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

    async def _redis_record(self, key: str, counter_type: str, amount: int) -> None:
        assert self._redis_client is not None
        wk = _window_keys()
        redis_key = self._redis_backend_key(key, counter_type)
        pipe = self._redis_client.pipeline()
        for period in ("m", "h", "d"):
            pipe.hincrby(redis_key, wk[period], amount)
        pipe.expire(redis_key, _REDIS_HASH_TTL_SECONDS)
        pipe.sadd(self._redis_known_keys_key(), key)
        await pipe.execute()

    async def _redis_reserve_request(self, key: str, limits: RateLimits) -> bool:
        assert self._redis_client is not None
        wk = _window_keys()
        script = """
local periods = {ARGV[1], ARGV[2], ARGV[3]}
for i = 1, 3 do
    local request_limit = tonumber(ARGV[3 + i])
    local token_limit = tonumber(ARGV[6 + i])
    if request_limit >= 0 and tonumber(redis.call('HGET', KEYS[1], periods[i]) or '0') >= request_limit then
        return 0
    end
    if token_limit >= 0 and tonumber(redis.call('HGET', KEYS[2], periods[i]) or '0') >= token_limit then
        return 0
    end
end
for i = 1, 3 do
    redis.call('HINCRBY', KEYS[1], periods[i], 1)
end
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[10]))
redis.call('SADD', KEYS[3], ARGV[11])
return 1
"""
        args = [
            wk["m"],
            wk["h"],
            wk["d"],
            limits.rpm if limits.rpm is not None else -1,
            limits.rph if limits.rph is not None else -1,
            limits.rpd if limits.rpd is not None else -1,
            limits.tpm if limits.tpm is not None else -1,
            limits.tph if limits.tph is not None else -1,
            limits.tpd if limits.tpd is not None else -1,
            _REDIS_HASH_TTL_SECONDS,
            key,
        ]
        result = await self._redis_client.eval(
            script,
            3,
            self._redis_backend_key(key, "requests"),
            self._redis_backend_key(key, "tokens"),
            self._redis_known_keys_key(),
            *args,
        )
        return bool(result)

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

    async def _redis_get(self, key: str) -> _Counters:
        assert self._redis_client is not None
        wk = _window_keys()
        current_keys = set(wk.values())
        pipe = self._redis_client.pipeline()
        pipe.hgetall(self._redis_backend_key(key, "requests"))
        pipe.hgetall(self._redis_backend_key(key, "tokens"))
        requests_raw, tokens_raw = await pipe.execute()
        counters = _Counters()
        for window_key, value in requests_raw.items():
            if window_key in current_keys:
                counters.requests[window_key] = self._coerce_redis_counter(value)
        for window_key, value in tokens_raw.items():
            if window_key in current_keys:
                counters.tokens[window_key] = self._coerce_redis_counter(value)
        return counters

    async def _redis_snapshot(self) -> dict[str, dict[str, dict[str, int]]]:
        assert self._redis_client is not None
        data: dict[str, dict[str, dict[str, int]]] = {}
        keys = sorted(await self._redis_client.smembers(self._redis_known_keys_key()))
        pipe = self._redis_client.pipeline()
        for key in keys:
            pipe.hgetall(self._redis_backend_key(key, "requests"))
            pipe.hgetall(self._redis_backend_key(key, "tokens"))
        raw_entries = await pipe.execute() if keys else []
        stale_keys: list[str] = []
        wk = _window_keys()
        current_keys = set(wk.values())
        for index, key in enumerate(keys):
            requests_raw = raw_entries[index * 2]
            tokens_raw = raw_entries[index * 2 + 1]
            counters = _Counters()
            for window_key, value in requests_raw.items():
                if window_key in current_keys:
                    counters.requests[window_key] = self._coerce_redis_counter(value)
            for window_key, value in tokens_raw.items():
                if window_key in current_keys:
                    counters.tokens[window_key] = self._coerce_redis_counter(value)
            entry: dict[str, dict[str, int]] = {}
            if counters.requests:
                entry["requests"] = dict(counters.requests)
            if counters.tokens:
                entry["tokens"] = dict(counters.tokens)
            if entry:
                data[key] = entry
            else:
                stale_keys.append(key)
        if stale_keys:
            await self._redis_client.srem(self._redis_known_keys_key(), *stale_keys)
        return data
