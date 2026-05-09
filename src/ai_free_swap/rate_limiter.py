from __future__ import annotations

import atexit
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .config import RateLimits

logger = logging.getLogger(__name__)

_CLEANUP_INTERVAL = 500


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
    def __init__(self, state_file: str | Path | None = None, *, save_interval_seconds: float = 60.0) -> None:
        self._counters: dict[str, _Counters] = {}
        self._ops: int = 0
        self._state_file = Path(state_file) if state_file else None
        self._save_interval_seconds = save_interval_seconds
        self._last_save_monotonic = time.monotonic()
        self._dirty = False
        if self._state_file:
            self._load()
            atexit.register(self.save)

    def _get(self, key: str) -> _Counters:
        if key not in self._counters:
            self._counters[key] = _Counters()
        return self._counters[key]

    def is_allowed(self, key: str, limits: RateLimits) -> bool:
        c = self._get(key)
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
        c = self._get(key)
        wk = _window_keys()
        for period in ("m", "h", "d"):
            c.requests[wk[period]] += 1
        self._dirty = True
        self._tick_cleanup()

    def record_tokens(self, key: str, tokens: int) -> None:
        if tokens <= 0:
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
        if not self._state_file or not self._dirty:
            return None
        if time.monotonic() - self._last_save_monotonic < self._save_interval_seconds:
            return None
        return self.snapshot()

    def snapshot(self) -> tuple[Path, dict[str, dict[str, dict[str, int]]]] | None:
        if not self._state_file or not self._dirty:
            return None
        self._cleanup()
        data = self._to_json_data()
        self._dirty = False
        self._last_save_monotonic = time.monotonic()
        return self._state_file, data

    def mark_dirty(self) -> None:
        self._dirty = True

    def counters_snapshot(self) -> dict[str, dict[str, dict[str, int]]]:
        self._cleanup()
        return self._to_json_data()

    def _tick_cleanup(self) -> None:
        self._ops += 1
        if self._ops % _CLEANUP_INTERVAL == 0:
            self._cleanup()

    def _cleanup(self) -> None:
        wk = _window_keys()
        current_keys = set(wk.values())
        for counters in self._counters.values():
            for d in (counters.requests, counters.tokens):
                stale = [k for k in d if k not in current_keys]
                for k in stale:
                    del d[k]

    def _load(self) -> None:
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
