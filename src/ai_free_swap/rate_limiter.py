from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

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
    def __init__(self) -> None:
        self._counters: dict[int, _Counters] = {}
        self._ops: int = 0

    def _get(self, key: int) -> _Counters:
        if key not in self._counters:
            self._counters[key] = _Counters()
        return self._counters[key]

    def is_allowed(self, key: int, limits: RateLimits) -> bool:
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

    def record_request(self, key: int) -> None:
        c = self._get(key)
        wk = _window_keys()
        for period in ("m", "h", "d"):
            c.requests[wk[period]] += 1
        self._tick_cleanup()

    def record_tokens(self, key: int, tokens: int) -> None:
        if tokens <= 0:
            return
        c = self._get(key)
        wk = _window_keys()
        for period in ("m", "h", "d"):
            c.tokens[wk[period]] += tokens
        self._tick_cleanup()

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
