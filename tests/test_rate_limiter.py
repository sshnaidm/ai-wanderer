from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import ai_free_swap.rate_limiter as rate_limiter_module
from ai_free_swap.config import RateLimits, StateStoreConfig
from ai_free_swap.rate_limiter import RateLimiter, _window_keys


class _FakeRedisPipeline:
    def __init__(self, client):
        self.client = client
        self.ops = []

    def hincrby(self, key, field, amount):
        self.ops.append(("hincrby", key, field, amount))
        return self

    def expire(self, key, ttl):
        self.ops.append(("expire", key, ttl))
        return self

    def sadd(self, key, value):
        self.ops.append(("sadd", key, value))
        return self

    def hgetall(self, key):
        self.ops.append(("hgetall", key))
        return self

    async def execute(self):
        results = []
        for op in self.ops:
            name = op[0]
            if name == "hincrby":
                _, key, field, amount = op
                self.client.hincrby(key, field, amount)
                results.append(None)
            elif name == "expire":
                results.append(True)
                continue
            elif name == "sadd":
                _, key, value = op
                self.client.sadd(key, value)
                results.append(1)
            elif name == "hgetall":
                _, key = op
                results.append(self.client.hgetall(key))
        self.ops.clear()
        return results


class _FakeRedisClient:
    def __init__(self):
        self.hashes = {}
        self.sets = {}

    def pipeline(self):
        return _FakeRedisPipeline(self)

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def hincrby(self, key, field, amount):
        bucket = self.hashes.setdefault(key, {})
        bucket[field] = int(bucket.get(field, 0)) + amount

    def expire(self, key, ttl):
        return True

    def sadd(self, key, value):
        bucket = self.sets.setdefault(key, set())
        bucket.add(value)

    async def smembers(self, key):
        return set(self.sets.get(key, set()))

    async def srem(self, key, *values):
        bucket = self.sets.setdefault(key, set())
        removed = 0
        for value in values:
            if value in bucket:
                bucket.remove(value)
                removed += 1
        return removed

    async def eval(self, script, numkeys, request_key, token_key, registry_key, *args):
        periods = args[:3]
        request_limits = args[3:6]
        token_limits = args[6:9]
        backend_key = args[10]
        for period, request_limit, token_limit in zip(periods, request_limits, token_limits):
            request_count = int(self.hashes.get(request_key, {}).get(period, 0))
            token_count = int(self.hashes.get(token_key, {}).get(period, 0))
            if request_limit >= 0 and request_count >= request_limit:
                return 0
            if token_limit >= 0 and token_count >= token_limit:
                return 0
        for period in periods:
            self.hincrby(request_key, period, 1)
        self.sadd(registry_key, backend_key)
        return 1


class _FakeRedisModule:
    class Redis:
        @staticmethod
        def from_url(url, decode_responses=True):
            return _FakeRedisClient()


class TestWindowKeys:
    def test_returns_three_keys(self):
        wk = _window_keys()
        assert set(wk.keys()) == {"m", "h", "d"}

    def test_minute_key_contains_hour_and_minute(self):
        with patch("ai_free_swap.rate_limiter.time") as mock_time:
            mock_time.gmtime.return_value = type(
                "struct",
                (),
                {
                    "tm_year": 2026,
                    "tm_mon": 5,
                    "tm_mday": 9,
                    "tm_hour": 14,
                    "tm_min": 30,
                },
            )()
            wk = _window_keys()
        assert wk["m"] == "20260509T1430"
        assert wk["h"] == "20260509T14"
        assert wk["d"] == "20260509"


class TestRateLimiterIsAllowed:
    def test_no_limits_always_allowed(self):
        rl = RateLimiter()
        limits = RateLimits()
        assert rl.is_allowed("b1", limits) is True

    def test_under_rpm_allowed(self):
        rl = RateLimiter()
        limits = RateLimits(rpm=5)
        for _ in range(4):
            rl.record_request("b1")
        assert rl.is_allowed("b1", limits) is True

    def test_at_rpm_blocked(self):
        rl = RateLimiter()
        limits = RateLimits(rpm=5)
        for _ in range(5):
            rl.record_request("b1")
        assert rl.is_allowed("b1", limits) is False

    def test_rpd_blocked(self):
        rl = RateLimiter()
        limits = RateLimits(rpd=2)
        rl.record_request("b1")
        rl.record_request("b1")
        assert rl.is_allowed("b1", limits) is False

    def test_rph_blocked(self):
        rl = RateLimiter()
        limits = RateLimits(rph=3)
        for _ in range(3):
            rl.record_request("b1")
        assert rl.is_allowed("b1", limits) is False

    def test_tpm_blocked(self):
        rl = RateLimiter()
        limits = RateLimits(tpm=1000)
        rl.record_tokens("b1", 1000)
        assert rl.is_allowed("b1", limits) is False

    def test_tph_blocked(self):
        rl = RateLimiter()
        limits = RateLimits(tph=5000)
        rl.record_tokens("b1", 5000)
        assert rl.is_allowed("b1", limits) is False

    def test_tpd_blocked(self):
        rl = RateLimiter()
        limits = RateLimits(tpd=10000)
        rl.record_tokens("b1", 10000)
        assert rl.is_allowed("b1", limits) is False

    def test_under_tpm_allowed(self):
        rl = RateLimiter()
        limits = RateLimits(tpm=1000)
        rl.record_tokens("b1", 500)
        assert rl.is_allowed("b1", limits) is True

    def test_multiple_limits_any_blocks(self):
        rl = RateLimiter()
        limits = RateLimits(rpm=100, rpd=2)
        rl.record_request("b1")
        rl.record_request("b1")
        assert rl.is_allowed("b1", limits) is False

    def test_different_backends_independent(self):
        rl = RateLimiter()
        limits = RateLimits(rpd=2)
        rl.record_request("b1")
        rl.record_request("b1")
        assert rl.is_allowed("b1", limits) is False
        assert rl.is_allowed("b2", limits) is True


class TestRateLimiterRecording:
    def test_record_request_increments_all_windows(self):
        rl = RateLimiter()
        rl.record_request("b1")
        c = rl._get("b1")
        wk = _window_keys()
        assert c.requests[wk["m"]] == 1
        assert c.requests[wk["h"]] == 1
        assert c.requests[wk["d"]] == 1

    def test_record_tokens_increments_all_windows(self):
        rl = RateLimiter()
        rl.record_tokens("b1", 500)
        c = rl._get("b1")
        wk = _window_keys()
        assert c.tokens[wk["m"]] == 500
        assert c.tokens[wk["h"]] == 500
        assert c.tokens[wk["d"]] == 500

    def test_record_tokens_zero_ignored(self):
        rl = RateLimiter()
        rl.record_tokens("b1", 0)
        assert "b1" not in rl._counters

    def test_record_tokens_negative_ignored(self):
        rl = RateLimiter()
        rl.record_tokens("b1", -5)
        assert "b1" not in rl._counters


class TestWindowRollover:
    def test_new_minute_resets_rpm(self):
        rl = RateLimiter()
        limits = RateLimits(rpm=2)

        with patch("ai_free_swap.rate_limiter._window_keys") as mock_wk:
            mock_wk.return_value = {"m": "20260509T1430", "h": "20260509T14", "d": "20260509"}
            rl.record_request("b1")
            rl.record_request("b1")
            assert rl.is_allowed("b1", limits) is False

            mock_wk.return_value = {"m": "20260509T1431", "h": "20260509T14", "d": "20260509"}
            assert rl.is_allowed("b1", limits) is True

    def test_new_hour_resets_rph(self):
        rl = RateLimiter()
        limits = RateLimits(rph=1)

        with patch("ai_free_swap.rate_limiter._window_keys") as mock_wk:
            mock_wk.return_value = {"m": "20260509T1430", "h": "20260509T14", "d": "20260509"}
            rl.record_request("b1")
            assert rl.is_allowed("b1", limits) is False

            mock_wk.return_value = {"m": "20260509T1530", "h": "20260509T15", "d": "20260509"}
            assert rl.is_allowed("b1", limits) is True

    def test_new_day_resets_rpd(self):
        rl = RateLimiter()
        limits = RateLimits(rpd=1)

        with patch("ai_free_swap.rate_limiter._window_keys") as mock_wk:
            mock_wk.return_value = {"m": "20260509T1430", "h": "20260509T14", "d": "20260509"}
            rl.record_request("b1")
            assert rl.is_allowed("b1", limits) is False

            mock_wk.return_value = {"m": "20260510T0000", "h": "20260510T00", "d": "20260510"}
            assert rl.is_allowed("b1", limits) is True


class TestCleanup:
    def test_cleanup_removes_stale_keys(self):
        rl = RateLimiter()

        with patch("ai_free_swap.rate_limiter._window_keys") as mock_wk:
            mock_wk.return_value = {"m": "old_m", "h": "old_h", "d": "old_d"}
            rl.record_request("b1")

            mock_wk.return_value = {"m": "new_m", "h": "new_h", "d": "new_d"}
            rl._cleanup()

        c = rl._get("b1")
        assert "old_m" not in c.requests
        assert "old_h" not in c.requests
        assert "old_d" not in c.requests


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        state_file = tmp_path / "state.json"
        rl = RateLimiter(state_file=state_file)
        rl.record_request("backend-1")
        rl.record_tokens("backend-1", 500)
        rl._save()

        assert state_file.exists()

        rl2 = RateLimiter(state_file=state_file)
        wk = _window_keys()
        c = rl2._get("backend-1")
        assert c.requests[wk["d"]] == 1
        assert c.tokens[wk["d"]] == 500

    def test_load_rejects_non_dict_json(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text("[]")
        rl = RateLimiter(state_file=state_file)
        assert rl._counters == {}
        assert rl.is_allowed("b1", RateLimits(rpd=2)) is True

    def test_load_discards_stale_data(self, tmp_path):
        state_file = tmp_path / "state.json"
        data = {
            "backend-1": {
                "requests": {"20200101": 99, "20200101T00": 50},
                "tokens": {"20200101": 9999},
            }
        }
        state_file.write_text(json.dumps(data))

        rl = RateLimiter(state_file=state_file)
        c = rl._get("backend-1")
        assert c.requests["20200101"] == 0
        assert c.tokens["20200101"] == 0

    def test_load_keeps_current_day_data(self, tmp_path):
        state_file = tmp_path / "state.json"
        wk = _window_keys()
        data = {
            "backend-1": {
                "requests": {wk["d"]: 5, "20200101": 99},
                "tokens": {wk["d"]: 1000},
            }
        }
        state_file.write_text(json.dumps(data))

        rl = RateLimiter(state_file=state_file)
        c = rl._get("backend-1")
        assert c.requests[wk["d"]] == 5
        assert c.tokens[wk["d"]] == 1000
        assert c.requests["20200101"] == 0

    def test_no_state_file_no_crash(self):
        rl = RateLimiter()
        rl.record_request("b1")
        assert rl.is_allowed("b1", RateLimits(rpd=2)) is True

    def test_corrupt_state_file_ignored(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text("not valid json {{{")
        rl = RateLimiter(state_file=state_file)
        assert rl.is_allowed("b1", RateLimits(rpd=2)) is True

    def test_missing_state_file_ignored(self, tmp_path):
        state_file = tmp_path / "nonexistent.json"
        rl = RateLimiter(state_file=state_file)
        assert rl.is_allowed("b1", RateLimits(rpd=2)) is True

    def test_save_if_due_waits_for_interval(self, tmp_path):
        state_file = tmp_path / "state.json"
        rl = RateLimiter(state_file=state_file, save_interval_seconds=60)
        rl._last_save_monotonic = 100
        rl.record_request("backend-1")

        with patch("ai_free_swap.rate_limiter.time.monotonic", return_value=159):
            rl.save_if_due()

        assert not state_file.exists()

    def test_save_if_due_writes_after_interval(self, tmp_path):
        state_file = tmp_path / "state.json"
        rl = RateLimiter(state_file=state_file, save_interval_seconds=60)
        rl._last_save_monotonic = 100
        rl.record_request("backend-1")

        with patch("ai_free_swap.rate_limiter.time.monotonic", return_value=160):
            rl.save_if_due()

        assert state_file.exists()

    def test_save_drops_stale_windows(self, tmp_path):
        state_file = tmp_path / "state.json"
        rl = RateLimiter(state_file=state_file)

        with patch("ai_free_swap.rate_limiter._window_keys") as mock_wk:
            mock_wk.return_value = {"m": "old_m", "h": "old_h", "d": "old_d"}
            rl.record_request("backend-1")

            mock_wk.return_value = {"m": "new_m", "h": "new_h", "d": "new_d"}
            rl.save()

        data = json.loads(state_file.read_text())
        assert data == {}


class TestRedisStateStore:
    @pytest.mark.asyncio
    async def test_redis_backend_records_and_blocks(self, monkeypatch):
        monkeypatch.setattr(rate_limiter_module, "redis", _FakeRedisModule)
        rl = RateLimiter(
            state_store=StateStoreConfig(type="redis", redis_url="redis://example", redis_prefix="test"),
        )
        limits = RateLimits(rpm=2)
        await rl.record_request_async("backend-1")
        await rl.record_request_async("backend-1")
        assert await rl.is_allowed_async("backend-1", limits) is False

    @pytest.mark.asyncio
    async def test_redis_backend_snapshot_includes_counters(self, monkeypatch):
        monkeypatch.setattr(rate_limiter_module, "redis", _FakeRedisModule)
        rl = RateLimiter(
            state_store=StateStoreConfig(type="redis", redis_url="redis://example", redis_prefix="test"),
        )
        await rl.record_request_async("backend-1")
        await rl.record_tokens_async("backend-1", 50)
        snapshot = await rl.counters_snapshot_async()
        assert snapshot["backend-1"]["requests"][_window_keys()["d"]] == 1
        assert snapshot["backend-1"]["tokens"][_window_keys()["d"]] == 50

    @pytest.mark.asyncio
    async def test_redis_reservation_is_atomic_and_prunes_stale_registry(self, monkeypatch):
        monkeypatch.setattr(rate_limiter_module, "redis", _FakeRedisModule)
        rl = RateLimiter(
            state_store=StateStoreConfig(type="redis", redis_url="redis://example", redis_prefix="test"),
        )
        limits = RateLimits(rpm=1)

        assert await rl.reserve_request("backend-1", limits) is True
        assert await rl.reserve_request("backend-1", limits) is False

        client = rl._redis_client
        client.sets[rl._redis_known_keys_key()].add("stale-backend")
        await rl.counters_snapshot_async()
        assert "stale-backend" not in client.sets[rl._redis_known_keys_key()]
