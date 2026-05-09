from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_free_swap.config import RateLimits
from ai_free_swap.rate_limiter import RateLimiter, _window_keys


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
        assert rl.is_allowed(1, limits) is True

    def test_under_rpm_allowed(self):
        rl = RateLimiter()
        limits = RateLimits(rpm=5)
        for _ in range(4):
            rl.record_request(1)
        assert rl.is_allowed(1, limits) is True

    def test_at_rpm_blocked(self):
        rl = RateLimiter()
        limits = RateLimits(rpm=5)
        for _ in range(5):
            rl.record_request(1)
        assert rl.is_allowed(1, limits) is False

    def test_rpd_blocked(self):
        rl = RateLimiter()
        limits = RateLimits(rpd=2)
        rl.record_request(1)
        rl.record_request(1)
        assert rl.is_allowed(1, limits) is False

    def test_rph_blocked(self):
        rl = RateLimiter()
        limits = RateLimits(rph=3)
        for _ in range(3):
            rl.record_request(1)
        assert rl.is_allowed(1, limits) is False

    def test_tpm_blocked(self):
        rl = RateLimiter()
        limits = RateLimits(tpm=1000)
        rl.record_tokens(1, 1000)
        assert rl.is_allowed(1, limits) is False

    def test_tph_blocked(self):
        rl = RateLimiter()
        limits = RateLimits(tph=5000)
        rl.record_tokens(1, 5000)
        assert rl.is_allowed(1, limits) is False

    def test_tpd_blocked(self):
        rl = RateLimiter()
        limits = RateLimits(tpd=10000)
        rl.record_tokens(1, 10000)
        assert rl.is_allowed(1, limits) is False

    def test_under_tpm_allowed(self):
        rl = RateLimiter()
        limits = RateLimits(tpm=1000)
        rl.record_tokens(1, 500)
        assert rl.is_allowed(1, limits) is True

    def test_multiple_limits_any_blocks(self):
        rl = RateLimiter()
        limits = RateLimits(rpm=100, rpd=2)
        rl.record_request(1)
        rl.record_request(1)
        assert rl.is_allowed(1, limits) is False

    def test_different_backends_independent(self):
        rl = RateLimiter()
        limits = RateLimits(rpd=2)
        rl.record_request(1)
        rl.record_request(1)
        assert rl.is_allowed(1, limits) is False
        assert rl.is_allowed(2, limits) is True


class TestRateLimiterRecording:
    def test_record_request_increments_all_windows(self):
        rl = RateLimiter()
        rl.record_request(1)
        c = rl._get(1)
        wk = _window_keys()
        assert c.requests[wk["m"]] == 1
        assert c.requests[wk["h"]] == 1
        assert c.requests[wk["d"]] == 1

    def test_record_tokens_increments_all_windows(self):
        rl = RateLimiter()
        rl.record_tokens(1, 500)
        c = rl._get(1)
        wk = _window_keys()
        assert c.tokens[wk["m"]] == 500
        assert c.tokens[wk["h"]] == 500
        assert c.tokens[wk["d"]] == 500

    def test_record_tokens_zero_ignored(self):
        rl = RateLimiter()
        rl.record_tokens(1, 0)
        assert 1 not in rl._counters

    def test_record_tokens_negative_ignored(self):
        rl = RateLimiter()
        rl.record_tokens(1, -5)
        assert 1 not in rl._counters


class TestWindowRollover:
    def test_new_minute_resets_rpm(self):
        rl = RateLimiter()
        limits = RateLimits(rpm=2)

        with patch("ai_free_swap.rate_limiter._window_keys") as mock_wk:
            mock_wk.return_value = {"m": "20260509T1430", "h": "20260509T14", "d": "20260509"}
            rl.record_request(1)
            rl.record_request(1)
            assert rl.is_allowed(1, limits) is False

            mock_wk.return_value = {"m": "20260509T1431", "h": "20260509T14", "d": "20260509"}
            assert rl.is_allowed(1, limits) is True

    def test_new_hour_resets_rph(self):
        rl = RateLimiter()
        limits = RateLimits(rph=1)

        with patch("ai_free_swap.rate_limiter._window_keys") as mock_wk:
            mock_wk.return_value = {"m": "20260509T1430", "h": "20260509T14", "d": "20260509"}
            rl.record_request(1)
            assert rl.is_allowed(1, limits) is False

            mock_wk.return_value = {"m": "20260509T1530", "h": "20260509T15", "d": "20260509"}
            assert rl.is_allowed(1, limits) is True

    def test_new_day_resets_rpd(self):
        rl = RateLimiter()
        limits = RateLimits(rpd=1)

        with patch("ai_free_swap.rate_limiter._window_keys") as mock_wk:
            mock_wk.return_value = {"m": "20260509T1430", "h": "20260509T14", "d": "20260509"}
            rl.record_request(1)
            assert rl.is_allowed(1, limits) is False

            mock_wk.return_value = {"m": "20260510T0000", "h": "20260510T00", "d": "20260510"}
            assert rl.is_allowed(1, limits) is True


class TestCleanup:
    def test_cleanup_removes_stale_keys(self):
        rl = RateLimiter()

        with patch("ai_free_swap.rate_limiter._window_keys") as mock_wk:
            mock_wk.return_value = {"m": "old_m", "h": "old_h", "d": "old_d"}
            rl.record_request(1)

            mock_wk.return_value = {"m": "new_m", "h": "new_h", "d": "new_d"}
            rl._cleanup()

        c = rl._get(1)
        assert "old_m" not in c.requests
        assert "old_h" not in c.requests
        assert "old_d" not in c.requests
