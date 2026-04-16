"""Tests for retry.py — exponential backoff + circuit breaker."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aifactory.core.retry import (
    is_circuit_open,
    record_failure,
    record_success,
    reset_circuit,
    retry_sync_with_backoff,
    retry_with_backoff,
)


@pytest.fixture(autouse=True)
def _reset_circuit():
    """Reset circuit-breaker state before each test."""
    reset_circuit()
    yield
    reset_circuit()


# ── Async retry ───────────────────────────────────────────────────────────────


class TestRetryWithBackoff:
    async def test_succeeds_on_first_attempt(self):
        calls = []

        async def _ok():
            calls.append(1)
            return "result"

        result = await retry_with_backoff(lambda: _ok(), max_retries=3)
        assert result == "result"
        assert len(calls) == 1

    async def test_retries_on_generic_exception(self):
        attempt = 0

        async def _flaky():
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise ValueError("transient")
            return "ok"

        with patch("aifactory.core.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await retry_with_backoff(lambda: _flaky(), max_retries=3, base_delay=0.001)

        assert result == "ok"
        assert attempt == 3

    async def test_raises_after_all_retries_exhausted(self):
        async def _always_fails():
            raise RuntimeError("always bad")

        with (
            patch("aifactory.core.retry.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RuntimeError, match="always bad"),
        ):
            await retry_with_backoff(lambda: _always_fails(), max_retries=3, base_delay=0.001)

    async def test_rate_limit_uses_long_delay(self):
        """RateLimitError should trigger a 5–15 s jitter sleep, not base_delay."""
        import litellm

        sleep_calls: list[float] = []

        async def _capture_sleep(delay):
            sleep_calls.append(delay)

        attempt = 0

        async def _rate_limited():
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise litellm.RateLimitError("too many", llm_provider="openai", model="gpt-4o")
            return "done"

        with patch("aifactory.core.retry.asyncio.sleep", side_effect=_capture_sleep):
            result = await retry_with_backoff(
                lambda: _rate_limited(), max_retries=3, base_delay=0.001, jitter=False
            )

        assert result == "done"
        assert len(sleep_calls) == 1
        assert 5.0 <= sleep_calls[0] <= 15.0

    async def test_records_success_on_model_id(self):
        async def _ok():
            return "x"

        record_failure("some-model")
        record_failure("some-model")
        await retry_with_backoff(lambda: _ok(), model_id="some-model")
        # Success resets the counter
        assert not is_circuit_open("some-model", threshold=3)

    async def test_records_failure_on_model_id(self):
        async def _fail():
            raise RuntimeError("fail")

        with (
            patch("aifactory.core.retry.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RuntimeError),
        ):
            await retry_with_backoff(
                lambda: _fail(), max_retries=3, base_delay=0.001, model_id="bad-model"
            )

        assert is_circuit_open("bad-model", threshold=3)


# ── Circuit breaker ───────────────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_circuit_closed_initially(self):
        assert not is_circuit_open("gpt-4o", threshold=3)

    def test_circuit_opens_after_threshold(self):
        for _ in range(3):
            record_failure("gpt-4o")
        assert is_circuit_open("gpt-4o", threshold=3)

    def test_circuit_does_not_open_below_threshold(self):
        record_failure("gpt-4o")
        record_failure("gpt-4o")
        assert not is_circuit_open("gpt-4o", threshold=3)

    def test_circuit_resets_on_success(self):
        record_failure("gpt-4o")
        record_failure("gpt-4o")
        record_failure("gpt-4o")
        assert is_circuit_open("gpt-4o", threshold=3)
        record_success("gpt-4o")
        assert not is_circuit_open("gpt-4o", threshold=3)

    def test_reset_clears_all(self):
        record_failure("model-a")
        record_failure("model-b")
        reset_circuit()
        assert not is_circuit_open("model-a", threshold=1)
        assert not is_circuit_open("model-b", threshold=1)

    def test_reset_single_model(self):
        record_failure("model-a")
        record_failure("model-b")
        reset_circuit("model-a")
        assert not is_circuit_open("model-a", threshold=1)
        assert is_circuit_open("model-b", threshold=1)


# ── Sync retry ────────────────────────────────────────────────────────────────


class TestRetrySyncWithBackoff:
    def test_succeeds_on_first_attempt(self):
        calls = []

        def _ok():
            calls.append(1)
            return "result"

        result = retry_sync_with_backoff(_ok, max_retries=3)
        assert result == "result"
        assert len(calls) == 1

    def test_retries_on_failure(self):
        attempt = 0

        def _flaky():
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise ValueError("transient")
            return "ok"

        with patch("aifactory.core.retry.time.sleep"):
            result = retry_sync_with_backoff(_flaky, max_retries=3, base_delay=0.001)

        assert result == "ok"
        assert attempt == 3

    def test_raises_after_all_retries(self):
        def _always_fails():
            raise RuntimeError("always bad")

        with (
            patch("aifactory.core.retry.time.sleep"),
            pytest.raises(RuntimeError, match="always bad"),
        ):
            retry_sync_with_backoff(_always_fails, max_retries=3, base_delay=0.001)

    def test_rate_limit_detected_via_predicate(self):
        sleep_calls: list[float] = []

        def _capture_sleep(delay):
            sleep_calls.append(delay)

        attempt = 0

        class FakeHTTPError(Exception):
            pass

        def _rate_limited():
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise FakeHTTPError("429 Too Many Requests")
            return "done"

        def _is_429(exc):
            return "429" in str(exc)

        with patch("aifactory.core.retry.time.sleep", side_effect=_capture_sleep):
            result = retry_sync_with_backoff(
                _rate_limited,
                max_retries=3,
                base_delay=0.001,
                jitter=False,
                is_rate_limit_exc=_is_429,
            )

        assert result == "done"
        assert len(sleep_calls) == 1
        assert 5.0 <= sleep_calls[0] <= 15.0
