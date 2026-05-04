"""
Shared retry + exponential-backoff utility for LLM calls and tool calls.

Used by:
  - dynamic_pipeline.py  — LiteLLM retry + circuit breaker + fallback model
  - composio.py          — Composio SDK 429 retry
  - mcp_provider.py      — MCP session.call_tool 429 retry

Circuit-breaker state: Redis-first (atomic INCR/EXPIRE), JSON-file fallback.
Redis keys: aifactory:cb:{model_id}  — integer failure count with TTL.
JSON fallback: ~/.aifactory/circuit_state.json (when Redis is unavailable).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ── Circuit-breaker constants ─────────────────────────────────────────────────

_CIRCUIT_OPEN_THRESHOLD: int = 3  # overridden by settings at call time
_CB_KEY_PREFIX = "aifactory:cb:"

# ── JSON-file fallback (used when Redis is unavailable) ───────────────────────
# Stored as {model_id: {"count": int, "last_failure_ts": float}}
_circuit_failures: dict[str, dict[str, Any]] = {}
_STATE_FILE = Path.home() / ".aifactory" / "circuit_state.json"
_state_loaded: bool = False

# ── Sync Redis client singleton ───────────────────────────────────────────────

_sync_redis: Any = None  # redis.Redis instance
_redis_checked: bool = False  # True once we've attempted to connect


def _get_sync_redis() -> Any:
    """
    Return a sync Redis client, or None if Redis is unavailable.

    Lazy-initialised on first call.  Silently returns None if REDIS_URL is not
    set or if the connection fails — callers fall through to the JSON-file path.
    """
    global _sync_redis, _redis_checked
    if _redis_checked:
        return _sync_redis
    _redis_checked = True

    import os

    redis_url = os.getenv("REDIS_URL", "")
    if not redis_url:
        return None
    try:
        import redis as _redis_mod

        client = _redis_mod.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
        )
        client.ping()
        _sync_redis = client
        logger.debug("Circuit breaker: Redis sync client connected")
    except Exception as exc:
        logger.debug("Circuit breaker: Redis unavailable (%s) — using JSON fallback", exc)
    return _sync_redis


# ── TTL ───────────────────────────────────────────────────────────────────────


def _get_ttl() -> int:
    """Return the configured TTL in seconds (default 3600)."""
    try:
        from aifactory.config.settings import get_settings

        return max(1, int(get_settings().circuit_breaker_ttl_s))
    except Exception:
        return 3600


# ── JSON-file helpers (fallback only) ────────────────────────────────────────


def _load_state() -> None:
    global _circuit_failures, _state_loaded
    if _state_loaded:
        return
    _state_loaded = True
    try:
        if _STATE_FILE.exists():
            data = json.loads(_STATE_FILE.read_text())
            if isinstance(data, dict):
                _circuit_failures = data
    except Exception:
        _circuit_failures = {}


def _save_state() -> None:
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(_circuit_failures))
        tmp.replace(_STATE_FILE)
    except Exception:
        pass


def _current_failures_json(model_id: str) -> int:
    """Return current failure count from JSON fallback, honouring TTL."""
    _load_state()
    ttl = _get_ttl()
    entry = _circuit_failures.get(model_id)
    if entry is None:
        return 0
    if ttl > 0 and (time.time() - entry.get("last_failure_ts", 0.0)) > ttl:
        _circuit_failures.pop(model_id, None)
        _save_state()
        return 0
    return int(entry.get("count", 0))


# ── Public circuit-breaker API ────────────────────────────────────────────────


def is_circuit_open(model_id: str, threshold: int = _CIRCUIT_OPEN_THRESHOLD) -> bool:
    """Return True if the circuit for *model_id* is open (too many failures)."""
    r = _get_sync_redis()
    if r is not None:
        try:
            count = r.get(f"{_CB_KEY_PREFIX}{model_id}")
            return int(count or 0) >= threshold
        except Exception:
            pass
    return _current_failures_json(model_id) >= threshold


def record_failure(model_id: str) -> None:
    """Increment consecutive failure count for a model."""
    ttl = _get_ttl()
    r = _get_sync_redis()
    if r is not None:
        try:
            key = f"{_CB_KEY_PREFIX}{model_id}"
            count = r.incr(key)
            r.expire(key, ttl)
            logger.warning(
                "Circuit breaker: %d consecutive failure(s) for '%s' (threshold=%d)",
                count,
                model_id,
                _CIRCUIT_OPEN_THRESHOLD,
            )
            return
        except Exception:
            pass
    # JSON fallback
    _load_state()
    count = _current_failures_json(model_id) + 1
    _circuit_failures[model_id] = {"count": count, "last_failure_ts": time.time()}
    _save_state()
    logger.warning(
        "Circuit breaker: %d consecutive failure(s) for '%s' (threshold=%d)",
        count,
        model_id,
        _CIRCUIT_OPEN_THRESHOLD,
    )


def record_success(model_id: str) -> None:
    """Reset consecutive failure count after a successful call."""
    r = _get_sync_redis()
    if r is not None:
        try:
            r.delete(f"{_CB_KEY_PREFIX}{model_id}")
            return
        except Exception:
            pass
    # JSON fallback
    _load_state()
    if model_id in _circuit_failures:
        _circuit_failures.pop(model_id)
        _save_state()


def reset_circuit(model_id: str | None = None) -> None:
    """Reset circuit state — clears Redis and JSON fallback."""
    global _state_loaded
    r = _get_sync_redis()
    if r is not None:
        try:
            if model_id:
                r.delete(f"{_CB_KEY_PREFIX}{model_id}")
            else:
                # Scan and delete all circuit breaker keys
                keys = list(r.scan_iter(f"{_CB_KEY_PREFIX}*"))
                if keys:
                    r.delete(*keys)
        except Exception:
            pass
    # Also clear JSON fallback
    _load_state()
    if model_id:
        _circuit_failures.pop(model_id, None)
    else:
        _circuit_failures.clear()
        _state_loaded = True
    _save_state()


# ── Async retry (for LLM + MCP async calls) ──────────────────────────────────


async def retry_with_backoff[T](
    coro_fn: Callable[[], Coroutine[Any, Any, T]],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    jitter: bool = True,
    model_id: str = "",
    is_rate_limit_exc: Callable[[Exception], bool] | None = None,
) -> T:
    """
    Execute an async callable with exponential backoff retry.

    Retry delays (multiplier ×3 each step, from base_delay):
      attempt 1 → fail  → sleep base_delay
      attempt 2 → fail  → sleep base_delay * 3
      attempt 3 → raise

    On litellm.RateLimitError (or when is_rate_limit_exc returns True):
      Sleep a random 5–15 s jitter delay instead of the shorter backoff.

    On all retries exhausted: re-raises the last exception.

    Args:
        coro_fn:          Zero-arg factory that returns a fresh coroutine each call.
        max_retries:      Total attempts (first try counts, so 3 = 1 + 2 retries).
        base_delay:       Base sleep seconds between retries.
        jitter:           Add ±30 % random jitter to avoid thundering herd.
        model_id:         Used for circuit-breaker tracking (empty = disabled).
        is_rate_limit_exc: Optional predicate to detect 429-equivalent errors
                          from non-litellm clients (e.g. MCP, Composio).
    """
    try:
        import litellm as _litellm

        _rate_limit_type: type = _litellm.RateLimitError
    except Exception:
        _rate_limit_type = type(None)  # type: ignore[assignment]

    delays = [base_delay * (3**i) for i in range(max_retries - 1)]
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            result = await coro_fn()
            if model_id:
                record_success(model_id)
            return result

        except Exception as exc:
            last_exc = exc
            if model_id:
                record_failure(model_id)

            if attempt >= max_retries - 1:
                break  # all attempts exhausted — raise below

            is_429 = isinstance(exc, _rate_limit_type) or (
                is_rate_limit_exc(exc) if is_rate_limit_exc else False
            )

            if is_429:
                delay = random.uniform(5.0, 15.0)
                logger.warning(
                    "Rate-limit on attempt %d/%d for '%s' — sleeping %.1f s",
                    attempt + 1,
                    max_retries,
                    model_id or "?",
                    delay,
                )
            else:
                delay = delays[attempt]
                if jitter:
                    delay += random.uniform(0.0, delay * 0.3)
                logger.warning(
                    "LLM call failed (attempt %d/%d) for '%s': %s — retrying in %.1f s",
                    attempt + 1,
                    max_retries,
                    exc,
                    model_id or "?",
                    delay,
                )

            await asyncio.sleep(delay)

    raise last_exc  # type: ignore[misc]


# ── Sync retry (for blocking SDK calls, e.g. Composio) ───────────────────────


def retry_sync_with_backoff[T](
    fn: Callable[[], T],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    jitter: bool = True,
    is_rate_limit_exc: Callable[[Exception], bool] | None = None,
) -> T:
    """
    Synchronous retry with exponential backoff.

    Same retry logic as the async version, using time.sleep instead of
    asyncio.sleep.  Used for blocking SDK calls (e.g. Composio SDK).
    """
    delays = [base_delay * (3**i) for i in range(max_retries - 1)]
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc

            if attempt >= max_retries - 1:
                break

            is_429 = is_rate_limit_exc(exc) if is_rate_limit_exc else False
            if is_429:
                delay = random.uniform(5.0, 15.0)
                logger.warning(
                    "Rate-limit on sync attempt %d/%d — sleeping %.1f s [%s]",
                    attempt + 1,
                    max_retries,
                    delay,
                    exc,
                )
            else:
                delay = delays[attempt]
                if jitter:
                    delay += random.uniform(0.0, delay * 0.3)
                logger.warning(
                    "Sync call failed (attempt %d/%d): %s — retrying in %.1f s",
                    attempt + 1,
                    max_retries,
                    exc,
                    delay,
                )
            time.sleep(delay)

    raise last_exc  # type: ignore[misc]
