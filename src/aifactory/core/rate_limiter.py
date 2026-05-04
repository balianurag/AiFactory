"""
API Rate Limiting for AiFactory.

Uses `slowapi` (built on top of `limits`) to enforce per-endpoint
request rate limits. Limits are applied per client IP address.

Tiers:
  - Pipeline endpoints (POST /api/tasks): strict  — 10/minute
  - Read endpoints (GET /api/tasks, health, models): generous — 60/minute
  - Memory endpoints: moderate — 30/minute
  - WebSocket: 5 connections/minute per IP

Configuration via environment variables:
  AIFACTORY_RATE_LIMIT_ENABLED:  true/false (default: true)
  AIFACTORY_RATE_LIMIT_TASKS:    "10/minute" (default)
  AIFACTORY_RATE_LIMIT_READ:     "60/minute" (default)
  AIFACTORY_RATE_LIMIT_MEMORY:   "30/minute" (default)
  AIFACTORY_RATE_LIMIT_WS:       "5/minute"  (default)

When rate-limited, the API returns HTTP 429 Too Many Requests with a JSON body:
  { "error": "Rate limit exceeded", "detail": "...", "retry_after": <seconds> }

Headers included on every response:
  X-RateLimit-Limit: <max requests>
  X-RateLimit-Remaining: <requests left>
  X-RateLimit-Reset: <epoch timestamp when window resets>
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


def _is_rate_limiting_enabled() -> bool:
    """Check if rate limiting is enabled via env var."""
    return os.environ.get("AIFACTORY_RATE_LIMIT_ENABLED", "true").lower() in (
        "true",
        "1",
        "yes",
    )


def _get_limit(env_var: str, default: str) -> str:
    """Read a rate-limit string from env or return default."""
    return os.environ.get(env_var, default)


# ── Rate-limit strings (read from env at import time, overridable) ──────────

RATE_LIMIT_TASKS: str = _get_limit("AIFACTORY_RATE_LIMIT_TASKS", "10/minute")
RATE_LIMIT_READ: str = _get_limit("AIFACTORY_RATE_LIMIT_READ", "60/minute")
RATE_LIMIT_MEMORY: str = _get_limit("AIFACTORY_RATE_LIMIT_MEMORY", "30/minute")
RATE_LIMIT_WS: str = _get_limit("AIFACTORY_RATE_LIMIT_WS", "5/minute")


def _storage_uri() -> str:
    """
    Use Redis as the rate-limit store when REDIS_URL is set.

    Redis-backed storage survives process restarts and is shared across
    multiple workers in production.  Falls back to in-memory (single-process)
    when Redis is unavailable.
    """
    redis_url = os.environ.get("REDIS_URL", "").strip()
    if redis_url:
        return redis_url
    return "memory://"


# ── Limiter singleton ──────────────────────────────────────────────────────────

limiter = Limiter(
    key_func=get_remote_address,
    enabled=_is_rate_limiting_enabled(),
    default_limits=[],  # No default — each endpoint sets its own limit
    storage_uri=_storage_uri(),  # Redis when REDIS_URL is set; in-memory otherwise
)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Custom 429 handler with JSON body + Retry-After header."""
    # Extract retry window from the exception message
    retry_after = 60  # default fallback
    detail = str(exc.detail) if hasattr(exc, "detail") else "Rate limit exceeded"

    response = JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": detail,
            "retry_after": retry_after,
        },
    )
    response.headers["Retry-After"] = str(retry_after)
    return response


def get_rate_limit_info() -> dict[str, Any]:
    """Return current rate limit configuration (for /api/health)."""
    return {
        "enabled": _is_rate_limiting_enabled(),
        "limits": {
            "tasks_submit": RATE_LIMIT_TASKS,
            "read": RATE_LIMIT_READ,
            "memory": RATE_LIMIT_MEMORY,
            "websocket": RATE_LIMIT_WS,
        },
    }
