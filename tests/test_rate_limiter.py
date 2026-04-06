"""
Tests for API rate limiting.

Verifies:
  - Rate limiter module configuration
  - 429 responses when limits are exceeded
  - Custom JSON error body
  - Rate limit info in health response
  - Disabling rate limiting via env var
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from aifactory.api.main import app
from aifactory.core.rate_limiter import (
    RATE_LIMIT_MEMORY,
    RATE_LIMIT_READ,
    RATE_LIMIT_TASKS,
    RATE_LIMIT_WS,
    get_rate_limit_info,
    limiter,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

# We need to mock DB/Redis for all tests (same as test_api.py)
_fake_db: dict[str, dict] = {}
_fake_agents: dict[str, list] = {}
_fake_messages: dict[str, list] = {}


def _reset_fake_db() -> None:
    _fake_db.clear()
    _fake_agents.clear()
    _fake_messages.clear()


async def _fake_create_task(task_id, status, created_at, request):
    _fake_db[task_id] = {
        "task_id": task_id,
        "status": status,
        "created_at": created_at,
        "updated_at": created_at,
        "progress": 0.0,
        "request": request,
        "result": None,
        "error": None,
        "metrics": {},
    }


async def _fake_update_task(task_id, **fields):
    if task_id in _fake_db:
        _fake_db[task_id].update(fields)


async def _fake_get_task(task_id):
    return _fake_db.get(task_id)


async def _fake_list_tasks(status_filter=None, limit=50):
    items = list(_fake_db.values())
    if status_filter:
        items = [t for t in items if t.get("status") == status_filter]
    return items[:limit]


async def _fake_upsert_agents(task_id, agents):
    _fake_agents[task_id] = agents


async def _fake_get_agents(task_id):
    return _fake_agents.get(task_id, [])


async def _fake_get_messages(task_id):
    return _fake_messages.get(task_id, [])


async def _fake_pg_health():
    return True


async def _fake_redis_health():
    return True


async def _fake_cache_noop(*_args, **_kwargs):
    return None


async def _fake_get_cache(*_args, **_kwargs):
    return None


async def _fake_invalidate(*_args, **_kwargs):
    return None


@pytest.fixture(autouse=True)
def _mock_database():
    """Mock all database + Redis calls so tests run without infrastructure."""
    _reset_fake_db()
    with (
        patch("aifactory.api.main.db.create_task", side_effect=_fake_create_task),
        patch("aifactory.api.main.db.update_task", side_effect=_fake_update_task),
        patch("aifactory.api.main.db.get_task", side_effect=_fake_get_task),
        patch("aifactory.api.main.db.list_tasks", side_effect=_fake_list_tasks),
        patch("aifactory.api.main.db.upsert_agents", side_effect=_fake_upsert_agents),
        patch("aifactory.api.main.db.get_agents", side_effect=_fake_get_agents),
        patch("aifactory.api.main.db.get_messages", side_effect=_fake_get_messages),
        patch("aifactory.api.main.db.pg_health", side_effect=_fake_pg_health),
        patch("aifactory.api.main.rc.redis_health", side_effect=_fake_redis_health),
        patch("aifactory.api.main.rc.cache_task", side_effect=_fake_cache_noop),
        patch("aifactory.api.main.rc.get_cached_task", side_effect=_fake_get_cache),
        patch("aifactory.api.main.rc.cache_task_list", side_effect=_fake_cache_noop),
        patch("aifactory.api.main.rc.get_cached_task_list", side_effect=_fake_get_cache),
        patch("aifactory.api.main.rc.invalidate_task", side_effect=_fake_invalidate),
        patch("aifactory.api.main.rc.invalidate_task_list", side_effect=_fake_invalidate),
    ):
        yield


@pytest.fixture
def client() -> TestClient:
    """Create a TestClient for the FastAPI app."""
    return TestClient(app, raise_server_exceptions=False)


# =============================================================================
# Rate Limiter Module Tests
# =============================================================================


class TestRateLimiterModule:
    """Test the rate_limiter.py module configuration."""

    def test_default_rate_limit_strings(self) -> None:
        """Default rate limits are set correctly."""
        assert RATE_LIMIT_TASKS == "10/minute"
        assert RATE_LIMIT_READ == "60/minute"
        assert RATE_LIMIT_MEMORY == "30/minute"
        assert RATE_LIMIT_WS == "5/minute"

    def test_limiter_is_singleton(self) -> None:
        """limiter object is importable and has correct type."""
        from slowapi import Limiter

        assert isinstance(limiter, Limiter)

    def test_get_rate_limit_info(self) -> None:
        """get_rate_limit_info returns the expected structure."""
        info = get_rate_limit_info()
        assert "enabled" in info
        assert "limits" in info
        assert "tasks_submit" in info["limits"]
        assert "read" in info["limits"]
        assert "memory" in info["limits"]
        assert "websocket" in info["limits"]

    def test_rate_limit_info_values(self) -> None:
        """Rate limit info returns actual configured values."""
        info = get_rate_limit_info()
        assert info["limits"]["tasks_submit"] == "10/minute"
        assert info["limits"]["read"] == "60/minute"
        assert info["limits"]["memory"] == "30/minute"
        assert info["limits"]["websocket"] == "5/minute"


# =============================================================================
# Health Endpoint — Rate Limit Info
# =============================================================================


class TestHealthRateLimitInfo:
    """Health endpoint now includes rate_limiting section."""

    def test_health_includes_rate_limiting(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "rate_limiting" in data
        rl = data["rate_limiting"]
        assert "enabled" in rl
        assert "limits" in rl

    def test_health_rate_limit_has_all_tiers(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        rl = resp.json()["rate_limiting"]
        limits = rl["limits"]
        assert "tasks_submit" in limits
        assert "read" in limits
        assert "memory" in limits
        assert "websocket" in limits


# =============================================================================
# Rate Limit Enforcement (429 responses)
# =============================================================================


class TestRateLimitEnforcement:
    """Test that endpoints return 429 when rate limit is exceeded."""

    def test_submit_task_rate_limit(self, client: TestClient) -> None:
        """POST /api/tasks should return 429 after exceeding limit."""
        # Reset limiter storage for clean test
        limiter.reset()

        # The limit is 10/minute. Send 11 requests to trigger 429.
        for i in range(10):
            resp = client.post(
                "/api/tasks",
                json={"task": f"Task {i}"},
            )
            assert resp.status_code == 202, (
                f"Request {i + 1} should succeed, got {resp.status_code}"
            )

        # 11th request should be rate-limited
        resp = client.post(
            "/api/tasks",
            json={"task": "One too many"},
        )
        assert resp.status_code == 429
        data = resp.json()
        assert data["error"] == "Rate limit exceeded"
        assert "retry_after" in data

    def test_rate_limit_429_json_body(self, client: TestClient) -> None:
        """429 responses have the correct JSON structure."""
        limiter.reset()

        # Exhaust the task submission limit
        for i in range(10):
            client.post("/api/tasks", json={"task": f"Task {i}"})

        resp = client.post("/api/tasks", json={"task": "Overflow"})
        assert resp.status_code == 429
        data = resp.json()
        assert "error" in data
        assert "detail" in data
        assert "retry_after" in data
        assert isinstance(data["retry_after"], int)

    def test_rate_limit_retry_after_header(self, client: TestClient) -> None:
        """429 responses include a Retry-After header."""
        limiter.reset()

        for i in range(10):
            client.post("/api/tasks", json={"task": f"Task {i}"})

        resp = client.post("/api/tasks", json={"task": "Overflow"})
        assert resp.status_code == 429
        assert "retry-after" in resp.headers

    def test_read_endpoints_have_higher_limit(self, client: TestClient) -> None:
        """GET endpoints should allow 60/min — much higher than write endpoints."""
        limiter.reset()

        # Health endpoint: send 30 requests — all should succeed (limit is 60)
        for i in range(30):
            resp = client.get("/api/health")
            assert resp.status_code == 200, f"Health request {i + 1} should succeed"

    def test_different_endpoints_have_separate_limits(self, client: TestClient) -> None:
        """Task submission and health check have independent rate limits."""
        limiter.reset()

        # Exhaust POST /api/tasks limit (10/min)
        for i in range(10):
            resp = client.post("/api/tasks", json={"task": f"Task {i}"})
            assert resp.status_code == 202

        # POST should now be limited
        resp = client.post("/api/tasks", json={"task": "Overflow"})
        assert resp.status_code == 429

        # But GET /api/health should still work (separate limit)
        resp = client.get("/api/health")
        assert resp.status_code == 200


# =============================================================================
# Rate Limit Enable/Disable
# =============================================================================


class TestRateLimitToggle:
    """Test enabling/disabling rate limiting via env var."""

    def test_rate_limit_info_reports_enabled(self) -> None:
        """When enabled, get_rate_limit_info reports enabled=True."""
        with patch.dict(os.environ, {"AIFACTORY_RATE_LIMIT_ENABLED": "true"}):
            info = get_rate_limit_info()
            assert info["enabled"] is True

    def test_rate_limit_info_reports_disabled(self) -> None:
        """When disabled, get_rate_limit_info reports enabled=False."""
        with patch.dict(os.environ, {"AIFACTORY_RATE_LIMIT_ENABLED": "false"}):
            info = get_rate_limit_info()
            assert info["enabled"] is False

    def test_env_var_accepts_multiple_true_values(self) -> None:
        """Env var accepts 'true', '1', 'yes'."""
        for val in ("true", "1", "yes", "True", "YES"):
            with patch.dict(os.environ, {"AIFACTORY_RATE_LIMIT_ENABLED": val}):
                info = get_rate_limit_info()
                assert info["enabled"] is True, f"Should be enabled for '{val}'"

    def test_env_var_rejects_non_true_values(self) -> None:
        """Env var treats anything else as disabled."""
        for val in ("false", "0", "no", "disabled"):
            with patch.dict(os.environ, {"AIFACTORY_RATE_LIMIT_ENABLED": val}):
                info = get_rate_limit_info()
                assert info["enabled"] is False, f"Should be disabled for '{val}'"


# =============================================================================
# Memory Endpoint Rate Limits
# =============================================================================


class TestMemoryRateLimits:
    """Memory endpoints use the MEMORY tier (30/min)."""

    def test_memory_stats_succeeds_within_limit(self, client: TestClient) -> None:
        """GET /api/memory/stats should succeed within rate limit."""
        limiter.reset()
        resp = client.get("/api/memory/stats")
        # May error due to no Qdrant, but should NOT be 429
        assert resp.status_code != 429

    def test_memory_store_succeeds_within_limit(self, client: TestClient) -> None:
        """POST /api/memory/store should succeed within rate limit."""
        limiter.reset()
        resp = client.post(
            "/api/memory/store",
            json={"level": "agent", "content": "test", "agent_id": "a1"},
        )
        assert resp.status_code != 429
