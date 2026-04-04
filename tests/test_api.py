"""
Tests for api/main.py — FastAPI endpoints and WebSocket.

All database operations are mocked so tests run without PostgreSQL/Redis.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from aifactory.api.main import app

# =============================================================================
# Fixtures
# =============================================================================

# A simple in-memory store that simulates Postgres for tests
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
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
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
    """Create a FastAPI TestClient."""
    return TestClient(app, raise_server_exceptions=False)


# =============================================================================
# Health Endpoint
# =============================================================================


class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.3.0"
        assert "phase" in data
        assert "timestamp" in data
        assert "services" in data

    def test_health_has_correct_fields(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        data = resp.json()
        assert {"status", "version", "phase", "timestamp", "services"} == set(data.keys())

    def test_health_services_connected(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        data = resp.json()
        assert data["services"]["postgres"] == "connected"
        assert data["services"]["redis"] == "connected"


# =============================================================================
# Task Submission
# =============================================================================


class TestSubmitTask:
    def test_submit_returns_202(self, client: TestClient) -> None:
        resp = client.post(
            "/api/tasks",
            json={"task": "Build a calculator", "context": "", "num_workers": 2},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "pending"
        assert "created_at" in data

    def test_submit_stores_task_in_db(self, client: TestClient) -> None:
        resp = client.post(
            "/api/tasks",
            json={"task": "Test task"},
        )
        task_id = resp.json()["task_id"]
        assert task_id in _fake_db
        assert _fake_db[task_id]["status"] in ("pending", "running", "failed")

    def test_submit_default_workers(self, client: TestClient) -> None:
        resp = client.post(
            "/api/tasks",
            json={"task": "Test"},
        )
        task_id = resp.json()["task_id"]
        assert _fake_db[task_id]["request"]["num_workers"] == 3

    def test_submit_custom_workers(self, client: TestClient) -> None:
        resp = client.post(
            "/api/tasks",
            json={"task": "Test", "num_workers": 5},
        )
        task_id = resp.json()["task_id"]
        assert _fake_db[task_id]["request"]["num_workers"] == 5

    def test_submit_missing_task_field(self, client: TestClient) -> None:
        resp = client.post("/api/tasks", json={})
        assert resp.status_code == 422


# =============================================================================
# Task Status Polling
# =============================================================================


class TestGetTask:
    def test_get_existing_task(self, client: TestClient) -> None:
        submit_resp = client.post(
            "/api/tasks",
            json={"task": "Test task"},
        )
        task_id = submit_resp.json()["task_id"]

        resp = client.get(f"/api/tasks/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == task_id
        assert "status" in data
        assert "created_at" in data
        assert "updated_at" in data

    def test_get_nonexistent_task(self, client: TestClient) -> None:
        resp = client.get("/api/tasks/nonexistent-id-12345")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_task_has_all_status_fields(self, client: TestClient) -> None:
        submit_resp = client.post(
            "/api/tasks",
            json={"task": "Test"},
        )
        task_id = submit_resp.json()["task_id"]
        resp = client.get(f"/api/tasks/{task_id}")
        data = resp.json()

        expected_fields = {
            "task_id",
            "status",
            "created_at",
            "updated_at",
            "progress",
            "agents",
            "messages",
            "result",
            "error",
            "metrics",
        }
        assert expected_fields.issubset(set(data.keys()))


# =============================================================================
# Task Listing
# =============================================================================


class TestListTasks:
    def test_list_empty(self, client: TestClient) -> None:
        resp = client.get("/api/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tasks"] == []
        assert data["total"] == 0

    def test_list_after_submit(self, client: TestClient) -> None:
        client.post("/api/tasks", json={"task": "Task 1"})
        client.post("/api/tasks", json={"task": "Task 2"})
        resp = client.get("/api/tasks")
        data = resp.json()
        assert data["total"] >= 2


# =============================================================================
# WebSocket
# =============================================================================


class TestWebSocket:
    def test_ws_unknown_task(self, client: TestClient) -> None:
        """Connecting to WS for non-existent task should get error."""
        with client.websocket_connect("/api/ws/nonexistent") as ws:
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "not found" in data["message"].lower()


# =============================================================================
# CORS
# =============================================================================


class TestCORS:
    def test_cors_allows_localhost(self, client: TestClient) -> None:
        resp = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code in (200, 204, 405)


# =============================================================================
# OpenAPI / Docs
# =============================================================================


class TestDocs:
    def test_openapi_json(self, client: TestClient) -> None:
        resp = client.get("/api/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["info"]["title"] == "AiFactory API"
        assert data["info"]["version"] == "0.3.0"

    def test_docs_page(self, client: TestClient) -> None:
        resp = client.get("/api/docs")
        assert resp.status_code == 200


# =============================================================================
# Observability Endpoints
# =============================================================================


class TestObservability:
    def test_observability_status(self, client: TestClient) -> None:
        resp = client.get("/api/observability/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "enabled" in data
        assert isinstance(data["enabled"], bool)

    def test_list_runs_without_key(self, client: TestClient) -> None:
        resp = client.get("/api/observability/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert "runs" in data
        assert isinstance(data["runs"], list)

    def test_list_runs_with_limit(self, client: TestClient) -> None:
        resp = client.get("/api/observability/runs?limit=5")
        assert resp.status_code == 200

    def test_get_run_without_key(self, client: TestClient) -> None:
        from unittest.mock import patch as _p

        with _p("aifactory.core.observability.is_tracing_enabled", return_value=False):
            resp = client.get("/api/observability/runs/fake-run-id")
            assert resp.status_code == 503
