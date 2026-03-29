"""
Tests for api/main.py — FastAPI endpoints and WebSocket.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aifactory.api.main import app, tasks

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _clear_task_store() -> None:
    """Clear the in-memory task store between tests."""
    tasks.clear()


@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI TestClient."""
    return TestClient(app)


# =============================================================================
# Health Endpoint
# =============================================================================


class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert "phase" in data
        assert "timestamp" in data

    def test_health_has_correct_fields(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        data = resp.json()
        assert set(data.keys()) == {"status", "version", "phase", "timestamp"}


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

    def test_submit_stores_task(self, client: TestClient) -> None:
        resp = client.post(
            "/api/tasks",
            json={"task": "Test task"},
        )
        task_id = resp.json()["task_id"]
        assert task_id in tasks
        # Status may be pending, running, or failed (background task races)
        assert tasks[task_id]["status"] in ("pending", "running", "failed")

    def test_submit_default_workers(self, client: TestClient) -> None:
        resp = client.post(
            "/api/tasks",
            json={"task": "Test"},
        )
        task_id = resp.json()["task_id"]
        assert tasks[task_id]["request"]["num_workers"] == 3

    def test_submit_custom_workers(self, client: TestClient) -> None:
        resp = client.post(
            "/api/tasks",
            json={"task": "Test", "num_workers": 5},
        )
        task_id = resp.json()["task_id"]
        assert tasks[task_id]["request"]["num_workers"] == 5

    def test_submit_missing_task_field(self, client: TestClient) -> None:
        resp = client.post("/api/tasks", json={})
        assert resp.status_code == 422  # Validation error


# =============================================================================
# Task Status Polling
# =============================================================================


class TestGetTask:
    def test_get_existing_task(self, client: TestClient) -> None:
        # Submit first
        submit_resp = client.post(
            "/api/tasks",
            json={"task": "Test task"},
        )
        task_id = submit_resp.json()["task_id"]

        # Poll
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
# WebSocket
# =============================================================================


class TestWebSocket:
    def test_ws_unknown_task(self, client: TestClient) -> None:
        """Connecting to WS for non-existent task should get error."""
        with client.websocket_connect("/api/ws/nonexistent") as ws:
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "not found" in data["message"].lower()

    def test_ws_connects_to_existing_task(self, client: TestClient) -> None:
        """Can connect to WS for an existing task."""
        submit_resp = client.post(
            "/api/tasks",
            json={"task": "Test"},
        )
        task_id = submit_resp.json()["task_id"]

        # WebSocket should connect and register as subscriber
        with client.websocket_connect(f"/api/ws/{task_id}") as _ws:
            # The subscriber list should include our connection
            assert len(tasks[task_id]["subscribers"]) >= 1


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
        # Should not return 403
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
        assert data["info"]["version"] == "0.1.0"

    def test_docs_page(self, client: TestClient) -> None:
        resp = client.get("/api/docs")
        assert resp.status_code == 200
