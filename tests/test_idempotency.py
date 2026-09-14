"""
Tests for Fix 1 — Idempotency-Key support on POST /api/tasks.

Covers:
  - HIT  : same key + same body → original task_id, HTTP 200, Idempotency-Replayed header
  - MISS : first time a key is used → fresh task_id, HTTP 202
  - BODY MISMATCH : same key, different body → HTTP 422 with structured error
  - TTL EXPIRY    : record older than ttl_seconds is treated as a miss
  - REDIS-DOWN FALLBACK : in-process store works when Redis singleton is absent
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from aifactory.api.main import app
from aifactory.core.idempotency import (
    HEADER_NAME,
    IdempotencyConflict,
    IdempotencyStore,
    canonical_body_hash,
    get_idempotency_store,
    reset_idempotency_store,
)

# ── Reuse the in-memory fake DB from test_api.py to avoid PG dependency ──────

_fake_db: dict[str, dict[str, Any]] = {}
_fake_agents: dict[str, list[Any]] = {}
_fake_messages: dict[str, list[Any]] = {}


def _reset_fake_db() -> None:
    _fake_db.clear()
    _fake_agents.clear()
    _fake_messages.clear()


async def _fake_create_task(
    task_id: str, status: str, created_at: str, request: dict[str, Any]
) -> None:
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


async def _fake_update_task(task_id: str, **fields: Any) -> None:
    if task_id in _fake_db:
        _fake_db[task_id].update(fields)


async def _fake_get_task(task_id: str) -> Any:
    return _fake_db.get(task_id)


async def _fake_list_tasks(*, status_filter: str | None = None, limit: int = 50) -> list[Any]:
    items = list(_fake_db.values())
    if status_filter:
        items = [t for t in items if t.get("status") == status_filter]
    return items[:limit]


async def _fake_upsert_agents(task_id: str, agents: list[Any]) -> None:
    _fake_agents[task_id] = agents


async def _fake_get_agents(task_id: str) -> list[Any]:
    return _fake_agents.get(task_id, [])


async def _fake_get_messages(task_id: str) -> list[Any]:
    return _fake_messages.get(task_id, [])


async def _fake_pg_health() -> bool:
    return True


async def _fake_redis_health() -> bool:
    return False


async def _fake_cache_noop(*_args: Any, **_kwargs: Any) -> None:
    return None


async def _fake_get_cache(*_args: Any, **_kwargs: Any) -> Any:
    return None


async def _fake_invalidate(*_args: Any, **_kwargs: Any) -> None:
    return None


@pytest.fixture(autouse=True)
def _isolate_idempotency_and_db() -> Any:
    _reset_fake_db()
    reset_idempotency_store()

    # Force Redis-unavailable so we exercise the in-memory fallback by default
    with patch("aifactory.core.idempotency.IdempotencyStore._maybe_redis", return_value=None):
        from aifactory.core.rate_limiter import limiter

        limiter.reset()
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
    return TestClient(app, raise_server_exceptions=False)


# ─── Unit tests on the IdempotencyStore ──────────────────────────────────────


class TestIdempotencyStore:
    @pytest.mark.asyncio
    async def test_canonical_body_hash_is_key_order_insensitive(self) -> None:
        a = canonical_body_hash({"task": "Build", "context": "X"})
        b = canonical_body_hash({"context": "X", "task": "Build"})
        assert a == b

    @pytest.mark.asyncio
    async def test_canonical_body_hash_differs_on_value_change(self) -> None:
        a = canonical_body_hash({"task": "Build"})
        b = canonical_body_hash({"task": "build"})  # case matters
        assert a != b

    @pytest.mark.asyncio
    async def test_reserve_returns_none_when_key_unseen(self) -> None:
        store = IdempotencyStore()
        result = await store.reserve_or_replay(key="k1", body={"task": "x"})
        assert result is None

    @pytest.mark.asyncio
    async def test_replay_returns_record_for_same_body(self) -> None:
        store = IdempotencyStore()
        body = {"task": "x"}
        await store.store_response(
            key="k1",
            body=body,
            run_id="run-001",
            status="pending",
            response_snapshot={"task_id": "run-001", "status": "pending"},
        )
        record = await store.reserve_or_replay(key="k1", body=body)
        assert record is not None
        assert record.run_id == "run-001"

    @pytest.mark.asyncio
    async def test_conflict_on_body_mismatch(self) -> None:
        store = IdempotencyStore()
        await store.store_response(
            key="k1",
            body={"task": "A"},
            run_id="run-A",
            status="pending",
            response_snapshot={"task_id": "run-A"},
        )
        with pytest.raises(IdempotencyConflict):
            await store.reserve_or_replay(key="k1", body={"task": "B"})

    @pytest.mark.asyncio
    async def test_ttl_expiry_treats_record_as_miss(self) -> None:
        store = IdempotencyStore(ttl_seconds=1)
        body = {"task": "x"}
        await store.store_response(
            key="k1",
            body=body,
            run_id="run-001",
            status="pending",
            response_snapshot={"task_id": "run-001"},
        )
        # Force the stored_at into the past so the entry is well past its TTL
        record = await store.get("k1")
        assert record is not None
        async with store._memory_lock:  # type: ignore[attr-defined]
            store._memory["k1"].stored_at = time.time() - 10  # type: ignore[attr-defined]
        assert await store.get("k1") is None
        assert await store.reserve_or_replay(key="k1", body=body) is None

    @pytest.mark.asyncio
    async def test_empty_key_is_treated_as_no_key(self) -> None:
        store = IdempotencyStore()
        await store.put_response_for_testing_only()  # no-op, just to exercise empty path
        assert await store.get("") is None
        assert await store.reserve_or_replay(key="", body={}) is None


# Patch in a tiny helper used in the test above
async def _put_response_noop(self: IdempotencyStore) -> None:
    return None


IdempotencyStore.put_response_for_testing_only = _put_response_noop  # type: ignore[attr-defined]


# ─── End-to-end API tests ────────────────────────────────────────────────────


class TestSubmitTaskIdempotency:
    def test_no_idempotency_key_returns_fresh_task_id_each_call(self, client: TestClient) -> None:
        a = client.post("/api/tasks", json={"task": "T"}).json()
        b = client.post("/api/tasks", json={"task": "T"}).json()
        assert a["task_id"] != b["task_id"]

    def test_same_key_same_body_returns_original_run_id(self, client: TestClient) -> None:
        body = {"task": "Build a calculator"}
        first = client.post("/api/tasks", json=body, headers={HEADER_NAME: "K-1"})
        assert first.status_code == 202
        first_id = first.json()["task_id"]

        second = client.post("/api/tasks", json=body, headers={HEADER_NAME: "K-1"})
        # Replay must return 200 + the original task_id
        assert second.status_code == 200
        assert second.json()["task_id"] == first_id
        assert second.headers.get("Idempotency-Replayed") == "true"

    def test_body_mismatch_returns_422_with_structured_error(self, client: TestClient) -> None:
        first = client.post(
            "/api/tasks",
            json={"task": "Build a calculator"},
            headers={HEADER_NAME: "K-2"},
        )
        assert first.status_code == 202

        second = client.post(
            "/api/tasks",
            json={"task": "Build a DIFFERENT thing"},
            headers={HEADER_NAME: "K-2"},
        )
        assert second.status_code == 422
        detail = second.json()["detail"]
        assert detail["error"] == "idempotency_key_conflict"
        assert detail["idempotency_key"] == "K-2"

    def test_key_reuse_after_ttl_acts_like_fresh_request(self, client: TestClient) -> None:
        first = client.post(
            "/api/tasks",
            json={"task": "X"},
            headers={HEADER_NAME: "K-3"},
        )
        assert first.status_code == 202

        # Simulate TTL expiry by pushing stored_at backwards on the singleton
        store = get_idempotency_store()
        record = store._memory["K-3"]  # type: ignore[attr-defined]
        record.stored_at = time.time() - store.ttl_seconds - 10

        second = client.post(
            "/api/tasks",
            json={"task": "X"},
            headers={HEADER_NAME: "K-3"},
        )
        # Expired record acts like a miss → fresh 202 + new id
        assert second.status_code == 202
        assert second.json()["task_id"] != first.json()["task_id"]

    def test_redis_down_uses_in_memory_fallback(self, client: TestClient) -> None:
        """When Redis is unavailable, the in-memory store still enforces idempotency."""
        body = {"task": "T"}
        first = client.post("/api/tasks", json=body, headers={HEADER_NAME: "K-4"})
        second = client.post("/api/tasks", json=body, headers={HEADER_NAME: "K-4"})
        assert first.status_code == 202
        assert second.status_code == 200
        assert first.json()["task_id"] == second.json()["task_id"]

    def test_key_order_in_body_does_not_break_idempotency(self, client: TestClient) -> None:
        a = client.post(
            "/api/tasks",
            json={"task": "X", "context": "Y"},
            headers={HEADER_NAME: "K-5"},
        )
        # Same logical body, different JSON key ordering
        b = client.post(
            "/api/tasks",
            json={"context": "Y", "task": "X"},
            headers={HEADER_NAME: "K-5"},
        )
        assert a.status_code == 202
        assert b.status_code == 200
        assert a.json()["task_id"] == b.json()["task_id"]
