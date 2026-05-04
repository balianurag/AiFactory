"""
Tests for core/run_store.py — RunRecord dataclass, RunStore backends, build_run_record().

Covers:
  - RunRecord creation, to_dict, from_dict round-trip
  - RunStore.save() + list_recent() + get() against in-memory Qdrant
  - RunStore JSONL fallback (when Qdrant _qdrant attribute set to None)
  - build_run_record() task-type inference and field mapping
  - get_run_store() lazy singleton
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

from aifactory.core.run_store import (
    RunRecord,
    RunStore,
    build_run_record,
    get_run_store,
)

# =============================================================================
# RunRecord dataclass
# =============================================================================


class TestRunRecord:
    def _make_record(self, **overrides) -> RunRecord:
        defaults = dict(
            run_id="test-run-001",
            timestamp="2026-04-16T12:00:00+00:00",
            query="Build a REST API",
            mode="balanced",
            elapsed_seconds=12.5,
            gate="",
            model_selections={"master": "gpt-4o"},
            task_count=3,
            quality_flagged_count=0,
            composite_quality=8.5,
            task_types=["coding"],
            final_response_preview="Here is the REST API...",
        )
        defaults.update(overrides)
        return RunRecord(**defaults)

    def test_basic_creation(self) -> None:
        rec = self._make_record()
        assert rec.run_id == "test-run-001"
        assert rec.mode == "balanced"
        assert rec.task_count == 3

    def test_to_dict_contains_all_fields(self) -> None:
        rec = self._make_record()
        d = rec.to_dict()
        assert d["run_id"] == "test-run-001"
        assert d["mode"] == "balanced"
        assert d["composite_quality"] == 8.5
        assert isinstance(d["quality_per_task"], list)
        assert isinstance(d["plan"], list)
        assert isinstance(d["metadata"], dict)

    def test_from_dict_round_trip(self) -> None:
        rec = self._make_record(
            quality_per_task=[{"task_id": "t1", "quality_flagged": False, "model": "gpt-4o"}],
            plan=[{"description": "Write the API"}],
            metadata={"extra": "info"},
        )
        d = rec.to_dict()
        restored = RunRecord.from_dict(d)
        assert restored.run_id == rec.run_id
        assert restored.composite_quality == rec.composite_quality
        assert restored.plan == rec.plan

    def test_from_dict_ignores_unknown_keys(self) -> None:
        d = self._make_record().to_dict()
        d["unknown_future_key"] = "should be ignored"
        # Should not raise
        rec = RunRecord.from_dict(d)
        assert rec.run_id == "test-run-001"

    def test_defaults_for_optional_fields(self) -> None:
        rec = self._make_record()
        assert rec.quality_per_task == []
        assert rec.plan == []
        assert rec.metadata == {}


# =============================================================================
# RunStore — in-memory Qdrant backend
# =============================================================================


class TestRunStore:
    def _make_store(self, tmp_path=None) -> RunStore:
        """Create a RunStore backed by in-memory Qdrant (no PostgreSQL)."""
        import tempfile

        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        import aifactory.core.run_store as rs_mod

        client = QdrantClient(":memory:")
        client.create_collection(
            "aifactory_run_store",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

        store = RunStore.__new__(RunStore)
        store._pg = False
        store._qdrant = client
        store._embed_fn = lambda text: [0.0] * 1536
        store._dim = 1536

        # Redirect JSONL to a temp location so tests don't write to ~
        _td = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
        rs_mod._FALLBACK_FILE = _td / "run_store.jsonl"
        return store

    def _make_record(self, run_id: str = "test-run-001", query: str = "test query") -> RunRecord:
        return RunRecord(
            run_id=run_id,
            timestamp=datetime.now(UTC).isoformat(),
            query=query,
            mode="balanced",
            elapsed_seconds=5.0,
            gate="",
            model_selections={},
            task_count=1,
            quality_flagged_count=0,
            composite_quality=8.0,
            task_types=["general"],
            final_response_preview="Answer here",
        )

    def test_save_and_list_recent(self) -> None:
        store = self._make_store()
        record = self._make_record()
        store.save(record)
        records = store.list_recent(n=10)
        assert len(records) == 1
        assert records[0].run_id == "test-run-001"

    def test_list_recent_returns_newest_first(self) -> None:
        store = self._make_store()
        for i in range(3):
            rec = self._make_record(
                run_id=f"run-{i}",
                query=f"query {i}",
            )
            # Give different timestamps by mutating
            rec.timestamp = f"2026-04-16T1{i}:00:00+00:00"
            store.save(rec)
        records = store.list_recent(n=10)
        # Should be sorted newest first
        timestamps = [r.timestamp for r in records]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_get_by_run_id(self) -> None:
        store = self._make_store()
        record = self._make_record(run_id="specific-run-xyz")
        store.save(record)
        found = store.get("specific-run-xyz")
        assert found is not None
        assert found.run_id == "specific-run-xyz"

    def test_get_returns_none_for_missing(self) -> None:
        store = self._make_store()
        found = store.get("nonexistent-run-id")
        assert found is None

    def test_search_similar_does_not_raise(self) -> None:
        """search_similar uses stub embeddings (all zeros) so similarity
        scores may not produce hits; verify it completes without raising."""
        store = self._make_store()
        record = self._make_record(query="Build a FastAPI REST API endpoint")
        store.save(record)
        results = store.search_similar("FastAPI REST", top_k=3)
        assert isinstance(results, list)  # may be empty with stub embeddings

    def test_list_recent_respects_limit(self) -> None:
        store = self._make_store()
        for i in range(5):
            store.save(self._make_record(run_id=f"run-{i}", query=f"query {i}"))
        records = store.list_recent(n=3)
        assert len(records) <= 3

    def test_save_fails_gracefully(self) -> None:
        """save() should not raise even when Qdrant fails."""
        store = self._make_store()
        store._qdrant = None  # Simulate Qdrant unavailable
        # JSONL fallback — temp dir to avoid polluting ~
        with tempfile.TemporaryDirectory() as td:
            import aifactory.core.run_store as rs

            orig = rs._FALLBACK_FILE
            rs._FALLBACK_FILE = Path(td) / "run_store.jsonl"
            try:
                store.save(self._make_record())  # should not raise
            finally:
                rs._FALLBACK_FILE = orig


# =============================================================================
# RunStore — JSONL fallback
# =============================================================================


class TestRunStoreJSONL:
    def test_save_and_list_jsonl(self) -> None:
        import aifactory.core.run_store as rs

        with tempfile.TemporaryDirectory() as td:
            fallback = Path(td) / "run_store.jsonl"
            orig = rs._FALLBACK_FILE
            rs._FALLBACK_FILE = fallback
            try:
                store = RunStore.__new__(RunStore)
                store._pg = False
                store._qdrant = None
                store._embed_fn = None
                store._dim = 1536

                rec = RunRecord(
                    run_id="jsonl-run-001",
                    timestamp="2026-04-16T12:00:00+00:00",
                    query="Test JSONL",
                    mode="balanced",
                    elapsed_seconds=1.0,
                    gate="",
                    model_selections={},
                    task_count=1,
                    quality_flagged_count=0,
                    composite_quality=9.0,
                    task_types=["general"],
                    final_response_preview="ok",
                )
                store.save(rec)
                assert fallback.exists()

                records = store.list_recent(n=10)
                assert len(records) == 1
                assert records[0].run_id == "jsonl-run-001"
            finally:
                rs._FALLBACK_FILE = orig

    def test_list_jsonl_handles_corrupt_lines(self) -> None:
        import aifactory.core.run_store as rs

        with tempfile.TemporaryDirectory() as td:
            fallback = Path(td) / "run_store.jsonl"
            orig = rs._FALLBACK_FILE
            rs._FALLBACK_FILE = fallback
            try:
                fallback.write_text(
                    '{"not": "a run record"}\n'
                    "{{invalid json}}\n"
                    + json.dumps(
                        RunRecord(
                            run_id="valid-run",
                            timestamp="2026-04-16T12:00:00+00:00",
                            query="valid",
                            mode="balanced",
                            elapsed_seconds=1.0,
                            gate="",
                            model_selections={},
                            task_count=1,
                            quality_flagged_count=0,
                            composite_quality=8.0,
                            task_types=[],
                            final_response_preview="",
                        ).to_dict()
                    )
                    + "\n"
                )
                store = RunStore.__new__(RunStore)
                store._pg = False
                store._qdrant = None
                store._embed_fn = None
                store._dim = 1536
                records = store.list_recent(n=10)
                assert any(r.run_id == "valid-run" for r in records)
            finally:
                rs._FALLBACK_FILE = orig


# =============================================================================
# build_run_record
# =============================================================================


class TestBuildRunRecord:
    def _make_registry(self) -> MagicMock:
        agent = MagicMock()
        agent.name = "master"
        agent.model = "gpt-4o"
        registry = MagicMock()
        registry.all.return_value = [agent]
        return registry

    def test_basic_fields(self) -> None:
        state = {
            "final_response": "The answer is 42",
            "tasks": [],
            "task_results": [],
            "judge_metrics": {"composite_quality": 9.0},
        }
        rec = build_run_record(
            run_id="run-001",
            query="What is the meaning of life?",
            mode="balanced",
            elapsed=5.2,
            gate="",
            state=state,
            registry=self._make_registry(),
        )
        assert rec.run_id == "run-001"
        assert rec.query == "What is the meaning of life?"
        assert rec.mode == "balanced"
        assert rec.elapsed_seconds == 5.2
        assert rec.composite_quality == 9.0
        assert rec.final_response_preview.startswith("The answer is 42")

    def test_model_selections_from_registry(self) -> None:
        state = {"final_response": "ok", "tasks": [], "task_results": [], "judge_metrics": {}}
        rec = build_run_record(
            run_id="run-002",
            query="test",
            mode="best",
            elapsed=1.0,
            gate="",
            state=state,
            registry=self._make_registry(),
        )
        assert "master" in rec.model_selections
        assert rec.model_selections["master"] == "gpt-4o"

    def test_task_type_coding_inferred(self) -> None:
        state = {
            "final_response": "code here",
            "tasks": [{"description": "implement a sort function"}],
            "task_results": [],
            "judge_metrics": {},
        }
        rec = build_run_record(
            run_id="run-003",
            query="code task",
            mode="balanced",
            elapsed=2.0,
            gate="",
            state=state,
            registry=self._make_registry(),
        )
        assert "coding" in rec.task_types

    def test_task_type_research_inferred(self) -> None:
        state = {
            "final_response": "research results",
            "tasks": [{"description": "research neural networks"}],
            "task_results": [],
            "judge_metrics": {},
        }
        rec = build_run_record(
            run_id="run-004",
            query="research task",
            mode="balanced",
            elapsed=2.0,
            gate="",
            state=state,
            registry=self._make_registry(),
        )
        assert "research" in rec.task_types

    def test_task_type_writing_inferred(self) -> None:
        state = {
            "final_response": "article",
            "tasks": [{"description": "write a blog post about AI"}],
            "task_results": [],
            "judge_metrics": {},
        }
        rec = build_run_record(
            run_id="run-005",
            query="writing task",
            mode="balanced",
            elapsed=2.0,
            gate="",
            state=state,
            registry=self._make_registry(),
        )
        assert "writing" in rec.task_types

    def test_task_type_general_fallback(self) -> None:
        state = {
            "final_response": "some answer",
            "tasks": [{"description": "plan the project kickoff meeting"}],
            "task_results": [],
            "judge_metrics": {},
        }
        rec = build_run_record(
            run_id="run-006",
            query="generic task",
            mode="balanced",
            elapsed=2.0,
            gate="",
            state=state,
            registry=self._make_registry(),
        )
        assert "general" in rec.task_types

    def test_quality_flagged_count(self) -> None:
        state = {
            "final_response": "ok",
            "tasks": [],
            "task_results": [
                {"task_id": "t1", "quality_flagged": True, "model": "gpt-4o"},
                {"task_id": "t2", "quality_flagged": False, "model": "gpt-4o"},
                {"task_id": "t3", "quality_flagged": True, "model": "gpt-4o"},
            ],
            "judge_metrics": {},
        }
        rec = build_run_record(
            run_id="run-007",
            query="test",
            mode="balanced",
            elapsed=1.0,
            gate="",
            state=state,
            registry=self._make_registry(),
        )
        assert rec.quality_flagged_count == 2

    def test_query_truncated_to_max(self) -> None:
        long_query = "x" * 3000
        state = {"final_response": "ok", "tasks": [], "task_results": [], "judge_metrics": {}}
        rec = build_run_record(
            run_id="run-008",
            query=long_query,
            mode="balanced",
            elapsed=1.0,
            gate="",
            state=state,
            registry=self._make_registry(),
        )
        assert len(rec.query) <= 2000

    def test_registry_error_handled_gracefully(self) -> None:
        bad_registry = MagicMock()
        bad_registry.all.side_effect = Exception("registry exploded")
        state = {"final_response": "ok", "tasks": [], "task_results": [], "judge_metrics": {}}
        rec = build_run_record(
            run_id="run-009",
            query="test",
            mode="balanced",
            elapsed=1.0,
            gate="",
            state=state,
            registry=bad_registry,
        )
        assert rec.model_selections == {}


# =============================================================================
# Singleton
# =============================================================================


class TestGetRunStore:
    def test_returns_run_store_instance(self) -> None:
        import aifactory.core.run_store as rs

        orig = rs._store
        rs._store = None
        try:
            store = get_run_store()
            assert isinstance(store, RunStore)
        finally:
            rs._store = orig

    def test_returns_same_instance_on_repeat_calls(self) -> None:
        import aifactory.core.run_store as rs

        orig = rs._store
        rs._store = None
        try:
            a = get_run_store()
            b = get_run_store()
            assert a is b
        finally:
            rs._store = orig
