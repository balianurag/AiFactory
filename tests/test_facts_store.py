"""
Tests for core/facts_store.py — filesystem REMEMBER: facts persistence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aifactory.core.facts_store import FactsStore


class TestFactsStore:
    @pytest.fixture
    def store(self, tmp_path: Path) -> FactsStore:
        return FactsStore(store_dir=tmp_path)

    def test_empty_store(self, store: FactsStore) -> None:
        assert store.size() == 0
        assert store.load_all() == []

    def test_store_and_load(self, store: FactsStore) -> None:
        store.store("worker_1", "Bhuvan WMS base URL: https://bhuvan.nrsc.gov.in/wms")
        facts = store.load_all()
        assert len(facts) == 1
        assert "Bhuvan WMS" in facts[0]

    def test_deduplication(self, store: FactsStore) -> None:
        fact = "PostGIS table: geofences(id, name, polygon)"
        store.store("worker_1", fact)
        store.store("worker_2", fact)  # duplicate
        assert store.size() == 1

    def test_multiple_facts(self, store: FactsStore) -> None:
        store.store("w1", "fact one")
        store.store("w2", "fact two")
        store.store("w3", "fact three")
        assert store.size() == 3
        facts = store.load_all()
        assert "fact one" in facts
        assert "fact two" in facts
        assert "fact three" in facts

    def test_empty_fact_ignored(self, store: FactsStore) -> None:
        store.store("w1", "")
        store.store("w1", "   ")
        assert store.size() == 0

    def test_clear(self, store: FactsStore) -> None:
        store.store("w1", "fact")
        store.clear()
        assert store.size() == 0
        assert store.load_all() == []

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        s1 = FactsStore(store_dir=tmp_path)
        s1.store("agent_x", "MapmyIndia API key required for route planning")

        s2 = FactsStore(store_dir=tmp_path)
        facts = s2.load_all()
        assert any("MapmyIndia" in f for f in facts)

    def test_run_id_stored(self, tmp_path: Path) -> None:
        store = FactsStore(store_dir=tmp_path)
        store.store("w1", "some fact", run_id="run_abc123")
        raw = (tmp_path / "facts.json").read_text()
        assert "run_abc123" in raw

    def test_tolerates_corrupt_file(self, tmp_path: Path) -> None:
        (tmp_path / "facts.json").write_text("not valid json", encoding="utf-8")
        store = FactsStore(store_dir=tmp_path)
        assert store.load_all() == []
        assert store.size() == 0


class TestMemoryBridgeFactsStore:
    """Integration: MemoryBridge wires FactsStore correctly."""

    def test_store_remember_facts_writes_to_filesystem(self, tmp_path: Path) -> None:
        from aifactory.core.facts_store import FactsStore
        from aifactory.core.memory_bridge import MemoryBridge

        bridge = MemoryBridge(router=None, run_id="test_run")
        bridge._facts_store = FactsStore(store_dir=tmp_path)

        bridge.store_remember_facts("worker_1", ["Bhuvan WMS URL found", "PostGIS schema ready"])

        facts = bridge._facts_store.load_all()
        assert "Bhuvan WMS URL found" in facts
        assert "PostGIS schema ready" in facts

    def test_store_remember_facts_no_router_still_writes_fs(self, tmp_path: Path) -> None:
        """Facts must persist even with no Qdrant router."""
        from aifactory.core.facts_store import FactsStore
        from aifactory.core.memory_bridge import MemoryBridge

        bridge = MemoryBridge(router=None, run_id="r1")
        bridge._facts_store = FactsStore(store_dir=tmp_path)

        bridge.store_remember_facts("w", ["cross-run fact"])
        assert bridge._facts_store.size() == 1

    def test_retrieve_global_context_falls_back_to_filesystem(self, tmp_path: Path) -> None:
        """When Qdrant returns nothing, filesystem facts are returned."""
        from aifactory.core.facts_store import FactsStore
        from aifactory.core.memory_bridge import MemoryBridge

        bridge = MemoryBridge(router=None, run_id="r1")
        fs = FactsStore(store_dir=tmp_path)
        fs.store("worker_1", "INSAT satellite provides 4km resolution imagery")
        bridge._facts_store = fs

        ctx = bridge.retrieve_global_context("satellite imagery")
        assert "INSAT" in ctx

    def test_retrieve_global_context_empty_when_no_facts(self, tmp_path: Path) -> None:
        from aifactory.core.facts_store import FactsStore
        from aifactory.core.memory_bridge import MemoryBridge

        bridge = MemoryBridge(router=None, run_id="r1")
        bridge._facts_store = FactsStore(store_dir=tmp_path)

        ctx = bridge.retrieve_global_context("anything")
        assert ctx == ""

    def test_filesystem_caps_at_top_k_times_two(self, tmp_path: Path) -> None:
        """Only top_k * 2 facts are returned to avoid prompt bloat."""
        from aifactory.core.facts_store import FactsStore
        from aifactory.core.memory_bridge import MemoryBridge

        bridge = MemoryBridge(router=None, run_id="r1")
        fs = FactsStore(store_dir=tmp_path)
        for i in range(10):
            fs.store("w", f"fact {i}")
        bridge._facts_store = fs

        ctx = bridge.retrieve_global_context("query", top_k=2)
        # top_k=2 → max 4 facts returned
        fact_lines = [line for line in ctx.splitlines() if line.startswith("-")]
        assert len(fact_lines) <= 4
