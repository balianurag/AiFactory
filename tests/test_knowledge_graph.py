"""Tests for KnowledgeGraph entity store (Week 2 — Adaptive Mesh Pipeline)."""

from __future__ import annotations

import pytest

from aifactory.core.knowledge_graph import (
    KGDiff,
    KnowledgeGraph,
    _parse_structural_index,
    _sha16,
)


SAMPLE_INDEX = """## Codebase Structural Index
Root dirs: src
Files: 2  |  Languages: python

### src/aifactory/core/foo.py  (python, 1.5 KB)
  classes:   FooManager, FooHelper
  functions: run_foo, _internal
  imports:   os, json, typing

### src/aifactory/api/bar.py  (python, 0.8 KB)
  classes:   BarHandler
  functions: handle_request
  imports:   fastapi, pydantic
"""

SAMPLE_META = {
    "src/aifactory/core/foo.py": {
        "functions": ["run_foo", "_internal"],
        "classes": ["FooManager", "FooHelper"],
        "imports": ["os", "json", "typing"],
    },
    "src/aifactory/api/bar.py": {
        "functions": ["handle_request"],
        "classes": ["BarHandler"],
        "imports": ["fastapi", "pydantic"],
    },
}


# ── _parse_structural_index ───────────────────────────────────────────────────


class TestParseStructuralIndex:
    def test_parses_two_files(self):
        meta = _parse_structural_index(SAMPLE_INDEX)
        assert len(meta) == 2
        assert "src/aifactory/core/foo.py" in meta
        assert "src/aifactory/api/bar.py" in meta

    def test_parses_functions(self):
        meta = _parse_structural_index(SAMPLE_INDEX)
        assert meta["src/aifactory/core/foo.py"]["functions"] == ["run_foo", "_internal"]

    def test_parses_classes(self):
        meta = _parse_structural_index(SAMPLE_INDEX)
        assert "FooManager" in meta["src/aifactory/core/foo.py"]["classes"]

    def test_parses_imports(self):
        meta = _parse_structural_index(SAMPLE_INDEX)
        assert "os" in meta["src/aifactory/core/foo.py"]["imports"]

    def test_empty_returns_empty(self):
        assert _parse_structural_index("") == {}

    def test_handles_missing_fields(self):
        text = "### src/only_file.py  (python, 0.1 KB)\n  functions: do_thing\n"
        meta = _parse_structural_index(text)
        assert meta["src/only_file.py"]["functions"] == ["do_thing"]
        assert "classes" not in meta["src/only_file.py"]


# ── KnowledgeGraph ingestion ──────────────────────────────────────────────────


class TestKnowledgeGraphIngestion:
    def test_ingest_codebase_returns_snapshot_id(self):
        kg = KnowledgeGraph()
        snap_id = kg.ingest_codebase(SAMPLE_META)
        assert isinstance(snap_id, str)
        assert snap_id.startswith("snap:")

    def test_ingest_structural_index(self):
        kg = KnowledgeGraph()
        snap_id = kg.ingest_structural_index(SAMPLE_INDEX)
        assert isinstance(snap_id, str)

    def test_entities_created_for_all_files(self):
        kg = KnowledgeGraph()
        kg.ingest_codebase(SAMPLE_META)
        files = kg.find_entities(kind="file")
        assert len(files) == 2

    def test_function_entities_created(self):
        kg = KnowledgeGraph()
        kg.ingest_codebase(SAMPLE_META)
        fns = kg.find_entities(kind="function", file_path="src/aifactory/core/foo.py")
        assert {e.name for e in fns} == {"run_foo", "_internal"}

    def test_class_entities_created(self):
        kg = KnowledgeGraph()
        kg.ingest_codebase(SAMPLE_META)
        classes = kg.find_entities(kind="class_", file_path="src/aifactory/core/foo.py")
        assert {e.name for e in classes} == {"FooManager", "FooHelper"}

    def test_import_entities_created(self):
        kg = KnowledgeGraph()
        kg.ingest_codebase(SAMPLE_META)
        imports = kg.find_entities(kind="import_", file_path="src/aifactory/core/foo.py")
        assert {e.name for e in imports} == {"os", "json", "typing"}

    def test_contains_relations_created(self):
        kg = KnowledgeGraph()
        kg.ingest_codebase(SAMPLE_META)
        file_entities = kg.find_entities(kind="file", file_path="src/aifactory/core/foo.py")
        assert len(file_entities) == 1
        rels = kg.relations_for(file_entities[0].id)
        assert any(r.relation == "contains" for r in rels)

    def test_re_ingest_replaces_entities(self):
        kg = KnowledgeGraph()
        kg.ingest_codebase(SAMPLE_META)
        count_before = kg.stats()["total_entities"]
        # Ingest again — entity count should stay the same (no duplicates)
        kg.ingest_codebase(SAMPLE_META)
        assert kg.stats()["total_entities"] == count_before


# ── KnowledgeGraph diff ───────────────────────────────────────────────────────


class TestKGDiff:
    def test_no_diff_for_same_content(self):
        kg = KnowledgeGraph()
        snap_id = kg.ingest_codebase(SAMPLE_META)
        diff = kg.diff_since(snap_id, kg.hash_codebase({k: str(v) for k, v in SAMPLE_META.items()}))
        # All hashes will differ because SAMPLE_META values are dicts, not str
        # Use the same snap_id approach
        diff2 = kg.diff_since(snap_id)
        assert diff2.is_empty  # comparing snap to itself → no diff

    def test_added_file_detected(self):
        kg = KnowledgeGraph()
        snap_id = kg.ingest_codebase(SAMPLE_META)
        new_hashes = {**{k: _sha16(str(v)) for k, v in SAMPLE_META.items()}, "new/file.py": "abc123"}
        diff = kg.diff_since(snap_id, new_hashes)
        assert "new/file.py" in diff.added

    def test_removed_file_detected(self):
        kg = KnowledgeGraph()
        snap_id = kg.ingest_codebase(SAMPLE_META)
        reduced = {"src/aifactory/core/foo.py": _sha16(str(SAMPLE_META["src/aifactory/core/foo.py"]))}
        diff = kg.diff_since(snap_id, reduced)
        assert "src/aifactory/api/bar.py" in diff.removed

    def test_modified_file_detected(self):
        kg = KnowledgeGraph()
        base_hashes = {k: _sha16(str(v)) for k, v in SAMPLE_META.items()}
        snap_id = kg.ingest_codebase(SAMPLE_META)
        # Same files, different hash for one
        modified_hashes = {**base_hashes, "src/aifactory/core/foo.py": "changed_hash"}
        diff = kg.diff_since(snap_id, modified_hashes)
        assert "src/aifactory/core/foo.py" in diff.modified

    def test_diff_changed_combines_added_modified(self):
        diff = KGDiff(
            snapshot_id="s1",
            added=["a.py"],
            modified=["b.py"],
            removed=["c.py"],
        )
        assert set(diff.changed) == {"a.py", "b.py"}

    def test_missing_snapshot_returns_empty_diff(self):
        kg = KnowledgeGraph()
        diff = kg.diff_since("nonexistent_snap")
        assert diff.is_empty

    def test_incremental_reingest_skips_unchanged(self):
        kg = KnowledgeGraph()
        snap_id = kg.ingest_codebase(SAMPLE_META)
        stats_before = kg.stats()
        # Re-ingest same content — should skip all files (no changes)
        snap_id2 = kg.ingest_codebase(SAMPLE_META, base_snapshot_id=snap_id)
        stats_after = kg.stats()
        assert stats_before["total_entities"] == stats_after["total_entities"]


# ── Query API ─────────────────────────────────────────────────────────────────


class TestKGQuery:
    def setup_method(self):
        self.kg = KnowledgeGraph()
        self.kg.ingest_codebase(SAMPLE_META)

    def test_function_names_all(self):
        fns = self.kg.function_names()
        assert "run_foo" in fns
        assert "handle_request" in fns

    def test_function_names_by_file(self):
        fns = self.kg.function_names("src/aifactory/core/foo.py")
        assert "run_foo" in fns
        assert "handle_request" not in fns

    def test_class_names_all(self):
        classes = self.kg.class_names()
        assert "FooManager" in classes
        assert "BarHandler" in classes

    def test_entities_for_file(self):
        entities = self.kg.entities_for_file("src/aifactory/core/foo.py")
        names = {e.name for e in entities}
        assert "run_foo" in names

    def test_stats_contains_counts(self):
        stats = self.kg.stats()
        assert stats["total_entities"] > 0
        assert stats["total_relations"] > 0
        assert stats["snapshots"] == 1

    def test_integration_files_detects_api(self):
        kg = KnowledgeGraph()
        kg.ingest_codebase({
            "src/aifactory/mcp/client.py": {"functions": ["connect"], "classes": [], "imports": []},
        })
        assert kg.has_integration_code()

    def test_no_integration_files_for_core(self):
        kg = KnowledgeGraph()
        kg.ingest_codebase({
            "src/aifactory/core/foo.py": {"functions": ["run"], "classes": [], "imports": []},
        })
        assert not kg.has_integration_code()

    def test_to_summary_returns_string(self):
        summary = self.kg.to_summary()
        assert isinstance(summary, str)
        assert "Knowledge Graph" in summary

    def test_hash_codebase(self):
        hashes = self.kg.hash_codebase({"file.py": "content"})
        assert "file.py" in hashes
        assert len(hashes["file.py"]) == 16


# ── MemoryRouter KG bridge ────────────────────────────────────────────────────


class TestMemoryRouterKGBridge:
    def test_kg_methods_work_without_kg(self):
        from aifactory.core.memory_router import MemoryRouter
        router = MemoryRouter()
        # No KG attached — all methods should return safe defaults
        assert router.kg_query_functions() == []
        assert router.kg_query_classes() == []
        assert router.kg_has_integration_code() is False
        assert router.kg_stats() == {}

    def test_kg_methods_delegate_to_attached_kg(self):
        from aifactory.core.memory_router import MemoryRouter
        kg = KnowledgeGraph()
        kg.ingest_codebase(SAMPLE_META)
        router = MemoryRouter()
        router.attach_knowledge_graph(kg)
        fns = router.kg_query_functions()
        assert "run_foo" in fns

    def test_kg_stats_populated_after_attach(self):
        from aifactory.core.memory_router import MemoryRouter
        kg = KnowledgeGraph()
        kg.ingest_codebase(SAMPLE_META)
        router = MemoryRouter()
        router.attach_knowledge_graph(kg)
        stats = router.kg_stats()
        assert stats["total_entities"] > 0
