"""Tests for GraphSpec and GraphCompiler (Week 1 — Adaptive Mesh Pipeline)."""

from __future__ import annotations

from aifactory.graphs.graph_compiler import (
    CompileResult,
    GraphCompiler,
    GraphSpec,
    TopologyShape,
)

# ── GraphSpec unit tests ─────────────────────────────────────────────────────


class TestGraphSpec:
    def test_defaults(self):
        spec = GraphSpec()
        assert spec.topology == TopologyShape.FULL
        assert spec.needs_research is False
        assert spec.needs_judge is True
        assert spec.worker_strategy == "parallel"

    def test_from_dict_full(self):
        spec = GraphSpec.from_dict(
            {
                "topology": "FULL",
                "needs_research": True,
                "needs_judge": True,
                "worker_strategy": "parallel",
                "task_count": 3,
                "reasoning": "multi-step task",
            }
        )
        assert spec.topology == TopologyShape.FULL
        assert spec.needs_research is True
        assert spec.task_count == 3

    def test_from_dict_minimal(self):
        spec = GraphSpec.from_dict({"topology": "MINIMAL"})
        assert spec.topology == TopologyShape.MINIMAL
        assert spec.is_lite is True

    def test_from_dict_case_insensitive(self):
        spec = GraphSpec.from_dict({"topology": "standard"})
        assert spec.topology == TopologyShape.STANDARD

    def test_from_dict_unknown_topology_defaults_to_full(self):
        spec = GraphSpec.from_dict({"topology": "QUANTUM"})
        assert spec.topology == TopologyShape.FULL

    def test_from_json_valid(self):
        raw = '{"topology": "STANDARD", "needs_research": false, "task_count": 2}'
        spec = GraphSpec.from_json(raw)
        assert spec.topology == TopologyShape.STANDARD
        assert spec.task_count == 2

    def test_from_json_embedded_in_text(self):
        raw = 'Here is my analysis. {"topology": "MINIMAL"} Done.'
        spec = GraphSpec.from_json(raw)
        assert spec.topology == TopologyShape.MINIMAL

    def test_from_json_invalid_returns_defaults(self):
        spec = GraphSpec.from_json("not json at all")
        assert spec.topology == TopologyShape.FULL

    def test_to_dict_roundtrip(self):
        spec = GraphSpec(
            topology=TopologyShape.EXTENDED,
            needs_research=True,
            worker_strategy="sequential",
            task_count=5,
        )
        d = spec.to_dict()
        restored = GraphSpec.from_dict(d)
        assert restored.topology == TopologyShape.EXTENDED
        assert restored.needs_research is True
        assert restored.worker_strategy == "sequential"
        assert restored.task_count == 5

    def test_is_lite_only_for_minimal(self):
        assert GraphSpec(topology=TopologyShape.MINIMAL).is_lite is True
        assert GraphSpec(topology=TopologyShape.STANDARD).is_lite is False
        assert GraphSpec(topology=TopologyShape.FULL).is_lite is False
        assert GraphSpec(topology=TopologyShape.EXTENDED).is_lite is False

    def test_has_research_loop_full_with_research(self):
        spec = GraphSpec(topology=TopologyShape.FULL, needs_research=True)
        assert spec.has_research_loop is True

    def test_has_research_loop_false_for_standard(self):
        spec = GraphSpec(topology=TopologyShape.STANDARD, needs_research=True)
        assert spec.has_research_loop is False

    def test_effective_topology_upgrades_standard_with_research(self):
        spec = GraphSpec(topology=TopologyShape.STANDARD, needs_research=True)
        assert spec.effective_topology == TopologyShape.FULL

    def test_effective_topology_downgrades_full_without_judge(self):
        spec = GraphSpec(topology=TopologyShape.FULL, needs_judge=False)
        assert spec.effective_topology == TopologyShape.STANDARD

    def test_effective_topology_no_change_when_consistent(self):
        spec = GraphSpec(topology=TopologyShape.FULL, needs_research=True, needs_judge=True)
        assert spec.effective_topology == TopologyShape.FULL

    def test_effective_topology_minimal_unchanged(self):
        spec = GraphSpec(topology=TopologyShape.MINIMAL, needs_judge=False)
        assert spec.effective_topology == TopologyShape.MINIMAL


# ── GraphSpec.spec_from_shared_ctx ───────────────────────────────────────────


class TestSpecFromSharedCtx:
    def test_returns_none_when_missing(self):
        assert GraphCompiler.spec_from_shared_ctx({}) is None
        assert GraphCompiler.spec_from_shared_ctx({"other": "key"}) is None

    def test_parses_dict_value(self):
        ctx = {"graph_spec": {"topology": "STANDARD"}}
        spec = GraphCompiler.spec_from_shared_ctx(ctx)
        assert spec is not None
        assert spec.topology == TopologyShape.STANDARD

    def test_parses_json_string_value(self):
        ctx = {"graph_spec": '{"topology": "MINIMAL"}'}
        spec = GraphCompiler.spec_from_shared_ctx(ctx)
        assert spec is not None
        assert spec.topology == TopologyShape.MINIMAL

    def test_returns_none_for_unexpected_type(self):
        ctx = {"graph_spec": 42}
        spec = GraphCompiler.spec_from_shared_ctx(ctx)
        assert spec is None


# ── CompileResult structure ───────────────────────────────────────────────────


class TestCompileResult:
    def test_fields_exist(self):
        spec = GraphSpec()
        result = CompileResult(
            graph=object(),
            topology=TopologyShape.FULL,
            lite_mode=False,
            node_count=7,
            spec=spec,
        )
        assert result.topology == TopologyShape.FULL
        assert result.lite_mode is False
        assert result.node_count == 7
        assert result.spec is spec


# ── TopologyShape enum ────────────────────────────────────────────────────────


class TestTopologyShape:
    def test_all_values_present(self):
        shapes = {s.value for s in TopologyShape}
        assert shapes == {"MINIMAL", "STANDARD", "FULL", "EXTENDED"}

    def test_string_equality(self):
        assert TopologyShape.FULL == "FULL"
        assert TopologyShape.MINIMAL == "MINIMAL"

    def test_ordering_ascending(self):
        order = [
            TopologyShape.MINIMAL,
            TopologyShape.STANDARD,
            TopologyShape.FULL,
            TopologyShape.EXTENDED,
        ]
        assert order[0] != order[-1]
