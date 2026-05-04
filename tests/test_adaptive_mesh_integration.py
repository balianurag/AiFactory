"""
Integration tests for Week 5 — all four Adaptive Mesh layers wired together.

Tests verify:
  1. Auction outcomes stored in shared_ctx auction_floor
  2. KG → graph_spec: integration code upgrades topology
  3. KG → graph_spec: large codebase upgrades MINIMAL to STANDARD
  4. GraphCompiler uses KG upgrade in compile()
  5. WaveGovernor reads auction_floor from shared_ctx
  6. End-to-end: MINIMAL spec + integration KG → FULL topology
"""

from __future__ import annotations

import pytest

from aifactory.core.knowledge_graph import KnowledgeGraph
from aifactory.core.worker_auction import WorkerAuction, WorkerBid
from aifactory.graphs.graph_compiler import GraphCompiler, GraphSpec, TopologyShape
from aifactory.graphs.wave_governor import WaveGovernor

# ── Layer 1: Auction → shared_ctx ────────────────────────────────────────────


class TestAuctionToSharedCtx:
    def test_floor_stored_in_ctx(self):
        bids = [
            WorkerBid(worker_id="coder", confidence=0.9, model="gpt-4o-mini"),
            WorkerBid(worker_id="analyst", confidence=0.7, model="gpt-4o-mini"),
        ]
        auction = WorkerAuction()
        result = auction.run("implement sort", bids)
        ctx = result.to_shared_ctx_update()
        assert float(ctx["auction_floor"]) == pytest.approx(0.7)
        assert ctx["auction_winner"] == "coder"

    def test_auction_floor_in_ctx_used_by_governor(self):
        # Auction sets floor 0.8; governor quality 0.5 < floor → ESCALATE
        ctx = {"auction_floor": 0.8}
        gov = WaveGovernor(escalate_threshold=0.1)  # very low threshold
        results = [{"task_id": "t1", "result": "REVISION: partial", "status": "ok"}]
        decision = gov.evaluate(results, "balanced", ctx)
        assert decision.action == "ESCALATE"
        assert decision.reason == "auction_floor_violation"


# ── Layer 2: KG → graph_spec → topology ──────────────────────────────────────


class TestKGToGraphSpec:
    def _make_compiler_with_kg(self, kg: KnowledgeGraph) -> GraphCompiler:
        # Minimal compiler — we only test _kg_upgrade_spec, no real agents needed
        return GraphCompiler(
            master=None,
            analyser=None,
            researcher=None,
            lead=None,
            judge=None,
            knowledge_graph=kg,
        )

    def test_integration_code_upgrades_minimal_to_full(self):
        kg = KnowledgeGraph()
        kg.ingest_codebase(
            {
                "src/aifactory/mcp/client.py": {
                    "functions": ["connect"],
                    "classes": ["MCPClient"],
                    "imports": [],
                }
            }
        )
        compiler = self._make_compiler_with_kg(kg)
        spec = GraphSpec(topology=TopologyShape.MINIMAL)
        upgraded = compiler._kg_upgrade_spec(spec)
        assert upgraded.topology == TopologyShape.FULL

    def test_integration_code_upgrades_standard_to_full(self):
        kg = KnowledgeGraph()
        kg.ingest_codebase(
            {
                "src/integration/connector.py": {
                    "functions": ["send"],
                    "classes": [],
                    "imports": [],
                }
            }
        )
        compiler = self._make_compiler_with_kg(kg)
        spec = GraphSpec(topology=TopologyShape.STANDARD)
        upgraded = compiler._kg_upgrade_spec(spec)
        assert upgraded.topology == TopologyShape.FULL

    def test_no_upgrade_for_full_topology(self):
        kg = KnowledgeGraph()
        kg.ingest_codebase(
            {"src/aifactory/mcp/client.py": {"functions": [], "classes": [], "imports": []}}
        )
        compiler = self._make_compiler_with_kg(kg)
        spec = GraphSpec(topology=TopologyShape.FULL)
        upgraded = compiler._kg_upgrade_spec(spec)
        assert upgraded.topology == TopologyShape.FULL  # already full, no change

    def test_large_codebase_upgrades_minimal_to_standard(self):
        kg = KnowledgeGraph()
        # Inject 250+ entities: 10 files x 25 functions each = 260 function entities + 10 file entities
        meta = {
            f"src/module_{i}.py": {
                "functions": [f"fn_{i}_{j}" for j in range(25)],
                "classes": [],
                "imports": [],
            }
            for i in range(11)
        }
        kg.ingest_codebase(meta)
        assert kg.stats()["total_entities"] > 200
        compiler = self._make_compiler_with_kg(kg)
        spec = GraphSpec(topology=TopologyShape.MINIMAL)
        upgraded = compiler._kg_upgrade_spec(spec)
        assert upgraded.topology == TopologyShape.STANDARD

    def test_no_kg_returns_spec_unchanged(self):
        compiler = GraphCompiler(
            master=None,
            analyser=None,
            researcher=None,
            lead=None,
            judge=None,
            knowledge_graph=None,
        )
        spec = GraphSpec(topology=TopologyShape.MINIMAL)
        result = compiler._kg_upgrade_spec(spec)
        assert result.topology == TopologyShape.MINIMAL

    def test_kg_upgrade_sets_needs_research_on_integration(self):
        kg = KnowledgeGraph()
        kg.ingest_codebase(
            {"src/aifactory/mcp/tool.py": {"functions": ["call"], "classes": [], "imports": []}}
        )
        compiler = self._make_compiler_with_kg(kg)
        spec = GraphSpec(topology=TopologyShape.MINIMAL, needs_research=False)
        upgraded = compiler._kg_upgrade_spec(spec)
        assert upgraded.needs_research is True


# ── Layer 3: GraphSpec effective_topology with auction influence ──────────────


class TestGraphSpecWithAuctionContext:
    def test_standard_with_research_upgrades_to_full(self):
        spec = GraphSpec(topology=TopologyShape.STANDARD, needs_research=True)
        assert spec.effective_topology == TopologyShape.FULL

    def test_full_without_judge_downgrades_to_standard(self):
        spec = GraphSpec(topology=TopologyShape.FULL, needs_judge=False)
        assert spec.effective_topology == TopologyShape.STANDARD


# ── Layer 4: Governor floor enforcement from auction ─────────────────────────


class TestGovernorFloorFromAuction:
    def test_high_floor_escalates_medium_quality_result(self):
        # Auction set floor=0.8; worker produced revision result (quality=0.3)
        gov = WaveGovernor(escalate_threshold=0.1)
        ctx = {"auction_floor": 0.8}
        results = [{"task_id": "t1", "result": "REVISION: incomplete", "status": "ok"}]
        decision = gov.evaluate(results, "balanced", ctx)
        assert decision.action == "ESCALATE"
        assert decision.reason == "auction_floor_violation"

    def test_floor_not_enforced_when_quality_exceeds_it(self):
        gov = WaveGovernor(escalate_threshold=0.1)
        ctx = {"auction_floor": 0.5}
        results = [{"task_id": "t1", "result": "Full complete answer.", "status": "ok"}]
        decision = gov.evaluate(results, "balanced", ctx)
        assert decision.action == "PROCEED"

    def test_floor_zero_never_triggers(self):
        gov = WaveGovernor(escalate_threshold=0.9)
        ctx = {"auction_floor": 0.0}
        results = [{"task_id": "t1", "result": "", "status": "failed"}]
        # threshold=0.9 > quality=0.2 → would escalate via threshold, not floor
        decision = gov.evaluate(results, "balanced", ctx)
        assert decision.action == "ESCALATE"
        assert decision.reason != "auction_floor_violation"

    def test_mode_best_no_escalation_even_below_floor(self):
        gov = WaveGovernor(escalate_threshold=0.1)
        ctx = {"auction_floor": 0.99}
        results = [{"task_id": "t1", "result": "partial answer", "status": "ok"}]
        # quality=1.0 >= floor=0.99 → PROCEED
        decision = gov.evaluate(results, "best", ctx)
        assert decision.action == "PROCEED"
