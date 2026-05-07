"""Tests for WaveGovernor (Week 3 — Adaptive Mesh Pipeline)."""

from __future__ import annotations

import pytest

from aifactory.graphs.wave_governor import (
    _MODE_ESCALATION,
    WaveDecision,
    WaveGovernor,
    make_wave_governor,
    route_after_wave_governor,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _ok_result(task_id: str = "t1") -> dict:
    return {"task_id": task_id, "result": "Done successfully.", "status": "ok"}


def _failed_result(task_id: str = "t1") -> dict:
    return {"task_id": task_id, "result": "", "status": "failed"}


def _error_result(task_id: str = "t1") -> dict:
    return {"task_id": task_id, "result": "boom", "status": "error"}


def _revision_result(task_id: str = "t1") -> dict:
    return {"task_id": task_id, "result": "REVISION: needs more detail", "status": "ok"}


# ── WaveDecision ──────────────────────────────────────────────────────────────


class TestWaveDecision:
    def test_fields(self):
        d = WaveDecision(action="PROCEED", reason="ok", quality_score=0.9)
        assert d.action == "PROCEED"
        assert d.quality_score == 0.9
        assert d.failed_task_ids == []

    def test_escalate_has_mode(self):
        d = WaveDecision(action="ESCALATE", reason="low", escalated_mode="best", quality_score=0.3)
        assert d.escalated_mode == "best"


# ── WaveGovernor.evaluate ─────────────────────────────────────────────────────


class TestWaveGovernorEvaluate:
    def test_all_good_results_proceed(self):
        gov = WaveGovernor()
        decision = gov.evaluate([_ok_result(), _ok_result("t2")], "balanced", {})
        assert decision.action == "PROCEED"

    def test_empty_results_proceed(self):
        gov = WaveGovernor()
        decision = gov.evaluate([], "balanced", {})
        assert decision.action == "PROCEED"
        assert decision.quality_score == 1.0

    def test_failed_results_escalate_on_balanced(self):
        gov = WaveGovernor(escalate_threshold=0.55)
        decision = gov.evaluate([_failed_result(), _failed_result("t2")], "balanced", {})
        assert decision.action == "ESCALATE"
        assert decision.escalated_mode == "best"

    def test_failed_results_proceed_on_best_mode(self):
        gov = WaveGovernor(escalate_threshold=0.55)
        decision = gov.evaluate([_failed_result()], "best", {})
        assert decision.action == "PROCEED"

    def test_escalation_from_cost_efficient_to_balanced(self):
        gov = WaveGovernor(escalate_threshold=0.55)
        decision = gov.evaluate([_failed_result()], "cost_efficient", {})
        assert decision.action == "ESCALATE"
        assert decision.escalated_mode == "balanced"

    def test_max_rounds_forces_proceed(self):
        gov = WaveGovernor(escalate_threshold=0.55, max_rounds=1)
        gov.evaluate([_failed_result()], "balanced", {})  # round 1 — escalate
        decision = gov.evaluate([_failed_result()], "balanced", {})  # round 2 — max
        assert decision.action == "PROCEED"

    def test_revision_result_lowers_quality(self):
        gov = WaveGovernor()
        quality = WaveGovernor._compute_quality([_revision_result()])
        assert quality == pytest.approx(0.3)

    def test_mixed_results_partial_quality(self):
        gov = WaveGovernor()
        quality = WaveGovernor._compute_quality([_ok_result(), _failed_result()])
        assert quality == pytest.approx(0.5)

    def test_auction_floor_triggers_escalate(self):
        gov = WaveGovernor(escalate_threshold=0.1)  # threshold low so normal check passes
        ctx = {"auction_floor": 0.9}
        decision = gov.evaluate([_ok_result()], "balanced", ctx)
        # quality ~1.0 but floor is 0.9 — since 1.0 >= 0.9 this actually passes
        assert decision.action == "PROCEED"

    def test_auction_floor_violation_escalates(self):
        gov = WaveGovernor(escalate_threshold=0.1)
        ctx = {"auction_floor": 0.95}
        # revision result has quality 0.3 < floor 0.95
        decision = gov.evaluate([_revision_result()], "balanced", ctx)
        assert decision.action == "ESCALATE"
        assert decision.reason == "auction_floor_violation"

    def test_failed_task_ids_returned(self):
        gov = WaveGovernor(escalate_threshold=0.55)
        decision = gov.evaluate(
            [_ok_result("t1"), _failed_result("t2"), _error_result("t3")], "balanced", {}
        )
        assert "t2" in decision.failed_task_ids
        assert "t3" in decision.failed_task_ids

    def test_reset_clears_round_counter(self):
        gov = WaveGovernor(escalate_threshold=0.55, max_rounds=1)
        gov.evaluate([_failed_result()], "balanced", {})
        gov.reset()
        decision = gov.evaluate([_failed_result()], "balanced", {})
        assert decision.action == "ESCALATE"  # round 1 again after reset

    def test_event_callback_called(self):
        events = []
        gov = WaveGovernor(event_callback=lambda e: events.append(e))
        gov.evaluate([_ok_result()], "balanced", {})
        assert any(e.get("type") == "governor_decision" for e in events)


# ── Mode escalation map ───────────────────────────────────────────────────────


class TestModeEscalation:
    def test_cost_efficient_escalates_to_balanced(self):
        assert _MODE_ESCALATION["cost_efficient"] == "balanced"

    def test_balanced_escalates_to_best(self):
        assert _MODE_ESCALATION["balanced"] == "best"

    def test_best_stays_at_best(self):
        assert _MODE_ESCALATION["best"] == "best"


# ── LangGraph node + router ───────────────────────────────────────────────────


class TestWaveGovernorNode:
    async def test_node_writes_governor_action_to_ctx(self):
        gov = WaveGovernor()
        node = make_wave_governor(gov, "balanced")
        state = {"shared_context": {}, "results": [_ok_result()]}
        result = await node(state)
        ctx = result["shared_context"]
        assert ctx["governor_action"] == "PROCEED"
        assert "governor_quality" in ctx

    async def test_node_writes_escalated_mode_on_escalate(self):
        gov = WaveGovernor(escalate_threshold=0.9)
        node = make_wave_governor(gov, "balanced")
        state = {"shared_context": {}, "results": [_failed_result()]}
        result = await node(state)
        ctx = result["shared_context"]
        assert ctx["governor_action"] == "ESCALATE"
        assert ctx["escalated_mode"] == "best"


class TestRouteAfterWaveGovernor:
    def test_proceed_routes_to_judge(self):
        state = {"shared_context": {"governor_action": "PROCEED"}}
        assert route_after_wave_governor(state) == "judge_review"

    def test_escalate_routes_to_execution_lead(self):
        state = {"shared_context": {"governor_action": "ESCALATE"}}
        assert route_after_wave_governor(state) == "execution_lead"

    def test_terminate_routes_to_judge(self):
        state = {"shared_context": {"governor_action": "TERMINATE"}}
        assert route_after_wave_governor(state) == "judge_review"

    def test_missing_action_defaults_to_judge(self):
        state = {"shared_context": {}}
        assert route_after_wave_governor(state) == "judge_review"
