"""
Tests for S1 (FINDINGS replaces prior_results) and S3 (lite mode routing/judge).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aifactory.graphs.dynamic_pipeline import (
    route_after_researcher,
    route_after_researcher_lite,
)

# =============================================================================
# S1: FINDINGS: replaces prior-results prose — logic unit tests
# =============================================================================


def _build_task_input(
    prior_results: list[dict],
    findings_context: str,
) -> str:
    """
    Replicate the task_input construction logic from _run_worker_task so we
    can test the if/else branch without invoking the full function.
    """
    if findings_context:
        return findings_context + "\n\nExecute your assigned task."
    else:
        prior = "\n\n".join(
            f"[{r['task_id']}]\n{r['result']}"
            for r in prior_results
            if r.get("result") and r["result"].strip() != "NOT_NEEDED"
        )
        return "Execute your assigned task." + (
            f"\n\nPrior task results:\n{prior}" if prior else ""
        )


class TestFindingsReplacesPrior:
    """When findings_context is set, prior_results prose must NOT be injected."""

    def test_findings_context_used_exclusively(self) -> None:
        fc = (
            "## Prior Task Findings\n"
            "[task_1]\n"
            "  - FastAPI app at main.py — routes: /health /route /geofence\n"
            "  - PostGIS table: geofences(id, name, polygon)"
        )
        prior = [{"task_id": "task_1", "result": "A" * 6000, "skipped": False}]

        result = _build_task_input(prior, fc)

        assert "## Prior Task Findings" in result
        assert "[task_1]" in result
        # Full 6000-char prose must NOT appear
        assert "A" * 100 not in result

    def test_findings_replaces_prior_not_concatenates(self) -> None:
        fc = "## Prior Task Findings\n[t1]\n  - done"
        prior = [{"task_id": "t1", "result": "UNIQUE_PRIOR_MARKER", "skipped": False}]

        result = _build_task_input(prior, fc)

        # findings present
        assert "## Prior Task Findings" in result
        # raw prior result absent — not concatenated
        assert "UNIQUE_PRIOR_MARKER" not in result

    def test_no_findings_falls_back_to_prior_results(self) -> None:
        prior = [{"task_id": "t1", "result": "UNIQUE_PRIOR_CONTENT", "skipped": False}]

        result = _build_task_input(prior, "")

        assert "UNIQUE_PRIOR_CONTENT" in result
        assert "Prior task results" in result

    def test_no_findings_no_prior_clean_prompt(self) -> None:
        result = _build_task_input([], "")
        assert result == "Execute your assigned task."
        assert "Prior task results" not in result

    def test_not_needed_skipped_in_fallback(self) -> None:
        prior = [
            {"task_id": "t1", "result": "NOT_NEEDED", "skipped": False},
            {"task_id": "t2", "result": "REAL_RESULT", "skipped": False},
        ]
        result = _build_task_input(prior, "")
        assert "REAL_RESULT" in result
        assert "NOT_NEEDED" not in result

    def test_empty_findings_triggers_fallback(self) -> None:
        prior = [{"task_id": "t1", "result": "MARKER", "skipped": False}]
        result = _build_task_input(prior, "")
        assert "MARKER" in result


# =============================================================================
# S3: Lite-mode routing
# =============================================================================


class TestLiteModeRouting:
    def test_standard_routing_with_research_for_analysis(self) -> None:
        state = {"research_for_analysis": True}
        assert route_after_researcher(state) == "reanalyse"

    def test_standard_routing_without_research_for_analysis(self) -> None:
        state = {"research_for_analysis": False}
        assert route_after_researcher(state) == "execution_lead"

    def test_lite_routing_always_skips_reanalyse(self) -> None:
        """In lite mode, even when research_for_analysis=True, go straight to execution_lead."""
        state_with = {"research_for_analysis": True}
        state_without = {"research_for_analysis": False}
        assert route_after_researcher_lite(state_with) == "execution_lead"
        assert route_after_researcher_lite(state_without) == "execution_lead"


class TestLiteModeJudgeReview:
    @pytest.mark.asyncio
    async def test_judge_review_lite_skips_llm(self) -> None:
        """In lite mode, judge_review returns synthetic metrics without calling agent.think()."""
        from aifactory.core.agent_factory import AgentRegistry
        from aifactory.graphs.dynamic_pipeline import make_judge_review

        mock_judge = MagicMock()
        mock_judge.think = AsyncMock(side_effect=Exception("Should not be called"))
        mock_judge.name = "judge"
        mock_judge.total_tokens = 0
        mock_judge.cfg = {"role": "Judge"}
        mock_judge._actions = []

        registry = AgentRegistry()

        judge_fn = make_judge_review(mock_judge, registry, run_id="r1", lite_mode=True)
        result = await judge_fn(
            {
                "user_query": "test",
                "shared_context": {},
                "task_results": [
                    {"task_id": "t1", "result": "done", "skipped": False, "model": "m"}
                ],
            }
        )

        assert "judge_metrics" in result
        assert result["judge_metrics"]["pipeline_efficiency"] == "lite"
        # LLM was never called
        mock_judge.think.assert_not_called()

    @pytest.mark.asyncio
    async def test_judge_review_lite_status_done_when_tasks_completed(self) -> None:
        from aifactory.core.agent_factory import AgentRegistry
        from aifactory.graphs.dynamic_pipeline import make_judge_review

        mock_judge = MagicMock()
        mock_judge.think = AsyncMock()
        mock_judge.total_tokens = 0
        mock_judge.cfg = {}
        mock_judge._actions = []

        registry = AgentRegistry()
        judge_fn = make_judge_review(mock_judge, registry, run_id="r1", lite_mode=True)

        result = await judge_fn(
            {
                "user_query": "q",
                "shared_context": {},
                "task_results": [{"task_id": "t1", "result": "x", "skipped": False}],
            }
        )
        assert result["judge_metrics"]["overall_status"] == "done"

    @pytest.mark.asyncio
    async def test_judge_review_lite_status_partial_when_no_tasks(self) -> None:
        from aifactory.core.agent_factory import AgentRegistry
        from aifactory.graphs.dynamic_pipeline import make_judge_review

        mock_judge = MagicMock()
        mock_judge.think = AsyncMock()
        mock_judge.total_tokens = 0
        mock_judge.cfg = {}
        mock_judge._actions = []

        registry = AgentRegistry()
        judge_fn = make_judge_review(mock_judge, registry, run_id="r1", lite_mode=True)

        result = await judge_fn({"user_query": "q", "shared_context": {}, "task_results": []})
        assert result["judge_metrics"]["overall_status"] == "partial"
