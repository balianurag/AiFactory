"""
Tests for RL-lite scorer — record_performance, get_model_rl_scores,
infer_task_type, record_pipeline_performances, and model_selector integration.

The scorer uses PostgreSQL when DATABASE_URL is set; all tests here run against
the in-memory fallback list (_mem_records) by patching out the DB functions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aifactory.core.rl_scorer import (
    get_all_stats,
    get_model_rl_scores,
    infer_task_type,
    record_performance,
    record_pipeline_performances,
)

# ── Test isolation — reset in-memory fallback between tests ──────────────────


@pytest.fixture(autouse=True)
def _clear_mem_records(monkeypatch):
    """Ensure _mem_records is empty before every test and DB calls fail-through."""
    import aifactory.core.rl_scorer as rl_mod

    # Empty the in-memory list
    monkeypatch.setattr(rl_mod, "_mem_records", [])

    # Patch database functions to raise so the in-memory fallback is used
    def _fail(*a, **kw):
        raise RuntimeError("no DB in tests")

    monkeypatch.setattr("aifactory.core.database.insert_rl_record", _fail, raising=False)
    monkeypatch.setattr("aifactory.core.database.query_rl_scores", _fail, raising=False)
    monkeypatch.setattr("aifactory.core.database.get_all_rl_stats", _fail, raising=False)
    monkeypatch.setattr("aifactory.core.database.get_sync_engine", lambda: None, raising=False)


# ═════════════════════════════════════════════════════════════════════════════
# Task type inference
# ═════════════════════════════════════════════════════════════════════════════


class TestInferTaskType:
    def test_coding_keywords(self):
        assert infer_task_type("Write a Python function to sort a list") == "coding"

    def test_math_keywords(self):
        assert infer_task_type("Calculate the derivative of x^2 + 3x") == "math"

    def test_research_keywords(self):
        assert infer_task_type("Research the history of neural networks") == "research"

    def test_writing_keywords(self):
        assert infer_task_type("Draft a blog post about AI trends") == "writing"

    def test_analysis_keywords(self):
        assert infer_task_type("Analyze the pros and cons of microservices") == "analysis"

    def test_tool_use_keywords(self):
        assert infer_task_type("Create a GitHub issue for the login bug") == "tool_use"

    def test_general_fallback(self):
        assert infer_task_type("Do something") == "general"

    def test_case_insensitive(self):
        assert infer_task_type("IMPLEMENT a CLASS in PYTHON") == "coding"


# ═════════════════════════════════════════════════════════════════════════════
# record_performance + get_model_rl_scores (in-memory fallback)
# ═════════════════════════════════════════════════════════════════════════════


class TestRecordAndRetrieve:
    def test_record_and_retrieve(self):
        for score in [7.0, 8.0, 9.0]:
            record_performance("gpt-4o", "coding", score, tokens_used=500, run_id="run1")

        scores = get_model_rl_scores("coding", min_samples=3)
        assert "gpt-4o" in scores
        assert abs(scores["gpt-4o"] - 8.0) < 0.1

    def test_min_samples_threshold(self):
        record_performance("gpt-4o-mini", "writing", 6.0)
        record_performance("gpt-4o-mini", "writing", 7.0)

        scores = get_model_rl_scores("writing", min_samples=3)
        assert "gpt-4o-mini" not in scores

        scores = get_model_rl_scores("writing", min_samples=2)
        assert "gpt-4o-mini" in scores

    def test_different_task_types_isolated(self):
        for _ in range(3):
            record_performance("claude-3-5-haiku-20241022", "coding", 9.0)
            record_performance("claude-3-5-haiku-20241022", "research", 4.0)

        coding_scores = get_model_rl_scores("coding", min_samples=1)
        research_scores = get_model_rl_scores("research", min_samples=1)

        assert coding_scores["claude-3-5-haiku-20241022"] > 8.0
        assert research_scores["claude-3-5-haiku-20241022"] < 5.0

    def test_empty_result_for_unknown_type(self):
        scores = get_model_rl_scores("unknown_type", min_samples=1)
        assert scores == {}

    def test_record_non_fatal_on_error(self, monkeypatch):
        """record_performance must not raise even if storage is broken."""
        import aifactory.core.rl_scorer as rl_mod

        def _broken(*a, **kw):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(rl_mod, "_mem_records", MagicMock(side_effect=_broken))
        # Should not raise
        record_performance("some-model", "coding", 7.5)


# ═════════════════════════════════════════════════════════════════════════════
# get_all_stats
# ═════════════════════════════════════════════════════════════════════════════


class TestGetAllStats:
    def test_empty_state(self):
        stats = get_all_stats()
        assert stats["total_records"] == 0
        assert stats["by_task_type"] == {}

    def test_populated_stats(self):
        for _ in range(3):
            record_performance("gpt-4o", "coding", 8.0)
        for _ in range(2):
            record_performance("gpt-4o-mini", "coding", 5.0)

        stats = get_all_stats()
        assert stats["total_records"] == 5
        assert "coding" in stats["by_task_type"]

        coding = stats["by_task_type"]["coding"]
        assert coding["records"] == 5
        assert coding["models"] == 2
        assert coding["best_model"] == "gpt-4o"
        assert coding["best_avg_score"] == 8.0


# ═════════════════════════════════════════════════════════════════════════════
# record_pipeline_performances
# ═════════════════════════════════════════════════════════════════════════════


class TestRecordPipelinePerformances:
    def test_batch_record(self):
        task_results = [
            {
                "agent": "Worker-task_1",
                "model": "gpt-4o",
                "description": "Implement a sorting algorithm",
                "skipped": False,
            },
            {
                "agent": "Worker-task_2",
                "model": "gpt-4o-mini",
                "description": "Research neural networks",
                "skipped": False,
            },
        ]
        assessments = [
            {"agent": "Worker-task_1", "status": "successful", "quality": "3", "tokens": "800"},
            {"agent": "Worker-task_2", "status": "successful", "quality": "2", "tokens": "400"},
        ]

        record_pipeline_performances(task_results, assessments, run_id="test_run")

        coding_scores = get_model_rl_scores("coding", min_samples=1)
        assert "gpt-4o" in coding_scores
        assert abs(coding_scores["gpt-4o"] - 10.0) < 0.1  # quality=3 → 10/10

        research_scores = get_model_rl_scores("research", min_samples=1)
        assert "gpt-4o-mini" in research_scores
        assert abs(research_scores["gpt-4o-mini"] - 6.667) < 0.1  # quality=2 → 6.67/10

    def test_skipped_workers_excluded(self):
        task_results = [
            {
                "agent": "Worker-task_1",
                "model": "gpt-4o",
                "description": "Write code",
                "skipped": True,
            },
        ]
        assessments = [
            {"agent": "Worker-task_1", "status": "not_required", "quality": "0", "tokens": "0"}
        ]

        record_pipeline_performances(task_results, assessments)
        scores = get_model_rl_scores("coding", min_samples=1)
        assert "gpt-4o" not in scores

    def test_respects_rl_enabled_false(self, monkeypatch):
        """No records written when rl_enabled=False."""
        from aifactory.config.settings import Settings

        mock_settings = MagicMock(spec=Settings)
        mock_settings.rl_enabled = False

        with patch("aifactory.config.settings.get_settings", return_value=mock_settings):
            record_pipeline_performances(
                [
                    {
                        "agent": "Worker-task_1",
                        "model": "gpt-4o",
                        "description": "code",
                        "skipped": False,
                    }
                ],
                [{"agent": "Worker-task_1", "quality": "3", "tokens": "500"}],
            )

        assert get_model_rl_scores("coding", min_samples=1) == {}


# ═════════════════════════════════════════════════════════════════════════════
# select_model with RL scores
# ═════════════════════════════════════════════════════════════════════════════


class TestSelectModelWithRL:
    def _minimal_models(self):
        return [
            {
                "id": "model-a",
                "reasoning": 3,
                "coding": 3,
                "math": 2,
                "writing": 2,
                "instruction_following": 2,
                "tool_use": 1,
                "knowledge": 2,
                "speed": 1,
                "cost_efficiency": 1,
                "context_handling": 2,
            },
            {
                "id": "model-b",
                "reasoning": 2,
                "coding": 2,
                "math": 1,
                "writing": 2,
                "instruction_following": 2,
                "tool_use": 1,
                "knowledge": 2,
                "speed": 2,
                "cost_efficiency": 2,
                "context_handling": 2,
            },
        ]

    def test_no_rl_scores_unchanged(self):
        from aifactory.core.model_selector import select_model

        models = self._minimal_models()
        reqs = {"coding": 3, "reasoning": 3}

        model_a, _ = select_model(reqs, models, "balanced", rl_scores=None)
        model_b, _ = select_model(reqs, models, "balanced")
        assert model_a == model_b

    def test_rl_boost_changes_selection(self):
        from aifactory.core.model_selector import select_model

        models = self._minimal_models()
        reqs = {"coding": 3, "reasoning": 3}

        static_winner, _ = select_model(reqs, models, "best")
        assert static_winner == "model-a"

        rl_scores = {"model-a": 1.0, "model-b": 9.5}
        rl_winner, _ = select_model(reqs, models, "best", rl_scores=rl_scores, rl_weight=0.5)
        assert rl_winner == "model-b"

    def test_rl_weight_zero_disables_rl(self):
        from aifactory.core.model_selector import select_model

        models = self._minimal_models()
        reqs = {"coding": 3}
        rl_scores = {"model-b": 10.0, "model-a": 0.0}

        winner, _ = select_model(reqs, models, "best", rl_scores=rl_scores, rl_weight=0.0)
        assert winner == "model-a"

    def test_rl_model_not_in_scores_unaffected(self):
        from aifactory.core.model_selector import select_model

        models = self._minimal_models()
        reqs = {"coding": 3, "reasoning": 3}

        rl_scores = {"model-a": 0.0}
        winner, _ = select_model(reqs, models, "best", rl_scores=rl_scores, rl_weight=0.5)
        assert winner == "model-b"
