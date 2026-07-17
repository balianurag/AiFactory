"""
Tests for judge synthetic-metrics fallback.

Regression: when the real LLM judge failed (flaky model / unparseable JSON),
judge_review returned empty judge_metrics={} with no composite_quality, so
multi-worker runs had zero quality measurement. The synthesizer must always
produce usable metrics.
"""

from __future__ import annotations

from aifactory.graphs.dynamic_pipeline import _synthesize_judge_metrics


class _FakeAgent:
    def __init__(self, name: str, tokens: int) -> None:
        self.name = name
        self.total_tokens = tokens
        self._actions = [{"x": 1}] if tokens else []


class _FakeRegistry:
    def __init__(self, agents: list[_FakeAgent]) -> None:
        self._agents = agents

    def all(self) -> list[_FakeAgent]:
        return self._agents


def test_synthesize_produces_composite_quality() -> None:
    registry = _FakeRegistry([_FakeAgent("master", 500), _FakeAgent("Worker-task_1", 300)])
    task_results = [
        {"task_id": "task_1", "result": "Good work.", "skipped": False, "quality_flagged": False}
    ]
    assessments, metrics = _synthesize_judge_metrics(registry, task_results, total_tokens=800)
    assert metrics != {}
    assert "composite_quality" in metrics
    assert float(metrics["composite_quality"]) > 0
    assert len(assessments) == 2


def test_synthesize_flags_tool_failures() -> None:
    registry = _FakeRegistry([_FakeAgent("Worker-task_1", 100)])
    # " ✗]" marker (space-cross-bracket) indicates a failed tool call
    task_results = [{"task_id": "task_1", "result": "[github ✗] 404 not found", "quality_flagged": False}]
    _, metrics = _synthesize_judge_metrics(registry, task_results, total_tokens=100)
    assert metrics["overall_status"] == "failed"


def test_synthesize_never_empty_for_normal_run() -> None:
    registry = _FakeRegistry([_FakeAgent("master", 200)])
    task_results = [
        {"task_id": "task_1", "result": "A", "skipped": False, "quality_flagged": False},
        {"task_id": "task_2", "result": "B", "skipped": False, "quality_flagged": False},
    ]
    _, metrics = _synthesize_judge_metrics(registry, task_results, total_tokens=200)
    assert metrics.get("composite_quality")
