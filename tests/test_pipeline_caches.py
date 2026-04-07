"""
Tests for the plan/worker/completed caches, _elapsed(), write_dump(),
and judge plain-text report features ported from agent-arch.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from aifactory.core.agent_factory import AgentRegistry
from aifactory.graphs.dynamic_pipeline import (
    DUMPS_DIR,
    DynamicPipelineState,
    PipelineAgent,
    _elapsed,
    make_execution_lead,
    make_judge_review,
    write_dump,
)
from tests.helpers import make_llm_response

_LITELLM_PATCH = "aifactory.graphs.dynamic_pipeline.litellm.acompletion"


def _mock_litellm(content: str, prompt_tokens: int = 100, completion_tokens: int = 50):
    resp = make_llm_response(content, prompt_tokens, completion_tokens)
    return patch(_LITELLM_PATCH, AsyncMock(return_value=resp))


# =============================================================================
# _elapsed() helper
# =============================================================================


class TestElapsed:
    def test_no_actions(self) -> None:
        agent = PipelineAgent("test", {"system_prompt": "test"})
        assert _elapsed(agent) == "-"

    def test_single_action(self) -> None:
        agent = PipelineAgent("test", {"system_prompt": "test"})
        agent._actions = [{"elapsed_s": 1.5}]
        assert _elapsed(agent) == "1.5s"

    def test_multiple_actions(self) -> None:
        agent = PipelineAgent("test", {"system_prompt": "test"})
        agent._actions = [
            {"elapsed_s": 1.0},
            {"elapsed_s": 2.5},
            {"elapsed_s": 0.3},
        ]
        assert _elapsed(agent) == "3.8s"

    def test_missing_elapsed_s_treated_as_zero(self) -> None:
        agent = PipelineAgent("test", {"system_prompt": "test"})
        agent._actions = [{"elapsed_s": 1.0}, {"some_key": "value"}]
        assert _elapsed(agent) == "1.0s"


# =============================================================================
# Plan cache / Worker cache / Completed cache
# =============================================================================


class TestExecutionLeadCaches:
    @pytest.mark.asyncio
    async def test_plan_cache_created_on_first_run(self) -> None:
        """On first run, execution lead creates a _plan_cache on the lead agent."""
        lead = PipelineAgent("execution_lead", {"system_prompt": "Lead.", "max_tokens": 1000})
        worker_cfg = {"system_prompt": "Worker.", "max_tokens": 500, "abilities": {}}
        registry = AgentRegistry()
        registry.register(lead)

        node = make_execution_lead(lead, worker_cfg, [], "balanced", registry)

        # Plan response + worker response
        plan_json = json.dumps(
            {
                "plan": [{"id": "task_1", "reasoning": 2, "coding": 1}],
                "prompts": {"task_1": "Do the task."},
            }
        )
        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_llm_response(plan_json)
            elif call_count <= 3:
                return make_llm_response("Worker result for task_1")
            else:
                return make_llm_response("Consolidated answer.")

        with patch(_LITELLM_PATCH, AsyncMock(side_effect=side_effect)):
            state: DynamicPipelineState = {
                "user_query": "Do something",
                "shared_context": {},
                "tasks": [{"id": "task_1", "description": "Do the task"}],
            }
            await node(state)

        # Plan cache should be set on the lead agent
        assert hasattr(lead, "_plan_cache")
        assert lead._plan_cache["key"] == (("task_1", "Do the task"),)
        assert "completed" in lead._plan_cache
        assert "workers" in lead._plan_cache
        assert "task_1" in lead._plan_cache["completed"]

    @pytest.mark.asyncio
    async def test_plan_cache_reused_on_resume(self) -> None:
        """When plan cache key matches, the planning call is skipped."""
        lead = PipelineAgent("execution_lead", {"system_prompt": "Lead.", "max_tokens": 1000})
        worker_cfg = {"system_prompt": "Worker.", "max_tokens": 500, "abilities": {}}
        registry = AgentRegistry()
        registry.register(lead)

        # Pre-populate cache (simulating a previous run)
        lead._plan_cache = {
            "key": (("task_1", "Do the task"),),
            "worker_prompts": {"task_1": "Do the task."},
            "ability_reqs": {"task_1": {"reasoning": 2}},
            "completed": {
                "task_1": {
                    "task_id": "task_1",
                    "agent": "Worker-task_1",
                    "model": "gpt-4o-mini",
                    "description": "Do the task",
                    "result": "Already done",
                    "skipped": False,
                }
            },
            "workers": {},
        }

        node = make_execution_lead(lead, worker_cfg, [], "balanced", registry)

        # Consolidation call only (no planning, no worker call since task_1 is completed)
        with _mock_litellm("Consolidated from cache."):
            state: DynamicPipelineState = {
                "user_query": "Do something",
                "shared_context": {},
                "tasks": [{"id": "task_1", "description": "Do the task"}],
            }
            result = await node(state)

        assert "Consolidated from cache." in result["final_response"]
        # Only 1 LLM call: consolidation (plan was cached, worker was completed)
        assert lead.total_tokens > 0

    @pytest.mark.asyncio
    async def test_completed_tasks_skipped(self) -> None:
        """Tasks already in the completed cache are skipped."""
        lead = PipelineAgent("execution_lead", {"system_prompt": "Lead.", "max_tokens": 1000})
        worker_cfg = {"system_prompt": "Worker.", "max_tokens": 500, "abilities": {}}
        registry = AgentRegistry()
        registry.register(lead)

        # Pre-populate cache with task_1 completed, task_2 not
        lead._plan_cache = {
            "key": (("task_1", "Task one"), ("task_2", "Task two")),
            "worker_prompts": {"task_1": "Do one.", "task_2": "Do two."},
            "ability_reqs": {},
            "completed": {
                "task_1": {
                    "task_id": "task_1",
                    "agent": "Worker-task_1",
                    "model": "gpt-4o-mini",
                    "description": "Task one",
                    "result": "Done",
                    "skipped": False,
                }
            },
            "workers": {},
        }

        node = make_execution_lead(lead, worker_cfg, [], "balanced", registry)

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_llm_response("Worker result for task_2")
            else:
                return make_llm_response("Consolidated.")

        with patch(_LITELLM_PATCH, AsyncMock(side_effect=side_effect)):
            state: DynamicPipelineState = {
                "user_query": "Do two things",
                "shared_context": {},
                "tasks": [
                    {"id": "task_1", "description": "Task one"},
                    {"id": "task_2", "description": "Task two"},
                ],
            }
            result = await node(state)

        # task_2 should now also be in completed
        assert "task_2" in lead._plan_cache["completed"]
        assert len(result["task_results"]) == 2

    @pytest.mark.asyncio
    async def test_worker_cache_stores_worker(self) -> None:
        """Worker agents are cached in workers_cache after creation."""
        lead = PipelineAgent("execution_lead", {"system_prompt": "Lead.", "max_tokens": 1000})
        worker_cfg = {"system_prompt": "Worker.", "max_tokens": 500, "abilities": {}}
        registry = AgentRegistry()
        registry.register(lead)

        node = make_execution_lead(lead, worker_cfg, [], "balanced", registry)

        plan_json = json.dumps(
            {
                "plan": [{"id": "task_1"}],
                "prompts": {"task_1": "Do it."},
            }
        )
        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_llm_response(plan_json)
            elif call_count == 2:
                return make_llm_response("Worker done.")
            else:
                return make_llm_response("Consolidated.")

        with patch(_LITELLM_PATCH, AsyncMock(side_effect=side_effect)):
            state: DynamicPipelineState = {
                "user_query": "Test",
                "shared_context": {},
                "tasks": [{"id": "task_1", "description": "Do it"}],
            }
            await node(state)

        assert "task_1" in lead._plan_cache["workers"]
        worker = lead._plan_cache["workers"]["task_1"]
        assert isinstance(worker, PipelineAgent)
        assert worker.name == "Worker-task_1"


# =============================================================================
# write_dump()
# =============================================================================


class TestWriteDump:
    def setup_method(self) -> None:
        """Clean up test dump files."""
        self._test_dir = DUMPS_DIR / "runs"
        self._test_dir.mkdir(parents=True, exist_ok=True)

    def teardown_method(self) -> None:
        """Remove test dumps."""
        test_file = self._test_dir / "test_run.json"
        if test_file.exists():
            test_file.unlink()

    def test_creates_dump_file(self) -> None:
        registry = AgentRegistry()
        agent = PipelineAgent("master", {"system_prompt": "m", "role": "Orchestrator"})
        registry.register(agent)

        result = {
            "shared_context": {"objective": "Test"},
            "tasks": [{"id": "task_1", "description": "A task"}],
            "task_results": [],
            "final_response": "Done.",
        }
        cfg = {"model": "gpt-4o-mini", "mode": "balanced"}

        path = write_dump("test_run", cfg, result, registry, 5.0, "Test query")
        assert path.exists()

        data = json.loads(path.read_text())
        assert data["run_id"] == "test_run"
        assert data["elapsed_seconds"] == 5.0
        assert data["config"]["model"] == "gpt-4o-mini"
        assert data["config"]["mode"] == "balanced"
        assert data["config"]["request"] == "Test query"
        assert data["final_response"] == "Done."
        assert len(data["agents"]) == 1
        assert data["agents"][0]["name"] == "master"
        assert data["totals"]["agents"] == 1

    def test_dump_includes_totals(self) -> None:
        registry = AgentRegistry()
        a1 = PipelineAgent("a1", {"system_prompt": "x"})
        a1.total_tokens = 100
        a1._actions = [{"elapsed_s": 1.0}]
        a2 = PipelineAgent("a2", {"system_prompt": "y"})
        a2.total_tokens = 200
        a2._actions = [{"elapsed_s": 2.0}, {"elapsed_s": 3.0}]
        registry.register(a1)
        registry.register(a2)

        path = write_dump("test_run", {"model": "x"}, {}, registry, 10.0)
        data = json.loads(path.read_text())
        assert data["totals"]["total_tokens"] == 300
        assert data["totals"]["total_actions"] == 3


# =============================================================================
# Judge plain-text report
# =============================================================================


class TestJudgeReport:
    def teardown_method(self) -> None:
        """Clean up test judge dumps."""
        judge_dir = DUMPS_DIR / "judge"
        for f in judge_dir.glob("test_judge_*.txt"):
            f.unlink()

    @pytest.mark.asyncio
    async def test_judge_review_writes_report(self) -> None:
        judge = PipelineAgent("judge", {"system_prompt": "You judge.", "max_tokens": 500})
        registry = AgentRegistry()
        master = PipelineAgent("master", {"system_prompt": "m", "role": "Orchestrator"})
        master.total_tokens = 150
        master.prompt_tokens = 100
        master.completion_tokens = 50
        master._actions = [{"elapsed_s": 1.2}]
        registry.register(master)
        registry.register(judge)

        node = make_judge_review(judge, registry, run_id="test_judge_run")

        review_json = json.dumps(
            {
                "assessments": [
                    {
                        "agent": "master",
                        "status": "successful",
                        "quality": 3,
                        "tokens": 150,
                        "notes": "Good",
                    },
                    {
                        "agent": "judge",
                        "status": "successful",
                        "quality": 2,
                        "tokens": 300,
                        "notes": "Reviewed",
                    },
                ],
                "metrics": {
                    "overall_status": "successful",
                    "pipeline_efficiency": "high",
                },
            }
        )
        with _mock_litellm(review_json):
            await node(
                {
                    "user_query": "Test query",
                    "shared_context": {},
                    "task_results": [
                        {
                            "task_id": "task_1",
                            "agent": "Worker-task_1",
                            "model": "gpt-4o-mini",
                            "description": "Test task",
                            "result": "Result here",
                            "skipped": False,
                        },
                    ],
                }
            )

        # Check the report file was created
        report_path = DUMPS_DIR / "judge" / "test_judge_run.txt"
        assert report_path.exists()

        report = report_path.read_text()
        assert "test_judge_run" in report
        assert "Test query" in report
        assert "EXECUTION PIPELINE" in report
        assert "PIPELINE METRICS" in report

    @pytest.mark.asyncio
    async def test_judge_review_includes_elapsed_in_summary(self) -> None:
        """Judge review should include elapsed time in the agent rows."""
        judge = PipelineAgent("judge", {"system_prompt": "Judge.", "max_tokens": 500})
        registry = AgentRegistry()
        master = PipelineAgent("master", {"system_prompt": "m"})
        master._actions = [{"elapsed_s": 2.5}]
        registry.register(master)
        registry.register(judge)

        node = make_judge_review(judge, registry, run_id="test_judge_elapsed")

        review_json = json.dumps(
            {
                "assessments": [],
                "metrics": {"overall_status": "successful"},
            }
        )

        with _mock_litellm(review_json) as mock_call:
            await node(
                {
                    "user_query": "Test",
                    "shared_context": {},
                    "task_results": [],
                }
            )

        # Check that elapsed time was included in the prompt to the judge
        call_kwargs = mock_call.call_args.kwargs
        messages = call_kwargs["messages"]
        user_msg = messages[-1]["content"]
        assert "2.5s" in user_msg

    @pytest.mark.asyncio
    async def test_judge_review_default_run_id(self) -> None:
        """Judge review with default run_id='unknown' still works."""
        judge = PipelineAgent("judge", {"system_prompt": "Judge.", "max_tokens": 500})
        registry = AgentRegistry()
        registry.register(judge)

        node = make_judge_review(judge, registry)  # default run_id="unknown"

        review_json = json.dumps({"assessments": [], "metrics": {}})
        with _mock_litellm(review_json):
            result = await node(
                {
                    "user_query": "Test",
                    "shared_context": {},
                    "task_results": [],
                }
            )

        assert "agent_assessments" in result
        report_path = DUMPS_DIR / "judge" / "unknown.txt"
        assert report_path.exists()
        # Clean up
        report_path.unlink()


# =============================================================================
# build_dynamic_pipeline with run_id
# =============================================================================


class TestBuildPipelineRunId:
    def test_accepts_run_id(self) -> None:
        from aifactory.graphs.dynamic_pipeline import build_dynamic_pipeline

        with patch(_LITELLM_PATCH):
            compiled, _registry = build_dynamic_pipeline(mode="balanced", run_id="test_1234")
        assert compiled is not None

    def test_auto_generates_run_id(self) -> None:
        from aifactory.graphs.dynamic_pipeline import build_dynamic_pipeline

        with patch(_LITELLM_PATCH):
            compiled, _registry = build_dynamic_pipeline(mode="balanced")
        assert compiled is not None
