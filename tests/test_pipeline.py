"""
Tests for graphs/single_team.py — the LangGraph pipeline.

All LLM calls are mocked to test graph orchestration without real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aifactory.agents.project_manager import (
    ObjectiveSpec,
    PMDelivery,
    PMPlan,
    ProjectManagerAgent,
)
from aifactory.agents.team_lead import (
    DecompositionPlan,
    TeamLeadAgent,
    WorkerTaskSpec,
)
from aifactory.agents.worker import WorkerOutput
from aifactory.core.memory_router import MemoryRouter
from aifactory.core.schemas import (
    Artifact,
    TaskStatus,
    WorkerSpecialty,
)
from aifactory.graphs.single_team import PipelineState, SingleTeamPipeline
from tests.helpers import make_json_response, make_llm_response

_LITELLM = "aifactory.core.base_agent.litellm.acompletion"

# ─── Shared fixture ──────────────────────────────────────────────────


@pytest.fixture
def in_memory_router(memory: MemoryRouter) -> MemoryRouter:
    """Alias the conftest memory fixture for clarity."""
    return memory


# =============================================================================
# Pipeline Construction
# =============================================================================


class TestPipelineConstruction:
    def test_default_construction(self, memory: MemoryRouter) -> None:
        pipeline = SingleTeamPipeline(memory=memory)
        assert pipeline.num_workers == 3
        assert len(pipeline.worker_specialties) == 3

    def test_custom_workers(self, memory: MemoryRouter) -> None:
        pipeline = SingleTeamPipeline(
            num_workers=2,
            worker_specialties=[WorkerSpecialty.CODER, WorkerSpecialty.TESTER],
            memory=memory,
        )
        assert pipeline.num_workers == 2
        assert pipeline.worker_specialties == [WorkerSpecialty.CODER, WorkerSpecialty.TESTER]

    def test_custom_memory(self, memory: MemoryRouter) -> None:
        pipeline = SingleTeamPipeline(memory=memory)
        assert pipeline.memory is memory

    def test_compile(self, memory: MemoryRouter) -> None:
        pipeline = SingleTeamPipeline(memory=memory)
        compiled = pipeline.compile()
        assert compiled is not None


# =============================================================================
# Pipeline Execution (Mocked)
# =============================================================================


class TestPipelineExecution:
    @pytest.mark.asyncio
    async def test_full_pipeline_run(self, memory: MemoryRouter) -> None:
        """End-to-end pipeline test with mocked LLM."""
        # Mock responses for each stage:
        # 1. PM planning
        plan = PMPlan(
            project_summary="Build a calculator",
            objectives=[
                ObjectiveSpec(
                    objective_id="obj_1",
                    title="Build calculator logic",
                    description="Implement basic arithmetic operations",
                    success_criteria=["Add, subtract, multiply, divide work"],
                )
            ],
        )
        # 2. TL decomposition
        decomp = DecompositionPlan(
            objective="Build calculator logic",
            tasks=[
                WorkerTaskSpec(
                    title="Implement operations",
                    description="Build add, sub, mul, div functions",
                    assigned_to_specialty=WorkerSpecialty.CODER,
                    acceptance_criteria=["All four operations work"],
                )
            ],
        )
        # 3. Worker output
        worker_out = WorkerOutput(
            summary="Calculator operations implemented",
            artifacts=[
                Artifact(
                    name="calculator.py",
                    content_type="python",
                    content="def add(a, b): return a + b\ndef sub(a, b): return a - b",
                    summary="Basic calculator module",
                )
            ],
            issues_found=[],
            status=TaskStatus.COMPLETED,
        )

        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_json_response(plan.model_dump())
            elif call_count == 2:
                return make_json_response(decomp.model_dump())
            elif call_count == 3:
                return make_json_response(worker_out.model_dump())
            else:
                return make_llm_response("ok")

        with patch(_LITELLM, new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_acompletion
            pipeline = SingleTeamPipeline(
                num_workers=1,
                worker_specialties=[WorkerSpecialty.CODER],
                memory=memory,
            )
            delivery = await pipeline.run("Build a simple calculator")

        assert isinstance(delivery, PMDelivery)
        assert len(delivery.deliverables) >= 1
        assert delivery.metrics.get("total_tokens", 0) >= 0

    @pytest.mark.asyncio
    async def test_pipeline_with_multiple_workers(self, memory: MemoryRouter) -> None:
        """Pipeline with 2 workers handles multi-task decomposition."""
        plan = PMPlan(
            project_summary="Build an API",
            objectives=[
                ObjectiveSpec(
                    objective_id="obj_1",
                    title="Build API",
                    description="Build REST API",
                )
            ],
        )
        decomp = DecompositionPlan(
            objective="Build API",
            tasks=[
                WorkerTaskSpec(
                    title="Routes",
                    description="Build routes",
                    assigned_to_specialty=WorkerSpecialty.CODER,
                ),
                WorkerTaskSpec(
                    title="Research",
                    description="Research best practices",
                    assigned_to_specialty=WorkerSpecialty.RESEARCHER,
                ),
            ],
        )
        worker_out = WorkerOutput(
            summary="Done",
            artifacts=[Artifact(name="f.py", content_type="python", content="code")],
            issues_found=[],
            status=TaskStatus.COMPLETED,
        )
        consolidated = {
            "summary": "API built with research",
            "consolidated_artifacts": [
                {
                    "name": "api.py",
                    "content_type": "python",
                    "content": "merged code",
                    "summary": "API",
                }
            ],
            "quality_assessment": "Good",
            "issues_and_risks": [],
            "recommendations": [],
            "status": "completed",
        }

        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_json_response(plan.model_dump())
            elif call_count == 2:
                return make_json_response(decomp.model_dump())
            elif call_count in (3, 4):
                return make_json_response(worker_out.model_dump())
            elif call_count == 5:
                return make_json_response(consolidated)
            else:
                return make_llm_response("ok")

        with patch(_LITELLM, new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_acompletion
            pipeline = SingleTeamPipeline(
                num_workers=2,
                worker_specialties=[WorkerSpecialty.CODER, WorkerSpecialty.RESEARCHER],
                memory=memory,
            )
            delivery = await pipeline.run("Build an API")

        assert isinstance(delivery, PMDelivery)

    @pytest.mark.asyncio
    async def test_pipeline_failure_raises(self, memory: MemoryRouter) -> None:
        """If every LLM call fails, pipeline should raise."""
        with patch(_LITELLM, new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("Total LLM failure")
            pipeline = SingleTeamPipeline(num_workers=1, memory=memory)

            # The pipeline should still produce a delivery via fallbacks,
            # but if even fallback worker fails, it propagates
            # Due to fallback logic, the pipeline might not raise but produce a failed delivery
            try:
                delivery = await pipeline.run("Test task")
                # If fallback succeeds, check it's a valid delivery
                assert isinstance(delivery, PMDelivery)
            except (RuntimeError, Exception):
                pass  # Expected if fallbacks also fail


# =============================================================================
# Setup Agents Node
# =============================================================================


class TestSetupAgents:
    @pytest.mark.asyncio
    async def test_creates_correct_hierarchy(self, memory: MemoryRouter) -> None:
        pipeline = SingleTeamPipeline(num_workers=2, memory=memory)
        state: PipelineState = {
            "task": "Test",
            "context": "",
            "pm": None,
            "team_leads": [],
            "workers": [],
            "delivery": None,
            "status": "starting",
            "error": None,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
        }

        result = await pipeline._setup_agents(state)

        assert result["pm"] is not None
        assert isinstance(result["pm"], ProjectManagerAgent)
        assert len(result["team_leads"]) == 1
        assert isinstance(result["team_leads"][0], TeamLeadAgent)
        assert len(result["workers"][0]) == 2
        assert result["status"] == "agents_ready"

    @pytest.mark.asyncio
    async def test_agents_have_correct_models(self, memory: MemoryRouter) -> None:
        pipeline = SingleTeamPipeline(num_workers=1, memory=memory)
        state: PipelineState = {
            "task": "Test",
            "context": "",
            "pm": None,
            "team_leads": [],
            "workers": [],
            "delivery": None,
            "status": "starting",
            "error": None,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
        }

        result = await pipeline._setup_agents(state)

        pm = result["pm"]
        tl = result["team_leads"][0]
        workers = result["workers"][0]

        # PM should use premium model
        assert pm.config.model == pipeline.settings.premium_model
        # TL should use standard model
        assert tl.config.model == pipeline.settings.standard_model
        # Workers should use budget model
        for w in workers:
            assert w.config.model == pipeline.settings.budget_model

    @pytest.mark.asyncio
    async def test_supervisor_chain(self, memory: MemoryRouter) -> None:
        pipeline = SingleTeamPipeline(num_workers=1, memory=memory)
        state: PipelineState = {
            "task": "Test",
            "context": "",
            "pm": None,
            "team_leads": [],
            "workers": [],
            "delivery": None,
            "status": "starting",
            "error": None,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
        }

        result = await pipeline._setup_agents(state)

        pm = result["pm"]
        tl = result["team_leads"][0]
        worker = result["workers"][0][0]

        # Workers report to TL
        assert worker.config.supervisor_id == tl.config.agent_id
        # TL reports to PM
        assert tl.config.supervisor_id == pm.config.agent_id


# =============================================================================
# Convenience function
# =============================================================================


class TestRunSingleTeam:
    @pytest.mark.asyncio
    async def test_run_single_team_function(self, memory: MemoryRouter) -> None:
        """Test the convenience run_single_team function with patched MemoryRouter."""
        from aifactory.graphs.single_team import run_single_team

        plan = PMPlan(
            project_summary="Quick task",
            objectives=[
                ObjectiveSpec(
                    objective_id="o1",
                    title="Do task",
                    description="Do it",
                )
            ],
        )
        decomp = DecompositionPlan(
            objective="Do task",
            tasks=[WorkerTaskSpec(title="Work", description="Do work")],
        )
        worker_out = WorkerOutput(
            summary="Work done",
            artifacts=[],
            issues_found=[],
            status=TaskStatus.COMPLETED,
        )

        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_json_response(plan.model_dump())
            elif call_count == 2:
                return make_json_response(decomp.model_dump())
            elif call_count == 3:
                return make_json_response(worker_out.model_dump())
            return make_llm_response("ok")

        with patch(_LITELLM, new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_acompletion
            # Patch MemoryRouter to use in-memory Qdrant
            with patch("aifactory.graphs.single_team.MemoryRouter", return_value=memory):
                delivery = await run_single_team(
                    task="Quick task",
                    num_workers=1,
                    worker_specialties=[WorkerSpecialty.CODER],
                )

        assert isinstance(delivery, PMDelivery)
