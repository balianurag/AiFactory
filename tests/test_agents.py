"""
Tests for agents — Worker, TeamLead, ProjectManager.

All LLM calls are mocked to avoid real API calls.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from aifactory.agents.project_manager import (
    ObjectiveSpec,
    PMDelivery,
    PMPlan,
    ProjectManagerAgent,
)
from aifactory.agents.team_lead import (
    ConsolidatedOutput,
    DecompositionPlan,
    TeamLeadAgent,
    WorkerTaskSpec,
)
from aifactory.agents.worker import WorkerAgent, WorkerOutput
from aifactory.core.memory_router import MemoryRouter
from aifactory.core.schemas import (
    AgentConfig,
    AgentRole,
    Artifact,
    TaskAssignment,
    TaskStatus,
    WorkerSpecialty,
)

from tests.helpers import make_json_response, make_llm_response, make_task_assignment


# =============================================================================
# WorkerAgent
# =============================================================================


class TestWorkerAgent:
    @pytest.fixture
    def worker(self, worker_config: AgentConfig, memory: MemoryRouter) -> WorkerAgent:
        return WorkerAgent(config=worker_config, memory=memory)

    def test_init_loads_prompt(self, worker: WorkerAgent) -> None:
        """Worker should have a system prompt (from file or fallback)."""
        assert worker._system_message  # Not empty
        assert "worker" in worker._system_message.lower()

    def test_init_fallback_prompt(self) -> None:
        """When prompt file doesn't exist, fallback is used."""
        config = AgentConfig(
            role=AgentRole.WORKER,
            name="Test",
            model="test",
            specialty=WorkerSpecialty.CODER,
        )
        # The prompt file may or may not exist, but agent should still init
        worker = WorkerAgent(config=config)
        assert worker._system_message

    @pytest.mark.asyncio
    async def test_execute_task_string(self, worker: WorkerAgent) -> None:
        """Worker handles a plain string task."""
        worker_output = WorkerOutput(
            summary="Built the feature",
            artifacts=[
                Artifact(name="main.py", content_type="python", content="print('hello')", summary="Main file")
            ],
            issues_found=[],
            status=TaskStatus.COMPLETED,
        )
        mock_resp = make_json_response(worker_output.model_dump())

        with patch("aifactory.core.base_agent.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            result = await worker.execute_task("Build a hello world app")

        assert result.status == TaskStatus.COMPLETED
        assert result.summary == "Built the feature"
        assert len(result.artifacts) == 1

    @pytest.mark.asyncio
    async def test_execute_task_assignment(self, worker: WorkerAgent) -> None:
        """Worker handles a TaskAssignment object."""
        task = make_task_assignment(
            title="Implement auth",
            description="Build JWT authentication",
        )
        worker_output = WorkerOutput(
            summary="Implemented JWT auth",
            artifacts=[
                Artifact(name="auth.py", content_type="python", content="def auth(): pass", summary="Auth")
            ],
            issues_found=[],
            status=TaskStatus.COMPLETED,
        )
        mock_resp = make_json_response(worker_output.model_dump())

        with patch("aifactory.core.base_agent.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            result = await worker.execute_task(task=task)

        assert result.status == TaskStatus.COMPLETED
        assert result.task_id == task.task_id
        assert result.from_agent == worker.config.agent_id

    @pytest.mark.asyncio
    async def test_execute_task_llm_failure(self, worker: WorkerAgent) -> None:
        """Worker returns FAILED result when LLM call fails."""
        with patch("aifactory.core.base_agent.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("API timeout")
            result = await worker.execute_task("Test task")

        assert result.status == TaskStatus.FAILED
        assert "failed" in result.summary.lower() or "API timeout" in result.summary

    @pytest.mark.asyncio
    async def test_execute_stores_memory(
        self, worker: WorkerAgent, memory: MemoryRouter
    ) -> None:
        """Worker stores decision in memory after completing a task."""
        worker_output = WorkerOutput(
            summary="Done",
            artifacts=[],
            issues_found=[],
            status=TaskStatus.COMPLETED,
        )
        mock_resp = make_json_response(worker_output.model_dump())

        with patch("aifactory.core.base_agent.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            await worker.execute_task("Test task")

        stats = memory.get_stats()
        assert stats["agent"] >= 1


# =============================================================================
# WorkerOutput model
# =============================================================================


class TestWorkerOutput:
    def test_creation(self) -> None:
        output = WorkerOutput(
            summary="Built the API",
            artifacts=[Artifact(name="api.py", content_type="python", content="app = FastAPI()")],
            issues_found=["No tests yet"],
            status=TaskStatus.COMPLETED,
        )
        assert output.summary == "Built the API"
        assert len(output.artifacts) == 1
        assert len(output.issues_found) == 1

    def test_defaults(self) -> None:
        output = WorkerOutput(summary="Done")
        assert output.artifacts == []
        assert output.issues_found == []
        assert output.status == TaskStatus.COMPLETED


# =============================================================================
# TeamLeadAgent
# =============================================================================


class TestTeamLeadAgent:
    @pytest.fixture
    def team_lead(
        self, tl_config: AgentConfig, worker_config: AgentConfig, memory: MemoryRouter
    ) -> TeamLeadAgent:
        tl = TeamLeadAgent(config=tl_config, memory=memory)
        w = WorkerAgent(config=worker_config, memory=memory)
        tl.add_worker(w)
        return tl

    def test_init(self, team_lead: TeamLeadAgent) -> None:
        assert len(team_lead.workers) == 1

    def test_add_worker(self, tl_config: AgentConfig) -> None:
        tl = TeamLeadAgent(config=tl_config)
        assert len(tl.workers) == 0

        w_config = AgentConfig(
            role=AgentRole.WORKER,
            name="Worker",
            model="test",
            specialty=WorkerSpecialty.CODER,
        )
        tl.add_worker(WorkerAgent(config=w_config))
        assert len(tl.workers) == 1
        assert w_config.agent_id in tl.config.subordinate_ids

    @pytest.mark.asyncio
    async def test_execute_objective_full_flow(self, team_lead: TeamLeadAgent) -> None:
        """Full lifecycle: decompose → delegate → consolidate."""
        # Mock decomposition response
        decomp_plan = DecompositionPlan(
            objective="Build auth",
            tasks=[
                WorkerTaskSpec(
                    title="Implement JWT",
                    description="Build JWT token generation and validation",
                    assigned_to_specialty=WorkerSpecialty.CODER,
                    acceptance_criteria=["Tokens generated", "Tokens validated"],
                )
            ],
        )
        # Mock worker output
        worker_output = WorkerOutput(
            summary="JWT implemented",
            artifacts=[
                Artifact(name="jwt.py", content_type="python", content="# JWT code", summary="JWT module")
            ],
            issues_found=[],
            status=TaskStatus.COMPLETED,
        )

        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # TL decomposition
                return make_json_response(decomp_plan.model_dump())
            elif call_count == 2:
                # Worker execution
                return make_json_response(worker_output.model_dump())
            else:
                # Should not get here for single-worker single-task
                return make_llm_response("fallback")

        with patch("aifactory.core.base_agent.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_acompletion
            result = await team_lead.execute_objective("Build JWT authentication")

        assert result.status == TaskStatus.COMPLETED
        assert len(result.artifacts) >= 1

    @pytest.mark.asyncio
    async def test_decomposition_fallback(self, team_lead: TeamLeadAgent) -> None:
        """When decomposition LLM fails, fallback plan is used."""
        worker_output = WorkerOutput(
            summary="Done",
            artifacts=[],
            issues_found=[],
            status=TaskStatus.COMPLETED,
        )

        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("LLM decomposition failed")
            else:
                return make_json_response(worker_output.model_dump())

        with patch("aifactory.core.base_agent.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_acompletion
            result = await team_lead.execute_objective("Test objective")

        # Should still succeed with fallback
        assert result.status == TaskStatus.COMPLETED

    def test_pick_worker_by_specialty(self, tl_config: AgentConfig) -> None:
        tl = TeamLeadAgent(config=tl_config)

        coder_config = AgentConfig(
            role=AgentRole.WORKER,
            name="Coder",
            model="test",
            specialty=WorkerSpecialty.CODER,
        )
        researcher_config = AgentConfig(
            role=AgentRole.WORKER,
            name="Researcher",
            model="test",
            specialty=WorkerSpecialty.RESEARCHER,
        )
        tl.add_worker(WorkerAgent(config=coder_config))
        tl.add_worker(WorkerAgent(config=researcher_config))

        picked = tl._pick_worker(WorkerSpecialty.RESEARCHER, 0)
        assert picked.config.specialty == WorkerSpecialty.RESEARCHER

    def test_pick_worker_fallback_round_robin(self, tl_config: AgentConfig) -> None:
        tl = TeamLeadAgent(config=tl_config)
        w_config = AgentConfig(
            role=AgentRole.WORKER,
            name="Worker",
            model="test",
            specialty=WorkerSpecialty.CODER,
        )
        tl.add_worker(WorkerAgent(config=w_config))

        # Request a specialty that doesn't exist — should fallback
        picked = tl._pick_worker(WorkerSpecialty.TESTER, 0)
        assert picked.config.specialty == WorkerSpecialty.CODER

    def test_pick_worker_no_workers_raises(self, tl_config: AgentConfig) -> None:
        tl = TeamLeadAgent(config=tl_config)
        with pytest.raises(RuntimeError, match="has no workers"):
            tl._pick_worker(WorkerSpecialty.CODER, 0)

    def test_total_team_tokens(self, team_lead: TeamLeadAgent) -> None:
        team_lead.total_tokens_used = 100
        team_lead.workers[0].total_tokens_used = 200
        assert team_lead._total_team_tokens() == 300


# =============================================================================
# DecompositionPlan & ConsolidatedOutput models
# =============================================================================


class TestTeamLeadModels:
    def test_decomposition_plan(self) -> None:
        plan = DecompositionPlan(
            objective="Build auth",
            tasks=[
                WorkerTaskSpec(title="JWT", description="Build JWT"),
                WorkerTaskSpec(title="OAuth", description="Build OAuth"),
            ],
        )
        assert len(plan.tasks) == 2

    def test_consolidated_output(self) -> None:
        output = ConsolidatedOutput(
            summary="Auth system built",
            consolidated_artifacts=[
                Artifact(name="auth.py", content_type="python", content="code")
            ],
            quality_assessment="High quality",
            issues_and_risks=["No MFA yet"],
            recommendations=["Add MFA support"],
            status=TaskStatus.COMPLETED,
        )
        assert len(output.consolidated_artifacts) == 1


# =============================================================================
# ProjectManagerAgent
# =============================================================================


class TestProjectManagerAgent:
    @pytest.fixture
    def pm(
        self,
        pm_config: AgentConfig,
        tl_config: AgentConfig,
        worker_config: AgentConfig,
        memory: MemoryRouter,
    ) -> ProjectManagerAgent:
        pm = ProjectManagerAgent(config=pm_config, memory=memory)
        tl = TeamLeadAgent(config=tl_config, memory=memory)
        w = WorkerAgent(config=worker_config, memory=memory)
        tl.add_worker(w)
        pm.add_team_lead(tl)
        return pm

    def test_init(self, pm: ProjectManagerAgent) -> None:
        assert len(pm.team_leads) == 1

    def test_add_team_lead(self, pm_config: AgentConfig) -> None:
        pm = ProjectManagerAgent(config=pm_config)
        assert len(pm.team_leads) == 0

        tl_config = AgentConfig(
            role=AgentRole.TEAM_LEAD,
            name="TL",
            model="test",
        )
        tl = TeamLeadAgent(config=tl_config)
        pm.add_team_lead(tl)
        assert len(pm.team_leads) == 1
        assert tl_config.agent_id in pm.config.subordinate_ids

    @pytest.mark.asyncio
    async def test_execute_project_full_flow(self, pm: ProjectManagerAgent) -> None:
        """Full lifecycle: plan → delegate → consolidate → deliver."""
        # Mock PM plan
        plan = PMPlan(
            project_summary="Build a todo API",
            objectives=[
                ObjectiveSpec(
                    objective_id="obj_1",
                    title="Build REST endpoints",
                    description="Create CRUD endpoints for todos",
                    success_criteria=["All endpoints work"],
                )
            ],
        )
        # Mock TL decomposition
        decomp = DecompositionPlan(
            objective="Build REST endpoints",
            tasks=[
                WorkerTaskSpec(
                    title="Create API routes",
                    description="Implement FastAPI routes",
                    assigned_to_specialty=WorkerSpecialty.CODER,
                )
            ],
        )
        # Mock Worker output
        worker_out = WorkerOutput(
            summary="API routes created",
            artifacts=[
                Artifact(name="routes.py", content_type="python", content="# routes", summary="Routes")
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
                return make_llm_response("fallback")

        with patch("aifactory.core.base_agent.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_acompletion
            delivery = await pm.execute_project("Build a todo API")

        assert isinstance(delivery, PMDelivery)
        assert len(delivery.deliverables) >= 1
        assert delivery.metrics.get("total_tokens", 0) >= 0

    @pytest.mark.asyncio
    async def test_plan_fallback(self, pm: ProjectManagerAgent) -> None:
        """When planning LLM fails, fallback plan is used."""
        decomp = DecompositionPlan(
            objective="Execute project",
            tasks=[WorkerTaskSpec(title="Do it", description="Do the thing")],
        )
        worker_out = WorkerOutput(
            summary="Done",
            artifacts=[],
            issues_found=[],
            status=TaskStatus.COMPLETED,
        )

        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Planning failed")
            elif call_count == 2:
                return make_json_response(decomp.model_dump())
            else:
                return make_json_response(worker_out.model_dump())

        with patch("aifactory.core.base_agent.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_acompletion
            delivery = await pm.execute_project("Test project")

        assert isinstance(delivery, PMDelivery)


# =============================================================================
# PMPlan & PMDelivery models
# =============================================================================


class TestPMModels:
    def test_pm_plan(self) -> None:
        plan = PMPlan(
            project_summary="Test",
            objectives=[
                ObjectiveSpec(
                    objective_id="o1",
                    title="Task 1",
                    description="Do something",
                )
            ],
            human_questions=["What's the deadline?"],
            risks=["Scope creep"],
        )
        assert len(plan.objectives) == 1
        assert len(plan.human_questions) == 1

    def test_pm_delivery(self) -> None:
        delivery = PMDelivery(
            project_title="Todo API",
            summary="Built successfully",
            deliverables=[
                Artifact(name="api.py", content_type="python", content="code")
            ],
            quality_assessment="High",
            open_issues=["No auth yet"],
            recommendations=["Add JWT"],
            metrics={"total_tokens": 1000},
        )
        assert delivery.project_title == "Todo API"
        assert len(delivery.deliverables) == 1
