"""
Single-Team LangGraph Pipeline — The MVP execution graph.

This is the Phase 1 graph with an intelligent complexity gate:
  [START] → assess_task → (trivial? → direct_worker | full_pipeline) → [END]

Trivial tasks skip PM + TL overhead and go straight to a single Worker.
Moderate/complex tasks use the full hierarchy: PM → Team Lead → Workers.

State flows through a typed dict that accumulates results at each stage.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from aifactory.agents.project_manager import PMDelivery, ProjectManagerAgent
from aifactory.agents.team_lead import TeamLeadAgent
from aifactory.agents.worker import WorkerAgent
from aifactory.config.settings import get_settings
from aifactory.core.assessor import (
    AssessmentResult,
    TaskComplexity,
    assess_complexity,
    cost_usd,
)
from aifactory.core.memory_router import MemoryRouter
from aifactory.core.schemas import (
    AgentConfig,
    AgentRole,
    WorkerSpecialty,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Graph State
# =============================================================================


class PipelineState(TypedDict, total=False):
    """State that flows through the LangGraph pipeline."""

    # Input
    task: str
    context: str

    # Assessment gate
    assessment: AssessmentResult | None
    route: str  # "trivial" | "full"

    # Runtime
    pm: ProjectManagerAgent | None
    team_leads: list[TeamLeadAgent]
    workers: list[list[WorkerAgent]]

    # Output
    delivery: PMDelivery | None
    status: str
    error: str | None

    # Metrics
    total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float


# =============================================================================
# Pipeline Builder
# =============================================================================


class SingleTeamPipeline:
    """
    Builds and runs a single-team LangGraph pipeline with complexity gating.

    Flow:
      1. assess_task → classify complexity (cheap gpt-4o-mini call)
      2. Route:
         - trivial  → direct_worker (single Worker, no PM/TL)
         - moderate/complex → setup_agents → execute_project (PM → TL → Workers)
    """

    def __init__(
        self,
        num_workers: int = 3,
        worker_specialties: list[WorkerSpecialty] | None = None,
        memory: MemoryRouter | None = None,
        pm_model: str | None = None,
        worker_model: str | None = None,
        skip_assessment: bool = False,
    ) -> None:
        self.settings = get_settings()
        self.memory = memory or MemoryRouter()
        self.num_workers = num_workers
        self.worker_specialties = worker_specialties or [
            WorkerSpecialty.CODER,
            WorkerSpecialty.RESEARCHER,
            WorkerSpecialty.WRITER,
        ]
        # Model overrides — fall back to settings defaults
        self.pm_model = pm_model or self.settings.premium_model
        self.worker_model = worker_model or self.settings.budget_model
        self.skip_assessment = skip_assessment

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph state machine with complexity routing."""
        graph = StateGraph(PipelineState)

        # Add nodes
        graph.add_node("assess_task", self._assess_task)
        graph.add_node("direct_worker", self._direct_worker)
        graph.add_node("setup_agents", self._setup_agents)
        graph.add_node("execute_project", self._execute_project)

        # Wire edges — assess first, then route
        graph.add_edge(START, "assess_task")
        graph.add_conditional_edges(
            "assess_task",
            self._route_by_complexity,
            {"trivial": "direct_worker", "full": "setup_agents"},
        )
        graph.add_edge("direct_worker", END)
        graph.add_edge("setup_agents", "execute_project")
        graph.add_edge("execute_project", END)

        return graph

    def compile(self):
        """Compile the graph into a runnable."""
        return self.graph.compile()

    async def run(self, task: str, context: str = "") -> PMDelivery:
        """
        Execute the full pipeline.

        Args:
            task: The project / task description.
            context: Additional context.

        Returns:
            PMDelivery with all results.
        """
        app = self.compile()

        initial_state: PipelineState = {
            "task": task,
            "context": context,
            "assessment": None,
            "route": "",
            "pm": None,
            "team_leads": [],
            "workers": [],
            "delivery": None,
            "status": "starting",
            "error": None,
            "total_tokens": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
        }

        result = await app.ainvoke(initial_state)

        if result.get("error"):
            logger.error("Pipeline failed: %s", result["error"])
            raise RuntimeError(result["error"])

        delivery = result.get("delivery")
        if delivery is None:
            raise RuntimeError("Pipeline completed but no delivery was produced")

        return delivery

    # ─── Graph Nodes ──────────────────────────────────────────────────

    async def _assess_task(self, state: PipelineState) -> dict[str, Any]:
        """Classify task complexity and decide routing."""
        if self.skip_assessment:
            # Skip assessment — go straight to full pipeline (for tests)
            return {
                "assessment": AssessmentResult(
                    complexity=TaskComplexity.MODERATE,
                    reasoning="Assessment skipped",
                ),
                "route": "full",
                "status": "assessed",
            }

        assessment = await assess_complexity(state["task"])

        route = "trivial" if assessment.complexity == TaskComplexity.TRIVIAL else "full"

        logger.info(
            "Task assessed: %s → route=%s (reason: %s)",
            assessment.complexity.value,
            route,
            assessment.reasoning,
        )

        return {
            "assessment": assessment,
            "route": route,
            "status": "assessed",
        }

    @staticmethod
    def _route_by_complexity(state: PipelineState) -> Literal["trivial", "full"]:
        """Conditional edge: decide which path to take after assessment."""
        route = state.get("route", "full")
        return "trivial" if route == "trivial" else "full"

    async def _direct_worker(self, state: PipelineState) -> dict[str, Any]:
        """
        Trivial-task shortcut: a single Worker, no PM or TL overhead.

        This is the fast path for simple tasks — saves ~60-70% of tokens.
        """
        assessment = state.get("assessment")
        specialty = WorkerSpecialty.CODER
        if assessment and assessment.suggested_specialties:
            try:
                specialty = WorkerSpecialty(assessment.suggested_specialties[0])
            except ValueError:
                specialty = WorkerSpecialty.CODER

        worker_config = AgentConfig(
            role=AgentRole.WORKER,
            name="Worker-Direct",
            model=self.worker_model,
            specialty=specialty,
            team_id="direct",
            max_iterations=10,
            token_budget=8000,
            temperature=0.2,
        )
        worker = WorkerAgent(config=worker_config, memory=self.memory)

        try:
            result = await worker.execute_task(state["task"], context=state.get("context", ""))

            # Calculate cost
            total_in = worker.total_input_tokens
            total_out = worker.total_output_tokens
            total_cost = cost_usd(worker.config.model, total_in, total_out)

            delivery = PMDelivery(
                project_title=state["task"][:100],
                summary=result.summary,
                deliverables=result.artifacts,
                quality_assessment=f"Direct worker execution (trivial task). "
                f"Assessment: {assessment.reasoning if assessment else 'N/A'}",
                open_issues=result.issues_found,
                recommendations=[],
                metrics={
                    "total_iterations": worker.iteration,
                    "total_tokens": worker.total_tokens_used,
                    "total_input_tokens": total_in,
                    "total_output_tokens": total_out,
                    "teams_used": 0,
                    "agents_used": 1,
                },
            )

            return {
                "delivery": delivery,
                "workers": [[worker]],
                "status": "completed",
                "total_tokens": worker.total_tokens_used,
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
                "total_cost_usd": total_cost,
            }

        except Exception as e:
            logger.error("Direct worker failed: %s", e, exc_info=True)
            return {"error": str(e), "status": "failed"}

    async def _setup_agents(self, state: PipelineState) -> dict[str, Any]:
        """Create the agent hierarchy: PM → TL → Workers."""
        settings = self.settings

        # Use assessment to tune worker count
        assessment = state.get("assessment")
        num_workers = self.num_workers
        if assessment and assessment.complexity == TaskComplexity.MODERATE:
            num_workers = min(num_workers, max(1, assessment.suggested_workers))

        team_id = "team_alpha"

        # Create workers
        workers: list[WorkerAgent] = []
        for i in range(num_workers):
            specialty = self.worker_specialties[i % len(self.worker_specialties)]
            worker_config = AgentConfig(
                role=AgentRole.WORKER,
                name=f"Worker-{specialty.value.title()}-{i + 1}",
                model=self.worker_model,
                specialty=specialty,
                team_id=team_id,
                max_iterations=10,
                token_budget=8000,
                temperature=0.2,
            )
            worker = WorkerAgent(config=worker_config, memory=self.memory)
            workers.append(worker)

        # Create Team Lead
        tl_config = AgentConfig(
            role=AgentRole.TEAM_LEAD,
            name="Team-Lead-Alpha",
            model=settings.standard_model,
            team_id=team_id,
            max_iterations=10,
            token_budget=12000,
            temperature=0.3,
        )
        tl = TeamLeadAgent(config=tl_config, memory=self.memory)
        for w in workers:
            tl.add_worker(w)
            w.config.supervisor_id = tl.config.agent_id

        # Create PM
        pm_config = AgentConfig(
            role=AgentRole.PROJECT_MANAGER,
            name="PM-Main",
            model=self.pm_model,
            max_iterations=8,
            token_budget=16000,
            temperature=0.3,
        )
        pm = ProjectManagerAgent(config=pm_config, memory=self.memory)
        pm.add_team_lead(tl)
        tl.config.supervisor_id = pm.config.agent_id

        logger.info(
            "Pipeline setup: PM=%s, TL=%s, Workers=%s",
            pm.config.name,
            tl.config.name,
            [w.config.name for w in workers],
        )

        return {
            "pm": pm,
            "team_leads": [tl],
            "workers": [workers],
            "status": "agents_ready",
        }

    async def _execute_project(self, state: PipelineState) -> dict[str, Any]:
        """Run the PM's full project execution with cost tracking."""
        pm = state["pm"]
        if pm is None:
            return {"error": "PM not initialized", "status": "failed"}

        task = state["task"]
        context = state.get("context", "")

        try:
            delivery = await pm.execute_project(
                project_description=task,
                context=context,
            )

            # Calculate total tokens and cost across all agents
            total_tokens = 0
            total_in = 0
            total_out = 0
            total_cost = 0.0

            # PM
            total_tokens += pm.total_tokens_used
            total_in += pm.total_input_tokens
            total_out += pm.total_output_tokens
            total_cost += cost_usd(pm.config.model, pm.total_input_tokens, pm.total_output_tokens)

            # TLs + Workers
            for tl in state.get("team_leads", []):
                total_tokens += tl.total_tokens_used
                total_in += tl.total_input_tokens
                total_out += tl.total_output_tokens
                total_cost += cost_usd(
                    tl.config.model, tl.total_input_tokens, tl.total_output_tokens
                )
                for w in tl.workers:
                    total_tokens += w.total_tokens_used
                    total_in += w.total_input_tokens
                    total_out += w.total_output_tokens
                    total_cost += cost_usd(
                        w.config.model, w.total_input_tokens, w.total_output_tokens
                    )

            return {
                "delivery": delivery,
                "status": "completed",
                "total_tokens": total_tokens,
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
                "total_cost_usd": total_cost,
            }

        except Exception as e:
            logger.error("Project execution failed: %s", e, exc_info=True)
            return {
                "error": str(e),
                "status": "failed",
            }


# =============================================================================
# Convenience function
# =============================================================================


async def run_single_team(
    task: str,
    context: str = "",
    num_workers: int = 3,
    worker_specialties: list[WorkerSpecialty] | None = None,
    pm_model: str | None = None,
    worker_model: str | None = None,
) -> PMDelivery:
    """
    Quick-start function to run a single-team pipeline.

    Args:
        task:               The project / task description.
        context:            Additional context.
        num_workers:        Number of worker agents.
        worker_specialties: List of worker specialties.
        pm_model:           Override the PM model (must be in allowed list).
        worker_model:       Override the worker model (must be in allowed list).

    Example:
        delivery = await run_single_team("Build a FastAPI CRUD app for a todo list")
    """
    pipeline = SingleTeamPipeline(
        num_workers=num_workers,
        worker_specialties=worker_specialties,
        pm_model=pm_model,
        worker_model=worker_model,
    )
    return await pipeline.run(task=task, context=context)
