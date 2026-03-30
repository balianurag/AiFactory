"""
Single-Team LangGraph Pipeline — The MVP execution graph.

This is the Phase 1 graph: PM → Team Lead → Workers.
It proves the core loop works end-to-end before we add dynamic hierarchy.

Graph Structure:
  [START] → plan_project → delegate_to_teams → execute_team → consolidate → [END]

State flows through a typed dict that accumulates results at each stage.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from aifactory.agents.project_manager import PMDelivery, ProjectManagerAgent
from aifactory.agents.team_lead import TeamLeadAgent
from aifactory.agents.worker import WorkerAgent
from aifactory.config.settings import get_settings
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
    total_cost_usd: float


# =============================================================================
# Pipeline Builder
# =============================================================================


class SingleTeamPipeline:
    """
    Builds and runs a single-team LangGraph pipeline.

    This is the MVP: 1 PM → 1 Team Lead → N Workers.
    Use this to prove the core agent loop works before scaling to multi-team.
    """

    def __init__(
        self,
        num_workers: int = 3,
        worker_specialties: list[WorkerSpecialty] | None = None,
        memory: MemoryRouter | None = None,
        pm_model: str | None = None,
        worker_model: str | None = None,
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

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph state machine."""
        graph = StateGraph(PipelineState)

        # Add nodes
        graph.add_node("setup_agents", self._setup_agents)
        graph.add_node("execute_project", self._execute_project)

        # Wire edges
        graph.add_edge(START, "setup_agents")
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
            "pm": None,
            "team_leads": [],
            "workers": [],
            "delivery": None,
            "status": "starting",
            "error": None,
            "total_tokens": 0,
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

    async def _setup_agents(self, state: PipelineState) -> dict[str, Any]:
        """Create the agent hierarchy: PM → TL → Workers."""
        settings = self.settings

        team_id = "team_alpha"

        # Create workers
        workers: list[WorkerAgent] = []
        for i in range(self.num_workers):
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

        # Create Team Lead (uses standard model from settings — not overridable per-TL yet)
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
        """Run the PM's full project execution."""
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

            # Calculate total tokens
            total_tokens = pm.total_tokens_used
            for tl in state.get("team_leads", []):
                total_tokens += tl.total_tokens_used
                for w in tl.workers:
                    total_tokens += w.total_tokens_used

            return {
                "delivery": delivery,
                "status": "completed",
                "total_tokens": total_tokens,
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
