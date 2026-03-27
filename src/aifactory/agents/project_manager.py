"""
Project Manager Agent — The strategic coordinator of the AI Factory.

PMs receive a project plan from the Assessment Team, break it into team objectives,
delegate to Team Leads, and consolidate the final output.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from aifactory.agents.team_lead import TeamLeadAgent
from aifactory.core.base_agent import BaseAgent
from aifactory.core.memory_router import MemoryRouter
from aifactory.core.schemas import (
    AgentConfig,
    Artifact,
    MemoryLevel,
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "pm.md"


# ─── Structured Output Models ────────────────────────────────────────


class ObjectiveSpec(BaseModel):
    """A single objective to assign to a Team Lead."""

    objective_id: str
    title: str
    description: str
    success_criteria: list[str] = Field(default_factory=list)
    priority: str = "high"
    iteration_budget: int = 15


class PMPlan(BaseModel):
    """PM's breakdown of the project into team objectives."""

    project_summary: str
    objectives: list[ObjectiveSpec]
    cross_team_dependencies: str = ""
    human_questions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class PMDelivery(BaseModel):
    """PM's final consolidated delivery."""

    project_title: str
    summary: str
    deliverables: list[Artifact] = Field(default_factory=list)
    quality_assessment: str = ""
    open_issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    metrics: dict[str, int] = Field(default_factory=dict)


# ─── Project Manager Agent ───────────────────────────────────────────


class ProjectManagerAgent(BaseAgent):
    """
    Strategic coordinator that manages Team Leads.

    Lifecycle:
      1. Receive project plan / task description
      2. Break into team objectives (via LLM)
      3. Assign to Team Leads
      4. Collect and consolidate results
      5. Deliver final output
    """

    def __init__(
        self,
        config: AgentConfig,
        memory: MemoryRouter | None = None,
        team_leads: list[TeamLeadAgent] | None = None,
    ) -> None:
        if not config.system_prompt:
            if _PROMPT_PATH.exists():
                config.system_prompt = _PROMPT_PATH.read_text()
            else:
                config.system_prompt = "You are a Project Manager. Plan, delegate, and deliver."

        super().__init__(config=config, memory=memory)
        self.team_leads = team_leads or []

    def add_team_lead(self, tl: TeamLeadAgent) -> None:
        """Add a Team Lead to this PM's roster."""
        self.team_leads.append(tl)
        if tl.config.agent_id not in self.config.subordinate_ids:
            self.config.subordinate_ids.append(tl.config.agent_id)

    async def execute_project(
        self,
        project_description: str,
        context: str = "",
    ) -> PMDelivery:
        """
        Full project lifecycle: plan → delegate → consolidate → deliver.

        Args:
            project_description: What needs to be built/done.
            context: Additional context (from Assessment Team or human).

        Returns:
            PMDelivery with all deliverables.
        """
        logger.info("PM %s starting project: %s...", self.config.name, project_description[:80])

        # ── Step 1: Plan ──────────────────────────────────────────────
        plan = await self._create_plan(project_description, context)

        # Log human questions if any
        if plan.human_questions:
            logger.warning(
                "PM %s has questions for the human: %s",
                self.config.name,
                plan.human_questions,
            )
            # In the full system, these would be routed to the human.
            # For MVP, we continue without answers.

        # ── Step 2: Delegate to Team Leads ────────────────────────────
        team_results = await self._delegate(plan, context)

        # ── Step 3: Consolidate & Deliver ─────────────────────────────
        delivery = await self._consolidate(project_description, plan, team_results)

        # Store project-level summary
        self.store_memory(
            content=f"Project delivered: {delivery.summary}. "
            f"Quality: {delivery.quality_assessment}. "
            f"Open issues: {delivery.open_issues}",
            entry_type="decision",
            level=MemoryLevel.TEAM,
        )

        logger.info(
            "PM %s delivered project — deliverables=%d, open_issues=%d",
            self.config.name,
            len(delivery.deliverables),
            len(delivery.open_issues),
        )

        return delivery

    # ─── Internal Steps ───────────────────────────────────────────────

    async def _create_plan(self, project_description: str, context: str) -> PMPlan:
        """Use LLM to break the project into team objectives."""
        team_info = f"Available Team Leads: {len(self.team_leads)}"
        for i, tl in enumerate(self.team_leads):
            worker_specs = [w.config.specialty.value for w in tl.workers if w.config.specialty]
            team_info += (
                f"\n  Team {i + 1} ({tl.config.name}): "
                f"workers={len(tl.workers)}, "
                f"specialties={worker_specs}"
            )

        prompt = (
            f"## Project\n{project_description}\n\n"
            f"## Resources\n{team_info}\n\n"
            "Create a project plan. Break into 1-3 objectives (one per Team Lead).\n"
            "Each objective should be independently executable.\n\n"
            "Respond with valid JSON matching PMPlan:\n"
            "{\n"
            '  "project_summary": "...",\n'
            '  "objectives": [{"objective_id": "obj_1", '
            '"title": "...", "description": "...", '
            '"success_criteria": ["..."], '
            '"priority": "high|medium|low", '
            '"iteration_budget": 15}],\n'
            '  "cross_team_dependencies": "...",\n'
            '  "human_questions": ["questions for human, if any"],\n'
            '  "risks": ["known risks"]\n'
            "}"
        )

        try:
            plan = await self.think(
                user_message=prompt,
                response_model=PMPlan,
                extra_context=context,
            )
            assert isinstance(plan, PMPlan)
        except Exception as e:
            logger.warning("PM %s plan creation failed, using fallback: %s", self.config.name, e)
            plan = PMPlan(
                project_summary=project_description[:200],
                objectives=[
                    ObjectiveSpec(
                        objective_id="obj_1",
                        title="Execute project",
                        description=project_description,
                        success_criteria=["Complete the project as described"],
                    )
                ],
            )

        logger.info(
            "PM %s planned %d objectives: %s",
            self.config.name,
            len(plan.objectives),
            [o.title for o in plan.objectives],
        )
        return plan

    async def _delegate(self, plan: PMPlan, context: str) -> list[TaskResult]:
        """Assign objectives to Team Leads and collect results."""
        results: list[TaskResult] = []

        for i, objective in enumerate(plan.objectives):
            if i >= len(self.team_leads):
                logger.warning(
                    "PM %s has more objectives (%d) than Team Leads (%d). "
                    "Assigning extra to last TL.",
                    self.config.name,
                    len(plan.objectives),
                    len(self.team_leads),
                )
                tl = self.team_leads[-1]
            else:
                tl = self.team_leads[i]

            objective_text = (
                f"## Objective: {objective.title}\n\n"
                f"{objective.description}\n\n"
                f"### Success Criteria\n" + "\n".join(f"- {c}" for c in objective.success_criteria)
            )

            full_context = (
                f"Project: {plan.project_summary}\n"
                f"Cross-team dependencies: {plan.cross_team_dependencies}\n"
                f"Risks: {', '.join(plan.risks)}"
            )
            if context:
                full_context = context + "\n\n" + full_context

            logger.info(
                "PM %s delegating '%s' to TL %s",
                self.config.name,
                objective.title,
                tl.config.name,
            )

            result = await tl.execute_objective(
                objective=objective_text,
                context=full_context,
            )
            results.append(result)

        return results

    async def _consolidate(
        self,
        project_description: str,
        plan: PMPlan,
        results: list[TaskResult],
    ) -> PMDelivery:
        """Merge all team results into the final delivery."""
        # Collect all artifacts and issues
        all_artifacts = []
        all_issues = []
        total_tokens = self.total_tokens_used

        team_summaries = []
        for i, r in enumerate(results):
            team_summaries.append(
                f"- Team {i + 1} [{r.status.value}]: {r.summary} "
                f"(artifacts: {len(r.artifacts)}, "
                f"issues: {len(r.issues_found)}, "
                f"tokens: {r.tokens_used})"
            )
            all_artifacts.extend(r.artifacts)
            all_issues.extend(r.issues_found)
            total_tokens += r.tokens_used

        # Simple merge for single-team projects
        if len(results) == 1 and results[0].status == TaskStatus.COMPLETED:
            return PMDelivery(
                project_title=plan.project_summary[:100],
                summary=results[0].summary,
                deliverables=all_artifacts,
                quality_assessment="Single team execution — direct pass-through.",
                open_issues=all_issues,
                recommendations=[],
                metrics={
                    "total_iterations": self.iteration + sum(r.iterations_used for r in results),
                    "total_tokens": total_tokens,
                    "teams_used": len(results),
                    "agents_used": 1
                    + sum(1 + len(tl.workers) for tl in self.team_leads[: len(results)]),
                },
            )

        # LLM consolidation for multi-team projects
        artifact_summaries = [
            f"- **{a.name}** ({a.content_type}): {a.summary or a.content[:100]}"
            for a in all_artifacts
        ]

        prompt = (
            f"## Project\n{project_description[:500]}\n\n"
            f"## Team Results\n" + "\n".join(team_summaries) + "\n\n"
            "## Artifacts\n" + "\n".join(artifact_summaries) + "\n\n"
            "## Issues\n" + "\n".join(f"- {i}" for i in all_issues) + "\n\n"
            "Create the final project delivery. Merge and assess quality.\n\n"
            "Respond with valid JSON matching PMDelivery:\n"
            "{\n"
            '  "project_title": "...",\n'
            '  "summary": "...",\n'
            '  "deliverables": [{"name": "...", '
            '"content_type": "...", "content": "...", '
            '"summary": "..."}],\n'
            '  "quality_assessment": "...",\n'
            '  "open_issues": ["..."],\n'
            '  "recommendations": ["..."],\n'
            '  "metrics": {"total_iterations": 0, '
            '"total_tokens": 0, "teams_used": 0, '
            '"agents_used": 0}\n'
            "}"
        )

        try:
            delivery = await self.think(
                user_message=prompt,
                response_model=PMDelivery,
            )
            assert isinstance(delivery, PMDelivery)
            # Fill in real metrics
            delivery.metrics = {
                "total_iterations": self.iteration + sum(r.iterations_used for r in results),
                "total_tokens": total_tokens,
                "teams_used": len(results),
                "agents_used": 1
                + sum(1 + len(tl.workers) for tl in self.team_leads[: len(results)]),
            }
            return delivery
        except Exception as e:
            logger.warning("PM consolidation failed, using raw merge: %s", e)
            return PMDelivery(
                project_title=plan.project_summary[:100],
                summary=f"Merged {len(results)} team results. {all_issues or 'No issues.'}",
                deliverables=all_artifacts,
                quality_assessment="Auto-merged (consolidation LLM failed)",
                open_issues=all_issues,
                metrics={
                    "total_iterations": self.iteration,
                    "total_tokens": total_tokens,
                    "teams_used": len(results),
                    "agents_used": 1 + len(self.team_leads),
                },
            )
