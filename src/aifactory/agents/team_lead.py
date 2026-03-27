"""
Team Lead Agent — The tactical coordinator of the AI Factory.

Team Leads receive objectives from their PM, decompose them into worker tasks,
delegate, and consolidate the results into a unified deliverable.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from aifactory.agents.worker import WorkerAgent
from aifactory.core.base_agent import BaseAgent
from aifactory.core.memory_router import MemoryRouter
from aifactory.core.schemas import (
    AgentConfig,
    Artifact,
    MemoryLevel,
    TaskAssignment,
    TaskResult,
    TaskStatus,
    WorkerSpecialty,
)

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "team_lead.md"


# ─── Structured Output Models ────────────────────────────────────────


class WorkerTaskSpec(BaseModel):
    """A single task to assign to a worker."""

    title: str
    description: str
    assigned_to_specialty: WorkerSpecialty = WorkerSpecialty.CODER
    acceptance_criteria: list[str] = Field(default_factory=list)
    context: str = ""
    priority: str = "medium"
    max_iterations: int = 5


class DecompositionPlan(BaseModel):
    """Team Lead's plan for breaking down an objective."""

    objective: str
    tasks: list[WorkerTaskSpec]
    dependencies: str = ""
    estimated_total_tokens: int = 10000


class ConsolidatedOutput(BaseModel):
    """Team Lead's merged output from all workers."""

    summary: str
    consolidated_artifacts: list[Artifact] = Field(default_factory=list)
    quality_assessment: str = ""
    issues_and_risks: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.COMPLETED


# ─── Team Lead Agent ──────────────────────────────────────────────────


class TeamLeadAgent(BaseAgent):
    """
    Coordinates a team of workers.

    Lifecycle for each objective:
      1. Receive objective from PM
      2. Decompose into worker tasks (via LLM)
      3. Spawn/assign workers and run them
      4. Collect results
      5. Consolidate into a unified deliverable
      6. Return to PM
    """

    def __init__(
        self,
        config: AgentConfig,
        memory: MemoryRouter | None = None,
        workers: list[WorkerAgent] | None = None,
    ) -> None:
        if not config.system_prompt:
            if _PROMPT_PATH.exists():
                config.system_prompt = _PROMPT_PATH.read_text()
            else:
                config.system_prompt = (
                    "You are a Team Lead. Decompose objectives, "
                    "delegate to workers, and consolidate results."
                )

        super().__init__(config=config, memory=memory)
        self.workers = workers or []

    def add_worker(self, worker: WorkerAgent) -> None:
        """Add a worker to this team."""
        self.workers.append(worker)
        if worker.config.agent_id not in self.config.subordinate_ids:
            self.config.subordinate_ids.append(worker.config.agent_id)

    async def execute_objective(
        self,
        objective: str,
        context: str = "",
    ) -> TaskResult:
        """
        Full lifecycle: decompose → delegate → consolidate.

        Args:
            objective: The team objective to accomplish.
            context: Project-level context from the PM.

        Returns:
            TaskResult with consolidated deliverables.
        """
        logger.info("Team Lead %s starting objective: %s", self.config.name, objective[:80])

        # ── Step 1: Decompose ─────────────────────────────────────────
        plan = await self._decompose(objective, context)

        # ── Step 2: Delegate & Execute ────────────────────────────────
        worker_results = await self._delegate_and_execute(plan, context)

        # ── Step 3: Consolidate ───────────────────────────────────────
        consolidated = await self._consolidate(objective, worker_results)

        # Build final TaskResult
        result = TaskResult(
            from_agent=self.config.agent_id,
            to_agent=self.config.supervisor_id or "",
            team_id=self.config.team_id,
            task_id=f"team_{self.config.team_id}_objective",
            status=consolidated.status,
            summary=consolidated.summary,
            artifacts=consolidated.consolidated_artifacts,
            issues_found=consolidated.issues_and_risks,
            tokens_used=self._total_team_tokens(),
            iterations_used=self.iteration,
        )

        # Store team-level memory
        self.store_memory(
            content=f"Team objective completed: {consolidated.summary}. "
            f"Quality: {consolidated.quality_assessment}. "
            f"Issues: {consolidated.issues_and_risks}",
            entry_type="decision",
            level=MemoryLevel.TEAM,
        )

        logger.info(
            "Team Lead %s completed — status=%s, artifacts=%d",
            self.config.name,
            consolidated.status.value,
            len(consolidated.consolidated_artifacts),
        )

        return result

    # ─── Internal Steps ───────────────────────────────────────────────

    async def _decompose(self, objective: str, context: str) -> DecompositionPlan:
        """Use LLM to break down the objective into worker tasks."""
        available_specialties = [
            w.config.specialty.value for w in self.workers if w.config.specialty
        ]
        worker_info = (
            f"Available workers ({len(self.workers)}): "
            f"{', '.join(available_specialties) or 'general'}"
        )

        prompt = (
            f"## Objective\n{objective}\n\n"
            f"## Team\n{worker_info}\n\n"
            "Decompose this objective into specific tasks for workers.\n"
            "Create 1-3 tasks max. Each must be independently executable.\n\n"
            "Respond with valid JSON matching DecompositionPlan:\n"
            "{\n"
            '  "objective": "...",\n'
            '  "tasks": [{"title": "...", "description": "...", '
            '"assigned_to_specialty": '
            '"coder|researcher|writer|reviewer|tester", '
            '"acceptance_criteria": ["..."], "context": "...", '
            '"priority": "high|medium|low", '
            '"max_iterations": 5}],\n'
            '  "dependencies": "...",\n'
            '  "estimated_total_tokens": 10000\n'
            "}"
        )

        try:
            plan = await self.think(
                user_message=prompt,
                response_model=DecompositionPlan,
                extra_context=context,
            )
            assert isinstance(plan, DecompositionPlan)
        except Exception as e:
            logger.warning(
                "Decomposition failed for TL %s, using fallback: %s",
                self.config.name,
                e,
            )
            plan = DecompositionPlan(
                objective=objective,
                tasks=[
                    WorkerTaskSpec(
                        title="Execute objective",
                        description=objective,
                        acceptance_criteria=["Complete the objective as described"],
                    )
                ],
            )

        logger.info(
            "TL %s decomposed into %d tasks: %s",
            self.config.name,
            len(plan.tasks),
            [t.title for t in plan.tasks],
        )
        return plan

    async def _delegate_and_execute(
        self,
        plan: DecompositionPlan,
        context: str,
    ) -> list[TaskResult]:
        """Assign tasks to workers and collect results."""
        results: list[TaskResult] = []

        for i, task_spec in enumerate(plan.tasks):
            # Pick the best worker for this task
            worker = self._pick_worker(task_spec.assigned_to_specialty, i)

            # Build TaskAssignment
            assignment = TaskAssignment(
                from_agent=self.config.agent_id,
                to_agent=worker.config.agent_id,
                team_id=self.config.team_id,
                title=task_spec.title,
                description=task_spec.description,
                acceptance_criteria=task_spec.acceptance_criteria,
                context=task_spec.context or context,
                max_iterations=task_spec.max_iterations,
            )

            logger.info(
                "TL %s assigning '%s' to worker %s",
                self.config.name,
                task_spec.title,
                worker.config.name,
            )

            # Execute
            result = await worker.execute_task(task=assignment, context=context)
            results.append(result)

        return results

    async def _consolidate(
        self,
        objective: str,
        results: list[TaskResult],
    ) -> ConsolidatedOutput:
        """Merge all worker results into a unified deliverable."""
        # Build a summary of what each worker produced
        worker_summaries = []
        all_artifacts = []
        all_issues = []

        for r in results:
            worker_summaries.append(
                f"- [{r.status.value}] {r.summary} "
                f"(artifacts: {len(r.artifacts)}, "
                f"issues: {len(r.issues_found)})"
            )
            all_artifacts.extend(r.artifacts)
            all_issues.extend(r.issues_found)

        # If all workers succeeded with simple output, skip LLM consolidation
        all_completed = all(r.status == TaskStatus.COMPLETED for r in results)

        if all_completed and len(results) == 1:
            return ConsolidatedOutput(
                summary=results[0].summary,
                consolidated_artifacts=all_artifacts,
                quality_assessment="Single worker task — passed through directly.",
                issues_and_risks=all_issues,
                status=TaskStatus.COMPLETED,
            )

        # Use LLM to merge
        artifact_previews = []
        for a in all_artifacts:
            preview = a.content[:300] + "..." if len(a.content) > 300 else a.content
            artifact_previews.append(f"**{a.name}** ({a.content_type}): {preview}")

        prompt = (
            f"## Objective\n{objective}\n\n"
            f"## Worker Results\n" + "\n".join(worker_summaries) + "\n\n"
            "## Artifacts Produced\n" + "\n\n".join(artifact_previews) + "\n\n"
            "## Issues Found\n" + "\n".join(f"- {i}" for i in all_issues) + "\n\n"
            "Consolidate these results into a unified deliverable.\n"
            "Merge artifacts where needed, assess quality, flag remaining issues.\n\n"
            "Respond with valid JSON matching ConsolidatedOutput:\n"
            "{\n"
            '  "summary": "...",\n'
            '  "consolidated_artifacts": [{"name": "...", '
            '"content_type": "...", "content": "...", '
            '"summary": "..."}],\n'
            '  "quality_assessment": "...",\n'
            '  "issues_and_risks": ["..."],\n'
            '  "recommendations": ["..."],\n'
            '  "status": "completed|needs_review|revision_needed"\n'
            "}"
        )

        try:
            output = await self.think(
                user_message=prompt,
                response_model=ConsolidatedOutput,
            )
            assert isinstance(output, ConsolidatedOutput)
            # If consolidated artifacts are empty, use the raw ones
            if not output.consolidated_artifacts:
                output.consolidated_artifacts = all_artifacts
            return output
        except Exception as e:
            logger.warning("Consolidation LLM failed, using raw merge: %s", e)
            return ConsolidatedOutput(
                summary=f"Merged {len(results)} worker results for: {objective[:60]}",
                consolidated_artifacts=all_artifacts,
                quality_assessment="Auto-merged (consolidation LLM failed)",
                issues_and_risks=all_issues,
                status=TaskStatus.NEEDS_REVIEW if all_issues else TaskStatus.COMPLETED,
            )

    def _pick_worker(self, specialty: WorkerSpecialty, task_index: int) -> WorkerAgent:
        """Pick the best available worker for a task."""
        # Try to match specialty
        for w in self.workers:
            if w.config.specialty == specialty:
                return w

        # Fallback: round-robin
        if self.workers:
            return self.workers[task_index % len(self.workers)]

        raise RuntimeError(f"Team Lead {self.config.name} has no workers!")

    def _total_team_tokens(self) -> int:
        """Sum of all tokens used by this TL + workers."""
        total = self.total_tokens_used
        for w in self.workers:
            total += w.total_tokens_used
        return total
