"""
Worker Agent — The execution specialist of the AI Factory.

Workers receive specific tasks from their Team Lead and produce artifacts.
Each worker has a specialty (coder, researcher, writer, reviewer, tester, etc.).
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from aifactory.core.base_agent import BaseAgent
from aifactory.core.memory_router import MemoryRouter
from aifactory.core.schemas import (
    AgentConfig,
    Artifact,
    TaskAssignment,
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "worker.md"


# ─── Structured Output Models ────────────────────────────────────────


class WorkerOutput(BaseModel):
    """What a worker produces for each task."""

    summary: str = Field(description="One-line summary of what was produced")
    artifacts: list[Artifact] = Field(default_factory=list)
    issues_found: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.COMPLETED


# ─── Worker Agent ─────────────────────────────────────────────────────


class WorkerAgent(BaseAgent):
    """
    Execution agent that produces concrete deliverables.

    Specialties:
      - coder: writes code, scripts, configs
      - researcher: analyses, summarises, finds information
      - writer: documentation, reports, copy
      - reviewer: code review, quality checks
      - tester: test cases, test plans, validation
    """

    def __init__(
        self,
        config: AgentConfig,
        memory: MemoryRouter | None = None,
    ) -> None:
        # Inject the worker system prompt if not already set
        if not config.system_prompt:
            if _PROMPT_PATH.exists():
                config.system_prompt = _PROMPT_PATH.read_text()
            else:
                config.system_prompt = (
                    "You are a worker agent. Execute tasks precisely "
                    "and produce high-quality artifacts."
                )

        super().__init__(config=config, memory=memory)

    async def execute_task(
        self,
        task: TaskAssignment | str,
        context: str = "",
    ) -> TaskResult:
        """
        Execute a task and return a structured TaskResult.

        Args:
            task: Either a TaskAssignment message or a plain string description.
            context: Additional context from the team/project.

        Returns:
            TaskResult with artifacts and status.
        """
        # Normalise input
        if isinstance(task, str):
            task_desc = task
            task_id = "inline_task"
        else:
            task_desc = (
                f"## Task: {task.title}\n\n"
                f"{task.description}\n\n"
                f"### Acceptance Criteria\n"
                + "\n".join(f"- {c}" for c in task.acceptance_criteria)
            )
            task_id = task.task_id
            if task.context:
                context = task.context + ("\n\n" + context if context else "")

        # Build the prompt
        prompt = (
            f"{task_desc}\n\n"
            "Respond with valid JSON matching the WorkerOutput schema:\n"
            "{\n"
            '  "summary": "one-line summary",\n'
            '  "artifacts": [{"name": "...", "content_type": "...", '
            '"content": "...", "summary": "..."}],\n'
            '  "issues_found": ["..."],\n'
            '  "status": "completed|needs_review|blocked"\n'
            "}"
        )

        # Call the LLM
        try:
            output = await self.think(
                user_message=prompt,
                response_model=WorkerOutput,
                extra_context=context,
            )
            assert isinstance(output, WorkerOutput)
        except Exception as e:
            logger.error("Worker %s failed on task %s: %s", self.config.name, task_id, e)
            # Return a failure result
            return TaskResult(
                from_agent=self.config.agent_id,
                to_agent=self.config.supervisor_id or "",
                team_id=self.config.team_id,
                task_id=task_id,
                status=TaskStatus.FAILED,
                summary=f"Worker failed: {e}",
                artifacts=[],
                issues_found=[str(e)],
                tokens_used=self.total_tokens_used,
                iterations_used=self.iteration,
            )

        # Build TaskResult
        result = TaskResult(
            from_agent=self.config.agent_id,
            to_agent=self.config.supervisor_id or "",
            team_id=self.config.team_id,
            task_id=task_id,
            status=output.status,
            summary=output.summary,
            artifacts=output.artifacts,
            issues_found=output.issues_found,
            tokens_used=self.total_tokens_used,
            iterations_used=self.iteration,
        )

        # Store the result as a decision in memory
        self.store_memory(
            content=f"Task '{task_id}' completed: {output.summary}. "
                    f"Artifacts: {len(output.artifacts)}. Issues: {output.issues_found}",
            entry_type="decision",
        )

        logger.info(
            "Worker %s completed task %s — status=%s, artifacts=%d, issues=%d",
            self.config.name, task_id, output.status.value,
            len(output.artifacts), len(output.issues_found),
        )

        return result
