"""
Worker Agent — The execution specialist of the AI Factory.

Workers receive specific tasks from their Team Lead and produce artifacts.
Each worker has a specialty (coder, researcher, writer, reviewer, tester, etc.).

Code artifacts are automatically validated in a sandbox — if execution fails,
the worker self-corrects up to MAX_FIX_ATTEMPTS times.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from aifactory.core.base_agent import BaseAgent
from aifactory.core.memory_router import MemoryRouter
from aifactory.core.sandbox import ExecutionResult, execute_python
from aifactory.core.schemas import (
    AgentConfig,
    Artifact,
    TaskAssignment,
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "worker.md"

MAX_FIX_ATTEMPTS = 2  # How many times to retry after sandbox failure


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

        For coder workers, Python artifacts are validated in the sandbox.
        If execution fails, the worker self-corrects up to MAX_FIX_ATTEMPTS times.

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
                f"### Acceptance Criteria\n" + "\n".join(f"- {c}" for c in task.acceptance_criteria)
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

        # ── Sandbox Validation: run Python artifacts and self-correct ──
        output = await self._validate_and_fix(output, task_desc, context)

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
            self.config.name,
            task_id,
            output.status.value,
            len(output.artifacts),
            len(output.issues_found),
        )

        return result

    # ─── Sandbox Validation ───────────────────────────────────────────

    async def _validate_and_fix(
        self,
        output: WorkerOutput,
        task_desc: str,
        context: str,
    ) -> WorkerOutput:
        """
        Validate Python artifacts in the sandbox. If they fail, ask the LLM
        to fix them (up to MAX_FIX_ATTEMPTS retries).

        Non-Python artifacts and non-coder workers skip validation.
        """
        # Only coder workers validate code
        if self.config.specialty and self.config.specialty.value != "coder":
            return output

        python_artifacts = [a for a in output.artifacts if a.content_type in ("python", "py")]
        if not python_artifacts:
            return output

        for attempt in range(MAX_FIX_ATTEMPTS + 1):
            # Run each Python artifact in the sandbox
            all_passed = True
            sandbox_results: list[tuple[Artifact, ExecutionResult]] = []

            for artifact in python_artifacts:
                exec_result = execute_python(artifact.content, timeout=15)
                sandbox_results.append((artifact, exec_result))

                if exec_result.success:
                    logger.info(
                        "Worker %s: %s passed sandbox ✓",
                        self.config.name,
                        artifact.name,
                    )
                else:
                    all_passed = False
                    logger.warning(
                        "Worker %s: %s failed sandbox (attempt %d/%d): %s",
                        self.config.name,
                        artifact.name,
                        attempt + 1,
                        MAX_FIX_ATTEMPTS + 1,
                        exec_result.error_summary,
                    )

            if all_passed:
                # Add sandbox validation note to issues_found
                if attempt > 0:
                    output.issues_found.append(f"Code fixed after {attempt} sandbox retry(s)")
                return output

            # Out of retries — return with sandbox errors as issues
            if attempt >= MAX_FIX_ATTEMPTS:
                for artifact, exec_result in sandbox_results:
                    if not exec_result.success:
                        output.issues_found.append(
                            f"[sandbox] {artifact.name}: {exec_result.error_summary}"
                        )
                output.status = TaskStatus.NEEDS_REVIEW
                return output

            # ── Self-correct: ask LLM to fix the broken code ──────────
            fix_details = []
            for artifact, exec_result in sandbox_results:
                if not exec_result.success:
                    fix_details.append(
                        f"### {artifact.name}\n"
                        f"**Error:** {exec_result.error_summary}\n"
                        f"**Stderr:**\n```\n{exec_result.stderr[:500]}\n```\n"
                        f"**Current code:**\n```python\n{artifact.content}\n```"
                    )

            fix_prompt = (
                "The code you produced has execution errors. "
                "Fix ONLY the broken artifacts and return the COMPLETE "
                "corrected WorkerOutput JSON.\n\n"
                f"## Original Task\n{task_desc[:300]}\n\n"
                f"## Sandbox Errors\n" + "\n\n".join(fix_details) + "\n\n"
                "Return the FULL corrected JSON with ALL artifacts "
                "(fixed + unchanged ones)."
            )

            try:
                fixed = await self.think(
                    user_message=fix_prompt,
                    response_model=WorkerOutput,
                    extra_context=context,
                )
                assert isinstance(fixed, WorkerOutput)
                output = fixed
                # Update the python_artifacts list for next iteration
                python_artifacts = [
                    a for a in output.artifacts if a.content_type in ("python", "py")
                ]
            except Exception as e:
                logger.warning(
                    "Worker %s fix attempt %d failed: %s",
                    self.config.name,
                    attempt + 1,
                    e,
                )
                output.issues_found.append(f"Self-correction attempt {attempt + 1} failed: {e}")
                return output

        return output
