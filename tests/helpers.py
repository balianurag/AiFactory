"""
Shared test helpers — factory functions for creating mock objects.

Separated from conftest.py so they can be imported explicitly by test files.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from aifactory.core.schemas import (
    Artifact,
    MemoryEntry,
    MemoryLevel,
    TaskAssignment,
    TaskResult,
    TaskStatus,
)

# ─── Mock LLM Response Factory ───────────────────────────────────────


def make_llm_response(
    content: str,
    input_tokens: int = 100,
    output_tokens: int = 200,
) -> MagicMock:
    """Create a mock LiteLLM response object."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = input_tokens
    response.usage.completion_tokens = output_tokens
    return response


def make_json_response(
    data: dict[str, Any],
    input_tokens: int = 100,
    output_tokens: int = 200,
) -> MagicMock:
    """Create a mock LiteLLM response with JSON content."""
    return make_llm_response(
        content=json.dumps(data),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


# ─── Test Data Factories ─────────────────────────────────────────────


def make_task_assignment(
    title: str = "Test Task",
    description: str = "Implement the test feature",
    from_agent: str = "tl_test_001",
    to_agent: str = "worker_test_001",
) -> TaskAssignment:
    """Create a TaskAssignment for testing."""
    return TaskAssignment(
        from_agent=from_agent,
        to_agent=to_agent,
        team_id="team_test",
        title=title,
        description=description,
        acceptance_criteria=["Feature works correctly", "Tests pass"],
        context="This is part of a test project",
        max_iterations=5,
        token_budget=4000,
    )


def make_task_result(
    status: TaskStatus = TaskStatus.COMPLETED,
    from_agent: str = "worker_test_001",
    to_agent: str = "tl_test_001",
) -> TaskResult:
    """Create a TaskResult for testing."""
    return TaskResult(
        from_agent=from_agent,
        to_agent=to_agent,
        team_id="team_test",
        task_id="task_test_001",
        status=status,
        summary="Implemented the test feature successfully",
        artifacts=[
            Artifact(
                name="feature.py",
                content_type="python",
                content="def feature(): return True",
                summary="Main feature implementation",
            )
        ],
        issues_found=[],
        tokens_used=300,
        iterations_used=2,
    )


def make_memory_entry(
    level: MemoryLevel = MemoryLevel.AGENT,
    agent_id: str = "agent_test_001",
    team_id: str = "team_test",
    content: str = "Test memory entry",
    turn_number: int = 1,
) -> MemoryEntry:
    """Create a MemoryEntry for testing."""
    return MemoryEntry(
        level=level,
        agent_id=agent_id,
        team_id=team_id,
        content=content,
        entry_type="observation",
        turn_number=turn_number,
    )
