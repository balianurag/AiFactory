"""
Tests for the schema models.
"""

from __future__ import annotations

from aifactory.core.schemas import (
    AgentConfig,
    AgentPerformanceRecord,
    AgentRole,
    Artifact,
    MemoryEntry,
    MemoryLevel,
    MessageType,
    ProjectPlan,
    TaskAssignment,
    TaskResult,
    TaskStatus,
)


def test_task_assignment_creation() -> None:
    """TaskAssignment should create with valid defaults."""
    msg = TaskAssignment(
        from_agent="pm_001",
        to_agent="worker_001",
        title="Build auth module",
        description="Implement JWT-based authentication",
        acceptance_criteria=["Tokens expire after 1h", "Refresh token support"],
    )
    assert msg.type == MessageType.TASK_ASSIGNMENT
    assert msg.title == "Build auth module"
    assert len(msg.acceptance_criteria) == 2
    assert msg.task_id  # Should auto-generate


def test_task_result_creation() -> None:
    """TaskResult should serialize cleanly."""
    result = TaskResult(
        from_agent="worker_001",
        to_agent="tl_001",
        task_id="task_abc",
        status=TaskStatus.COMPLETED,
        summary="Built the auth module with JWT",
        artifacts=[
            Artifact(
                name="auth.py",
                content_type="python",
                content="def login(): ...",
                summary="Authentication module",
            )
        ],
    )
    data = result.model_dump()
    assert data["status"] == "completed"
    assert len(data["artifacts"]) == 1


def test_agent_config_defaults() -> None:
    """AgentConfig should have sensible defaults."""
    config = AgentConfig(
        role=AgentRole.WORKER,
        name="Test Worker",
        model="google/gemini-2.5-flash",
    )
    assert config.agent_id  # Auto-generated
    assert config.temperature == 0.3
    assert config.max_iterations == 15


def test_performance_record_composite_score() -> None:
    """Composite score should weight metrics correctly."""
    record = AgentPerformanceRecord(
        agent_id="agent_001",
        task_id="task_001",
        role=AgentRole.WORKER,
        model="google/gemini-2.5-flash",
        correctness=8.0,
        completeness=7.0,
        edge_case_handling=6.0,
        communication_quality=9.0,
        efficiency=8.0,
    )
    score = record.composite_score
    # 8*0.3 + 7*0.2 + 6*0.15 + 9*0.1 + 8*0.15 = 2.4+1.4+0.9+0.9+1.2 = 6.8
    assert abs(score - 6.8) < 0.01


def test_performance_with_peer_scores() -> None:
    """Peer scores should contribute 10% to composite."""
    record = AgentPerformanceRecord(
        agent_id="agent_001",
        task_id="task_001",
        role=AgentRole.WORKER,
        model="test",
        correctness=8.0,
        completeness=8.0,
        edge_case_handling=8.0,
        communication_quality=8.0,
        efficiency=8.0,
        peer_judge_scores=[9.0, 8.5],
    )
    score = record.composite_score
    assert score > 7.0  # Should be boosted by peer scores


def test_memory_entry_defaults() -> None:
    """MemoryEntry should have correct defaults."""
    entry = MemoryEntry(
        level=MemoryLevel.AGENT,
        content="Test memory entry",
    )
    assert entry.id  # Auto-generated
    assert entry.entry_type == "observation"
    assert entry.promoted is False
    assert entry.turn_number == 0


def test_project_plan_serialization() -> None:
    """ProjectPlan should round-trip through JSON."""
    plan = ProjectPlan(
        title="Test Project",
        description="A test project",
        complexity="medium",
        estimated_teams=1,
        risks=["timeline risk"],
        success_criteria=["works"],
    )
    json_str = plan.model_dump_json()
    restored = ProjectPlan.model_validate_json(json_str)
    assert restored.title == "Test Project"
    assert restored.complexity == "medium"
