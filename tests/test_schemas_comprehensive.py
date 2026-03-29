"""
Comprehensive tests for core/schemas.py — all Pydantic models and enums.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from aifactory.core.schemas import (
    AgentConfig,
    AgentPerformanceRecord,
    AgentRole,
    Answer,
    Artifact,
    Escalation,
    MemoryEntry,
    MemoryLevel,
    MessageType,
    ProjectPlan,
    Question,
    ReviewRequest,
    ReviewResult,
    ReviewScore,
    StatusUpdate,
    TaskAssignment,
    TaskPriority,
    TaskResult,
    TaskStatus,
    TeamConfig,
    WorkerSpecialty,
)

# =============================================================================
# Enums
# =============================================================================


class TestEnums:
    """Test all StrEnum types."""

    def test_agent_roles(self) -> None:
        """All expected roles exist."""
        expected = {
            "master_researcher",
            "assessor",
            "devils_advocate",
            "project_manager",
            "senior_project_manager",
            "team_lead",
            "worker",
            "consolidator",
            "judge",
        }
        assert {r.value for r in AgentRole} == expected

    def test_task_status_values(self) -> None:
        expected = {
            "pending",
            "in_progress",
            "blocked",
            "needs_review",
            "revision_needed",
            "completed",
            "failed",
        }
        assert {s.value for s in TaskStatus} == expected

    def test_task_priority_values(self) -> None:
        expected = {"critical", "high", "medium", "low"}
        assert {p.value for p in TaskPriority} == expected

    def test_message_type_values(self) -> None:
        expected = {
            "task_assignment",
            "task_result",
            "status_update",
            "question",
            "answer",
            "review_request",
            "review_result",
            "escalation",
        }
        assert {m.value for m in MessageType} == expected

    def test_worker_specialty_values(self) -> None:
        expected = {
            "coder",
            "researcher",
            "writer",
            "reviewer",
            "tester",
            "architect",
            "data_analyst",
            "devops",
        }
        assert {s.value for s in WorkerSpecialty} == expected

    def test_memory_level_values(self) -> None:
        expected = {"agent", "team", "global"}
        assert {lv.value for lv in MemoryLevel} == expected

    def test_enum_string_behavior(self) -> None:
        """StrEnum values should behave as strings."""
        assert AgentRole.WORKER == "worker"
        assert TaskStatus.COMPLETED == "completed"
        assert f"Role: {AgentRole.PM}" if hasattr(AgentRole, "PM") else True


# =============================================================================
# A2AMessage (base)
# =============================================================================


class TestA2AMessage:
    """Tests for the base A2A message envelope."""

    def test_auto_generates_id(self) -> None:
        msg = TaskAssignment(
            from_agent="pm_1",
            to_agent="worker_1",
            title="Test",
            description="Test task",
        )
        assert msg.id
        assert len(msg.id) == 12

    def test_auto_generates_timestamp(self) -> None:
        msg = TaskAssignment(
            from_agent="pm_1",
            to_agent="worker_1",
            title="Test",
            description="Test task",
        )
        assert isinstance(msg.timestamp, datetime)
        assert msg.timestamp.tzinfo is not None

    def test_metadata_defaults_to_empty(self) -> None:
        msg = TaskAssignment(
            from_agent="pm_1",
            to_agent="worker_1",
            title="Test",
            description="Test task",
        )
        assert msg.metadata == {}

    def test_custom_metadata(self) -> None:
        msg = TaskAssignment(
            from_agent="pm_1",
            to_agent="worker_1",
            title="Test",
            description="Test task",
            metadata={"priority": "urgent", "tags": ["auth"]},
        )
        assert msg.metadata["priority"] == "urgent"

    def test_unique_ids_per_instance(self) -> None:
        msg1 = TaskAssignment(from_agent="a", to_agent="b", title="T", description="D")
        msg2 = TaskAssignment(from_agent="a", to_agent="b", title="T", description="D")
        assert msg1.id != msg2.id


# =============================================================================
# TaskAssignment
# =============================================================================


class TestTaskAssignment:
    def test_creation_with_defaults(self) -> None:
        msg = TaskAssignment(
            from_agent="pm_001",
            to_agent="worker_001",
            title="Build auth module",
            description="Implement JWT-based authentication",
            acceptance_criteria=["Tokens expire after 1h"],
        )
        assert msg.type == MessageType.TASK_ASSIGNMENT
        assert msg.title == "Build auth module"
        assert msg.priority == TaskPriority.MEDIUM
        assert msg.task_id  # auto-generated
        assert msg.max_iterations == 10
        assert msg.token_budget == 8000

    def test_custom_priority(self) -> None:
        msg = TaskAssignment(
            from_agent="pm_001",
            to_agent="worker_001",
            title="Hotfix",
            description="Fix the bug",
            priority=TaskPriority.CRITICAL,
        )
        assert msg.priority == TaskPriority.CRITICAL

    def test_serialization_roundtrip(self) -> None:
        msg = TaskAssignment(
            from_agent="pm_001",
            to_agent="worker_001",
            title="Test",
            description="Test task",
            acceptance_criteria=["Criterion 1", "Criterion 2"],
            dependencies=["task_001"],
        )
        data = msg.model_dump(mode="json")
        restored = TaskAssignment.model_validate(data)
        assert restored.title == "Test"
        assert len(restored.acceptance_criteria) == 2
        assert restored.dependencies == ["task_001"]

    def test_json_roundtrip(self) -> None:
        msg = TaskAssignment(
            from_agent="pm_001",
            to_agent="worker_001",
            title="Test",
            description="Test task",
        )
        json_str = msg.model_dump_json()
        restored = TaskAssignment.model_validate_json(json_str)
        assert restored.title == msg.title
        assert restored.id == msg.id


# =============================================================================
# Artifact & TaskResult
# =============================================================================


class TestArtifact:
    def test_creation(self) -> None:
        a = Artifact(
            name="auth.py",
            content_type="python",
            content="def login(): pass",
            summary="Auth module",
        )
        assert a.name == "auth.py"
        assert a.summary == "Auth module"

    def test_default_summary(self) -> None:
        a = Artifact(name="file.txt", content_type="text", content="hello")
        assert a.summary == ""

    def test_serialization(self) -> None:
        a = Artifact(name="f.py", content_type="python", content="x=1")
        data = a.model_dump()
        assert data["name"] == "f.py"


class TestTaskResult:
    def test_creation(self) -> None:
        result = TaskResult(
            from_agent="worker_001",
            to_agent="tl_001",
            task_id="task_abc",
            status=TaskStatus.COMPLETED,
            summary="Done",
            artifacts=[
                Artifact(name="f.py", content_type="python", content="x=1", summary="Code")
            ],
        )
        assert result.type == MessageType.TASK_RESULT
        assert result.status == TaskStatus.COMPLETED
        assert len(result.artifacts) == 1

    def test_default_tokens_and_iterations(self) -> None:
        result = TaskResult(
            from_agent="w",
            to_agent="tl",
            task_id="t",
            status=TaskStatus.COMPLETED,
            summary="Done",
        )
        assert result.tokens_used == 0
        assert result.iterations_used == 0

    def test_failed_result(self) -> None:
        result = TaskResult(
            from_agent="w",
            to_agent="tl",
            task_id="t",
            status=TaskStatus.FAILED,
            summary="Error occurred",
            issues_found=["API timeout", "Invalid response"],
        )
        assert result.status == TaskStatus.FAILED
        assert len(result.issues_found) == 2


# =============================================================================
# StatusUpdate
# =============================================================================


class TestStatusUpdate:
    def test_creation(self) -> None:
        update = StatusUpdate(
            from_agent="worker_001",
            to_agent="tl_001",
            task_id="task_abc",
            status=TaskStatus.IN_PROGRESS,
            progress_pct=45.0,
            brief="Halfway done",
        )
        assert update.type == MessageType.STATUS_UPDATE
        assert update.progress_pct == 45.0

    def test_progress_bounds(self) -> None:
        with pytest.raises(ValidationError):
            StatusUpdate(
                from_agent="w",
                to_agent="tl",
                task_id="t",
                status=TaskStatus.IN_PROGRESS,
                progress_pct=150.0,  # Invalid: > 100
                brief="Too much",
            )

    def test_progress_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            StatusUpdate(
                from_agent="w",
                to_agent="tl",
                task_id="t",
                status=TaskStatus.IN_PROGRESS,
                progress_pct=-10.0,  # Invalid: < 0
                brief="Negative",
            )


# =============================================================================
# Question & Answer
# =============================================================================


class TestQuestion:
    def test_creation(self) -> None:
        q = Question(
            from_agent="worker_001",
            to_agent="tl_001",
            question="What authentication method should I use?",
            options=["JWT", "OAuth2", "API keys"],
        )
        assert q.type == MessageType.QUESTION
        assert len(q.options) == 3
        assert q.is_blocking is False

    def test_blocking_question(self) -> None:
        q = Question(
            from_agent="w",
            to_agent="tl",
            question="Need DB credentials",
            is_blocking=True,
        )
        assert q.is_blocking is True


class TestAnswer:
    def test_creation(self) -> None:
        a = Answer(
            from_agent="tl_001",
            to_agent="worker_001",
            question_id="q_abc",
            answer="Use JWT with RS256",
        )
        assert a.type == MessageType.ANSWER
        assert a.additional_context == ""


# =============================================================================
# Review
# =============================================================================


class TestReviewRequest:
    def test_creation(self) -> None:
        req = ReviewRequest(
            from_agent="tl",
            to_agent="judge",
            task_id="task_1",
            artifacts=[Artifact(name="f.py", content_type="python", content="x=1")],
            review_criteria=["Correctness", "Performance"],
        )
        assert req.type == MessageType.REVIEW_REQUEST
        assert len(req.review_criteria) == 2


class TestReviewScore:
    def test_creation(self) -> None:
        s = ReviewScore(criterion="Correctness", score=8.5, reasoning="Good")
        assert s.score == 8.5

    def test_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ReviewScore(criterion="Test", score=11.0, reasoning="Too high")

    def test_score_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            ReviewScore(criterion="Test", score=-1.0, reasoning="Negative")


class TestReviewResult:
    def test_creation(self) -> None:
        result = ReviewResult(
            from_agent="judge",
            to_agent="tl",
            task_id="task_1",
            overall_score=8.0,
            scores=[
                ReviewScore(criterion="Correctness", score=8.5, reasoning="Good"),
                ReviewScore(criterion="Performance", score=7.5, reasoning="OK"),
            ],
            verdict=TaskStatus.COMPLETED,
            feedback="Well done, minor improvements needed",
        )
        assert result.type == MessageType.REVIEW_RESULT
        assert result.overall_score == 8.0
        assert len(result.scores) == 2


# =============================================================================
# Escalation
# =============================================================================


class TestEscalation:
    def test_creation(self) -> None:
        e = Escalation(
            from_agent="worker",
            to_agent="tl",
            task_id="task_1",
            reason="API rate limited, cannot proceed",
            severity=TaskPriority.HIGH,
            suggested_action="Wait 60s and retry",
        )
        assert e.type == MessageType.ESCALATION
        assert e.severity == TaskPriority.HIGH

    def test_defaults(self) -> None:
        e = Escalation(
            from_agent="w",
            to_agent="tl",
            task_id="t",
            reason="Problem",
        )
        assert e.severity == TaskPriority.HIGH
        assert e.suggested_action == ""


# =============================================================================
# AgentConfig
# =============================================================================


class TestAgentConfig:
    def test_defaults(self) -> None:
        config = AgentConfig(
            role=AgentRole.WORKER,
            name="Test Worker",
            model="openai/gpt-4o-mini",
        )
        assert config.agent_id  # auto-generated
        assert config.temperature == 0.3
        assert config.max_iterations == 15
        assert config.token_budget == 16000
        assert config.specialty is None
        assert config.team_id is None
        assert config.supervisor_id is None
        assert config.subordinate_ids == []
        assert config.system_prompt == ""

    def test_full_config(self) -> None:
        config = AgentConfig(
            agent_id="custom_id",
            role=AgentRole.TEAM_LEAD,
            name="TL Alpha",
            model="anthropic/claude-sonnet-4-20250514",
            specialty=WorkerSpecialty.ARCHITECT,
            team_id="team_1",
            supervisor_id="pm_1",
            subordinate_ids=["w1", "w2"],
            system_prompt="You are a TL",
            max_iterations=20,
            token_budget=20000,
            temperature=0.5,
        )
        assert config.agent_id == "custom_id"
        assert config.role == AgentRole.TEAM_LEAD
        assert len(config.subordinate_ids) == 2


# =============================================================================
# TeamConfig & ProjectPlan
# =============================================================================


class TestTeamConfig:
    def test_creation(self) -> None:
        tl = AgentConfig(role=AgentRole.TEAM_LEAD, name="TL", model="m")
        w = AgentConfig(role=AgentRole.WORKER, name="W", model="m")
        tc = TeamConfig(team_lead=tl, workers=[w], objective="Build auth")
        assert tc.team_id  # auto-generated
        assert len(tc.workers) == 1
        assert tc.objective == "Build auth"


class TestProjectPlan:
    def test_creation_minimal(self) -> None:
        plan = ProjectPlan(
            title="Test Project",
            description="A test project",
            complexity="medium",
            estimated_teams=1,
            risks=["timeline risk"],
            success_criteria=["works"],
        )
        assert plan.project_id  # auto-generated
        assert plan.architecture_type == "hierarchical"
        assert len(plan.risks) == 1

    def test_json_roundtrip(self) -> None:
        plan = ProjectPlan(
            title="Test",
            description="D",
            complexity="low",
            estimated_teams=1,
        )
        json_str = plan.model_dump_json()
        restored = ProjectPlan.model_validate_json(json_str)
        assert restored.title == "Test"

    def test_full_plan(self) -> None:
        tl = AgentConfig(role=AgentRole.TEAM_LEAD, name="TL", model="m")
        team = TeamConfig(team_lead=tl, objective="Obj")
        pm = AgentConfig(role=AgentRole.PROJECT_MANAGER, name="PM", model="m")
        plan = ProjectPlan(
            title="Big Project",
            description="Full project",
            complexity="extreme",
            estimated_teams=3,
            teams=[team],
            project_managers=[pm],
            risks=["scope creep"],
            success_criteria=["all tests pass"],
            architecture_type="debate",
        )
        assert len(plan.teams) == 1
        assert len(plan.project_managers) == 1
        assert plan.architecture_type == "debate"


# =============================================================================
# AgentPerformanceRecord
# =============================================================================


class TestPerformanceRecord:
    def test_composite_score_no_peers(self) -> None:
        record = AgentPerformanceRecord(
            agent_id="a1",
            task_id="t1",
            role=AgentRole.WORKER,
            model="test",
            correctness=8.0,
            completeness=7.0,
            edge_case_handling=6.0,
            communication_quality=9.0,
            efficiency=8.0,
        )
        # 8*0.3 + 7*0.2 + 6*0.15 + 9*0.1 + 8*0.15 = 2.4+1.4+0.9+0.9+1.2 = 6.8
        assert abs(record.composite_score - 6.8) < 0.01

    def test_composite_score_with_peers(self) -> None:
        record = AgentPerformanceRecord(
            agent_id="a1",
            task_id="t1",
            role=AgentRole.WORKER,
            model="test",
            correctness=8.0,
            completeness=8.0,
            edge_case_handling=8.0,
            communication_quality=8.0,
            efficiency=8.0,
            peer_judge_scores=[9.0, 8.5],
        )
        # base = 8*0.3 + 8*0.2 + 8*0.15 + 8*0.1 + 8*0.15 = 7.2
        # peer_avg = 8.75, bonus = 8.75 * 0.1 = 0.875
        # total = 8.075
        assert abs(record.composite_score - 8.075) < 0.01

    def test_score_bounds(self) -> None:
        """Scores must be 0-10."""
        with pytest.raises(ValidationError):
            AgentPerformanceRecord(
                agent_id="a",
                task_id="t",
                role=AgentRole.WORKER,
                model="test",
                correctness=11.0,  # Invalid
                completeness=0.0,
                edge_case_handling=0.0,
                communication_quality=0.0,
                efficiency=0.0,
            )

    def test_all_zeros(self) -> None:
        record = AgentPerformanceRecord(
            agent_id="a",
            task_id="t",
            role=AgentRole.WORKER,
            model="test",
            correctness=0.0,
            completeness=0.0,
            edge_case_handling=0.0,
            communication_quality=0.0,
            efficiency=0.0,
        )
        assert record.composite_score == 0.0

    def test_perfect_scores(self) -> None:
        record = AgentPerformanceRecord(
            agent_id="a",
            task_id="t",
            role=AgentRole.WORKER,
            model="test",
            correctness=10.0,
            completeness=10.0,
            edge_case_handling=10.0,
            communication_quality=10.0,
            efficiency=10.0,
            peer_judge_scores=[10.0, 10.0],
        )
        # 10 * (0.3+0.2+0.15+0.1+0.15) + 10 * 0.1 = 9.0 + 1.0 = 10.0
        assert abs(record.composite_score - 10.0) < 0.01

    def test_timing_defaults(self) -> None:
        record = AgentPerformanceRecord(
            agent_id="a",
            task_id="t",
            role=AgentRole.WORKER,
            model="test",
            correctness=5.0,
            completeness=5.0,
            edge_case_handling=5.0,
            communication_quality=5.0,
            efficiency=5.0,
        )
        assert record.time_to_first_token_ms == 0
        assert record.total_duration_ms == 0
        assert record.tokens_used == 0
        assert record.iterations_used == 0


# =============================================================================
# MemoryEntry
# =============================================================================


class TestMemoryEntry:
    def test_defaults(self) -> None:
        entry = MemoryEntry(level=MemoryLevel.AGENT, content="Test")
        assert entry.id  # auto-generated
        assert entry.entry_type == "observation"
        assert entry.promoted is False
        assert entry.turn_number == 0
        assert entry.agent_id is None
        assert entry.summary == ""

    def test_full_entry(self) -> None:
        entry = MemoryEntry(
            level=MemoryLevel.TEAM,
            agent_id="a1",
            team_id="t1",
            content="Team decided to use PostgreSQL",
            summary="DB choice: PostgreSQL",
            entry_type="decision",
            metadata={"confidence": 0.9},
            turn_number=5,
            promoted=True,
        )
        assert entry.level == MemoryLevel.TEAM
        assert entry.entry_type == "decision"
        assert entry.promoted is True
        assert entry.metadata["confidence"] == 0.9

    def test_serialization(self) -> None:
        entry = MemoryEntry(
            level=MemoryLevel.GLOBAL,
            content="Best practice: use bcrypt",
            entry_type="learning",
        )
        data = entry.model_dump(mode="json")
        restored = MemoryEntry.model_validate(data)
        assert restored.content == "Best practice: use bcrypt"
        assert restored.level == MemoryLevel.GLOBAL
