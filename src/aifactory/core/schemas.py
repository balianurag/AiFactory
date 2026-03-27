"""
Structured A2A message schemas for the AI Factory.

Every inter-agent message uses these Pydantic models.
This eliminates parsing errors, enforces contracts, and cuts token usage 5-10x
compared to free-form text communication.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# Enums
# =============================================================================


class AgentRole(StrEnum):
    """All possible agent roles in the hierarchy."""

    MASTER_RESEARCHER = "master_researcher"
    ASSESSOR = "assessor"
    DEVILS_ADVOCATE = "devils_advocate"
    PROJECT_MANAGER = "project_manager"
    SENIOR_PROJECT_MANAGER = "senior_project_manager"
    TEAM_LEAD = "team_lead"
    WORKER = "worker"
    CONSOLIDATOR = "consolidator"
    JUDGE = "judge"


class TaskStatus(StrEnum):
    """Status of a task through its lifecycle."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    NEEDS_REVIEW = "needs_review"
    REVISION_NEEDED = "revision_needed"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(StrEnum):
    """Priority levels for task scheduling."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MessageType(StrEnum):
    """Types of structured A2A messages."""

    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    STATUS_UPDATE = "status_update"
    QUESTION = "question"
    ANSWER = "answer"
    REVIEW_REQUEST = "review_request"
    REVIEW_RESULT = "review_result"
    ESCALATION = "escalation"


class WorkerSpecialty(StrEnum):
    """Worker specialization types."""

    CODER = "coder"
    RESEARCHER = "researcher"
    WRITER = "writer"
    REVIEWER = "reviewer"
    TESTER = "tester"
    ARCHITECT = "architect"
    DATA_ANALYST = "data_analyst"
    DEVOPS = "devops"


# =============================================================================
# Base Message
# =============================================================================


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> datetime:
    return datetime.now(UTC)


class A2AMessage(BaseModel):
    """Base envelope for every agent-to-agent message."""

    id: str = Field(default_factory=_new_id, description="Unique message ID")
    type: MessageType
    from_agent: str = Field(description="Sender agent ID")
    to_agent: str = Field(description="Recipient agent ID")
    team_id: str | None = Field(default=None, description="Team scope")
    timestamp: datetime = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Task Assignment & Results
# =============================================================================


class TaskAssignment(A2AMessage):
    """Sent by PM/TL to assign work to a subordinate."""

    type: MessageType = MessageType.TASK_ASSIGNMENT

    task_id: str = Field(default_factory=_new_id, description="Unique task ID")
    title: str = Field(description="Short task title")
    description: str = Field(description="Detailed task specification")
    priority: TaskPriority = TaskPriority.MEDIUM
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="Measurable conditions that define 'done'",
    )
    context: str = Field(default="", description="Relevant background info (summarised)")
    dependencies: list[str] = Field(
        default_factory=list,
        description="Task IDs this depends on",
    )
    max_iterations: int = Field(default=10, description="Iteration budget for this task")
    token_budget: int = Field(default=8000, description="Max tokens the agent may use")


class Artifact(BaseModel):
    """A deliverable produced by an agent (code, doc, analysis, etc.)."""

    name: str
    content_type: str = Field(description="e.g. 'python', 'markdown', 'json', 'text'")
    content: str
    summary: str = Field(default="", description="One-line description")


class TaskResult(A2AMessage):
    """Sent by a worker/TL back to its supervisor with completed work."""

    type: MessageType = MessageType.TASK_RESULT

    task_id: str
    status: TaskStatus
    summary: str = Field(description="Brief summary of what was done")
    artifacts: list[Artifact] = Field(default_factory=list)
    issues_found: list[str] = Field(
        default_factory=list,
        description="Problems, edge cases, or risks discovered",
    )
    tokens_used: int = Field(default=0, description="Actual tokens consumed")
    iterations_used: int = Field(default=0, description="Actual iterations taken")


# =============================================================================
# Status & Communication
# =============================================================================


class StatusUpdate(A2AMessage):
    """Periodic progress ping from any agent to its supervisor."""

    type: MessageType = MessageType.STATUS_UPDATE

    task_id: str
    status: TaskStatus
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    brief: str = Field(description="One-line status")
    blockers: list[str] = Field(default_factory=list)


class Question(A2AMessage):
    """Agent asks a question to its supervisor (or to the human)."""

    type: MessageType = MessageType.QUESTION

    task_id: str | None = None
    question: str
    options: list[str] = Field(
        default_factory=list,
        description="Suggested answers (if any)",
    )
    is_blocking: bool = Field(
        default=False,
        description="If true, agent cannot continue without answer",
    )


class Answer(A2AMessage):
    """Response to a Question."""

    type: MessageType = MessageType.ANSWER

    question_id: str = Field(description="ID of the Question message being answered")
    answer: str
    additional_context: str = ""


# =============================================================================
# Review / QA
# =============================================================================


class ReviewRequest(A2AMessage):
    """Sent to a Judge/peer to request quality review of work."""

    type: MessageType = MessageType.REVIEW_REQUEST

    task_id: str
    artifacts: list[Artifact] = Field(default_factory=list)
    review_criteria: list[str] = Field(
        default_factory=list,
        description="What the reviewer should check for",
    )


class ReviewScore(BaseModel):
    """Individual criterion score."""

    criterion: str
    score: float = Field(ge=0.0, le=10.0)
    reasoning: str


class ReviewResult(A2AMessage):
    """Judge's assessment of submitted work."""

    type: MessageType = MessageType.REVIEW_RESULT

    task_id: str
    overall_score: float = Field(ge=0.0, le=10.0)
    scores: list[ReviewScore] = Field(default_factory=list)
    verdict: TaskStatus = Field(description="COMPLETED if passes, REVISION_NEEDED if not")
    feedback: str = Field(description="Actionable feedback for the agent")


class Escalation(A2AMessage):
    """Agent escalates an issue to a higher level in the hierarchy."""

    type: MessageType = MessageType.ESCALATION

    task_id: str
    reason: str
    severity: TaskPriority = TaskPriority.HIGH
    suggested_action: str = ""


# =============================================================================
# Agent Configuration & Hierarchy
# =============================================================================


class AgentConfig(BaseModel):
    """Runtime configuration for spawning an agent."""

    agent_id: str = Field(default_factory=_new_id)
    role: AgentRole
    name: str = Field(description="Human-readable name, e.g. 'Backend Coder #1'")
    model: str = Field(
        description="LiteLLM model string, e.g. 'anthropic/claude-sonnet-4-20250514'",
    )
    specialty: WorkerSpecialty | None = None
    team_id: str | None = None
    supervisor_id: str | None = None
    subordinate_ids: list[str] = Field(default_factory=list)
    system_prompt: str = Field(default="", description="Full system prompt for this agent")
    max_iterations: int = 15
    token_budget: int = 16000
    temperature: float = 0.3


class TeamConfig(BaseModel):
    """Configuration for a team unit (1 TL + N workers)."""

    team_id: str = Field(default_factory=_new_id)
    team_lead: AgentConfig
    workers: list[AgentConfig] = Field(default_factory=list)
    objective: str = Field(description="What this team is responsible for")


class ProjectPlan(BaseModel):
    """Output of the Assessment Team — the full hierarchy blueprint."""

    project_id: str = Field(default_factory=_new_id)
    title: str
    description: str
    complexity: str = Field(description="low / medium / high / extreme")
    estimated_teams: int
    teams: list[TeamConfig] = Field(default_factory=list)
    project_managers: list[AgentConfig] = Field(default_factory=list)
    senior_pm: AgentConfig | None = None
    risks: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    architecture_type: str = Field(
        default="hierarchical",
        description="'hierarchical' or 'debate' — decided by assessment team",
    )


# =============================================================================
# RL Sidecar Schemas
# =============================================================================


class AgentPerformanceRecord(BaseModel):
    """Performance record for one agent on one task — fed to RL sidecar."""

    agent_id: str
    task_id: str
    role: AgentRole
    model: str
    timestamp: datetime = Field(default_factory=_now)

    # Core metrics (0-10 scale)
    correctness: float = Field(ge=0.0, le=10.0)
    completeness: float = Field(ge=0.0, le=10.0, description="Holism / coverage")
    edge_case_handling: float = Field(ge=0.0, le=10.0)
    communication_quality: float = Field(ge=0.0, le=10.0, description="A2A clarity")
    efficiency: float = Field(ge=0.0, le=10.0, description="Tokens used vs. quality")

    # Timing
    time_to_first_token_ms: int = 0
    total_duration_ms: int = 0
    tokens_used: int = 0
    iterations_used: int = 0

    # Peer scores
    peer_judge_scores: list[float] = Field(default_factory=list)

    @property
    def composite_score(self) -> float:
        """Weighted composite — the single number the RL sidecar optimises."""
        weights = {
            "correctness": 0.30,
            "completeness": 0.20,
            "edge_case_handling": 0.15,
            "communication_quality": 0.10,
            "efficiency": 0.15,
        }
        base = (
            self.correctness * weights["correctness"]
            + self.completeness * weights["completeness"]
            + self.edge_case_handling * weights["edge_case_handling"]
            + self.communication_quality * weights["communication_quality"]
            + self.efficiency * weights["efficiency"]
        )
        # Add peer bonus (10% weight, average of peer scores)
        if self.peer_judge_scores:
            peer_avg = sum(self.peer_judge_scores) / len(self.peer_judge_scores)
            base += peer_avg * 0.10
        return round(base, 3)


# =============================================================================
# Memory Layer Schemas
# =============================================================================


class MemoryLevel(StrEnum):
    """The three tiers of the memory hierarchy."""

    AGENT = "agent"
    TEAM = "team"
    GLOBAL = "global"


class MemoryEntry(BaseModel):
    """A single entry in the memory system."""

    id: str = Field(default_factory=_new_id)
    level: MemoryLevel
    agent_id: str | None = None
    team_id: str | None = None
    content: str
    summary: str = Field(default="", description="Compressed version for promotion")
    entry_type: str = Field(
        default="observation",
        description="observation / decision / artifact / learning / error",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_now)
    turn_number: int = 0
    promoted: bool = Field(default=False, description="Already promoted to next layer?")
