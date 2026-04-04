"""
Tests for core/assessor.py — cost calculation, complexity classification, and routing.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aifactory.core.assessor import (
    AssessmentResult,
    TaskComplexity,
    assess_complexity,
    cost_usd,
)

# =============================================================================
# cost_usd()
# =============================================================================


class TestCostUsd:
    def test_gpt4o_known_pricing(self) -> None:
        # gpt-4o: $2.50 / $10.00 per 1M tokens
        cost = cost_usd("gpt-4o", input_tokens=1000, output_tokens=500)
        expected = (1000 * 2.50 + 500 * 10.00) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_gpt4o_mini_known_pricing(self) -> None:
        # gpt-4o-mini: $0.15 / $0.60 per 1M tokens
        cost = cost_usd("gpt-4o-mini", input_tokens=10000, output_tokens=5000)
        expected = (10000 * 0.15 + 5000 * 0.60) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_unknown_model_returns_zero(self) -> None:
        cost = cost_usd("unknown-model-xyz", input_tokens=100000, output_tokens=50000)
        assert cost == 0.0

    def test_zero_tokens(self) -> None:
        cost = cost_usd("gpt-4o", input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_large_token_counts(self) -> None:
        # 1M tokens each
        cost = cost_usd("gpt-4o", input_tokens=1_000_000, output_tokens=1_000_000)
        expected = 2.50 + 10.00  # $12.50
        assert cost == pytest.approx(expected)


# =============================================================================
# AssessmentResult model
# =============================================================================


class TestAssessmentResult:
    def test_trivial(self) -> None:
        result = AssessmentResult(
            complexity=TaskComplexity.TRIVIAL,
            reasoning="Simple utility function",
            suggested_workers=1,
            suggested_specialties=["coder"],
        )
        assert result.complexity == TaskComplexity.TRIVIAL
        assert result.suggested_workers == 1

    def test_complex(self) -> None:
        result = AssessmentResult(
            complexity=TaskComplexity.COMPLEX,
            reasoning="Multi-component distributed system",
            suggested_workers=3,
            suggested_specialties=["coder", "architect", "tester"],
        )
        assert result.complexity == TaskComplexity.COMPLEX
        assert result.suggested_workers == 3

    def test_defaults(self) -> None:
        result = AssessmentResult(
            complexity=TaskComplexity.MODERATE,
            reasoning="Default",
        )
        assert result.suggested_workers == 1
        assert result.suggested_specialties == ["coder"]

    def test_complexity_enum_values(self) -> None:
        assert TaskComplexity.TRIVIAL.value == "trivial"
        assert TaskComplexity.MODERATE.value == "moderate"
        assert TaskComplexity.COMPLEX.value == "complex"


# =============================================================================
# assess_complexity() — mocked LLM
# =============================================================================


def _mock_assessment_response(
    complexity: str = "trivial",
    reasoning: str = "Simple function",
    workers: int = 1,
) -> MagicMock:
    """Create a mock OpenAI chat completion for the assessor."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps(
        {
            "complexity": complexity,
            "reasoning": reasoning,
            "suggested_workers": workers,
            "suggested_specialties": ["coder"],
        }
    )
    response.usage = MagicMock()
    response.usage.prompt_tokens = 150
    response.usage.completion_tokens = 50
    return response


class TestAssessComplexity:
    @pytest.mark.asyncio
    async def test_trivial_classification(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_assessment_response("trivial", "Simple utility function")
        )

        with patch("aifactory.core.assessor.openai.AsyncOpenAI", return_value=mock_client):
            result = await assess_complexity("Write a function to check if a number is prime")

        assert result.complexity == TaskComplexity.TRIVIAL
        assert result.suggested_workers == 1

    @pytest.mark.asyncio
    async def test_complex_classification(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_assessment_response(
                "complex", "Multi-service architecture", workers=3
            )
        )

        with patch("aifactory.core.assessor.openai.AsyncOpenAI", return_value=mock_client):
            result = await assess_complexity("Build a microservices architecture with Kafka")

        assert result.complexity == TaskComplexity.COMPLEX
        assert result.suggested_workers == 3

    @pytest.mark.asyncio
    async def test_fallback_on_error(self) -> None:
        """When the LLM call fails, should default to MODERATE."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API down"))

        with patch("aifactory.core.assessor.openai.AsyncOpenAI", return_value=mock_client):
            result = await assess_complexity("Any task")

        assert result.complexity == TaskComplexity.MODERATE
        assert "failed" in result.reasoning.lower() or "defaulting" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_malformed_json_fallback(self) -> None:
        """When the LLM returns non-JSON, should default to MODERATE."""
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "not json at all"
        response.usage = MagicMock()
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 20

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        with patch("aifactory.core.assessor.openai.AsyncOpenAI", return_value=mock_client):
            result = await assess_complexity("Some task")

        assert result.complexity == TaskComplexity.MODERATE


# =============================================================================
# Pipeline routing integration
# =============================================================================


class TestPipelineRouting:
    @pytest.mark.asyncio
    async def test_trivial_route_skips_pm(self, memory) -> None:
        """Trivial tasks should use _direct_worker, not the full hierarchy."""
        from aifactory.agents.worker import WorkerOutput
        from aifactory.core.schemas import Artifact, TaskStatus
        from aifactory.graphs.single_team import SingleTeamPipeline
        from tests.helpers import make_json_response, patch_llm

        worker_output = WorkerOutput(
            summary="Prime checker implemented",
            artifacts=[
                Artifact(
                    name="is_prime.py",
                    content_type="python",
                    content="def is_prime(n: int) -> bool: ...",
                    summary="Prime check function",
                )
            ],
            issues_found=[],
            status=TaskStatus.COMPLETED,
        )

        # Mock the assessor to return TRIVIAL
        trivial_assessment = AssessmentResult(
            complexity=TaskComplexity.TRIVIAL,
            reasoning="Simple utility function",
            suggested_workers=1,
            suggested_specialties=["coder"],
        )

        with (
            patch(
                "aifactory.graphs.single_team.assess_complexity",
                return_value=trivial_assessment,
            ),
            patch_llm(make_json_response(worker_output.model_dump())),
        ):
            pipeline = SingleTeamPipeline(num_workers=1, memory=memory)
            delivery = await pipeline.run("Write a function to check if a number is prime")

        # Should have been handled by direct worker — no PM/TL
        assert delivery is not None
        assert len(delivery.deliverables) >= 1
        qa = delivery.quality_assessment.lower()
        assert "trivial" in qa or "direct" in qa
        # Metrics should show 1 agent, 0 teams
        assert delivery.metrics.get("agents_used") == 1
        assert delivery.metrics.get("teams_used") == 0

    @pytest.mark.asyncio
    async def test_complex_route_uses_full_pipeline(self, memory) -> None:
        """Complex tasks should use the full PM → TL → Workers pipeline."""
        from aifactory.agents.project_manager import ObjectiveSpec, PMPlan
        from aifactory.agents.team_lead import DecompositionPlan, WorkerTaskSpec
        from aifactory.agents.worker import WorkerOutput
        from aifactory.core.schemas import Artifact, TaskStatus, WorkerSpecialty
        from aifactory.graphs.single_team import SingleTeamPipeline
        from tests.helpers import make_json_response, make_llm_response, patch_llm

        # Assessor returns COMPLEX
        complex_assessment = AssessmentResult(
            complexity=TaskComplexity.COMPLEX,
            reasoning="Multi-component architecture",
            suggested_workers=3,
        )

        plan = PMPlan(
            project_summary="Build a REST API",
            objectives=[
                ObjectiveSpec(
                    objective_id="obj_1",
                    title="Build REST API",
                    description="Create CRUD endpoints",
                )
            ],
        )
        decomp = DecompositionPlan(
            objective="Build REST API",
            tasks=[
                WorkerTaskSpec(
                    title="Implement routes",
                    description="Build FastAPI routes",
                    assigned_to_specialty=WorkerSpecialty.CODER,
                )
            ],
        )
        worker_out = WorkerOutput(
            summary="Routes built",
            artifacts=[Artifact(name="routes.py", content_type="python", content="code")],
            issues_found=[],
            status=TaskStatus.COMPLETED,
        )

        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_json_response(plan.model_dump())
            elif call_count == 2:
                return make_json_response(decomp.model_dump())
            elif call_count == 3:
                return make_json_response(worker_out.model_dump())
            return make_llm_response("ok")

        with (
            patch(
                "aifactory.graphs.single_team.assess_complexity",
                return_value=complex_assessment,
            ),
            patch_llm() as mock_llm,
        ):
            mock_llm.side_effect = mock_acompletion
            pipeline = SingleTeamPipeline(num_workers=1, memory=memory)
            delivery = await pipeline.run("Build a full REST API with auth and database")

        assert delivery is not None
        assert delivery.metrics.get("agents_used", 0) > 1
