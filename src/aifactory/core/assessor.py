"""
Task Complexity Assessor — The intelligent gate for the AI Factory.

This module provides a lightweight, cheap LLM call that classifies incoming
tasks by complexity before the pipeline decides how many agents to spin up.

Routing logic:
  trivial   → single Worker (skip PM + TL overhead)
  moderate  → 1 PM + 1 TL + 1-2 Workers
  complex   → full hierarchy (PM + TL + 3 Workers)

Cost utility:
  cost_usd() calculates per-model pricing from input/output token counts.
"""

from __future__ import annotations

import enum
import json
import logging
from typing import Any

import litellm
from pydantic import BaseModel, Field

from aifactory.core.prompt_cache import build_cached_system

logger = logging.getLogger(__name__)


# =============================================================================
# Cost Tracking
# =============================================================================

# Pricing per 1M tokens (input / output) — updated June 2025
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Direct OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    # OpenRouter pass-through (kept for future multi-provider)
    "openrouter/anthropic/claude-sonnet-4-5-20250514": (3.00, 15.00),
    "openrouter/google/gemini-2.5-flash": (0.15, 0.60),
    "openrouter/google/gemini-2.5-pro": (1.25, 10.00),
}


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Calculate cost in USD for a given model and token counts.

    Returns 0.0 for unknown models (safe fallback).
    """
    pricing = _MODEL_PRICING.get(model)
    if pricing is None:
        logger.debug("No pricing found for model %r — returning $0.00", model)
        return 0.0
    input_rate, output_rate = pricing
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


# =============================================================================
# Complexity Classification
# =============================================================================


class TaskComplexity(enum.StrEnum):
    """How complex an incoming task is — determines pipeline routing."""

    TRIVIAL = "trivial"
    MODERATE = "moderate"
    COMPLEX = "complex"


class AssessmentResult(BaseModel):
    """Output from the complexity assessor."""

    complexity: TaskComplexity = Field(description="trivial, moderate, or complex")
    reasoning: str = Field(description="One-line explanation of the classification")
    suggested_workers: int = Field(
        default=1,
        description="Recommended number of workers",
        ge=1,
        le=5,
    )
    suggested_specialties: list[str] = Field(
        default_factory=lambda: ["coder"],
        description="Recommended worker specialties",
    )


_ASSESS_SYSTEM = (
    "You are a task complexity assessor. Your ONLY job is to classify an "
    "incoming task as trivial, moderate, or complex.\n\n"
    "Rules:\n"
    "- **trivial**: single-function, single-file, well-defined scope, "
    "can be done in <50 lines. Examples: write a utility function, "
    "fix a typo, convert units, simple CRUD endpoint.\n"
    "- **moderate**: multi-file or multi-step, needs some design but "
    "scope is clear. Examples: REST API with 3-5 endpoints, small CLI tool, "
    "data pipeline with a few stages.\n"
    "- **complex**: requires architecture decisions, multiple interacting "
    "components, research, or ambiguous scope. Examples: full application, "
    "distributed system, ML pipeline, refactoring a large codebase.\n\n"
    "Respond ONLY with valid JSON — no extra text."
)

_ASSESS_PROMPT = (
    'Classify this task:\n\n"{task}"\n\n'
    "Respond with JSON:\n"
    '{{"complexity": "trivial|moderate|complex", '
    '"reasoning": "one-line explanation", '
    '"suggested_workers": 1, '
    '"suggested_specialties": ["coder"]}}'
)


async def assess_complexity(
    task: str,
    model: str = "gpt-4o-mini",
) -> AssessmentResult:
    """
    Run a cheap LLM call to classify task complexity.

    Uses gpt-4o-mini (~0.15 / 0.60 per 1M tokens) — typically costs
    < $0.001 per assessment.

    Args:
        task: The task description to classify.
        model: Which model to use for assessment (should be cheap).

    Returns:
        AssessmentResult with complexity level and routing suggestions.
    """
    try:
        # Phase 3.5: _ASSESS_SYSTEM is a fixed classifier prompt reused on
        # every incoming task, so attach cache_control when the caller opts
        # into a Claude-family assessor model. No-op (plain string) for the
        # default gpt-4o-mini path — helper gates on Claude route + setting.
        # ``api_base=""`` — assess_complexity() calls litellm directly without
        # a ``base_url`` override; LITELLM_API_BASE (dynamic_pipeline's proxy
        # switch) is irrelevant here and must not gate caching.
        _sys_content = build_cached_system(_ASSESS_SYSTEM, model, api_base="")
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": _sys_content},
                {"role": "user", "content": _ASSESS_PROMPT.format(task=task[:500])},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        data: dict[str, Any] = json.loads(raw)

        # Track assessment cost
        usage = response.usage
        input_tok = usage.prompt_tokens if usage else 0
        output_tok = usage.completion_tokens if usage else 0
        assess_cost = cost_usd(model, input_tok, output_tok)

        result = AssessmentResult(
            complexity=TaskComplexity(data.get("complexity", "moderate")),
            reasoning=data.get("reasoning", "Could not determine reasoning"),
            suggested_workers=data.get("suggested_workers", 1),
            suggested_specialties=data.get("suggested_specialties", ["coder"]),
        )

        logger.info(
            "Task assessed: complexity=%s, workers=%d, cost=$%.6f — %s",
            result.complexity.value,
            result.suggested_workers,
            assess_cost,
            result.reasoning,
        )
        return result

    except Exception as e:
        logger.warning("Complexity assessment failed, defaulting to moderate: %s", e)
        return AssessmentResult(
            complexity=TaskComplexity.MODERATE,
            reasoning=f"Assessment failed ({e}), defaulting to moderate",
        )
