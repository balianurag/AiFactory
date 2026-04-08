"""
Model Selector — Ability-based model scoring for the AI Factory.

Replaces the old tier-based (premium/standard/budget) model assignment with
a multi-dimensional dot-product scoring system.

Each model is rated 0–3 on 10 abilities. Each agent/task declares its ability
requirements. The best-matching model is selected via:

    score = Σ [ ((1 - α) × task_req_i  +  α × mode_profile_i) × model_ability_i ]

Modes control the quality/cost bias:
  best            — α=1.0, always picks the most capable model
  balanced        — α=0.35, task requirements lead, mode nudges
  cost_efficient  — α=0.75, strongly biases toward speed/cheap models

Ported from agent-arch/main.py.
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── Canonical ability list (order matters for dot-product) ─────────────

ABILITIES: list[str] = [
    "reasoning",
    "coding",
    "math",
    "writing",
    "instruction_following",
    "tool_use",
    "knowledge",
    "speed",
    "cost_efficiency",
    "context_handling",
]

# ── Quality abilities (excludes speed/cost — used for best_model) ─────

QUALITY_ABILITIES: list[str] = [a for a in ABILITIES if a not in ("speed", "cost_efficiency")]

# ── Efficiency abilities — NOT capped by max_quality in scoring ────────

_EFFICIENCY_ABILITIES: frozenset[str] = frozenset({"speed", "cost_efficiency"})

# ── Mode profiles — baked-in alpha + preference weights ───────────────
#
# max_quality: cap applied to model ability scores for non-efficiency dims.
#   best=3       — no cap, full 0-3 model scores, always picks most capable
#   balanced=2   — any model scoring ≥2 on quality looks equal; speed/cost
#                  break the tie → mid-tier fast/cheap models win vs top models
#   cost_efficient=3 — quality profile weights are already ~0, no need to cap

MODE_PROFILES: dict[str, dict[str, Any]] = {
    "best": {
        "alpha": 1.0,
        "max_quality": 3,
        "reasoning": 3,
        "coding": 3,
        "math": 2,
        "writing": 3,
        "instruction_following": 3,
        "tool_use": 3,
        "knowledge": 3,
        "speed": 0,
        "cost_efficiency": 0,
        "context_handling": 3,
    },
    "balanced": {
        "alpha": 0.35,
        "max_quality": 2,
        **{a: 2 for a in ABILITIES},
    },
    "cost_efficient": {
        "alpha": 0.75,
        "max_quality": 3,
        "reasoning": 0,
        "coding": 0,
        "math": 0,
        "writing": 0,
        "instruction_following": 0,
        "tool_use": 0,
        "knowledge": 0,
        "speed": 3,
        "cost_efficiency": 3,
        "context_handling": 0,
    },
}


# ── YAML loaders ──────────────────────────────────────────────────────

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@functools.lru_cache(maxsize=1)
def load_models() -> list[dict[str, Any]]:
    """Load and cache the model registry from config/model_registry.yaml."""
    path = _CONFIG_DIR / "model_registry.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("models", [])


def load_available_models() -> list[dict[str, Any]]:
    """
    Load models filtered to those accessible with currently-configured API keys.

    When LITELLM_API_BASE is set (proxy mode), all models are available since the
    proxy handles provider routing. Otherwise, checks which provider keys are set
    and filters the registry accordingly.

    Provider detection heuristics:
      - gpt-*, o1-*, o3-*, o4-*         → OPENAI_API_KEY
      - us.anthropic.*, claude-*         → ANTHROPIC_API_KEY or LITELLM_API_BASE
      - gemini-*                         → GOOGLE_API_KEY or GEMINI_API_KEY
      - us.meta.*, us.deepseek.*         → LITELLM_API_BASE (Bedrock)
      - perplexity/*                     → PERPLEXITY_API_KEY
      - mistral*, openai.gpt-oss-*      → LITELLM_API_BASE (custom endpoints)
    """
    all_models = load_models()

    # If a proxy is configured, all models are available
    if os.getenv("LITELLM_API_BASE", "").strip():
        return all_models

    # Check which provider keys are set
    has_openai = bool(os.getenv("OPENAI_API_KEY", "").strip())
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY", "").strip())
    has_google = bool(
        os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    )
    has_xai = bool(os.getenv("XAI_API_KEY", "").strip())
    has_perplexity = bool(os.getenv("PERPLEXITY_API_KEY", "").strip())
    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY", "").strip())

    def _is_accessible(model_id: str) -> bool:
        mid = model_id.lower()

        # OpenRouter can access everything
        if has_openrouter and mid.startswith("openrouter/"):
            return True

        # OpenAI models
        if mid.startswith(("gpt-", "o1-", "o3-", "o4-")):
            return has_openai
        # Anthropic (direct or Bedrock ARN)
        if mid.startswith(("claude-", "us.anthropic.")):
            return has_anthropic
        # Google
        if mid.startswith("gemini-"):
            return has_google
        # xAI
        if mid.startswith("grok-"):
            return has_xai
        # Perplexity
        if mid.startswith("perplexity/"):
            return has_perplexity
        # Bedrock / custom endpoints — need proxy
        if mid.startswith(("us.meta.", "us.deepseek.", "openai.gpt-oss-", "mistral")):
            return False
        # Unknown — allow if any key is set
        return has_openai or has_anthropic or has_google

    available = [m for m in all_models if _is_accessible(m["id"])]
    if not available:
        logger.warning(
            "No models accessible with current API keys — falling back to full registry"
        )
        return all_models

    logger.info(
        "Filtered model registry: %d/%d models available",
        len(available),
        len(all_models),
    )
    return available


@functools.lru_cache(maxsize=1)
def load_graph_config() -> dict[str, Any]:
    """Load and cache the graph config from config/graph.yaml."""
    path = _CONFIG_DIR / "graph.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_agent_cfg(name: str) -> dict[str, Any]:
    """Load a single agent config from config/agents/<name>.yaml."""
    path = _CONFIG_DIR / "agents" / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


# ── Model selection ───────────────────────────────────────────────────


def select_model(
    requirements: dict[str, float],
    models: list[dict[str, Any]] | None = None,
    mode: str = "balanced",
) -> tuple[str, float]:
    """
    Blended dot-product scoring.

    Effective requirement per ability:
        (1 - alpha) * task_requirement  +  alpha * mode_profile

    alpha is baked into each mode profile:
      best/cost_efficient = high alpha (mode overrides task)
      balanced            = low alpha  (task leads, mode nudges)

    Args:
        requirements: dict mapping ability names to 0–3 scores.
        models: list of model dicts from models.yaml. Auto-loaded if None.
        mode: "best", "balanced", or "cost_efficient".

    Returns:
        (model_id, score) — the best-matching model and its score.
    """
    if models is None:
        models = load_models()

    if not models:
        raise ValueError("No models available in registry")

    profile = MODE_PROFILES.get(mode, MODE_PROFILES["balanced"])
    alpha: float = profile["alpha"]
    max_q: float = float(profile.get("max_quality", 3))

    best_id: str = models[0]["id"]
    best_score: float = -1.0

    for m in models:
        score = sum(
            ((1 - alpha) * requirements.get(a, 0.0) + alpha * profile[a])
            * (
                float(m.get(a, 0.0))
                if a in _EFFICIENCY_ABILITIES
                else min(float(m.get(a, 0.0)), max_q)
            )
            for a in ABILITIES
        )
        if score > best_score:
            best_score, best_id = score, m["id"]

    return best_id, round(best_score, 4)


def best_model(models: list[dict[str, Any]] | None = None) -> str:
    """
    Returns the model with the highest average quality score.

    Speed and cost_efficiency excluded — used exclusively for the master agent.
    Always returns the objectively best model regardless of mode.

    Args:
        models: list of model dicts from models.yaml. Auto-loaded if None.

    Returns:
        model_id of the best model.
    """
    if models is None:
        models = load_models()

    if not models:
        raise ValueError("No models available in registry")

    best_id: str = models[0]["id"]
    best_avg: float = -1.0

    for m in models:
        avg = sum(float(m.get(a, 0.0)) for a in QUALITY_ABILITIES) / len(QUALITY_ABILITIES)
        if avg > best_avg:
            best_avg, best_id = avg, m["id"]

    return best_id


def model_for_agent(
    agent_cfg: dict[str, Any],
    models: list[dict[str, Any]] | None = None,
    mode: str = "balanced",
    fallback: str = "gpt-4o-mini",
) -> str:
    """
    Select a model for a named agent using its YAML ability profile + mode bias.

    Args:
        agent_cfg: Agent config dict from load_agent_cfg().
        models: list of model dicts. Auto-loaded if None.
        mode: "best", "balanced", or "cost_efficient".
        fallback: model ID to use if agent has no ability profile.

    Returns:
        model_id of the selected model.
    """
    abilities = agent_cfg.get("abilities")
    if not abilities:
        return fallback
    chosen, _ = select_model(abilities, models, mode)
    return chosen


def cost_usd_from_registry(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    models: list[dict[str, Any]] | None = None,
) -> float:
    """
    Calculate cost in USD using pricing from models.yaml.

    Falls back to 0.0 for models without pricing data.

    Args:
        model_id: The model ID (may include openai/ prefix from proxy).
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        models: Model registry. Auto-loaded if None.

    Returns:
        Cost in USD.
    """
    if models is None:
        models = load_models()

    # Strip proxy prefix if present
    clean_id = model_id.removeprefix("openai/")

    for m in models:
        if m["id"] == clean_id:
            inp = m.get("input_cost_per_1m")
            out = m.get("output_cost_per_1m")
            if inp is not None and out is not None:
                return (input_tokens * inp + output_tokens * out) / 1_000_000
            break

    return 0.0
