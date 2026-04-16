"""
Tests for core/model_selector.py — ability-based model scoring.
"""

from __future__ import annotations

import pytest

from aifactory.core.model_selector import (
    _EFFICIENCY_ABILITIES,
    ABILITIES,
    MODE_PROFILES,
    QUALITY_ABILITIES,
    best_model,
    cost_usd_from_registry,
    load_agent_cfg,
    load_graph_config,
    load_models,
    model_for_agent,
    select_model,
)

# ── Sample models for deterministic testing ──────────────────────────

_TEST_MODELS = [
    {
        "id": "model-premium",
        "reasoning": 3,
        "coding": 3,
        "math": 3,
        "writing": 3,
        "instruction_following": 3,
        "tool_use": 3,
        "knowledge": 3,
        "speed": 0,
        "cost_efficiency": 0,
        "context_handling": 3,
        "input_cost_per_1m": 5.00,
        "output_cost_per_1m": 25.00,
    },
    {
        "id": "model-balanced",
        "reasoning": 2,
        "coding": 2,
        "math": 2,
        "writing": 2,
        "instruction_following": 2,
        "tool_use": 2,
        "knowledge": 2,
        "speed": 2,
        "cost_efficiency": 2,
        "context_handling": 2,
        "input_cost_per_1m": 1.00,
        "output_cost_per_1m": 5.00,
    },
    {
        "id": "model-cheap",
        "reasoning": 1,
        "coding": 1,
        "math": 0,
        "writing": 1,
        "instruction_following": 1,
        "tool_use": 1,
        "knowledge": 1,
        "speed": 3,
        "cost_efficiency": 3,
        "context_handling": 1,
        "input_cost_per_1m": 0.10,
        "output_cost_per_1m": 0.40,
    },
]


# =============================================================================
# Constants
# =============================================================================


class TestConstants:
    def test_abilities_count(self) -> None:
        assert len(ABILITIES) == 10

    def test_quality_abilities_excludes_speed_cost(self) -> None:
        assert "speed" not in QUALITY_ABILITIES
        assert "cost_efficiency" not in QUALITY_ABILITIES
        assert len(QUALITY_ABILITIES) == 8

    def test_mode_profiles_exist(self) -> None:
        assert "best" in MODE_PROFILES
        assert "balanced" in MODE_PROFILES
        assert "cost_efficient" in MODE_PROFILES

    def test_mode_profiles_have_alpha(self) -> None:
        assert MODE_PROFILES["best"]["alpha"] == 1.0
        assert MODE_PROFILES["balanced"]["alpha"] == 0.35
        assert MODE_PROFILES["cost_efficient"]["alpha"] == 0.75

    def test_mode_profiles_have_max_quality(self) -> None:
        """Each mode should declare a max_quality cap."""
        assert MODE_PROFILES["best"]["max_quality"] == 3
        assert MODE_PROFILES["balanced"]["max_quality"] == 2
        assert MODE_PROFILES["cost_efficient"]["max_quality"] == 1

    def test_efficiency_abilities_set(self) -> None:
        """_EFFICIENCY_ABILITIES contains speed and cost_efficiency only."""
        assert {"speed", "cost_efficiency"} == _EFFICIENCY_ABILITIES

    def test_max_quality_cap_in_balanced_mode(self) -> None:
        """In balanced mode, a model with quality=3 should be scored the same as quality=2."""
        # model-cap-test: quality abilities all=3, speed=0, cost=0
        # model-at-cap: quality abilities all=2, speed=0, cost=0
        # With max_quality=2, both should score identically (tied → first wins)
        models_cap = [
            {
                "id": "model-above-cap",
                **{a: 3 for a in QUALITY_ABILITIES},
                "speed": 0,
                "cost_efficiency": 0,
            },
            {
                "id": "model-at-cap",
                **{a: 2 for a in QUALITY_ABILITIES},
                "speed": 0,
                "cost_efficiency": 0,
            },
        ]
        # With no requirements, mode profile drives selection. Both models have
        # equal post-cap quality abilities — first model wins on tie-break.
        _, _ = select_model({}, models_cap, mode="balanced")
        # Scores must be equal since 3 is capped to 2
        _, score_above = select_model({}, [models_cap[0]], mode="balanced")
        _, score_at = select_model({}, [models_cap[1]], mode="balanced")
        assert abs(score_above - score_at) < 1e-6

    def test_efficiency_abilities_uncapped(self) -> None:
        """Speed and cost_efficiency are never capped, even in cost_efficient mode."""
        # model-fast: speed=3, cost=3, quality=1
        # In cost_efficient mode, max_quality=1, so quality abilities capped at 1.
        # But speed/cost pass through uncapped.
        models_eff = [
            {
                "id": "model-fast",
                **{a: 1 for a in QUALITY_ABILITIES},
                "speed": 3,
                "cost_efficiency": 3,
            },
        ]
        mid, score = select_model({}, models_eff, mode="cost_efficient")
        assert mid == "model-fast"
        assert score > 0


# =============================================================================
# select_model()
# =============================================================================


class TestSelectModel:
    def test_best_mode_picks_premium(self) -> None:
        """In 'best' mode, the highest-quality model should win."""
        mid, score = select_model(
            requirements={a: 2 for a in ABILITIES},
            models=_TEST_MODELS,
            mode="best",
        )
        assert mid == "model-premium"
        assert score > 0

    def test_cost_efficient_mode_picks_cheap(self) -> None:
        """In 'cost_efficient' mode, the cheapest/fastest model should win."""
        mid, score = select_model(
            requirements={a: 2 for a in ABILITIES},
            models=_TEST_MODELS,
            mode="cost_efficient",
        )
        assert mid == "model-cheap"
        assert score > 0

    def test_task_specific_requirements(self) -> None:
        """When task needs high coding, premium should win in best mode (no quality cap)."""
        mid, _score = select_model(
            requirements={"coding": 3, "reasoning": 3, "tool_use": 3},
            models=_TEST_MODELS,
            mode="best",
        )
        assert mid == "model-premium"

    def test_returns_tuple(self) -> None:
        mid, score = select_model(
            requirements={"reasoning": 1},
            models=_TEST_MODELS,
        )
        assert isinstance(mid, str)
        assert isinstance(score, float)

    def test_empty_requirements_still_selects(self) -> None:
        """Even with empty requirements, mode profile provides scores."""
        mid, score = select_model(
            requirements={},
            models=_TEST_MODELS,
            mode="best",
        )
        assert mid == "model-premium"
        assert score > 0

    def test_raises_on_empty_models(self) -> None:
        with pytest.raises(ValueError, match="No models"):
            select_model(requirements={"reasoning": 1}, models=[])

    def test_unknown_mode_falls_back_to_balanced(self) -> None:
        """Unknown mode name should fall back to balanced (max_quality=2 cap applied)."""
        mid, _score = select_model(
            requirements={a: 3 for a in ABILITIES},
            models=_TEST_MODELS,
            mode="nonexistent",
        )
        # balanced mode applies max_quality=2 cap — quality abilities are equalised,
        # so model-balanced wins due to its speed=2 + cost_efficiency=2 advantage.
        assert mid == "model-balanced"


# =============================================================================
# best_model()
# =============================================================================


class TestBestModel:
    def test_picks_highest_quality(self) -> None:
        """best_model excludes speed/cost, so premium should always win."""
        mid = best_model(models=_TEST_MODELS)
        assert mid == "model-premium"

    def test_ignores_speed_cost(self) -> None:
        """A model with high speed/cost but low quality should not win."""
        # model-cheap has speed=3, cost_efficiency=3 but low quality
        mid = best_model(models=_TEST_MODELS)
        assert mid != "model-cheap"

    def test_raises_on_empty_models(self) -> None:
        with pytest.raises(ValueError, match="No models"):
            best_model(models=[])


# =============================================================================
# model_for_agent()
# =============================================================================


class TestModelForAgent:
    def test_uses_ability_profile(self) -> None:
        """Agent with ability profile should get a model selected by scoring."""
        cfg = {"abilities": {"reasoning": 3, "coding": 3, "tool_use": 3}}
        # Use 'best' mode so max_quality=3 (no cap) and premium wins on quality
        mid = model_for_agent(cfg, models=_TEST_MODELS, mode="best")
        assert mid == "model-premium"

    def test_fallback_without_abilities(self) -> None:
        """Agent without ability profile should get the fallback model."""
        cfg = {"role": "test"}
        mid = model_for_agent(cfg, models=_TEST_MODELS, fallback="gpt-4o-mini")
        assert mid == "gpt-4o-mini"


# =============================================================================
# cost_usd_from_registry()
# =============================================================================


class TestCostFromRegistry:
    def test_known_model_pricing(self) -> None:
        # model-premium: $5.00/$25.00 per 1M tokens
        cost = cost_usd_from_registry(
            "model-premium", input_tokens=1000, output_tokens=500, models=_TEST_MODELS
        )
        expected = (1000 * 5.00 + 500 * 25.00) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_unknown_model_returns_zero(self) -> None:
        cost = cost_usd_from_registry(
            "nonexistent-model", input_tokens=100, output_tokens=100, models=_TEST_MODELS
        )
        assert cost == 0.0

    def test_strips_openai_prefix(self) -> None:
        """Should strip openai/ prefix (used by LiteLLM proxy routing)."""
        cost = cost_usd_from_registry(
            "openai/model-cheap", input_tokens=1000, output_tokens=1000, models=_TEST_MODELS
        )
        expected = (1000 * 0.10 + 1000 * 0.40) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_zero_tokens(self) -> None:
        cost = cost_usd_from_registry(
            "model-premium", input_tokens=0, output_tokens=0, models=_TEST_MODELS
        )
        assert cost == 0.0


# =============================================================================
# YAML loaders (integration — real config files)
# =============================================================================


class TestYamlLoaders:
    def test_load_models_returns_list(self) -> None:
        models = load_models()
        assert isinstance(models, list)
        assert len(models) > 0
        # Each model should have an id and ability scores
        for m in models[:3]:
            assert "id" in m
            assert "reasoning" in m

    def test_load_graph_config(self) -> None:
        cfg = load_graph_config()
        assert "model" in cfg
        assert "mode" in cfg
        assert cfg["mode"] in ("best", "balanced", "cost_efficient")

    def test_load_agent_cfg_master(self) -> None:
        cfg = load_agent_cfg("master")
        assert "system_prompt" in cfg
        assert "max_tokens" in cfg

    def test_load_agent_cfg_analyser(self) -> None:
        cfg = load_agent_cfg("analyser")
        assert "abilities" in cfg
        assert "system_prompt" in cfg
        assert cfg["abilities"]["reasoning"] >= 0

    def test_load_agent_cfg_worker(self) -> None:
        cfg = load_agent_cfg("worker")
        assert "abilities" in cfg
        assert "max_tokens" in cfg
