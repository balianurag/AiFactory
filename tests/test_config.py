"""
Tests for config/settings.py — settings loading and validation.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from aifactory.config.settings import Settings, get_settings

# =============================================================================
# Settings Defaults
# =============================================================================


class TestSettingsDefaults:
    def test_default_models(self, settings: Settings) -> None:
        assert settings.premium_model == "gpt-4o"
        assert settings.standard_model == "gpt-4o-mini"
        assert settings.budget_model == "gpt-4o-mini"

    def test_default_qdrant(self, settings: Settings) -> None:
        assert "localhost" in settings.qdrant_url or "http" in settings.qdrant_url
        assert settings.qdrant_collection_prefix == "aifactory"

    def test_default_memory_settings(self, settings: Settings) -> None:
        assert settings.memory_promotion_interval == 10
        assert settings.memory_max_agent_entries == 200
        assert settings.memory_max_team_entries == 500
        assert settings.memory_retrieval_top_k == 5
        assert settings.embedding_dimensions == 1536

    def test_default_hierarchy(self, settings: Settings) -> None:
        assert settings.max_subordinates_per_lead == 3
        assert settings.max_project_managers == 5
        assert settings.sr_pm_threshold == 2

    def test_default_budget(self, settings: Settings) -> None:
        assert settings.max_budget_usd == 5.00
        assert settings.max_iterations == 50
        assert settings.max_tokens_per_agent_turn == 2048
        assert settings.max_total_tokens == 500_000

    def test_default_rl(self, settings: Settings) -> None:
        assert settings.rl_min_score_for_promotion == 6.0
        assert settings.rl_judge_model == "gpt-4o-mini"

    def test_log_level(self, settings: Settings) -> None:
        assert settings.log_level == "INFO"


# =============================================================================
# Settings from Environment
# =============================================================================


class TestSettingsFromEnv:
    def test_override_premium_model(self) -> None:
        get_settings.cache_clear()
        with patch.dict(os.environ, {"AIFACTORY_PREMIUM_MODEL": "openai/gpt-4o"}):
            get_settings.cache_clear()
            s = Settings()
            assert s.premium_model == "openai/gpt-4o"

    def test_override_budget(self) -> None:
        with patch.dict(os.environ, {"AIFACTORY_MAX_BUDGET_USD": "10.0"}):
            s = Settings()
            assert s.max_budget_usd == 10.0

    def test_override_max_iterations(self) -> None:
        with patch.dict(os.environ, {"AIFACTORY_MAX_ITERATIONS": "100"}):
            s = Settings()
            assert s.max_iterations == 100

    def test_api_keys_default_empty(self, settings: Settings) -> None:
        # By default (no .env loaded in test), keys should be empty strings
        # or whatever the .env file provides
        assert isinstance(settings.openai_api_key, str)
        assert isinstance(settings.anthropic_api_key, str)
        assert isinstance(settings.xai_api_key, str)
        assert isinstance(settings.google_api_key, str)


# =============================================================================
# Singleton Behavior
# =============================================================================


class TestSingleton:
    def test_get_settings_returns_same_instance(self) -> None:
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_clear_returns_new_instance(self) -> None:
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        # Different object (but same values)
        assert s1 is not s2
