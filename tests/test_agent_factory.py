"""
Tests for core/agent_factory.py — SHARE: protocol, JSON parsing, agent creation.
"""

from __future__ import annotations

import pytest

from aifactory.core.agent_factory import (
    AgentRegistry,
    build_shared_context_block,
    create_pipeline_agents,
    create_worker_config,
    extract_share,
    merge_share,
    parse_json_robust,
)

# =============================================================================
# SHARE: protocol
# =============================================================================


class TestExtractShare:
    def test_no_share_block(self) -> None:
        result, updates = extract_share("Just a normal response")
        assert result == "Just a normal response"
        assert updates == {}

    def test_basic_share(self) -> None:
        raw = "Here is my work.\nSHARE:\nframework: FastAPI\ndb: PostgreSQL"
        result, updates = extract_share(raw)
        assert result == "Here is my work."
        assert updates == {"framework": "FastAPI", "db": "PostgreSQL"}

    def test_share_preserves_result(self) -> None:
        raw = "Line 1\nLine 2\nSHARE:\nkey: value"
        result, updates = extract_share(raw)
        assert result == "Line 1\nLine 2"
        assert updates == {"key": "value"}

    def test_empty_share_block(self) -> None:
        raw = "Result\nSHARE:\n"
        result, updates = extract_share(raw)
        assert result == "Result"
        assert updates == {}

    def test_multiple_colons_in_value(self) -> None:
        raw = "Work done\nSHARE:\nurl: http://example.com:8080"
        result, updates = extract_share(raw)
        assert updates == {"url": "http://example.com:8080"}


class TestMergeShare:
    def test_merge_adds_keys(self) -> None:
        ctx = {"a": "1"}
        result = merge_share(ctx, {"b": "2"})
        assert result == {"a": "1", "b": "2"}

    def test_merge_overwrites(self) -> None:
        ctx = {"a": "old"}
        result = merge_share(ctx, {"a": "new"})
        assert result == {"a": "new"}

    def test_empty_updates_returns_same(self) -> None:
        ctx = {"a": "1"}
        result = merge_share(ctx, {})
        assert result is ctx  # same object, not copied


class TestBuildSharedContextBlock:
    def test_empty_context(self) -> None:
        assert build_shared_context_block({}) == ""

    def test_formats_context(self) -> None:
        block = build_shared_context_block({"framework": "FastAPI", "db": "PostgreSQL"})
        assert "## Shared Context" in block
        assert "framework: FastAPI" in block
        assert "db: PostgreSQL" in block

    def test_skips_empty_values(self) -> None:
        block = build_shared_context_block({"key": "", "real": "value"})
        assert "key:" not in block
        assert "real: value" in block


# =============================================================================
# JSON parsing
# =============================================================================


class TestParseJsonRobust:
    def test_clean_json(self) -> None:
        result = parse_json_robust('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_code_fences(self) -> None:
        raw = '```json\n{"key": "value"}\n```'
        result = parse_json_robust(raw)
        assert result == {"key": "value"}

    def test_json_with_leading_prose(self) -> None:
        raw = 'Here is the result:\n{"key": "value"}'
        result = parse_json_robust(raw)
        assert result == {"key": "value"}

    def test_json_with_share_block(self) -> None:
        raw = '{"key": "value"}\nSHARE:\nfoo: bar'
        result = parse_json_robust(raw)
        assert result == {"key": "value"}

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError, match="no valid JSON"):
            parse_json_robust("not json at all")


# =============================================================================
# AgentRegistry
# =============================================================================


class TestAgentRegistry:
    def test_register_and_all(self) -> None:
        reg = AgentRegistry()
        assert reg.count() == 0
        reg.register({"name": "agent1"})
        reg.register({"name": "agent2"})
        assert reg.count() == 2
        assert len(reg.all()) == 2

    def test_get_by_name(self) -> None:
        """get() searches for .name attribute."""

        class FakeAgent:
            def __init__(self, name: str):
                self.name = name

        reg = AgentRegistry()
        a1 = FakeAgent("master")
        a2 = FakeAgent("judge")
        reg.register(a1)
        reg.register(a2)
        assert reg.get("master") is a1
        assert reg.get("judge") is a2
        assert reg.get("nonexistent") is None


# =============================================================================
# create_pipeline_agents() — integration test with real YAML configs
# =============================================================================


class TestCreatePipelineAgents:
    def test_creates_all_named_agents(self) -> None:
        agents = create_pipeline_agents(mode="balanced")
        assert "master" in agents
        assert "judge" in agents
        assert "analyser" in agents
        assert "researcher" in agents
        assert "execution_lead" in agents
        # Each agent should have a model assigned
        for name, cfg in agents.items():
            assert "model" in cfg, f"{name} missing model"
            assert cfg["model"], f"{name} has empty model"

    def test_master_uses_best_model(self) -> None:
        """Master should always get best_model regardless of mode."""
        agents_balanced = create_pipeline_agents(mode="balanced")
        agents_cheap = create_pipeline_agents(mode="cost_efficient")
        # Master model should be the same in both modes (always best)
        assert agents_balanced["master"]["model"] == agents_cheap["master"]["model"]

    def test_cost_efficient_differs_from_best(self) -> None:
        """Non-master agents should get different models in different modes."""
        agents_best = create_pipeline_agents(mode="best")
        agents_cheap = create_pipeline_agents(mode="cost_efficient")
        # At least one non-master agent should differ
        differ = any(
            agents_best[n]["model"] != agents_cheap[n]["model"]
            for n in ["analyser", "researcher", "execution_lead"]
        )
        assert differ, "Expected mode to affect model selection for non-master agents"


# =============================================================================
# create_worker_config() — per-task model selection
# =============================================================================


class TestCreateWorkerConfig:
    def test_creates_worker_with_model(self) -> None:
        cfg = create_worker_config(
            task_id="task_1",
            task_description="Build a REST API",
            mode="balanced",
        )
        assert cfg["model"]
        assert cfg["_name"] == "Worker-task_1"
        assert "QUESTION:" in cfg["system_prompt"]
        assert "NOT_NEEDED" in cfg["system_prompt"]

    def test_custom_ability_requirements(self) -> None:
        cfg = create_worker_config(
            task_id="task_1",
            task_description="Write code",
            ability_requirements={"coding": 3, "reasoning": 3, "tool_use": 3},
            mode="best",
        )
        assert cfg["model"]

    def test_custom_worker_prompt(self) -> None:
        cfg = create_worker_config(
            task_id="task_1",
            task_description="Research topic",
            worker_prompt="Research the following topic thoroughly.",
            mode="balanced",
        )
        assert "Research the following topic thoroughly" in cfg["system_prompt"]
        assert "QUESTION:" in cfg["system_prompt"]

    def test_model_score_included(self) -> None:
        cfg = create_worker_config(
            task_id="task_1",
            task_description="Code task",
            mode="balanced",
        )
        assert "model_score" in cfg
        assert isinstance(cfg["model_score"], float)
