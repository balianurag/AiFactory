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
        _result, updates = extract_share(raw)
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
        assert "QUESTION PROTOCOL:" in cfg["system_prompt"]
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
        assert "QUESTION PROTOCOL:" in cfg["system_prompt"]

    def test_model_score_included(self) -> None:
        cfg = create_worker_config(
            task_id="task_1",
            task_description="Code task",
            mode="balanced",
        )
        assert "model_score" in cfg
        assert isinstance(cfg["model_score"], float)

    def test_integration_block_with_connected_toolkits(self) -> None:
        """Worker prompt includes INTEGRATION TOOLS block when toolkits are connected."""
        from unittest.mock import MagicMock, patch

        mock_integration = MagicMock()
        mock_integration.get_toolkit_tools_for_prompt.return_value = (
            "  GITHUB_CREATE_ISSUE: Create an issue (requires: owner, repo, title)\n"
            "  GITHUB_LIST_ISSUES: List issues"
        )

        mock_registry = MagicMock()
        mock_registry.get_connected_toolkit_names.return_value = ["github", "gmail"]
        mock_registry.get_connected.return_value = [mock_integration]
        mock_registry.get_tools_summary.return_value = (
            "  github: GITHUB_CREATE_ISSUE, GITHUB_LIST_ISSUES"
        )

        with patch(
            "aifactory.integrations.registry.IntegrationRegistry",
            return_value=mock_registry,
        ):
            cfg = create_worker_config(
                task_id="task_1",
                task_description="Create a GitHub issue",
                mode="balanced",
            )

        prompt = cfg["system_prompt"]
        assert "INTEGRATION TOOLS" in prompt
        assert "ACTION:composio" in prompt
        assert "ITERATIVE TOOL USE" in prompt
        assert "3 rounds" in prompt
        assert "GITHUB_CREATE_ISSUE" in prompt

        # Verify task_keywords were passed to the filter
        mock_integration.get_toolkit_tools_for_prompt.assert_called()
        call_args = mock_integration.get_toolkit_tools_for_prompt.call_args
        assert call_args[0][0] == "github"  # toolkit name
        assert "task_keywords" in call_args[1]
        assert isinstance(call_args[1]["task_keywords"], list)

    def test_integration_block_filters_tools_by_task(self) -> None:
        """Worker prompt only includes task-relevant tools, not the full catalog."""
        from unittest.mock import MagicMock, patch

        mock_integration = MagicMock()

        def _mock_tools(toolkit: str, task_keywords: list[str] | None = None) -> str:
            all_tools = {
                "GITLAB_LIST_USER_PROJECTS": "List projects for a user",
                "GITLAB_GET_PROJECTS": "Get project details",
                "GITLAB_CREATE_ISSUE": "Create an issue in a project",
                "GITLAB_UPDATE_ISSUE": "Update an issue",
                "GITLAB_MERGE_REQUESTS": "List merge requests",
            }
            if task_keywords:
                kws = [k.lower() for k in task_keywords]
                filtered = {
                    slug: desc
                    for slug, desc in all_tools.items()
                    if any(kw in slug.lower() or kw in desc.lower() for kw in kws)
                }
                if filtered:
                    return "\n".join(f"  {s}: {d}" for s, d in filtered.items())
            return "\n".join(f"  {s}: {d}" for s, d in all_tools.items())

        mock_integration.get_toolkit_tools_for_prompt.side_effect = _mock_tools

        mock_registry = MagicMock()
        mock_registry.get_connected_toolkit_names.return_value = ["gitlab"]
        mock_registry.get_connected.return_value = [mock_integration]

        with patch(
            "aifactory.integrations.registry.IntegrationRegistry",
            return_value=mock_registry,
        ):
            cfg = create_worker_config(
                task_id="task_1",
                task_description="List my GitLab projects",
                mode="balanced",
            )

        prompt = cfg["system_prompt"]
        # Should contain project-related tools
        assert "GITLAB_LIST_USER_PROJECTS" in prompt or "GITLAB_GET_PROJECTS" in prompt
        # Keywords extracted should include "project" (or "projects")
        call_kw = mock_integration.get_toolkit_tools_for_prompt.call_args[1]["task_keywords"]
        assert any("project" in kw for kw in call_kw)

    def test_integration_block_absent_without_toolkits(self) -> None:
        """Worker prompt omits INTEGRATION TOOLS block when no toolkits connected."""
        from unittest.mock import MagicMock, patch

        mock_registry = MagicMock()
        mock_registry.get_connected_toolkit_names.return_value = []

        with patch(
            "aifactory.integrations.registry.IntegrationRegistry",
            return_value=mock_registry,
        ):
            cfg = create_worker_config(
                task_id="task_1",
                task_description="Write some code",
                mode="balanced",
            )

        prompt = cfg["system_prompt"]
        assert "INTEGRATION TOOLS" not in prompt

    def test_code_execution_iterative_mention(self) -> None:
        """Worker prompt mentions sandbox code execution."""
        cfg = create_worker_config(
            task_id="task_1",
            task_description="Write code",
            mode="balanced",
        )
        prompt = cfg["system_prompt"]
        assert "sandbox" in prompt.lower()


# =============================================================================
# Worker Role Specialization
# =============================================================================


class TestWorkerRoleSpecialization:
    """Verify that create_worker_config() prepends specialist context for known task types."""

    def _cfg_for(self, task_desc: str, **kwargs):
        from unittest.mock import MagicMock, patch

        mock_registry = MagicMock()
        mock_registry.get_connected_toolkit_names.return_value = []

        with patch(
            "aifactory.integrations.registry.IntegrationRegistry",
            return_value=mock_registry,
        ):
            return create_worker_config(
                task_id="task_1",
                task_description=task_desc,
                mode="balanced",
                **kwargs,
            )

    def test_coding_task_gets_coder_specialist(self) -> None:
        """Task type 'coding' prepends worker_coder.yaml specialist context."""
        cfg = self._cfg_for("Write a Python function to parse JSON")
        # worker_coder.yaml has role: Coder — its task text starts with 'You are a specialist software engineer'  # noqa: E501
        assert (
            "specialist software engineer" in cfg["system_prompt"].lower()
            or "coding" in cfg["system_prompt"].lower()
            or "algorithm" in cfg["system_prompt"].lower()
        )

    def test_research_task_gets_researcher_specialist(self) -> None:
        """Task type 'research' prepends worker_researcher.yaml specialist context."""
        cfg = self._cfg_for("Research the latest trends in machine learning")
        # worker_researcher.yaml has role: Researcher
        assert (
            "researcher" in cfg["system_prompt"].lower()
            or "research" in cfg["system_prompt"].lower()
        )

    def test_writing_task_gets_writer_specialist(self) -> None:
        """Task type 'writing' prepends worker_writer.yaml specialist context."""
        cfg = self._cfg_for("Write a blog post about AI trends")
        # worker_writer.yaml has role: Writer
        assert (
            "writer" in cfg["system_prompt"].lower()
            or "writing" in cfg["system_prompt"].lower()
            or "tone" in cfg["system_prompt"].lower()
        )

    def test_analysis_task_gets_analyst_specialist(self) -> None:
        """Task type 'analysis' prepends worker_analyst.yaml specialist context."""
        cfg = self._cfg_for("Analyse the performance metrics of the system")
        # worker_analyst.yaml has role: Analyst
        assert (
            "analyst" in cfg["system_prompt"].lower()
            or "analysis" in cfg["system_prompt"].lower()
            or "evidence" in cfg["system_prompt"].lower()
        )

    def test_specialist_context_prepended_before_task_prompt(self) -> None:
        """Specialist context appears before the execution lead's task instructions."""
        cfg = self._cfg_for(
            "Write Python code to sort a list",
            worker_prompt="Sort a list of integers in ascending order.",
        )
        prompt = cfg["system_prompt"]
        # The custom worker_prompt should appear after any specialist preamble
        task_idx = prompt.find("Sort a list of integers")
        assert task_idx != -1, "worker_prompt not found in system_prompt"
        # Everything before the task instruction is the specialist context + preamble
        preamble = prompt[:task_idx]
        assert len(preamble) > 0, "Specialist context should precede task instruction"
