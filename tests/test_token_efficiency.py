"""
Phase 3.5 — Token Efficiency tests.

Covers three optimisations added in Phase 3.5:
  1. Anthropic prompt caching  — _is_anthropic_model() + cache_control injection
  2. Smart judge               — skip LLM call for simple single-task runs
  3. Shared context pruning    — _prune_shared_context() + calls in pipeline nodes

All tests use mocks so no real LLM calls are made.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ─── helpers to build fake litellm response ─────────────────────────────────


def _fake_resp(content: str = "ok", prompt_tokens: int = 10, completion_tokens: int = 5):
    """Minimal litellm response mock."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    msg = MagicMock()
    msg.content = content
    msg.refusal = None

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


# =============================================================================
# 1. Anthropic Prompt Caching
# =============================================================================


class TestIsAnthropicModel:
    """Unit tests for _is_anthropic_model()."""

    def test_claude_prefix_detected(self):
        from aifactory.graphs.dynamic_pipeline import _is_anthropic_model

        assert _is_anthropic_model("claude-3-5-haiku-20241022") is True

    def test_claude_opus_detected(self):
        from aifactory.graphs.dynamic_pipeline import _is_anthropic_model

        assert _is_anthropic_model("claude-3-opus-20240229") is True

    def test_anthropic_slash_prefix(self):
        from aifactory.graphs.dynamic_pipeline import _is_anthropic_model

        assert _is_anthropic_model("anthropic/claude-3-5-sonnet") is True

    def test_gpt4o_not_anthropic(self):
        from aifactory.graphs.dynamic_pipeline import _is_anthropic_model

        assert _is_anthropic_model("gpt-4o") is False

    def test_openai_slash_prefix_not_anthropic(self):
        from aifactory.graphs.dynamic_pipeline import _is_anthropic_model

        assert _is_anthropic_model("openai/gpt-4o") is False

    def test_proxy_claude_not_matched(self):
        """openai/claude-* proxy routing — should NOT be treated as native Anthropic."""
        from aifactory.graphs.dynamic_pipeline import _is_anthropic_model

        assert _is_anthropic_model("openai/claude-3-5-haiku") is False

    def test_gemini_not_anthropic(self):
        from aifactory.graphs.dynamic_pipeline import _is_anthropic_model

        assert _is_anthropic_model("gemini/gemini-pro") is False

    def test_case_insensitive(self):
        from aifactory.graphs.dynamic_pipeline import _is_anthropic_model

        assert _is_anthropic_model("Claude-3-Haiku") is True

    # ── OpenRouter Claude (all tiers) ────────────────────────────────────────

    def test_openrouter_anthropic_claude_detected(self):
        """OpenRouter-routed Claude models should match (passes cache_control through)."""
        from aifactory.graphs.dynamic_pipeline import _is_anthropic_model

        assert _is_anthropic_model("openrouter/anthropic/claude-3-5-haiku") is True

    def test_openrouter_anthropic_claude_opus(self):
        from aifactory.graphs.dynamic_pipeline import _is_anthropic_model

        assert _is_anthropic_model("openrouter/anthropic/claude-3-opus") is True

    def test_openrouter_openai_not_matched(self):
        from aifactory.graphs.dynamic_pipeline import _is_anthropic_model

        assert _is_anthropic_model("openrouter/openai/gpt-4o") is False

    def test_openrouter_disabled_via_param(self):
        """include_openrouter=False should exclude OpenRouter routes."""
        from aifactory.graphs.dynamic_pipeline import _is_anthropic_model

        assert (
            _is_anthropic_model("openrouter/anthropic/claude-3-5-haiku", include_openrouter=False)
            is False
        )

    def test_openrouter_disabled_still_matches_direct(self):
        """Disabling OpenRouter detection shouldn't affect direct Anthropic models."""
        from aifactory.graphs.dynamic_pipeline import _is_anthropic_model

        assert _is_anthropic_model("claude-3-5-haiku", include_openrouter=False) is True


class TestPromptCachingInjection:
    """PipelineAgent.think() injects cache_control for Claude models."""

    @pytest.fixture()
    def agent_cfg(self):
        return {"system_prompt": "You are a test agent.", "max_tokens": 100}

    async def test_anthropic_model_uses_cache_control(self, agent_cfg):
        """With a Claude model and no proxy, system content should be a list."""
        from aifactory.graphs.dynamic_pipeline import PipelineAgent

        agent = PipelineAgent("test", agent_cfg, model="claude-3-5-haiku-20241022")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _fake_resp("answer")
            with (
                patch("aifactory.graphs.dynamic_pipeline._api_base", ""),
                patch("aifactory.graphs.dynamic_pipeline._is_anthropic_model", return_value=True),
            ):
                await agent.think("hello", {})

        call_kwargs = mock_llm.call_args
        messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][1]
        system_msg = next(m for m in messages if m["role"] == "system")
        content = system_msg["content"]
        assert isinstance(content, list), "Anthropic model should use list content"
        assert content[0]["type"] == "text"
        assert content[0]["cache_control"] == {"type": "ephemeral"}

    async def test_openai_model_uses_string_system(self, agent_cfg):
        """With an OpenAI model, system content should be a plain string."""
        from aifactory.graphs.dynamic_pipeline import PipelineAgent

        agent = PipelineAgent("test", agent_cfg, model="gpt-4o-mini")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _fake_resp("answer")
            with (
                patch("aifactory.graphs.dynamic_pipeline._api_base", ""),
                patch("aifactory.graphs.dynamic_pipeline._is_anthropic_model", return_value=False),
            ):
                await agent.think("hello", {})

        call_kwargs = mock_llm.call_args
        messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][1]
        system_msg = next(m for m in messages if m["role"] == "system")
        content = system_msg["content"]
        assert isinstance(content, str), "OpenAI model should use string content"

    async def test_no_caching_when_proxy_active(self, agent_cfg):
        """When _api_base is set (proxy routing), skip cache_control even for Claude."""
        from aifactory.graphs.dynamic_pipeline import PipelineAgent

        agent = PipelineAgent("test", agent_cfg, model="claude-3-5-haiku-20241022")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _fake_resp("answer")
            # Simulate proxy being active
            with patch("aifactory.graphs.dynamic_pipeline._api_base", "http://proxy:4000"):
                await agent.think("hello", {})

        call_kwargs = mock_llm.call_args
        messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][1]
        system_msg = next(m for m in messages if m["role"] == "system")
        content = system_msg["content"]
        assert isinstance(content, str), "Proxy mode should not inject cache_control"

    async def test_no_caching_when_setting_disabled(self, agent_cfg):
        """When prompt_cache_enabled=False, skip cache_control even for Claude."""
        from aifactory.graphs.dynamic_pipeline import PipelineAgent

        agent = PipelineAgent("test", agent_cfg, model="claude-3-5-haiku-20241022")

        mock_settings = MagicMock()
        mock_settings.prompt_cache_enabled = False
        mock_settings.memory_context_top_k = 3

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _fake_resp("answer")
            with (
                patch("aifactory.graphs.dynamic_pipeline._api_base", ""),
                patch("aifactory.config.settings.get_settings", return_value=mock_settings),
            ):
                await agent.think("hello", {})

        call_kwargs = mock_llm.call_args
        messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][1]
        system_msg = next(m for m in messages if m["role"] == "system")
        content = system_msg["content"]
        assert isinstance(content, str), "Disabled setting: no cache_control"

    async def test_cache_content_contains_system_prompt(self, agent_cfg):
        """The cache_control content block must include the full system text."""
        from aifactory.graphs.dynamic_pipeline import PipelineAgent

        agent = PipelineAgent("test", agent_cfg, model="claude-3-5-haiku-20241022")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _fake_resp("answer")
            with (
                patch("aifactory.graphs.dynamic_pipeline._api_base", ""),
                patch("aifactory.graphs.dynamic_pipeline._is_anthropic_model", return_value=True),
            ):
                await agent.think("hello", {})

        messages = mock_llm.call_args[1]["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        block_text = system_msg["content"][0]["text"]
        assert "You are a test agent." in block_text


# =============================================================================
# 2. Smart Judge — skip LLM for simple runs
# =============================================================================


class TestSmartJudge:
    """make_judge_review() skips LLM call for single-task runs with no research."""

    def _make_registry(self, names: list[str]):
        from aifactory.core.agent_factory import AgentRegistry
        from aifactory.graphs.dynamic_pipeline import PipelineAgent

        reg = AgentRegistry()
        for n in names:
            ag = PipelineAgent(n, {"system_prompt": ""}, "gpt-4o-mini")
            ag._actions.append({"input": "x", "output": "y", "elapsed_s": 0.1})
            ag.total_tokens = 50
            ag.prompt_tokens = 30
            ag.completion_tokens = 20
            reg.register(ag)
        return reg

    def _simple_state(self, n_tasks: int = 1) -> dict[str, Any]:
        tasks = [
            {
                "task_id": f"t{i}",
                "agent": f"Worker-t{i}",
                "model": "gpt-4o-mini",
                "description": "do something",
                "result": "done",
                "skipped": False,
            }
            for i in range(1, n_tasks + 1)
        ]
        return {
            "user_query": "what is 2+2",
            "shared_context": {"objective": "what is 2+2"},
            "task_results": tasks,
            "final_response": "4",
            "research_summary": "",
            "analysis_research_summary": "",
        }

    async def test_single_task_skips_llm(self):
        """1 task + no research → judge skips LLM call."""
        reg = self._make_registry(["master", "judge", "analyser", "execution_lead", "Worker-t1"])
        judge_agent = reg.get("judge")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _fake_resp("{}")
            with patch("aifactory.graphs.dynamic_pipeline.DUMPS_DIR") as mock_dir:
                mock_dir.__truediv__ = lambda s, x: MagicMock(
                    __truediv__=lambda s2, y: MagicMock(mkdir=MagicMock(), write_text=MagicMock())
                )
                from aifactory.graphs.dynamic_pipeline import make_judge_review

                node = make_judge_review(judge_agent, reg, run_id="test-run")
                await node(self._simple_state(n_tasks=1))

        mock_llm.assert_not_called()

    async def test_multi_task_calls_llm(self):
        """2+ tasks → judge makes LLM call."""
        reg = self._make_registry(
            ["master", "judge", "analyser", "execution_lead", "Worker-t1", "Worker-t2"]
        )
        judge_agent = reg.get("judge")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _fake_resp(
                '{"assessments": [], "metrics": {"overall_status": "successful", '
                '"total_tokens": "100", "pipeline_efficiency": "high", '
                '"cost_rating": "3", "recommendations": "ok"}}'
            )
            with patch("aifactory.graphs.dynamic_pipeline.DUMPS_DIR") as mock_dir:
                mock_dir.__truediv__ = lambda s, x: MagicMock(
                    __truediv__=lambda s2, y: MagicMock(mkdir=MagicMock(), write_text=MagicMock())
                )
                from aifactory.graphs.dynamic_pipeline import make_judge_review

                node = make_judge_review(judge_agent, reg, run_id="test-run")
                await node(self._simple_state(n_tasks=2))

        mock_llm.assert_called_once()

    async def test_single_task_with_research_calls_llm(self):
        """Single task BUT research ran → judge makes full LLM call."""
        reg = self._make_registry(
            ["master", "judge", "analyser", "researcher", "execution_lead", "Worker-t1"]
        )
        judge_agent = reg.get("judge")

        state = self._simple_state(n_tasks=1)
        state["research_summary"] = "Found some info"

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _fake_resp(
                '{"assessments": [], "metrics": {"overall_status": "successful", '
                '"total_tokens": "200", "pipeline_efficiency": "medium", '
                '"cost_rating": "2", "recommendations": "ok"}}'
            )
            with patch("aifactory.graphs.dynamic_pipeline.DUMPS_DIR") as mock_dir:
                mock_dir.__truediv__ = lambda s, x: MagicMock(
                    __truediv__=lambda s2, y: MagicMock(mkdir=MagicMock(), write_text=MagicMock())
                )
                from aifactory.graphs.dynamic_pipeline import make_judge_review

                node = make_judge_review(judge_agent, reg, run_id="test-run")
                await node(state)

        mock_llm.assert_called_once()

    async def test_synthetic_assessment_has_all_agents(self):
        """Synthetic assessment must include one entry per registered agent."""
        agents = ["master", "judge", "analyser", "execution_lead", "Worker-t1"]
        reg = self._make_registry(agents)
        judge_agent = reg.get("judge")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _fake_resp("{}")
            with patch("aifactory.graphs.dynamic_pipeline.DUMPS_DIR") as mock_dir:
                mock_dir.__truediv__ = lambda s, x: MagicMock(
                    __truediv__=lambda s2, y: MagicMock(mkdir=MagicMock(), write_text=MagicMock())
                )
                from aifactory.graphs.dynamic_pipeline import make_judge_review

                node = make_judge_review(judge_agent, reg, run_id="test-run")
                result = await node(self._simple_state(n_tasks=1))

        found_agents = {a["agent"] for a in result["agent_assessments"]}
        assert found_agents == set(agents)

    async def test_synthetic_metrics_structure(self):
        """Synthetic metrics must have all required fields."""
        reg = self._make_registry(["master", "judge", "execution_lead", "Worker-t1"])
        judge_agent = reg.get("judge")

        with (
            patch("litellm.acompletion", new_callable=AsyncMock),
            patch("aifactory.graphs.dynamic_pipeline.DUMPS_DIR") as mock_dir,
        ):
            mock_dir.__truediv__ = lambda s, x: MagicMock(
                __truediv__=lambda s2, y: MagicMock(mkdir=MagicMock(), write_text=MagicMock())
            )
            from aifactory.graphs.dynamic_pipeline import make_judge_review

            node = make_judge_review(judge_agent, reg, run_id="test-run")
            result = await node(self._simple_state())

        m = result["judge_metrics"]
        assert "overall_status" in m
        assert "pipeline_efficiency" in m
        assert "cost_rating" in m
        assert "recommendations" in m
        assert "total_tokens" in m

    async def test_skip_can_be_disabled_via_setting(self):
        """With judge_skip_simple_runs=False, always calls LLM even for single task."""
        reg = self._make_registry(["master", "judge", "execution_lead", "Worker-t1"])
        judge_agent = reg.get("judge")

        mock_settings = MagicMock()
        mock_settings.judge_skip_simple_runs = False
        mock_settings.judge_deep_eval_enabled = False  # isolate: test skip setting only

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _fake_resp(
                '{"assessments": [], "metrics": {"overall_status": "successful", '
                '"total_tokens": "50", "pipeline_efficiency": "high", '
                '"cost_rating": "3", "recommendations": "ok"}}'
            )
            with patch("aifactory.graphs.dynamic_pipeline.DUMPS_DIR") as mock_dir:
                mock_dir.__truediv__ = lambda s, x: MagicMock(
                    __truediv__=lambda s2, y: MagicMock(mkdir=MagicMock(), write_text=MagicMock())
                )
                with patch("aifactory.config.settings.get_settings", return_value=mock_settings):
                    from aifactory.graphs.dynamic_pipeline import make_judge_review

                    node = make_judge_review(judge_agent, reg, run_id="test-run")
                    await node(self._simple_state())

        mock_llm.assert_called_once()


# =============================================================================
# 3. Shared Context Pruning
# =============================================================================


class TestSharedContextPruning:
    """Unit tests for _prune_shared_context()."""

    def test_no_pruning_under_limit(self):
        from aifactory.graphs.dynamic_pipeline import _prune_shared_context

        ctx = {f"key{i}": f"val{i}" for i in range(10)}
        result = _prune_shared_context(ctx, max_keys=25)
        assert result == ctx

    def test_exact_limit_unchanged(self):
        from aifactory.graphs.dynamic_pipeline import _prune_shared_context

        ctx = {f"k{i}": "v" for i in range(25)}
        result = _prune_shared_context(ctx, max_keys=25)
        assert len(result) == 25

    def test_prunes_excess_keys(self):
        from aifactory.graphs.dynamic_pipeline import _prune_shared_context

        ctx = {f"k{i}": "v" for i in range(30)}
        result = _prune_shared_context(ctx, max_keys=25)
        assert len(result) == 25

    def test_priority_keys_always_kept(self):
        from aifactory.graphs.dynamic_pipeline import _prune_shared_context

        # Fill with 30 non-priority keys + priority keys
        ctx = {f"other_{i}": "v" for i in range(30)}
        ctx["objective"] = "my goal"
        ctx["tasks_count"] = "3"
        ctx["connected_integrations"] = "github"
        result = _prune_shared_context(ctx, max_keys=10)
        assert "objective" in result
        assert "tasks_count" in result
        assert "connected_integrations" in result

    def test_most_recent_non_priority_kept(self):
        """When pruning, the last-inserted non-priority keys should survive."""
        from aifactory.graphs.dynamic_pipeline import _prune_shared_context

        ctx = {f"old_{i}": "v" for i in range(20)}  # these are old
        ctx["objective"] = "goal"
        ctx["new_a"] = "recent1"
        ctx["new_b"] = "recent2"
        # max_keys=5: keep objective + 4 most recent non-priority
        result = _prune_shared_context(ctx, max_keys=5)
        assert "new_a" in result
        assert "new_b" in result
        assert "objective" in result

    def test_empty_context(self):
        from aifactory.graphs.dynamic_pipeline import _prune_shared_context

        assert _prune_shared_context({}) == {}

    def test_priority_keys_alone_not_pruned(self):
        from aifactory.graphs.dynamic_pipeline import _prune_shared_context

        ctx = {"objective": "x", "tasks_count": "2"}
        result = _prune_shared_context(ctx, max_keys=1)
        # Both are priority — result should contain all priority keys
        assert "objective" in result
        assert "tasks_count" in result

    def test_result_is_new_dict(self):
        """Pruning should return a new dict, not modify in place."""
        from aifactory.graphs.dynamic_pipeline import _prune_shared_context

        ctx = {f"k{i}": "v" for i in range(30)}
        result = _prune_shared_context(ctx, max_keys=10)
        assert result is not ctx

    def test_no_excess_does_not_copy(self):
        """When under limit, return the original dict unchanged (no copy)."""
        from aifactory.graphs.dynamic_pipeline import _prune_shared_context

        ctx = {"a": "1", "b": "2"}
        result = _prune_shared_context(ctx, max_keys=25)
        assert result is ctx


class TestPruningInPipelineNodes:
    """Pruning is called inside pipeline node closures."""

    async def test_analyse_prunes_on_return(self):
        """make_analyse returns a pruned shared_context when keys are excessive."""
        from aifactory.graphs.dynamic_pipeline import PipelineAgent, make_analyse

        cfg = {"system_prompt": "You are an analyser.", "max_tokens": 200}
        agent = PipelineAgent("analyser", cfg, "gpt-4o-mini")

        # Build a shared_ctx that already has 30 keys
        bloated_ctx = {f"extra_{i}": f"v{i}" for i in range(30)}
        bloated_ctx["objective"] = "do stuff"

        analyser_output = (
            '{"direct_answer": "", "research_for_analysis": false, '
            '"research_for_analysis_query": "", "research_needed": false, '
            '"research_query": "", "tasks": [{"id": "t1", "description": "do it"}]}'
        )

        mock_settings = MagicMock()
        mock_settings.shared_context_max_keys = 25
        mock_settings.memory_context_top_k = 3
        mock_settings.prompt_cache_enabled = False

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _fake_resp(analyser_output)
            with patch("aifactory.config.settings.get_settings", return_value=mock_settings):
                node = make_analyse(agent)
                result = await node(
                    {
                        "user_query": "do stuff",
                        "shared_context": bloated_ctx,
                    }
                )

        assert len(result["shared_context"]) <= 25

    async def test_execution_lead_prunes_context(self):
        """make_execution_lead prunes shared_context before returning."""
        from aifactory.core.agent_factory import AgentRegistry
        from aifactory.graphs.dynamic_pipeline import (
            PipelineAgent,
            make_execution_lead,
        )

        cfg = {"system_prompt": "You are a lead.", "max_tokens": 500}
        lead = PipelineAgent("execution_lead", cfg, "gpt-4o-mini")
        worker_cfg = {
            "system_prompt": "You are a worker.",
            "max_tokens": 500,
            "role": "Worker",
        }
        reg = AgentRegistry()

        # Plan response + worker response + consolidation response
        plan_json = (
            '{"plan": [{"id": "t1", "reasoning": 2, "coding": 2, "math": 0, "writing": 1, '
            '"instruction_following": 2, "tool_use": 0, "knowledge": 1, "speed": 0, '
            '"cost_efficiency": 0, "context_handling": 1, "toolkits": []}], '
            '"prompts": {"t1": "do the task"}}'
        )
        responses = [_fake_resp(plan_json), _fake_resp("result here")]

        bloated_ctx = {f"old_{i}": f"v{i}" for i in range(30)}
        bloated_ctx["objective"] = "task"
        state = {
            "user_query": "task",
            "shared_context": bloated_ctx,
            "tasks": [{"id": "t1", "description": "do it"}],
            "research_summary": "",
        }

        mock_settings = MagicMock()
        mock_settings.shared_context_max_keys = 25
        mock_settings.memory_context_top_k = 3
        mock_settings.prompt_cache_enabled = False

        call_count = 0

        async def _fake_llm(*args, **kwargs):
            nonlocal call_count
            r = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return r

        with (
            patch("litellm.acompletion", side_effect=_fake_llm),
            patch("aifactory.config.settings.get_settings", return_value=mock_settings),
            patch("aifactory.graphs.dynamic_pipeline.load_available_models", return_value=[]),
            patch(
                "aifactory.graphs.dynamic_pipeline.create_worker_config",
                return_value={**worker_cfg, "model": "gpt-4o-mini"},
            ),
            patch(
                "aifactory.graphs.dynamic_pipeline.execute_code_blocks",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "aifactory.graphs.dynamic_pipeline.execute_integration_actions",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            node = make_execution_lead(lead, worker_cfg, [], "balanced", reg)
            result = await node(state)

        assert len(result["shared_context"]) <= 25


# =============================================================================
# 4. Settings integration
# =============================================================================


class TestTokenEfficiencySettings:
    """New settings fields exist with correct defaults."""

    def test_prompt_cache_enabled_default(self):
        from aifactory.config.settings import Settings

        s = Settings()
        assert s.prompt_cache_enabled is True

    def test_judge_skip_simple_runs_default(self):
        from aifactory.config.settings import Settings

        s = Settings()
        assert s.judge_skip_simple_runs is True

    def test_shared_context_max_keys_default(self):
        from aifactory.config.settings import Settings

        s = Settings()
        assert s.shared_context_max_keys == 25

    def test_response_cache_enabled_default(self):
        from aifactory.config.settings import Settings

        s = Settings()
        assert s.response_cache_enabled is True

    def test_response_cache_ttl_default(self):
        from aifactory.config.settings import Settings

        s = Settings()
        assert s.response_cache_ttl == 3600

    def test_prompt_cache_openrouter_default(self):
        from aifactory.config.settings import Settings

        s = Settings()
        assert s.prompt_cache_openrouter is True

    def test_can_override_via_env(self, monkeypatch):
        import importlib

        monkeypatch.setenv("AIFACTORY_PROMPT_CACHE_ENABLED", "false")
        monkeypatch.setenv("AIFACTORY_JUDGE_SKIP_SIMPLE_RUNS", "false")
        monkeypatch.setenv("AIFACTORY_SHARED_CONTEXT_MAX_KEYS", "10")
        monkeypatch.setenv("AIFACTORY_RESPONSE_CACHE_ENABLED", "false")
        monkeypatch.setenv("AIFACTORY_RESPONSE_CACHE_TTL", "600")
        monkeypatch.setenv("AIFACTORY_PROMPT_CACHE_OPENROUTER", "false")

        from aifactory.config import settings as _settings_module

        importlib.reload(_settings_module)
        s = _settings_module.Settings()
        assert s.prompt_cache_enabled is False
        assert s.judge_skip_simple_runs is False
        assert s.shared_context_max_keys == 10
        assert s.response_cache_enabled is False
        assert s.response_cache_ttl == 600
        assert s.prompt_cache_openrouter is False


class TestResponseCacheInit:
    """_init_response_cache() initialises litellm.cache for all providers."""

    def test_initialises_when_enabled(self):
        import litellm

        from aifactory.graphs.dynamic_pipeline import _init_response_cache

        mock_settings = MagicMock()
        mock_settings.response_cache_enabled = True
        mock_settings.response_cache_ttl = 3600

        original_cache = getattr(litellm, "cache", None)
        try:
            litellm.cache = None  # Reset
            with (
                patch("aifactory.config.settings.get_settings", return_value=mock_settings),
                patch("litellm.Cache") as mock_cache_cls,
            ):
                mock_cache_cls.return_value = MagicMock()
                _init_response_cache()
            assert litellm.cache is not None
        finally:
            litellm.cache = original_cache

    def test_skips_when_disabled(self):
        import litellm

        from aifactory.graphs.dynamic_pipeline import _init_response_cache

        mock_settings = MagicMock()
        mock_settings.response_cache_enabled = False

        original_cache = getattr(litellm, "cache", None)
        try:
            litellm.cache = None
            with patch("aifactory.config.settings.get_settings", return_value=mock_settings):
                _init_response_cache()
            assert litellm.cache is None
        finally:
            litellm.cache = original_cache

    def test_idempotent_does_not_reset_existing_cache(self):
        """If cache already set, _init_response_cache should not replace it."""
        import litellm

        from aifactory.graphs.dynamic_pipeline import _init_response_cache

        mock_settings = MagicMock()
        mock_settings.response_cache_enabled = True
        mock_settings.response_cache_ttl = 3600

        sentinel = MagicMock(name="existing_cache")
        original_cache = getattr(litellm, "cache", None)
        try:
            litellm.cache = sentinel
            with patch("aifactory.config.settings.get_settings", return_value=mock_settings):
                _init_response_cache()
            assert litellm.cache is sentinel  # unchanged
        finally:
            litellm.cache = original_cache

    def test_ttl_zero_passes_none(self):
        """TTL=0 should be passed as None (no expiry) to litellm.Cache."""
        import litellm

        from aifactory.graphs.dynamic_pipeline import _init_response_cache

        mock_settings = MagicMock()
        mock_settings.response_cache_enabled = True
        mock_settings.response_cache_ttl = 0

        original_cache = getattr(litellm, "cache", None)
        try:
            litellm.cache = None
            with (
                patch("aifactory.config.settings.get_settings", return_value=mock_settings),
                patch("litellm.Cache") as mock_cache_cls,
            ):
                mock_cache_cls.return_value = MagicMock()
                _init_response_cache()
            # Cache() should have been called with ttl=None
            mock_cache_cls.assert_called_once_with(ttl=None)
        finally:
            litellm.cache = original_cache

    def test_failure_is_silent(self):
        """Cache init failure must never raise — pipeline continues without cache."""
        import litellm

        from aifactory.graphs.dynamic_pipeline import _init_response_cache

        original_cache = getattr(litellm, "cache", None)
        try:
            litellm.cache = None
            with patch(
                "aifactory.config.settings.get_settings",
                side_effect=RuntimeError("settings broken"),
            ):
                # Should not raise
                _init_response_cache()
        finally:
            litellm.cache = original_cache


class TestOpenRouterPromptCaching:
    """OpenRouter-routed Claude models get cache_control when configured."""

    @pytest.fixture()
    def agent_cfg(self):
        return {"system_prompt": "You are a test agent.", "max_tokens": 100}

    async def test_openrouter_claude_uses_cache_control(self, agent_cfg):
        """openrouter/anthropic/claude-* should inject cache_control."""
        from aifactory.graphs.dynamic_pipeline import PipelineAgent

        agent = PipelineAgent("test", agent_cfg, model="openrouter/anthropic/claude-3-5-haiku")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _fake_resp("answer")
            with (
                patch("aifactory.graphs.dynamic_pipeline._api_base", ""),
                patch("aifactory.graphs.dynamic_pipeline._is_anthropic_model", return_value=True),
            ):
                await agent.think("hello", {})

        messages = mock_llm.call_args[1]["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert isinstance(system_msg["content"], list)
        assert system_msg["content"][0].get("cache_control") == {"type": "ephemeral"}

    async def test_openrouter_can_be_disabled(self, agent_cfg):
        """When prompt_cache_openrouter=False, OpenRouter Claude gets no cache_control."""
        from aifactory.graphs.dynamic_pipeline import PipelineAgent

        agent = PipelineAgent("test", agent_cfg, model="openrouter/anthropic/claude-3-5-haiku")

        mock_settings = MagicMock()
        mock_settings.prompt_cache_enabled = True
        mock_settings.prompt_cache_openrouter = False  # ← disabled
        mock_settings.memory_context_top_k = 3

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _fake_resp("answer")
            with (
                patch("aifactory.graphs.dynamic_pipeline._api_base", ""),
                patch("aifactory.config.settings.get_settings", return_value=mock_settings),
            ):
                await agent.think("hello", {})

        messages = mock_llm.call_args[1]["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        # With include_openrouter=False, openrouter/anthropic/claude-* is not Anthropic
        # So it gets string system content
        assert isinstance(system_msg["content"], str)


# =============================================================================
# 5. Regression — existing behaviour unchanged
# =============================================================================


class TestNonRegression:
    """Ensure Phase 3.5 changes do not break existing pipeline behaviour."""

    async def test_think_still_returns_reply(self):
        """think() return value is unaffected by prompt caching logic."""
        from aifactory.graphs.dynamic_pipeline import PipelineAgent

        cfg = {"system_prompt": "sys", "max_tokens": 100}
        agent = PipelineAgent("a", cfg, "gpt-4o-mini")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _fake_resp("the answer")
            with patch("aifactory.graphs.dynamic_pipeline._api_base", ""):
                reply = await agent.think("question", {})

        assert reply == "the answer"

    async def test_think_tracks_tokens(self):
        from aifactory.graphs.dynamic_pipeline import PipelineAgent

        cfg = {"system_prompt": "sys", "max_tokens": 100}
        agent = PipelineAgent("a", cfg, "gpt-4o-mini")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _fake_resp("reply", prompt_tokens=42, completion_tokens=13)
            with patch("aifactory.graphs.dynamic_pipeline._api_base", ""):
                await agent.think("question", {})

        assert agent.prompt_tokens == 42
        assert agent.completion_tokens == 13
        assert agent.total_tokens == 55

    def test_prune_preserves_all_values(self):
        """_prune_shared_context must not corrupt surviving values."""
        from aifactory.graphs.dynamic_pipeline import _prune_shared_context

        ctx = {f"k{i}": f"v{i}" for i in range(10)}
        result = _prune_shared_context(ctx, max_keys=25)
        for k, v in ctx.items():
            assert result[k] == v
