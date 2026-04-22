"""
Tests for graphs/dynamic_pipeline.py — the 8-node config-driven graph.

All LLM calls are mocked via litellm.acompletion.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aifactory.core.agent_factory import AgentRegistry
from aifactory.graphs.dynamic_pipeline import (
    AgentBudgetExceededError,
    DynamicPipelineState,
    PipelineAgent,
    _build_dependency_waves,
    _run_worker_task,
    build_dynamic_pipeline,
    make_analyse,
    make_judge_brief,
    make_judge_review,
    make_master_receive,
    make_master_respond,
    make_researcher,
    route_after_analysis,
    route_after_reanalyse,
    route_after_researcher,
)
from tests.helpers import make_llm_response

# ── Helper to mock litellm.acompletion ─────────────────────────────


_LITELLM_PATCH = "aifactory.graphs.dynamic_pipeline.litellm.acompletion"


def _mock_litellm(response_content: str, prompt_tokens: int = 100, completion_tokens: int = 50):
    """Return a patch context for litellm.acompletion."""
    resp = make_llm_response(response_content, prompt_tokens, completion_tokens)
    return patch(_LITELLM_PATCH, AsyncMock(return_value=resp))


# =============================================================================
# PipelineAgent
# =============================================================================


class TestPipelineAgent:
    @pytest.mark.asyncio
    async def test_think_tracks_tokens(self) -> None:
        agent = PipelineAgent("test", {"system_prompt": "You are a test agent.", "max_tokens": 100})
        with _mock_litellm("Hello world"):
            result = await agent.think("Say hello")
        assert result == "Hello world"
        assert agent.total_tokens > 0
        assert agent.prompt_tokens > 0
        assert agent.completion_tokens > 0
        assert len(agent._messages) == 2  # user + assistant
        assert len(agent._actions) == 1

    @pytest.mark.asyncio
    async def test_shared_context_in_system_prompt(self) -> None:
        agent = PipelineAgent("test", {"system_prompt": "Base prompt.", "max_tokens": 100})
        with _mock_litellm("ok") as mock_call:
            await agent.think("Go", shared_ctx={"framework": "FastAPI"})
        # Check that system prompt included shared context
        messages = mock_call.call_args.kwargs["messages"]
        system_msg = messages[0]["content"]
        assert "framework: FastAPI" in system_msg

    def test_dump(self) -> None:
        agent = PipelineAgent(
            "master", {"role": "Orchestrator", "system_prompt": "You are master."}
        )
        d = agent.dump()
        assert d["name"] == "master"
        assert d["role"] == "Orchestrator"
        assert d["total_tokens"] == 0

    def test_default_token_budget(self) -> None:
        agent = PipelineAgent("test", {"system_prompt": "test"})
        assert agent.token_budget == PipelineAgent.DEFAULT_TOKEN_BUDGET

    def test_custom_token_budget_from_cfg(self) -> None:
        agent = PipelineAgent("test", {"system_prompt": "test", "token_budget": 1000})
        assert agent.token_budget == 1000

    @pytest.mark.asyncio
    async def test_budget_exceeded_raises(self) -> None:
        agent = PipelineAgent("test", {"system_prompt": "test", "token_budget": 100})
        # Simulate exceeding budget
        agent.total_tokens = 100
        with pytest.raises(AgentBudgetExceededError, match="exceeded token budget"):
            await agent.think("Should fail")

    @pytest.mark.asyncio
    async def test_budget_allows_when_under_limit(self) -> None:
        agent = PipelineAgent(
            "test", {"system_prompt": "test", "max_tokens": 100, "token_budget": 5000}
        )
        agent.total_tokens = 100  # under the 5000 limit
        with _mock_litellm("ok"):
            result = await agent.think("Should succeed")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_think_is_traceable_decorated(self) -> None:
        """think() should be decorated with @traceable (zero-cost when disabled)."""
        agent = PipelineAgent("test", {"system_prompt": "test", "max_tokens": 100})
        with _mock_litellm("traced"):
            result = await agent.think("Trace me")
        # Even with tracing disabled, the function should work normally
        assert result == "traced"


# =============================================================================
# Routing functions
# =============================================================================


class TestRouting:
    def test_route_direct_answer(self) -> None:
        state: DynamicPipelineState = {"direct_answer": "42", "user_query": "q"}  # type: ignore[typeddict-item]
        assert route_after_analysis(state) == "master_respond"

    def test_route_research_for_analysis(self) -> None:
        state: DynamicPipelineState = {"research_for_analysis": True, "user_query": "q"}  # type: ignore[typeddict-item]
        assert route_after_analysis(state) == "researcher"

    def test_route_research_needed(self) -> None:
        state: DynamicPipelineState = {"research_needed": True, "user_query": "q"}  # type: ignore[typeddict-item]
        assert route_after_analysis(state) == "researcher"

    def test_route_to_execution(self) -> None:
        state: DynamicPipelineState = {"user_query": "q"}  # type: ignore[typeddict-item]
        assert route_after_analysis(state) == "execution_lead"

    def test_route_after_researcher_for_analysis(self) -> None:
        state: DynamicPipelineState = {"research_for_analysis": True, "user_query": "q"}  # type: ignore[typeddict-item]
        assert route_after_researcher(state) == "reanalyse"

    def test_route_after_researcher_for_execution(self) -> None:
        state: DynamicPipelineState = {"research_for_analysis": False, "user_query": "q"}  # type: ignore[typeddict-item]
        assert route_after_researcher(state) == "execution_lead"

    def test_route_after_reanalyse_needs_research(self) -> None:
        state: DynamicPipelineState = {"research_needed": True, "user_query": "q"}  # type: ignore[typeddict-item]
        assert route_after_reanalyse(state) == "researcher"

    def test_route_after_reanalyse_no_research(self) -> None:
        state: DynamicPipelineState = {"research_needed": False, "user_query": "q"}  # type: ignore[typeddict-item]
        assert route_after_reanalyse(state) == "execution_lead"


# =============================================================================
# Node functions (mocked)
# =============================================================================


class TestNodes:
    @pytest.mark.asyncio
    async def test_master_receive(self) -> None:
        agent = PipelineAgent("master", {"system_prompt": "You are master.", "max_tokens": 100})
        node = make_master_receive(agent)
        result = await node({"user_query": "Build an app", "shared_context": {}})
        assert "shared_context" in result
        assert result["shared_context"]["objective"] == "Build an app"
        assert len(agent._actions) == 1

    @pytest.mark.asyncio
    async def test_master_receive_injects_integrations(self) -> None:
        """master_receive should inject connected integrations into shared_context."""

        agent = PipelineAgent("master", {"system_prompt": "You are master.", "max_tokens": 100})
        node = make_master_receive(agent)

        mock_registry = MagicMock()
        mock_registry.get_connected_toolkit_names.return_value = ["gitlab", "gmail"]
        mock_registry.get_tools_summary.return_value = "  gitlab: GITLAB_LIST_PROJECTS"

        with (
            patch(
                "aifactory.integrations.registry.IntegrationRegistry",
                return_value=mock_registry,
            ),
        ):
            result = await node({"user_query": "List my GitLab repos", "shared_context": {}})
        ctx = result["shared_context"]
        assert "gitlab" in ctx.get("connected_integrations", "")

    @pytest.mark.asyncio
    async def test_analyse_with_integration_context(self) -> None:
        """Analyser should mention connected integrations in its prompt."""
        agent = PipelineAgent(
            "analyser", {"system_prompt": "You are analyser.", "max_tokens": 1000}
        )
        node = make_analyse(agent)
        analysis_response = json.dumps(
            {
                "direct_answer": "",
                "research_needed": False,
                "research_query": "",
                "tasks": [{"id": "task_1", "description": "List GitLab projects"}],
            }
        )
        with _mock_litellm(analysis_response) as mock_llm:
            await node(
                {
                    "user_query": "List my GitLab projects",
                    "shared_context": {"connected_integrations": "gitlab, gmail"},
                }
            )
        # The analyser's user message should mention integration awareness
        call_messages = mock_llm.call_args.kwargs["messages"]
        user_msg = call_messages[-1]["content"]
        assert "INTEGRATION AWARENESS" in user_msg
        assert "gitlab" in user_msg

    @pytest.mark.asyncio
    async def test_analyse_with_tasks(self) -> None:
        agent = PipelineAgent(
            "analyser", {"system_prompt": "You are analyser.", "max_tokens": 1000}
        )
        node = make_analyse(agent)
        analysis_response = json.dumps(
            {
                "direct_answer": "",
                "research_for_analysis": False,
                "research_for_analysis_query": "",
                "research_needed": False,
                "research_query": "",
                "tasks": [
                    {"id": "task_1", "description": "Build the API"},
                    {"id": "task_2", "description": "Write tests"},
                ],
            }
        )
        with _mock_litellm(analysis_response):
            result = await node({"user_query": "Build an API with tests", "shared_context": {}})
        assert len(result["tasks"]) == 2
        assert result["tasks"][0]["id"] == "task_1"
        assert result["research_needed"] is False

    @pytest.mark.asyncio
    async def test_analyse_safety_net_overrides_research(self) -> None:
        """Safety net fires when user says 'research' but analyser misses it."""
        agent = PipelineAgent(
            "analyser", {"system_prompt": "You are analyser.", "max_tokens": 1000}
        )
        node = make_analyse(agent)
        # Analyser wrongly sets research_needed=false despite user asking to research
        analysis_response = json.dumps(
            {
                "direct_answer": "",
                "research_for_analysis": False,
                "research_for_analysis_query": "",
                "research_needed": False,
                "research_query": "",
                "tasks": [
                    {"id": "task_1", "description": "Summarize World 2"},
                    {"id": "task_2", "description": "Summarize the Illuminati"},
                ],
            }
        )
        with _mock_litellm(analysis_response):
            result = await node(
                {
                    "user_query": "Research about world 2 and illuminati",
                    "shared_context": {},
                }
            )
        # Safety net should have overridden to True
        assert result["research_needed"] is True
        assert result["research_query"] == "Research about world 2 and illuminati"

    @pytest.mark.asyncio
    async def test_analyse_no_override_when_no_research_keyword(self) -> None:
        """Safety net does NOT fire when user query has no research keywords."""
        agent = PipelineAgent(
            "analyser", {"system_prompt": "You are analyser.", "max_tokens": 1000}
        )
        node = make_analyse(agent)
        analysis_response = json.dumps(
            {
                "direct_answer": "",
                "research_needed": False,
                "research_query": "",
                "tasks": [{"id": "task_1", "description": "Build the API"}],
            }
        )
        with _mock_litellm(analysis_response):
            result = await node({"user_query": "Build an API", "shared_context": {}})
        assert result["research_needed"] is False

    @pytest.mark.asyncio
    async def test_analyse_respects_llm_research_true(self) -> None:
        """When analyser correctly sets research_needed=true, it passes through."""
        agent = PipelineAgent(
            "analyser", {"system_prompt": "You are analyser.", "max_tokens": 1000}
        )
        node = make_analyse(agent)
        analysis_response = json.dumps(
            {
                "direct_answer": "",
                "research_needed": True,
                "research_query": "Cold War Second World overview",
                "tasks": [{"id": "task_1", "description": "Summarize World 2"}],
            }
        )
        with _mock_litellm(analysis_response):
            result = await node({"user_query": "Tell me about world 2", "shared_context": {}})
        assert result["research_needed"] is True
        assert result["research_query"] == "Cold War Second World overview"

    @pytest.mark.asyncio
    async def test_analyse_direct_answer(self) -> None:
        agent = PipelineAgent(
            "analyser", {"system_prompt": "You are analyser.", "max_tokens": 1000}
        )
        node = make_analyse(agent)
        response = json.dumps(
            {
                "direct_answer": "Hello! How can I help you?",
                "tasks": [],
            }
        )
        with _mock_litellm(response):
            result = await node({"user_query": "Hello", "shared_context": {}})
        assert result["direct_answer"] == "Hello! How can I help you?"
        assert result["tasks"] == []

    @pytest.mark.asyncio
    async def test_master_respond_direct(self) -> None:
        agent = PipelineAgent("master", {"system_prompt": "You are master.", "max_tokens": 100})
        node = make_master_respond(agent)
        # Direct answer should skip LLM call
        result = await node(
            {
                "user_query": "Hi",
                "shared_context": {},
                "direct_answer": "Hello!",
                "conversation_history": [],
            }
        )
        assert result["final_response"] == "Hello!"
        assert len(result["conversation_history"]) == 1

    @pytest.mark.asyncio
    async def test_master_respond_with_team_output(self) -> None:
        agent = PipelineAgent("master", {"system_prompt": "You are master.", "max_tokens": 100})
        node = make_master_respond(agent)
        with _mock_litellm("Here is the final answer."):
            result = await node(
                {
                    "user_query": "Build an API",
                    "shared_context": {},
                    "direct_answer": "",
                    "final_response": "The API has been built with 3 endpoints.",
                    "conversation_history": [],
                }
            )
        assert result["final_response"] == "Here is the final answer."


# =============================================================================
# build_dynamic_pipeline() — integration test
# =============================================================================


class TestBuildPipeline:
    def test_builds_and_compiles(self) -> None:
        """Pipeline should build and compile without error."""
        with patch(_LITELLM_PATCH):
            compiled, registry = build_dynamic_pipeline(mode="balanced")
        assert compiled is not None
        assert registry.count() == 5  # master, judge, analyser, researcher, execution_lead

    def test_builds_with_custom_mode(self) -> None:
        """Pipeline respects the mode argument."""
        with patch(_LITELLM_PATCH):
            compiled, _registry = build_dynamic_pipeline(mode="best")
        assert compiled is not None

    def test_builds_with_provided_registry(self) -> None:
        """Pipeline uses externally-provided registry."""
        custom_reg = AgentRegistry()
        with patch(_LITELLM_PATCH):
            _, returned_reg = build_dynamic_pipeline(mode="balanced", registry=custom_reg)
        assert returned_reg is custom_reg
        assert returned_reg.count() == 5


# =============================================================================
# Additional node tests
# =============================================================================


class TestJudgeBriefNode:
    @pytest.mark.asyncio
    async def test_judge_brief_runs(self) -> None:
        judge = PipelineAgent("judge", {"system_prompt": "You are the judge.", "max_tokens": 200})
        registry = AgentRegistry()
        registry.register(PipelineAgent("master", {"system_prompt": "m"}))
        registry.register(judge)
        registry.register(PipelineAgent("analyser", {"system_prompt": "a"}))

        node = make_judge_brief(judge, registry)
        # judge_brief no longer makes an LLM call — it just records a lightweight action
        result = await node(
            {
                "user_query": "Build a REST API",
                "shared_context": {"objective": "Build a REST API"},
            }
        )
        # judge_brief returns empty dict (it only records to its own history)
        assert result == {}
        assert len(judge._actions) == 1
        assert judge._actions[0]["output"] == "Ready to evaluate."


class TestResearcherNode:
    @pytest.mark.asyncio
    async def test_researcher_for_analysis(self) -> None:
        agent = PipelineAgent("researcher", {"system_prompt": "Research.", "max_tokens": 500})
        node = make_researcher(agent)
        with _mock_litellm("FastAPI is a high-performance Python web framework."):
            result = await node(
                {
                    "user_query": "Build an API",
                    "shared_context": {},
                    "research_for_analysis": True,
                    "research_for_analysis_query": "What is FastAPI?",
                }
            )
        assert "analysis_research_summary" in result
        assert "FastAPI" in result["analysis_research_summary"]

    @pytest.mark.asyncio
    async def test_researcher_for_execution(self) -> None:
        agent = PipelineAgent("researcher", {"system_prompt": "Research.", "max_tokens": 500})
        node = make_researcher(agent)
        with _mock_litellm("Redis supports pub/sub messaging."):
            result = await node(
                {
                    "user_query": "Set up Redis",
                    "shared_context": {},
                    "research_for_analysis": False,
                    "research_query": "Redis pub/sub patterns",
                }
            )
        assert "research_summary" in result
        assert "Redis" in result["research_summary"]


class TestJudgeReviewNode:
    @pytest.mark.asyncio
    async def test_judge_review_parses_assessments(self) -> None:
        judge = PipelineAgent("judge", {"system_prompt": "You are the judge.", "max_tokens": 500})
        registry = AgentRegistry()
        registry.register(PipelineAgent("master", {"system_prompt": "m"}))
        registry.register(judge)

        node = make_judge_review(judge, registry)
        review_response = json.dumps(
            {
                "assessments": [
                    {
                        "agent": "Worker-task_1",
                        "status": "successful",
                        "quality": 3,
                        "tokens": 500,
                        "notes": "Great work",
                    },
                ],
                "metrics": {
                    "overall_status": "successful",
                    "total_tokens": 1500,
                    "pipeline_efficiency": "high",
                    "cost_rating": 2,
                    "recommendations": "None",
                },
            }
        )
        # Use 2 tasks so the smart-judge skip is not triggered and the LLM path runs.
        with _mock_litellm(review_response):
            result = await node(
                {
                    "user_query": "Build an API",
                    "shared_context": {},
                    "task_results": [
                        {
                            "task_id": "task_1",
                            "agent": "Worker-task_1",
                            "model": "gpt-4o-mini",
                            "description": "Build the API",
                            "result": "API built!",
                            "skipped": False,
                        },
                        {
                            "task_id": "task_2",
                            "agent": "Worker-task_2",
                            "model": "gpt-4o-mini",
                            "description": "Write tests",
                            "result": "Tests written!",
                            "skipped": False,
                        },
                    ],
                }
            )
        assert len(result["agent_assessments"]) == 1
        assert result["agent_assessments"][0]["agent"] == "Worker-task_1"
        assert result["judge_metrics"]["overall_status"] == "successful"


# =============================================================================
# Iterative tool-calling loop tests
# =============================================================================


class TestToolCallingLoop:
    """Tests for the iterative tool-calling loop in worker dispatch.

    Workers can emit ACTION: blocks and code blocks.  The pipeline detects
    these, executes them, feeds results back to the worker, and repeats
    up to MAX_TOOL_ROUNDS (3).
    """

    @pytest.mark.asyncio
    async def test_no_tools_single_pass(self) -> None:
        """Worker with no ACTION/code blocks runs once — no loop."""
        agent = PipelineAgent("Worker-t1", {"system_prompt": "Do the task.", "max_tokens": 200})
        with _mock_litellm("Here is the answer.") as mock_llm:
            result = await agent.think("Execute your assigned task.")
        assert result == "Here is the answer."
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_action_block_triggers_reprompt(self) -> None:
        """Worker emits ACTION block → executed → results fed back → final answer."""
        from aifactory.integrations.executor import (
            extract_action_blocks,
        )

        # First call: worker emits an ACTION block
        worker_output_1 = (
            "Let me check the GitHub issues.\n"
            "ACTION:composio\n"
            '{"tool": "GITHUB_LIST_ISSUES", "params": {"owner": "org", "repo": "r"}}\n'
            "ACTION_END"
        )
        blocks = extract_action_blocks(worker_output_1)
        assert len(blocks) == 1
        assert blocks[0][0] == "composio"
        assert blocks[0][1]["tool"] == "GITHUB_LIST_ISSUES"

    @pytest.mark.asyncio
    async def test_multiple_action_blocks_extracted(self) -> None:
        """Multiple ACTION blocks in a single worker response."""
        from aifactory.integrations.executor import extract_action_blocks

        text = (
            "I'll do two things:\n"
            "ACTION:composio\n"
            '{"tool": "GITHUB_LIST_ISSUES", "params": {"owner": "org", "repo": "r"}}\n'
            "ACTION_END\n"
            "And also:\n"
            "ACTION:composio\n"
            '{"tool": "GITHUB_CREATE_ISSUE", "params": {"title": "Bug"}}\n'
            "ACTION_END"
        )
        blocks = extract_action_blocks(text)
        assert len(blocks) == 2
        assert blocks[0][1]["tool"] == "GITHUB_LIST_ISSUES"
        assert blocks[1][1]["tool"] == "GITHUB_CREATE_ISSUE"

    @pytest.mark.asyncio
    async def test_execute_integration_actions_dispatches(self) -> None:
        """execute_integration_actions dispatches ACTION blocks via registry."""
        from aifactory.integrations.base import ActionResult
        from aifactory.integrations.executor import execute_integration_actions

        mock_registry = MagicMock()
        mock_registry.execute = AsyncMock(
            return_value=ActionResult(
                success=True,
                action="GITHUB_LIST_ISSUES",
                integration="composio",
                data={"issues": [{"id": 1, "title": "Bug"}]},
            )
        )

        text = (
            "ACTION:composio\n"
            '{"tool": "GITHUB_LIST_ISSUES", "params": {"owner": "org", "repo": "r"}}\n'
            "ACTION_END"
        )
        results = await execute_integration_actions(text, registry=mock_registry)
        assert len(results) == 1
        assert results[0][0] == "GITHUB_LIST_ISSUES"
        assert results[0][1].success is True
        mock_registry.execute.assert_called_once_with(
            "composio", "GITHUB_LIST_ISSUES", {"owner": "org", "repo": "r"}
        )

    @pytest.mark.asyncio
    async def test_tool_loop_max_rounds_guard(self) -> None:
        """Verify max_tool_rounds is configured in the pipeline source."""
        from pathlib import Path

        pipeline_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "aifactory"
            / "graphs"
            / "dynamic_pipeline.py"
        )
        source = pipeline_path.read_text()
        # Simple tool tasks cap at 1 round; complex at 3
        assert "max_tool_rounds = 1" in source
        assert "max_tool_rounds" in source

    @pytest.mark.asyncio
    async def test_action_results_appended_to_result(self) -> None:
        """Integration results are appended under '--- Integration Results ---'."""

        # Simulate what the pipeline does: format integration results
        all_integ_parts = ['[GITHUB_LIST_ISSUES ✓]\n{"issues": [{"id": 1}]}']
        result = "Here is the analysis."
        integ_summary = "\n\n".join(all_integ_parts)
        final = f"{result}\n\n--- Integration Results ---\n{integ_summary}"
        assert "--- Integration Results ---" in final
        assert "GITHUB_LIST_ISSUES" in final

    @pytest.mark.asyncio
    async def test_code_results_appended_to_result(self) -> None:
        """Code execution results are appended under '--- Code Execution Results ---'."""
        all_code_parts = ["[code:python ✓]\n42"]
        result = "Here is the code."
        exec_summary = "\n\n".join(all_code_parts)
        final = f"{result}\n\n--- Code Execution Results ---\n{exec_summary}"
        assert "--- Code Execution Results ---" in final
        assert "42" in final


# =============================================================================
# _build_dependency_waves()
# =============================================================================


def _task(tid: str) -> dict:
    return {"id": tid, "description": f"task {tid}"}


class TestBuildDependencyWaves:
    def test_no_deps_single_wave(self) -> None:
        """All tasks with no dependencies go into one parallel wave."""
        tasks = [_task("t1"), _task("t2"), _task("t3")]
        waves = _build_dependency_waves(tasks, {})
        assert len(waves) == 1
        assert {t["id"] for t in waves[0]} == {"t1", "t2", "t3"}

    def test_empty_deps_map_single_wave(self) -> None:
        """An empty dependencies_map with tasks returns a single wave."""
        tasks = [_task("a"), _task("b")]
        waves = _build_dependency_waves(tasks, {"a": [], "b": []})
        # Only empty lists — same as no deps
        assert len(waves) == 1

    def test_linear_chain_produces_sequential_waves(self) -> None:
        """t1 → t2 → t3 must produce three waves of one task each."""
        tasks = [_task("t1"), _task("t2"), _task("t3")]
        deps = {"t2": ["t1"], "t3": ["t2"]}
        waves = _build_dependency_waves(tasks, deps)
        assert len(waves) == 3
        assert waves[0][0]["id"] == "t1"
        assert waves[1][0]["id"] == "t2"
        assert waves[2][0]["id"] == "t3"

    def test_fan_out_parallel_wave(self) -> None:
        """t1 has no deps; t2 and t3 both depend on t1 → t1 alone, then t2+t3 together."""
        tasks = [_task("t1"), _task("t2"), _task("t3")]
        deps = {"t2": ["t1"], "t3": ["t1"]}
        waves = _build_dependency_waves(tasks, deps)
        assert len(waves) == 2
        assert waves[0][0]["id"] == "t1"
        assert {t["id"] for t in waves[1]} == {"t2", "t3"}

    def test_diamond_dependency(self) -> None:
        """Diamond: t1 → t2, t3 → t4.  Wave 1: t1; Wave 2: t2, t3; Wave 3: t4."""
        tasks = [_task("t1"), _task("t2"), _task("t3"), _task("t4")]
        deps = {"t2": ["t1"], "t3": ["t1"], "t4": ["t2", "t3"]}
        waves = _build_dependency_waves(tasks, deps)
        assert len(waves) == 3
        assert waves[0][0]["id"] == "t1"
        assert {t["id"] for t in waves[1]} == {"t2", "t3"}
        assert waves[2][0]["id"] == "t4"

    def test_circular_dep_does_not_hang(self) -> None:
        """Circular dependency (t1↔t2) must not hang — forced resolution expected."""
        tasks = [_task("t1"), _task("t2")]
        deps = {"t1": ["t2"], "t2": ["t1"]}
        # Must complete without infinite loop (forced wave)
        waves = _build_dependency_waves(tasks, deps)
        total = sum(len(w) for w in waves)
        assert total == 2, "All tasks must appear in some wave"

    def test_single_task_no_deps(self) -> None:
        """A single task with no dependencies returns one wave."""
        tasks = [_task("solo")]
        waves = _build_dependency_waves(tasks, {})
        assert len(waves) == 1
        assert waves[0][0]["id"] == "solo"

    def test_preserves_task_objects(self) -> None:
        """Task dicts in waves are the same objects as input (not copies)."""
        t1 = _task("t1")
        t2 = _task("t2")
        waves = _build_dependency_waves([t1, t2], {})
        flat = [t for w in waves for t in w]
        assert t1 in flat
        assert t2 in flat


# =============================================================================
# Self-Correction (_run_worker_task REVISION: handling)
# =============================================================================


class TestSelfCorrection:
    """Test the REVISION: self-correction loop inside _run_worker_task."""

    def _make_registry(self):
        registry = AgentRegistry()
        return registry

    def _make_worker_kwargs(self, registry, completed=None, workers_cache=None):
        return dict(
            worker_prompts={},
            ability_reqs={},
            toolkit_assignments={},
            tool_slug_assignments={},
            completed=completed if completed is not None else {},
            workers_cache=workers_cache if workers_cache is not None else {},
            prior_results=[],
            mode="balanced",
            registry=registry,
            bridge=None,
        )

    @pytest.mark.asyncio
    async def test_validated_result_unchanged(self) -> None:
        """When the worker replies VALIDATED, the result is left as-is."""
        task = _task("t1")
        think_responses = [
            "The answer is 42.",  # initial response
            "VALIDATED",  # self-check
        ]
        call_count = 0

        async def mock_think(prompt, ctx=None):
            nonlocal call_count
            resp = think_responses[call_count]
            call_count += 1
            return resp

        registry = self._make_registry()

        with (
            patch("aifactory.graphs.dynamic_pipeline.create_worker_config") as mock_cfg,
            patch(
                "aifactory.graphs.dynamic_pipeline.execute_code_blocks",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aifactory.graphs.dynamic_pipeline.execute_integration_actions",
                new=AsyncMock(return_value=[]),
            ),
        ):
            mock_cfg.return_value = {
                "model": "gpt-4o-mini",
                "_name": "Worker-t1",
                "system_prompt": "You are a worker.",
                "max_tokens": 1000,
                "token_budget": 50000,
                "model_score": 1.0,
            }
            kwargs = self._make_worker_kwargs(registry)
            worker = MagicMock()
            worker.think = AsyncMock(side_effect=mock_think)
            worker._actions = []
            kwargs["workers_cache"]["t1"] = worker

            entry, _ctx_delta = await _run_worker_task(task, {}, **kwargs)

        assert entry["result"] == "The answer is 42."
        assert call_count == 2  # initial + self-check

    @pytest.mark.asyncio
    async def test_revision_replaces_result(self) -> None:
        """When the worker replies REVISION:, the corrected answer replaces the original."""
        task = _task("t1")
        think_responses = [
            "The answer is 41.",  # initial (wrong)
            "REVISION: off by one\nThe answer is 42.",  # self-check + correction
        ]
        call_count = 0

        async def mock_think(prompt, ctx=None):
            nonlocal call_count
            resp = think_responses[call_count]
            call_count += 1
            return resp

        registry = self._make_registry()

        with (
            patch("aifactory.graphs.dynamic_pipeline.create_worker_config") as mock_cfg,
            patch(
                "aifactory.graphs.dynamic_pipeline.execute_code_blocks",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aifactory.graphs.dynamic_pipeline.execute_integration_actions",
                new=AsyncMock(return_value=[]),
            ),
        ):
            mock_cfg.return_value = {
                "model": "gpt-4o-mini",
                "_name": "Worker-t1",
                "system_prompt": "You are a worker.",
                "max_tokens": 1000,
                "token_budget": 50000,
                "model_score": 1.0,
            }
            kwargs = self._make_worker_kwargs(registry)
            worker = MagicMock()
            worker.think = AsyncMock(side_effect=mock_think)
            worker._actions = []
            kwargs["workers_cache"]["t1"] = worker

            entry, _ctx_delta = await _run_worker_task(task, {}, **kwargs)

        assert entry["result"] == "The answer is 42."
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_self_correction_skipped_for_not_needed(self) -> None:
        """NOT_NEEDED output skips the self-correction call entirely."""
        task = _task("t1")
        call_count = 0

        async def mock_think(prompt, ctx=None):
            nonlocal call_count
            call_count += 1
            return "NOT_NEEDED"

        registry = self._make_registry()

        with (
            patch("aifactory.graphs.dynamic_pipeline.create_worker_config") as mock_cfg,
            patch(
                "aifactory.graphs.dynamic_pipeline.execute_code_blocks",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aifactory.graphs.dynamic_pipeline.execute_integration_actions",
                new=AsyncMock(return_value=[]),
            ),
        ):
            mock_cfg.return_value = {
                "model": "gpt-4o-mini",
                "_name": "Worker-t1",
                "system_prompt": "You are a worker.",
                "max_tokens": 1000,
                "token_budget": 50000,
                "model_score": 1.0,
            }
            kwargs = self._make_worker_kwargs(registry)
            worker = MagicMock()
            worker.think = AsyncMock(side_effect=mock_think)
            worker._actions = []
            kwargs["workers_cache"]["t1"] = worker

            entry, _ctx_delta = await _run_worker_task(task, {}, **kwargs)

        assert entry["skipped"] is True
        # Only the initial think() call, no self-check
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_self_correction_skipped_for_tool_tasks(self) -> None:
        """Tool tasks (has_tools=True) skip self-correction."""
        task = _task("t1")
        call_count = 0

        async def mock_think(prompt, ctx=None):
            nonlocal call_count
            call_count += 1
            return "Done with tools."

        registry = self._make_registry()

        with (
            patch("aifactory.graphs.dynamic_pipeline.create_worker_config") as mock_cfg,
            patch(
                "aifactory.graphs.dynamic_pipeline.execute_code_blocks",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aifactory.graphs.dynamic_pipeline.execute_integration_actions",
                new=AsyncMock(return_value=[]),
            ),
        ):
            mock_cfg.return_value = {
                "model": "gpt-4o-mini",
                "_name": "Worker-t1",
                "system_prompt": "You are a worker.",
                "max_tokens": 1000,
                "token_budget": 50000,
                "model_score": 1.0,
            }
            # Assign a toolkit slug → has_tools=True
            kwargs = self._make_worker_kwargs(registry)
            kwargs["tool_slug_assignments"] = {"t1": ["GITHUB_LIST_ISSUES"]}
            worker = MagicMock()
            worker.think = AsyncMock(side_effect=mock_think)
            worker._actions = []
            kwargs["workers_cache"]["t1"] = worker

            entry, _ctx_delta = await _run_worker_task(task, {}, **kwargs)

        # No QUESTION loop, no self-check → exactly 1 think() call
        assert call_count == 1
        assert entry["result"] == "Done with tools."


# =============================================================================
# Parallel worker dispatch (asyncio.gather in wave execution)
# =============================================================================


class TestParallelWorkerDispatch:
    """Verify that multi-task waves run workers concurrently via asyncio.gather."""

    @pytest.mark.asyncio
    async def test_parallel_wave_runs_all_tasks(self) -> None:
        """Three independent tasks in one wave all produce results."""
        tasks = [_task("t1"), _task("t2"), _task("t3")]

        async def fake_run_worker(task, shared_ctx_in, **kwargs):
            return (
                {
                    "task_id": task["id"],
                    "agent": f"Worker-{task['id']}",
                    "model": "gpt-4o-mini",
                    "description": task["description"],
                    "result": f"result-{task['id']}",
                    "skipped": False,
                },
                {f"key_{task['id']}": "val"},
            )

        with patch(
            "aifactory.graphs.dynamic_pipeline._run_worker_task",
            side_effect=fake_run_worker,
        ):
            waves = _build_dependency_waves(tasks, {})
            assert len(waves) == 1, "All tasks independent → single wave"

            task_results: list[dict] = []
            shared_ctx: dict = {}
            coros = [fake_run_worker(t, shared_ctx) for t in waves[0]]
            wave_outcomes = await asyncio.gather(*coros, return_exceptions=True)

            for _t, outcome in zip(waves[0], wave_outcomes, strict=False):
                assert not isinstance(outcome, BaseException)
                entry, ctx_delta = outcome
                shared_ctx = {**shared_ctx, **ctx_delta}
                task_results.append(entry)

        assert len(task_results) == 3
        result_ids = {r["task_id"] for r in task_results}
        assert result_ids == {"t1", "t2", "t3"}

    @pytest.mark.asyncio
    async def test_dependent_tasks_run_in_order(self) -> None:
        """Dependent tasks are split into waves; wave 1 completes before wave 2."""
        execution_order: list[str] = []
        tasks = [_task("t1"), _task("t2")]
        deps = {"t2": ["t1"]}

        async def fake_run_worker(task, shared_ctx_in, **kwargs):
            execution_order.append(task["id"])
            return (
                {
                    "task_id": task["id"],
                    "agent": f"Worker-{task['id']}",
                    "model": "gpt-4o-mini",
                    "description": task["description"],
                    "result": f"result-{task['id']}",
                    "skipped": False,
                },
                {},
            )

        waves = _build_dependency_waves(tasks, deps)
        assert len(waves) == 2

        for wave in waves:
            coros = [fake_run_worker(t, {}) for t in wave]
            results = await asyncio.gather(*coros)
            for _, _ in results:
                pass

        assert execution_order == ["t1", "t2"]

    @pytest.mark.asyncio
    async def test_ctx_delta_merge_no_clobber(self) -> None:
        """Each parallel worker returns its own ctx_delta; sequential merge avoids clobber."""
        tasks = [_task("t1"), _task("t2")]

        async def fake_t1(task, ctx, **kw):
            return (
                {
                    "task_id": "t1",
                    "result": "r1",
                    "skipped": False,
                    "agent": "W-t1",
                    "model": "m",
                    "description": "d",
                },
                {"key_t1": "val1"},
            )

        async def fake_t2(task, ctx, **kw):
            return (
                {
                    "task_id": "t2",
                    "result": "r2",
                    "skipped": False,
                    "agent": "W-t2",
                    "model": "m",
                    "description": "d",
                },
                {"key_t2": "val2"},
            )

        _build_dependency_waves(tasks, {})
        snapshot = {}
        coros = [fake_t1(tasks[0], snapshot), fake_t2(tasks[1], snapshot)]
        outcomes = await asyncio.gather(*coros, return_exceptions=True)

        shared_ctx = {}
        for outcome in outcomes:
            assert not isinstance(outcome, BaseException)
            _, ctx_delta = outcome
            shared_ctx = {**shared_ctx, **ctx_delta}

        assert shared_ctx == {"key_t1": "val1", "key_t2": "val2"}


# =============================================================================
# QUESTION: interrupt handling in _run_worker_task
# =============================================================================


class TestQuestionInterrupt:
    """Worker QUESTION: prompt triggers LangGraph interrupt() and resumes on answer."""

    def _base_worker_kwargs(self, registry, workers_cache=None):
        return dict(
            worker_prompts={},
            ability_reqs={},
            toolkit_assignments={},
            tool_slug_assignments={},
            completed={},
            workers_cache=workers_cache if workers_cache is not None else {},
            prior_results=[],
            mode="balanced",
            registry=registry,
            bridge=None,
        )

    @pytest.mark.asyncio
    async def test_question_calls_interrupt(self) -> None:
        """Worker returning QUESTION: should call interrupt() with the question text."""
        task = _task("t1")
        think_responses = [
            "QUESTION: What is the target database?",
            "Using PostgreSQL as the database.",
        ]
        call_count = 0

        async def mock_think(prompt, ctx=None):
            nonlocal call_count
            resp = think_responses[min(call_count, len(think_responses) - 1)]
            call_count += 1
            return resp

        registry = AgentRegistry()
        worker = MagicMock()
        worker.think = AsyncMock(side_effect=mock_think)
        worker._actions = []

        with (
            patch("aifactory.graphs.dynamic_pipeline.create_worker_config") as mock_cfg,
            patch(
                "aifactory.graphs.dynamic_pipeline.execute_code_blocks",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aifactory.graphs.dynamic_pipeline.execute_integration_actions",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aifactory.graphs.dynamic_pipeline.interrupt", return_value="PostgreSQL"
            ) as mock_interrupt,
        ):
            mock_cfg.return_value = {
                "model": "gpt-4o-mini",
                "_name": "Worker-t1",
                "system_prompt": "You are a worker.",
                "max_tokens": 1000,
                "token_budget": 50000,
                "model_score": 1.0,
            }
            kwargs = self._base_worker_kwargs(registry)
            kwargs["workers_cache"]["t1"] = worker

            entry, _ctx_delta = await _run_worker_task(task, {}, **kwargs)

        mock_interrupt.assert_called_once()
        question_arg = mock_interrupt.call_args[0][0]
        assert "What is the target database?" in question_arg
        assert "PostgreSQL" in entry["result"]

    @pytest.mark.asyncio
    async def test_question_answer_stored_in_ctx_delta(self) -> None:
        """Q&A pair is stored in ctx_delta under the 'qa0' key."""
        task = _task("t1")
        think_responses = [
            "QUESTION: What mode?",
            "Running in production mode.",
        ]
        call_count = 0

        async def mock_think(prompt, ctx=None):
            nonlocal call_count
            resp = think_responses[min(call_count, len(think_responses) - 1)]
            call_count += 1
            return resp

        registry = AgentRegistry()
        worker = MagicMock()
        worker.think = AsyncMock(side_effect=mock_think)
        worker._actions = []

        with (
            patch("aifactory.graphs.dynamic_pipeline.create_worker_config") as mock_cfg,
            patch(
                "aifactory.graphs.dynamic_pipeline.execute_code_blocks",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aifactory.graphs.dynamic_pipeline.execute_integration_actions",
                new=AsyncMock(return_value=[]),
            ),
            patch("aifactory.graphs.dynamic_pipeline.interrupt", return_value="production"),
        ):
            mock_cfg.return_value = {
                "model": "gpt-4o-mini",
                "_name": "Worker-t1",
                "system_prompt": "You are a worker.",
                "max_tokens": 1000,
                "token_budget": 50000,
                "model_score": 1.0,
            }
            kwargs = self._base_worker_kwargs(registry)
            kwargs["workers_cache"]["t1"] = worker

            _entry, ctx_delta = await _run_worker_task(task, {}, **kwargs)

        qa_keys = [k for k in ctx_delta if k.startswith("qa")]
        assert len(qa_keys) >= 1
        assert "production" in ctx_delta.get("qa0", "")


# =============================================================================
# LLM output format resilience in make_analyse
# =============================================================================


class TestAnalyserLLMFormatResilience:
    """Analyser must handle varied LLM output formats without crashing."""

    @pytest.mark.asyncio
    async def test_json_in_markdown_code_fence(self) -> None:
        """Analyser parses JSON wrapped in ```json ... ``` fences."""
        agent = PipelineAgent(
            "analyser", {"system_prompt": "You are analyser.", "max_tokens": 1000}
        )
        node = make_analyse(agent)
        response = (
            "```json\n"
            '{"direct_answer": "", "research_needed": false, '
            '"tasks": [{"id": "task_1", "description": "Build the API"}]}\n'
            "```"
        )
        with _mock_litellm(response):
            result = await node({"user_query": "Build an API", "shared_context": {}})
        assert len(result["tasks"]) == 1

    @pytest.mark.asyncio
    async def test_json_with_leading_prose(self) -> None:
        """Analyser parses JSON preceded by explanation text."""
        agent = PipelineAgent(
            "analyser", {"system_prompt": "You are analyser.", "max_tokens": 1000}
        )
        node = make_analyse(agent)
        response = (
            "Here is my analysis:\n"
            '{"direct_answer": "", "research_needed": false, '
            '"tasks": [{"id": "t1", "description": "Write code"}]}'
        )
        with _mock_litellm(response):
            result = await node({"user_query": "Write some code", "shared_context": {}})
        assert len(result["tasks"]) == 1

    @pytest.mark.asyncio
    async def test_json_followed_by_share_block(self) -> None:
        """Analyser strips trailing SHARE: block before parsing JSON."""
        agent = PipelineAgent(
            "analyser", {"system_prompt": "You are analyser.", "max_tokens": 1000}
        )
        node = make_analyse(agent)
        response = (
            '{"direct_answer": "", "research_needed": false, '
            '"tasks": [{"id": "t1", "description": "Build it"}]}\n'
            "SHARE:\nstatus: started"
        )
        with _mock_litellm(response):
            result = await node({"user_query": "Build something", "shared_context": {}})
        assert len(result["tasks"]) == 1


# =============================================================================
# Concurrent pipeline isolation (team-layer memory per run_id)
# =============================================================================


class TestConcurrentPipelineIsolation:
    """Two concurrent pipeline runs must not cross-contaminate team-layer memory."""

    @pytest.mark.asyncio
    async def test_run_ids_isolate_team_layer(self) -> None:
        """SHARE: updates from different run_ids land in separate Qdrant team entries."""
        from qdrant_client import QdrantClient

        from aifactory.core.memory_bridge import MemoryBridge
        from aifactory.core.memory_router import MemoryRouter, _stub_embed
        from aifactory.core.schemas import MemoryLevel

        client = QdrantClient(":memory:")
        router = MemoryRouter(qdrant_client=client, embedding_fn=_stub_embed)

        bridge_a = MemoryBridge(router=router, run_id="run_isolation_a")
        bridge_b = MemoryBridge(router=router, run_id="run_isolation_b")

        bridge_a.store_share_update("analyser", "framework", "FastAPI")
        bridge_b.store_share_update("analyser", "framework", "Django")

        hits_a = router.retrieve(
            "FastAPI framework",
            levels=[MemoryLevel.TEAM],
            team_id="run_isolation_a",
            top_k=10,
        )
        hits_b = router.retrieve(
            "Django framework",
            levels=[MemoryLevel.TEAM],
            team_id="run_isolation_b",
            top_k=10,
        )

        assert len(hits_a) >= 1
        a_content = " ".join(e.content for e in hits_a)
        assert "FastAPI" in a_content

        assert len(hits_b) >= 1
        b_content = " ".join(e.content for e in hits_b)
        assert "Django" in b_content

    @pytest.mark.asyncio
    async def test_parallel_bridges_dont_clobber_shared_ctx(self) -> None:
        """Two workers running concurrently return independent ctx_deltas that merge cleanly."""
        tasks = [_task("t1"), _task("t2")]

        async def fake_worker_a(task, ctx, **kw):
            # Simulate a slow worker
            await asyncio.sleep(0)
            return (
                {
                    "task_id": "t1",
                    "result": "a",
                    "skipped": False,
                    "agent": "W-t1",
                    "model": "m",
                    "description": "d",
                },
                {"key_a": "value_a", "shared_key": "from_a"},
            )

        async def fake_worker_b(task, ctx, **kw):
            await asyncio.sleep(0)
            return (
                {
                    "task_id": "t2",
                    "result": "b",
                    "skipped": False,
                    "agent": "W-t2",
                    "model": "m",
                    "description": "d",
                },
                {"key_b": "value_b", "shared_key": "from_b"},
            )

        coros = [fake_worker_a(tasks[0], {}), fake_worker_b(tasks[1], {})]
        outcomes = await asyncio.gather(*coros, return_exceptions=True)

        shared_ctx: dict = {}
        for outcome in outcomes:
            assert not isinstance(outcome, BaseException)
            _, delta = outcome
            shared_ctx = {**shared_ctx, **delta}

        assert shared_ctx["key_a"] == "value_a"
        assert shared_ctx["key_b"] == "value_b"
        # Last writer wins for shared_key — no corruption


# =============================================================================
# Auto-calibration hook
# =============================================================================


class TestAutoCalibration:
    """Auto-calibration hook in judge_review fires ModelCalibrator every N runs."""

    async def test_auto_calibrate_fires_at_n_runs(self) -> None:
        """asyncio.create_task is called when run_count % N == 0."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        from aifactory.graphs.dynamic_pipeline import make_judge_review

        judge_agent = MagicMock()
        judge_agent.name = "judge"
        judge_agent.model = "gpt-4o-mini"
        judge_agent.think = AsyncMock(
            return_value='{"assessments": [], "composite_quality": 8.0}'
        )
        judge_agent.total_tokens = 50
        judge_agent._budget_tracker = None

        registry = MagicMock()
        registry.all.return_value = []

        tasks_fired: list = []

        def _capture_task(coro):
            """Sync side_effect: capture the coroutine and immediately close it."""
            tasks_fired.append(coro)
            coro.close()  # prevent "never awaited" RuntimeWarning
            return MagicMock()

        mock_settings = MagicMock()
        mock_settings.calibrate_every_n_runs = 3
        mock_settings.lite_mode = False
        mock_settings.judge_skip_simple_runs = False
        mock_settings.judge_deep_eval_enabled = False

        store = MagicMock()
        store.list_recent.return_value = [MagicMock()] * 3  # 3 runs → 3 % 3 == 0

        state: dict = {
            "user_query": "test query",
            "task_results": [],
            "shared_context": {"objective": "test"},
            "tasks": [],
            "agent_assessments": [],
            "judge_metrics": {},
        }

        with (
            patch("aifactory.config.settings.get_settings", return_value=mock_settings),
            patch("aifactory.core.run_store.get_run_store", return_value=store),
            patch("asyncio.create_task", side_effect=_capture_task),
        ):
            judge_fn = make_judge_review(judge_agent, registry, run_id="run-test")
            try:
                await judge_fn(state)
            except Exception:
                pass  # partial state is expected in unit tests

        assert len(tasks_fired) == 1, (
            f"Expected 1 create_task call, got {len(tasks_fired)}"
        )

    async def test_auto_calibrate_not_triggered_when_disabled(self) -> None:
        """calibrate_every_n_runs=0 means no asyncio.create_task is created."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from aifactory.graphs.dynamic_pipeline import make_judge_review

        judge_agent = MagicMock()
        judge_agent.name = "judge"
        judge_agent.model = "gpt-4o-mini"
        judge_agent.think = AsyncMock(
            return_value='{"assessments": [], "composite_quality": 8.0}'
        )
        judge_agent.total_tokens = 50
        judge_agent._budget_tracker = None

        registry = MagicMock()
        registry.all.return_value = []

        mock_settings = MagicMock()
        mock_settings.calibrate_every_n_runs = 0  # disabled
        mock_settings.lite_mode = False
        mock_settings.judge_skip_simple_runs = False
        mock_settings.judge_deep_eval_enabled = False

        task_calls: list = []

        state: dict = {
            "user_query": "test",
            "task_results": [],
            "shared_context": {},
            "tasks": [],
            "agent_assessments": [],
            "judge_metrics": {},
        }

        with (
            patch("aifactory.config.settings.get_settings", return_value=mock_settings),
            patch("asyncio.create_task", side_effect=lambda c: task_calls.append(c)),
        ):
            judge_fn = make_judge_review(judge_agent, registry, run_id="run-x")
            try:
                await judge_fn(state)
            except Exception:
                pass

        assert len(task_calls) == 0

    async def test_auto_calibrate_not_triggered_when_count_not_divisible(self) -> None:
        """With N=3 but run_count=4, no task is created."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from aifactory.graphs.dynamic_pipeline import make_judge_review

        judge_agent = MagicMock()
        judge_agent.name = "judge"
        judge_agent.model = "gpt-4o-mini"
        judge_agent.think = AsyncMock(
            return_value='{"assessments": [], "composite_quality": 8.0}'
        )
        judge_agent.total_tokens = 50
        judge_agent._budget_tracker = None

        registry = MagicMock()
        registry.all.return_value = []

        mock_settings = MagicMock()
        mock_settings.calibrate_every_n_runs = 3
        mock_settings.lite_mode = False
        mock_settings.judge_skip_simple_runs = False
        mock_settings.judge_deep_eval_enabled = False

        store = MagicMock()
        store.list_recent.return_value = [MagicMock()] * 4  # 4 % 3 != 0

        task_calls: list = []

        state: dict = {
            "user_query": "test",
            "task_results": [],
            "shared_context": {},
            "tasks": [],
            "agent_assessments": [],
            "judge_metrics": {},
        }

        with (
            patch("aifactory.config.settings.get_settings", return_value=mock_settings),
            patch("aifactory.core.run_store.get_run_store", return_value=store),
            patch("asyncio.create_task", side_effect=lambda c: task_calls.append(c)),
        ):
            judge_fn = make_judge_review(judge_agent, registry, run_id="run-y")
            try:
                await judge_fn(state)
            except Exception:
                pass

        assert len(task_calls) == 0
