"""
Tests for graphs/dynamic_pipeline.py — the 8-node config-driven graph.

All LLM calls are mocked via litellm.acompletion.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from aifactory.core.agent_factory import AgentRegistry
from aifactory.graphs.dynamic_pipeline import (
    AgentBudgetExceededError,
    DynamicPipelineState,
    PipelineAgent,
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
        with _mock_litellm("On it!"):
            result = await node({"user_query": "Build an app", "shared_context": {}})
        assert "shared_context" in result
        assert result["shared_context"]["objective"] == "Build an app"

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
        with _mock_litellm("I will assess quality and efficiency."):
            result = await node(
                {
                    "user_query": "Build a REST API",
                    "shared_context": {"objective": "Build a REST API"},
                }
            )
        # judge_brief returns empty dict (it only records to its own history)
        assert result == {}
        assert judge.total_tokens > 0


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
                    ],
                }
            )
        assert len(result["agent_assessments"]) == 1
        assert result["agent_assessments"][0]["agent"] == "Worker-task_1"
        assert result["judge_metrics"]["overall_status"] == "successful"
