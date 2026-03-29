"""
Tests for core/base_agent.py — LLM calls, budget enforcement, memory integration.

All LLM calls are mocked to avoid real API calls.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, Field

from aifactory.core.base_agent import (
    AgentBudgetError,
    BaseAgent,
    StructuredOutputError,
)
from aifactory.core.memory_router import MemoryRouter
from aifactory.core.schemas import AgentConfig, AgentRole, MemoryLevel, WorkerSpecialty

from tests.helpers import make_json_response, make_llm_response


# ─── Structured Output Model for Tests ────────────────────────────────


class MockOutput(BaseModel):
    answer: str = Field(description="The answer")
    confidence: float = Field(default=0.8)


# =============================================================================
# Agent Construction
# =============================================================================


class TestBaseAgentInit:
    def test_initial_state(self, worker_config: AgentConfig) -> None:
        agent = BaseAgent(config=worker_config)
        assert agent.iteration == 0
        assert agent.total_tokens_used == 0
        assert agent.total_input_tokens == 0
        assert agent.total_output_tokens == 0
        assert agent.conversation_history == []

    def test_system_message_contains_role(self, worker_config: AgentConfig) -> None:
        agent = BaseAgent(config=worker_config)
        assert "worker" in agent._system_message.lower()
        assert worker_config.name in agent._system_message

    def test_system_message_contains_agent_id(self, worker_config: AgentConfig) -> None:
        agent = BaseAgent(config=worker_config)
        assert worker_config.agent_id in agent._system_message

    def test_system_message_contains_team(self, worker_config: AgentConfig) -> None:
        agent = BaseAgent(config=worker_config)
        assert "team_test" in agent._system_message

    def test_system_message_contains_specialty(self, worker_config: AgentConfig) -> None:
        agent = BaseAgent(config=worker_config)
        assert "coder" in agent._system_message.lower()

    def test_system_message_contains_budget_info(self, worker_config: AgentConfig) -> None:
        agent = BaseAgent(config=worker_config)
        assert str(worker_config.max_iterations) in agent._system_message
        assert str(worker_config.token_budget) in agent._system_message

    def test_custom_system_prompt_appended(self) -> None:
        config = AgentConfig(
            role=AgentRole.WORKER,
            name="Custom",
            model="test",
            system_prompt="Always respond in JSON format.",
        )
        agent = BaseAgent(config=config)
        assert "Always respond in JSON format" in agent._system_message

    def test_with_memory(
        self, worker_config: AgentConfig, memory: MemoryRouter
    ) -> None:
        agent = BaseAgent(config=worker_config, memory=memory)
        assert agent.memory is memory

    def test_without_memory(self, worker_config: AgentConfig) -> None:
        agent = BaseAgent(config=worker_config, memory=None)
        assert agent.memory is None


# =============================================================================
# think() — LLM Calls
# =============================================================================


class TestThink:
    @pytest.mark.asyncio
    async def test_basic_call(self, worker_config: AgentConfig) -> None:
        agent = BaseAgent(config=worker_config)
        mock_resp = make_llm_response("Hello, I completed the task.")

        with patch("aifactory.core.base_agent.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            result = await agent.think("Do something")

        assert result == "Hello, I completed the task."
        assert agent.iteration == 1
        assert agent.total_tokens_used == 300  # 100 + 200
        assert agent.total_input_tokens == 100
        assert agent.total_output_tokens == 200

    @pytest.mark.asyncio
    async def test_conversation_history_updated(self, worker_config: AgentConfig) -> None:
        agent = BaseAgent(config=worker_config)
        mock_resp = make_llm_response("Response 1")

        with patch("aifactory.core.base_agent.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            await agent.think("Question 1")

        assert len(agent.conversation_history) == 2  # user + assistant
        assert agent.conversation_history[0]["role"] == "user"
        assert agent.conversation_history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_structured_output_parsing(self, worker_config: AgentConfig) -> None:
        agent = BaseAgent(config=worker_config)
        data = {"answer": "42", "confidence": 0.95}
        mock_resp = make_json_response(data)

        with patch("aifactory.core.base_agent.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            result = await agent.think("What is the answer?", response_model=MockOutput)

        assert isinstance(result, MockOutput)
        assert result.answer == "42"
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_structured_output_from_markdown_fence(self, worker_config: AgentConfig) -> None:
        """Test JSON extraction from markdown code fences."""
        agent = BaseAgent(config=worker_config)
        content = '```json\n{"answer": "extracted", "confidence": 0.7}\n```'
        mock_resp = make_llm_response(content)

        with patch("aifactory.core.base_agent.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            result = await agent.think("Test", response_model=MockOutput)

        assert isinstance(result, MockOutput)
        assert result.answer == "extracted"

    @pytest.mark.asyncio
    async def test_structured_output_failure_raises(self, worker_config: AgentConfig) -> None:
        """When response can't be parsed, StructuredOutputError is raised."""
        agent = BaseAgent(config=worker_config)
        mock_resp = make_llm_response("This is not valid JSON at all")

        with patch("aifactory.core.base_agent.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            with pytest.raises(StructuredOutputError):
                await agent.think("Test", response_model=MockOutput)

    @pytest.mark.asyncio
    async def test_extra_context_injected(self, worker_config: AgentConfig) -> None:
        agent = BaseAgent(config=worker_config)
        mock_resp = make_llm_response("Done")

        with patch("aifactory.core.base_agent.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            await agent.think("Test", extra_context="Important background info")

        # The extra_context should appear in the messages sent to the LLM
        call_args = mock_llm.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[-1]["content"]
        assert "Important background info" in user_msg

    @pytest.mark.asyncio
    async def test_temperature_override(self, worker_config: AgentConfig) -> None:
        agent = BaseAgent(config=worker_config)
        mock_resp = make_llm_response("Done")

        with patch("aifactory.core.base_agent.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            await agent.think("Test", temperature=0.9)

        call_args = mock_llm.call_args
        assert call_args.kwargs["temperature"] == 0.9


# =============================================================================
# Budget Enforcement
# =============================================================================


class TestBudgetEnforcement:
    @pytest.mark.asyncio
    async def test_iteration_limit(self, worker_config: AgentConfig) -> None:
        agent = BaseAgent(config=worker_config)
        agent.iteration = worker_config.max_iterations  # At limit

        with pytest.raises(AgentBudgetError, match="exceeded max iterations"):
            await agent.think("Test")

    @pytest.mark.asyncio
    async def test_token_budget_limit(self, worker_config: AgentConfig) -> None:
        agent = BaseAgent(config=worker_config)
        agent.total_tokens_used = worker_config.token_budget  # At limit

        with pytest.raises(AgentBudgetError, match="exceeded token budget"):
            await agent.think("Test")

    @pytest.mark.asyncio
    async def test_multiple_calls_track_tokens(self, worker_config: AgentConfig) -> None:
        agent = BaseAgent(config=worker_config)

        with patch("aifactory.core.base_agent.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = make_llm_response("R1", input_tokens=50, output_tokens=100)
            await agent.think("Q1")

            mock_llm.return_value = make_llm_response("R2", input_tokens=60, output_tokens=120)
            await agent.think("Q2")

        assert agent.iteration == 2
        assert agent.total_tokens_used == 330  # 150 + 180
        assert agent.total_input_tokens == 110
        assert agent.total_output_tokens == 220


# =============================================================================
# Memory Integration
# =============================================================================


class TestMemoryIntegration:
    @pytest.mark.asyncio
    async def test_stores_to_agent_memory(
        self, worker_config: AgentConfig, memory: MemoryRouter
    ) -> None:
        agent = BaseAgent(config=worker_config, memory=memory)
        mock_resp = make_llm_response("Task completed")

        with patch("aifactory.core.base_agent.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            await agent.think("Do something")

        stats = memory.get_stats()
        assert stats["agent"] >= 1

    @pytest.mark.asyncio
    async def test_retrieves_memory_context(
        self, worker_config: AgentConfig, memory: MemoryRouter
    ) -> None:
        """Memory context is retrieved and injected when no extra_context given."""
        # Store a relevant memory first
        from aifactory.core.schemas import MemoryEntry

        memory.store(
            MemoryEntry(
                level=MemoryLevel.AGENT,
                agent_id=worker_config.agent_id,
                team_id=worker_config.team_id,
                content="Previously discussed PostgreSQL for the database",
                turn_number=0,
            )
        )

        agent = BaseAgent(config=worker_config, memory=memory)
        mock_resp = make_llm_response("Using PostgreSQL")

        with patch("aifactory.core.base_agent.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            await agent.think("What database should we use?")

        # Check that memory context was injected into the prompt
        call_args = mock_llm.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[-1]["content"]
        assert "relevant_memory" in user_msg or "PostgreSQL" in user_msg

    def test_store_memory_explicit(
        self, worker_config: AgentConfig, memory: MemoryRouter
    ) -> None:
        agent = BaseAgent(config=worker_config, memory=memory)
        agent.store_memory(
            content="Important decision made",
            entry_type="decision",
        )
        stats = memory.get_stats()
        assert stats["agent"] >= 1

    def test_store_memory_without_memory_router(self, worker_config: AgentConfig) -> None:
        """Storing memory should be a no-op when no MemoryRouter is configured."""
        agent = BaseAgent(config=worker_config, memory=None)
        agent.store_memory(content="This should not crash")  # Should not raise


# =============================================================================
# get_status()
# =============================================================================


class TestGetStatus:
    def test_returns_all_fields(self, worker_config: AgentConfig) -> None:
        agent = BaseAgent(config=worker_config)
        status = agent.get_status()

        assert status["agent_id"] == worker_config.agent_id
        assert status["name"] == worker_config.name
        assert status["role"] == "worker"
        assert status["model"] == worker_config.model
        assert status["iteration"] == 0
        assert status["max_iterations"] == worker_config.max_iterations
        assert status["tokens_used"] == 0
        assert status["token_budget"] == worker_config.token_budget


# =============================================================================
# _extract_json() static method
# =============================================================================


class TestExtractJson:
    def test_json_code_fence(self) -> None:
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        result = BaseAgent._extract_json(text)
        assert json.loads(result) == {"key": "value"}

    def test_generic_code_fence(self) -> None:
        text = 'Before\n```\n{"key": "value"}\n```\nAfter'
        result = BaseAgent._extract_json(text)
        assert json.loads(result) == {"key": "value"}

    def test_bare_json(self) -> None:
        text = 'Prefix {"key": "value"} suffix'
        result = BaseAgent._extract_json(text)
        assert json.loads(result) == {"key": "value"}

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="No JSON found"):
            BaseAgent._extract_json("No JSON here at all")
