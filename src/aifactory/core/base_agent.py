"""
Base Agent — The foundational abstraction for all agents in the AI Factory.

Every agent (Worker, Team Lead, PM, Master Researcher, Judge) extends this class.
It handles:
  - LLM communication via LiteLLM (any model, any provider)
  - Structured input/output via Pydantic schemas
  - Memory integration via MemoryRouter
  - Token tracking and budget enforcement
  - Iteration counting
  - Persona / system prompt management
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypeVar

import litellm
from pydantic import BaseModel

from aifactory.config.settings import get_settings
from aifactory.core.memory_router import MemoryRouter
from aifactory.core.observability import traceable
from aifactory.core.prompt_cache import build_cached_system
from aifactory.core.schemas import (
    AgentConfig,
    MemoryEntry,
    MemoryLevel,
)

logger = logging.getLogger(__name__)

# Suppress litellm debug logs — we handle our own logging
litellm.suppress_debug_info = True

T = TypeVar("T", bound=BaseModel)

import re as _re


def _strip_tool_artifacts(text: str) -> str:
    """Remove LLM-internal tool/code artifacts that leak into visible output."""
    # Gemini/OpenRouter tool_code blocks: ```tool_code ... ```
    text = _re.sub(r"```tool_code\b.*?```", "", text, flags=_re.DOTALL)
    # tool_outputs blocks
    text = _re.sub(r"```tool_outputs\b.*?```", "", text, flags=_re.DOTALL)
    # XML-style internal tags some models emit
    text = _re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=_re.DOTALL)
    text = _re.sub(r"<tool_response>.*?</tool_response>", "", text, flags=_re.DOTALL)

    # Comment-only code blocks (e.g. ```\n# Master Agent:\n# ...\n```)
    def _is_comment_only(block: str) -> bool:
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        return bool(lines) and all(l.startswith("#") for l in lines)

    def _maybe_strip_block(m: _re.Match) -> str:
        return "" if _is_comment_only(m.group(1)) else m.group(0)

    text = _re.sub(r"```(?:\w*\n)?(.*?)```", _maybe_strip_block, text, flags=_re.DOTALL)
    return text.strip()


class BaseAgent:
    """
    The atomic unit of the AI Factory.

    Each agent:
      1. Has a persona (role, name, system prompt)
      2. Communicates only via structured messages (Pydantic)
      3. Reads/writes to the shared memory system
      4. Tracks its own token usage and iteration count
      5. Can be scored by the RL sidecar
    """

    def __init__(
        self,
        config: AgentConfig,
        memory: MemoryRouter | None = None,
    ) -> None:
        self.config = config
        self.memory = memory

        # Runtime state
        self.iteration = 0
        self.total_tokens_used = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.conversation_history: list[dict[str, str]] = []

        # Build the system message
        self._system_message = self._build_system_message()

    # ─── Core LLM Interface ──────────────────────────────────────────

    @traceable(name="agent_think", run_type="llm")
    async def think(
        self,
        user_message: str,
        response_model: type[T] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_context: str = "",
    ) -> T | str:
        """
        Send a message to the LLM and get a response.

        Args:
            user_message: The input prompt / task for this turn.
            response_model: If provided, parse response into this Pydantic model.
            temperature: Override the agent's default temperature.
            max_tokens: Override max tokens for this call.
            extra_context: Additional context to inject (e.g. from memory retrieval).

        Returns:
            Parsed Pydantic model if response_model given, else raw string.
        """
        settings = get_settings()

        # Budget check
        if self.iteration >= self.config.max_iterations:
            logger.warning(
                "Agent %s (%s) hit iteration limit (%d)",
                self.config.name,
                self.config.agent_id,
                self.config.max_iterations,
            )
            raise AgentBudgetError(
                f"Agent {self.config.name} exceeded max iterations ({self.config.max_iterations})"
            )

        if self.total_tokens_used >= self.config.token_budget:
            logger.warning(
                "Agent %s (%s) hit token budget (%d)",
                self.config.name,
                self.config.agent_id,
                self.config.token_budget,
            )
            raise AgentBudgetError(
                f"Agent {self.config.name} exceeded token budget ({self.config.token_budget})"
            )

        # Retrieve relevant memory context
        memory_context = ""
        if self.memory and extra_context == "":
            memories = self.memory.retrieve(
                query=user_message,
                levels=[MemoryLevel.AGENT, MemoryLevel.TEAM],
                agent_id=self.config.agent_id,
                team_id=self.config.team_id,
                top_k=3,
            )
            if memories:
                memory_snippets = [
                    f"[{m.level.value}/{m.entry_type}] {m.content[:200]}" for m in memories
                ]
                memory_context = (
                    "\n<relevant_memory>\n" + "\n".join(memory_snippets) + "\n</relevant_memory>"
                )

        # Build messages.
        # Phase 3.5: apply Anthropic prompt caching to the system prefix — the
        # system message is stable across every turn of an agent's life, so
        # cache_control gives ~90% input-token cost reduction on repeat Claude
        # calls. Delegated to the shared helper so gating (setting flag +
        # Claude-only + no-proxy) stays in sync with PipelineAgent.think.
        # ``api_base=""`` — BaseAgent calls litellm.acompletion directly with
        # no ``base_url`` override, so a LITELLM_API_BASE env var (used only by
        # dynamic_pipeline's proxy routing) must not gate caching here.
        _sys_content = build_cached_system(
            self._system_message, self.config.model, api_base=""
        )
        messages: list[dict[str, Any]] = [{"role": "system", "content": _sys_content}]

        # Add recent conversation history (keep last 6 turns to stay lean)
        messages.extend(self.conversation_history[-6:])

        # Build user message with memory + extra context
        full_user = user_message
        if memory_context:
            full_user = memory_context + "\n\n" + full_user
        if extra_context:
            full_user = f"<context>\n{extra_context}\n</context>\n\n" + full_user

        messages.append({"role": "user", "content": full_user})

        # LLM call
        t0 = time.perf_counter()

        call_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or settings.max_tokens_per_agent_turn,
            "caching": True,  # use LiteLLM response cache when available (Redis or in-memory)
        }

        # If structured output requested, ask for JSON
        if response_model is not None:
            call_kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await litellm.acompletion(**call_kwargs)
        except Exception as e:
            logger.error("LLM call failed for %s: %s", self.config.name, e)
            raise

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        # Extract response
        choice = response.choices[0]  # type: ignore[union-attr]
        raw_content = _strip_tool_artifacts(choice.message.content or "")

        # Track tokens
        usage = response.usage  # type: ignore[union-attr]
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens_used += input_tokens + output_tokens
        self.iteration += 1

        logger.info(
            "[%s] iter=%d tokens=%d+%d time=%dms",
            self.config.name,
            self.iteration,
            input_tokens,
            output_tokens,
            elapsed_ms,
        )

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": raw_content})

        # Store in agent memory
        if self.memory:
            self.memory.store(
                MemoryEntry(
                    level=MemoryLevel.AGENT,
                    agent_id=self.config.agent_id,
                    team_id=self.config.team_id,
                    content=(
                        f"[Turn {self.iteration}] Task: "
                        f"{user_message[:100]}... -> "
                        f"Response: {raw_content[:200]}..."
                    ),
                    entry_type="observation",
                    turn_number=self.iteration,
                    metadata={
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "elapsed_ms": elapsed_ms,
                    },
                )
            )

            # Check if we should promote to team memory
            if self.memory.should_promote(self.config.agent_id) and self.config.team_id:
                await self._auto_promote()

        # Parse into Pydantic model if requested
        if response_model is not None:
            try:
                parsed = response_model.model_validate_json(raw_content)
                return parsed
            except Exception:
                # Fallback: try to extract JSON from response
                try:
                    json_str = self._extract_json(raw_content)
                    parsed = response_model.model_validate_json(json_str)
                    return parsed
                except Exception as e2:
                    logger.warning(
                        "Failed to parse structured output for %s: %s. Raw: %s...",
                        self.config.name,
                        e2,
                        raw_content[:200],
                    )
                    raise StructuredOutputError(
                        f"Could not parse response into {response_model.__name__}: {e2}"
                    ) from e2

        return raw_content

    # ─── Convenience Methods ──────────────────────────────────────────

    async def execute_task(
        self,
        task_description: str,
        response_model: type[T] | None = None,
        context: str = "",
    ) -> T | str:
        """
        High-level method to execute a task. Subclasses can override this
        to add role-specific logic (decomposition, delegation, etc.).
        """
        return await self.think(
            user_message=task_description,
            response_model=response_model,
            extra_context=context,
        )

    def store_memory(
        self,
        content: str,
        entry_type: str = "observation",
        level: MemoryLevel = MemoryLevel.AGENT,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Explicitly store something in memory (outside of auto-storage in think())."""
        if self.memory:
            self.memory.store(
                MemoryEntry(
                    level=level,
                    agent_id=self.config.agent_id,
                    team_id=self.config.team_id,
                    content=content,
                    entry_type=entry_type,
                    turn_number=self.iteration,
                    metadata=metadata or {},
                )
            )

    def get_status(self) -> dict[str, Any]:
        """Return agent's current runtime status."""
        return {
            "agent_id": self.config.agent_id,
            "name": self.config.name,
            "role": self.config.role.value,
            "model": self.config.model,
            "iteration": self.iteration,
            "max_iterations": self.config.max_iterations,
            "tokens_used": self.total_tokens_used,
            "token_budget": self.config.token_budget,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
        }

    # ─── Internal Helpers ─────────────────────────────────────────────

    def _build_system_message(self) -> str:
        """Construct the system prompt from config + role-specific content."""
        get_settings()
        parts = [
            f"You are {self.config.name}, a {self.config.role.value} "
            f"agent in the AI Factory system.",
            f"Your agent ID is: {self.config.agent_id}",
        ]

        if self.config.team_id:
            parts.append(f"You belong to team: {self.config.team_id}")

        if self.config.supervisor_id:
            parts.append(f"Your supervisor is: {self.config.supervisor_id}")

        if self.config.specialty:
            parts.append(f"Your specialty is: {self.config.specialty.value}")

        parts.append(
            "\n## Communication Rules\n"
            "- Always respond with valid JSON when a structured format is requested.\n"
            "- Be concise. Avoid filler words.\n"
            "- Report issues immediately via escalation.\n"
            "- Never repeat work already done by peers (check memory context).\n"
            f"- You have a budget of {self.config.max_iterations} iterations "
            f"and {self.config.token_budget} tokens.\n"
        )

        if self.config.system_prompt:
            parts.append(f"\n## Role-Specific Instructions\n{self.config.system_prompt}")

        return "\n".join(parts)

    async def _auto_promote(self) -> None:
        """
        Automatically promote agent memories to the team layer.
        Uses the LLM to generate a summary of recent work.
        """
        if not self.memory or not self.config.team_id:
            return

        # Get recent unpromoted memories
        recent = self.memory.retrieve(
            query="summary of recent work and decisions",
            levels=[MemoryLevel.AGENT],
            agent_id=self.config.agent_id,
            top_k=10,
        )

        if not recent:
            return

        # Use the LLM to summarise
        entries_text = "\n".join([f"- {m.content[:150]}" for m in recent])
        summary_prompt = (
            "Summarise the following agent activity into a brief "
            "paragraph (max 100 words) capturing key decisions, "
            "progress, and issues:\n\n"
            f"{entries_text}"
        )

        try:
            # Prompt-cache exempt: summary_prompt is fully dynamic (rebuilt
            # every promotion from the just-retrieved memory entries) and
            # there is no stable system prefix to cache — cache_control would
            # never hit and would only add a wasted marker on cold calls.
            summary_response = await litellm.acompletion(
                model=self.config.model,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            summary_text = summary_response.choices[0].message.content or ""

            self.memory.promote(
                agent_id=self.config.agent_id,
                team_id=self.config.team_id,
                summary=summary_text,
                from_level=MemoryLevel.AGENT,
                to_level=MemoryLevel.TEAM,
            )
            logger.info(
                "Auto-promoted memories for agent %s to team layer",
                self.config.name,
            )
        except Exception as e:
            logger.warning("Auto-promotion failed for %s: %s", self.config.name, e)

    @staticmethod
    def _extract_json(text: str) -> str:
        """Try to extract JSON from a text that might have markdown code fences."""
        # Look for ```json ... ``` blocks
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            return text[start:end].strip()
        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return text[start:end].strip()
        # Try finding { ... } directly
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return text[start:end]
        raise ValueError("No JSON found in response")


# ─── Exceptions ───────────────────────────────────────────────────────────


class AgentBudgetError(Exception):
    """Raised when an agent exceeds its iteration or token budget."""


class StructuredOutputError(Exception):
    """Raised when the LLM response can't be parsed into the expected schema."""
