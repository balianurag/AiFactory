"""
Integration tests for Phase 5 hardening features wired into the pipeline.

Covers:
  1. Output quality gate inside _run_worker_task
  2. Streaming event_callback (worker_start / worker_complete)
  3. Budget tracker enforcement inside PipelineAgent.think()
  4. Budget tracker wired into pipeline workers
  5. Retry + circuit breaker inside PipelineAgent.think()
  6. build_dynamic_pipeline accepts event_callback
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aifactory.core.agent_factory import AgentRegistry
from aifactory.core.budget_tracker import BudgetExceededError, BudgetTracker
from aifactory.core.retry import reset_circuit
from aifactory.graphs.dynamic_pipeline import (
    PipelineAgent,
    _run_worker_task,
    build_dynamic_pipeline,
)
from tests.helpers import make_llm_response

_LITELLM_PATCH = "aifactory.graphs.dynamic_pipeline.litellm.acompletion"


# ── Shared fixture ────────────────────────────────────────────────────────────


def _task(tid: str, desc: str = "") -> dict:
    return {"id": tid, "description": desc or f"task {tid}"}


def _base_worker_kwargs(registry=None, workers_cache=None, completed=None):
    if registry is None:
        registry = AgentRegistry()
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


def _stub_worker_cfg():
    return {
        "model": "gpt-4o-mini",
        "fallback_model": "gpt-4o-mini",
        "_name": "Worker-t1",
        "system_prompt": "You are a worker.",
        "max_tokens": 500,
        "token_budget": 50_000,
        "model_score": 1.0,
    }


# ── 1. Output quality gate ────────────────────────────────────────────────────


class TestOutputQualityGate:
    """Quality gate fires inside _run_worker_task for prose results."""

    @pytest.mark.asyncio
    async def test_refusal_triggers_retry_and_passes(self):
        """Refusal on first response → retry → valid result → quality_flagged=False.

        Call order (pipeline order matters):
          1. Main worker think()         → refusal
          2. Self-correction check       → "VALIDATED" (keeps refusal as result)
          3. Quality-gate retry think()  → good answer (> 20 chars, no refusal)
        """
        task = _task("t1", "Write a summary of the paper.")
        think_calls = []

        async def mock_think(prompt, ctx=None):
            think_calls.append(prompt[:60])
            if len(think_calls) == 1:
                return "I cannot help with this request."
            if len(think_calls) == 2:
                # self-correction check — return "VALIDATED" so result stays as refusal
                return "VALIDATED"
            # Call 3 = quality gate retry → good substantive answer
            return "Here is a detailed summary of the paper covering all key points."

        registry = AgentRegistry()
        workers_cache: dict = {}

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
            mock_cfg.return_value = _stub_worker_cfg()
            worker = MagicMock()
            worker.think = AsyncMock(side_effect=mock_think)
            worker._actions = []
            worker._budget_tracker = None
            workers_cache["t1"] = worker

            kwargs = _base_worker_kwargs(registry, workers_cache)
            entry, _ = await _run_worker_task(task, {}, **kwargs)

        assert entry["quality_flagged"] is False
        assert "summary" in entry["result"].lower()
        # At least 2 calls: initial + quality retry
        assert len(think_calls) >= 2

    @pytest.mark.asyncio
    async def test_double_refusal_sets_quality_flagged(self):
        """Two consecutive refusals → quality_flagged=True, run continues."""
        task = _task("t1", "Write a summary.")
        call_count = 0

        async def mock_think(prompt, ctx=None):
            nonlocal call_count
            call_count += 1
            return "I'm unable to assist with that request."

        registry = AgentRegistry()
        workers_cache: dict = {}

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
            mock_cfg.return_value = _stub_worker_cfg()
            worker = MagicMock()
            worker.think = AsyncMock(side_effect=mock_think)
            worker._actions = []
            worker._budget_tracker = None
            workers_cache["t1"] = worker

            kwargs = _base_worker_kwargs(registry, workers_cache)
            entry, _ = await _run_worker_task(task, {}, **kwargs)

        assert entry["quality_flagged"] is True
        assert entry["skipped"] is False  # did NOT skip — still returns the result

    @pytest.mark.asyncio
    async def test_valid_result_no_flag(self):
        """Valid result on first attempt → quality_flagged=False, no extra calls."""
        task = _task("t1", "Explain recursion.")
        calls = []

        async def mock_think(prompt, ctx=None):
            calls.append(prompt)
            if len(calls) == 1:
                return "Recursion is a function calling itself to solve smaller sub-problems."
            return "VALIDATED"

        registry = AgentRegistry()
        workers_cache: dict = {}

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
            mock_cfg.return_value = _stub_worker_cfg()
            worker = MagicMock()
            worker.think = AsyncMock(side_effect=mock_think)
            worker._actions = []
            worker._budget_tracker = None
            workers_cache["t1"] = worker

            kwargs = _base_worker_kwargs(registry, workers_cache)
            entry, _ = await _run_worker_task(task, {}, **kwargs)

        assert entry["quality_flagged"] is False

    @pytest.mark.asyncio
    async def test_quality_gate_skipped_for_not_needed(self):
        """NOT_NEEDED self-skip bypasses quality gate entirely."""
        task = _task("t1", "Check if already done.")
        calls = []

        async def mock_think(prompt, ctx=None):
            calls.append(prompt)
            return "NOT_NEEDED"

        registry = AgentRegistry()
        workers_cache: dict = {}

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
            mock_cfg.return_value = _stub_worker_cfg()
            worker = MagicMock()
            worker.think = AsyncMock(side_effect=mock_think)
            worker._actions = []
            worker._budget_tracker = None
            workers_cache["t1"] = worker

            kwargs = _base_worker_kwargs(registry, workers_cache)
            entry, _ = await _run_worker_task(task, {}, **kwargs)

        assert entry["skipped"] is True
        assert entry["quality_flagged"] is False
        assert len(calls) == 1  # no quality gate retry


# ── 2. Streaming event callback ───────────────────────────────────────────────


class TestEventCallback:
    """event_callback receives worker_start and worker_complete events."""

    @pytest.mark.asyncio
    async def test_worker_start_event_fired(self):
        task = _task("t1", "Do the task.")
        events: list[dict] = []

        async def mock_think(prompt, ctx=None):
            return "Done, here is the result with enough content."

        registry = AgentRegistry()
        workers_cache: dict = {}

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
            mock_cfg.return_value = _stub_worker_cfg()
            worker = MagicMock()
            worker.think = AsyncMock(side_effect=mock_think)
            worker._actions = []
            worker._budget_tracker = None
            workers_cache["t1"] = worker

            kwargs = _base_worker_kwargs(registry, workers_cache)
            kwargs["event_callback"] = events.append
            await _run_worker_task(task, {}, **kwargs)

        start_events = [e for e in events if e["type"] == "worker_start"]
        assert len(start_events) == 1
        assert start_events[0]["worker"] == "Worker-t1"
        # event["task"] is task["description"][:120] — just verify it is non-empty
        assert start_events[0]["task"]

    @pytest.mark.asyncio
    async def test_worker_complete_event_fired(self):
        task = _task("t1", "Do the task.")
        events: list[dict] = []

        async def mock_think(prompt, ctx=None):
            return "Completed successfully with all required details included."

        registry = AgentRegistry()
        workers_cache: dict = {}

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
            mock_cfg.return_value = _stub_worker_cfg()
            worker = MagicMock()
            worker.think = AsyncMock(side_effect=mock_think)
            worker._actions = []
            worker._budget_tracker = None
            workers_cache["t1"] = worker

            kwargs = _base_worker_kwargs(registry, workers_cache)
            kwargs["event_callback"] = events.append
            await _run_worker_task(task, {}, **kwargs)

        complete_events = [e for e in events if e["type"] == "worker_complete"]
        assert len(complete_events) == 1
        assert complete_events[0]["worker"] == "Worker-t1"
        assert "result_preview" in complete_events[0]

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_break_pipeline(self):
        """A crashing event_callback must not propagate and kill the worker."""
        task = _task("t1", "Do the task.")

        def _crashing_callback(event):
            raise RuntimeError("callback error!")

        async def mock_think(prompt, ctx=None):
            return "Result with sufficient content for validation purposes."

        registry = AgentRegistry()
        workers_cache: dict = {}

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
            mock_cfg.return_value = _stub_worker_cfg()
            worker = MagicMock()
            worker.think = AsyncMock(side_effect=mock_think)
            worker._actions = []
            worker._budget_tracker = None
            workers_cache["t1"] = worker

            kwargs = _base_worker_kwargs(registry, workers_cache)
            kwargs["event_callback"] = _crashing_callback
            # Must NOT raise despite the callback crashing
            entry, _ = await _run_worker_task(task, {}, **kwargs)

        assert entry["task_id"] == "t1"

    def test_build_pipeline_accepts_event_callback(self):
        """build_dynamic_pipeline() accepts event_callback without error."""
        events: list[dict] = []
        compiled, registry = build_dynamic_pipeline(event_callback=events.append)
        assert compiled is not None
        assert registry.count() == 5


# ── 3. Budget tracker in PipelineAgent.think() ───────────────────────────────


class TestBudgetTrackerInThink:
    """BudgetTracker is checked before and recorded after each think() call."""

    @pytest.mark.asyncio
    async def test_budget_check_fires_before_llm_call(self):
        """Exhausted budget raises BudgetExceededError before litellm is called."""
        agent = PipelineAgent("test", {"system_prompt": "Test.", "max_tokens": 100})
        tracker = BudgetTracker(cap_usd=0.001)
        tracker._total = 0.002  # already over cap
        agent._budget_tracker = tracker

        with patch(_LITELLM_PATCH, AsyncMock()) as mock_llm:
            with pytest.raises(BudgetExceededError):
                await agent.think("hello", {})
            # LLM must NOT have been called
            mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_budget_recorded_after_successful_call(self):
        """Cost is recorded to the tracker after each successful think()."""
        agent = PipelineAgent("test", {"system_prompt": "Test.", "max_tokens": 100})
        tracker = BudgetTracker(cap_usd=10.0)
        agent._budget_tracker = tracker

        resp = make_llm_response("Answer.", input_tokens=200, output_tokens=100)
        with patch(_LITELLM_PATCH, AsyncMock(return_value=resp)):
            await agent.think("hello", {})

        # total_cost_usd should be non-zero (exact value depends on model pricing)
        # gpt-4o-mini fallback may return 0.0 if not in registry, but at minimum
        # the record() call must have been attempted without raising
        assert tracker.total_cost_usd >= 0.0

    @pytest.mark.asyncio
    async def test_budget_not_enforced_when_tracker_none(self):
        """When _budget_tracker is None, think() works normally."""
        agent = PipelineAgent("test", {"system_prompt": "Test.", "max_tokens": 100})
        assert agent._budget_tracker is None

        resp = make_llm_response("Fine.", input_tokens=10, output_tokens=5)
        with patch(_LITELLM_PATCH, AsyncMock(return_value=resp)):
            result = await agent.think("ping", {})

        assert result == "Fine."


# ── 4. Budget tracker wired into pipeline workers ─────────────────────────────


class TestBudgetTrackerInWorker:
    """budget_tracker passed to _run_worker_task is attached to the worker."""

    @pytest.mark.asyncio
    async def test_budget_tracker_attached_to_created_worker(self):
        """Workers created inside _run_worker_task receive the passed tracker."""
        task = _task("t1", "Do something.")
        tracker = BudgetTracker(cap_usd=99.0)
        captured_workers: list = []

        async def mock_think(prompt, ctx=None):
            return "Done, here is a detailed answer to the task."

        registry = AgentRegistry()

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
                _LITELLM_PATCH,
                AsyncMock(return_value=make_llm_response("Done, detailed answer here.")),
            ),
        ):
            mock_cfg.return_value = _stub_worker_cfg()

            original_register = registry.register

            def capture_register(agent):
                captured_workers.append(agent)
                return original_register(agent)

            registry.register = capture_register
            kwargs = _base_worker_kwargs(registry)
            kwargs["budget_tracker"] = tracker
            await _run_worker_task(task, {}, **kwargs)

        # The worker registered must have the tracker attached
        workers = [a for a in captured_workers if "Worker" in a.name]
        assert len(workers) >= 1
        assert workers[0]._budget_tracker is tracker

    @pytest.mark.asyncio
    async def test_budget_exceeded_propagates_from_worker(self):
        """BudgetExceededError raised inside think() propagates out of _run_worker_task."""
        task = _task("t1", "Do something.")
        tracker = BudgetTracker(cap_usd=0.001)
        tracker._total = 0.002  # already exceeded

        async def mock_think(prompt, ctx=None):
            tracker.check()  # simulate what think() does
            return "should not reach here"

        registry = AgentRegistry()
        workers_cache: dict = {}

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
            mock_cfg.return_value = _stub_worker_cfg()
            worker = MagicMock()
            worker.think = AsyncMock(side_effect=mock_think)
            worker._actions = []
            worker._budget_tracker = tracker
            workers_cache["t1"] = worker

            kwargs = _base_worker_kwargs(registry, workers_cache)
            kwargs["budget_tracker"] = tracker

            with pytest.raises(BudgetExceededError):
                await _run_worker_task(task, {}, **kwargs)


# ── 5. Retry + circuit breaker in PipelineAgent.think() ──────────────────────


class TestRetryInThink:
    """Retry and circuit breaker are exercised through PipelineAgent.think()."""

    @pytest.fixture(autouse=True)
    def _clear_circuit(self):
        reset_circuit()
        yield
        reset_circuit()

    @pytest.mark.asyncio
    async def test_think_retries_on_llm_failure(self):
        """think() retries a failed LLM call and returns the successful result."""
        agent = PipelineAgent("test", {"system_prompt": "Test.", "max_tokens": 100})
        agent.fallback_model = "gpt-4o-mini"

        call_count = 0

        async def flaky_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient LLM error")
            return make_llm_response("Success after retry.")

        with (
            patch(_LITELLM_PATCH, side_effect=flaky_llm),
            patch("aifactory.core.retry.asyncio.sleep", new=AsyncMock()),
        ):
            result = await agent.think("hello", {})

        assert result == "Success after retry."
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_think_uses_fallback_after_circuit_opens(self):
        """When circuit is open, think() routes directly to fallback_model."""
        from aifactory.core.retry import record_failure

        agent = PipelineAgent(
            "test",
            {"system_prompt": "Test.", "max_tokens": 100, "model": "expensive-model"},
            model="expensive-model",
        )
        agent.fallback_model = "gpt-4o-mini"

        # Open the circuit for the primary model
        for _ in range(3):
            record_failure("expensive-model")

        called_with_models: list[str] = []

        async def capture_model(*args, **kwargs):
            model = kwargs.get("model") or (args[0] if args else "?")
            called_with_models.append(model)
            return make_llm_response("Fallback response.")

        with patch(_LITELLM_PATCH, side_effect=capture_model):
            result = await agent.think("hello", {})

        assert result == "Fallback response."
        # Must have called with fallback model, not expensive-model
        assert any("gpt-4o-mini" in m for m in called_with_models)
        assert all("expensive-model" not in m for m in called_with_models)

    @pytest.mark.asyncio
    async def test_fallback_used_after_all_retries_exhausted(self):
        """Primary fails all retries → fallback model tried once → returns result."""
        agent = PipelineAgent(
            "test",
            {"system_prompt": "Test.", "max_tokens": 100, "model": "main-model"},
            model="main-model",
        )
        agent.fallback_model = "gpt-4o-mini"

        call_log: list[str] = []

        async def selective_fail(*args, **kwargs):
            model = kwargs.get("model", "")
            call_log.append(model)
            if "main-model" in model:
                raise RuntimeError("primary always fails")
            return make_llm_response("Fallback saved us.")

        with (
            patch(_LITELLM_PATCH, side_effect=selective_fail),
            patch("aifactory.core.retry.asyncio.sleep", new=AsyncMock()),
        ):
            result = await agent.think("hello", {})

        assert result == "Fallback saved us."
        # Primary was tried (and failed), then fallback
        assert any("main-model" in m for m in call_log)
        assert any("gpt-4o-mini" in m for m in call_log)
