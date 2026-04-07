"""
Dynamic Pipeline — The config-driven 8-node LangGraph agent hierarchy.

Ported from agent-arch/main.py into production with full integration:
  - LiteLLM for cross-provider LLM calls
  - Ability-based model selection per agent and per worker task
  - YAML-driven agent configs (config/agents/*.yaml)
  - 3-layer Qdrant memory (Agent → Team → Global)
  - SHARE: protocol for lightweight inter-agent context broadcasting
  - LangGraph MemorySaver checkpointing for human-in-the-loop interrupt()
  - WebSocket-compatible event broadcasting

Graph topology:
  START → master_receive → judge_brief → analyse
        → [researcher → reanalyse]?   (pre-analysis research loop)
        → [researcher]?               (execution research)
        → execution_lead              (spawns N workers dynamically)
        → judge_review
        → master_respond → END

Three conditional edges:
  1. After analyse   → direct_answer | researcher | execution_lead
  2. After researcher → reanalyse (if pre-analysis) | execution_lead
  3. After reanalyse  → researcher (if still needs research) | execution_lead
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import litellm
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from aifactory.core.agent_factory import (
    AgentRegistry,
    build_shared_context_block,
    create_worker_config,
    extract_share,
    merge_share,
    parse_json_robust,
)
from aifactory.core.model_selector import (
    ABILITIES,
    best_model,
    load_agent_cfg,
    load_available_models,
    load_graph_config,
    load_models,
    model_for_agent,
)
from aifactory.core.observability import traceable
from aifactory.core.sandbox import execute_code_blocks

logger = logging.getLogger(__name__)

# Dumps directory for run/judge artefacts
DUMPS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "dumps"

# Suppress litellm debug logs and disable SSL verification for internal proxies
litellm.suppress_debug_info = True
litellm.ssl_verify = False

# ── LiteLLM proxy config (populated from env, injected into every call) ───────

import os as _os  # noqa: E402

_LITELLM_EXTRA: dict[str, str] = {}
_api_base = _os.getenv("LITELLM_API_BASE", "").strip()
_api_key = _os.getenv("LITELLM_API_KEY", "").strip()
if _api_base:
    _LITELLM_EXTRA["base_url"] = _api_base
    # Only pass LITELLM_API_KEY when routing through a proxy.
    # Without a base_url, litellm uses the native provider key (OPENAI_API_KEY, etc.)
    if _api_key:
        _LITELLM_EXTRA["api_key"] = _api_key


def _proxy_model(model_id: str) -> str:
    """
    When routing through a LiteLLM proxy, prefix with 'openai/' so litellm
    calls the proxy's /v1/chat/completions instead of a native provider API.
    When no proxy is configured, return the model_id unchanged.
    """
    if not _api_base:
        return model_id
    if model_id.startswith("openai/"):
        return model_id
    return f"openai/{model_id}"


# =============================================================================
# Helpers
# =============================================================================


def _elapsed(agent: PipelineAgent) -> str:
    """Sum per-call elapsed times recorded in each action, formatted as a string."""
    if not agent._actions:
        return "-"
    total = sum(act.get("elapsed_s", 0.0) for act in agent._actions)
    return f"{total:.1f}s"


# =============================================================================
# Lightweight Agent — single class for all roles (config-driven)
# =============================================================================


class AgentBudgetExceededError(Exception):
    """Raised when a PipelineAgent exceeds its per-agent token budget."""


class PipelineAgent:
    """
    A single LLM agent with private message history.

    This is the production version of agent-arch's Agent class, integrated with:
      - litellm.acompletion for cross-provider LLM calls
      - Token tracking per agent
      - Actions audit log for judge review and run dumps
      - SHARE: protocol support
      - Per-agent token budget enforcement
      - LangSmith tracing via @traceable

    Role-specific behavior comes entirely from the YAML config + system_prompt.
    No Python subclasses needed — one class handles all roles.
    """

    #: Default per-agent token budget. Can be overridden via cfg["token_budget"].
    DEFAULT_TOKEN_BUDGET: int = 50_000

    def __init__(
        self,
        name: str,
        cfg: dict[str, Any],
        model: str | None = None,
    ) -> None:
        self.name = name
        self.model = model or cfg.get("model", "gpt-4o-mini")
        self.cfg = cfg
        self._system_prompt: str = cfg.get("system_prompt", "")
        self._messages: list[dict[str, str]] = []  # private history
        self._actions: list[dict[str, Any]] = []  # audit log
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.token_budget: int = int(cfg.get("token_budget", self.DEFAULT_TOKEN_BUDGET))

    @traceable(name="pipeline_agent_think", run_type="llm")
    async def think(
        self,
        user_msg: str,
        shared_ctx: dict[str, str] | None = None,
    ) -> str:
        """
        Send a message to the LLM and get a response.

        Injects shared_context into system prompt. Maintains private
        conversation history. Tracks tokens per call.

        Raises:
            AgentBudgetExceededError: when total_tokens reaches token_budget.
        """
        if self.total_tokens >= self.token_budget:
            raise AgentBudgetExceededError(
                f"Agent {self.name} exceeded token budget ({self.total_tokens}/{self.token_budget})"
            )

        system = self._build_system(shared_ctx)
        self._messages.append({"role": "user", "content": user_msg})

        t_start = datetime.now(UTC)
        resp = await litellm.acompletion(
            model=_proxy_model(self.model),
            messages=[{"role": "system", "content": system}, *self._messages],
            max_tokens=self.cfg.get("max_tokens", 1000),
            **_LITELLM_EXTRA,
        )
        elapsed_s = round((datetime.now(UTC) - t_start).total_seconds(), 2)

        raw_content = resp.choices[0].message.content
        if raw_content is None:
            # Some models/proxies return content differently — check for refusal or fallback
            logger.warning(
                "[%s] message.content is None (finish_reason=%s) — checking alternatives",
                self.name,
                getattr(resp.choices[0], "finish_reason", "?"),
            )
            # Try refusal field (OpenAI safety refusals)
            raw_content = getattr(resp.choices[0].message, "refusal", None) or ""
        reply = raw_content.strip()
        usage = resp.usage
        self.total_tokens += usage.total_tokens
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self._messages.append({"role": "assistant", "content": reply})

        logger.info(
            "[%s] prompt=%d completion=%d total=%d time=%.1fs",
            self.name,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
            elapsed_s,
        )

        self._actions.append(
            {
                "timestamp": t_start.isoformat(),
                "elapsed_s": elapsed_s,
                "input": user_msg,
                "output": reply,
                "shared_context_snapshot": dict(shared_ctx) if shared_ctx else {},
                "tokens": {
                    "prompt": usage.prompt_tokens,
                    "completion": usage.completion_tokens,
                    "total": usage.total_tokens,
                },
            }
        )

        return reply

    def _build_system(self, shared_ctx: dict[str, str] | None) -> str:
        base = self._system_prompt.strip()
        ctx_block = build_shared_context_block(shared_ctx or {})
        return base + ctx_block if ctx_block else base

    def dump(self) -> dict[str, Any]:
        """Return a complete snapshot of this agent for run dumps."""
        return {
            "name": self.name,
            "model": self.model,
            "role": self.cfg.get("role", self.name),
            "system_prompt": self._system_prompt[:200] + "..."
            if len(self._system_prompt) > 200
            else self._system_prompt,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "actions_count": len(self._actions),
        }


# =============================================================================
# LangGraph State
# =============================================================================


class DynamicPipelineState(TypedDict, total=False):
    """State that flows through the 8-node LangGraph pipeline."""

    # Input
    user_query: str
    final_response: str
    direct_answer: str  # set by analyser → bypasses all workers/judge

    # Shared context (SHARE: protocol)
    shared_context: dict[str, str]

    # Task breakdown
    tasks: list[dict[str, Any]]

    # Pre-analysis research
    research_for_analysis: bool
    research_for_analysis_query: str
    analysis_research_summary: str

    # Execution research
    research_needed: bool
    research_query: str
    research_summary: str

    # Results
    task_results: list[dict[str, Any]]
    agent_assessments: list[dict[str, Any]]
    judge_metrics: dict[str, Any]

    # Conversation history (for post-pipeline chat)
    conversation_history: list[dict[str, str]]


# =============================================================================
# Node factories
# =============================================================================


def make_master_receive(agent: PipelineAgent):
    async def master_receive(state: DynamicPipelineState) -> dict[str, Any]:
        query = state["user_query"]
        shared_ctx: dict[str, str] = {"objective": query}
        await agent.think(
            f'Request received: "{query}"\n'
            "One sentence: confirm you'll coordinate your team on this.",
            shared_ctx,
        )
        return {"shared_context": shared_ctx}

    return master_receive


def make_judge_brief(agent: PipelineAgent, registry: AgentRegistry):
    async def judge_brief(state: DynamicPipelineState) -> dict[str, Any]:
        shared_ctx = state.get("shared_context", {})
        named_agents = [a.name for a in registry.all() if a.name != agent.name]
        agents_str = ", ".join(named_agents) + ", Worker-task_N (one per task)"
        await agent.think(
            f'Pipeline starting for: "{state["user_query"]}"\n'
            f"Agents in this run: {agents_str}\n"
            "In one paragraph: what a successful outcome looks like "
            "and what you will look for when judging each agent.",
            shared_ctx,
        )
        return {}

    return judge_brief


def make_analyse(agent: PipelineAgent):
    async def analyse(state: DynamicPipelineState) -> dict[str, Any]:
        shared_ctx = state.get("shared_context", {})

        raw = await agent.think(
            f"Break down this request into atomic tasks.\n\n"
            f'Request: "{state["user_query"]}"\n\n'
            "Each task description must be an actionable goal — what the worker must accomplish. "
            "Never a statement of completed fact, never a method. "
            "Do not assume what info is available — workers can ask the user directly.\n\n"
            "RESEARCH FLAGS — set these carefully:\n"
            "- research_needed = true when the request involves researching, investigating, "
            "exploring, analysing, or comparing real-world topics/entities, OR when workers "
            "would benefit from a pre-gathered factual summary. If the user explicitly "
            "says 'research', always set this to true.\n"
            "- research_query = a concise query the researcher should answer.\n"
            "- research_for_analysis = true when YOU need background knowledge before "
            "you can decompose the request into tasks properly.\n"
            "- research_for_analysis_query = what you need the researcher to look up.\n\n"
            "Output a JSON object with keys: direct_answer (str, empty if not direct), "
            "research_for_analysis (bool), research_for_analysis_query (str), "
            "research_needed (bool), research_query (str), tasks (array of {id, description}).",
            shared_ctx,
        )

        try:
            parsed = parse_json_robust(raw)
        except ValueError as e:
            logger.warning("Analyser JSON parse failed — falling back to single task: %s", e)
            parsed = {
                "tasks": [{"id": "task_1", "description": state["user_query"]}],
                "research_needed": False,
                "research_query": "",
            }

        direct_answer: str = parsed.get("direct_answer") or ""
        if direct_answer:
            logger.info("Analyser: DIRECT ANSWER — skipping workers")
            return {
                "direct_answer": direct_answer,
                "tasks": [],
                "research_for_analysis": False,
                "research_for_analysis_query": "",
                "research_needed": False,
                "research_query": "",
                "shared_context": shared_ctx,
            }

        raw_tasks: list[dict] = parsed.get("tasks", [])
        tasks = [
            {
                "id": str(t.get("id") or f"task_{i + 1}"),
                "description": t.get("description") or t.get("task") or str(t),
            }
            for i, t in enumerate(raw_tasks)
        ]

        research_needed = bool(parsed.get("research_needed", False))
        research_query = parsed.get("research_query") or ""

        # Safety net: if the user explicitly asks for research but the
        # analyser missed it, override to ensure the researcher runs.
        research_signals = ("research", "investigate", "look up", "find out about", "deep dive")
        query_lower = state["user_query"].lower()
        if not research_needed and any(sig in query_lower for sig in research_signals):
            research_needed = True
            research_query = research_query or state["user_query"]
            logger.info("Analyser safety-net: overriding research_needed → True")

        return {
            "tasks": tasks,
            "research_for_analysis": bool(parsed.get("research_for_analysis", False)),
            "research_for_analysis_query": parsed.get("research_for_analysis_query") or "",
            "research_needed": research_needed,
            "research_query": research_query,
            "shared_context": {**shared_ctx, "tasks_count": str(len(tasks))},
        }

    return analyse


def make_researcher(agent: PipelineAgent):
    async def researcher(state: DynamicPipelineState) -> dict[str, Any]:
        for_analysis = state.get("research_for_analysis", False)
        if for_analysis:
            query = state.get("research_for_analysis_query") or state["user_query"]
        else:
            query = state.get("research_query") or state["user_query"]

        shared_ctx = state.get("shared_context", {})
        raw = await agent.think(
            f"Research: {query}\n\nReturn a dense, actionable summary — facts and findings only.",
            shared_ctx,
        )
        summary, updates = extract_share(raw)
        shared_ctx = merge_share(shared_ctx, updates, agent.name)

        if for_analysis:
            return {
                "analysis_research_summary": summary,
                "shared_context": {**shared_ctx, "analysis_research": summary[:200]},
            }
        else:
            return {
                "research_summary": summary,
                "shared_context": {**shared_ctx, "research_snippet": summary[:200]},
            }

    return researcher


def make_reanalyse(agent: PipelineAgent):
    async def reanalyse(state: DynamicPipelineState) -> dict[str, Any]:
        shared_ctx = state.get("shared_context", {})
        research = state.get("analysis_research_summary", "")

        raw = await agent.think(
            f"Refine your task breakdown using this research:\n\n{research}\n\n"
            f'Request: "{state["user_query"]}"\n\n'
            "Output a JSON object with keys: research_needed (bool), research_query (str), "
            "tasks (array of {id, description}).",
            shared_ctx,
        )

        try:
            parsed = parse_json_robust(raw)
        except ValueError as e:
            logger.warning("Reanalyse JSON parse failed — keeping original tasks: %s", e)
            return {"research_for_analysis": False}

        raw_tasks: list[dict] = parsed.get("tasks", [])
        tasks = [
            {
                "id": str(t.get("id") or f"task_{i + 1}"),
                "description": t.get("description") or t.get("task") or str(t),
            }
            for i, t in enumerate(raw_tasks)
        ]

        return {
            "tasks": tasks if tasks else state.get("tasks", []),
            "research_for_analysis": False,
            "research_needed": bool(parsed.get("research_needed", False)),
            "research_query": parsed.get("research_query") or "",
            "shared_context": {**shared_ctx, "tasks_count": str(len(tasks))},
        }

    return reanalyse


def make_execution_lead(
    lead: PipelineAgent,
    worker_cfg: dict[str, Any],
    models: list[dict[str, Any]],
    mode: str,
    registry: AgentRegistry,
):
    async def execution_lead(state: DynamicPipelineState) -> dict[str, Any]:
        shared_ctx = dict(state.get("shared_context", {}))
        tasks = state.get("tasks") or [{"id": "task_1", "description": state["user_query"]}]
        research = state.get("research_summary", "")

        # ── Planning call — skipped on resume after a worker QUESTION interrupt ─
        # LangGraph re-runs the entire node after interrupt() resume, so we cache
        # the plan on the lead agent object (same Python instance in the closure).
        # Cache key = task content, so a new request with different tasks re-plans.
        cache_key = tuple((t["id"], t["description"]) for t in tasks)
        cached_plan = getattr(lead, "_plan_cache", None)
        if cached_plan and cached_plan["key"] == cache_key:
            logger.info("Execution lead: resuming (plan cached)")
            worker_prompts: dict[str, str] = cached_plan["worker_prompts"]
            ability_reqs: dict[str, dict[str, float]] = cached_plan["ability_reqs"]
        else:
            tasks_list = "\n".join(f"  {t['id']}: {t['description']}" for t in tasks)
            research_block = (
                f"\nResearch summary (share with workers):\n{research[:500]}" if research else ""
            )
            plan_raw = await lead.think(
                f"Plan worker dispatch:{research_block}\n\n{tasks_list}\n\n"
                "JSON only — no prose:\n"
                '{"plan": [{"id": "task_1", "reasoning": 0-3, "coding": 0-3, '
                '"knowledge": 0-3, "writing": 0-3, "math": 0-3, '
                '"instruction_following": 0-3, "tool_use": 0-3, '
                '"speed": 0-3, "cost_efficiency": 0-3, "context_handling": 0-3}],\n'
                ' "prompts": {"task_1": "<concise worker instruction>"}}',
                shared_ctx,
            )

            worker_prompts = {}
            ability_reqs = {}
            try:
                parsed_plan = parse_json_robust(plan_raw)
                for row in parsed_plan.get("plan", []):
                    tid = str(row.get("id", ""))
                    ability_reqs[tid] = {a: float(row.get(a, 0.0)) for a in ABILITIES}
                worker_prompts = {str(k): str(v) for k, v in parsed_plan.get("prompts", {}).items()}
            except ValueError as e:
                logger.warning("Execution lead plan parse failed: %s", e)

            lead._plan_cache = {
                "key": cache_key,
                "worker_prompts": worker_prompts,
                "ability_reqs": ability_reqs,
                "completed": {},  # tid -> result dict, persists across interrupt resumes
            }

        completed: dict[str, dict[str, Any]] = lead._plan_cache["completed"]
        workers_cache: dict[str, PipelineAgent] = lead._plan_cache.setdefault("workers", {})

        # ── Worker dispatch ──────────────────────────────────────────
        task_results: list[dict[str, Any]] = []

        for task in tasks:
            tid = task["id"]
            worker_name = f"Worker-{tid}"

            # Skip workers that fully completed before an interrupt resume
            if tid in completed:
                task_results.append(completed[tid])
                logger.info("[%s] already completed (pre-interrupt) — skipping", worker_name)
                continue

            # Create worker with per-task model selection
            w_config = create_worker_config(
                task_id=tid,
                task_description=task["description"],
                ability_requirements=ability_reqs.get(tid),
                worker_prompt=worker_prompts.get(tid),
                mode=mode,
            )
            chosen_model = w_config["model"]

            # Build worker input with prior results
            prior = "\n\n".join(
                f"[{r['task_id']}]\n{r['result']}"
                for r in task_results
                if r.get("result") and r["result"].strip() != "NOT_NEEDED"
            )
            task_input = "Execute your assigned task." + (
                f"\n\nPrior task results:\n{prior}" if prior else ""
            )

            # Reuse cached worker agent if this task was interrupted mid-execution
            # (i.e., the worker itself asked a QUESTION before finishing).
            # On LangGraph resume, interrupt() returns the answer immediately — so
            # we feed the cached output back into the while loop without a redundant LLM call.
            cached_worker = workers_cache.get(tid)
            if cached_worker is not None:
                last_action = cached_worker._actions[-1] if cached_worker._actions else None
                if last_action and re.search(
                    r"(?m)^QUESTION:", last_action["output"], re.IGNORECASE
                ):
                    worker = cached_worker
                    raw = last_action["output"]
                    logger.info("[%s] resuming from cached question (no re-call)", worker_name)
                else:
                    worker = cached_worker
                    raw = await worker.think(task_input, shared_ctx)
            else:
                worker = registry.register(PipelineAgent(worker_name, w_config, chosen_model))
                workers_cache[tid] = worker
                raw = await worker.think(task_input, shared_ctx)

            # Handle QUESTION: interrupts
            while True:
                q_match = re.search(r"(?m)^QUESTION:\s*(.+)", raw, re.IGNORECASE)
                if not q_match:
                    break
                question = q_match.group(1).strip()
                logger.info("[%s] QUESTION: %s", worker_name, question)
                answer: str = interrupt(f"[{worker_name}] {question}")
                logger.info("[%s] ANSWER: %s", worker_name, answer)
                qa_key = f"qa{sum(1 for k in shared_ctx if k.startswith('qa'))}"
                shared_ctx = {**shared_ctx, qa_key: f"Q:{question}|A:{answer}"}
                raw = await worker.think(
                    f'The human answered: "{answer}"\n\nNow complete your task fully.',
                    shared_ctx,
                )

            result, updates = extract_share(raw)
            shared_ctx = merge_share(shared_ctx, updates, worker_name)

            # Guard: if extract_share stripped everything but worker produced output
            if not result.strip() and raw.strip():
                logger.warning(
                    "[%s] extract_share returned empty from %d-char response — using raw",
                    worker_name,
                    len(raw),
                )
                result = raw.strip()

            # ── Code execution: detect and run code blocks in worker output ──
            code_runs = await execute_code_blocks(raw, timeout=30)
            if code_runs:
                exec_parts: list[str] = []
                for lang, _code_str, exec_result in code_runs:
                    status = "✓" if exec_result.success else "✗"
                    logger.info(
                        "[%s] code exec [%s/%s]: %s",
                        worker_name,
                        lang,
                        exec_result.backend,
                        status,
                    )
                    exec_parts.append(f"[code:{lang} {status}]\n{exec_result.output[:500]}")
                exec_summary = "\n\n".join(exec_parts)
                result = f"{result}\n\n--- Code Execution Results ---\n{exec_summary}"

            not_needed = result.strip().upper() == "NOT_NEEDED"
            result_entry = {
                "task_id": tid,
                "agent": worker_name,
                "model": chosen_model,
                "description": task["description"],
                "result": result,
                "skipped": not_needed,
            }
            completed[tid] = result_entry  # persist across interrupt resumes
            if not not_needed:
                task_results.append(result_entry)

        # ── Consolidation ────────────────────────────────────────────
        # Truncate each result to limit consolidation input tokens
        max_chars = 600
        results_text = "\n\n".join(
            f"### {r['task_id']}: {r['description']}\n"
            f"{r['result'][:max_chars]}{'…' if len(r['result']) > max_chars else ''}"
            for r in task_results
        )
        consolidated = await lead.think(
            f"Consolidate into one coherent answer. Plain text only.\n\n{results_text}",
            shared_ctx,
        )
        clean_consolidated, updates = extract_share(consolidated)
        shared_ctx = merge_share(shared_ctx, updates, lead.name)

        return {
            "task_results": task_results,
            "shared_context": shared_ctx,
            "final_response": clean_consolidated,
        }

    return execution_lead


def make_judge_review(agent: PipelineAgent, registry: AgentRegistry, run_id: str = "unknown"):
    async def judge_review(state: DynamicPipelineState) -> dict[str, Any]:
        shared_ctx = state.get("shared_context", {})

        # Build run summary for judge (include elapsed times)
        agent_rows = "\n".join(
            f"{a.name},{a.cfg.get('role', a.name)},{len(a._actions)},{a.total_tokens},{_elapsed(a)}"
            for a in registry.all()
        )
        task_results = state.get("task_results", [])
        task_rows = (
            "\n".join(
                f"  {r['task_id']}: {r['description'][:70]}"
                f" | {len(r['result'])} chars | model: {r['model']}"
                for r in task_results
            )
            or "  none"
        )
        total_tokens = sum(a.total_tokens for a in registry.all())

        run_summary = (
            f"agents ({len(registry.all())}):\n{agent_rows}\n\n"
            f"tasks_completed: {len(task_results)}\n"
            f"task_detail:\n{task_rows}\n\n"
            f"research_for_analysis_ran: {bool(state.get('analysis_research_summary'))}\n"
            f"execution_research_ran: {bool(state.get('research_summary'))}\n"
            f"final_output_chars: {len(state.get('final_response', ''))}\n"
            f"total_tokens: {total_tokens}"
        )

        agent_count = len(registry.all())
        raw = await agent.think(
            f'Judge this pipeline run for: "{state["user_query"]}"\n\n'
            f"{run_summary}\n\n"
            f"Output a JSON object with two keys:\n"
            f"- assessments: array of {agent_count} objects, one per agent, each with: "
            f"agent (str), status (successful|unsuccessful|not_required), quality (0-3), "
            f"tokens (int), notes (str)\n"
            f"- metrics: object with: overall_status "
            f"(successful|partial|failed), total_tokens (int), "
            f"pipeline_efficiency (high|medium|low), "
            f"cost_rating (0-3), recommendations (str)",
            shared_ctx,
        )

        assessments: list[dict] = []
        metrics: dict = {}
        try:
            parsed = parse_json_robust(raw)
            assessments = [
                {
                    "agent": str(r.get("agent", "?")),
                    "status": str(r.get("status", "?")),
                    "quality": str(r.get("quality", "?")),
                    "tokens": str(r.get("tokens", "?")),
                    "notes": str(r.get("notes", "")),
                }
                for r in parsed.get("assessments", [])
            ]
            raw_metrics = parsed.get("metrics", {})
            if isinstance(raw_metrics, dict):
                metrics = {k: str(v) for k, v in raw_metrics.items()}
        except ValueError as e:
            logger.warning("Judge JSON parse failed: %s", e)

        # ── Build hierarchical plain-text report ─────────────────────
        _pricing = {m["id"]: m for m in load_models()}

        def _fmt_model(m: str) -> str:
            m = m.removeprefix("openai/")
            return m[-36:] if len(m) > 36 else m

        def _status_sym(s: str) -> str:
            return {"successful": "✓", "unsuccessful": "✗", "not_required": "—"}.get(s, "—")

        def _agent_cost(ag: PipelineAgent) -> str:
            model_id = ag.model.removeprefix("openai/")
            p = _pricing.get(model_id, {})
            inp = p.get("input_cost_per_1m")
            out = p.get("output_cost_per_1m")
            if inp is None or out is None:
                return "—"
            cost = (ag.prompt_tokens * inp + ag.completion_tokens * out) / 1_000_000
            return f"${cost:.4f}"

        col_w = 82
        sep = "═" * col_w
        line = "─" * col_w
        assess_map = {a["agent"]: a for a in assessments}
        ag_map = {a.name: a for a in registry.all()}

        def _agent_row(name: str, indent: str = "") -> list[str]:
            ag = ag_map.get(name)
            asmnt = assess_map.get(name, {})
            status = asmnt.get("status", "—")
            quality = "—" if status == "not_required" else asmnt.get("quality", "-")
            tokens = ag.total_tokens if ag else 0
            elapsed_str = _elapsed(ag) if ag else "—"
            model = _fmt_model(ag.model) if ag else "—"
            cost = _agent_cost(ag) if ag else "—"
            sym = _status_sym(status)
            label = f"{indent}{name}"
            return [
                f"  {label:<28} {model:<36} {tokens:>5} tok"
                f"  {elapsed_str:>6}  {cost:>9}  {sym:<12} {quality}"
            ]

        report_lines: list[str] = [
            sep,
            f"  RUN    {run_id}",
            f"  QUERY  {state.get('user_query', '')}",
            sep,
            "",
            "  EXECUTION PIPELINE",
            line,
            f"  {'agent':<28} {'model':<36} {'tokens':>9}"
            f"  {'time':>6}  {'cost':>9}  {'status':<12} quality",
            line,
        ]

        named_order = ["master", "judge", "analyser", "researcher", "execution_lead"]
        for name in named_order:
            if name not in ag_map:
                asmnt = assess_map.get(name, {})
                status = asmnt.get("status", "not_required")
                sym = _status_sym(status)
                report_lines.append(
                    f"  {name:<28} {'—':<36} {'—':>9}  {'—':>6}  {'—':>9}  {sym:<12} —"
                )
            else:
                report_lines.extend(_agent_row(name))
            if name == "execution_lead":
                for r in task_results:
                    wname = r["agent"]
                    report_lines.extend(_agent_row(wname, indent="  └─ "))
                    desc = r["description"][:60]
                    report_lines.append(f"     {'':28} task: {desc}")

        total_cost_known = sum(
            (
                ag.prompt_tokens * _pricing[mid].get("input_cost_per_1m", 0)
                + ag.completion_tokens * _pricing[mid].get("output_cost_per_1m", 0)
            )
            / 1_000_000
            for ag in registry.all()
            for mid in [ag.model.removeprefix("openai/")]
            if mid in _pricing and _pricing[mid].get("input_cost_per_1m") is not None
        )
        total_cost_str = f"${total_cost_known:.4f}" if total_cost_known > 0 else "—"

        report_lines += [
            line,
            f"  {'TOTAL':<28} {'':36} {total_tokens:>5} tok  {'':>6}  {total_cost_str:>9}",
            "",
            "  PIPELINE METRICS",
            line,
        ]
        for k, v in metrics.items():
            report_lines.append(f"  {k:<28} {v}")
        report_lines += ["", sep]

        report = "\n".join(report_lines)

        # ── Write judge dump (plain text report) ─────────────────────
        judge_dir = DUMPS_DIR / "judge"
        judge_dir.mkdir(parents=True, exist_ok=True)
        dump_path = judge_dir / f"{run_id}.txt"
        dump_path.write_text(report, encoding="utf-8")
        logger.info("Judge report saved: %s", dump_path)

        return {"agent_assessments": assessments, "judge_metrics": metrics}

    return judge_review


def make_master_respond(agent: PipelineAgent):
    async def master_respond(state: DynamicPipelineState) -> dict[str, Any]:
        shared_ctx = state.get("shared_context", {})
        history: list[dict] = list(state.get("conversation_history") or [])

        direct = state.get("direct_answer", "")
        if direct:
            logger.info("Master: direct answer — no consolidation needed")
            history.append({"role": "assistant", "content": direct})
            return {"final_response": direct, "conversation_history": history}

        team_output = state.get("final_response", "")
        final = await agent.think(
            f'Request: "{state["user_query"]}"\n\n'
            f"Team output:\n\n{team_output}\n\n"
            "Deliver this to the human. No mention of agents, teams, or internal process.",
            shared_ctx,
        )
        # Fallback: if master's synthesis is empty (truncated), use team output
        if not final.strip() and team_output.strip():
            logger.warning("Master respond returned empty — falling back to team output")
            final = team_output
        history.append({"role": "assistant", "content": final})
        return {"final_response": final, "conversation_history": history}

    return master_respond


# =============================================================================
# Dump — full JSON run artefact
# =============================================================================


def write_dump(
    run_id: str,
    cfg: dict[str, Any],
    result: dict[str, Any],
    registry: AgentRegistry,
    elapsed: float,
    request: str = "",
) -> Path:
    """
    Write a complete JSON dump of the pipeline run.

    Args:
        run_id: Unique run identifier (timestamp-based).
        cfg: Graph config dict (mode, model, etc.).
        result: Final pipeline state dict.
        registry: AgentRegistry with all agents.
        elapsed: Wall-clock time in seconds.
        request: Original human request text.

    Returns:
        Path to the written dump file.
    """
    runs_dir = DUMPS_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    dump_path = runs_dir / f"{run_id}.json"

    payload = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "config": {
            "model": cfg.get("model", "unknown"),
            "mode": cfg.get("mode", "balanced"),
            "request": request,
        },
        "shared_context_final": result.get("shared_context", {}),
        "tasks": result.get("tasks", []),
        "research_needed": result.get("research_needed", False),
        "research_summary": result.get("research_summary", ""),
        "task_results": result.get("task_results", []),
        "agent_assessments": result.get("agent_assessments", []),
        "judge_metrics": result.get("judge_metrics", {}),
        "final_response": result.get("final_response", ""),
        "agents": [a.dump() for a in registry.all()],
        "totals": {
            "agents": len(registry.all()),
            "total_tokens": sum(a.total_tokens for a in registry.all()),
            "total_actions": sum(len(a._actions) for a in registry.all()),
        },
    }

    dump_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("Run dump saved: %s", dump_path)
    return dump_path


# =============================================================================
# Routing
# =============================================================================


def route_after_analysis(state: DynamicPipelineState) -> str:
    if state.get("direct_answer"):
        return "master_respond"
    if state.get("research_for_analysis"):
        return "researcher"
    if state.get("research_needed"):
        return "researcher"
    return "execution_lead"


def route_after_researcher(state: DynamicPipelineState) -> str:
    return "reanalyse" if state.get("research_for_analysis") else "execution_lead"


def route_after_reanalyse(state: DynamicPipelineState) -> str:
    return "researcher" if state.get("research_needed") else "execution_lead"


# =============================================================================
# Graph builder
# =============================================================================


def build_dynamic_pipeline(
    mode: str | None = None,
    registry: AgentRegistry | None = None,
    checkpointer: MemorySaver | None = None,
    run_id: str | None = None,
) -> tuple[Any, AgentRegistry]:
    """
    Build and compile the 8-node dynamic pipeline graph.

    Args:
        mode: "best", "balanced", or "cost_efficient". Auto-loaded if None.
        registry: Optional AgentRegistry. Created if None.
        checkpointer: Optional MemorySaver for interrupt/resume support.
        run_id: Optional run identifier for dump files. Auto-generated if None.

    Returns:
        (compiled_graph, registry) — the compiled LangGraph app and the agent registry.
    """
    if registry is None:
        registry = AgentRegistry()
    if run_id is None:
        run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    graph_cfg = load_graph_config()
    if mode is None:
        mode = graph_cfg.get("mode", "balanced")

    fallback_model = graph_cfg.get("model", "gpt-4o-mini")
    models = load_available_models()

    # Load agent configs
    agent_cfgs = {
        n: load_agent_cfg(n)
        for n in ["master", "judge", "analyser", "researcher", "execution_lead", "worker"]
    }

    # Master always uses best_model (mode-independent)
    master_model = best_model(models)

    # Named agents: ability profile x mode bias
    judge_model = model_for_agent(agent_cfgs["judge"], models, mode, fallback_model)
    analyser_model = model_for_agent(agent_cfgs["analyser"], models, mode, fallback_model)
    researcher_model = model_for_agent(agent_cfgs["researcher"], models, mode, fallback_model)
    lead_model = model_for_agent(agent_cfgs["execution_lead"], models, mode, fallback_model)

    logger.info(
        "Pipeline models [mode=%s]: master=%s, judge=%s, analyser=%s, researcher=%s, lead=%s",
        mode,
        master_model,
        judge_model,
        analyser_model,
        researcher_model,
        lead_model,
    )

    # Create agent instances
    master = registry.register(
        PipelineAgent("master", {**agent_cfgs["master"], "model": master_model}, master_model)
    )
    judge = registry.register(
        PipelineAgent("judge", {**agent_cfgs["judge"], "model": judge_model}, judge_model)
    )
    analyser = registry.register(
        PipelineAgent(
            "analyser", {**agent_cfgs["analyser"], "model": analyser_model}, analyser_model
        )
    )
    researcher_ag = registry.register(
        PipelineAgent(
            "researcher", {**agent_cfgs["researcher"], "model": researcher_model}, researcher_model
        )
    )
    lead = registry.register(
        PipelineAgent(
            "execution_lead", {**agent_cfgs["execution_lead"], "model": lead_model}, lead_model
        )
    )

    # Build LangGraph
    g = StateGraph(DynamicPipelineState)
    g.add_node("master_receive", make_master_receive(master))
    g.add_node("judge_brief", make_judge_brief(judge, registry))
    g.add_node("analyse", make_analyse(analyser))
    g.add_node("researcher", make_researcher(researcher_ag))
    g.add_node("reanalyse", make_reanalyse(analyser))
    g.add_node(
        "execution_lead",
        make_execution_lead(lead, agent_cfgs["worker"], models, mode, registry),
    )
    g.add_node("judge_review", make_judge_review(judge, registry, run_id))
    g.add_node("master_respond", make_master_respond(master))

    g.add_edge(START, "master_receive")
    g.add_edge("master_receive", "judge_brief")
    g.add_edge("judge_brief", "analyse")
    g.add_conditional_edges(
        "analyse",
        route_after_analysis,
        {
            "master_respond": "master_respond",
            "researcher": "researcher",
            "execution_lead": "execution_lead",
        },
    )
    g.add_conditional_edges(
        "researcher",
        route_after_researcher,
        {
            "reanalyse": "reanalyse",
            "execution_lead": "execution_lead",
        },
    )
    g.add_conditional_edges(
        "reanalyse",
        route_after_reanalyse,
        {
            "researcher": "researcher",
            "execution_lead": "execution_lead",
        },
    )
    g.add_edge("execution_lead", "judge_review")
    g.add_edge("judge_review", "master_respond")
    g.add_edge("master_respond", END)

    compiled = g.compile(checkpointer=checkpointer)
    logger.info("Dynamic pipeline compiled [8 nodes, 3 conditional edges]")
    return compiled, registry
