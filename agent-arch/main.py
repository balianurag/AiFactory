#!/usr/bin/env python3
"""
Multi-Agent LangGraph Experiment

Architecture:
  Human → Master Agent
            ├── Analyser          (breaks query → tasks[], flags research)
            ├── Researcher        (conditional — only if analyser flags it)
            └── Execution Lead    (spawns one Worker per task, consolidates)
                     ├── Worker-task_1 … Worker-task_N  (dynamic)
            └── Master Agent      (presents final answer to human)

Context model:
  - Each agent has a private message history (never shared)
  - A shared_context dict is injected into every agent's system prompt
  - Only Master Agent sends/receives human-facing messages

Run:
    uv run python experiments/main.py
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

ROOT = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent / "config"
DUMPS_DIR = Path(__file__).parent / "dumps"

sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).parent / ".env")

import os  # noqa: E402

import litellm  # noqa: E402
import yaml  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command, interrupt  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.tree import Tree  # noqa: E402

litellm.suppress_debug_info = True
litellm.ssl_verify = False
console = Console()

# LiteLLM proxy config — populated from .env, injected into every completion call
_LITELLM_EXTRA: dict = {}
_api_base = os.getenv("LITELLM_API_BASE", "").strip()
_api_key = os.getenv("LITELLM_API_KEY", "").strip()
if _api_base:
    _LITELLM_EXTRA["base_url"] = _api_base
if _api_key:
    _LITELLM_EXTRA["api_key"] = _api_key


def _proxy_model(model_id: str) -> str:
    """
    When routing through a LiteLLM proxy, force OpenAI-compatible format by
    prefixing the model ID with 'openai/'. Without this, LiteLLM inspects the
    model string (e.g. 'us.anthropic.*' or 'gemini-*') and calls the native
    provider API instead of the proxy's /v1/chat/completions endpoint.
    """
    if not _api_base:
        return model_id
    if model_id.startswith("openai/"):
        return model_id
    return f"openai/{model_id}"


# =============================================================================
# Config loaders
# =============================================================================


ABILITIES: list[str] = [
    "reasoning",
    "coding",
    "math",
    "writing",
    "instruction_following",
    "tool_use",
    "knowledge",
    "speed",
    "cost_efficiency",
    "context_handling",
]

# Mode blending: each mode defines a preference profile + a fixed alpha.
# alpha controls how strongly the mode overrides per-task requirements:
#   0.0 = pure task requirements, 1.0 = pure mode profile
# best          → high alpha so quality always wins regardless of task
# balanced      → low alpha so task requirements lead, mode nudges
# cost_efficient → high alpha so speed/cost always wins regardless of task
MODE_PROFILES: dict[str, dict] = {
    "best": {
        "alpha": 1.0,
        "reasoning": 3,
        "coding": 3,
        "math": 2,
        "writing": 3,
        "instruction_following": 3,
        "tool_use": 3,
        "knowledge": 3,
        "speed": 0,
        "cost_efficiency": 0,
        "context_handling": 3,
    },
    "balanced": {
        "alpha": 0.35,
        **{a: 2 for a in ABILITIES},
    },
    "cost_efficient": {
        "alpha": 0.75,
        "reasoning": 0,
        "coding": 0,
        "math": 0,
        "writing": 0,
        "instruction_following": 0,
        "tool_use": 0,
        "knowledge": 0,
        "speed": 3,
        "cost_efficiency": 3,
        "context_handling": 0,
    },
}


def load_graph_config() -> dict:
    return yaml.safe_load((CONFIG_DIR / "graph.yaml").read_text())


def load_agent_cfg(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / "agents" / f"{name}.yaml").read_text())


def load_models() -> list[dict]:
    return yaml.safe_load((CONFIG_DIR / "models.yaml").read_text())["models"]


def select_model(
    requirements: dict[str, float],
    models: list[dict],
    mode: str = "balanced",
) -> tuple[str, float]:
    """
    Blended dot-product scoring.

    Effective requirement per ability:
        (1 - alpha) * task_requirement  +  alpha * mode_profile

    alpha is baked into each mode profile:
      best/cost_efficient = 0.75 (mode mostly overrides task)
      balanced            = 0.35 (task leads, mode nudges)
    Returns (model_id, score).
    """
    profile = MODE_PROFILES.get(mode, MODE_PROFILES["balanced"])
    alpha = profile["alpha"]
    best_id, best_score = models[0]["id"], -1.0
    for m in models:
        score = sum(
            ((1 - alpha) * requirements.get(a, 0.0) + alpha * profile[a]) * float(m.get(a, 0.0))
            for a in ABILITIES
        )
        if score > best_score:
            best_score, best_id = score, m["id"]
    return best_id, round(best_score, 4)


def best_model(models: list[dict]) -> str:
    """
    Returns the model with the highest average quality score.
    Speed and cost_efficiency excluded — used exclusively for master agent.
    """
    quality = [a for a in ABILITIES if a not in ("speed", "cost_efficiency")]
    best_id, best_avg = models[0]["id"], -1.0
    for m in models:
        avg = sum(float(m.get(a, 0.0)) for a in quality) / len(quality)
        if avg > best_avg:
            best_avg, best_id = avg, m["id"]
    return best_id


# =============================================================================
# Prompt generation — Master briefs the team
# =============================================================================


# =============================================================================
# Logging
# =============================================================================


def _section(title: str, color: str = "blue") -> None:
    console.print(Rule(f"[bold {color}]{title}[/bold {color}]"))


def _log(agent: str, label: str, msg: str) -> None:
    console.print(f"  [dim][{agent}][/dim] [dim]{label}:[/dim] {msg}")


# =============================================================================
# Shared-context helpers
# =============================================================================

_SHARE_MARKER = "\nSHARE:"


def _parse_json(raw: str) -> dict:
    """
    Robustly extract a JSON object from a model response.
    Handles: SHARE blocks appended after JSON, code fences, leading/trailing prose.
    Raises ValueError if no valid JSON object is found.
    """
    # Strip SHARE block first — models often append it after their JSON output
    text = raw.split("\nSHARE:", 1)[0] if "\nSHARE:" in raw else raw
    # Strip code fences
    text = re.sub(r"```[a-z]*\n?", "", text).replace("```", "").strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back: find the outermost { ... } block (handles leading prose)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"no valid JSON object found in: {text[:300]!r}")


def extract_share(response: str) -> tuple[str, dict]:
    """
    Split an agent response into (result, shared_updates).

    Agents may append a SHARE: block with key: value lines to broadcast
    facts/decisions to all subsequent agents:

        [actual work result]
        SHARE:
        framework: FastAPI
        db: PostgreSQL

    Returns the cleaned result and a dict of updates to merge into
    shared_context. If no SHARE block is present, updates is {}.
    """
    if _SHARE_MARKER not in response:
        return response.strip(), {}

    result_part, share_part = response.split(_SHARE_MARKER, 1)
    updates: dict = {}
    for line in share_part.strip().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip()
            if k:
                updates[k] = v
    return result_part.strip(), updates


def _merge_share(shared_ctx: dict, updates: dict, agent_name: str) -> dict:
    if not updates:
        return shared_ctx
    _log(agent_name, "→ shared", f"[dim]{list(updates.keys())}[/dim]")
    return {**shared_ctx, **updates}


# =============================================================================
# Agent Registry — collects all agent instances for the end-of-run dump
# =============================================================================


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: list[Agent] = []

    def register(self, agent: Agent) -> Agent:
        self._agents.append(agent)
        return agent

    def get(self, name: str) -> Agent | None:
        for a in self._agents:
            if a.name == name:
                return a
        return None

    def all(self) -> list[Agent]:
        return list(self._agents)


# =============================================================================
# Agent
# =============================================================================


class Agent:
    """
    A single LLM agent with private message history.

    Private context  : self._messages  — never exposed to other agents
    Shared context   : injected into system prompt on every call
    Actions log      : self._actions   — timestamped record of every call
    """

    def __init__(
        self,
        name: str,
        cfg: dict,
        default_model: str,
        generated_prompt: str | None = None,
    ) -> None:
        self.name = name
        self.model = cfg.get("model", default_model)
        self.cfg = cfg
        # Use master-generated prompt if provided, fall back to yaml system_prompt
        self._system_prompt: str = generated_prompt or cfg.get("system_prompt", "")
        self.prompt_source: str = "generated" if generated_prompt else "yaml"
        self._messages: list[dict] = []  # private conversation history
        self._actions: list[dict] = []  # audit log of every think() call
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    async def think(self, user_msg: str, shared_ctx: dict | None = None) -> str:
        system = self._build_system(shared_ctx)
        self._messages.append({"role": "user", "content": user_msg})

        _log(
            self.name,
            "→ input",
            f"[yellow]{user_msg[:120]}{'...' if len(user_msg) > 120 else ''}[/yellow]",
        )
        _log(self.name, "model", f"[dim]{self.model}[/dim]")
        _log(
            self.name, "history", f"[dim]{len(self._messages)} message(s) in private context[/dim]"
        )

        t_start = datetime.now(UTC)
        resp = await litellm.acompletion(
            model=_proxy_model(self.model),
            messages=[{"role": "system", "content": system}] + self._messages,
            max_tokens=self.cfg.get("max_tokens", 1000),
            **_LITELLM_EXTRA,
        )
        elapsed_s = round((datetime.now(UTC) - t_start).total_seconds(), 2)

        reply = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        self.total_tokens += usage.total_tokens
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self._messages.append({"role": "assistant", "content": reply})

        _log(
            self.name,
            "tokens",
            f"prompt={usage.prompt_tokens}  completion={usage.completion_tokens}  total={usage.total_tokens}  time={elapsed_s}s",
        )
        _log(
            self.name,
            "← output",
            f"[green]{reply[:200]}{'...' if len(reply) > 200 else ''}[/green]",
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

    def _build_system(self, shared_ctx: dict | None) -> str:
        base = self._system_prompt.strip()
        if shared_ctx:
            lines = [f"{k}: {v}" for k, v in shared_ctx.items() if v]
            if lines:
                return base + "\n\n## Shared\n" + "\n".join(lines)
        return base

    def dump(self) -> dict:
        """Return a complete snapshot of this agent for the run dump."""
        return {
            "name": self.name,
            "model": self.model,
            "role": self.cfg.get("role", self.name),
            "prompt_source": self.prompt_source,
            "system_prompt": self._system_prompt,
            "total_tokens": self.total_tokens,
            "private_history": self._messages,
            "actions": self._actions,
        }


# =============================================================================
# LangGraph State
# =============================================================================


class SharedState(TypedDict, total=False):
    user_query: str
    final_response: str
    direct_answer: str  # set by analyser → bypasses all workers/judge; master delivers directly
    shared_context: dict
    tasks: list[dict]
    # Pre-analysis research (feeds back into analyser for refined breakdown)
    research_for_analysis: bool
    research_for_analysis_query: str
    analysis_research_summary: str
    # Execution research (feeds into workers)
    research_needed: bool
    research_query: str
    research_summary: str
    task_results: list[dict]
    agent_assessments: list[dict]  # [{agent, status, quality, tokens, notes}] from judge
    judge_metrics: (
        dict  # {overall_status, total_tokens, pipeline_efficiency, cost_rating, recommendations}
    )
    conversation_history: list[dict]


# =============================================================================
# Node factories
# =============================================================================


def make_master_receive(agent: Agent):
    async def master_receive(state: SharedState) -> dict[str, Any]:
        _section("Master Agent  ·  Receive Human Request", "blue")
        query = state["user_query"]
        shared_ctx: dict = {"objective": query}
        await agent.think(
            f'Request received: "{query}"\nOne sentence: confirm you\'ll coordinate your team on this.',
            shared_ctx,
        )
        return {"shared_context": shared_ctx}

    return master_receive


def make_analyse(agent: Agent, default_model: str):
    async def analyse(state: SharedState) -> dict[str, Any]:
        _section("Analyser Agent  ·  Break Down Query", "cyan")
        shared_ctx = state.get("shared_context", {})

        raw = await agent.think(
            f"Break down this request into atomic tasks.\n\n"
            f'Request: "{state["user_query"]}"\n\n'
            "Each task description must be an actionable goal — what the worker must accomplish. "
            "Never a statement of completed fact ('name is known'), never a method. "
            "Do not assume what info is available — workers can ask the user directly.\n\n"
            "Output a JSON object with keys: research_for_analysis (bool), research_for_analysis_query (str), "
            "research_needed (bool), research_query (str), tasks (array of {id, description}).",
            shared_ctx,
        )

        try:
            parsed = _parse_json(raw)
        except ValueError as e:
            _log(
                agent.name,
                "warn",
                f"[yellow]JSON parse failed — falling back to single task ({e})[/yellow]",
            )
            parsed = {
                "tasks": [{"id": "task_1", "description": state["user_query"]}],
                "research_needed": False,
                "research_query": "",
            }
        direct_answer: str = parsed.get("direct_answer") or ""
        if direct_answer:
            _log(agent.name, "route", "[bold green]DIRECT ANSWER — skipping workers[/bold green]")
            _log(agent.name, "answer", f"[green]{direct_answer[:120]}[/green]")
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
        tasks: list[dict] = [
            {
                "id": str(t.get("id") or f"task_{i + 1}"),
                "description": t.get("description") or t.get("task") or str(t),
            }
            for i, t in enumerate(raw_tasks)
        ]
        research_for_analysis: bool = bool(parsed.get("research_for_analysis", False))
        research_for_analysis_query: str = parsed.get("research_for_analysis_query") or ""
        research_needed: bool = bool(parsed.get("research_needed", False))
        research_query: str = parsed.get("research_query") or ""

        t = Table(title="Task Breakdown", border_style="dim")
        t.add_column("ID", style="cyan", no_wrap=True)
        t.add_column("Description")
        for task in tasks:
            t.add_row(task.get("id", "?"), task.get("description", "?"))
        console.print(t)
        _log(
            agent.name,
            "research_for_analysis",
            f"[{'yellow' if research_for_analysis else 'dim'}]{research_for_analysis}[/]",
        )
        _log(
            agent.name,
            "research_needed",
            f"[{'green' if research_needed else 'dim'}]{research_needed}[/]",
        )

        return {
            "tasks": tasks,
            "research_for_analysis": research_for_analysis,
            "research_for_analysis_query": research_for_analysis_query,
            "research_needed": research_needed,
            "research_query": research_query,
            "shared_context": {**shared_ctx, "tasks_count": str(len(tasks))},
        }

    return analyse


def make_researcher(agent: Agent):
    async def researcher(state: SharedState) -> dict[str, Any]:
        for_analysis = state.get("research_for_analysis", False)
        if for_analysis:
            _section("Researcher Agent  ·  Pre-Analysis Research", "yellow")
            query = state.get("research_for_analysis_query") or state["user_query"]
        else:
            _section("Researcher Agent  ·  Execution Research", "yellow")
            query = state.get("research_query") or state["user_query"]

        shared_ctx = state.get("shared_context", {})
        raw = await agent.think(
            f"Research: {query}\n\nReturn a dense, actionable summary — facts and findings only, no padding.",
            shared_ctx,
        )
        summary, updates = extract_share(raw)
        shared_ctx = _merge_share(shared_ctx, updates, agent.name)

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


def make_reanalyse(agent: Agent):
    async def reanalyse(state: SharedState) -> dict[str, Any]:
        _section("Analyser Agent  ·  Refined Breakdown (post-research)", "cyan")
        shared_ctx = state.get("shared_context", {})
        research = state.get("analysis_research_summary", "")

        raw = await agent.think(
            f"Refine your task breakdown using this research:\n\n{research}\n\n"
            f'Request: "{state["user_query"]}"\n\n'
            "Task descriptions must be outcomes only, not methods. "
            "Set research_needed to true and provide research_query if workers still need external knowledge.\n\n"
            "Output a JSON object with keys: research_needed (bool), research_query (str), "
            "tasks (array of {id, description}).",
            shared_ctx,
        )

        try:
            parsed = _parse_json(raw)
        except ValueError as e:
            _log(
                agent.name,
                "warn",
                f"[yellow]JSON parse failed on reanalyse — keeping original tasks ({e})[/yellow]",
            )
            return {"research_for_analysis": False}

        raw_tasks: list[dict] = parsed.get("tasks", [])
        tasks: list[dict] = [
            {
                "id": str(t.get("id") or f"task_{i + 1}"),
                "description": t.get("description") or t.get("task") or str(t),
            }
            for i, t in enumerate(raw_tasks)
        ]
        research_needed: bool = bool(parsed.get("research_needed", False))
        research_query: str = parsed.get("research_query") or ""

        t = Table(title="Refined Task Breakdown", border_style="dim")
        t.add_column("ID", style="cyan", no_wrap=True)
        t.add_column("Description")
        for task in tasks:
            t.add_row(task.get("id", "?"), task.get("description", "?"))
        console.print(t)
        _log(
            agent.name,
            "research_needed",
            f"[{'green' if research_needed else 'dim'}]{research_needed}[/]",
        )

        return {
            "tasks": tasks if tasks else state.get("tasks", []),
            "research_for_analysis": False,  # clear so researcher routes correctly next time
            "research_needed": research_needed,
            "research_query": research_query,
            "shared_context": {**shared_ctx, "tasks_count": str(len(tasks))},
        }

    return reanalyse


def make_execution_lead(
    lead: Agent,
    worker_cfg: dict,
    models: list[dict],
    default_model: str,
    mode: str,
    registry: AgentRegistry,
):
    async def execution_lead(state: SharedState) -> dict[str, Any]:
        shared_ctx = state.get("shared_context", {})
        tasks = state.get("tasks") or [
            {"id": "task_1", "description": state["user_query"], "type": "other"}
        ]
        research = state.get("research_summary", "")

        # ── Planning call — skipped on resume after a worker QUESTION interrupt ─
        # LangGraph re-runs the entire node after interrupt() resume, so we cache
        # the plan on the lead agent object (same Python instance in the closure).
        # Cache key = task content, so a new request with different tasks re-plans.
        cache_key = tuple((t["id"], t["description"]) for t in tasks)
        cached_plan = getattr(lead, "_plan_cache", None)
        if cached_plan and cached_plan["key"] == cache_key:
            _section("Execution Lead  ·  Resuming (plan cached)", "magenta")
            worker_prompts: dict[str, str] = cached_plan["worker_prompts"]
            ability_reqs: dict[str, dict] = cached_plan["ability_reqs"]
        else:
            _section("Execution Lead  ·  Planning", "magenta")
            tasks_list = "\n".join(f"  {t['id']}: {t['description']}" for t in tasks)
            research_block = f"\nResearch available:\n{research}" if research else ""
            plan_raw = await lead.think(
                f"Dispatch workers for these tasks:{research_block}\n\n{tasks_list}\n\n"
                "Output a JSON object with two keys:\n"
                "- plan: array of {id, reasoning, coding, math, writing, instruction_following, tool_use, knowledge, speed, cost_efficiency, context_handling} "
                "where each ability is scored 0-3 (0=not needed, 1=basic, 2=solid, 3=critical)\n"
                "- prompts: object mapping each task id to a complete worker spec (what to produce, exact format, all constraints)",
                shared_ctx,
            )
            worker_prompts = {}
            ability_reqs = {}
            try:
                parsed_plan = _parse_json(plan_raw)
                for row in parsed_plan.get("plan", []):
                    tid = str(row.get("id", ""))
                    ability_reqs[tid] = {a: float(row.get(a, 0.0)) for a in ABILITIES}
                worker_prompts = {str(k): str(v) for k, v in parsed_plan.get("prompts", {}).items()}
            except ValueError as e:
                _log(lead.name, "warn", f"[yellow]JSON parse failed: {e}[/yellow]")
            lead._plan_cache = {
                "key": cache_key,
                "worker_prompts": worker_prompts,
                "ability_reqs": ability_reqs,
                "completed": {},  # tid -> result dict, persists across interrupt resumes
            }

        completed: dict = lead._plan_cache["completed"]
        workers_cache: dict = lead._plan_cache.setdefault("workers", {})

        # ── Model selection + worker dispatch ─────────────────────────────────
        _section("Execution Lead  ·  Spawning Workers", "magenta")
        _log(lead.name, "workers", f"[cyan]Spawning {len(tasks)} worker(s)[/cyan]")

        # Print model selection table
        sel_table = Table(title="Worker Model Selection", border_style="dim")
        sel_table.add_column("Worker", style="cyan")
        sel_table.add_column("Model")
        sel_table.add_column("Score", justify="right")

        task_results: list[dict] = []
        for task in tasks:
            tid = task["id"]
            worker_name = f"Worker-{tid}"

            # Skip workers that fully completed before an interrupt resume
            if tid in completed:
                task_results.append(completed[tid])
                _log(worker_name, "skip", "[dim]already completed (pre-interrupt)[/dim]")
                continue

            # Select model via dot-product scoring with mode bias
            reqs = (
                ability_reqs.get(tid) or worker_cfg.get("abilities") or {a: 0.5 for a in ABILITIES}
            )
            chosen_model, score = select_model(reqs, models, mode)
            sel_table.add_row(worker_name, f"[dim]{chosen_model}[/dim]", str(score))

            # Worker cfg with selected model
            w_cfg = {**worker_cfg, "model": chosen_model}
            # Build base prompt from execution lead's plan, or fall back to minimal spec.
            # The QUESTION channel is always appended in Python — never left to the LLM to include.
            _base_prompt = worker_prompts.get(tid) or (
                f"Task: {task['description']}\n"
                + (f"Research context:\n{research}\n" if research else "")
                + "Deliver a complete, precise result."
            )
            prompt = (
                _base_prompt.rstrip()
                + "\n\nIMPORTANT: If you are missing any information needed to complete this task, "
                "you MUST stop and output ONLY this single line — nothing before it, nothing after it:\n"
                "QUESTION: <your question here>\n"
                "Do not attempt to answer, proceed, or guess. Ask first.\n\n"
                "If this task is not applicable given the results of prior tasks, output exactly:\n"
                "NOT_NEEDED"
            )

            _section(f"  {worker_name}  ·  {task['description'][:60]}", "green")
            _log(worker_name, "model", f"[dim]{chosen_model}[/dim]  score={score}")

            # Inject prior completed results so workers can make conditional decisions.
            # Full result text — no truncation — so downstream workers have complete context.
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
                    _log(
                        worker_name,
                        "resume",
                        "[dim]resuming from cached question (no re-call)[/dim]",
                    )
                else:
                    worker = cached_worker
                    raw = await worker.think(task_input, shared_ctx)
            else:
                worker = registry.register(Agent(worker_name, w_cfg, chosen_model, prompt))
                workers_cache[tid] = worker
                raw = await worker.think(task_input, shared_ctx)

            # Handle clarifying questions — loop until no more QUESTION: lines
            while True:
                _q_match = re.search(r"(?m)^QUESTION:\s*(.+)", raw, re.IGNORECASE)
                if not _q_match:
                    break
                question = _q_match.group(1).strip()
                _log(worker_name, "question", f"[yellow]{question}[/yellow]")
                answer: str = interrupt(f"[{worker_name}] {question}")
                _log(worker_name, "← answer", f"[cyan]{answer}[/cyan]")
                qa_key = f"qa{sum(1 for k in shared_ctx if k.startswith('qa'))}"
                shared_ctx = {**shared_ctx, qa_key: f"Q:{question}|A:{answer}"}
                raw = await worker.think(
                    f'The human answered: "{answer}"\n\nNow complete your task fully.',
                    shared_ctx,
                )

            result, updates = extract_share(raw)
            shared_ctx = _merge_share(shared_ctx, updates, worker_name)

            # Detect self-skip
            not_needed = result.strip().upper() == "NOT_NEEDED"
            if not_needed:
                _log(worker_name, "skip", "[dim]NOT_NEEDED — task skipped by worker[/dim]")
            else:
                _log(worker_name, "done", f"[green]{len(result)} chars[/green]")

            result_entry = {
                "task_id": tid,
                "agent": worker_name,
                "model": chosen_model,
                "score": score,
                "description": task["description"],
                "result": result,
                "skipped": not_needed,
            }
            completed[tid] = result_entry
            if not not_needed:
                task_results.append(result_entry)

        console.print(sel_table)

        # ── Consolidation ─────────────────────────────────────────────────────
        _section("Execution Lead  ·  Consolidating", "magenta")
        results_text = "\n\n".join(
            f"### {r['task_id']}: {r['description']}\n{r['result']}" for r in task_results
        )
        raw_consolidated = await lead.think(
            f"Your workers have completed their tasks. Consolidate their results into a "
            f"single coherent answer for the user. Plain text only — no TOON, no JSON, no code fences.\n\n"
            f"{results_text}",
            shared_ctx,
        )
        consolidated, updates = extract_share(raw_consolidated)
        shared_ctx = _merge_share(shared_ctx, updates, lead.name)
        return {
            "task_results": task_results,
            "shared_context": shared_ctx,
            "final_response": consolidated,
        }

    return execution_lead


def make_judge_brief(agent: Agent, registry: AgentRegistry):
    async def judge_brief(state: SharedState) -> dict[str, Any]:
        _section("Judge  ·  Briefing", "bright_blue")
        shared_ctx = state.get("shared_context", {})
        named_agents = [a.name for a in registry.all() if a.name != agent.name]
        agents_str = ", ".join(named_agents) + ", Worker-task_N (one per task)"
        await agent.think(
            f'Pipeline starting for: "{state["user_query"]}"\n'
            f"Agents in this run: {agents_str}\n"
            "In one paragraph: what a successful outcome looks like and what you will look for when judging each agent.",
            shared_ctx,
        )
        return {}

    return judge_brief


def make_judge_review(agent: Agent, registry: AgentRegistry, run_id: str):
    async def judge_review(state: SharedState) -> dict[str, Any]:
        _section("Judge  ·  Review", "bright_blue")
        shared_ctx = state.get("shared_context", {})

        # ── Sum per-call elapsed times recorded in each action ────────────────
        def _elapsed(a: Agent) -> str:
            if not a._actions:
                return "-"
            total = sum(act.get("elapsed_s", 0.0) for act in a._actions)
            return f"{total:.1f}s"

        # ── Rich run summary ──────────────────────────────────────────────────
        agent_rows = "\n".join(
            f"{a.name},{a.cfg.get('role', a.name)},{len(a._actions)},{a.total_tokens},{_elapsed(a)}"
            for a in registry.all()
        )
        task_results = state.get("task_results", [])
        task_rows = (
            "\n".join(
                f"  {r['task_id']}: {r['description'][:70]} | {len(r['result'])} chars | model: {r['model']}"
                for r in task_results
            )
            or "  none"
        )
        total_tokens = sum(a.total_tokens for a in registry.all())

        num_agents = len(list(registry.all()))
        run_summary = (
            f"agents ({num_agents}):\n{agent_rows}\n\n"
            f"tasks_completed: {len(task_results)}\n"
            f"task_detail:\n{task_rows}\n\n"
            f"research_for_analysis_ran: {bool(state.get('analysis_research_summary'))}\n"
            f"execution_research_ran: {bool(state.get('research_summary'))}\n"
            f"final_output_chars: {len(state.get('final_response', ''))}\n"
            f"total_tokens: {total_tokens}"
        )

        agent_count = len(list(registry.all()))
        raw = await agent.think(
            f'Judge this pipeline run for: "{state["user_query"]}"\n\n'
            f"{run_summary}\n\n"
            f"Output a JSON object with two keys:\n"
            f"- assessments: array of {agent_count} objects, one per agent, each with: "
            f"agent (str), status (successful|unsuccessful|not_required), quality (0-3), tokens (int), notes (str)\n"
            f"- metrics: object with: overall_status (successful|partial|failed), total_tokens (int), "
            f"pipeline_efficiency (high|medium|low), cost_rating (0-3), recommendations (str)",
            shared_ctx,
        )

        assessments: list[dict] = []
        metrics: dict = {}
        try:
            parsed = _parse_json(raw)
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
            _log(agent.name, "warn", f"[yellow]JSON parse failed: {e}[/yellow]")

        # ── Print assessment table ────────────────────────────────────────────
        t = Table(title="Agent Assessments", border_style="dim")
        t.add_column("Agent", style="bold")
        t.add_column("Status")
        t.add_column("Quality", justify="center")
        t.add_column("Tokens", justify="right")
        t.add_column("Notes")
        status_colour = {"successful": "green", "unsuccessful": "red", "not_required": "dim"}
        for a in assessments:
            c = status_colour.get(a["status"], "white")
            q_display = "—" if a["status"] == "not_required" else a["quality"]
            t.add_row(
                a["agent"],
                f"[{c}]{a['status']}[/{c}]",
                f"[dim]{q_display}[/dim]",
                f"[dim]{a['tokens']}[/dim]",
                f"[dim]{a['notes']}[/dim]",
            )
        console.print(t)

        # ── Print metrics summary ─────────────────────────────────────────────
        if metrics:
            m = Table(title="Pipeline Metrics", border_style="dim", show_header=False, box=None)
            m.add_column(style="bold")
            m.add_column()
            colour_map = {
                "successful": "green",
                "partial": "yellow",
                "failed": "red",
                "high": "green",
                "medium": "yellow",
                "low": "red",
            }
            for k, v in metrics.items():
                col = colour_map.get(str(v).lower(), "cyan")
                m.add_row(k, f"[{col}]{v}[/{col}]")
            console.print(m)

        # ── Build hierarchical text report ───────────────────────────────────
        _pricing = {m["id"]: m for m in load_models()}

        def _fmt_model(m: str) -> str:
            m = m.removeprefix("openai/")
            return m[-36:] if len(m) > 36 else m

        def _status_sym(s: str) -> str:
            return {"successful": "✓", "unsuccessful": "✗", "not_required": "—"}.get(s, "—")

        def _agent_cost(ag: Agent) -> str:
            """Return formatted dollar cost string, or '—' if pricing unknown."""
            model_id = ag.model.removeprefix("openai/")
            p = _pricing.get(model_id, {})
            inp = p.get("input_cost_per_1m")
            out = p.get("output_cost_per_1m")
            if inp is None or out is None:
                return "—"
            cost = (ag.prompt_tokens * inp + ag.completion_tokens * out) / 1_000_000
            return f"${cost:.4f}"

        W = 82
        SEP = "═" * W
        LINE = "─" * W
        assess_map = {a["agent"]: a for a in assessments}
        ag_map = {a.name: a for a in registry.all()}
        task_results = state.get("task_results", [])
        total_tokens = sum(a.total_tokens for a in registry.all())

        def _agent_row(name: str, indent: str = "") -> list[str]:
            ag = ag_map.get(name)
            asmnt = assess_map.get(name, {})
            status = asmnt.get("status", "—")
            quality = "—" if status == "not_required" else asmnt.get("quality", "-")
            tokens = ag.total_tokens if ag else 0
            elapsed = _elapsed(ag) if ag else "—"
            model = _fmt_model(ag.model) if ag else "—"
            cost = _agent_cost(ag) if ag else "—"
            sym = _status_sym(status)
            label = f"{indent}{name}"
            return [
                f"  {label:<28} {model:<36} {tokens:>5} tok  {elapsed:>6}  {cost:>9}  {sym:<12} {quality}"
            ]

        report_lines: list[str] = [
            SEP,
            f"  RUN    {run_id}",
            f"  QUERY  {state.get('user_query', '')}",
            SEP,
            "",
            "  EXECUTION PIPELINE",
            LINE,
            f"  {'agent':<28} {'model':<36} {'tokens':>9}  {'time':>6}  {'cost':>9}  {'status':<12} quality",
            LINE,
        ]

        # Fixed pipeline order; workers nested under execution_lead
        named_order = ["master", "judge", "analyser", "researcher", "execution_lead"]
        for name in named_order:
            if name not in ag_map:
                asmnt = assess_map.get(name, {})
                status = asmnt.get("status", "not_required")
                report_lines.append(
                    f"  {name:<28} {'—':<36} {'—':>9}  {'—':>6}  {'—':>9}  {_status_sym(status):<12} —"
                )
            else:
                report_lines.extend(_agent_row(name))
            # Workers appear after execution_lead
            if name == "execution_lead":
                for r in task_results:
                    wname = r["agent"]
                    row = _agent_row(wname, indent="  └─ ")
                    report_lines.extend(row)
                    report_lines.append(f"     {'':28} task: {r['description'][:60]}")

        # Total cost across all agents with known pricing
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
            LINE,
            f"  {'TOTAL':<28} {'':36} {total_tokens:>5} tok  {'':>6}  {total_cost_str:>9}",
            "",
            "  PIPELINE METRICS",
            LINE,
        ]
        for k, v in metrics.items():
            report_lines.append(f"  {k:<28} {v}")
        report_lines += ["", SEP]

        report = "\n".join(report_lines)

        # ── Write judge dump (plain text report) ─────────────────────────────
        judge_dir = DUMPS_DIR / "judge"
        judge_dir.mkdir(parents=True, exist_ok=True)
        dump_path = judge_dir / f"{run_id}.txt"
        dump_path.write_text(report, encoding="utf-8")
        _log(agent.name, "dump", f"[cyan]{dump_path}[/cyan]")

        return {"agent_assessments": assessments, "judge_metrics": metrics}

    return judge_review


def make_master_respond(agent: Agent):
    async def master_respond(state: SharedState) -> dict[str, Any]:
        _section("Master Agent  ·  Respond to Human", "blue")
        shared_ctx = state.get("shared_context", {})
        history: list[dict] = list(state.get("conversation_history") or [])

        direct = state.get("direct_answer", "")
        if direct:
            # Simple query answered by analyser — master just delivers, no LLM call needed
            _log(agent.name, "route", "[dim]direct answer — no consolidation needed[/dim]")
            history.append({"role": "assistant", "content": direct})
            return {"final_response": direct, "conversation_history": history}

        final = await agent.think(
            f'Request: "{state["user_query"]}"\n\n'
            f"Team output:\n\n{state.get('final_response', '')}\n\n"
            "Deliver this to the human. No mention of agents, teams, or internal process.",
            shared_ctx,
        )
        history.append({"role": "assistant", "content": final})
        return {"final_response": final, "conversation_history": history}

    return master_respond


# =============================================================================
# Routing
# =============================================================================


def route_after_analysis(state: SharedState) -> str:
    if state.get("direct_answer"):
        return "master_respond"
    if state.get("research_for_analysis"):
        return "researcher"
    if state.get("research_needed"):
        return "researcher"
    return "execution_lead"


def route_after_researcher(state: SharedState) -> str:
    # If researcher ran for analysis, feed result back into analyser
    return "reanalyse" if state.get("research_for_analysis") else "execution_lead"


def route_after_reanalyse(state: SharedState) -> str:
    return "researcher" if state.get("research_needed") else "execution_lead"


# =============================================================================
# Dump
# =============================================================================


def write_dump(
    run_id: str,
    cfg: dict,
    result: SharedState,
    registry: AgentRegistry,
    elapsed: float,
    request: str = "",
) -> Path:
    runs_dir = DUMPS_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    dump_path = runs_dir / f"{run_id}.json"

    payload = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "config": {
            "model": cfg["model"],
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
    return dump_path


# =============================================================================
# Graph builder
# =============================================================================


def print_graph_structure() -> None:
    _section("Graph Structure", "blue")
    tree = Tree("[bold]Multi-Agent StateGraph[/bold]")
    m = tree.add("[blue]master_receive[/blue]  [dim]← human input[/dim]")
    a = m.add("[cyan]analyse[/cyan]  [dim]→ tasks[], research_needed[/dim]")
    r = a.add("[yellow]researcher[/yellow]  [dim](if research_needed=true)[/dim]")
    el = a.add("[magenta]execution_lead[/magenta]  [dim](always)[/dim]")
    r.add("[magenta]execution_lead[/magenta]")
    el.add("[green]Worker-task_1 … Worker-task_N[/green]  [dim](dynamic)[/dim]")
    el.add("[blue]master_respond[/blue]  [dim]→ human output[/dim]")
    console.print(tree)


def print_agent_roster(
    master_model: str,
    judge_model: str,
    analyser_model: str,
    researcher_model: str,
    lead_model: str,
) -> None:
    t = Table(title="Agent Roster", border_style="dim")
    t.add_column("Agent", style="bold")
    t.add_column("Role")
    t.add_column("Model")
    t.add_column("Human-facing", justify="center")
    for name, role, model, human in [
        ("master", "Orchestrator + human I/O", master_model, "[green]yes[/green]"),
        ("judge", "Pipeline quality assessor", judge_model, "no"),
        ("analyser", "Query → task breakdown", analyser_model, "no"),
        ("researcher", "Optional research (conditional)", researcher_model, "no"),
        ("execution_lead", "Spawn workers + consolidate", lead_model, "no"),
        ("worker (×N)", "One per task, created at runtime", lead_model, "no"),
    ]:
        t.add_row(name, role, f"[dim]{model}[/dim]", human)
    console.print(t)


def _model_for(agent_cfg: dict, models: list[dict], mode: str, fallback: str) -> str:
    """Select model for a named agent using its ability profile + mode bias."""
    abilities = agent_cfg.get("abilities")
    if not abilities:
        return fallback
    chosen, _ = select_model(abilities, models, mode)
    return chosen


def build_graph(cfg: dict, registry: AgentRegistry, run_id: str, checkpointer=None):
    fallback_model = cfg["model"]
    mode = cfg.get("mode", "balanced")
    agent_cfgs = {
        n: load_agent_cfg(n)
        for n in ["master", "judge", "analyser", "researcher", "execution_lead", "worker"]
    }
    models = load_models()
    print_graph_structure()

    # Master always uses the best available model — mode has no effect
    master_model = best_model(models)
    _log(
        "build", "master model", f"[bold]{master_model}[/bold]  [dim](best, mode-independent)[/dim]"
    )

    # All other agents: ability profile × mode bias
    judge_model = _model_for(agent_cfgs["judge"], models, mode, fallback_model)
    analyser_model = _model_for(agent_cfgs["analyser"], models, mode, fallback_model)
    researcher_model = _model_for(agent_cfgs["researcher"], models, mode, fallback_model)
    lead_model = _model_for(agent_cfgs["execution_lead"], models, mode, fallback_model)

    sel_table = Table(title=f"Agent Model Selection  [mode: {mode}]", border_style="dim")
    sel_table.add_column("Agent", style="bold")
    sel_table.add_column("Model")
    for name, model in [
        ("master", master_model),
        ("judge", judge_model),
        ("analyser", analyser_model),
        ("researcher", researcher_model),
        ("execution_lead", lead_model),
    ]:
        sel_table.add_row(name, f"[dim]{model}[/dim]")
    console.print(sel_table)

    print_agent_roster(master_model, judge_model, analyser_model, researcher_model, lead_model)

    master = registry.register(
        Agent("master", {**agent_cfgs["master"], "model": master_model}, master_model)
    )
    judge = registry.register(
        Agent("judge", {**agent_cfgs["judge"], "model": judge_model}, judge_model)
    )
    analyser = registry.register(
        Agent("analyser", {**agent_cfgs["analyser"], "model": analyser_model}, analyser_model)
    )
    researcher_ag = registry.register(
        Agent(
            "researcher", {**agent_cfgs["researcher"], "model": researcher_model}, researcher_model
        )
    )
    lead = registry.register(
        Agent("execution_lead", {**agent_cfgs["execution_lead"], "model": lead_model}, lead_model)
    )

    g = StateGraph(SharedState)
    g.add_node("master_receive", make_master_receive(master))
    g.add_node("judge_brief", make_judge_brief(judge, registry))
    g.add_node("analyse", make_analyse(analyser, fallback_model))
    g.add_node("researcher", make_researcher(researcher_ag))
    g.add_node("reanalyse", make_reanalyse(analyser))
    g.add_node(
        "execution_lead",
        make_execution_lead(lead, agent_cfgs["worker"], models, fallback_model, mode, registry),
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

    _log("graph", "compile", "[green]compiled and ready[/green]")
    return g.compile(checkpointer=checkpointer)


# =============================================================================
# Main
# =============================================================================


async def _async_input(prompt: str) -> str:
    """Read a line from stdin without blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)


QUIT_WORDS = {"quit", "exit", "done", "bye", "q"}


async def main() -> None:
    import time

    cfg = load_graph_config()
    registry = AgentRegistry()

    mode = cfg.get("mode", "balanced")
    mode_colour = {"best": "green", "balanced": "yellow", "cost_efficient": "cyan"}.get(
        mode, "white"
    )
    _mode_alpha = MODE_PROFILES.get(mode, MODE_PROFILES["balanced"])["alpha"]
    console.print(
        Panel(
            "[bold white]Multi-Agent LangGraph  ·  Chat Mode[/bold white]\n"
            "Master  →  Analyser  →  [Researcher?]  →  Execution Lead  →  Workers\n"
            f"Mode: [{mode_colour}]{mode}[/{mode_colour}]  "
            f"[dim]alpha={_mode_alpha}   Type  quit / exit / done  to end.[/dim]",
            border_style="blue",
            expand=False,
        )
    )
    console.print()

    # ── Read request from terminal ────────────────────────────────────────────
    request = (await _async_input("What would you like to work on?\n[You] > ")).strip()
    console.print()
    if not request or request.lower() in QUIT_WORDS:
        console.print("[dim]Exiting.[/dim]")
        return

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    checkpointer = MemorySaver()
    app = build_graph(cfg, registry, run_id, checkpointer)
    thread_cfg: dict = {"configurable": {"thread_id": run_id}}

    console.print(
        Panel(f"[bold]Human Request:[/bold]\n\n{request}", border_style="yellow", expand=False)
    )
    console.print()

    # ── Run the pipeline, pausing whenever a worker asks a question ──────────
    t0 = time.perf_counter()
    current_input: Any = {
        "user_query": request,
        "shared_context": {},
        "tasks": [],
        "task_results": [],
        "research_needed": False,
        "research_query": "",
        "research_summary": "",
        "final_response": "",
        "conversation_history": [],
    }
    result: SharedState = {}

    while True:
        result = await app.ainvoke(current_input, config=thread_cfg)
        snap = app.get_state(thread_cfg)
        if not snap.next:
            break  # pipeline complete

        # A worker raised a question — surface it to the user
        interrupt_msg = ""
        for task in snap.tasks:
            for intr in task.interrupts:
                interrupt_msg = str(intr.value)
                break
            if interrupt_msg:
                break

        _section("Worker Question", "yellow")
        console.print(Panel(interrupt_msg, border_style="yellow", expand=False))
        console.print()
        answer = (await _async_input("[You] > ")).strip()
        console.print()
        if not answer or answer.lower() in QUIT_WORDS:
            console.print("[dim]Exiting.[/dim]")
            elapsed = time.perf_counter() - t0
            partial = app.get_state(thread_cfg).values or {}
            dump_path = write_dump(run_id, cfg, partial, registry, elapsed, request)
            _log("dump", "saved (partial)", f"[cyan]{dump_path}[/cyan]")
            return
        current_input = Command(resume=answer)
    elapsed = time.perf_counter() - t0

    # ── Chat loop — talk directly to master agent ────────────────────────────
    # The graph is done. Master already has full private context (_messages).
    # We continue the conversation by calling master.think() directly.
    master = registry.get("master")
    shared_ctx = result.get("shared_context", {})
    history: list[dict] = list(result.get("conversation_history", []))

    _section("Master Agent  ·  Response", "green")
    final_response = result.get("final_response", "")
    if final_response and final_response.strip():
        console.print(Panel(final_response, border_style="green", expand=False))
    else:
        console.print("[yellow]⚠ No final response generated.[/yellow]")
    console.print()

    while master is not None:
        user_input = (await _async_input("[You] > ")).strip()
        console.print()

        if not user_input or user_input.lower() in QUIT_WORDS:
            break

        history.append({"role": "user", "content": user_input})
        reply = await master.think(
            f'The human replied: "{user_input}"\n\nRespond helpfully.',
            shared_ctx,
        )
        history.append({"role": "assistant", "content": reply})

        _section("Master Agent", "green")
        console.print(Panel(reply, border_style="green", expand=False))
        console.print()

    result["conversation_history"] = history

    # ── Summary ──────────────────────────────────────────────────────────────
    _section("Session Complete", "blue")
    t = Table(title="Run Summary", show_header=False, box=None, padding=(0, 2))
    t.add_column(style="bold")
    t.add_column()
    t.add_row("model", f"[cyan]{cfg['model']}[/cyan]")
    t.add_row("agents", f"[cyan]{len(registry.all())}[/cyan]")
    t.add_row("tasks executed", f"[cyan]{len(result.get('task_results', []))}[/cyan]")
    t.add_row("researched", f"[cyan]{result.get('research_needed', False)}[/cyan]")
    t.add_row("turns", f"[cyan]{len(history) // 2}[/cyan]")
    t.add_row("total tokens", f"[cyan]{sum(a.total_tokens for a in registry.all()):,}[/cyan]")
    t.add_row("wall time", f"[cyan]{elapsed:.2f}s[/cyan]")
    console.print(t)
    console.print()

    dump_path = write_dump(run_id, cfg, result, registry, elapsed, request)
    _log("dump", "saved", f"[cyan]{dump_path}[/cyan]")


if __name__ == "__main__":
    asyncio.run(main())
