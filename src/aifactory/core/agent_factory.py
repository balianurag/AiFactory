"""
Agent Factory — Build agents from YAML config + ability-based model selection.

This module provides the bridge between:
  - YAML agent configs (config/agents/*.yaml)
  - The ability-based model selector (core/model_selector.py)
  - The existing BaseAgent hierarchy (core/base_agent.py)

Key concepts:
  - Each agent role is defined in a YAML file with system_prompt, abilities, max_tokens
  - Models are selected per-agent using dot-product scoring against ability profiles
  - The SHARE: protocol lets agents broadcast facts into a shared_context dict
  - An AgentRegistry tracks all agent instances in a pipeline run

Ported from agent-arch/main.py, integrated with production BaseAgent.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from aifactory.core.model_selector import (
    best_model,
    load_agent_cfg,
    load_graph_config,
    load_models,
    model_for_agent,
)

logger = logging.getLogger(__name__)

# ── FINDINGS: protocol ────────────────────────────────────────────────
# Workers append a FINDINGS block with discrete key-value facts for
# subsequent workers.  extract_findings() strips it before extract_share().

_FINDINGS_MARKER = "\nFINDINGS:"


def extract_findings(response: str) -> tuple[str, dict[str, str]]:
    """
    Split a worker response into (result_without_findings, findings_dict).

    Workers append a FINDINGS block after their prose:
        [prose result]
        FINDINGS:
        language: python3
        function_name: is_palindrome

    Returns the cleaned result (FINDINGS block removed) and a dict of findings.
    Must be called before extract_share so neither function sees the other's marker.
    """
    if _FINDINGS_MARKER not in response:
        return response, {}

    result_part, findings_part = response.split(_FINDINGS_MARKER, 1)
    # findings may be followed by a SHARE block — stop there
    findings_raw = findings_part.split("\nSHARE:", 1)[0]
    findings: dict[str, str] = {}
    for line in findings_raw.strip().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip()
            if k:
                findings[k] = v
    return result_part, findings


# ── SHARE: protocol ───────────────────────────────────────────────────

_SHARE_MARKER = "\nSHARE:"


def extract_share(response: str) -> tuple[str, dict[str, str]]:
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
    updates: dict[str, str] = {}
    for line in share_part.strip().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip()
            if k:
                updates[k] = v
    return result_part.strip(), updates


def merge_share(
    shared_ctx: dict[str, str],
    updates: dict[str, str],
    agent_name: str = "",
) -> dict[str, str]:
    """Merge SHARE: updates into shared context."""
    if not updates:
        return shared_ctx
    if agent_name:
        logger.debug("[%s] → shared: %s", agent_name, list(updates.keys()))
    return {**shared_ctx, **updates}


def build_shared_context_block(shared_ctx: dict[str, str]) -> str:
    """
    Format shared_context as a text block for injection into system prompts.

    Returns empty string if no shared context.
    """
    if not shared_ctx:
        return ""
    lines = [f"{k}: {v}" for k, v in shared_ctx.items() if v]
    if not lines:
        return ""
    return "\n\n## Shared Context\n" + "\n".join(lines)


# ── JSON parsing (robust) ────────────────────────────────────────────


def parse_json_robust(raw: str) -> dict[str, Any]:
    """
    Robustly extract a JSON object from a model response.

    Handles: SHARE blocks appended after JSON, code fences, leading/trailing prose.
    Raises ValueError if no valid JSON object is found.
    """
    import json

    # Strip SHARE block first
    text = raw.split(_SHARE_MARKER, 1)[0] if _SHARE_MARKER in raw else raw
    # Strip code fences
    text = re.sub(r"```[a-z]*\n?", "", text).replace("```", "").strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back: find the outermost { ... } block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"no valid JSON object found in: {text[:300]!r}")


# ── Agent Registry ───────────────────────────────────────────────────


class AgentRegistry:
    """
    Collects all agent instances in a pipeline run.

    Used by the Judge for post-run evaluation, and for run dumps.
    """

    def __init__(self) -> None:
        self._agents: list[Any] = []

    def register(self, agent: Any) -> Any:
        """Register an agent and return it (for chaining)."""
        self._agents.append(agent)
        return agent

    def get(self, name: str) -> Any | None:
        """Find an agent by name."""
        for a in self._agents:
            n = getattr(a, "name", None) or (getattr(a, "config", None) and a.config.name)
            if n == name:
                return a
        return None

    def all(self) -> list[Any]:
        """Return all registered agents."""
        return list(self._agents)

    def count(self) -> int:
        return len(self._agents)


# ── Agent Factory ────────────────────────────────────────────────────


def create_pipeline_agents(
    mode: str | None = None,
    registry: AgentRegistry | None = None,
) -> dict[str, Any]:
    """
    Create all named pipeline agents from YAML configs with ability-based
    model selection.

    Returns a dict with agent instances keyed by role name:
      {"master": Agent, "judge": Agent, "analyser": Agent,
       "researcher": Agent, "execution_lead": Agent}

    Workers are NOT created here — they are spawned dynamically by the
    execution lead at runtime based on the analyser's task breakdown.

    Args:
        mode: "best", "balanced", or "cost_efficient". Auto-loaded from
              graph.yaml if None.
        registry: Optional AgentRegistry to register agents into.

    Returns:
        Dict of role_name → agent config dict (with "model" key set).
    """

    graph_cfg = load_graph_config()
    if mode is None:
        mode = graph_cfg.get("mode", "balanced")

    fallback_model = graph_cfg.get("model", "gpt-4o-mini")
    models = load_models()

    # Master always uses best_model (mode-independent)
    master_model = best_model(models)

    # Named agents: ability profile x mode bias
    agent_names = ["master", "judge", "analyser", "researcher", "execution_lead"]
    agent_cfgs: dict[str, dict[str, Any]] = {}

    for name in agent_names:
        cfg = load_agent_cfg(name)
        if name == "master":
            cfg["model"] = master_model
        else:
            cfg["model"] = model_for_agent(cfg, models, mode, fallback_model)
        cfg["_name"] = name
        agent_cfgs[name] = cfg

    logger.info(
        "Pipeline agents created [mode=%s]: %s",
        mode,
        {n: c["model"] for n, c in agent_cfgs.items()},
    )

    return agent_cfgs


def create_worker_config(
    task_id: str,
    task_description: str,
    ability_requirements: dict[str, float] | None = None,
    worker_prompt: str | None = None,
    mode: str = "balanced",
) -> dict[str, Any]:
    """
    Create a single worker config with per-task model selection.

    The execution lead calls this for each task, providing per-task
    ability requirements from its planning LLM call.

    Args:
        task_id: e.g. "task_1"
        task_description: What the worker must accomplish.
        ability_requirements: Per-ability scores (0-3) for this task.
        worker_prompt: Custom prompt from execution lead's plan.
        mode: Quality/cost mode.

    Returns:
        Worker config dict with selected model and prompt.
    """
    from aifactory.core.model_selector import select_model

    models = load_models()
    worker_cfg = load_agent_cfg("worker")

    # Use task-specific ability requirements, fall back to worker defaults
    reqs = ability_requirements or worker_cfg.get("abilities", {})
    chosen_model, score = select_model(reqs, models, mode)

    # Build prompt: execution lead's custom prompt or fallback
    if not worker_prompt:
        worker_prompt = f"Task: {task_description}\nDeliver a complete, precise result."

    # ── Integration tools block (only if integrations are connected) ──
    integration_block = ""
    try:
        from aifactory.integrations.registry import IntegrationRegistry

        reg = IntegrationRegistry()
        connected_toolkits = reg.get_connected_toolkit_names()
        if connected_toolkits:
            integration_block = (
                "\n\nINTEGRATION TOOLS: You have access to external service integrations. "
                "To invoke a tool, output an ACTION block:\n"
                "ACTION:composio\n"
                '{"tool": "TOOL_SLUG", "params": {"key": "value"}}\n'
                "ACTION_END\n"
                "The system will execute it and append the result. "
                f"Available toolkits: {', '.join(connected_toolkits)}. "
                "Common tools: GITHUB_CREATE_ISSUE, GITHUB_LIST_ISSUES, "
                "GITLAB_LIST_PROJECTS, GITLAB_CREATE_ISSUE, "
                "JIRA_CREATE_ISSUE, NOTION_GET_PAGE, LINEAR_LIST_TEAMS. "
                "Only use ACTION blocks when the task requires interacting with "
                "external services. Do NOT use ACTION for coding tasks."
            )
    except Exception:
        pass

    # Append QUESTION and NOT_NEEDED instructions (always present)
    full_prompt = (
        worker_prompt.rstrip()
        + integration_block
        + "\n\nCODING EXCELLENCE: Use the most optimal data structure and algorithm. "
        "Include type hints and a brief docstring with time/space complexity as a "
        "one-line comment (e.g. # O(n) time, O(k) space). Prefer standard library "
        "tools (Counter, heapq, bisect, itertools) when optimal. Be concise."
        "\n\nCODE EXECUTION: If the task involves writing code, algorithms, "
        "data analysis, or anything that benefits from being executed, wrap "
        "executable code in a fenced code block with the language tag:\n"
        "```python\n# your code here\n```\n"
        "The system will execute it in a sandbox and append the output. "
        "Supported: python, javascript, bash.\n"
        "CRITICAL: Each execution runs in a FRESH ISOLATED sandbox with NO prior state. "
        "Write fully self-contained code — include ALL imports, function definitions, "
        "and data. If prior tasks produced code, re-define everything you need."
        "\n\nIMPORTANT: If you are missing any information needed to complete this task, "
        "you MUST stop and output ONLY this single line — nothing before it, nothing after it:\n"
        "QUESTION: <your question here>\n"
        "Do not attempt to answer, proceed, or guess. Ask first.\n\n"
        "If this task is not applicable given the results of prior tasks, output exactly:\n"
        "NOT_NEEDED\n\n"
        "FINDINGS RULE: After completing your task (when NOT asking a QUESTION), if your output "
        "contains facts that subsequent tasks may need (e.g. a language choice, a function name, "
        "a boolean outcome, a key decision), append a FINDINGS block using REAL descriptive key "
        "names, one per line, like this example:\n"
        "FINDINGS:\n"
        "language: python3\n"
        "function_name: is_palindrome\n"
        "Do NOT use 'key' or 'value' as key names — use meaningful names specific to your output. "
        "Minimal — discrete facts only, no prose. Omit if nothing structured to share."
    )

    return {
        **worker_cfg,
        "model": chosen_model,
        "model_score": score,
        "system_prompt": full_prompt,
        "_name": f"Worker-{task_id}",
        "_task_id": task_id,
        "_task_description": task_description,
    }
