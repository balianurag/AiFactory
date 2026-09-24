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
from functools import lru_cache
from pathlib import Path
from typing import Any

from aifactory.core.model_selector import (
    best_model,
    load_agent_cfg,
    load_graph_config,
    load_models,
    model_for_agent,
)

logger = logging.getLogger(__name__)


# ── Package registry (sandbox environment description) ────────────────


@lru_cache(maxsize=1)
def _load_package_registry_snippet() -> str:
    """
    Load the prompt_snippet from config/package_registry.yaml.

    Returns empty string on any error so callers never break.
    """
    try:
        import yaml  # type: ignore[import]

        registry_path = Path(__file__).resolve().parents[2] / "config" / "package_registry.yaml"
        if not registry_path.exists():
            return ""
        with registry_path.open() as fh:
            data = yaml.safe_load(fh)
        return str(data.get("prompt_snippet", "")).strip()
    except Exception as exc:  # pragma: no cover
        logger.debug("package_registry.yaml load failed: %s", exc)
        return ""


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


def _match_context_key(key: str, pattern: str) -> bool:
    """
    Return True if `key` matches `pattern`.

    Patterns are either exact strings (``"objective"``) or a wildcard suffix
    with a trailing ``*`` (``"findings_*"`` matches ``findings_wave_0``,
    ``findings_wave_1``, …).  No other glob metacharacters are supported —
    this is intentional to keep behaviour deterministic and cheap.
    """
    if pattern.endswith("*"):
        return key.startswith(pattern[:-1])
    return key == pattern


def filter_shared_context(
    shared_ctx: dict[str, str],
    allowed_patterns: list[str] | None,
) -> dict[str, str]:
    """
    Return a filtered view of ``shared_ctx`` containing only the keys that
    match at least one pattern in ``allowed_patterns``.

    - ``allowed_patterns is None`` — return the input unchanged (default
      behaviour, i.e. every agent sees the whole shared context).
    - Empty list — returns an empty dict (agent opts out of shared context).
    - Patterns absent from ``shared_ctx`` are silently ignored — no error.

    Iteration order matches the input dict so the rendered prompt block
    remains stable / cache-friendly.
    """
    if allowed_patterns is None:
        return shared_ctx
    if not shared_ctx:
        return {}
    return {
        k: v
        for k, v in shared_ctx.items()
        if any(_match_context_key(k, p) for p in allowed_patterns)
    }


def build_shared_context_block(
    shared_ctx: dict[str, str],
    allowed_patterns: list[str] | None = None,
) -> str:
    """
    Format shared_context as a text block for injection into system prompts.

    When ``allowed_patterns`` is provided, keys are filtered per
    :func:`filter_shared_context` before rendering — this is how per-role
    context filtering shrinks the prompt (and therefore the Anthropic prompt
    cache prefix) for agents that only need a subset of shared state.

    Returns empty string if no shared context remains after filtering.
    """
    if allowed_patterns is not None:
        shared_ctx = filter_shared_context(shared_ctx, allowed_patterns)
    if not shared_ctx:
        return ""
    lines = [f"{k}: {v}" for k, v in shared_ctx.items() if v]
    if not lines:
        return ""
    return "\n\n## Shared Context\n" + "\n".join(lines)


# ── FINDINGS: protocol ───────────────────────────────────────────────
# Workers append compact FINDINGS: blocks to pass structured facts to
# downstream workers at 80-90% fewer tokens than full prose repetition.

_FINDINGS_MARKER = "\nFINDINGS:"

# Models (esp. gemini-2.0-flash) frequently render the FINDINGS: protocol marker
# as a markdown heading ("### Findings"), bold ("**FINDINGS:**"), or lowercase.
# This regex matches the marker at the start of a line in any of those variants
# so the block is never silently dropped as prose.
_FINDINGS_LINE_RE = re.compile(
    r"^[ \t>]*(?:#{1,6}\s*|[*_]{1,2}\s*)?findings\s*:?\s*[*_]{0,2}\s*$",
    re.IGNORECASE,
)
# Downstream protocol markers that terminate the findings block, same variants.
_OTHER_MARKER_RE = re.compile(
    r"^[ \t>]*(?:#{1,6}\s*|[*_]{1,2}\s*)?(?:remember|share)\s*:?\s*[*_]{0,2}\s*$",
    re.IGNORECASE,
)


def extract_findings(response: str) -> tuple[str, list[str]]:
    """
    Split a worker response into (result, findings_facts).

    Workers may append a FINDINGS: block with concise bullet points that
    capture key facts for downstream workers:

        [actual work result]
        FINDINGS:
        - Framework: FastAPI with async lifespan
        - DB schema: users(id, email, hash), sessions(id, user_id, token)
        - Auth: JWT, 1h expiry, refresh via /auth/refresh

    The marker is matched leniently — "FINDINGS:", "### Findings", "**FINDINGS:**"
    and lowercase variants all work, because models format it inconsistently.

    Returns the cleaned result and a list of fact strings. If no FINDINGS
    block is present, facts is [].
    """
    lines = response.splitlines()
    marker_idx = next(
        (i for i, line in enumerate(lines) if _FINDINGS_LINE_RE.match(line)),
        None,
    )
    if marker_idx is None:
        return response.strip(), []

    result_part = "\n".join(lines[:marker_idx])
    facts: list[str] = []
    for line in lines[marker_idx + 1 :]:
        # Stop when another protocol marker (REMEMBER/SHARE, any format) appears
        if _OTHER_MARKER_RE.match(line):
            break
        # Strip markdown bullet/emphasis noise from the fact itself
        stripped = line.strip().lstrip("-*•").strip().strip("*_").strip()
        if stripped:
            facts.append(stripped)
    return result_part.strip(), facts


_MAX_FINDINGS_PER_TASK = 5  # facts per individual task
_MAX_FINDING_CHARS = 120  # chars per individual fact
_MAX_FINDINGS_TOTAL = 20  # hard global cap across all tasks


def format_findings_context(findings_by_task: dict[str, list[str]]) -> str:
    """
    Format cross-worker findings for injection into a subsequent worker's context.

    Applies a three-level cap to keep the injected block predictably small:
      1. Per-fact: truncate at _MAX_FINDING_CHARS (120 chars)
      2. Per-task: keep only the top _MAX_FINDINGS_PER_TASK (5) facts
      3. Global:   hard-cap the total at _MAX_FINDINGS_TOTAL (20) facts,
                   preferring the most recently completed tasks (last keys in dict)
                   since they carry the most relevant forward context.

    Returns a compact multi-line block, or empty string if nothing to inject.
    """
    if not findings_by_task:
        return ""

    # Collect (task_id, fact) pairs from newest task to oldest, then reverse
    # so the final output is chronological (earliest task first).
    pairs: list[tuple[str, str]] = []
    for task_id, facts in reversed(list(findings_by_task.items())):
        for f in facts[:_MAX_FINDINGS_PER_TASK]:
            if len(pairs) >= _MAX_FINDINGS_TOTAL:
                break
            truncated = f[:_MAX_FINDING_CHARS] + ("…" if len(f) > _MAX_FINDING_CHARS else "")
            pairs.append((task_id, truncated))
        if len(pairs) >= _MAX_FINDINGS_TOTAL:
            break
    pairs.reverse()

    lines = ["## Prior Task Findings"]
    current_task: str | None = None
    for task_id, fact in pairs:
        if task_id != current_task:
            lines.append(f"[{task_id}]")
            current_task = task_id
        lines.append(f"  - {fact}")
    return "\n".join(lines)


# ── REMEMBER: protocol ────────────────────────────────────────────────
# Workers can persist cross-run facts to global LTM so future pipeline runs
# benefit from proven solutions without re-discovering them.

_REMEMBER_MARKER = "\nREMEMBER:"


def extract_remember(response: str) -> tuple[str, list[str]]:
    """
    Split a worker response into (result, remember_facts).

    Workers may append a REMEMBER: block to store long-term facts in global
    memory, seeded into all agents on future pipeline runs:

        [actual work result]
        REMEMBER:
        - FastAPI + Redis best for async task queues in this stack
        - Users table always needs soft-delete (deleted_at column)

    Returns the cleaned result and a list of facts to persist globally.
    If no REMEMBER block is present, facts is [].
    """
    if _REMEMBER_MARKER not in response:
        return response.strip(), []

    result_part, remember_part = response.split(_REMEMBER_MARKER, 1)
    facts: list[str] = []
    for line in remember_part.strip().splitlines():
        raw_stripped = line.strip()
        # Stop when another protocol marker is encountered
        if raw_stripped in ("FINDINGS:", "SHARE:"):
            break
        stripped = raw_stripped.lstrip("-").strip()
        if stripped:
            facts.append(stripped)
    return result_part.strip(), facts


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


# ── Worker config helper functions ────────────────────────────────────────────


def _build_slug_catalog(reg: Any, slugs: list[str]) -> str:
    """Build a concise tool catalog from a pre-selected list of slug strings."""
    slug_set = set(slugs)
    lines: list[str] = []
    for integration in reg.get_connected():
        meta_store = getattr(integration, "_tool_metadata", {})
        for tk_meta in meta_store.values():
            for m in tk_meta:
                if m["slug"] in slug_set:
                    desc = m.get("description", "")
                    req = m.get("required", [])
                    req_str = f" (requires: {', '.join(req)})" if req else ""
                    lines.append(
                        f"  {m['slug']}: {desc}{req_str}" if desc else f"  {m['slug']}{req_str}"
                    )
    if not lines:
        # Metadata not fetched yet — just list the slugs
        lines = [f"  {s}" for s in slugs]
    return "\n".join(lines)


def _build_toolkit_catalog(
    reg: Any,
    relevant_toolkits: list[str],
    task_description: str,
    connected_toolkits: list[str],
) -> tuple[str, str]:
    """
    Build a keyword-filtered tool catalog for the given toolkits.

    Returns a tuple of (tools_section, action_prefix).  Uses MCPProvider
    for toolkits it covers; skips toolkits not covered by MCP.
    """
    import re as _re

    _stop_words = frozenset(
        [
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "shall",
            "should",
            "may",
            "might",
            "can",
            "could",
            "of",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "and",
            "but",
            "or",
            "not",
            "no",
            "my",
            "me",
            "i",
            "you",
            "your",
            "we",
            "our",
            "they",
            "their",
            "it",
            "this",
            "that",
            "from",
            "by",
            "as",
            "all",
            "any",
            "each",
            "how",
            "what",
            "which",
            "who",
            "whom",
            "when",
            "where",
            "why",
            "if",
            "then",
            "so",
            "than",
            "too",
            "very",
            "just",
            "about",
            "also",
            "into",
            "over",
            "under",
            "between",
            "through",
            "after",
            "before",
            "during",
            "please",
            "using",
            "use",
        ]
    )
    toolkit_names_lower = {tk.lower() for tk in connected_toolkits}
    _task_words = _re.findall(r"[a-z]{3,}", task_description.lower())
    seen: set[str] = set()
    task_keywords: list[str] = []
    for kw in _task_words:
        if kw not in seen and kw not in _stop_words and kw not in toolkit_names_lower:
            seen.add(kw)
            task_keywords.append(kw)
    task_keywords = task_keywords[:15]

    # Prefer MCP for toolkits it covers; skip any toolkit not covered by MCP.
    # Only use get_provider_by_type when the registry is a real IntegrationRegistry
    # (not a MagicMock in tests) — guard by checking the return value has the
    # correct provider_type attribute.
    _get_prov_by_type = getattr(reg, "get_provider_by_type", None)
    mcp_prov = None
    if callable(_get_prov_by_type):
        _maybe_mcp = _get_prov_by_type("mcp")
        if _maybe_mcp is not None and getattr(_maybe_mcp, "provider_type", "") == "mcp":
            mcp_prov = _maybe_mcp

    mcp_toolkits = set(mcp_prov.get_available_toolkits()) if mcp_prov else set()
    mcp_covered = [tk for tk in relevant_toolkits if tk in mcp_toolkits]

    parts: list[str] = []
    used_prefix = "mcp"

    if mcp_covered and mcp_prov:
        for tk in mcp_covered:
            tk_tools = mcp_prov.get_toolkit_tools_for_prompt(tk, task_keywords=task_keywords)
            if tk_tools:
                parts.append(f"[{tk}]\n{tk_tools}")

    # Legacy fallback: iterate connected integrations (no provider_type support)
    if not parts:
        for integration in reg.get_connected():
            if hasattr(integration, "get_toolkit_tools_for_prompt"):
                for tk in relevant_toolkits:
                    tk_tools = integration.get_toolkit_tools_for_prompt(
                        tk, task_keywords=task_keywords
                    )
                    if tk_tools:
                        parts.append(f"[{tk}]\n{tk_tools}")
                if parts:
                    break

    tools_section = (
        "\n".join(parts) if parts else f"Connected toolkits: {', '.join(relevant_toolkits)}"
    )

    logger.info(
        "[create_worker_config] toolkit catalog: relevant=%s prefix=%s keywords=%s len=%d",
        relevant_toolkits,
        used_prefix,
        task_keywords,
        len(tools_section),
    )
    return tools_section, used_prefix


def resolve_slugs_for_task(
    task_description: str,
    toolkits: list[str],
    top_k: int = 6,
) -> list[str]:
    """
    Deterministically select the top-K tool slugs for a task without LLM involvement.

    Called by the execution lead post-planning to auto-fill empty tool_slug
    assignments.  Instead of relying on the LLM to name slugs (which causes
    hallucination), we score every tool in the assigned toolkits against the
    task description keywords and return the highest-scoring matches.

    The result is passed as `assigned_tool_slugs` to `create_worker_config`,
    triggering the tight `_build_slug_catalog` path (5-6 tools) rather than
    the broad keyword-filtered catalog (up to 15 tools), dramatically reducing
    the surface area for slug hallucination.
    """
    import re as _re

    _stop = frozenset(
        [
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "shall",
            "should",
            "may",
            "might",
            "can",
            "could",
            "of",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "and",
            "but",
            "or",
            "not",
            "no",
            "my",
            "me",
            "i",
            "you",
            "your",
            "we",
            "our",
            "they",
            "their",
            "it",
            "this",
            "that",
            "from",
            "by",
            "as",
            "all",
            "any",
            "each",
            "how",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "if",
            "then",
            "so",
            "than",
            "too",
            "very",
            "just",
            "about",
            "also",
            "into",
            "over",
            "under",
            "between",
            "through",
            "after",
            "before",
            "please",
            "using",
            "use",
        ]
    )
    try:
        from aifactory.integrations.registry import IntegrationRegistry

        reg = IntegrationRegistry()
        raw_words = _re.findall(r"[a-z]{3,}", task_description.lower())
        kw = [w for w in dict.fromkeys(raw_words) if w not in _stop][:12]
        if not kw:
            return []

        scored: list[tuple[int, str]] = []
        for integration in reg.get_connected():
            meta_store = getattr(integration, "_tool_metadata", {})
            for tk in toolkits:
                for m in meta_store.get(tk, []):
                    slug_l = m["slug"].lower()
                    desc_l = m.get("description", "").lower()
                    score = sum(1 for w in kw if w in slug_l or w in desc_l)
                    if score > 0:
                        scored.append((score, m["slug"]))

        scored.sort(key=lambda x: -x[0])
        seen: set[str] = set()
        result: list[str] = []
        for _, slug in scored:
            if slug not in seen:
                seen.add(slug)
                result.append(slug)
            if len(result) >= top_k:
                break
        return result
    except Exception:
        return []


def _build_integration_block(tools_section: str, action_prefix: str = "mcp") -> str:
    """Wrap a tool catalog in the standard ACTION block instruction."""
    action_example = (
        "ACTION:mcp\n"
        '{"tool": "github::create_issue", "params": {"owner": "...", "repo": "...", "title": "Bug title"}}\n'  # noqa: E501
        "ACTION_END"
    )
    slug_rule = (
        "CRITICAL — copy the slug exactly as shown (format: server::tool_name). "
        "NEVER invent or modify slugs. If the right tool is not listed, say so."
    )

    return (
        "\n\nINTEGRATION TOOLS: You have access to external service integrations. "
        "ALL integrations are ALREADY CONNECTED and AUTHENTICATED — "
        "do NOT ask for host URLs, API tokens, credentials, or endpoint details. "
        "Just call the tool directly.\n"
        "To invoke a tool, output an ACTION block:\n"
        f"{action_example}\n\n"
        "RULES:\n"
        f"  1. {slug_rule}\n"
        "  2. Include ALL required params shown in parentheses.\n"
        "  3. Call tools immediately — do NOT ask for confirmation.\n\n"
        "ITERATIVE TOOL USE: After you emit an ACTION block, the system "
        "executes it and returns results. Up to 3 rounds.\n\n"
        f"TOOL CATALOG (exact slugs only — do NOT alter these names):\n{tools_section}"
    )


def create_worker_config(
    task_id: str,
    task_description: str,
    ability_requirements: dict[str, float] | None = None,
    worker_prompt: str | None = None,
    mode: str = "balanced",
    assigned_toolkits: list[str] | None = None,
    assigned_tool_slugs: list[str] | None = None,
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
        assigned_toolkits: Toolkit names the execution lead picked for
            this task.  When provided, only these toolkits' tools are
            injected into the worker prompt.  Empty list means "no tools".
            None means "fall back to auto-detection" (backward compat).
        assigned_tool_slugs: Exact action slugs the execution lead selected
            (e.g. ["GITHUB_CREATE_ISSUE"]).  When provided, the worker
            prompt shows ONLY these slugs — no full catalog injection.
            Takes priority over assigned_toolkits for prompt building.

    Returns:
        Worker config dict with selected model and prompt.
    """
    from aifactory.core.model_selector import select_model

    models = load_models()
    worker_cfg = load_agent_cfg("worker")

    # Use task-specific ability requirements, fall back to worker defaults
    reqs = ability_requirements or worker_cfg.get("abilities", {})

    # ── RL-lite: fetch historical performance scores for this task type ──
    rl_scores: dict[str, float] | None = None
    task_type = "general"
    try:
        from aifactory.config.settings import get_settings as _gs

        _rl_settings = _gs()
        if _rl_settings.rl_enabled:
            from aifactory.core.rl_scorer import get_model_rl_scores, infer_task_type

            task_type = infer_task_type(task_description)
            rl_scores = (
                get_model_rl_scores(task_type, min_samples=_rl_settings.rl_min_samples) or None
            )
            if rl_scores:
                logger.debug("[RL] task_type=%s rl_scores=%s", task_type, rl_scores)
    except Exception as _rl_exc:
        logger.debug("[RL] fetch failed (non-fatal): %s", _rl_exc)

    chosen_model, score = select_model(reqs, models, mode, rl_scores=rl_scores)

    # ── Worker role specialization — load specialist context by task type ──
    # Maps inferred task type to a specialist YAML config.
    # The specialist's "task" block is prepended to the prompt as role context,
    # giving the worker a domain-specific persona before the task instruction.
    _role_yaml: dict[str, str] = {
        "coding": "worker_coder",
        "math": "worker_coder",
        "research": "worker_researcher",
        "writing": "worker_writer",
        "analysis": "worker_analyst",
    }
    specialist_context = ""
    _specialist_yaml = _role_yaml.get(task_type)
    if _specialist_yaml:
        try:
            specialist_cfg = load_agent_cfg(_specialist_yaml)
            _ctx = str(specialist_cfg.get("task", "")).strip()
            if _ctx:
                specialist_context = _ctx + "\n\n"
                logger.debug("[worker] role specialization: %s (%s)", _specialist_yaml, task_type)
            # Apply specialist max_tokens when higher than base — coding tasks
            # in particular need room to write complete code without truncation.
            # Truncation leaves unclosed code fences → sandbox can't execute.
            _spec_max = specialist_cfg.get("max_tokens")
            if _spec_max and int(_spec_max) > int(worker_cfg.get("max_tokens", 0)):
                worker_cfg = dict(worker_cfg)  # shallow copy — don't mutate cached config
                worker_cfg["max_tokens"] = int(_spec_max)
                logger.debug(
                    "[worker] max_tokens override from %s: %d → %d",
                    _specialist_yaml,
                    worker_cfg.get("max_tokens", 0),
                    int(_spec_max),
                )
        except Exception as _spec_exc:
            logger.debug("[worker] specialist config load skipped: %s", _spec_exc)

    # Build prompt: execution lead's custom prompt or fallback
    if not worker_prompt:
        worker_prompt = f"Task: {task_description}\nDeliver a complete, precise result."

    # ── Integration tools block (only if integrations are connected) ──
    # Priority:
    #   1. assigned_tool_slugs (exact slugs from execution lead) — most deterministic
    #   2. assigned_toolkits (toolkit-level, keyword-filtered catalog)
    #   3. Empty list assigned → no tools (lead said skip)
    #   4. None → auto-detect (backward compat)
    integration_block = ""
    try:
        from aifactory.integrations.registry import IntegrationRegistry

        reg = IntegrationRegistry()
        connected_toolkits = reg.get_connected_toolkit_names()
        logger.debug(
            "[create_worker_config] connected_toolkits=%s for task=%s",
            connected_toolkits,
            task_description[:80],
        )
        if connected_toolkits:
            # ── Fast path: exact slugs pre-selected by execution lead ──
            # Only activate when slugs are actually provided (non-empty).
            # If the lead returned tool_slugs=[] but toolkits=['github'], we
            # fall through to toolkit-level handling so the worker still gets
            # a tool catalog. The LLM may not always emit specific slugs even
            # when it knows the toolkit is needed.
            if assigned_tool_slugs:
                # Build a minimal catalog showing only the pre-selected slugs
                tools_section = _build_slug_catalog(reg, assigned_tool_slugs)
                logger.info(
                    "[create_worker_config] task=%s → slug-level assignment: %s",
                    task_id,
                    assigned_tool_slugs,
                )
                integration_block = _build_integration_block(tools_section)

            elif assigned_toolkits is not None:
                # Toolkit-level assignment (fallback when no slugs provided)
                if not assigned_toolkits:
                    logger.info(
                        "[create_worker_config] task=%s → lead assigned NO toolkits, skipping",
                        task_id,
                    )
                else:
                    relevant_toolkits = [
                        tk
                        for tk in assigned_toolkits
                        if tk.lower() in {c.lower() for c in connected_toolkits}
                    ]
                    if not relevant_toolkits:
                        logger.warning(
                            "[create_worker_config] lead assigned %s but none connected (%s) — falling back",  # noqa: E501
                            assigned_toolkits,
                            connected_toolkits,
                        )
                        relevant_toolkits = connected_toolkits[:1]

                    tools_section, action_prefix = _build_toolkit_catalog(
                        reg, relevant_toolkits, task_description, connected_toolkits
                    )
                    other_note = ""
                    other_toolkits = [
                        tk for tk in connected_toolkits if tk not in relevant_toolkits
                    ]
                    if other_toolkits:
                        other_note = f"\nOther available toolkits: {', '.join(other_toolkits)}"
                    integration_block = _build_integration_block(
                        tools_section + other_note, action_prefix=action_prefix
                    )
            else:
                # Auto-detect from task text (backward compat)
                task_text = (task_description + " " + (worker_prompt or "")).lower()
                relevant_toolkits = [tk for tk in connected_toolkits if tk.lower() in task_text]
                if not relevant_toolkits:
                    relevant_toolkits = connected_toolkits[:2]

                tools_section, action_prefix = _build_toolkit_catalog(
                    reg, relevant_toolkits, task_description, connected_toolkits
                )
                integration_block = _build_integration_block(
                    tools_section, action_prefix=action_prefix
                )
    except Exception as exc:
        logger.warning("[create_worker_config] integration block failed: %s", exc)

    # ── Build the full worker prompt ────────────────────────────────────
    # For tool-only tasks (has integration block, no coding needed)
    # skip the coding instructions to save ~150 prompt tokens.
    is_tool_task = bool(integration_block)

    coding_block = ""
    if not is_tool_task:
        # Inject sandbox environment description when available (Docker backend)
        sandbox_env_block = ""
        pkg_snippet = _load_package_registry_snippet()
        if pkg_snippet:
            sandbox_env_block = f"\n\n{pkg_snippet}"

        coding_block = (
            "\n\nCODING EXCELLENCE: Use the most optimal data structure and algorithm. "
            "Include type hints. Be concise."
            "\n\nCODE EXECUTION: Wrap executable code in a fenced block with the "
            "language tag (python/javascript/bash). The system will execute it "
            "in a sandbox and return output. Each run is a FRESH ISOLATED sandbox "
            "— include ALL imports and definitions." + sandbox_env_block
        )

    question_block = "\n\nQUESTION PROTOCOL: Almost NEVER ask questions. "
    if is_tool_task:
        question_block += (
            "You have pre-configured integration tools — DO NOT ask questions. "
            "Never ask for host URLs, credentials, namespaces, or confirmation. "
            "Just call the tools with the information from your task description."
        )
    else:
        question_block += (
            "If the user provided a name, username, path, or identifier — USE IT. "
            "Only ask if a required parameter is truly absent AND cannot be defaulted. "
            "If you must ask, output ONLY: QUESTION: <your question>"
        )

    full_prompt = (
        specialist_context  # Role-specific persona/standards (Coder/Researcher/etc.)
        + worker_prompt.rstrip()  # Execution lead's task-specific instruction
        + integration_block
        + coding_block
        + question_block
        + "\n\nIf this task is not applicable given prior results, output exactly: NOT_NEEDED"
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
