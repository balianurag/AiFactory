#!/usr/bin/env python3
"""
Multi-Agent LangGraph Experiment

Architecture:
  Human → Leader
            ├── Judge             (quality review before + after)
            ├── Program Manager   (plan tasks with ability ratings + prompts, execute workers, consolidate)
            │         ├── Researcher   (conditional — pre-analysis or execution research)
            │         └── Worker-task_1 … Worker-task_N  (dynamic, per-task model selection)
            └── Leader            (delivers final answer; Judge can flag quality issues)

Context model:
  - Each agent has a private message history (never shared)
  - A shared_context dict is injected into every agent's system prompt
  - Leader and Judge are the only human-facing agents

Run:
    uv run python experiments/main.py
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

ROOT = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent / "config"
DUMPS_DIR    = Path(__file__).parent / "dumps"
MEMORIES_DIR = Path(__file__).parent / "memories"

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

litellm.suppress_debug_info = True
litellm.ssl_verify = False
console = Console()

# LiteLLM proxy config — populated from .env, injected into every completion call
_LITELLM_EXTRA: dict = {}
_api_base = os.getenv("LITELLM_API_BASE", "").strip()
_api_key  = os.getenv("LITELLM_API_KEY",  "").strip()
if _api_base:
    _LITELLM_EXTRA["base_url"] = _api_base
if _api_key:
    _LITELLM_EXTRA["api_key"] = _api_key


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


def _norm_query(query: str) -> str:
    """Normalise a query string to a stable cache key."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", query.lower())).strip()


def _word_overlap(a: str, b: str) -> float:
    """Word-overlap similarity between two normalised query strings (0–1)."""
    wa, wb = set(a.split()), set(b.split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


async def _embed_query(query: str) -> list[float] | None:
    """Embed a query string via LiteLLM. Returns None on failure (cache falls back to word-overlap)."""
    try:
        resp = await litellm.aembedding(
            model=_proxy_model("text-embedding-ada-002"),
            input=[query],
            **_LITELLM_EXTRA,
        )
        return resp.data[0]["embedding"]
    except Exception:
        return None


# =============================================================================
# Tool implementations  (called by ToolExecutor)
# =============================================================================


async def _tool_python_exec(code: str, timeout: int = 15) -> str:
    """Execute Python code in a subprocess; return stdout+stderr."""
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        out = stdout.decode(errors="replace")
        err = stderr.decode(errors="replace")
        parts: list[str] = []
        if out:
            parts.append(f"stdout:\n{out}")
        if err:
            parts.append(f"stderr:\n{err}")
        if proc.returncode != 0:
            parts.append(f"[exit code {proc.returncode}]")
        return "\n".join(parts).strip() or "[no output]"
    except asyncio.TimeoutError:
        return f"[timeout after {timeout}s]"
    except Exception as e:
        return f"[error: {e}]"


def _tool_read_file(path: str) -> str:
    try:
        p = Path(path)
        if not p.exists():
            return f"[file not found: {path}]"
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[error reading {path}: {e}]"


def _tool_write_file(path: str, content: str) -> str:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"[wrote {len(content)} chars to {path}]"
    except Exception as e:
        return f"[error writing {path}: {e}]"


def _tool_edit_file(path: str, old_string: str, new_string: str) -> str:
    """Precise string replacement — rejected if old_string is not unique."""
    try:
        p = Path(path)
        if not p.exists():
            return f"[file not found: {path}]"
        content = p.read_text(encoding="utf-8", errors="replace")
        count = content.count(old_string)
        if count == 0:
            return f"[edit_file: old_string not found in {path}]"
        if count > 1:
            return (
                f"[edit_file: old_string matched {count} locations in {path} — "
                "must be unique. Add more surrounding context to make it unique.]"
            )
        p.write_text(content.replace(old_string, new_string, 1), encoding="utf-8")
        return f"[edited {path} — 1 replacement made]"
    except Exception as e:
        return f"[error editing {path}: {e}]"


def _tool_glob_search(pattern: str, path: str = ".") -> str:
    try:
        matches = sorted(Path(path).glob(pattern))
        files = [str(m) for m in matches if m.is_file()]
        if not files:
            return "No files found."
        truncated = files[:100]
        suffix = f"\n[{len(files) - 100} more results omitted]" if len(files) > 100 else ""
        return "\n".join(truncated) + suffix
    except Exception as e:
        return f"[glob_search error: {e}]"


def _tool_grep_search(
    pattern: str,
    path: str = ".",
    glob: str = "**/*",
    case_insensitive: bool = False,
) -> str:
    try:
        flags = re.IGNORECASE if case_insensitive else 0
        compiled = re.compile(pattern, flags)
        results: list[str] = []
        for f in sorted(Path(path).glob(glob)):
            if not f.is_file():
                continue
            try:
                for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if compiled.search(line):
                        results.append(f"{f}:{i}: {line.rstrip()}")
                        if len(results) >= 100:
                            break
            except Exception:
                continue
            if len(results) >= 100:
                break
        if not results:
            return "No matches found."
        suffix = "\n[truncated at 100 results]" if len(results) >= 100 else ""
        return "\n".join(results) + suffix
    except Exception as e:
        return f"[grep_search error: {e}]"


async def _tool_web_search(query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo; return titled results with snippets."""
    try:
        from duckduckgo_search import DDGS  # noqa: PLC0415
    except ImportError:
        return "[web_search unavailable: run `uv add duckduckgo-search`]"
    try:
        loop = asyncio.get_event_loop()
        max_results = max(1, min(int(max_results), 10))

        def _search() -> list[dict]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        results = await loop.run_in_executor(None, _search)
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r.get('title', '')}\n    URL: {r.get('href', '')}\n    {r.get('body', '')}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"[web_search error: {e}]"


async def _tool_web_crawl(url: str, max_chars: int = 8000) -> str:
    """Fetch a URL and return clean extracted text (up to max_chars characters)."""
    try:
        import httpx  # noqa: PLC0415
        from bs4 import BeautifulSoup  # noqa: PLC0415
    except ImportError as exc:
        return f"[web_crawl unavailable: {exc} — run `uv add httpx beautifulsoup4`]"
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15, verify=False) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; researcher-bot/1.0)"})
            resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text[:max_chars] + ("…" if len(text) > max_chars else "")
    except Exception as e:
        return f"[web_crawl error: {e}]"


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
    "reasoning", "coding", "math", "writing", "instruction_following",
    "tool_use", "knowledge", "speed", "cost_efficiency", "context_handling",
    "embedding",
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
        "max_quality": 3,   # no cap — full 0-3 model scores used
        "reasoning": 3, "coding": 3, "math": 2, "writing": 3,
        "instruction_following": 3, "tool_use": 3, "knowledge": 3,
        "speed": 0, "cost_efficiency": 0, "context_handling": 3,
        "embedding": 0,
    },
    "balanced": {
        "alpha": 0.35,
        "max_quality": 2,   # cap model quality at 2 — "good enough" not "exceptional"
        **{a: 2 for a in ABILITIES if a != "embedding"}, "embedding": 0,
    },
    "cost_efficient": {
        "alpha": 0.75,
        "max_quality": 3,   # quality already near-zero from profile weights
        "reasoning": 0, "coding": 0, "math": 0, "writing": 0,
        "instruction_following": 0, "tool_use": 0, "knowledge": 0,
        "speed": 3, "cost_efficiency": 3, "context_handling": 0,
        "embedding": 0,
    },
}


def load_graph_config() -> dict:
    return yaml.safe_load((CONFIG_DIR / "graph.yaml").read_text())


def load_agent_cfg(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / "agents" / f"{name}.yaml").read_text())


def load_models() -> list[dict]:
    return yaml.safe_load((CONFIG_DIR / "models.yaml").read_text())["models"]


def load_tools_cfg() -> list[dict]:
    path = CONFIG_DIR / "tools.yaml"
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text()).get("tools", [])


def load_researcher_tools_cfg() -> list[dict]:
    path = CONFIG_DIR / "researcher_tools.yaml"
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text()).get("tools", [])


def _expand_env(obj: Any) -> Any:
    """Recursively expand ${VAR} and ${VAR:-default} patterns using os.environ."""
    if isinstance(obj, str):
        def _sub(m: re.Match) -> str:
            var, sep, default = m.group(1).partition(":-")
            val = os.environ.get(var)
            if val is not None:
                return val
            return default if sep else m.group(0)  # keep ${VAR} if unset and no default
        return re.sub(r"\$\{([^}]+)\}", _sub, obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(i) for i in obj]
    return obj


def load_mcp_servers_cfg() -> list[dict]:
    path = CONFIG_DIR / "mcp_servers.yaml"
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text()).get("servers", []) or []
    return _expand_env(raw)


# =============================================================================
# McpClient — manages lifecycle and tool dispatch for external MCP servers
# =============================================================================


class McpClient:
    """
    Connects to all servers in config/mcp_servers.yaml, discovers their tools,
    and dispatches tool calls.

    Tool names are prefixed:  mcp__{server_id}__{tool_name}
    This lets the PM treat MCP tools identically to built-in tools.
    """

    def __init__(self, servers_cfg: list[dict]) -> None:
        self._servers_cfg = servers_cfg
        self._sessions: dict[str, Any] = {}   # server_id -> ClientSession
        self._tools: list[dict] = []           # flat list; each entry is a tools_cfg-compatible dict
        self._exit_stack: Any = None

    async def connect(self) -> None:
        """Connect to all configured servers and discover tools."""
        from contextlib import AsyncExitStack

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            _log("mcp", "warn", "[yellow]mcp package not installed — run: uv sync[/yellow]")
            return

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        for srv in self._servers_cfg:
            sid = srv["id"]
            srv_type = srv.get("type", "stdio")

            # Skip if any env value is still an unexpanded ${VAR} placeholder
            unresolved = [k for k, v in (srv.get("env") or {}).items() if isinstance(v, str) and v.startswith("${")]
            if unresolved:
                _log("mcp", sid, f"[yellow]skipped — unset env vars: {unresolved}[/yellow]")
                continue

            try:
                if srv_type == "stdio":
                    params = StdioServerParameters(
                        command=srv["command"],
                        args=srv.get("args", []),
                        env=srv.get("env") or None,
                    )
                    read, write = await self._exit_stack.enter_async_context(stdio_client(params))
                elif srv_type == "sse":
                    from mcp.client.sse import sse_client
                    read, write = await self._exit_stack.enter_async_context(sse_client(srv["url"]))
                else:
                    _log("mcp", "skip", f"[dim]unknown server type '{srv_type}' — skipping {sid}[/dim]")
                    continue

                session = await self._exit_stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._sessions[sid] = session

                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    self._tools.append({
                        "id": f"mcp__{sid}__{tool.name}",
                        "name": tool.name,
                        "server_id": sid,
                        "description": tool.description or f"MCP tool from {sid}",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                    })
                _log("mcp", sid, f"[green]connected — {len(tools_result.tools)} tools[/green]")

            except Exception as exc:
                _log("mcp", sid, f"[red]connection failed: {exc}[/red]")

    async def close(self) -> None:
        if self._exit_stack:
            await self._exit_stack.__aexit__(None, None, None)

    def tools(self) -> list[dict]:
        """All discovered MCP tool dicts (prefixed IDs, tools_cfg-compatible)."""
        return list(self._tools)

    async def call(self, prefixed_name: str, arguments: dict) -> str:
        """Dispatch a call to an MCP tool by its prefixed name."""
        parts = prefixed_name.split("__", 2)
        if len(parts) != 3:
            return f"[mcp: invalid tool name format: {prefixed_name}]"
        _, sid, tool_name = parts
        session = self._sessions.get(sid)
        if not session:
            return f"[mcp: no active session for server '{sid}']"
        try:
            result = await session.call_tool(tool_name, arguments)
            texts = [block.text for block in result.content if hasattr(block, "text")]
            return "\n".join(texts) if texts else "[mcp: empty result]"
        except Exception as exc:
            return f"[mcp: call failed: {exc}]"


# =============================================================================
# ToolExecutor — resolves schemas and dispatches tool calls for a worker
# =============================================================================


class ToolExecutor:
    """
    Given a list of tool IDs and the full tools config, builds LiteLLM-compatible
    function schemas and dispatches execution.

    Workers receive a ToolExecutor pre-loaded with their assigned tools only.
    Agent.think() passes it to litellm.acompletion() and runs the tool loop.
    """

    def __init__(
        self,
        tool_ids: list[str],
        tools_cfg: list[dict],
        mcp_client: "McpClient | None" = None,
    ) -> None:
        id_set = set(tool_ids)
        # Built-in tools
        self._tools: dict[str, dict] = {t["id"]: t for t in tools_cfg if t["id"] in id_set}
        # MCP tools — indexed by prefixed ID
        self._mcp_client = mcp_client
        if mcp_client:
            for t in mcp_client.tools():
                if t["id"] in id_set:
                    self._tools[t["id"]] = t

    def __bool__(self) -> bool:
        return bool(self._tools)

    def schemas(self) -> list[dict]:
        """LiteLLM-compatible function calling schemas for assigned tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["id"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in self._tools.values()
        ]

    async def execute(self, name: str, arguments: dict) -> str:
        """Dispatch a tool call and return a string result."""
        # MCP tools — route to McpClient
        if name.startswith("mcp__"):
            if self._mcp_client:
                return await self._mcp_client.call(name, arguments)
            return f"[mcp: no client available for tool: {name}]"
        # Built-in tools
        if name == "python_exec":
            return await _tool_python_exec(arguments.get("code", ""))
        if name == "read_file":
            return _tool_read_file(arguments.get("path", ""))
        if name == "write_file":
            return _tool_write_file(arguments.get("path", ""), arguments.get("content", ""))
        if name == "edit_file":
            return _tool_edit_file(arguments.get("path", ""), arguments.get("old_string", ""), arguments.get("new_string", ""))
        if name == "glob_search":
            return _tool_glob_search(arguments.get("pattern", ""), arguments.get("path", "."))
        if name == "grep_search":
            return _tool_grep_search(arguments.get("pattern", ""), arguments.get("path", "."), arguments.get("glob", "**/*"), arguments.get("case_insensitive", False))
        if name == "web_search":
            return await _tool_web_search(arguments.get("query", ""), arguments.get("max_results", 5))
        if name == "web_crawl":
            return await _tool_web_crawl(arguments.get("url", ""))
        return f"[unknown tool: {name}]"


_EFFICIENCY_ABILITIES = {"speed", "cost_efficiency"}


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
      best           = 1.0 (mode fully overrides task)
      balanced       = 0.35 (task leads, mode nudges)
      cost_efficient = 0.75 (mode strongly overrides task)

    max_quality caps model ability scores on quality dimensions (non-speed/cost).
    In balanced mode (max_quality=2), a model that scores 3 on reasoning is treated
    the same as one that scores 2 — you're buying "good enough", not "exceptional".
    This prevents premium models from always winning on quality alone.
    Returns (model_id, score).
    """
    profile = MODE_PROFILES.get(mode, MODE_PROFILES["balanced"])
    alpha = profile["alpha"]
    max_q = float(profile.get("max_quality", 3))
    best_id, best_score = models[0]["id"], -1.0
    for m in models:
        score = sum(
            ((1 - alpha) * requirements.get(a, 0.0) + alpha * profile[a])
            * (float(m.get(a, 0.0)) if a in _EFFICIENCY_ABILITIES
               else min(float(m.get(a, 0.0)), max_q))
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
    quality = [a for a in ABILITIES if a not in ("speed", "cost_efficiency", "embedding")]
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

_SHARE_MARKER    = "\nSHARE:"
_FINDINGS_MARKER = "\nFINDINGS:"
_REMEMBER_MARKER = "\nREMEMBER:"


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


def extract_findings(response: str) -> tuple[str, dict]:
    """
    Split a worker response into (prose_output, findings_dict).

    Workers append a FINDINGS: block to broadcast compact structured facts to
    subsequent workers. This replaces passing full prose inter-worker, cutting
    token cost by 80-90% on multi-task pipelines.

    Expected worker output order:
        [prose work result]
        FINDINGS:
        key: value
        SHARE:          ← optional, for global shared_context broadcast
        key: value

    FINDINGS are consumed only by subsequent workers; SHARE is consumed by all
    agents. extract_findings() must be called before extract_share() so the
    SHARE block is not accidentally included in findings parsing.

    If no FINDINGS block is present, returns ({}) and the full response as-is
    — caller falls back to passing full prose (backward compatible).
    """
    if _FINDINGS_MARKER not in response:
        return response.strip(), {}

    result_part, findings_raw = response.split(_FINDINGS_MARKER, 1)
    # Stop at REMEMBER: or SHARE: — both may follow FINDINGS in worker output
    for _stop in (_REMEMBER_MARKER, _SHARE_MARKER):
        findings_raw = findings_raw.split(_stop, 1)[0]
    findings_text = findings_raw
    findings: dict = {}
    for line in findings_text.strip().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            if k:
                findings[k] = v
    return result_part.strip(), findings


def extract_remember(response: str) -> tuple[str, dict]:
    """
    Split an agent response into (result_without_remember, persistent_facts).

    REMEMBER: is the cross-run analogue of SHARE:. Facts in a REMEMBER block
    are saved to memories/facts/persistent.json and survive across runs.
    They are also merged into shared_context for the current run (like SHARE:).

    Expected output order (all sections optional):
        [prose]
        FINDINGS:   ← inter-worker structured facts (workers only)
        REMEMBER:   ← persistent cross-run facts
        SHARE:      ← ephemeral run-scoped broadcast

    Must be called after extract_findings() and before extract_share().
    """
    if _REMEMBER_MARKER not in response:
        return response, {}

    result_part, remember_raw = response.split(_REMEMBER_MARKER, 1)
    # Stop at SHARE: if it follows REMEMBER:
    remember_text = remember_raw.split(_SHARE_MARKER, 1)[0]
    _SKIP = {"REMEMBER", "SHARE", "FINDINGS"}
    facts: dict = {}
    for line in remember_text.strip().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            if k and k.upper() not in _SKIP:
                facts[k] = v
    return result_part.strip(), facts


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

    def get(self, name: str) -> "Agent | None":
        for a in self._agents:
            if a.name == name:
                return a
        return None

    def all(self) -> list[Agent]:
        return list(self._agents)


# =============================================================================
# Memory
# =============================================================================
#
# Architecture:
#   STM (ShortTermMemory)  — in-memory, current run scope
#                            seeded from LTM facts at run start
#                            accumulated via SHARE: broadcasts during the run
#   LTM (LongTermMemory)   — cross-run persistence via pluggable backend
#     facts     — key-value facts (REMEMBER: protocol)
#     research  — query → summary semantic cache
#
# Backend is swappable: FilesystemBackend now, QdrantBackend/RedisBackend later.
# Callers always talk to Memory.stm / Memory.ltm — never to the backend directly.


class MemoryBackend(ABC):
    """Pluggable persistence layer. Swap without changing any callers."""

    @abstractmethod
    def load(self, collection: str) -> dict: ...

    @abstractmethod
    def save(self, collection: str, data: dict) -> None: ...


class FilesystemBackend(MemoryBackend):
    """JSON files: memories/<collection>.json"""

    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir
        self._base.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _migrate(self) -> None:
        """One-time migration from old subdirectory layout to flat files."""
        for old_stem, collection in [("facts/persistent", "facts"), ("research/cache", "research")]:
            old_path = self._base / f"{old_stem}.json"
            new_path = self._base / f"{collection}.json"
            if old_path.exists() and not new_path.exists():
                new_path.write_text(old_path.read_text(encoding="utf-8"), encoding="utf-8")

    def load(self, collection: str) -> dict:
        path = self._base / f"{collection}.json"
        if not path.exists():
            return {}
        raw = path.read_text(encoding="utf-8")
        return json.loads(raw) if raw.strip() else {}

    def save(self, collection: str, data: dict) -> None:
        path = self._base / f"{collection}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class ShortTermMemory:
    """In-memory store for the current run. Not persisted. Cleared on each run start."""

    def __init__(self, seed: dict | None = None) -> None:
        self._store: dict = dict(seed or {})

    def get(self, key: str, default=None):
        return self._store.get(key, default)

    def set(self, key: str, value) -> None:
        self._store[key] = value

    def merge(self, data: dict) -> None:
        self._store.update(data)

    def all(self) -> dict:
        return dict(self._store)


class LongTermMemory:
    """Cross-run persistence via pluggable backend.

    Collections:
      facts     — key-value facts about user/project (REMEMBER: protocol)
      research  — query → result semantic cache
    """

    _SKIP = frozenset({"REMEMBER", "SHARE", "FINDINGS"})

    def __init__(self, backend: MemoryBackend) -> None:
        self._b = backend

    # ── Facts ──────────────────────────────────────────────────────────────────

    def get_facts(self) -> dict:
        return self._b.load("facts")

    def save_facts(self, updates: dict) -> None:
        filtered = {k: v for k, v in updates.items() if k.upper() not in self._SKIP}
        if not filtered:
            return
        facts = self._b.load("facts")
        facts.update(filtered)
        self._b.save("facts", facts)

    # ── Research cache ─────────────────────────────────────────────────────────

    def get_research(self, query: str, query_embedding: list[float] | None = None) -> dict | None:
        """
        Return a cached research entry for query, or None on miss.

        Lookup order:
          1. Exact normalised-key match
          2. Cosine similarity on stored embeddings (threshold 0.82) — if embedding provided
          3. Word-overlap fallback (threshold 0.72) — if no embeddings available
        """
        cache = self._b.load("research")
        key = _norm_query(query)
        if key in cache:
            return cache[key]
        if query_embedding:
            best_score, best_entry = 0.0, None
            for entry in cache.values():
                emb = entry.get("embedding")
                if emb:
                    score = _cosine(query_embedding, emb)
                    if score > best_score:
                        best_score, best_entry = score, entry
            if best_score >= 0.82:
                return best_entry
        best_score, best_entry = 0.0, None
        for cached_key, entry in cache.items():
            score = _word_overlap(key, cached_key)
            if score > best_score:
                best_score, best_entry = score, entry
        return best_entry if best_score >= 0.72 else None

    def save_research(self, query: str, summary: str, run_id: str,
                      query_embedding: list[float] | None = None) -> None:
        cache = self._b.load("research")
        entry: dict = {
            "query": query,
            "summary": summary,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if query_embedding:
            entry["embedding"] = query_embedding
        cache[_norm_query(query)] = entry
        self._b.save("research", cache)


class Memory:
    """Unified memory facade. One object passed to all graph nodes.

    stm — current run, in-memory, cleared each run
    ltm — cross-run, persisted via backend

    To swap backend:  Memory(backend=QdrantBackend(url=...))
    """

    def __init__(self, backend: MemoryBackend | None = None) -> None:
        _backend = backend or FilesystemBackend(MEMORIES_DIR)
        self.ltm = LongTermMemory(_backend)
        self.stm = ShortTermMemory(seed=self.ltm.get_facts())



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
        self._messages: list[dict] = []       # private conversation history
        self._actions: list[dict] = []        # audit log of every think() call
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    async def think(
        self,
        user_msg: str,
        shared_ctx: dict | None = None,
        tool_executor: "ToolExecutor | None" = None,
    ) -> str:
        """
        Call the LLM, optionally running a tool-call loop until the model
        produces a plain text response.  Always returns a string.

        If tool_executor is provided, available tools are passed to
        litellm.acompletion() as `tools=`.  When the model issues tool_calls,
        each tool is executed and results are fed back; the loop continues
        until the model returns a final text reply.
        """
        system = self._build_system(shared_ctx)
        self._messages.append({"role": "user", "content": user_msg})

        _log(self.name, "→ input",  f"[yellow]{user_msg[:120]}{'...' if len(user_msg) > 120 else ''}[/yellow]")
        _log(self.name, "model",    f"[dim]{self.model}[/dim]")

        tool_schemas = tool_executor.schemas() if tool_executor else None
        t_start = datetime.now(timezone.utc)
        call_prompt = call_completion = call_total = 0
        final_reply = ""
        tool_calls_log: list[dict] = []  # every tool call made during this think() invocation

        while True:
            extra: dict = {"tools": tool_schemas} if tool_schemas else {}
            resp = await litellm.acompletion(
                model=_proxy_model(self.model),
                messages=[{"role": "system", "content": system}] + self._messages,
                max_tokens=self.cfg.get("max_tokens", 1000),
                **_LITELLM_EXTRA,
                **extra,
            )
            usage = resp.usage
            self.total_tokens      += usage.total_tokens
            self.prompt_tokens     += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            call_prompt     += usage.prompt_tokens
            call_completion += usage.completion_tokens
            call_total      += usage.total_tokens

            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []

            if tool_calls and tool_executor:
                # Append assistant tool-call turn
                self._messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                })
                # Execute each tool, log result, append to messages
                for tc in tool_calls:
                    args = json.loads(tc.function.arguments or "{}")
                    _log(self.name, f"→ tool:{tc.function.name}", f"[yellow]{str(args)[:80]}[/yellow]")
                    tool_result = await tool_executor.execute(tc.function.name, args)
                    _log(self.name, f"← tool:{tc.function.name}", f"[cyan]{str(tool_result)[:120]}[/cyan]")
                    tool_calls_log.append({
                        "tool": tc.function.name,
                        "args": args,
                        "result_preview": str(tool_result)[:300],
                    })
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(tool_result),
                    })
                # Continue — model will incorporate tool results
            else:
                # Final text response — exit loop
                final_reply = (msg.content or "").strip()
                self._messages.append({"role": "assistant", "content": final_reply})
                break

        elapsed_s = round((datetime.now(timezone.utc) - t_start).total_seconds(), 2)
        _log(self.name, "tokens",   f"prompt={call_prompt}  completion={call_completion}  total={call_total}  time={elapsed_s}s")
        _log(self.name, "← output", f"[green]{final_reply[:200]}{'...' if len(final_reply) > 200 else ''}[/green]")

        self._actions.append({
            "timestamp": t_start.isoformat(),
            "elapsed_s": elapsed_s,
            "input": user_msg,
            "output": final_reply,
            "tool_calls": tool_calls_log,
            "shared_context_snapshot": dict(shared_ctx) if shared_ctx else {},
            "tokens": {
                "prompt": call_prompt,
                "completion": call_completion,
                "total": call_total,
            },
        })

        return final_reply

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
    direct_answer: str              # set by pm_plan → bypasses workers/judge; leader delivers directly
    shared_context: dict
    tasks: list[dict]               # each task: {id, description, ability scores..., prompt}
    # Pre-analysis research (feeds back into pm_replan for refined breakdown)
    research_for_analysis: bool
    research_for_analysis_query: str
    analysis_research_summary: str
    # Execution research (feeds into workers via shared_context)
    research_needed: bool
    research_query: str
    research_summary: str
    task_results: list[dict]
    agent_assessments: list[dict]   # [{agent, status, quality, tokens, notes}] from judge
    judge_metrics: dict             # {overall_status, total_tokens, pipeline_efficiency, cost_rating, recommendations}
    judge_human_message: str        # optional message from judge surfaced to human before leader responds
    conversation_history: list[dict]


# =============================================================================
# Node factories
# =============================================================================


def make_leader_receive(agent: Agent):
    async def leader_receive(state: SharedState) -> dict[str, Any]:
        _section("Leader  ·  Receive Request", "blue")
        query = state["user_query"]
        # Merge: preserve incoming shared_context (persistent facts from memory) and add objective on top
        shared_ctx: dict = {**state.get("shared_context", {}), "objective": query}
        # Routing-only — no LLM call. Seeds shared_context for the pipeline.
        _log(agent.name, "route", "[dim]request received — routing to pipeline[/dim]")
        return {"shared_context": shared_ctx}
    return leader_receive


def _parse_pm_tasks(parsed: dict, fallback_query: str) -> list[dict]:
    """Normalise the tasks array from a PM plan JSON response."""
    raw_tasks: list[dict] = parsed.get("tasks", [])
    tasks: list[dict] = []
    for i, t in enumerate(raw_tasks):
        task = {
            "id":          str(t.get("id") or f"task_{i + 1}"),
            "description": t.get("description") or t.get("task") or str(t),
            "prompt":      t.get("prompt") or "",
            "tools":       [str(tid) for tid in (t.get("tools") or [])],
        }
        for a in ABILITIES:
            task[a] = float(t.get(a, 0.0))
        tasks.append(task)
    if not tasks:
        tasks = [{"id": "task_1", "description": fallback_query, "prompt": "",
                  **{a: 0.5 for a in ABILITIES}}]
    return tasks


def make_pm_plan(agent: Agent, memory: Memory):
    """
    Program Manager — planning node.

    Single LLM call that replaces the old two-call analyser → execution_lead
    planning sequence. Outputs tasks with ability ratings and worker prompts
    already embedded, so pm_execute needs no additional planning call.
    """
    async def pm_plan(state: SharedState) -> dict[str, Any]:
        _section("Program Manager  ·  Plan", "cyan")
        shared_ctx = state.get("shared_context", {})

        raw = await agent.think(
            f"Plan work for this request.\n\nRequest: \"{state['user_query']}\"\n\n"
            "Task descriptions must be actionable goals — what the worker must accomplish. "
            "Never a completed fact, never a method. Do not assume what info is available. "
            "Each task must be a meaningful phase — tightly coupled steps (e.g. write + test + run code) belong in one task, not several.",
            shared_ctx,
        )

        try:
            parsed = _parse_json(raw)
        except ValueError as e:
            _log(agent.name, "warn", f"[yellow]JSON parse failed — single-task fallback ({e})[/yellow]")
            parsed = {"tasks": [{"id": "task_1", "description": state["user_query"]}]}

        # Save any facts the PM extracted (e.g. user stating "team is based in India")
        facts = {k: str(v) for k, v in (parsed.get("facts") or {}).items() if k and v}
        facts = {k: v for k, v in facts.items() if k not in shared_ctx}  # skip already-known
        if facts:
            memory.ltm.save_facts(facts)
            memory.stm.merge(facts)
            shared_ctx = {**shared_ctx, **facts}
            _log(agent.name, "→ memory", f"[magenta]{list(facts.keys())}[/magenta]")

        direct_answer: str = parsed.get("direct_answer") or ""
        if direct_answer:
            _log(agent.name, "route", "[bold green]DIRECT ANSWER — skipping workers[/bold green]")
            parsed["tasks"] = []   # enforce: direct_answer and tasks are mutually exclusive
            return {
                "direct_answer": direct_answer,
                "tasks": [],
                "research_for_analysis": False,
                "research_for_analysis_query": "",
                "research_needed": False,
                "research_query": "",
                "shared_context": shared_ctx,
            }

        tasks = _parse_pm_tasks(parsed, state["user_query"])
        research_for_analysis: bool = bool(parsed.get("research_for_analysis", False))
        research_for_analysis_query: str = parsed.get("research_for_analysis_query") or ""
        research_needed: bool = bool(parsed.get("research_needed", False))
        research_query: str = parsed.get("research_query") or ""

        tbl = Table(title="PM Plan", border_style="dim")
        tbl.add_column("ID", style="cyan", no_wrap=True)
        tbl.add_column("Description")
        tbl.add_column("Has Prompt", justify="center")
        for task in tasks:
            tbl.add_row(task["id"], task["description"], "✓" if task.get("prompt") else "—")
        console.print(tbl)
        _log(agent.name, "research_for_analysis", f"[{'yellow' if research_for_analysis else 'dim'}]{research_for_analysis}[/]")
        _log(agent.name, "research_needed",        f"[{'green'  if research_needed       else 'dim'}]{research_needed}[/]")

        return {
            "tasks": tasks,
            "research_for_analysis": research_for_analysis,
            "research_for_analysis_query": research_for_analysis_query,
            "research_needed": research_needed,
            "research_query": research_query,
            "shared_context": {**shared_ctx, "tasks_count": str(len(tasks))},
        }
    return pm_plan


async def _do_research(
    query: str,
    researcher: Agent,
    memory: Memory,
    run_id: str,
    shared_ctx: dict,
    caller: str = "researcher",
    tools_cfg: list[dict] | None = None,
) -> str:
    """
    Sole entry point for research LTM access.

    Cache hit  → return cached summary, no LLM call.
    Cache miss → call researcher LLM (with web tools if provided), save to cache, return summary.

    Only called from make_researcher (primary path) and pm_execute (inline RESEARCH: signals).
    No other code touches memory.ltm.get_research / save_research.
    """
    query_emb = await _embed_query(query)
    cached = memory.ltm.get_research(query, query_embedding=query_emb)
    if cached:
        _log(caller, "memory hit", f"[green]cache from run {cached['run_id']}[/green] — skipping LLM call")
        return cached["summary"]
    tool_exec = ToolExecutor([t["id"] for t in tools_cfg], tools_cfg) if tools_cfg else None
    raw = await researcher.think(
        f"Research: {query}\n\nDense factual summary — facts and findings only.",
        shared_ctx,
        tool_executor=tool_exec,
    )
    summary, _ = extract_share(raw)
    memory.ltm.save_research(query, summary, run_id, query_embedding=query_emb)
    _log(caller, "→ research cache", f"[dim]saved ({len(summary)} chars)[/dim]")
    return summary


def make_researcher(agent: Agent, memory: Memory, run_id: str, tools_cfg: list[dict] | None = None):
    async def researcher(state: SharedState) -> dict[str, Any]:
        for_analysis = state.get("research_for_analysis", False)
        if for_analysis:
            _section("Researcher Agent  ·  Pre-Analysis Research", "yellow")
            query = state.get("research_for_analysis_query") or state["user_query"]
        else:
            _section("Researcher Agent  ·  Execution Research", "yellow")
            query = state.get("research_query") or state["user_query"]

        shared_ctx = state.get("shared_context", {})

        # ── Memory check + LLM call via shared research helper ────────
        # For the primary researcher path, we need REMEMBER: extraction from the
        # raw response. Check cache first; on miss, call the LLM directly here
        # so we can extract persistent facts before delegating save to _do_research.
        query_emb = await _embed_query(query)
        cached = memory.ltm.get_research(query, query_embedding=query_emb)
        if cached:
            _log(agent.name, "memory hit", f"[green]cache from run {cached['run_id']}[/green] — skipping LLM call")
            summary = cached["summary"]
        else:
            tool_exec = ToolExecutor([t["id"] for t in tools_cfg], tools_cfg) if tools_cfg else None
            raw = await agent.think(
                f"Research: {query}\n\nReturn a dense, actionable summary — facts and findings only, no padding.\n\n"
                "REMEMBER RULE: If you discover facts that are project-level and should persist across all future runs "
                "(e.g. the project's language, framework, key constraints, user preferences), append a REMEMBER block:\n"
                "REMEMBER:\n"
                "key: value\n"
                "Use meaningful key names. Minimal — only genuinely stable facts. Omit if nothing permanent to note.",
                shared_ctx,
                tool_executor=tool_exec,
            )
            # Extract REMEMBER: (persistent facts) then SHARE: (ephemeral broadcast)
            raw_no_remember, persistent = extract_remember(raw)
            summary, updates = extract_share(raw_no_remember)
            shared_ctx = _merge_share(shared_ctx, updates, agent.name)
            memory.stm.merge(updates)
            if persistent:
                _log(agent.name, "→ memory", f"[magenta]{list(persistent.keys())}[/magenta]")
                memory.ltm.save_facts(persistent)
                memory.stm.merge(persistent)
                shared_ctx = {**shared_ctx, **persistent}
            memory.ltm.save_research(query, summary, run_id, query_embedding=query_emb)
            _log(agent.name, "→ research cache", f"[dim]saved ({len(summary)} chars)[/dim]")

        if for_analysis:
            return {
                "analysis_research_summary": summary,
                "shared_context": shared_ctx,
            }
        else:
            return {
                "research_summary": summary,
                "shared_context": shared_ctx,
            }
    return researcher


def make_pm_replan(agent: Agent):
    """
    Program Manager — replan node (post-research).

    Called after the researcher runs for pre-analysis research. Uses the same
    merged JSON format as pm_plan so pm_execute has ability ratings + prompts.
    """
    async def pm_replan(state: SharedState) -> dict[str, Any]:
        _section("Program Manager  ·  Replan (post-research)", "cyan")
        shared_ctx = state.get("shared_context", {})
        research = state.get("analysis_research_summary", "")

        raw = await agent.think(
            f"Refine your plan using this research:\n\n{research}\n\n"
            f"Request: \"{state['user_query']}\"\n\n"
            "Produce the full plan including ability scores and worker prompts. "
            "Set research_needed if workers still need external knowledge.",
            shared_ctx,
        )

        try:
            parsed = _parse_json(raw)
        except ValueError as e:
            _log(agent.name, "warn", f"[yellow]JSON parse failed on replan — keeping original plan ({e})[/yellow]")
            return {"research_for_analysis": False}

        tasks = _parse_pm_tasks(parsed, state["user_query"])
        if not tasks:
            tasks = state.get("tasks", [])

        research_needed: bool = bool(parsed.get("research_needed", False))
        research_query: str = parsed.get("research_query") or ""

        tbl = Table(title="PM Replan", border_style="dim")
        tbl.add_column("ID", style="cyan", no_wrap=True)
        tbl.add_column("Description")
        tbl.add_column("Has Prompt", justify="center")
        for task in tasks:
            tbl.add_row(task["id"], task["description"], "✓" if task.get("prompt") else "—")
        console.print(tbl)
        _log(agent.name, "research_needed", f"[{'green' if research_needed else 'dim'}]{research_needed}[/]")

        return {
            "tasks": tasks,
            "research_for_analysis": False,   # clear so researcher routes correctly next time
            "research_needed": research_needed,
            "research_query": research_query,
            "shared_context": {**shared_ctx, "tasks_count": str(len(tasks))},
        }
    return pm_replan


_WORKER_CTX_EXCLUDE = {"analysis_research", "research_snippet", "tasks_count"}
_WORKER_RULES = (
    "\n\n## Context\n"
    "You are one worker in a pipeline assembled by a Program Manager to serve a human request. "
    "The human is the end user who made the original request — they do not see your output directly "
    "unless you are the final worker. Build on the work done so far; do not repeat it.\n\n"
    "## Rules\n"
    "TOOLS: Tools are available for you to use when they advance your task. Use your judgment — "
    "if your task involves writing code that should be executed, run it rather than just printing it as text. "
    "If the task is already fully done by prior work, signal NOT_NEEDED instead.\n\n"
    "CODE OUTPUT: When you write code, always include the complete runnable source in your text output "
    "alongside any execution results. The next worker receives only your text — not files, not interpreter state. "
    "If your code is not in your output, it is lost.\n\n"
    "CONSISTENCY: Your work done so far shows what prior workers produced. Verify it is consistent with "
    "the original human request — if a prior worker used a weaker or incorrect approach, correct it. "
    "Do not silently accept a regression.\n\n"
    "Signals (use alone on a single line, nothing else):\n"
    "QUESTION: <text>   — ONLY for hard-blocked missing input with no reasonable default "
    "(e.g. a required credential, a specific name the user must supply). "
    "Never use for confirmations, approvals, or 'is this good?' checks — make your best decision and proceed.\n"
    "RESEARCH: <query>  — need external research\n"
    "NOT_NEEDED         — task not applicable given prior work\n\n"
    "REMEMBER:\nkey: value  — append after your result for permanent facts about the user, project, or team"
)


def make_pm_execute(pm: Agent, worker_cfg: dict, models: list[dict], default_model: str,
                    mode: str, registry: AgentRegistry, memory: Memory,
                    researcher_ag: Agent, run_id: str, tools_cfg: list[dict],
                    researcher_tools_cfg: list[dict] | None = None,
                    mcp_client: "McpClient | None" = None):
    """
    Program Manager — execute node.

    Workers run sequentially. Each receives the cumulative work summary from the
    previous worker. Workers signal the PM via:
      RESEARCH: <query>  — PM fetches research inline, worker is resumed with results
      QUESTION: <text>   — PM pauses via interrupt(), leader surfaces it to user
      NOT_NEEDED         — worker self-skips if task is no longer applicable
    """
    async def pm_execute(state: SharedState) -> dict[str, Any]:
        shared_ctx = dict(state.get("shared_context", {}))
        tasks = state.get("tasks") or [{"id": "task_1", "description": state["user_query"],
                                        "prompt": "", **{a: 0.5 for a in ABILITIES}}]
        research = state.get("research_summary", "")

        # ── Resume guard ───────────────────────────────────────────────────────
        cache_key = tuple((t["id"], t["description"]) for t in tasks)
        cached = getattr(pm, "_exec_cache", None)
        if cached and cached["key"] == cache_key:
            _section("Program Manager  ·  Resuming", "magenta")
        else:
            _section("Program Manager  ·  Executing", "magenta")
            pm._exec_cache = {"key": cache_key, "completed": {}, "workers": {}}

        completed: dict = pm._exec_cache["completed"]
        workers_cache: dict = pm._exec_cache["workers"]

        # ── Worker dispatch ────────────────────────────────────────────────────
        sel_table = Table(title="Worker Model Selection", border_style="dim")
        sel_table.add_column("Worker", style="cyan")
        sel_table.add_column("Model")
        sel_table.add_column("Score", justify="right")

        task_results: list[dict] = []
        for task in tasks:
            tid = task["id"]
            worker_name = f"Worker-{tid}"

            if tid in completed:
                if not completed[tid].get("skipped"):
                    task_results.append(completed[tid])
                _log(worker_name, "skip", "[dim]already completed[/dim]")
                continue

            reqs = {a: float(task.get(a, 0.0)) for a in ABILITIES}
            if all(v == 0.0 for v in reqs.values()):
                reqs = worker_cfg.get("abilities") or {a: 0.5 for a in ABILITIES}
            chosen_model, score = select_model(reqs, models, mode)
            sel_table.add_row(worker_name, f"[dim]{chosen_model}[/dim]", str(score))

            w_cfg = {**worker_cfg, "model": chosen_model}
            base_prompt = task.get("prompt") or (
                f"Task: {task['description']}\n"
                + (f"Research context:\n{research}\n" if research else "")
                + ("Produce the complete, human-ready final answer." if task == tasks[-1]
                   else "Deliver a complete, precise result.")
            )
            prompt = base_prompt.rstrip() + _WORKER_RULES

            _section(f"  {worker_name}  ·  {task['description'][:60]}", "green")
            _log(worker_name, "model", f"[dim]{chosen_model}[/dim]  score={score}")

            # Pass the running summary from the previous worker as context
            work_so_far = task_results[-1]["result"] if task_results else ""
            is_last = (task == tasks[-1])
            task_input = (
                f"Human request: {state['user_query']}\n\n"
                + ("Continue the work." if work_so_far else "Begin the task.")
                + (f"\n\nWork done so far:\n{work_so_far}" if work_so_far else "")
                + ("\n\nYou are the final worker. Your output is delivered directly to the human."
                   if is_last else "")
            )

            worker_shared_ctx = {k: v for k, v in shared_ctx.items() if k not in _WORKER_CTX_EXCLUDE}

            # Build ToolExecutor for this task (empty if no tools assigned)
            task_tool_ids = task.get("tools") or []
            tool_exec = ToolExecutor(task_tool_ids, tools_cfg, mcp_client=mcp_client) if task_tool_ids else None
            if task_tool_ids:
                _log(worker_name, "tools", f"[dim]{task_tool_ids}[/dim]")

            # Reuse cached worker on interrupt resume — start from where it paused
            cached_worker = workers_cache.get(tid)
            if cached_worker is not None:
                last_action = cached_worker._actions[-1] if cached_worker._actions else None
                if last_action and re.search(r"(?m)^QUESTION:", last_action["output"], re.IGNORECASE):
                    worker = cached_worker
                    raw = last_action["output"]
                    _log(worker_name, "resume", "[dim]resuming from paused question[/dim]")
                else:
                    worker = cached_worker
                    raw = await worker.think(task_input, worker_shared_ctx, tool_executor=tool_exec)
            else:
                worker = registry.register(Agent(worker_name, w_cfg, chosen_model, prompt))
                workers_cache[tid] = worker
                raw = await worker.think(task_input, worker_shared_ctx, tool_executor=tool_exec)

            # ── Signal loop: RESEARCH (inline) → QUESTION (interrupt) ─────────
            while True:
                # RESEARCH: resolved inline via researcher — cache check + LLM + save
                _r_match = re.search(r"(?m)^RESEARCH:\s*(.+)", raw, re.IGNORECASE)
                if _r_match:
                    query = _r_match.group(1).strip()
                    _log(worker_name, "→ research", f"[yellow]{query}[/yellow]")
                    research_result = await _do_research(
                        query, researcher_ag, memory, run_id, worker_shared_ctx,
                        caller=worker_name, tools_cfg=researcher_tools_cfg,
                    )
                    raw = await worker.think(
                        f"Research results:\n{research_result}\n\nNow complete your task.",
                        worker_shared_ctx,
                        tool_executor=tool_exec,
                    )
                    continue

                # QUESTION: PM pauses, leader surfaces it to the user
                _q_match = re.search(r"(?m)^QUESTION:\s*(.+)", raw, re.IGNORECASE)
                if not _q_match:
                    break
                question = _q_match.group(1).strip()
                _log(worker_name, "question", f"[yellow]{question}[/yellow]")
                answer: str = interrupt(f"[{worker_name}] {question}")
                _log(worker_name, "← answer", f"[cyan]{answer}[/cyan]")
                qa_key = f"qa{sum(1 for k in shared_ctx if k.startswith('qa'))}"
                shared_ctx = {**shared_ctx, qa_key: f"Q:{question}|A:{answer}"}
                worker_shared_ctx = {k: v for k, v in shared_ctx.items() if k not in _WORKER_CTX_EXCLUDE}
                raw = await worker.think(
                    f"User clarification: \"{answer}\"\n\nNow complete your task.",
                    worker_shared_ctx,
                    tool_executor=tool_exec,
                )

            # Extract REMEMBER → SHARE
            raw_no_remember, persistent = extract_remember(raw)
            result, updates = extract_share(raw_no_remember)
            # Strip NOT_NEEDED signal if worker appended it after valid content
            result = re.sub(r"(?im)\n?^NOT_NEEDED\s*$", "", result).strip()
            findings: dict = {}
            shared_ctx = _merge_share(shared_ctx, updates, worker_name)
            memory.stm.merge(updates)
            if persistent:
                _log(worker_name, "→ memory", f"[magenta]{list(persistent.keys())}[/magenta]")
                memory.ltm.save_facts(persistent)
                memory.stm.merge(persistent)
                shared_ctx = {**shared_ctx, **persistent}

            not_needed = result.strip().upper() == "NOT_NEEDED"
            # Collect unique tool names actually called (order-preserving)
            all_tc = [tc for act in worker._actions for tc in act.get("tool_calls", [])]
            tools_used = list(dict.fromkeys(tc["tool"] for tc in all_tc))
            if not_needed:
                _log(worker_name, "skip", "[dim]NOT_NEEDED — task skipped by worker[/dim]")
            else:
                if tools_used:
                    _log(worker_name, "tools used", f"[magenta]{tools_used}[/magenta]")
                _log(worker_name, "done", f"[green]{len(result)} chars[/green]")

            result_entry = {
                "task_id": tid, "agent": worker_name, "model": chosen_model,
                "score": score, "description": task["description"],
                "result": result, "findings": findings, "skipped": not_needed,
                "tools_used": tools_used if not not_needed else [],
            }
            completed[tid] = result_entry
            if not not_needed:
                task_results.append(result_entry)

        console.print(sel_table)

        # ── Final response = last worker's output ──────────────────────────────
        # No consolidation LLM call — the task list is a sequential pipeline
        # designed so the last worker produces the complete human-ready answer.
        if not task_results:
            _log(pm.name, "warn", "[yellow]no task results — all workers skipped[/yellow]")
            return {"task_results": [], "shared_context": shared_ctx, "final_response": ""}

        # Use last worker's output; fall back to last non-empty result if final worker returned nothing
        final_response = task_results[-1]["result"]
        if not final_response.strip():
            fallback = next((r["result"] for r in reversed(task_results) if r["result"].strip()), "")
            if fallback:
                _log(pm.name, "warn", "[yellow]final worker returned empty — using last non-empty worker output[/yellow]")
                final_response = fallback
        _log(pm.name, "final", f"[green]final response ready ({len(final_response)} chars)[/green]")
        return {"task_results": task_results, "shared_context": shared_ctx, "final_response": final_response}

    return pm_execute



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

        # ── Pricing lookup (used in summary and report) ───────────────────────
        _pricing = {m["id"]: m for m in load_models()}

        def _agent_cost_usd(a: Agent) -> float | None:
            mid = a.model.removeprefix("openai/")
            p = _pricing.get(mid, {})
            inp, out = p.get("input_cost_per_1m"), p.get("output_cost_per_1m")
            if inp is None or out is None:
                return None
            return (a.prompt_tokens * inp + a.completion_tokens * out) / 1_000_000

        def _cost_str(val: float | None) -> str:
            return f"${val:.4f}" if val is not None else "—"

        # ── Run summary sent to judge ──────────────────────────────────────────
        def _agent_note(a: Agent) -> str:
            if a.name == "leader":
                return "routing-only (no LLM call by design)"
            return ""

        task_results = state.get("task_results", [])
        total_tokens = sum(a.total_tokens for a in registry.all())
        total_cost = sum(c for a in registry.all() if (c := _agent_cost_usd(a)) is not None)

        agent_rows = "\n".join(
            f"{a.name},{a.cfg.get('role', a.name)},{len(a._actions)},{a.total_tokens},"
            f"{_elapsed(a)},{_cost_str(_agent_cost_usd(a))},{_agent_note(a)}"
            for a in registry.all()
        )

        # Workers: FINDINGS (compact) for intermediate; full result for the last one
        _SNIPPET = 400
        # Build task_id → tools_assigned map from the plan
        _task_tools_assigned = {t["id"]: t.get("tools") or [] for t in state.get("tasks", [])}

        task_rows_parts = []
        for i, r in enumerate(task_results):
            is_last = (i == len(task_results) - 1)
            assigned = _task_tools_assigned.get(r["task_id"], [])
            used     = r.get("tools_used") or []
            tools_line = (
                f"\n  tools_assigned: {', '.join(assigned) or 'none'}"
                f"  tools_used: {', '.join(used) or 'none'}"
            ) if assigned or used else ""
            if is_last:
                snippet = r["result"][:_SNIPPET] + ("…" if len(r["result"]) > _SNIPPET else "")
                task_rows_parts.append(
                    f"  {r['task_id']} [FINAL]: {r['description'][:60]}{tools_line}\n"
                    f"  output: {snippet}"
                )
            elif r.get("findings"):
                kv = " | ".join(f"{k}: {v}" for k, v in r["findings"].items())
                task_rows_parts.append(f"  {r['task_id']}: {r['description'][:60]}{tools_line} → {kv}")
            else:
                snippet = r["result"][:_SNIPPET] + ("…" if len(r["result"]) > _SNIPPET else "")
                task_rows_parts.append(
                    f"  {r['task_id']}: {r['description'][:60]}{tools_line}\n"
                    f"  output: {snippet}"
                )
        task_rows = "\n".join(task_rows_parts) or "  none"

        num_agents = len(list(registry.all()))
        run_summary = (
            f"agents ({num_agents}): name,role,llm_calls,tokens,elapsed,cost,note\n{agent_rows}\n\n"
            f"total_tokens: {total_tokens}  total_cost: {_cost_str(total_cost)}\n\n"
            f"tasks_completed: {len(task_results)}\n"
            f"task_detail:\n{task_rows}\n\n"
            f"research_for_analysis_ran: {bool(state.get('analysis_research_summary'))}\n"
            f"execution_research_ran: {bool(state.get('research_summary'))}"
        )

        assessments: list[dict] = []
        metrics: dict = {}
        judge_human_message: str = ""

        if state.get("direct_answer"):
            # Direct-answer fast path — no LLM call; synthesise assessment from actual counts
            _log(agent.name, "route", "[dim]direct answer — skipping LLM call[/dim]")
            for a in registry.all():
                if a.name == "program_manager":
                    assessments.append({
                        "agent": a.name, "status": "successful", "quality": "3",
                        "tokens": str(a.total_tokens), "cost_usd": _cost_str(_agent_cost_usd(a)),
                        "notes": "direct answer — workers bypassed",
                    })
                elif a.name == "leader":
                    assessments.append({
                        "agent": a.name, "status": "not_required", "quality": None,
                        "tokens": str(a.total_tokens), "cost_usd": _cost_str(_agent_cost_usd(a)),
                        "notes": "routing-only",
                    })
                else:
                    assessments.append({
                        "agent": a.name, "status": "not_required", "quality": None,
                        "tokens": "0", "cost_usd": "—", "notes": "",
                    })
            metrics = {
                "overall_status": "successful",
                "pipeline_efficiency": "high",
                "cost_rating": "3",
                "recommendations": "none",
            }
        else:
            agent_count = len(list(registry.all()))
            raw = await agent.think(
                f"Judge this pipeline run for: \"{state['user_query']}\"\n\n"
                f"{run_summary}\n\n"
                f"Output a JSON object with three keys:\n"
                f"- assessments: array of {agent_count} objects, one per agent, each with: "
                f"agent (str), status (successful|unsuccessful|not_required), quality (0-3 or null if not_required), "
                f"tokens (int), cost_usd (str), notes (str)\n"
                f"- metrics: object with: overall_status (successful|partial|failed), "
                f"pipeline_efficiency (high|medium|low), cost_rating (0-3), recommendations (str)\n"
                f"- human_message: empty string if run is satisfactory; brief message for human if a quality issue "
                f"is worth flagging (e.g. incomplete output, skipped task, partial answer, assigned tools not used, "
                f"regression from prior work). Be specific and concise.",
                shared_ctx,
            )
            try:
                parsed = _parse_json(raw)
                assessments = [
                    {
                        "agent":    str(r.get("agent", "?")),
                        "status":   str(r.get("status", "?")),
                        "quality":  None if r.get("quality") is None else str(r.get("quality")),
                        "tokens":   str(r.get("tokens", "?")),
                        "cost_usd": str(r.get("cost_usd", "—")),
                        "notes":    str(r.get("notes", "")),
                    }
                    for r in parsed.get("assessments", [])
                ]
                raw_metrics = parsed.get("metrics", {})
                if isinstance(raw_metrics, dict):
                    metrics = {k: str(v) for k, v in raw_metrics.items()}
                judge_human_message = str(parsed.get("human_message") or "").strip()
                if judge_human_message:
                    _log(agent.name, "→ human", f"[yellow]{judge_human_message}[/yellow]")
            except ValueError as e:
                _log(agent.name, "warn", f"[yellow]JSON parse failed: {e}[/yellow]")

        # ── Inject accurate totals from registry (judge can't know its own tokens or post-pipeline leader)
        metrics["total_tokens"] = str(total_tokens)
        metrics["total_cost_usd"] = _cost_str(total_cost if total_cost > 0 else None)

        # ── Print assessment table ────────────────────────────────────────────
        # Build per-agent tools_used map from task_results
        _agent_tools: dict[str, list[str]] = {}
        for tr in task_results:
            if tr.get("tools_used"):
                _agent_tools[tr["agent"]] = tr["tools_used"]

        t = Table(title="Agent Assessments", border_style="dim")
        t.add_column("Agent", style="bold")
        t.add_column("Status")
        t.add_column("Quality", justify="center")
        t.add_column("Tokens", justify="right")
        t.add_column("Cost", justify="right")
        t.add_column("Tools used", style="magenta")
        t.add_column("Notes")
        status_colour = {"successful": "green", "unsuccessful": "red", "not_required": "dim"}
        for a in assessments:
            c = status_colour.get(a["status"], "white")
            q_display = "—" if a["quality"] is None else a["quality"]
            tools_display = ", ".join(_agent_tools.get(a["agent"], [])) or "—"
            t.add_row(
                a["agent"],
                f"[{c}]{a['status']}[/{c}]",
                f"[dim]{q_display}[/dim]",
                f"[dim]{a['tokens']}[/dim]",
                f"[dim]{a.get('cost_usd', '—')}[/dim]",
                tools_display,
                f"[dim]{a['notes']}[/dim]",
            )
        console.print(t)

        # ── Print metrics summary ─────────────────────────────────────────────
        if metrics:
            m = Table(title="Pipeline Metrics", border_style="dim", show_header=False, box=None)
            m.add_column(style="bold")
            m.add_column()
            colour_map = {
                "successful": "green", "partial": "yellow", "failed": "red",
                "high": "green", "medium": "yellow", "low": "red",
            }
            for k, v in metrics.items():
                col = colour_map.get(str(v).lower(), "cyan")
                m.add_row(k, f"[{col}]{v}[/{col}]")
            console.print(m)

        # ── Build hierarchical text report ───────────────────────────────────
        def _fmt_model(m: str) -> str:
            m = m.removeprefix("openai/")
            return m[-36:] if len(m) > 36 else m

        def _status_sym(s: str) -> str:
            return {"successful": "✓", "unsuccessful": "✗", "not_required": "—"}.get(s, "—")

        def _agent_cost(ag: Agent) -> str:
            return _cost_str(_agent_cost_usd(ag))

        W = 82
        SEP  = "═" * W
        LINE = "─" * W
        assess_map = {a["agent"]: a for a in assessments}
        ag_map     = {a.name: a for a in registry.all()}
        task_results = state.get("task_results", [])
        total_tokens = sum(a.total_tokens for a in registry.all())

        def _agent_row(name: str, indent: str = "") -> list[str]:
            ag  = ag_map.get(name)
            asmnt = assess_map.get(name, {})
            status  = asmnt.get("status", "—")
            quality = "—" if asmnt.get("quality") is None else asmnt.get("quality", "—")
            tokens  = ag.total_tokens if ag else 0
            elapsed = _elapsed(ag) if ag else "—"
            model   = _fmt_model(ag.model) if ag else "—"
            cost    = _agent_cost(ag) if ag else "—"
            sym     = _status_sym(status)
            label   = f"{indent}{name}"
            return [f"  {label:<28} {model:<36} {tokens:>5} tok  {elapsed:>6}  {cost:>9}  {sym:<12} {quality}"]

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

        # Fixed pipeline order; workers nested under program_manager
        named_order = ["leader", "judge", "program_manager", "researcher"]
        for name in named_order:
            if name not in ag_map:
                asmnt  = assess_map.get(name, {})
                status = asmnt.get("status", "not_required")
                report_lines.append(f"  {name:<28} {'—':<36} {'—':>9}  {'—':>6}  {'—':>9}  {_status_sym(status):<12} —")
            else:
                report_lines.extend(_agent_row(name))
            # Workers appear after program_manager
            if name == "program_manager":
                _tr_tools_assigned = {t["id"]: t.get("tools") or [] for t in state.get("tasks", [])}
                for r in task_results:
                    wname = r["agent"]
                    row   = _agent_row(wname, indent="  └─ ")
                    report_lines.extend(row)
                    report_lines.append(f"     {'':28} task: {r['description'][:60]}")
                    assigned = _tr_tools_assigned.get(r["task_id"], [])
                    used     = r.get("tools_used") or []
                    if assigned or used:
                        report_lines.append(
                            f"     {'':28} tools assigned: {', '.join(assigned) or 'none'}"
                            f"  used: {', '.join(used) or 'none'}"
                        )

        total_cost_str = _cost_str(total_cost if total_cost > 0 else None)

        report_lines += [
            LINE,
            f"  {'TOTAL':<28} {'':36} {total_tokens:>5} tok  {'':>6}  {total_cost_str:>9}",
            "",
            "  PIPELINE METRICS",
            LINE,
        ]
        for k, v in metrics.items():
            report_lines.append(f"  {k:<28} {v}")

        # ── Tool call detail section ───────────────────────────────────────────
        all_tool_workers = [r for r in task_results if r.get("tools_used")]
        if all_tool_workers:
            report_lines += ["", "  TOOL CALLS", LINE]
            for r in all_tool_workers:
                report_lines.append(f"  {r['agent']:<28} tools: {', '.join(r['tools_used'])}")
                # Show each call with args preview
                wag = ag_map.get(r["agent"])
                if wag:
                    all_tc = [tc for act in wag._actions for tc in act.get("tool_calls", [])]
                    for tc in all_tc:
                        args_preview = str(tc.get("args", {}))[:60]
                        result_preview = tc.get("result_preview", "")[:80]
                        report_lines.append(f"    {tc['tool']:<26} args: {args_preview}")
                        if result_preview:
                            report_lines.append(f"    {'':26} result: {result_preview}")

        report_lines += ["", SEP]

        report = "\n".join(report_lines)

        # ── Write judge dump (plain text report) ─────────────────────────────
        judge_dir = DUMPS_DIR / "judge"
        judge_dir.mkdir(parents=True, exist_ok=True)
        dump_path = judge_dir / f"{run_id}.txt"
        dump_path.write_text(report, encoding="utf-8")
        _log(agent.name, "dump", f"[cyan]{dump_path}[/cyan]")

        return {"agent_assessments": assessments, "judge_metrics": metrics, "judge_human_message": judge_human_message}
    return judge_review


def make_leader_respond(agent: Agent):
    async def leader_respond(state: SharedState) -> dict[str, Any]:
        _section("Leader  ·  Respond to Human", "blue")
        shared_ctx = state.get("shared_context", {})
        history: list[dict] = list(state.get("conversation_history") or [])

        # Surface judge quality note if set (before delivering the answer)
        judge_note = state.get("judge_human_message", "").strip()
        if judge_note:
            _log(agent.name, "judge note", f"[yellow]{judge_note}[/yellow]")

        direct = state.get("direct_answer", "")
        if direct:
            _log(agent.name, "route", "[dim]direct answer — passing through[/dim]")
            agent._messages.append({"role": "user", "content": state["user_query"]})
            agent._messages.append({"role": "assistant", "content": direct})
            history.append({"role": "assistant", "content": direct})
            return {"final_response": direct, "conversation_history": history}

        # Pass PM's consolidated output directly — no LLM paraphrase needed.
        # PM already produces clean prose. Saves ~1000–2000 tokens on best_model.
        # Leader's private history is populated so the follow-up chat loop has full context.
        consolidated = state.get("final_response", "")
        _log(agent.name, "route", "[dim]passing PM consolidation directly (no LLM call)[/dim]")
        agent._messages.append({"role": "user", "content": f"Request: \"{state['user_query']}\"\n\nTeam output:\n\n{consolidated}"})
        agent._messages.append({"role": "assistant", "content": consolidated})
        history.append({"role": "assistant", "content": consolidated})
        return {"final_response": consolidated, "conversation_history": history}
    return leader_respond


# =============================================================================
# Routing
# =============================================================================


def route_after_pm_plan(state: SharedState) -> str:
    if state.get("direct_answer"):
        return "judge_review"
    if state.get("research_for_analysis"):
        return "researcher"
    if state.get("research_needed"):
        return "researcher"
    return "pm_execute"


def route_after_researcher(state: SharedState) -> str:
    # If researcher ran for analysis, feed result back into PM for replan
    return "pm_replan" if state.get("research_for_analysis") else "pm_execute"


def route_after_pm_replan(state: SharedState) -> str:
    return "researcher" if state.get("research_needed") else "pm_execute"


# =============================================================================
# Dump
# =============================================================================


def _build_graph_lines(
    result: "SharedState",
    registry: "AgentRegistry",
    run_id: str,
    elapsed: float,
    request: str = "",
) -> list[str]:
    """
    Build a list of plain-text lines rendering the pipeline as a vertical graph.

    Active nodes:  solid box  (┌─┐ / └─┘)
    Skipped nodes: dashed box (┌╌┐ / └╌┘)
    Workers:       nested box inside pm_execute
    Edges:         │ / ▼ arrows between boxes
    """
    # ── Box geometry ───────────────────────────────────────────────────────────
    OW = 60   # outer box: chars of content between │ borders (excl. 1-space padding)
    NW = 54   # nested worker box content width  (= OW - 6, fits exactly inside outer row)
    # Outer box total width:  2(indent) + 1(│) + 1(space) + OW + 1(space) + 1(│) = OW + 6 = 66
    # Nested box total width: 2(indent) + 1(│) + 1(space) + NW + 1(space) + 1(│) = NW + 6 = 60 = OW ✓

    def s_top(w: int = OW) -> str: return "  ┌" + "─" * (w + 2) + "┐"
    def s_bot(w: int = OW) -> str: return "  └" + "─" * (w + 2) + "┘"
    def d_top(w: int = OW) -> str: return "  ┌" + "╌" * (w + 2) + "┐"
    def d_bot(w: int = OW) -> str: return "  └" + "╌" * (w + 2) + "┘"

    def s_row(text: str, w: int = OW) -> str:
        return "  │ " + text.ljust(w) + " │"

    def d_row(text: str, w: int = OW) -> str:
        return "  ╎ " + text.ljust(w) + " ╎"

    # Arrow: centred on the outer box (outer box │ borders at col 2 and col 65)
    _ARROW_COL = 34
    ARROW = ["  " + " " * _ARROW_COL + "│", "  " + " " * _ARROW_COL + "▼"]

    # ── Data helpers ───────────────────────────────────────────────────────────
    ag_map       = {a.name: a for a in registry.all()}
    task_map     = {t["id"]: t for t in result.get("tasks", [])}
    task_results = result.get("task_results", [])
    pricing      = {m["id"]: m for m in load_models()}

    def _agent_cost(ag: "Agent") -> float | None:
        mid = ag.model.removeprefix("openai/")
        p = pricing.get(mid, {})
        inp, out = p.get("input_cost_per_1m"), p.get("output_cost_per_1m")
        return (ag.prompt_tokens * inp + ag.completion_tokens * out) / 1_000_000 if inp and out else None

    def _cost_str(v: float | None) -> str:
        return f"${v:.4f}" if v is not None else "—"

    def _ag_elapsed(ag: "Agent") -> str:
        if not ag._actions:
            return "—"
        return f"{sum(a.get('elapsed_s', 0) for a in ag._actions):.1f}s"

    total_tokens = sum(a.total_tokens for a in registry.all())
    total_cost   = sum(c for a in registry.all() if (c := _agent_cost(a)) is not None)

    # ── Node renderers ─────────────────────────────────────────────────────────

    def _title(sym: str, name: str, note: str = "") -> str:
        """Left-align name, right-align note within OW."""
        base = f"{sym}  {name}"
        if not note:
            return base
        pad = OW - len(base) - len(note)
        return base + " " * max(2, pad) + note

    def active_node(name: str, model: str = "", tokens: str = "",
                    time_: str = "", note: str = "") -> list[str]:
        rows = [s_top(), s_row(_title("✓", name, note))]
        if model and tokens:
            rows.append(s_row(f"   {model}  ·  {tokens} tok  ·  {time_}"))
        rows.append(s_bot())
        return rows

    def skipped_node(name: str) -> list[str]:
        return [d_top(), d_row(_title("—", name, "skipped")), d_bot()]

    def worker_box(name: str, model: str, tokens: str, time_: str,
                   tools: list[str], desc: str) -> list[str]:
        """Render a worker as a nested box (content width = NW = 60 chars total = OW)."""
        rows = [s_top(NW), s_row(f"✓  {name}", NW)]
        rows.append(s_row(f"   {model}  ·  {tokens} tok  ·  {time_}", NW))
        if tools:
            rows.append(s_row(f"   tools: {', '.join(tools)}", NW))
        if desc:
            rows.append(s_row(f"   {desc}", NW))
        rows.append(s_bot(NW))
        return rows

    # ── Output builder ─────────────────────────────────────────────────────────
    SEP = "  " + "═" * 66
    out: list[str] = [
        "",
        SEP,
        f"  RUN    {run_id}",
        f"  QUERY  {request}",
        f"  TOTAL  {total_tokens:,} tok  ·  {elapsed:.1f}s  ·  {_cost_str(total_cost if total_cost > 0 else None)}",
        SEP,
        "",
    ]

    pm         = ag_map.get("program_manager")
    researcher = ag_map.get("researcher")
    judge      = ag_map.get("judge")

    def add(node_lines: list[str]) -> None:
        """Append a node and an arrow below it."""
        out.extend(node_lines)
        out.extend(ARROW)
        out.append("")

    # leader_receive
    add(active_node("leader_receive", note="routing"))

    # pm_plan
    if pm and pm._actions:
        a0 = pm._actions[0]
        add(active_node("pm_plan", _fmt_model(pm.model), str(a0["tokens"]["total"]), f"{a0['elapsed_s']:.1f}s"))
    else:
        add(skipped_node("pm_plan"))

    # researcher
    if researcher and researcher._actions:
        a0 = researcher._actions[0]
        add(active_node("researcher", _fmt_model(researcher.model), str(a0["tokens"]["total"]), f"{a0['elapsed_s']:.1f}s"))
    else:
        add(skipped_node("researcher"))

    # pm_replan  (second action on program_manager, only if research_for_analysis ran)
    if pm and len(pm._actions) >= 2:
        a1 = pm._actions[1]
        add(active_node("pm_replan", _fmt_model(pm.model), str(a1["tokens"]["total"]), f"{a1['elapsed_s']:.1f}s"))
    else:
        add(skipped_node("pm_replan"))

    # pm_execute (outer box containing nested worker boxes)
    if task_results:
        pm_rows: list[str] = [
            s_top(),
            s_row(_title("✓", "pm_execute", f"{len(task_results)} worker(s)")),
            s_row(""),
        ]
        for tr in task_results:
            wag   = ag_map.get(tr["agent"])
            task  = task_map.get(tr["task_id"], {})
            # Prefer actually-called tools; fall back to PM-assigned list
            tools = tr.get("tools_used") or task.get("tools") or []
            wtok  = str(wag.total_tokens) if wag else "—"
            wtime = _ag_elapsed(wag)       if wag else "—"
            desc  = tr["description"][:50] + ("…" if len(tr["description"]) > 50 else "")
            status = "skipped" if tr.get("skipped") else ""
            name_display = tr["agent"] + (f"  [{status}]" if status else "")
            # Each worker_box line is exactly OW chars (NW+6=60=OW) → fits as outer row content
            for wline in worker_box(name_display, _fmt_model(tr["model"]), wtok, wtime, tools, desc):
                pm_rows.append(s_row(wline))
            pm_rows.append(s_row(""))
        pm_rows.append(s_bot())
        add(pm_rows)
    elif result.get("direct_answer"):
        add(active_node("pm_execute", note="direct answer — workers skipped"))
    else:
        add(skipped_node("pm_execute"))

    # judge_review
    if judge and judge._actions:
        a0 = judge._actions[0]
        add(active_node("judge_review", _fmt_model(judge.model), str(a0["tokens"]["total"]), f"{a0['elapsed_s']:.1f}s"))
    else:
        add(skipped_node("judge_review"))

    # leader_respond — last node, no arrow after
    out.extend(active_node("leader_respond", note="pass-through"))
    out += ["", SEP, ""]
    return out


def write_graph_dump(
    result: "SharedState",
    registry: "AgentRegistry",
    run_id: str,
    elapsed: float,
    request: str = "",
) -> Path:
    """Write the pipeline graph to dumps/graphs/<run_id>.txt."""
    graphs_dir = DUMPS_DIR / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    dump_path = graphs_dir / f"{run_id}.txt"
    dump_path.write_text("\n".join(_build_graph_lines(result, registry, run_id, elapsed, request)), encoding="utf-8")
    return dump_path


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
        "timestamp": datetime.now(timezone.utc).isoformat(),
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


def print_startup_table(
    mode: str,
    leader_model: str,
    judge_model: str,
    pm_model: str,
    res_model: str,
    tools_cfg: list[dict],
    researcher_tools_cfg: list[dict] | None = None,
    mcp_tools: list[dict] | None = None,
) -> None:
    """Single startup table: mode, agents, models, roles."""
    _section(f"AI Factory  ·  mode: {mode}", "blue")
    t = Table(border_style="dim", show_header=True, padding=(0, 1))
    t.add_column("Agent",        style="bold", no_wrap=True)
    t.add_column("Role",         style="dim")
    t.add_column("Model",        no_wrap=True)
    t.add_column("Selection",    style="dim")
    t.add_column("Human-facing", justify="center")
    for name, role, model, selection, human in [
        ("leader",          "Orchestrator + human I/O",          leader_model, "best_model()",  "[green]yes[/green]"),
        ("judge",           "Quality assessor",                  judge_model,  f"mode:{mode}",  "[green]yes[/green]"),
        ("program_manager", "Plan + spawn workers",              pm_model,     f"mode:{mode}",  "no"),
        ("researcher",      "Conditional research",              res_model,    f"mode:{mode}",  "no"),
        ("worker (×N)",     "Per-task, spawned at runtime",      "dynamic",    "per-task reqs", "no"),
    ]:
        t.add_row(name, role, f"[cyan]{model}[/cyan]", selection, human)
    t.add_section()
    if researcher_tools_cfg:
        t.add_row("[dim]researcher tools[/dim]", f"[dim]{', '.join(tc['id'] for tc in researcher_tools_cfg)}[/dim]", "", "", "")
    if tools_cfg:
        t.add_row("[dim]worker tools[/dim]", f"[dim]{', '.join(tc['id'] for tc in tools_cfg)}[/dim]", "", "", "")
    if mcp_tools:
        # Group by server for a readable summary
        by_server: dict[str, list[str]] = {}
        for mt in mcp_tools:
            by_server.setdefault(mt["server_id"], []).append(mt["name"])
        for sid, names in by_server.items():
            t.add_row(f"[dim]mcp:{sid}[/dim]", f"[dim]{', '.join(names)}[/dim]", "", "", "")
    console.print(t)


def _fmt_model(model: str) -> str:
    m = model.removeprefix("openai/")
    return m[-32:] if len(m) > 32 else m


def print_pipeline_graph(
    result: "SharedState",
    registry: "AgentRegistry",
    run_id: str,
    elapsed: float,
    request: str = "",
) -> None:
    """Print the pipeline graph to the console using plain Unicode box-drawing chars."""
    for line in _build_graph_lines(result, registry, run_id, elapsed, request):
        console.print(line)


def _model_for(agent_cfg: dict, models: list[dict], mode: str, fallback: str) -> str:
    """Select model for a named agent using its ability profile + mode bias."""
    abilities = agent_cfg.get("abilities")
    if not abilities:
        return fallback
    chosen, _ = select_model(abilities, models, mode)
    return chosen


def build_graph(
    cfg: dict,
    registry: AgentRegistry,
    run_id: str,
    memory: Memory,
    checkpointer=None,
    mcp_client: "McpClient | None" = None,
):
    fallback_model = cfg["model"]
    mode  = cfg.get("mode", "balanced")
    agent_cfgs = {n: load_agent_cfg(n) for n in ["leader", "judge", "program_manager", "researcher", "worker"]}
    models = load_models()
    tools_cfg = load_tools_cfg()
    researcher_tools_cfg = load_researcher_tools_cfg()
    mcp_tools = mcp_client.tools() if mcp_client else []

    # All tools available to workers = built-in + MCP (used for PM menu + ToolExecutor lookup)
    all_worker_tools = tools_cfg + mcp_tools

    # Inject compact tool menu into PM system prompt (id + one-line description)
    if all_worker_tools:
        tool_menu = "\n".join(
            f"  - {t['id']}: {t['description'].split(chr(46))[0]}."
            for t in all_worker_tools
        )
        agent_cfgs["program_manager"]["system_prompt"] = (
            agent_cfgs["program_manager"]["system_prompt"].rstrip()
            + f"\n\n## Available Tools\nAssign IDs to tasks that need them via the \"tools\" list.\n{tool_menu}"
        )
    # Leader always uses the best available model — mode has no effect
    leader_model = best_model(models)

    # All other agents: ability profile × mode bias
    judge_model = _model_for(agent_cfgs["judge"],           models, mode, fallback_model)
    pm_model    = _model_for(agent_cfgs["program_manager"], models, mode, fallback_model)
    res_model   = _model_for(agent_cfgs["researcher"],      models, mode, fallback_model)

    print_startup_table(mode, leader_model, judge_model, pm_model, res_model, tools_cfg, researcher_tools_cfg, mcp_tools or None)

    leader        = registry.register(Agent("leader",          {**agent_cfgs["leader"],          "model": leader_model}, leader_model))
    judge         = registry.register(Agent("judge",           {**agent_cfgs["judge"],           "model": judge_model},  judge_model))
    pm            = registry.register(Agent("program_manager", {**agent_cfgs["program_manager"], "model": pm_model},     pm_model))
    researcher_ag = registry.register(Agent("researcher",      {**agent_cfgs["researcher"],      "model": res_model},    res_model))

    g = StateGraph(SharedState)
    g.add_node("leader_receive", make_leader_receive(leader))
    g.add_node("pm_plan",        make_pm_plan(pm, memory))
    g.add_node("researcher",     make_researcher(researcher_ag, memory, run_id, researcher_tools_cfg))
    g.add_node("pm_replan",      make_pm_replan(pm))
    g.add_node("pm_execute",     make_pm_execute(pm, agent_cfgs["worker"], models, fallback_model, mode, registry, memory, researcher_ag, run_id, all_worker_tools, researcher_tools_cfg, mcp_client=mcp_client))
    g.add_node("judge_review",   make_judge_review(judge, registry, run_id))
    g.add_node("leader_respond", make_leader_respond(leader))

    g.add_edge(START, "leader_receive")
    g.add_edge("leader_receive", "pm_plan")
    g.add_conditional_edges("pm_plan", route_after_pm_plan, {
        "judge_review":   "judge_review",
        "researcher":     "researcher",
        "pm_execute":     "pm_execute",
    })
    g.add_conditional_edges("researcher", route_after_researcher, {
        "pm_replan":  "pm_replan",
        "pm_execute": "pm_execute",
    })
    g.add_conditional_edges("pm_replan", route_after_pm_replan, {
        "researcher": "researcher",
        "pm_execute": "pm_execute",
    })
    g.add_edge("pm_execute",   "judge_review")
    g.add_edge("judge_review", "leader_respond")
    g.add_edge("leader_respond", END)

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
    mode_colour = {"best": "green", "balanced": "yellow", "cost_efficient": "cyan"}.get(mode, "white")
    _mode_alpha = MODE_PROFILES.get(mode, MODE_PROFILES["balanced"])["alpha"]
    console.print(Panel(
        "[bold white]Multi-Agent LangGraph  ·  Chat Mode[/bold white]\n"
        "Leader  →  Judge  →  Program Manager  →  [Researcher?]  →  Workers\n"
        f"Mode: [{mode_colour}]{mode}[/{mode_colour}]  "
        f"[dim]alpha={_mode_alpha}   Type  quit / exit / done  to end.[/dim]",
        border_style="blue", expand=False,
    ))
    console.print()

    # ── Read request from terminal ────────────────────────────────────────────
    request = (await _async_input("What would you like to work on?\n[You] > ")).strip()
    console.print()
    if not request or request.lower() in QUIT_WORDS:
        console.print("[dim]Exiting.[/dim]")
        return

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    memory = Memory()
    checkpointer = MemorySaver()

    # MCP — connect to configured external servers (no-op if mcp_servers.yaml is empty)
    mcp_servers_cfg = load_mcp_servers_cfg()
    mcp_client: McpClient | None = None
    if mcp_servers_cfg:
        mcp_client = McpClient(mcp_servers_cfg)
        await mcp_client.connect()

    try:
        app = build_graph(cfg, registry, run_id, memory, checkpointer, mcp_client=mcp_client)
        thread_cfg: dict = {"configurable": {"thread_id": run_id}}

        # STM is seeded from LTM facts at Memory() init — log what was loaded
        if memory.stm.all():
            _log("memory", "facts loaded", f"[magenta]{list(memory.stm.all().keys())}[/magenta]")

        console.print(Panel(f"[bold]Human Request:[/bold]\n\n{request}", border_style="yellow", expand=False))
        console.print()

        # ── Run the pipeline, pausing whenever a worker asks a question ──────
        t0 = time.perf_counter()
        current_input: Any = {
            "user_query": request,
            "shared_context": memory.stm.all(),  # LTM facts seeded into STM at Memory() init
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

            # ainvoke returns {**state, "__interrupt__": [Interrupt(...)]} when a node
            # calls interrupt().  This is the authoritative signal — more reliable than
            # snap.next which can be empty on the second interrupt within the same resume
            # invocation (LangGraph 1.1.x behaviour).
            pending_interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
            if not pending_interrupts:
                break  # pipeline complete — no pending interrupt

            interrupt_msg = str(pending_interrupts[0].value) if pending_interrupts else ""

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
                graph_path = write_graph_dump(partial, registry, run_id, elapsed, request)
                _log("dump", "graph (partial)", f"[cyan]{graph_path}[/cyan]")
                return
            current_input = Command(resume=answer)
        elapsed = time.perf_counter() - t0
        print_pipeline_graph(result, registry, run_id, elapsed, request)

        # ── Chat loop — talk directly to leader agent ─────────────────────────
        # The graph is done. Leader already has full private context (_messages).
        # We continue the conversation by calling leader.think() directly.
        leader = registry.get("leader")
        shared_ctx = result.get("shared_context", {})
        history: list[dict] = list(result.get("conversation_history", []))

        # Surface judge quality note if set
        judge_note = result.get("judge_human_message", "").strip()
        if judge_note:
            console.print(Panel(f"[bold yellow]Judge note:[/bold yellow] {judge_note}", border_style="yellow", expand=False))
            console.print()

        _section("Leader  ·  Response", "green")
        final_response = result.get("final_response", "")
        if final_response and final_response.strip():
            console.print(Panel(final_response, border_style="green", expand=False))
        else:
            console.print("[yellow]⚠ No final response generated.[/yellow]")
        console.print()

        while leader is not None:
            user_input = (await _async_input("[You] > ")).strip()
            console.print()

            if not user_input or user_input.lower() in QUIT_WORDS:
                break

            history.append({"role": "user", "content": user_input})
            reply = await leader.think(
                f"The human replied: \"{user_input}\"\n\nRespond helpfully.",
                shared_ctx,
            )
            reply_clean, persistent = extract_remember(reply)
            if persistent:
                memory.ltm.save_facts(persistent)
                memory.stm.merge(persistent)
                shared_ctx = {**shared_ctx, **persistent}
                leader._messages[-1]["content"] = reply_clean
                reply = reply_clean
                _log("leader", "→ memory", f"[magenta]{list(persistent.keys())}[/magenta]")
            history.append({"role": "assistant", "content": reply})

            _section("Leader", "green")
            console.print(Panel(reply, border_style="green", expand=False))
            console.print()

        result["conversation_history"] = history

        dump_path = write_dump(run_id, cfg, result, registry, elapsed, request)
        _log("dump", "saved", f"[cyan]{dump_path}[/cyan]")
        graph_path = write_graph_dump(result, registry, run_id, elapsed, request)
        _log("dump", "graph", f"[cyan]{graph_path}[/cyan]")

    finally:
        if mcp_client:
            await mcp_client.close()



if __name__ == "__main__":
    asyncio.run(main())
