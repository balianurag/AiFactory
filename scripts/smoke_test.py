#!/usr/bin/env python3
"""
smoke_test.py — AiFactory end-to-end functional smoke test.

Drives every pipeline layer systematically in cost_efficient mode + DEPLOY_MODE=local:
  [0] Environment probe — API keys, Docker MCP pool health, Qdrant, Redis
  [1] ComplexityGate routing — 4 tiers: math, unit, fact, full pipeline
  [2] Model selection + specialist role detection (cost_efficient mode)
  [3] Sandbox code execution — Docker MCP backend (falls back to local)
  [3b] Docker MCP pool — pool_status, warm containers, parallel execution
  [4] Memory layers — agent store / retrieve / promote → team (persistent Qdrant)
  [5] RL scorer — record performances, retrieve scores, ordering check
  [6] SHARE: protocol — extract, merge, context injection
  [7] Dependency wave construction — no-dep, linear, fan-out, diamond, circular
  [8] Full pipeline run — real LLM + Docker sandbox + persistent Qdrant memory
      · token tracking per agent
      · model selection diversity
      · judge metrics + RL records written
      · memory layer counts post-run (persistent across MemoryRouter instances)

Usage:
    uv run python scripts/smoke_test.py               # all sections
    uv run python scripts/smoke_test.py --skip-llm   # skip section 8
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import UTC
from pathlib import Path

# ── repo root on path ────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

# ── probe Docker MCP server BEFORE aifactory imports so settings load correctly
_MCP_URL = os.environ.get("AIFACTORY_MCP_SANDBOX_URL", "http://localhost:8100")
_DOCKER_AVAILABLE = False
_MCP_POOL_STATUS: dict = {}
try:
    with urllib.request.urlopen(f"{_MCP_URL}/health", timeout=2) as _r:
        import json as _json

        _MCP_POOL_STATUS = _json.loads(_r.read())
        _DOCKER_AVAILABLE = _MCP_POOL_STATUS.get("status") == "ok"
except Exception:
    pass

# ── force overrides BEFORE any aifactory imports ─────────────────────────────
os.environ["AIFACTORY_DEPLOY_MODE"] = "local"
# Use Docker sandbox only when MCP server is actually reachable
os.environ["AIFACTORY_SANDBOX_DOCKER_ENABLED"] = "true" if _DOCKER_AVAILABLE else "false"
os.environ["AIFACTORY_LLAMAINDEX_EMBEDDING_PROVIDER"] = "stub"  # SHA-256 embeddings, no API
os.environ["AIFACTORY_RL_MIN_SAMPLES"] = "1"  # lower threshold so smoke records show up

# ── terminal colours ─────────────────────────────────────────────────────────
_T = sys.stdout.isatty()
RESET = "\033[0m" if _T else ""
GREEN = "\033[32m" if _T else ""
RED = "\033[31m" if _T else ""
YELLOW = "\033[33m" if _T else ""
CYAN = "\033[36m" if _T else ""
BOLD = "\033[1m" if _T else ""
DIM = "\033[2m" if _T else ""


def hdr(n: int | str, title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─' * 64}{RESET}")
    print(f"{BOLD}{CYAN}[{n}] {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 64}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {DIM}·{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {msg}")


# ── result accumulator ───────────────────────────────────────────────────────
_results: dict[str, bool] = {}


def chk(label: str, cond: bool, msg: str) -> bool:
    _results[f"[{label}] {msg}"] = cond
    (ok if cond else fail)(msg)
    return cond


# =============================================================================
# SECTION 0 — Environment probe
# =============================================================================


def _probe_http(url: str, timeout: int = 2) -> bool:
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


def probe_env() -> dict[str, bool]:
    hdr(0, "Environment Probe")
    from aifactory.config.settings import get_settings

    s = get_settings()

    # ── API keys ─────────────────────────────────────────────────────────────
    keys: dict[str, bool] = {
        "openai": bool(s.openai_api_key),
        "anthropic": bool(s.anthropic_api_key),
        "openrouter": bool(s.openrouter_api_key),
        "e2b": bool(s.e2b_api_key),
        "composio": bool(s.composio_api_key),
    }
    for name, present in keys.items():
        (ok if present else warn)(f"{name.upper()}_API_KEY {'set' if present else 'not set'}")

    # ── Infrastructure services ───────────────────────────────────────────────
    print()
    qdrant_live = _probe_http(f"{s.qdrant_url}/collections")
    (ok if qdrant_live else warn)(
        f"Qdrant @ {s.qdrant_url} {'LIVE' if qdrant_live else 'unreachable — in-memory fallback'}"
    )

    redis_live = False
    try:
        import socket

        sock = socket.create_connection(("localhost", 6379), timeout=1)
        sock.close()
        redis_live = True
    except Exception:
        pass
    (ok if redis_live else warn)(
        f"Redis @ localhost:6379 {'LIVE' if redis_live else 'unreachable'}"
    )

    if _DOCKER_AVAILABLE:
        pool = _MCP_POOL_STATUS.get("pool", {})
        ok(
            f"Docker MCP server @ {_MCP_URL}  LIVE — pool: {pool.get('ready', 0)} ready / "
            f"{pool.get('total', 0)} total  image={pool.get('image', '?')}"
        )
    else:
        warn(f"Docker MCP server @ {_MCP_URL}  unreachable — sandbox section uses local subprocess")

    # ── Settings summary ──────────────────────────────────────────────────────
    print()
    info(f"deploy_mode                 = {s.deploy_mode}")
    info(
        f"sandbox_docker_enabled      = {os.environ['AIFACTORY_SANDBOX_DOCKER_ENABLED']} (auto-detected)"  # noqa: E501
    )
    info(f"sandbox_backend             = {'docker' if _DOCKER_AVAILABLE else 'local'}")
    info(f"embedding_provider          = {s.llamaindex_embedding_provider}")
    info(f"rl_enabled                  = {s.rl_enabled}")
    info(f"memory_pipeline_enabled     = {s.memory_pipeline_enabled}")
    info(f"prompt_cache_enabled        = {s.prompt_cache_enabled}")
    info(f"judge_skip_simple_runs      = {s.judge_skip_simple_runs}")

    has_llm = keys["openai"] or keys["anthropic"] or keys["openrouter"]
    print()
    if has_llm:
        ok("LLM provider available — section [8] full pipeline run ENABLED")
    else:
        warn("No LLM API key — section [8] will be skipped")

    return {**keys, "qdrant": qdrant_live, "redis": redis_live, "docker_mcp": _DOCKER_AVAILABLE}


# =============================================================================
# SECTION 1 — ComplexityGate routing (4 tiers)
# =============================================================================


def test_complexity_gate() -> bool:
    hdr(1, "ComplexityGate Routing (4 tiers)")
    from aifactory.core.complexity_gate import ComplexityGate

    gate = ComplexityGate()
    all_ok = True

    # ── Tier 1: pure math AST eval ───────────────────────────────────────────
    d = gate.evaluate("2 + 2")
    all_ok &= chk(
        "gate",
        d.bypass and "4" in d.answer,
        f"math '2+2' → bypass={d.bypass} answer='{d.answer}' method='{d.method}'",
    )

    d = gate.evaluate("15 * 7")
    all_ok &= chk(
        "gate",
        d.bypass and "105" in d.answer,
        f"math '15*7' → bypass={d.bypass} answer='{d.answer}'",
    )

    d = gate.evaluate("(100 + 200) / 3")
    all_ok &= chk(
        "gate",
        d.bypass and "100" in d.answer,
        f"math '(100+200)/3' → bypass={d.bypass} answer='{d.answer}'",
    )

    # ── Tier 2: unit conversion ──────────────────────────────────────────────
    d = gate.evaluate("100 celsius to fahrenheit")
    all_ok &= chk("gate", d.bypass, f"unit '100C→F' → bypass={d.bypass} answer='{d.answer}'")

    d = gate.evaluate("5 miles to kilometres")
    info(f"  unit '5 miles→km' → bypass={d.bypass} answer='{d.answer}' method='{d.method}'")

    # ── Tier 3: static fact ──────────────────────────────────────────────────
    d = gate.evaluate("capital of France")
    all_ok &= chk(
        "gate",
        d.bypass and "Paris" in d.answer,
        f"fact 'capital of France' → bypass={d.bypass} answer='{d.answer}'",
    )

    d = gate.evaluate("how many days in a year")
    info(f"  fact 'days in a year' → bypass={d.bypass} answer='{d.answer}' method='{d.method}'")

    # ── Tier 4: full pipeline (must NOT bypass) ──────────────────────────────
    d = gate.evaluate(
        "Build a production FastAPI REST API with OAuth2 authentication, "
        "PostgreSQL backend, Alembic migrations and Docker deployment"
    )
    all_ok &= chk(
        "gate",
        not d.bypass and not d.fast_llm,
        f"complex task → full pipeline: bypass={d.bypass} fast_llm={d.fast_llm}",
    )

    # ── fast_llm tier (simple factual Q&A) ──────────────────────────────────
    d = gate.evaluate("Who invented the telephone?")
    info(
        f"  Q&A 'Who invented the telephone?' → bypass={d.bypass} fast_llm={d.fast_llm} method='{d.method}'"  # noqa: E501
    )

    return all_ok


# =============================================================================
# SECTION 2 — Model selection + specialist routing (cost_efficient)
# =============================================================================


def test_model_selection() -> bool:
    hdr(2, "Model Selection + Specialist Routing (cost_efficient)")
    from unittest.mock import MagicMock, patch

    from aifactory.config.settings import get_settings
    from aifactory.core.agent_factory import create_worker_config

    s = get_settings()
    all_ok = True

    def _cfg(desc: str) -> dict:
        """Build worker config with no live integrations."""
        mock_reg = MagicMock()
        mock_reg.get_connected_toolkit_names.return_value = []
        with patch("aifactory.integrations.registry.IntegrationRegistry", return_value=mock_reg):
            return create_worker_config(
                task_id="smoke_t0",
                task_description=desc,
                worker_prompt=desc,
                mode="cost_efficient",
            )

    # coding task → worker_coder.yaml
    cfg = _cfg("Write a Python function to sort a list using quicksort algorithm")
    has_coder = any(
        kw in cfg["system_prompt"].lower()
        for kw in ["software engineer", "algorithm", "big-o", "coding excellence"]
    )
    all_ok &= chk(
        "model",
        has_coder,
        f"coding → coder specialist (model={cfg['model']}, score={cfg['model_score']:.3f})",
    )

    # research task → worker_researcher.yaml
    cfg = _cfg("Research the latest advances in transformer architecture for NLP")
    has_researcher = any(
        kw in cfg["system_prompt"].lower()
        for kw in ["research", "synthesis", "researcher", "knowledge"]
    )
    all_ok &= chk(
        "model", has_researcher, f"research → researcher specialist (model={cfg['model']})"
    )

    # writing task → worker_writer.yaml
    cfg = _cfg("Write a technical blog post about vector databases for developers")
    has_writer = any(
        kw in cfg["system_prompt"].lower()
        for kw in ["writer", "writing", "tone", "audience", "communication"]
    )
    all_ok &= chk("model", has_writer, f"writing → writer specialist (model={cfg['model']})")

    # analysis task → worker_analyst.yaml  (avoid coding keywords like "sort")
    cfg = _cfg(
        "Evaluate the trade-offs and pros and cons between monolith and microservices architecture"
    )
    has_analyst = any(
        kw in cfg["system_prompt"].lower()
        for kw in ["analyst", "analysis", "evidence", "reasoning", "structured"]
    )
    all_ok &= chk("model", has_analyst, f"analysis → analyst specialist (model={cfg['model']})")

    # cost_efficient model preference
    cfg = _cfg("Write a function to reverse a string")
    budget = s.budget_model
    info(f"cost_efficient selected: {cfg['model']}  (budget_model={budget})")
    info(f"QUESTION PROTOCOL present: {'QUESTION PROTOCOL:' in cfg['system_prompt']}")
    info(f"NOT_NEEDED present:        {'NOT_NEEDED' in cfg['system_prompt']}")

    # verify key prompt sections always present
    all_ok &= chk(
        "model",
        "QUESTION PROTOCOL:" in cfg["system_prompt"],
        "system_prompt contains QUESTION PROTOCOL block",
    )
    all_ok &= chk(
        "model",
        "NOT_NEEDED" in cfg["system_prompt"],
        "system_prompt contains NOT_NEEDED skip instruction",
    )

    return all_ok


# =============================================================================
# SECTION 3 — Sandbox code execution (local backend)
# =============================================================================


async def test_sandbox() -> bool:
    backend = "docker" if _DOCKER_AVAILABLE else "local"
    hdr(3, f"Sandbox Code Execution ({backend} backend)")
    from aifactory.core.sandbox import execute, execute_code_blocks, extract_code_blocks

    all_ok = True

    if _DOCKER_AVAILABLE:
        info(
            f"Docker MCP pool: {_MCP_POOL_STATUS.get('pool', {}).get('ready', '?')} warm containers ready"  # noqa: E501
        )

    # ── basic arithmetic ─────────────────────────────────────────────────────
    r = await execute("print(sum(range(1, 11)))", "python")
    all_ok &= chk(
        "sandbox",
        r.success and "55" in r.output,
        f"sum(1..10)=55 → success={r.success} output='{r.output.strip()}' backend={r.backend}",
    )

    # ── stdlib import ────────────────────────────────────────────────────────
    r = await execute("import math\nprint(round(math.pi, 4))", "python")
    all_ok &= chk(
        "sandbox",
        r.success and "3.1416" in r.output,
        f"math.pi → success={r.success} output='{r.output.strip()}' backend={r.backend}",
    )

    # ── multi-statement function ─────────────────────────────────────────────
    code = (
        "def fib(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n): a, b = b, a + b\n"
        "    return a\n"
        "print(fib(10))"
    )
    r = await execute(code, "python")
    all_ok &= chk(
        "sandbox",
        r.success and "55" in r.output,
        f"fibonacci(10)=55 → success={r.success} output='{r.output.strip()}' backend={r.backend}",
    )

    # ── error propagation ────────────────────────────────────────────────────
    r = await execute("1 / 0", "python")
    captured = not r.success or "ZeroDivisionError" in r.output
    all_ok &= chk(
        "sandbox", captured, f"ZeroDivisionError → success={r.success} backend={r.backend}"
    )

    # ── code block extraction from LLM text ──────────────────────────────────
    llm_text = "Here is the solution:\n```python\nprint('hello world')\n```\nDone."
    blocks = extract_code_blocks(llm_text)
    all_ok &= chk(
        "sandbox",
        len(blocks) == 1 and blocks[0][0] == "python",
        f"extract_code_blocks → {len(blocks)} block(s), lang='{blocks[0][0] if blocks else 'none'}'",  # noqa: E501
    )

    # ── execute_code_blocks (end-to-end, backend auto-selected) ──────────────
    results = await execute_code_blocks("```python\nprint(2**10)\n```")
    got_1024 = len(results) == 1 and results[0][2].success and "1024" in results[0][2].output
    all_ok &= chk(
        "sandbox",
        got_1024,
        f"execute_code_blocks 2**10=1024 → success={results[0][2].success if results else False} "
        f"backend={results[0][2].backend if results else '?'}",
    )

    # ── Big-O style timing code ───────────────────────────────────────────────
    timing_code = (
        "import time\n"
        "data = list(range(1000, 0, -1))\n"
        "t = time.perf_counter()\n"
        "data.sort()\n"
        "elapsed = time.perf_counter() - t\n"
        "print(f'sorted={data[0]}..{data[-1]} in {elapsed*1000:.2f}ms')"
    )
    r = await execute(timing_code, "python")
    all_ok &= chk(
        "sandbox",
        r.success and "sorted=1..1000" in r.output,
        f"timing sort → success={r.success} backend={r.backend}",
    )

    return all_ok


async def test_docker_mcp() -> bool:
    hdr("3b", "Docker MCP Sandbox Pool")

    if not _DOCKER_AVAILABLE:
        warn("Docker MCP not available — skipping Docker MCP section")
        return True

    import json as _json
    import urllib.request

    from aifactory.core.sandbox import execute

    all_ok = True

    # ── pool health via HTTP ─────────────────────────────────────────────────
    with urllib.request.urlopen(f"{_MCP_URL}/health", timeout=5) as resp:
        status = _json.loads(resp.read())
    pool = status.get("pool", {})
    all_ok &= chk(
        "docker",
        pool.get("ready", 0) >= 1,
        f"pool health: {pool.get('ready')} ready / {pool.get('total')} total  "
        f"utilisation={pool.get('utilization_pct', 0):.1f}%",
    )
    all_ok &= chk(
        "docker",
        pool.get("image", "").startswith("aifactory-sandbox"),
        f"pool image: {pool.get('image')}",
    )

    # ── single execution via Docker ──────────────────────────────────────────
    r = await execute("import sys; print(sys.version.split()[0])", "python", backend="docker")
    all_ok &= chk(
        "docker",
        r.success and r.backend == "docker",
        f"docker python exec → success={r.success} backend={r.backend} "
        f"python={r.output.strip()[:10]}",
    )

    # ── numpy available in sandbox image ─────────────────────────────────────
    r = await execute(
        "import numpy as np; print(np.sum(np.arange(1, 11)))", "python", backend="docker"
    )
    all_ok &= chk(
        "docker",
        r.success and "55" in r.output,
        f"numpy in sandbox → success={r.success} output='{r.output.strip()}'",
    )

    # ── pandas available ──────────────────────────────────────────────────────
    r = await execute(
        "import pandas as pd\n"
        "df = pd.DataFrame({'x': [1,2,3], 'y': [4,5,6]})\n"
        "print(df['x'].sum())",
        "python",
        backend="docker",
    )
    all_ok &= chk(
        "docker",
        r.success and "6" in r.output,
        f"pandas in sandbox → success={r.success} output='{r.output.strip()}'",
    )

    # ── no network access inside container (security) ────────────────────────
    r = await execute(
        "import urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('http://example.com', timeout=2)\n"
        "    print('NETWORK_OK')\n"
        "except Exception as e:\n"
        "    print(f'NETWORK_BLOCKED: {type(e).__name__}')",
        "python",
        backend="docker",
    )
    blocked = r.success and "NETWORK_OK" not in r.output
    all_ok &= chk("docker", blocked, f"network isolation → {r.output.strip()[:60]}")

    # ── parallel execution (2 containers at once) ────────────────────────────
    code_a = "import time; time.sleep(0.1); print('worker_A done')"
    code_b = "import time; time.sleep(0.1); print('worker_B done')"
    t0 = time.perf_counter()
    ra, rb = await asyncio.gather(
        execute(code_a, "python", backend="docker"),
        execute(code_b, "python", backend="docker"),
    )
    parallel_elapsed = time.perf_counter() - t0
    both_ok = ra.success and rb.success
    # Parallel should finish much faster than sequential (0.2s each = 0.4s total)
    is_parallel = parallel_elapsed < 0.35
    all_ok &= chk(
        "docker",
        both_ok,
        f"parallel docker exec → A={ra.success} B={rb.success} "
        f"elapsed={parallel_elapsed:.2f}s {'(parallel ✓)' if is_parallel else '(sequential?)'}",
    )

    # ── pool still healthy after executions ──────────────────────────────────
    with urllib.request.urlopen(f"{_MCP_URL}/health", timeout=5) as resp:
        after = _json.loads(resp.read()).get("pool", {})
    all_ok &= chk(
        "docker",
        after.get("ready", 0) >= 1,
        f"pool still healthy after use: {after.get('ready')} ready / {after.get('total')} total",
    )

    return all_ok


# =============================================================================
# SECTION 4 — Memory layer store / retrieve / promote
# =============================================================================


def test_memory() -> bool:
    hdr(4, "Memory Layers (agent → team → global)")
    from aifactory.core.memory_router import MemoryRouter
    from aifactory.core.schemas import MemoryEntry, MemoryLevel

    all_ok = True

    router = MemoryRouter()
    agent_id = f"smoke_agent_{int(time.time())}"
    team_id = f"smoke_team_{int(time.time())}"

    # ── store at AGENT level ─────────────────────────────────────────────────
    entry = MemoryEntry(
        level=MemoryLevel.AGENT,
        agent_id=agent_id,
        team_id=team_id,
        content="The task requires building a REST API with FastAPI and PostgreSQL.",
        entry_type="observation",
        metadata={"run_id": "smoke_test"},
    )
    eid = router.store(entry)
    all_ok &= chk(
        "memory", bool(eid), f"store AGENT-level entry → id='{eid[:16]}...' (stub embeddings)"
    )

    # ── retrieve from AGENT level ────────────────────────────────────────────
    hits = router.retrieve(
        "FastAPI REST API database",
        levels=[MemoryLevel.AGENT],
        agent_id=agent_id,
        top_k=5,
    )
    all_ok &= chk("memory", len(hits) >= 1, f"retrieve AGENT-level → {len(hits)} hit(s)")
    if hits:
        info(f"  top hit content: '{hits[0].content[:80]}'")

    # ── store a SHARE: decision at TEAM level ────────────────────────────────
    team_entry = MemoryEntry(
        level=MemoryLevel.TEAM,
        agent_id=agent_id,
        team_id=team_id,
        content="db_schema: users(id,email,password_hash), posts(id,user_id,title,body)",
        entry_type="decision",
        metadata={"run_id": "smoke_test"},
    )
    eid2 = router.store(team_entry)
    all_ok &= chk("memory", bool(eid2), f"store TEAM-level decision → id='{eid2[:16]}...'")

    # ── retrieve from TEAM level ─────────────────────────────────────────────
    team_hits = router.retrieve(
        "database schema users",
        levels=[MemoryLevel.TEAM],
        team_id=team_id,
        top_k=5,
    )
    all_ok &= chk("memory", len(team_hits) >= 1, f"retrieve TEAM-level → {len(team_hits)} hit(s)")

    # ── promote AGENT → TEAM ─────────────────────────────────────────────────
    promo_id = router.promote(
        agent_id=agent_id,
        team_id=team_id,
        summary="Smoke agent established FastAPI+PostgreSQL REST API context.",
        from_level=MemoryLevel.AGENT,
        to_level=MemoryLevel.TEAM,
    )
    all_ok &= chk("memory", bool(promo_id), f"promote AGENT→TEAM → id='{promo_id[:16]}...'")

    # ── retrieve promoted summary ────────────────────────────────────────────
    promo_hits = router.retrieve(
        "FastAPI PostgreSQL REST API",
        levels=[MemoryLevel.TEAM],
        team_id=team_id,
        top_k=10,
        entry_type="promoted_summary",
    )
    all_ok &= chk(
        "memory", len(promo_hits) >= 1, f"retrieve promoted_summary → {len(promo_hits)} hit(s)"
    )

    # ── cross-level retrieve (all levels) ───────────────────────────────────
    all_hits = router.retrieve(
        "REST API FastAPI",
        top_k=20,
    )
    all_ok &= chk(
        "memory",
        len(all_hits) >= 2,
        f"cross-level retrieve → {len(all_hits)} total hit(s) across all layers",
    )

    # ── persistence: store in one router, retrieve from fresh instance ───────
    qdrant_live = _probe_http("http://localhost:6333/collections")
    if qdrant_live:
        persist_key = f"smoke_persist_{int(time.time())}"
        router1 = MemoryRouter()
        router1.store(
            MemoryEntry(
                level=MemoryLevel.AGENT,
                agent_id=persist_key,
                team_id=persist_key,
                content=f"Persistent smoke test entry — key={persist_key}",
                entry_type="smoke_test",
            )
        )
        # Retrieve from a brand-new MemoryRouter instance (real Qdrant)
        router2 = MemoryRouter()
        hits2 = router2.retrieve(
            f"key={persist_key}",
            levels=[MemoryLevel.AGENT],
            agent_id=persist_key,
            top_k=5,
        )
        all_ok &= chk(
            "memory",
            len(hits2) >= 1,
            f"Qdrant persistence: stored via router1, retrieved via fresh router2 → {len(hits2)} hit(s)",  # noqa: E501
        )
    else:
        info("Qdrant not live — skipping persistence test (in-memory only)")

    return all_ok


# =============================================================================
# SECTION 5 — RL scorer record / retrieve cycle
# =============================================================================


def test_rl_scorer() -> bool:
    hdr(5, "RL Scorer — Record + Retrieve Cycle")
    from aifactory.core.rl_scorer import get_model_rl_scores, infer_task_type, record_performance

    all_ok = True

    # ── task type inference ──────────────────────────────────────────────────
    inference_cases = [
        ("Write a Python quicksort algorithm", "coding"),
        ("Research transformer architecture papers", "research"),
        ("Evaluate trade-offs and benchmark microservices vs monolith", "analysis"),
        ("Write a blog post about AI trends", "writing"),
        ("Calculate the derivative of x squared", "math"),
    ]
    for desc, expected in inference_cases:
        got = infer_task_type(desc)
        all_ok &= chk(
            "rl",
            got == expected,
            f"infer_task_type('{desc[:45]}') → '{got}' (expected '{expected}')",
        )

    # ── record 3 performances (rl_min_samples=1 from env) ───────────────────
    run_tag = f"smoke_{int(time.time())}"
    for score in [7.0, 8.5, 9.0]:
        record_performance(
            model_id="gpt-4o-mini",
            task_type="coding",
            quality_score=score,
            tokens_used=400,
            run_id=run_tag,
        )

    # ── retrieve RL scores ───────────────────────────────────────────────────
    scores = get_model_rl_scores("coding", min_samples=1)
    all_ok &= chk("rl", "gpt-4o-mini" in scores, f"get_model_rl_scores('coding') → {scores}")

    if "gpt-4o-mini" in scores:
        avg = scores["gpt-4o-mini"]
        expected = (7.0 + 8.5 + 9.0) / 3
        # Allow ±2.0 — there may be prior records from previous runs
        all_ok &= chk(
            "rl", abs(avg - expected) < 3.0, f"avg quality ≈ {expected:.2f}, got {avg:.2f}"
        )
        info(f"  gpt-4o-mini coding score: {avg:.2f}/10")

    # ── record a poor model for comparison ──────────────────────────────────
    record_performance("gpt-3.5-turbo", "coding", 3.5, run_id=run_tag)
    scores2 = get_model_rl_scores("coding", min_samples=1)
    if "gpt-3.5-turbo" in scores2 and "gpt-4o-mini" in scores2:
        ordered_ok = scores2["gpt-4o-mini"] > scores2["gpt-3.5-turbo"]
        all_ok &= chk(
            "rl",
            ordered_ok,
            f"gpt-4o-mini ({scores2['gpt-4o-mini']:.2f}) > gpt-3.5-turbo ({scores2['gpt-3.5-turbo']:.2f})",  # noqa: E501
        )

    # ── record and retrieve for research type ───────────────────────────────
    for s in [6.0, 7.5]:
        record_performance("claude-haiku-4-5-20251001", "research", s, run_id=run_tag)
    r_scores = get_model_rl_scores("research", min_samples=1)
    info(f"  research RL scores: {r_scores}")

    return all_ok


# =============================================================================
# SECTION 6 — SHARE: protocol extraction + merge + context injection
# =============================================================================


def test_share_protocol() -> bool:
    hdr(6, "SHARE: Protocol — Extract, Merge, Context Injection")
    from aifactory.core.agent_factory import build_shared_context_block, extract_share, merge_share

    all_ok = True

    # ── basic multi-field extraction (block format: \nSHARE:\nkey: val) ────────
    text = (
        "Analysis complete.\n"
        "SHARE:\n"
        "topic: transformer architecture\n"
        "depth: comprehensive\n"
        "key_finding: attention is all you need"
    )
    result, updates = extract_share(text)
    all_ok &= chk(
        "share",
        result.strip() == "Analysis complete.",
        f"extract text portion → '{result.strip()}'",
    )
    expected = {
        "topic": "transformer architecture",
        "depth": "comprehensive",
        "key_finding": "attention is all you need",
    }
    all_ok &= chk("share", updates == expected, f"extract updates → {updates}")

    # ── merge into existing context ──────────────────────────────────────────
    ctx = {"existing": "value", "objective": "build REST API"}
    merged = merge_share(ctx, updates, "analyser")
    all_ok &= chk(
        "share",
        "topic" in merged and "existing" in merged and "objective" in merged,
        f"merge preserves existing + adds new keys: {list(merged.keys())}",
    )

    # ── no SHARE: block → full text returned ────────────────────────────────
    r2, u2 = extract_share("Just a plain response with no share blocks.")
    all_ok &= chk(
        "share",
        r2 == "Just a plain response with no share blocks." and u2 == {},
        "no SHARE: block → full text returned, empty updates dict",
    )

    # ── SHARE: block amid other content (block terminates at end) ────────────
    text3 = "The API requires authentication.\nSHARE:\nauth_method: JWT tokens\ndb: PostgreSQL"
    _, u3 = extract_share(text3)
    all_ok &= chk(
        "share",
        u3.get("auth_method") == "JWT tokens",
        f"SHARE: block → auth_method='{u3.get('auth_method')}'",
    )

    # ── shared context block build ───────────────────────────────────────────
    ctx_block = build_shared_context_block(
        {
            "objective": "Build a REST API",
            "tech_stack": "FastAPI + PostgreSQL",
            "auth_method": "JWT tokens",
        }
    )
    all_ok &= chk(
        "share",
        "objective" in ctx_block and "tech_stack" in ctx_block,
        f"build_shared_context_block → {len(ctx_block)} chars, contains expected keys",
    )
    info(f"  context block preview: {ctx_block[:120].strip()!r}...")

    # ── SHARE: block with colon in value ────────────────────────────────────
    text4 = "Done.\nSHARE:\nendpoint: POST /api/v1/users"
    _, u4 = extract_share(text4)
    all_ok &= chk(
        "share",
        "endpoint" in u4 and "/api/v1/users" in u4.get("endpoint", ""),
        f"SHARE: block with colon in value → endpoint='{u4.get('endpoint')}'",
    )

    return all_ok


# =============================================================================
# SECTION 7 — Dependency wave construction
# =============================================================================


def test_dependency_waves() -> bool:
    hdr(7, "Dependency Wave Construction (_build_dependency_waves)")
    from aifactory.graphs.dynamic_pipeline import _build_dependency_waves

    all_ok = True

    def t(tid: str) -> dict:
        return {"id": tid, "description": f"task {tid}"}

    def ids(waves: list[list[dict]]) -> list[list[str]]:
        return [[x["id"] for x in w] for w in waves]

    # ── no deps → single parallel wave ──────────────────────────────────────
    w = _build_dependency_waves([t("a"), t("b"), t("c")], {})
    all_ok &= chk("waves", len(w) == 1 and len(w[0]) == 3, f"no deps → 1 wave of 3: {ids(w)}")

    # ── linear chain a→b→c ───────────────────────────────────────────────────
    w = _build_dependency_waves([t("a"), t("b"), t("c")], {"b": ["a"], "c": ["b"]})
    all_ok &= chk(
        "waves", len(w) == 3 and w[0][0]["id"] == "a", f"linear a→b→c → {len(w)} waves: {ids(w)}"
    )

    # ── fan-out a→(b, c) ─────────────────────────────────────────────────────
    w = _build_dependency_waves([t("a"), t("b"), t("c")], {"b": ["a"], "c": ["a"]})
    all_ok &= chk(
        "waves",
        len(w) == 2 and {x["id"] for x in w[1]} == {"b", "c"},
        f"fan-out a→(b,c) → {len(w)} waves: {ids(w)}",
    )

    # ── diamond a→(b,c)→d ────────────────────────────────────────────────────
    w = _build_dependency_waves(
        [t("a"), t("b"), t("c"), t("d")],
        {"b": ["a"], "c": ["a"], "d": ["b", "c"]},
    )
    all_ok &= chk(
        "waves",
        len(w) == 3 and {x["id"] for x in w[1]} == {"b", "c"} and w[2][0]["id"] == "d",
        f"diamond a→(b,c)→d → {len(w)} waves: {ids(w)}",
    )

    # ── single task, no deps ──────────────────────────────────────────────────
    w = _build_dependency_waves([t("solo")], {})
    all_ok &= chk(
        "waves", len(w) == 1 and w[0][0]["id"] == "solo", f"single task → 1 wave: {ids(w)}"
    )

    # ── circular dep — must not hang ─────────────────────────────────────────
    w = _build_dependency_waves([t("x"), t("y")], {"x": ["y"], "y": ["x"]})
    total = sum(len(wave) for wave in w)
    all_ok &= chk(
        "waves", total == 2, f"circular dep → resolved without hang, {total} tasks placed: {ids(w)}"
    )

    # ── all tasks preserved ──────────────────────────────────────────────────
    tasks = [t(f"t{i}") for i in range(6)]
    deps = {"t1": ["t0"], "t2": ["t0"], "t3": ["t1", "t2"], "t4": ["t2"], "t5": ["t3", "t4"]}
    w = _build_dependency_waves(tasks, deps)
    total = sum(len(wave) for wave in w)
    all_ok &= chk(
        "waves",
        total == 6,
        f"complex 6-task graph → all 6 tasks placed across {len(w)} waves: {ids(w)}",
    )

    return all_ok


# =============================================================================
# SECTION 8 — Full pipeline run (real LLM call)
# =============================================================================


async def test_full_pipeline(api_keys: dict[str, bool]) -> bool:
    hdr(8, "Full Pipeline Run  (mode=cost_efficient, deploy_mode=local)")

    has_llm = api_keys.get("openai") or api_keys.get("anthropic") or api_keys.get("openrouter")
    if not has_llm:
        warn("No LLM API key — skipping full pipeline test")
        warn("Set OPENAI_API_KEY or ANTHROPIC_API_KEY and re-run to exercise section [8]")
        return True  # not a failure — environment limitation

    from langgraph.checkpoint.memory import MemorySaver

    from aifactory.core.memory_bridge import MemoryBridge
    from aifactory.core.memory_router import MemoryRouter
    from aifactory.core.rl_scorer import get_model_rl_scores
    from aifactory.core.schemas import MemoryLevel
    from aifactory.graphs.dynamic_pipeline import build_dynamic_pipeline

    all_ok = True
    run_id = f"smoke_{int(time.time())}"

    # A task that exercises: analyser → coder worker + analyst worker → judge → RL write
    query = (
        "Write a Python function to compute the nth Fibonacci number using "
        "dynamic programming (memoization). Then briefly analyse its time "
        "and space complexity with Big-O notation."
    )

    info(f"run_id  = {run_id}")
    info("mode    = cost_efficient")
    info(f"query   = '{query[:80]}...'")
    t_start = time.perf_counter()

    try:
        # Create shared router so post-run inspection reads from the same instance
        shared_router = MemoryRouter()
        shared_bridge = MemoryBridge(router=shared_router, run_id=run_id)

        checkpointer = MemorySaver()
        app, registry = build_dynamic_pipeline(
            mode="cost_efficient",
            checkpointer=checkpointer,
            run_id=run_id,
            memory_bridge=shared_bridge,
        )

        config = {"configurable": {"thread_id": run_id}}
        state = await app.ainvoke({"user_query": query}, config=config)
        elapsed = time.perf_counter() - t_start

        # ── final response ───────────────────────────────────────────────────
        final = state.get("final_response", "")
        all_ok &= chk(
            "pipeline",
            bool(final) and len(final) > 50,
            f"final_response: {len(final)} chars in {elapsed:.1f}s",
        )

        has_fib = any(
            kw in final.lower() for kw in ["fibonacci", "fib", "def ", "memoization", "dynamic"]
        )
        all_ok &= chk("pipeline", has_fib, "final_response addresses fibonacci / DP task")

        # ── worker task results ──────────────────────────────────────────────
        task_results = state.get("task_results", [])
        all_ok &= chk(
            "pipeline", len(task_results) > 0, f"task_results: {len(task_results)} worker result(s)"
        )

        info("Worker breakdown:")
        for r in task_results:
            status = "SKIPPED" if r.get("skipped") else "DONE"
            info(f"  [{r['task_id']}] {r['agent']:<22} model={r['model']:<20} [{status}]")
            info(f"         desc: {r['description'][:70]}")

        # ── token tracking ───────────────────────────────────────────────────
        agents = registry.all()
        total_tokens = sum(getattr(a, "total_tokens", 0) for a in agents)
        all_ok &= chk(
            "pipeline",
            total_tokens > 0,
            f"token tracking: {total_tokens:,} total tokens consumed across {len(agents)} agents",
        )

        info("Per-agent token breakdown:")
        for a in sorted(agents, key=lambda x: -getattr(x, "total_tokens", 0)):
            toks = getattr(a, "total_tokens", 0)
            if toks > 0:
                info(f"  {a.name:<28} {toks:>6,} tokens")

        # ── model selection diversity ────────────────────────────────────────
        models_used = {r["model"] for r in task_results if r.get("model")}
        info(f"Worker models selected: {models_used}")

        # ── judge metrics ────────────────────────────────────────────────────
        judge_metrics = state.get("judge_metrics", {})
        all_ok &= chk(
            "pipeline", bool(judge_metrics), f"judge_metrics present: {list(judge_metrics.keys())}"
        )

        assessments = state.get("agent_assessments", [])
        info(f"Agent assessments ({len(assessments)} entries):")
        for a in assessments:
            info(f"  {a.get('agent', '?'):<26} quality={a.get('quality', '?')}")

        # ── shared context broadcasting ──────────────────────────────────────
        shared_ctx = state.get("shared_context", {})
        all_ok &= chk(
            "pipeline",
            len(shared_ctx) > 0,
            f"shared_context: {len(shared_ctx)} key(s) — {list(shared_ctx.keys())[:6]}",
        )

        # ── memory post-run: search for run artifacts (same router instance) ──
        info("Memory layer counts post-run (searching 'fibonacci'):")
        try:
            for level in [MemoryLevel.AGENT, MemoryLevel.TEAM, MemoryLevel.GLOBAL]:
                hits = shared_router.retrieve(
                    "fibonacci dynamic programming", levels=[level], top_k=30
                )
                info(f"  {level.value:<8} layer: {len(hits)} entry/entries")
            all_agent_hits = shared_router.retrieve("fibonacci", top_k=100)
            all_ok &= chk(
                "pipeline",
                len(all_agent_hits) > 0,
                f"memory persisted during run: {len(all_agent_hits)} total entries across all layers",  # noqa: E501
            )
        except Exception as exc:
            warn(f"Memory post-run check (non-fatal): {exc}")

        # ── RL scorer post-run ───────────────────────────────────────────────
        info("RL score table post-run:")
        for task_type in ["coding", "analysis", "general"]:
            try:
                rl = get_model_rl_scores(task_type, min_samples=1)
                if rl:
                    best = max(rl, key=lambda k: rl[k])
                    info(
                        f"  {task_type:<12} best={best} ({rl[best]:.2f}/10)  "
                        f"[{len(rl)} model(s) tracked]"
                    )
                else:
                    info(f"  {task_type:<12} no RL records yet")
            except Exception as exc:
                warn(f"RL post-run check (non-fatal): {exc}")

    except Exception as exc:
        elapsed = time.perf_counter() - t_start
        fail(f"Pipeline raised exception after {elapsed:.1f}s: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        all_ok = False

    return all_ok


# =============================================================================
# SUMMARY
# =============================================================================


def print_summary() -> bool:
    passed = sum(1 for v in _results.values() if v)
    failed = sum(1 for v in _results.values() if not v)
    total = len(_results)

    print(f"\n{BOLD}{'═' * 64}{RESET}")
    print(f"{BOLD}SMOKE TEST SUMMARY{RESET}")
    print(f"{'═' * 64}")
    print(f"  {GREEN}PASSED{RESET}: {passed}/{total}")
    if failed:
        print(f"  {RED}FAILED{RESET}: {failed}/{total}")
        print("\n  Failed checks:")
        for name, v in _results.items():
            if not v:
                print(f"    {RED}✗{RESET}  {name}")
    else:
        print(f"\n  {GREEN}{BOLD}All checks passed.{RESET}")
    print(f"{'═' * 64}\n")
    return failed == 0


# =============================================================================
# Entry point
# =============================================================================


async def main(skip_llm: bool) -> int:
    t0 = time.perf_counter()
    print(f"\n{BOLD}AiFactory Smoke Test{RESET}")
    print(f"mode=cost_efficient  deploy_mode=local  repo={_REPO.name}")

    api_keys = probe_env()

    test_complexity_gate()
    test_model_selection()
    await test_sandbox()
    await test_docker_mcp()
    test_memory()
    test_rl_scorer()
    test_share_protocol()
    test_dependency_waves()

    if not skip_llm:
        await test_full_pipeline(api_keys)
    else:
        hdr(8, "Full Pipeline Run — SKIPPED (--skip-llm)")
        warn("Pass without --skip-llm (and set an LLM API key) to run the real pipeline.")

    ok_overall = print_summary()
    info(f"Total wall time: {time.perf_counter() - t0:.1f}s")
    return 0 if ok_overall else 1


def write_report(skip_llm: bool) -> None:
    """Write a JSON report to logs/smoke_<run_id>.json."""
    import json as _json
    from datetime import datetime

    logs_dir = _REPO / "logs"
    logs_dir.mkdir(exist_ok=True)

    run_id = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    report_path = logs_dir / f"smoke_{run_id}.json"

    passed = sum(1 for v in _results.values() if v)
    failed = sum(1 for v in _results.values() if not v)

    report = {
        "run_id": run_id,
        "skip_llm": skip_llm,
        "total": len(_results),
        "passed": passed,
        "failed": failed,
        "ok": failed == 0,
        "checks": {name: "pass" if v else "fail" for name, v in _results.items()},
    }

    report_path.write_text(_json.dumps(report, indent=2))
    print(f"\n{DIM}Report written to {report_path}{RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AiFactory end-to-end smoke test")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip section [8] — full pipeline LLM call (no API key required)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write a JSON report to logs/smoke_<run_id>.json",
    )
    args = parser.parse_args()
    exit_code = asyncio.run(main(skip_llm=args.skip_llm))
    if args.report:
        write_report(skip_llm=args.skip_llm)
    sys.exit(exit_code)
