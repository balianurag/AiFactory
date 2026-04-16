# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. Also after new work keep updating the claude.md and spec.md

## Project Overview

**AI Factory** — A self-assembling, self-optimising multi-agent system that turns a raw human idea into production-grade output with minimal manual coordination.

### Architecture

```
Human Idea → Master (coordinator)
           → Judge (briefs quality criteria)
           → Analyser (task breakdown / direct-answer bypass)
           → [Researcher → Re-analyse]?   (optional pre-analysis research loop)
           → [Researcher]?                 (optional execution research)
           → Execution Lead (spawns N Workers dynamically, per-task model selection)
           → Judge (reviews pipeline run quality)
           → Master (synthesises final response)
```

Key features:
- **Config-driven agents** — all roles defined in YAML (`config/agents/*.yaml`)
- **10-ability model scoring** — each model rated on reasoning, coding, math, writing, etc. (`config/model_registry.yaml`)
- **3 mode profiles** — `best`, `balanced`, `cost_efficient` (alpha-blended quality vs cost)
- **LiteLLM** — cross-provider LLM abstraction (OpenAI, Anthropic, Bedrock, etc.)
- **SHARE: protocol** — lightweight inter-agent context broadcasting
- **Dynamic worker spawning** — Execution Lead creates per-task workers with ability-scored models
- **Direct-answer bypass** — trivial queries skip workers entirely (Analyser detects)
- **Docker MCP sandbox** — warm Docker container pool via MCP (10-50ms, gVisor+seccomp+cap-drop security). Falls back to E2B or local subprocess automatically
- **Package registry** — `config/package_registry.yaml` injected into worker prompts so LLMs never waste tokens on `pip install`
- **DSA-optimised coding** — workers use optimal algorithms, state Big-O complexity, prefer stdlib tools
- **3-layer memory**: Agent → Team → Global LTM (all via Qdrant)
- **Human-in-the-loop** — LangGraph `interrupt()` for worker QUESTION: prompts
- **MemorySaver checkpointing** — graph state persisted for resume after interrupts

### Tech Stack

- **Python 3.13+**, managed with `uv` + `pyproject.toml` (hatchling build)
- **LangGraph** — stateful graph orchestration
- **LlamaIndex** — memory management and retrieval
- **LiteLLM** — unified LLM interface (any provider)
- **Pydantic v2** — structured schemas for all A2A communication
- **Qdrant** — vector DB for 3-layer memory (in-memory mode for local dev)
- **Rich + Typer** — CLI interface
- **pytest** — testing

### Commands

All dev and app commands are available via `manage.py` at the repo root:

```bash
# One-time environment setup (Python check, uv, venv, deps, pre-commit)
python manage.py setup

# Dev tooling
python manage.py install          # uv sync --dev
python manage.py dev              # FastAPI dev server with auto-reload (port 8000)
python manage.py test             # pytest tests/
python manage.py test -k "name"   # Filter by test name
python manage.py test --cov       # With HTML coverage report
python manage.py lint             # ruff check src/ tests/
python manage.py lint --fix       # Auto-fix violations
python manage.py format           # ruff format src/ tests/
python manage.py format --check   # CI mode (no writes)
python manage.py typecheck        # mypy src/
python manage.py build            # uv build → dist/

# Application commands (delegates to aifactory CLI)
python manage.py run "task"                     # Run the agent pipeline
python manage.py run --task-file task.md        # Task from file
python manage.py info                           # Show config + API key status
python manage.py memory-stats                   # Memory layer stats
python manage.py models list                    # All roles + allowed models
python manage.py models list --role coder       # Filter by role
python manage.py models tiers                   # Premium/standard/budget assignments
python manage.py agents list                    # All roles, tiers, default models

# Memory commands (Phase 2.9 / 5.0)
uv run aifactory memory health                  # ★ Check Qdrant + embedding provider + Redis + collections
uv run aifactory memory stats                   # Layer counts (agent/team/global)
uv run aifactory memory search "FastAPI auth"   # Semantic search across memory
uv run aifactory memory search "query" --layer team --agent analyser
uv run aifactory memory clear --layer agent --yes   # Clear agent layer
uv run aifactory memory clear --agent Worker-task_1 # Clear specific agent

# API server (Phase 5.0)
uv run aifactory serve                          # Start persistent API server (localhost:8080)
uv run aifactory serve --port 9000              # Custom port
uv run aifactory serve --workers 4             # Production (multi-process)
uv run aifactory serve --reload --log-level debug  # Hot-reload dev mode

# RL-lite adaptive model scoring commands (Phase 3)
uv run aifactory rl stats                       # Show model performance by task type
uv run aifactory rl stats --type coding         # Filter to a single task type
uv run aifactory rl reset --yes                 # Clear all RL records (fresh start)
uv run aifactory rl reset --type coding         # Clear records for one task type

# Docker MCP Sandbox commands (Phase 4.0)
uv run aifactory sandbox build                  # Build aifactory-sandbox + aifactory-mcp-server images
uv run aifactory sandbox build --no-cache       # Force full rebuild
uv run aifactory sandbox pull                   # Pull base images (python:3.12-slim, socket-proxy)
uv run aifactory sandbox status                 # Pool health metrics (ready/in_use/total)
uv run aifactory sandbox status --url http://localhost:8100  # Override MCP server URL
uv run aifactory sandbox test                   # Smoke test: Python + numpy/pandas in container
uv run aifactory sandbox test --language bash   # Test bash execution
uv run aifactory sandbox test --language javascript  # Test Node.js execution
```

Raw `uv run aifactory <cmd>` also works for all application commands.

### Project Structure

```
src/aifactory/
├── core/
│   ├── base_agent.py       # Base agent with LiteLLM, memory, token tracking, @traceable
│   ├── assessor.py         # Task complexity classifier (trivial/moderate/complex)
│   ├── model_selector.py   # 10-ability dot-product model scoring engine
│   ├── agent_factory.py    # YAML agent loader, SHARE: protocol, AgentRegistry, pkg registry
│   ├── schemas.py          # ALL Pydantic models (A2A messages, configs, RL)
│   ├── memory_router.py    # 3-layer Qdrant memory (store/retrieve/promote)
│   ├── rate_limiter.py     # slowapi rate limiting (4 tiers)
│   ├── sandbox.py          # ★ 3-backend code execution (docker > e2b > local)
│   ├── sandbox_pool.py     # ★ Async warm Docker container pool (Phase 4.0)
│   ├── rl_scorer.py        # ★ RL-lite adaptive model scoring (Phase 3)
│   ├── embed_provider.py   # ★ Auto-select embedding provider + Redis embedding cache (Phase 5.0)
│   ├── semantic_cache.py   # ★ Qdrant-backed semantic query deduplication (Phase 5.0)
│   └── observability.py    # LangSmith setup, @traceable decorator, run fetchers
├── agents/
│   ├── worker.py           # Legacy worker (coder/researcher/writer) — kept for backward compat
│   ├── team_lead.py        # Legacy team lead — kept for backward compat
│   └── project_manager.py  # Legacy PM — kept for backward compat
├── graphs/
│   ├── dynamic_pipeline.py # ★ 8-node LangGraph: the new config-driven pipeline
│   └── single_team.py      # Legacy 4-node pipeline — kept for backward compat
├── api/
│   └── main.py             # FastAPI app: /api/health, /api/models, /api/tasks, /api/observability/*, WS /api/ws/{id}
├── prompts/                # System prompts per role (.md files)
├── mcp/                    # ★ MCP sandbox integration (Phase 4.0)
│   ├── __init__.py
│   ├── sandbox_server.py   #   FastMCP server: execute_python/bash/js + pool_status tools
│   └── client.py           #   MCP client factory (HTTP mode or in-process stdio)
├── config/
│   ├── settings.py         # Pydantic Settings (env vars + .env, incl. LangSmith config)
│   ├── models.yaml         # Legacy model registry for /api/models UI endpoint
│   ├── model_registry.yaml # ★ ~20+ models with 10-ability scores + pricing
│   ├── graph.yaml          # ★ Graph runtime config (mode, fallback model)
│   ├── package_registry.yaml # ★ Pre-installed sandbox packages (injected into worker prompts)
│   └── agents/             # ★ YAML configs per agent role:
│       ├── master.yaml     #   - Coordinator / synthesiser
│       ├── judge.yaml      #   - Quality reviewer
│       ├── analyser.yaml   #   - Task breakdown / direct-answer detection
│       ├── researcher.yaml #   - Pre-analysis and execution research
│       ├── execution_lead.yaml # - Worker dispatcher + consolidator
│       ├── worker.yaml     #   - Template for per-task workers
│       ├── worker_coder.yaml   # ★ Specialist: software engineer (coding/math tasks)
│       ├── worker_researcher.yaml # ★ Specialist: research & synthesis (research tasks)
│       ├── worker_writer.yaml  # ★ Specialist: writer & communicator (writing tasks)
│       └── worker_analyst.yaml # ★ Specialist: analyst & reasoner (analysis tasks)
└── cli/                    # ★ Modular Typer CLI
    ├── __init__.py         #   - App root + command registration
    ├── _helpers.py         #   - Shared console, dump loading, formatting
    ├── cmd_new.py          #   - `aifactory new` / `aifactory resume` (complexity gate → fast/full path)
    ├── cmd_run.py          #   - `aifactory run` (advanced, --mode, --context)
    ├── cmd_status.py       #   - `aifactory status` / `aifactory inspect`
    ├── cmd_connect.py      #   - `aifactory connect` / `aifactory tools`
    ├── cmd_ask.py          #   - `aifactory ask`
    ├── cmd_agents.py       #   - `aifactory agents list`
    ├── cmd_cost.py         #   - `aifactory cost`
    ├── cmd_info.py         #   - `aifactory info` / `aifactory models`
    ├── cmd_memory.py       #   ★ `aifactory memory stats/search/clear/health` (Phase 2.1/5.0)
    ├── cmd_sandbox.py      #   ★ `aifactory sandbox build/status/test/pull` (Phase 4.0)
    ├── cmd_rl.py           #   ★ `aifactory rl stats/reset` (Phase 3)
    └── cmd_serve.py        #   ★ `aifactory serve` — persistent API server with pre-warming (Phase 5.0)
```

### Implementation Status

- [x] Phase 0: Foundation (schemas, base agent, memory router, settings)
- [x] Phase 1: Single-team pipeline (PM → TL → Workers graph) — legacy, still in codebase
- [x] Phase 1.5: HTTP/WebSocket API (`/api/tasks`, `/api/models`, WS streaming) — deployed on Render
- [x] Phase 1.5: Curated model registry (`models.yaml`) — UI model picker endpoint
- [x] Phase 1.5: CLI model/agent commands (`models list`, `models tiers`, `agents list`)
- [x] Phase 2.0: LangSmith observability — `@traceable` on BaseAgent.think(), API endpoints, UI Traces view
- [x] Phase 2.1: LiteLLM migration — replaced OpenAI SDK with `litellm.acompletion()` everywhere
- [x] Phase 2.2: Ability-based model selection (`model_selector.py` + `model_registry.yaml`)
- [x] Phase 2.3: YAML config-driven agents (`agent_factory.py` + `config/agents/*.yaml`)
- [x] Phase 2.4: 8-node dynamic pipeline (`dynamic_pipeline.py`) — the production architecture
- [x] Phase 2.5: Pipeline wired to API (`_run_pipeline()` uses `build_dynamic_pipeline()`)
- [x] Phase 2.5: API rate limiting (slowapi, 4 tiers)
- [x] Phase 2.5: Code execution sandbox (local subprocess)
- [x] Phase 2.5: Worker self-correction loop
- [x] Phase 2.6: E2B cloud sandbox integration — dual-backend code execution for workers
- [x] Phase 2.7: Code optimisation flow — DSA excellence, concise prompts, task merging, isolated sandbox awareness
- [x] Phase 2.8: ComplexityGate v2 — 4-tier routing (bypass math, static facts, fast_llm, full pipeline), YAML-driven patterns
- [x] Phase 2.9: Memory Bridge (Phase 2.1) — `MemoryBridge` wired into every `PipelineAgent`; retrieval before think(), storage after; SHARE: updates persisted to team layer; promotion at N-turn threshold; CLI `memory stats/search/clear`
- [x] Phase 3.5: Token Efficiency — Anthropic prompt caching (direct + OpenRouter Claude, `cache_control: ephemeral`); LiteLLM in-memory response cache for ALL providers/tiers; smart judge (skip LLM for single-task runs, ~1000 token saving); shared context pruning (cap at 25 keys); 3 new settings
- [x] Phase 4.0: Docker MCP Sandbox — warm container pool (`sandbox_pool.py`), FastMCP server (`mcp/sandbox_server.py`), auto-detect client (`mcp/client.py`), `sandbox.py` updated (docker > e2b > local priority), package registry injected into worker prompts, CLI `aifactory sandbox build/status/test/pull`; 62 new tests
- [x] Phase 3: RL-lite scoring — `core/rl_scorer.py`; `(model, task_type) → avg_quality` stored in Qdrant `aifactory_rl_performance`; `select_model()` blends static ability scores + RL signal (±20% multiplicative at default rl_weight=0.2); judge_review records per-worker performance after each run; `create_worker_config()` fetches RL scores before model selection; CLI `aifactory rl stats/reset`; 3 new settings (`AIFACTORY_RL_ENABLED/WEIGHT/MIN_SAMPLES`); 22 new tests
- [x] Phase 4.1: MCP Catalog provider — `MCPProvider` alongside `ComposioProvider`; config-driven server catalog (`config/integrations/mcp_catalog.yaml`); 6 servers: GitHub (official Docker), Atlassian/Jira+Confluence (uvx), Notion (npx), Google Workspace (disabled until token vault), Linear (npx), Slack (disabled); `AsyncExitStack` session management; `server::tool_name` slug format for `ACTION:mcp` blocks; `_build_toolkit_catalog()` prefers MCP for covered toolkits, falls back to Composio for the rest; `AIFACTORY_MCP_CATALOG_ENABLED=false` feature flag; 63 new tests
- [x] Phase 4.2: DEPLOY_MODE routing — `AIFACTORY_DEPLOY_MODE=local|cloud`; `local` uses stdio subprocess (dev); `cloud` maintains async HTTP session pool per MCP server (N persistent sessions, acquire/release via `asyncio.Queue`); `url_env_var` + `pool_size` fields in `mcp_catalog.yaml`; `AIFACTORY_MCP_HTTP_POOL_SIZE` setting; 12 new tests (76 total in `test_mcp_provider.py`)
- [x] Phase 4.3: Parallel worker execution + role specialization + self-correction + dependency enforcement — `asyncio.gather()` wave-based dispatch; 4 specialist YAML configs (`worker_coder/researcher/writer/analyst.yaml`); REVISION: self-check loop; `dependencies_map` extracted from plan JSON; `_build_dependency_waves()` topological sort; 20 new tests
- [x] Phase 5.0: Production hardening — token streaming (`on_token` callback, `_think_stream()`, `AIFACTORY_STREAM_TOKENS_ENABLED`); Redis task store (run lifecycle tracked: running → completed, sorted-set index `aifactory:runs:all`, 7d TTL); semantic cache (`core/semantic_cache.py`, Qdrant collection `aifactory_semantic_cache`, cosine similarity dedup at query-intent level, `AIFACTORY_SEMANTIC_CACHE_*` settings); judge deep eval (post-metrics LLM scoring on relevance/accuracy/completeness/actionability, `AIFACTORY_JUDGE_DEEP_EVAL_ENABLED`); auto-embedding provider (`core/embed_provider.py`, priority: OpenAI → Voyage → Cohere → HuggingFace → stub, Redis embedding cache, clear stub warnings); `aifactory serve` command (`cmd_serve.py`, pre-warms all connections); `aifactory memory health` subcommand; ComplexityGate pattern expansion (7 new `simple_qa` patterns, narrowed research routing); budget warn threshold with `on_warn` callback (80% cap warning); 134 new tests (1024 total)
- [ ] Phase 5.1: Hardening (circuit breakers, budget hard caps enforcement, batch pipeline)

**Current test count: 1024 passing**

### Key Design Decisions

1. **Config-driven agents** — Agent behaviour comes from YAML files, not Python subclasses. One `PipelineAgent` class handles all roles.
2. **10-ability model scoring** — Each model rated 0-3 on reasoning, coding, math, writing, instruction_following, tool_use, knowledge, speed, cost_efficiency, context_handling. Agent YAML files declare required abilities. Dot-product scoring selects the best model per role+mode.
3. **3 execution modes** — `best` (quality-first), `balanced` (default), `cost_efficient`. Each mode has an alpha weight that blends quality vs cost/speed.
4. **SHARE: protocol** — Agents append `SHARE: key: value` lines to their output. These are extracted and merged into a shared context dict visible to downstream agents. Phase 2.9: SHARE: updates are also persisted to team-layer memory for cross-run retrieval.
5. **LiteLLM for model abstraction** — switch providers with a string change. Supports OpenAI, Anthropic, Bedrock, etc.
6. **Memory promotion every N turns** — agent layer summarised → team layer. Interval set via `AIFACTORY_MEMORY_PROMOTION_INTERVAL` (default: 10).
7. **In-memory Qdrant for dev** — no Docker needed. Falls back automatically if `QDRANT_URL` is unreachable.
8. **Token budgets per agent** — hard caps enforced in `BaseAgent.think()`. Raises `AgentBudgetError` when exceeded.
9. **uv for package management** — 10-100x faster than pip. Venv: `.venv` inside the project root.
10. **Stub embeddings in dev** — `MemoryRouter` uses a SHA-256 hash-based pseudo-embedding (1536-dim). Pass a real `embedding_fn` for production semantic search.
11. **Env var naming** — LLM API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) have **no prefix**. All other settings use the `AIFACTORY_` prefix. LangSmith uses `LANGCHAIN_` prefix.
12. **Tests are async-native** — `asyncio_mode = "auto"` in `pyproject.toml`; no `@pytest.mark.asyncio` needed.
13. **LangSmith tracing is zero-cost when disabled** — `@traceable` decorator is a passthrough when `LANGCHAIN_API_KEY` is absent.
14. **Stack: LangGraph + LlamaIndex + LangSmith + LiteLLM** — LangGraph for orchestration, LlamaIndex for memory/RAG, LangSmith for observability, LiteLLM for cross-provider LLM calls.
15. **Project-level `pip.conf`** — `pip.conf` at the repo root pins PyPI as the index.
16. **E2B sandbox** — `Sandbox.create(api_key=..., timeout=...)` for SDK v2.6+. Auto-detected via `E2B_API_KEY` env var → settings/.env fallback. Each worker's code runs in a fresh isolated micro-VM — no shared state between workers. Now third priority after Docker MCP.
22. **Docker MCP Sandbox (Phase 4.0)** — `core/sandbox_pool.py` + `mcp/sandbox_server.py` + `mcp/client.py`. Warm Docker container pool (min_idle=4, max_size=16). Every container runs with: `--network=none --cap-drop=ALL --read-only --tmpfs /workspace --pids-limit=64 --memory=512m` + custom seccomp profile. gVisor (`runsc`) runtime optional. FastMCP server exposes `execute_python/bash/javascript + pool_status`. Two client modes: HTTP (prod, bearer auth) or stdio/in-process (dev, zero config). Auto-selected via `AIFACTORY_MCP_SANDBOX_URL`. Backend priority: docker > e2b > local. Toggle with `AIFACTORY_SANDBOX_DOCKER_ENABLED=true`.
23. **Package registry injection** — `config/package_registry.yaml` loaded once (LRU-cached) by `agent_factory.py`. Its `prompt_snippet` field is appended to every worker system prompt so LLMs never attempt to `pip install` packages that are already pre-installed in the sandbox image (numpy, pandas, requests, sklearn, etc.).
17. **DSA-optimised coding prompts** — CODING EXCELLENCE instructions in worker.yaml + agent_factory.py. Concise: Big-O in code comments, not separate prose. Execution lead merges dependent coding tasks the analyser over-splits.
18. **Analyser CODING RULE** — write+test+run+show = 1 task. Prevents sandbox isolation errors from splitting tightly-coupled code across workers.
19. **MemoryBridge (Phase 2.9)** — `core/memory_bridge.py` is the fail-safe adapter between `PipelineAgent` and `MemoryRouter`. All ops are try/except-wrapped at both bridge and `think()` level — memory failures never break the pipeline. `build_dynamic_pipeline()` auto-creates one bridge per run (using `run_id` as `team_id` for isolation). Disable with `AIFACTORY_MEMORY_PIPELINE_ENABLED=false`. Tune injection with `AIFACTORY_MEMORY_CONTEXT_TOP_K` and `AIFACTORY_MEMORY_CONTEXT_MAX_CHARS`.
20. **ComplexityGate v2** — `core/complexity_gate.py`. 4-tier: pure math bypass (AST eval, 0 tokens) → static facts (dict lookup, 0 tokens) → fast_llm (single master call, ~1s) → full 8-node pipeline. YAML-driven patterns in `config/gate_patterns.yaml`. Wired into both `aifactory new` and `aifactory run`.
24. **RL-lite Adaptive Model Selection (Phase 3)** — `core/rl_scorer.py` stores `(model_id, task_type) → quality_score` in a dedicated Qdrant collection (`aifactory_rl_performance`). After each pipeline run, `judge_review` calls `record_pipeline_performances()` which normalises judge quality (0–3 → 0–10) per worker and persists to Qdrant. Before worker model selection, `create_worker_config()` calls `get_model_rl_scores(task_type)` and passes results to `select_model()`. The RL signal is applied as a multiplicative factor: `adj = 1 + rl_weight × (rl_score/5 − 1)`, capped at ±rl_weight (default ±20%). Only models with ≥ `AIFACTORY_RL_MIN_SAMPLES` records (default: 3) are adjusted — prevents a single run from skewing selection. Task type inferred by keyword matching (7 types). Falls back to in-memory Qdrant if remote unavailable. Toggle: `AIFACTORY_RL_ENABLED`. CLI: `aifactory rl stats [--type coding]` / `aifactory rl reset`.
25. **MCP Catalog Strategy (investigated April 2026)** — `docs/mcp-catalog-feasibility.md`. Decision: hybrid approach. Full Composio replacement is premature (token vault / per-tenant OAuth management is 3–4 weeks of infra). Phase 1: add `MCPProvider` class behind feature flag. Phase 2: migrate GitHub, Jira, Notion, Google Workspace to direct MCP servers (data residency + audit wins for B2B enterprise). Phase 3: full catalog + token vault.
26. **DEPLOY_MODE MCP routing (Phase 4.2)** — `AIFACTORY_DEPLOY_MODE=local` (default) uses stdio subprocess per MCP server; `cloud` uses an async HTTP session pool. Pool is `asyncio.Queue`-backed, N sessions per server (default 3, per-server `pool_size` override in `mcp_catalog.yaml`). Tool calls in `cloud` mode: `queue.get(timeout=30)` → call → `queue.put_nowait`. Pool cleanup in `disconnect()` drains stacks before closing stdio.
27. **Parallel worker execution (Phase 4.3)** — Workers execute in dependency-ordered waves. `_build_dependency_waves(tasks, deps_map)` topological sort → `list[list[task]]`. Single-task waves run directly (preserves `interrupt()` semantics). Multi-task waves: `asyncio.gather(*coros, return_exceptions=True)`. Each worker returns `(result_entry, ctx_delta)` — only net context changes are returned so parallel workers don't clobber shared_ctx. Deltas merged sequentially after gather.
28. **Worker role specialization (Phase 4.3)** — `create_worker_config()` detects task type (coding/math/research/writing/analysis) by keyword. Loads matching specialist YAML (`worker_coder/researcher/writer/analyst.yaml`). Specialist `task` text prepended before the execution lead's instruction, so the LLM assumes the role persona before reading the task. Falls back to generic `worker.yaml` for unrecognised types.
29. **Self-correction loop (Phase 4.3)** — After tool rounds complete, non-tool workers do one self-check `think()`: "Is this complete and accurate? Reply VALIDATED or REVISION: <note>\n<corrected>". `REVISION:` triggers a single corrected response that replaces the original result. Skipped for tool tasks (quality determined by tool success) and `NOT_NEEDED` outputs. One revision max — no infinite loop.
30. **Token streaming (Phase 5.0)** — `PipelineAgent.on_token` callback wired into `think()`. When set, `_think_stream()` uses `litellm.acompletion(..., stream=True)` and fires the callback for each delta token. Falls back to non-streaming automatically on pre-first-token errors. CLI `cmd_new.py` emits a `"token"` event that writes raw tokens to stdout. Toggle: `AIFACTORY_STREAM_TOKENS_ENABLED`.
31. **Auto-embedding provider (Phase 5.0)** — `core/embed_provider.py` auto-selects the best embedding model from available API keys: OpenAI `text-embedding-3-small` → Voyage `voyage-3-lite` → Cohere `embed-english-light-v3.0` → HuggingFace `all-MiniLM-L6-v2` (local) → SHA-256 stub. Stub mode emits a prominent warning panel and clearly instructs the fix. Redis embedding cache wraps raw embed calls (key: `sha256[:24]`, TTL: 7d). `MemoryRouter` calls `build_embed_fn()` instead of the old hardcoded model.
32. **Semantic cache (Phase 5.0)** — `core/semantic_cache.py`. Collection: `aifactory_semantic_cache`. `lookup(query)` → embed → kNN (limit=1, score_threshold) → TTL check → return cached `final_response` if hit. `store(query, final_response)` → embed → upsert. Wired into `master_receive` (check) and `master_respond` (store). Settings: `AIFACTORY_SEMANTIC_CACHE_ENABLED`, `AIFACTORY_SEMANTIC_CACHE_THRESHOLD` (default 0.92), `AIFACTORY_SEMANTIC_CACHE_TTL_DAYS`, `AIFACTORY_SEMANTIC_CACHE_MAX_SIZE`.
33. **Judge deep eval (Phase 5.0)** — After synthetic/LLM judge metrics, an additional focused LLM call scores `final_response` on relevance (0.30), accuracy (0.30), completeness (0.25), actionability (0.15) producing `composite_quality` and `eval_*` fields in judge metrics. Toggle: `AIFACTORY_JUDGE_DEEP_EVAL_ENABLED`. Model: `AIFACTORY_JUDGE_EVAL_MODEL` (default `openai/gpt-4o-mini`).
34. **`aifactory serve` (Phase 5.0)** — `cmd_serve.py`. Starts `aifactory.api.main:app` via uvicorn. Pre-warms Redis, Qdrant, embedding provider, LiteLLM cache, model registry, and Composio before accepting requests. Options: `--host`, `--port`, `--workers`, `--reload`, `--log-level`, `--no-warmup`. `serve`, `new`, and `run` are independent: `serve` is a persistent multi-request API server; `new`/`run` are one-shot CLI commands that spin up their own pipeline process.
35. **Redis task store (Phase 5.0)** — `cmd_new.py` + `cmd_run.py`. At run start: `SET aifactory:run:{rid}` (status=running) + `ZADD aifactory:runs:all`. At completion: updates status=completed with elapsed_s, total_tokens, final_response_preview, judge_status. TTL: 7d. Falls back silently if `REDIS_URL` is not set. Consumed by `api/main.py` and `aifactory runs` CLI.
21. **Token Efficiency (Phase 3.5)** — Three layered optimisations in `dynamic_pipeline.py`:
    - **Anthropic prompt caching** — `cache_control: ephemeral` on system messages for `claude-*`, `anthropic/claude-*`, and `openrouter/anthropic/claude-*`. 60–90% cost reduction on repeat calls with the same system prompt. Off by default when routing through a local LiteLLM proxy. Toggle with `AIFACTORY_PROMPT_CACHE_ENABLED` and `AIFACTORY_PROMPT_CACHE_OPENROUTER`.
    - **LiteLLM response cache** — `litellm.cache = Cache(ttl=3600)` initialised once per pipeline. All providers/tiers benefit: identical (model, messages) pairs return 0-token cached responses. Useful for QUESTION/ANSWER retries and repeated runs. `AIFACTORY_RESPONSE_CACHE_ENABLED` / `AIFACTORY_RESPONSE_CACHE_TTL`.
    - **Smart judge** — skip judge LLM call for single-task runs with no research (~800–1200 tokens saved). Synthetic assessment generated from run metadata. Toggle with `AIFACTORY_JUDGE_SKIP_SIMPLE_RUNS`.
    - **Shared context pruning** — `_prune_shared_context(ctx, max_keys=25)` called after analyse, researcher, and execution_lead nodes. Priority keys (objective, tasks_count, etc.) always kept. `AIFACTORY_SHARED_CONTEXT_MAX_KEYS`.

### Agent Execution Pattern

The 8-node dynamic pipeline follows this flow:

1. **master_receive** — Acknowledges the request, seeds shared context with the objective
2. **judge_brief** — Defines success criteria and what will be evaluated
3. **analyse** — Breaks request into atomic tasks (or returns a direct answer for trivial queries)
4. **researcher** (conditional) — Gathers research for pre-analysis or execution
5. **reanalyse** (conditional) — Refines task breakdown after research
6. **execution_lead** — Plans worker dispatch with per-task ability requirements and dependency graph, spawns N workers dynamically with per-task model + specialist role selection, runs workers in parallel dependency waves, consolidates results
7. **judge_review** — Evaluates each agent's performance, produces quality metrics
8. **master_respond** — Synthesises final response for the human (no mention of internal process)

Workers support:
- **NOT_NEEDED** self-skip — if a worker determines its task is already covered
- **QUESTION:** interrupts — worker can pause and ask the human a question (via LangGraph interrupt())
- **SHARE:** protocol — share discovered context with downstream agents
- **Specialist roles** — coding/math → Coder persona; research → Researcher persona; writing → Writer persona; analysis → Analyst persona
- **Self-correction** — one VALIDATED/REVISION: self-check after initial response (non-tool, non-skipped tasks)
- **Parallel execution** — tasks in the same dependency wave run concurrently via asyncio.gather()

### Documentation

Project docs live in `docs/`:
- `docs/architecture-v2.md` — full system architecture (LangGraph + LlamaIndex + LangSmith)
- `docs/sprints.md` — sprint plan with stories and acceptance criteria
- `docs/progress.md` — what’s built, current status, what’s next
- `docs/decisions.md` — key technical decisions and rationale
