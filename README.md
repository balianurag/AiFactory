# AI Factory

A self-assembling, self-optimising multi-agent system that turns a raw human idea into production-grade output with near-zero manual coordination.

---

## Quick Start

```bash
# 1. One-time setup (Python 3.13+ check, venv, deps, pre-commit hooks)
make setup

# 2. Add your API key
cp .env.example .env
# edit .env → add ANTHROPIC_API_KEY or OPENAI_API_KEY

# 3. Start databases (Qdrant + Redis + Postgres via Docker)
make db-up

# 4. Run a task
make run TASK="Build a FastAPI CRUD app for a todo list"

# Or use the CLI directly
uv run aifactory run "Build a FastAPI CRUD app for a todo list"
```

---

## Installation

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (installed automatically by `make setup`)
- Docker (for databases and sandbox)

### Full Setup

```bash
make setup        # installs uv, creates .venv, installs all deps, sets up pre-commit
make db-up        # starts Qdrant:6333, Redis:6379, Postgres:5432 in Docker
make info         # verify API keys + config
```

### Corporate / Zscaler Network

```bash
make setup CORPORATE=true    # routes pip through internal Artifactory
make deploy CORPORATE=true   # full Docker deploy with corporate certs
```

---

## Architecture

```
Human Idea
   ↓
Master (coordinator) → seeds shared context
   ↓
Judge (briefs quality criteria)
   ↓
Analyser (task breakdown / direct-answer bypass)
   ↓
[Researcher]?         — optional pre-analysis research
[Re-analyse]?         — optional refinement after research
   ↓
Execution Lead → spawns N Workers per task (parallel, dependency-ordered waves)
   │   ├── Worker (Coder)      — coding, math, DSA
   │   ├── Worker (Researcher) — research, synthesis
   │   ├── Worker (Writer)     — writing, communication
   │   └── Worker (Analyst)    — analysis, reasoning
   ↓
Judge (reviews quality, records RL scores)
   ↓
Master (synthesises final response)
```

### Routing Tiers (ComplexityGate)

Every request is classified before hitting the full pipeline:

| Tier | Trigger | Cost |
|---|---|---|
| Math bypass | Pure arithmetic expression | 0 tokens (AST eval) |
| Static facts | Lookup (capitals, conversions) | 0 tokens (dict lookup) |
| Fast LLM | Simple factual questions | ~1s, ~200 tokens |
| Full pipeline | Complex, multi-step tasks | Full 8-node pipeline |

---

## Features

| Capability | Detail |
|---|---|
| Config-driven agents | All roles defined in YAML (`config/agents/*.yaml`), no Python subclasses |
| 10-ability model scoring | Each model rated on reasoning, coding, math, writing, etc. — dot-product model selection |
| 3 execution modes | `best`, `balanced`, `cost_efficient` — blended quality vs cost |
| LiteLLM cross-provider | OpenAI, Anthropic, Bedrock, Google — swap with a string change |
| 4-tier complexity routing | Math bypass → static facts → fast LLM → full pipeline |
| Parallel worker execution | Dependency-ordered waves via `asyncio.gather()` |
| Worker role specialisation | Coder / Researcher / Writer / Analyst personas per task type |
| Worker self-correction | One VALIDATED/REVISION: self-check after initial response |
| Docker MCP sandbox | Warm container pool — `--network=none`, seccomp, cap-drop, gVisor optional |
| 3-layer memory | Agent → Team → Global LTM (Qdrant, auto-promoted every N turns) |
| RL-lite scoring | `(model, task_type) → quality` tracked in Qdrant; blended into model selection |
| Human-in-the-loop | LangGraph `interrupt()` for QUESTION: prompts |
| SHARE: protocol | Token-efficient inter-agent context broadcasting |
| FINDINGS: protocol | Workers append compact facts; injected into subsequent waves (80–90% token reduction) |
| RESEARCH: inline | Workers emit `RESEARCH: <query>` mid-task; result injected back inline |
| REMEMBER: cross-run | Workers seed global Qdrant layer for future runs |
| CODE_ANALYSIS: protocol | Workers request local codebase analysis; raw code never leaves the process |
| Token streaming | `on_token` callback; SSE streaming in API and CLI |
| Semantic cache | Qdrant-backed query deduplication (cosine similarity, configurable threshold) |
| Anthropic prompt caching | `cache_control: ephemeral` on Claude system messages |
| LiteLLM response cache | In-memory dedup for identical `(model, messages)` pairs |
| Redis task store | Run lifecycle tracked (running → completed), 7-day TTL |
| FastAPI + WebSocket API | `/api/tasks`, `/api/models`, SSE streaming, WS live output |
| Judge deep eval | Post-run LLM scoring on relevance / accuracy / completeness / actionability |
| LangSmith observability | `@traceable` on all agents; zero-cost passthrough when key absent |
| MCP integrations | GitHub, Jira, Confluence, Notion, Linear (catalog-driven, feature-flagged) |
| Test suite | **1053 tests passing** |

---

## Environment Variables

### Required

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key (for Claude models) |
| `OPENAI_API_KEY` | OpenAI API key (for GPT models) |

At least one LLM provider key is required. Both can be set.

### Optional

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | in-memory | Qdrant vector DB URL |
| `REDIS_URL` | — | Redis for task store + embedding cache |
| `E2B_API_KEY` | — | E2B cloud sandbox (fallback after Docker) |
| `LANGCHAIN_API_KEY` | — | LangSmith tracing |
| `LANGCHAIN_PROJECT` | `aifactory` | LangSmith project name |
| `AIFACTORY_MODE` | `balanced` | Default pipeline mode |
| `AIFACTORY_SANDBOX_DOCKER_ENABLED` | `false` | Enable Docker MCP sandbox |
| `AIFACTORY_RL_ENABLED` | `true` | Enable RL-lite model scoring |
| `AIFACTORY_SEMANTIC_CACHE_ENABLED` | `false` | Enable Qdrant semantic cache |
| `AIFACTORY_STREAM_TOKENS_ENABLED` | `true` | Token streaming to CLI |
| `AIFACTORY_JUDGE_DEEP_EVAL_ENABLED` | `false` | Enable judge LLM deep evaluation |
| `AIFACTORY_MEMORY_PIPELINE_ENABLED` | `true` | Wire MemoryBridge into every agent |
| `AIFACTORY_MCP_CATALOG_ENABLED` | `false` | Enable MCP tool catalog |
| `AIFACTORY_DEPLOY_MODE` | `local` | `local` (stdio) or `cloud` (HTTP session pool) |

See `.env.example` for the full list.

---

## CLI Reference

### `aifactory run` — Main entry point

```bash
uv run aifactory run "Build a REST API for a bookshop"

# Options
uv run aifactory run "task"                   # full pipeline (auto-routes via ComplexityGate)
uv run aifactory run "task" --mode best       # highest quality model selection
uv run aifactory run "task" --mode balanced   # default: quality vs cost
uv run aifactory run "task" --mode cheap      # alias for cost_efficient
uv run aifactory run --task-file task.md      # read task from file
uv run aifactory run "task" -c "extra notes"  # pass additional context
uv run aifactory run "task" --verbose         # show detailed per-agent logs
uv run aifactory run "task" --no-chat         # skip post-pipeline chat
uv run aifactory resume <run-id>              # resume a paused (QUESTION:) run
```

### `aifactory serve` — Persistent API server

```bash
uv run aifactory serve                          # localhost:8080, pre-warms all connections
uv run aifactory serve --port 9000
uv run aifactory serve --workers 4             # multi-process production mode
uv run aifactory serve --reload --log-level debug
```

### Memory

```bash
uv run aifactory memory health                  # check Qdrant + embedding provider + Redis
uv run aifactory memory stats                   # layer entry counts (agent/team/global)
uv run aifactory memory search "FastAPI auth"   # semantic search across memory
uv run aifactory memory clear --layer agent --yes
```

### Models & Agents

```bash
uv run aifactory agents list                    # all roles, tiers, default models
uv run aifactory models list                    # all models grouped by role
uv run aifactory models list --role coder       # filter by role
uv run aifactory models tiers                   # premium / standard / budget assignments
uv run aifactory info                           # config + API key status
```

### RL Scoring

```bash
uv run aifactory rl stats                       # model performance by task type
uv run aifactory rl stats --type coding         # filter to one task type
uv run aifactory rl reset --yes                 # clear all RL records
```

### Sandbox

```bash
uv run aifactory sandbox build                  # build Docker sandbox + MCP server images
uv run aifactory sandbox build --no-cache
uv run aifactory sandbox status                 # pool health (ready / in_use / total)
uv run aifactory sandbox test                   # smoke test: Python + numpy/pandas
uv run aifactory sandbox test --language bash
```

### Pipeline status

```bash
uv run aifactory status                         # list recent runs
uv run aifactory inspect <run-id>              # rich tree view of pipeline run + per-worker details
```

---

## Make Reference

```bash
# Setup
make setup              # first-time: venv, deps, .env, pre-commit
make install            # re-sync all deps
make env                # create .env from .env.example

# Databases
make db-up              # start Qdrant + Redis + Postgres in Docker
make db-down            # stop containers (data preserved)
make db-wipe            # stop + delete all volumes  ⚠ irreversible

# Sandbox
make sandbox-build      # build aifactory-sandbox Docker image
make sandbox-up         # build both sandbox + MCP server images
make sandbox-test       # smoke test the sandbox

# Pipeline
make run                # run pipeline (TASK="..." MODE=balanced)
make run-best           # TASK="..." with best models
make run-fast           # TASK="..." with cost_efficient models
make ask                # single-agent answer (TASK="...")

# Dev server
make dev                # FastAPI auto-reload on :8000
make serve              # aifactory serve on :8080 (pre-warmed)
make dev-full           # db-up + dev in one command

# Full Docker deployment
make deploy             # build all images + start all services
make deploy-down        # stop all services

# Tests
make test               # full suite
make test-fast          # stop on first failure
make test-k K="gate"    # filter by keyword
make test-cov           # HTML coverage report → htmlcov/

# Quality
make lint               # ruff check
make lint-fix           # ruff check --fix
make format             # ruff format
make typecheck          # mypy src/
make check              # lint + format-check + typecheck
make ci                 # install + check + test (full CI pipeline)

# AI Factory CLI shortcuts
make info               # config + API key status
make agents             # list all roles
make models             # list all models
make rl-stats           # RL performance scores
make memory-health      # memory system health check
make memory-stats       # layer entry counts

# Cleanup
make clean              # remove venv, dist, caches
make clean-cache        # remove __pycache__, .pytest_cache only
make clean-dumps        # delete local pipeline run dumps  ⚠ irreversible
```

---

## Project Structure

```
src/aifactory/
├── core/
│   ├── base_agent.py         # Base agent: LiteLLM, memory, token budget, @traceable
│   ├── agent_factory.py      # YAML loader, SHARE/FINDINGS/REMEMBER protocols, AgentRegistry
│   ├── model_selector.py     # 10-ability dot-product scoring + RL blend
│   ├── complexity_gate.py    # 4-tier routing (math / facts / fast_llm / full)
│   ├── memory_router.py      # 3-layer Qdrant memory (store / retrieve / promote)
│   ├── rl_scorer.py          # RL-lite: (model, task_type) → quality scores
│   ├── sandbox.py            # 3-backend execution (docker > e2b > local)
│   ├── sandbox_pool.py       # Warm Docker container pool
│   ├── semantic_cache.py     # Qdrant-backed query deduplication
│   ├── embed_provider.py     # Auto-select embedding provider + Redis cache
│   └── schemas.py            # All Pydantic models (A2A, configs, RL)
├── agents/
│   └── code_security_processor.py  # Local codebase analysis — raw code never leaves process
├── graphs/
│   └── dynamic_pipeline.py   # 8-node LangGraph pipeline (production)
├── api/
│   └── main.py               # FastAPI: /api/health, /api/tasks, /api/models, WS streaming
├── mcp/
│   ├── sandbox_server.py     # FastMCP server: execute_python/bash/js + pool_status
│   └── client.py             # MCP client (HTTP or stdio)
├── config/
│   ├── settings.py           # Pydantic Settings (env vars + .env)
│   ├── model_registry.yaml   # ~20 models with 10-ability scores + pricing
│   ├── graph.yaml            # Runtime config (mode, fallback model)
│   ├── gate_patterns.yaml    # ComplexityGate YAML-driven patterns
│   ├── tool_catalog.yaml     # Integration tool catalog
│   ├── package_registry.yaml # Pre-installed sandbox packages (injected into worker prompts)
│   └── agents/               # Per-role YAML configs (master, judge, analyser, worker, ...)
└── cli/
    ├── __init__.py           # App root + command registration
    ├── cmd_run.py            # `run` + `resume` — unified pipeline entry point
    ├── cmd_ask.py            # `ask` — single-agent answer
    ├── cmd_serve.py          # `serve` — persistent API server
    ├── cmd_status.py         # `status` + `inspect` — pipeline run viewer
    ├── cmd_agents.py         # `agents list`
    ├── cmd_cost.py           # `cost`
    ├── cmd_info.py           # `info` + `models`
    ├── cmd_memory.py         # `memory stats/search/clear/health`
    ├── cmd_sandbox.py        # `sandbox build/status/test/pull`
    └── cmd_rl.py             # `rl stats/reset`
```

---

## Tech Stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph (stateful 8-node graph) |
| LLM interface | LiteLLM (OpenAI, Anthropic, Bedrock, Google) |
| Memory / RAG | LlamaIndex + Qdrant (3-layer: agent / team / global) |
| Observability | LangSmith (`@traceable`, zero-cost when key absent) |
| Schemas | Pydantic v2 |
| Sandbox | Docker MCP → E2B → local subprocess (priority order) |
| Integrations | Composio + MCP catalog (GitHub, Jira, Notion, Linear, Slack) |
| API | FastAPI + uvicorn + WebSockets |
| Task store | Redis (run lifecycle, 7-day TTL) |
| CLI | Typer + Rich |
| Package manager | uv |
| Python | 3.13+ |
