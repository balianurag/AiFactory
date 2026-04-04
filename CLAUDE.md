# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. Also after new work keep updating the claude.md and spec.md

## Project Overview

**AI Factory** — A self-assembling, self-optimising multi-agent system that turns a raw human idea into production-grade output with minimal manual coordination.

### Architecture

```
Human Idea → Complexity Assessment (gpt-4o-mini gate)
  → TRIVIAL:    Direct Worker (skip PM/TL)
  → MODERATE+:  Project Manager → Team Lead(s) [1:3 ratio] → Worker Agents
  → Consolidation → Final Delivery
  → Persist to PostgreSQL + cache in Redis
```

Key features:
- **Dynamic hierarchy construction** at runtime (not pre-defined crews)
- **Complexity-based routing**: trivial tasks skip PM/TL overhead
- **3-layer memory system**: Agent → Team → Global LTM (all via Qdrant with real OpenAI embeddings)
- **Persistent storage**: PostgreSQL for tasks/agents/messages, Redis for hot caching
- **LangSmith observability**: every LLM call + memory op traced via `@traceable`
- **Structured A2A messaging** via Pydantic (5-10× token savings vs chatty)
- **Cost tracking**: per-agent input/output token counts + USD cost
- **Model routing**: premium models for leads, budget models for workers

### Tech Stack

- **Python 3.12+**, managed with `uv` + `pyproject.toml` (hatchling build)
- **LangGraph** — stateful graph orchestration
- **OpenAI SDK** (`openai>=1.40.0`) — async LLM calls + embeddings (`text-embedding-3-small`)
- **Pydantic v2** — structured schemas for all A2A communication
- **SQLAlchemy async + asyncpg** — PostgreSQL persistence (tasks, agents, messages)
- **Redis async** (`redis[hiredis]>=5.0`) — hot cache with TTLs (30s active, 5min completed)
- **Qdrant** — vector DB for 3-layer memory (Qdrant Cloud in production, in-memory for dev)
- **LangSmith** — observability: traces, cost tracking, run history
- **Rich + Typer** — CLI interface
- **pytest** — 208+ tests

### Commands

```bash
# Install (requires uv: https://docs.astral.sh/uv/)
uv sync --dev

# Run all tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_schemas.py -v

# Run a single test by name
uv run pytest tests/ -v -k "test_composite_score"

# Type check
uv run mypy src/

# CLI
uv run aifactory info              # Show config
uv run aifactory run "task"        # Run a task
uv run aifactory memory-stats      # Memory stats

# Model registry
uv run aifactory models list                    # All roles + allowed models
uv run aifactory models list --role coder       # Filter by agent role
uv run aifactory models tiers                   # Active premium/standard/budget assignments

# Agent management
uv run aifactory agents list                    # All roles, tiers, default models

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/
uv run ruff format --check src/ tests/  # CI check (no changes)

# Run a task from a file
uv run aifactory run --task-file task.md --workers 5 --verbose
```

### Project Structure

```
src/aifactory/
├── core/
│   ├── base_agent.py       # Base agent with LLM (OpenAI SDK), memory, token tracking, @traceable
│   ├── assessor.py         # Complexity gate (trivial/moderate/complex) + cost_usd() tracking
│   ├── database.py         # Async PostgreSQL CRUD (tasks, agents, messages) via SQLAlchemy
│   ├── redis_client.py     # Async Redis cache with TTLs and graceful degradation
│   ├── schemas.py          # ALL Pydantic models (A2A messages, configs, RL)
│   ├── memory_router.py    # 3-layer Qdrant memory (store/retrieve/promote) with @traceable
│   └── observability.py    # LangSmith setup, @traceable decorator, run/task-trace fetchers
├── agents/
│   ├── worker.py           # Execution specialist (coder/researcher/writer)
│   ├── team_lead.py        # Tactical coordinator (decompose → delegate → consolidate)
│   └── project_manager.py  # Strategic coordinator (plan → delegate → deliver)
├── graphs/
│   └── single_team.py      # Pipeline: assess → route → (direct_worker | PM→TL→Workers)
├── api/
│   └── main.py             # FastAPI v0.3.0: /api/health, /api/models, /api/tasks, /api/tasks/{id}/traces, /api/observability/*, WS /api/ws/{id}
├── prompts/                # System prompts per role (.md files)
├── config/
│   ├── settings.py         # Pydantic Settings (env vars + .env, incl. DB, Redis, LangSmith)
│   └── models.yaml         # Curated model registry: allowed models per agent role + metadata
└── cli.py                  # Typer CLI (run, info, memory-stats, models list/tiers, agents list)
```

### Implementation Status

- [x] Phase 0: Foundation (schemas, base agent, memory router, settings)
- [x] Phase 1: Single-team pipeline (PM → TL → Workers graph) — fully functional via CLI
- [x] Phase 1.5: HTTP/WebSocket API (`/api/tasks`, `/api/models`, WS streaming) — deployed on Render
- [x] Phase 1.5: Curated model registry (`models.yaml`) — 14 agent roles, 11 approved models, UI metadata
- [x] Phase 1.5: CLI model/agent commands (`models list`, `models tiers`, `agents list`)
- [x] Phase 1.5: Per-task model overrides (`pm_model`, `worker_tier` in `TaskRequest`)
- [x] Phase 1.5: Frontend ChatView + ObservabilityView wired to real backend
- [x] Phase 2.0: LangSmith observability — `@traceable` on BaseAgent.think() + MemoryRouter ops, API endpoints incl. /api/tasks/{id}/traces, UI Traces view
- [x] Phase 2.5: Real OpenAI embeddings (`text-embedding-3-small`) in MemoryRouter (fallback to stub when no key)
- [x] Phase 2.5: Complexity assessor gate — trivial tasks skip PM/TL overhead
- [x] Phase 2.5: Cost tracking per agent (input/output tokens + USD via `_MODEL_PRICING`)
- [x] Phase 3.0: PostgreSQL persistence — tasks, agents, messages (SQLAlchemy async + asyncpg)
- [x] Phase 3.0: Redis hot cache — 30s active tasks, 5min completed (graceful degradation)
- [x] Phase 3.0: Render managed Postgres + Redis provisioned and migrated
- [ ] Phase 3.1: Assessment Team agents (MasterResearcher, Assessor, DevilsAdvocate, HierarchyBuilder)
- [ ] Phase 3.2: Dynamic Pipeline Graph (multi-team execution)
- [ ] Phase 4: RL Sidecar + Agents-as-Judge + Model Router
- [ ] Phase 5: Hardening (circuit breakers, budget middleware, OpenTelemetry)

### Key Design Decisions

1. **All A2A messages are Pydantic models** — no free-form text between agents.
2. **Memory promotion every N turns** — agent layer summarised → team layer. Interval set via `AIFACTORY_MEMORY_PROMOTION_INTERVAL` (default: 10).
3. **In-memory Qdrant for dev** — no Docker needed. Falls back automatically if `QDRANT_URL` is unreachable.
4. **OpenAI SDK for LLM calls** — `openai.AsyncOpenAI` for agent LLM calls, sync `openai.OpenAI` for embeddings.
5. **1:3 ratio** — max 3 subordinates per lead. Configured via `AIFACTORY_MAX_SUBORDINATES_PER_LEAD`.
6. **Token budgets per agent** — hard caps enforced in `BaseAgent.think()`. Raises `AgentBudgetError` when exceeded.
7. **uv for package management** — 10-100x faster than pip. Venv: `aifactory_venv` (parallel to project dir).
8. **Real embeddings by default** — `MemoryRouter` uses OpenAI `text-embedding-3-small`. Falls back to SHA-256 stub only when `OPENAI_API_KEY` is absent.
9. **Env var naming** — LLM API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) have **no prefix**. All other settings use the `AIFACTORY_` prefix. LangSmith uses `LANGCHAIN_` prefix. DB/Redis use `DATABASE_URL` / `REDIS_URL`.
10. **Tests are async-native** — `asyncio_mode = "auto"` in `pyproject.toml`; no `@pytest.mark.asyncio` needed.
11. **LangSmith tracing is zero-cost when disabled** — `@traceable` decorator is a passthrough when `LANGCHAIN_API_KEY` is absent. No overhead, no failures. Memory ops (store/retrieve/promote) are also traced.
12. **PostgreSQL for persistence, Redis for speed** — all task state survives restarts. Redis is optional (graceful degradation).
13. **Complexity gate before execution** — `assess_complexity()` via gpt-4o-mini routes trivial tasks to a single Worker, saving PM/TL overhead.

### Agent Execution Pattern

Every level in the hierarchy follows the same **decompose → delegate → consolidate** lifecycle:

- **Worker** (`worker.py`): Receives a `TaskAssignment`, calls `think()` to produce work, returns `TaskResult` with `Artifact` objects.
- **Team Lead** (`team_lead.py`): Calls LLM to decompose an objective into 1–3 `WorkerTaskSpec`s, delegates to workers sequentially, then calls LLM to consolidate results into a single `TaskResult`.
- **Project Manager** (`project_manager.py`): Calls LLM to produce a `PMPlan` with 1–3 team objectives, delegates to Team Leads, consolidates into `PMDelivery` with quality metrics.

All LLM calls go through `BaseAgent.think()`, which enforces token budgets and iteration limits. Budget violations raise `AgentBudgetError`; failed structured-output parsing raises `StructuredOutputError`. Both TL and PM have fallback logic (raw text merge) if their consolidation LLM call fails.

### Documentation

Project docs live in `docs/`:
- `docs/architecture-v2.md` — full system architecture (LangGraph + LangSmith)
- `docs/architecture.md` — original v1 architecture (for reference)
- `docs/sprints.md` — sprint plan with stories and acceptance criteria
- `docs/progress.md` — what's built, current status, what's next
- `docs/decisions.md` — key technical decisions and rationale
