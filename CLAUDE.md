# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. Also after new work keep updating the claude.md and spec.md

## Project Overview

**AI Factory** — A self-assembling, self-optimising multi-agent system that turns a raw human idea into production-grade output with minimal manual coordination.

### Architecture

```
Human Idea → Master Researcher → Assessment Team (2 + Devil's Advocate)
  → Project Manager(s) [1:3 ratio] → Team Lead(s) [1:3 ratio] → Worker Agents
  → Consolidation → Final Delivery
```

Key features:
- **Dynamic hierarchy construction** at runtime (not pre-defined crews)
- **3-layer memory system**: Agent → Team → Global LTM (all via Qdrant)
- **RL Sidecar** with Agents-as-Judge for continuous agent scoring
- **Structured A2A messaging** via Pydantic (5-10× token savings vs chatty)
- **Model routing**: premium models for leads, budget models for workers

### Tech Stack

- **Python 3.12+**, managed with `uv` + `pyproject.toml` (hatchling build)
- **LangGraph** — stateful graph orchestration
- **LiteLLM** — unified LLM interface (any provider)
- **Pydantic v2** — structured schemas for all A2A communication
- **Qdrant** — vector DB for 3-layer memory (in-memory mode for local dev)
- **Rich + Typer** — CLI interface
- **pytest** — testing

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
│   ├── base_agent.py       # Base agent with LLM, memory, token tracking, @traceable
│   ├── schemas.py          # ALL Pydantic models (A2A messages, configs, RL)
│   ├── memory_router.py    # 3-layer Qdrant memory (store/retrieve/promote)
│   └── observability.py    # LangSmith setup, @traceable decorator, run fetchers
├── agents/
│   ├── worker.py           # Execution specialist (coder/researcher/writer)
│   ├── team_lead.py        # Tactical coordinator (decompose → delegate → consolidate)
│   └── project_manager.py  # Strategic coordinator (plan → delegate → deliver)
├── graphs/
│   └── single_team.py      # MVP pipeline: PM → TL → Workers (accepts pm_model/worker_model)
├── api/
│   └── main.py             # FastAPI app: /api/health, /api/models, /api/tasks, /api/observability/*, WS /api/ws/{id}
├── prompts/                # System prompts per role (.md files)
├── config/
│   ├── settings.py         # Pydantic Settings (env vars + .env, incl. LangSmith config)
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
- [x] Phase 1.5: Frontend E2E integration — all views wired to real backend API (no mock data)
- [x] Phase 2.0: LangSmith observability — `@traceable` on BaseAgent.think(), API endpoints, UI Traces view
- [ ] Phase 2.1: LlamaIndex memory migration (replace hash-based MemoryRouter with real embeddings)
- [ ] Phase 2.2: Assessment Team + Dynamic Hierarchy Builder
- [ ] Phase 3: RL Sidecar + Agents-as-Judge + Model Router
- [ ] Phase 4: Hardening (circuit breakers, budget caps, Redis task store)

### Key Design Decisions

1. **All A2A messages are Pydantic models** — no free-form text between agents.
2. **Memory promotion every N turns** — agent layer summarised → team layer. Interval set via `AIFACTORY_MEMORY_PROMOTION_INTERVAL` (default: 10).
3. **In-memory Qdrant for dev** — no Docker needed. Falls back automatically if `QDRANT_URL` is unreachable.
4. **LiteLLM for model abstraction** — switch providers with a string change in `.env`.
5. **1:3 ratio** — max 3 subordinates per lead. Configured via `AIFACTORY_MAX_SUBORDINATES_PER_LEAD`.
6. **Token budgets per agent** — hard caps enforced in `BaseAgent.think()`. Raises `AgentBudgetError` when exceeded.
7. **uv for package management** — 10-100x faster than pip. Venv: `aifactory_venv` (parallel to project dir).
8. **Stub embeddings in dev** — `MemoryRouter` uses a SHA-256 hash-based pseudo-embedding by default. Pass a real `embedding_fn` for production semantic search.
9. **Env var naming** — LLM API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) have **no prefix**. All other settings use the `AIFACTORY_` prefix. LangSmith uses `LANGCHAIN_` prefix.
10. **Tests are async-native** — `asyncio_mode = "auto"` in `pyproject.toml`; no `@pytest.mark.asyncio` needed.
11. **LangSmith tracing is zero-cost when disabled** — `@traceable` decorator is a passthrough when `LANGCHAIN_API_KEY` is absent. No overhead, no failures.
12. **Stack: LangGraph + LlamaIndex + LangSmith** — LangGraph for orchestration, LlamaIndex for memory/RAG (Phase 2.1), LangSmith for observability.

### Agent Execution Pattern

Every level in the hierarchy follows the same **decompose → delegate → consolidate** lifecycle:

- **Worker** (`worker.py`): Receives a `TaskAssignment`, calls `think()` to produce work, returns `TaskResult` with `Artifact` objects.
- **Team Lead** (`team_lead.py`): Calls LLM to decompose an objective into 1–3 `WorkerTaskSpec`s, delegates to workers sequentially, then calls LLM to consolidate results into a single `TaskResult`.
- **Project Manager** (`project_manager.py`): Calls LLM to produce a `PMPlan` with 1–3 team objectives, delegates to Team Leads, consolidates into `PMDelivery` with quality metrics.

All LLM calls go through `BaseAgent.think()`, which enforces token budgets and iteration limits. Budget violations raise `AgentBudgetError`; failed structured-output parsing raises `StructuredOutputError`. Both TL and PM have fallback logic (raw text merge) if their consolidation LLM call fails.

### Documentation

Project docs live in `docs/`:
- `docs/architecture-v2.md` — full system architecture (LangGraph + LlamaIndex + LangSmith)
- `docs/architecture.md` — original v1 architecture (for reference)
- `docs/sprints.md` — sprint plan with stories and acceptance criteria
- `docs/progress.md` — what’s built, current status, what’s next
- `docs/decisions.md` — key technical decisions and rationale
