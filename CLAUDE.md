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
```

Raw `uv run aifactory <cmd>` also works for all application commands.

### Project Structure

```
src/aifactory/
├── core/
│   ├── base_agent.py       # Base agent with LiteLLM, memory, token tracking, @traceable
│   ├── assessor.py         # Task complexity classifier (trivial/moderate/complex)
│   ├── model_selector.py   # 10-ability dot-product model scoring engine
│   ├── agent_factory.py    # YAML agent loader, SHARE: protocol, AgentRegistry
│   ├── schemas.py          # ALL Pydantic models (A2A messages, configs, RL)
│   ├── memory_router.py    # 3-layer Qdrant memory (store/retrieve/promote)
│   ├── rate_limiter.py     # slowapi rate limiting (4 tiers)
│   ├── sandbox.py          # Code execution sandbox
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
├── config/
│   ├── settings.py         # Pydantic Settings (env vars + .env, incl. LangSmith config)
│   ├── models.yaml         # Legacy model registry for /api/models UI endpoint
│   ├── model_registry.yaml # ★ ~20+ models with 10-ability scores + pricing
│   ├── graph.yaml          # ★ Graph runtime config (mode, fallback model)
│   └── agents/             # ★ YAML configs per agent role:
│       ├── master.yaml     #   - Coordinator / synthesiser
│       ├── judge.yaml      #   - Quality reviewer
│       ├── analyser.yaml   #   - Task breakdown / direct-answer detection
│       ├── researcher.yaml #   - Pre-analysis and execution research
│       ├── execution_lead.yaml # - Worker dispatcher + consolidator
│       └── worker.yaml     #   - Template for per-task workers
└── cli.py                  # Typer CLI (run, info, memory-stats, models list/tiers, agents list)
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
- [x] Phase 2.5: Code execution sandbox
- [x] Phase 2.5: Worker self-correction loop
- [ ] Phase 3: LlamaIndex memory integration (replace hash-based MemoryRouter with real embeddings)
- [ ] Phase 4: Hardening (circuit breakers, budget caps, Redis task store)

### Key Design Decisions

1. **Config-driven agents** — Agent behaviour comes from YAML files, not Python subclasses. One `PipelineAgent` class handles all roles.
2. **10-ability model scoring** — Each model rated 0-3 on reasoning, coding, math, writing, instruction_following, tool_use, knowledge, speed, cost_efficiency, context_handling. Agent YAML files declare required abilities. Dot-product scoring selects the best model per role+mode.
3. **3 execution modes** — `best` (quality-first), `balanced` (default), `cost_efficient`. Each mode has an alpha weight that blends quality vs cost/speed.
4. **SHARE: protocol** — Agents append `SHARE: key: value` lines to their output. These are extracted and merged into a shared context dict visible to downstream agents.
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

### Agent Execution Pattern

The 8-node dynamic pipeline follows this flow:

1. **master_receive** — Acknowledges the request, seeds shared context with the objective
2. **judge_brief** — Defines success criteria and what will be evaluated
3. **analyse** — Breaks request into atomic tasks (or returns a direct answer for trivial queries)
4. **researcher** (conditional) — Gathers research for pre-analysis or execution
5. **reanalyse** (conditional) — Refines task breakdown after research
6. **execution_lead** — Plans worker dispatch with per-task ability requirements, spawns N workers dynamically with per-task model selection, consolidates results
7. **judge_review** — Evaluates each agent's performance, produces quality metrics
8. **master_respond** — Synthesises final response for the human (no mention of internal process)

Workers support:
- **NOT_NEEDED** self-skip — if a worker determines its task is already covered
- **QUESTION:** interrupts — worker can pause and ask the human a question (via LangGraph interrupt())
- **SHARE:** protocol — share discovered context with downstream agents

### Documentation

Project docs live in `docs/`:
- `docs/architecture-v2.md` — full system architecture (LangGraph + LlamaIndex + LangSmith)
- `docs/architecture.md` — original v1 architecture (for reference)
- `docs/sprints.md` — sprint plan with stories and acceptance criteria
- `docs/progress.md` — what’s built, current status, what’s next
- `docs/decisions.md` — key technical decisions and rationale
