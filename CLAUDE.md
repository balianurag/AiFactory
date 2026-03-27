# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

# Run tests
uv run pytest tests/ -v

# CLI
uv run aifactory info              # Show config
uv run aifactory run "task"        # Run a task
uv run aifactory memory-stats      # Memory stats

# Lint
uv run ruff check src/ tests/
```

### Project Structure

```
src/aifactory/
├── core/
│   ├── base_agent.py       # Base agent with LLM, memory, token tracking
│   ├── schemas.py          # ALL Pydantic models (A2A messages, configs, RL)
│   └── memory_router.py    # 3-layer Qdrant memory (store/retrieve/promote)
├── agents/
│   ├── worker.py           # Execution specialist (coder/researcher/writer)
│   ├── team_lead.py        # Tactical coordinator (decompose → delegate → consolidate)
│   └── project_manager.py  # Strategic coordinator (plan → delegate → deliver)
├── graphs/
│   └── single_team.py      # MVP pipeline: PM → TL → Workers
├── prompts/                # System prompts per role (.md files)
├── config/
│   ├── settings.py         # Pydantic Settings (env vars + .env)
│   └── models.yaml         # Model configs & pricing per tier
└── cli.py                  # Typer CLI entry point
```

### Implementation Status

- [x] Phase 0: Foundation (schemas, base agent, memory router, settings)
- [x] Phase 1 (partial): Single-team pipeline (PM → TL → Workers graph)
- [ ] Phase 2: Assessment Team + Dynamic Hierarchy Builder
- [ ] Phase 3: RL Sidecar + Agents-as-Judge + Model Router
- [ ] Phase 4: Hardening (circuit breakers, budget caps, observability)

### Key Design Decisions

1. **All A2A messages are Pydantic models** — no free-form text between agents.
2. **Memory promotion every N turns** — agent layer summarised → team layer.
3. **In-memory Qdrant for dev** — no Docker needed to get started.
4. **LiteLLM for model abstraction** — switch providers with a string change.
5. **1:3 ratio** — max 3 subordinates per lead. Dynamic, not hardcoded.
6. **Token budgets per agent** — hard caps prevent runaway costs.
7. **uv for package management** — 10-100x faster than pip.

### Documentation

Project docs live in `docs/`:
- `docs/architecture.md` — full system architecture and design
- `docs/progress.md` — what's built, current status, what's next
- `docs/decisions.md` — key technical decisions and rationale
