# Progress Tracker

## Phase Overview

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 0** | Foundation — scaffolding, schemas, base agent, memory, config | ✅ Complete |
| **Phase 1** | Single-Team Pipeline — PM → TL → Workers end-to-end | ✅ Complete |
| **Phase 2** | Assessment + Dynamic Hierarchy — system designs its own structure | 🔜 Not started |
| **Phase 3** | Intelligence Layer — RL Sidecar, Agents-as-Judge, Model Router | 🔜 Not started |
| **Phase 4** | Hardening — circuit breakers, budget caps, observability, UI | 🔜 Not started |

---

## Phase 0: Foundation ✅

**Completed: 27 March 2026**

### What was built

| Component | File | Description |
|-----------|------|-------------|
| A2A Schemas | `core/schemas.py` | 25+ Pydantic models — every message type (TaskAssignment, TaskResult, StatusUpdate, Question, Answer, ReviewRequest, ReviewResult, Escalation), agent configs, team configs, project plans, RL performance records, memory entries |
| Base Agent | `core/base_agent.py` | Abstract agent with LLM calls via LiteLLM, memory integration, token tracking, budget enforcement, conversation history, auto-promotion to team memory |
| Memory Router | `core/memory_router.py` | 3-layer Qdrant memory system — store/retrieve/promote/dedup across Agent/Team/Global layers. In-memory mode for dev (no Docker needed) |
| Settings | `config/settings.py` | Pydantic Settings loaded from `.env` — API keys, model tiers, memory configs, hierarchy ratios, budget limits |
| Model Config | `config/models.yaml` | Model pricing, context windows, tier mappings per agent role |
| Project setup | `pyproject.toml` | Python 3.12+, uv-managed, all dependencies declared |

### Tests (6 passing)
- Memory store/retrieve, multi-level storage, promotion pipeline, auto-promotion threshold, deduplication, agent cleanup

---

## Phase 1: Single-Team Pipeline ✅

**Completed: 27 March 2026**

### What was built

| Component | File | Description |
|-----------|------|-------------|
| Worker Agent | `agents/worker.py` | Execution specialist with specialty (coder/researcher/writer/reviewer/tester). Takes TaskAssignment, produces TaskResult with artifacts. |
| Team Lead Agent | `agents/team_lead.py` | Tactical coordinator. Decomposes objectives → worker tasks, delegates, collects results, consolidates into unified deliverable. |
| Project Manager Agent | `agents/project_manager.py` | Strategic coordinator. Plans project → team objectives, delegates to TLs, merges into final PMDelivery. |
| LangGraph Pipeline | `graphs/single_team.py` | Stateful graph: `setup_agents → execute_project → END`. Creates PM → TL → 3 Workers and runs full lifecycle. |
| CLI | `cli.py` | `aifactory run "task"`, `aifactory info`, `aifactory memory-stats` — Rich-formatted output |
| System Prompts | `prompts/*.md` | Role-specific system prompts for worker, team_lead, pm, researcher, assessor |

### Tests (7 passing)
- Schema creation, serialisation, default values, composite score calculation, peer score integration, memory entry defaults, project plan round-trip

### How to run
```bash
# Set up
cp .env.example .env   # Add at least one API key
uv sync

# Run a task
uv run aifactory run "Build a FastAPI CRUD app for a todo list"

# Check config
uv run aifactory info
```

---

## Phase 2: Assessment + Dynamic Hierarchy (Next)

### Planned components

| Component | File | Description |
|-----------|------|-------------|
| Master Researcher | `agents/master_researcher.py` | Takes raw idea → research doc (market, tech, risks, SWOT) |
| Assessment Team | `agents/assessment_team.py` | 2 assessors + 1 Devil's Advocate → project plan + hierarchy blueprint |
| Hierarchy Builder | `core/hierarchy_builder.py` | Takes ProjectPlan → dynamically constructs the LangGraph with correct number of PMs/TLs/Workers |
| Consolidator | `agents/consolidator.py` | Master Agent merges all PM outputs into final delivery |
| Full Pipeline | `graphs/full_pipeline.py` | Complete graph: Research → Assess → Build Hierarchy → Execute → Consolidate |

---

## Phase 3: Intelligence Layer (Planned)

| Component | File | Description |
|-----------|------|-------------|
| RL Scorer | `core/rl_scorer.py` | Score every agent after task on 6 metrics, store in Global LTM |
| Agents-as-Judge | `core/judge.py` | Peer review step — agents score each other's output |
| Model Router | `core/model_router.py` | Auto-select LLM per agent based on RL scores + task complexity |
| Adaptive Promotion | `core/promotion_policy.py` | Replace fixed N-turn interval with dynamic trigger based on info entropy |

---

## Phase 4: Hardening (Planned)

- Circuit breakers (PM can flag "structure isn't working" mid-run → re-assess)
- Hard budget caps per project with cost tracking
- Auto-Architect Switch (Assessment Team can choose "debate" architecture for high-uncertainty tasks)
- Observability dashboard
- Web UI for human-in-the-loop
