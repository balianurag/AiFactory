# AiFactory — Feature Specification

**Version:** 0.6.0
**Phase:** 6 — Memory Bridge + ComplexityGate v2
**Last updated:** 2026-04-11

This document is the authoritative spec for what has been built. Use it as the
baseline for spec-driven development of future phases. Each new feature should
be specced here before implementation begins.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Backend — Current Architecture (Built)](#2-backend--current-architecture-built)
3. [Memory Bridge — Phase 2.9 (Built)](#3-memory-bridge--phase-29-built)
4. [ComplexityGate v2 — Phase 2.8 (Built)](#4-complexitygate-v2--phase-28-built)
5. [Frontend — Phase 1 (Built)](#5-frontend--phase-1-built)
6. [Deployment & Infrastructure (Built)](#6-deployment--infrastructure-built)
7. [Data Contracts](#7-data-contracts)
8. [Configuration Reference](#8-configuration-reference)
9. [Roadmap — Future Phases (Not Yet Built)](#9-roadmap--future-phases-not-yet-built)

---

## 1. System Overview

AiFactory is a self-assembling, self-optimising multi-agent orchestration system
that accepts a raw human task description and returns production-grade output by
dynamically constructing an 8-node agent pipeline from YAML configs.

```
Human Task
  → POST /api/tasks {task, context?, mode?}
      → build_dynamic_pipeline(mode)
          → master_receive    (coordinator acknowledges request)
          → judge_brief       (defines quality criteria)
          → analyse           (task breakdown or direct-answer bypass)
          → [researcher]?     (optional pre-analysis research)
          → [reanalyse]?      (refine tasks after research)
          → [researcher]?     (optional execution research)
          → execution_lead    (spawns N workers dynamically, per-task model selection)
          → judge_review      (quality assessment of all agents)
          → master_respond    (synthesises final response)
      → result
  ← result via GET /api/tasks/{id} or WS /api/ws/{id}
```

**Key properties:**
- All agents defined in YAML configs (`config/agents/*.yaml`)
- 10-ability model scoring: each model rated on reasoning, coding, math, writing, etc.
- 3 mode profiles: `best`, `balanced` (default), `cost_efficient`
- Dynamic worker spawning: Execution Lead creates per-task workers with ability-scored models
- **ComplexityGate v2**: 4-tier pre-pipeline routing (math bypass → static facts → fast_llm → full)
- **MemoryBridge**: every PipelineAgent retrieves relevant memories before thinking and stores outputs after
- SHARE: protocol for inter-agent context broadcasting + team-layer memory persistence
- LLM calls via LiteLLM (`litellm.acompletion()`) for cross-provider support
- 3-layer vector memory: Agent → Team → Global (Qdrant + LlamaIndex; HuggingFace/OpenAI/stub embeddings)
- LangSmith observability: every LLM call + memory op traced
- API rate limiting: per-IP via slowapi (10/min tasks, 60/min reads, 30/min memory, 5/min WS)
- Human-in-the-loop: LangGraph `interrupt()` for worker QUESTION: prompts

---

## 2. Backend — Phase 1 (Built)

### 2.1 HTTP API

All routes are prefixed `/api`. FastAPI auto-generates Swagger at `/api/docs`.

#### `GET /api/health`

| Field | Value |
|---|---|
| Auth | None |
| Response | `200 OK` |
| Body | `{status, version, phase, timestamp}` |
| Used by | Render health-check, uptime monitors |

#### `GET /api/models`

| Field | Value |
|---|---|
| Auth | None |
| Response | `200 OK` |
| Body | `{ "roles": { [role]: { "description": str, "models": [ModelInfo] } } }` |
| Source | `config/models.yaml` (LRU-cached, read once per process) |
| Used by | UI model picker in ChatView; `aifactory models list` CLI |

`ModelInfo` shape: `{ id, display, provider, context_k, input_per_1m, output_per_1m, strength, tags }`

**14 agent roles:** `orchestrator`, `project_manager`, `devils_advocate`, `assessor`, `team_lead`, `coder`, `researcher`, `writer`, `tester`, `reviewer`, `architect`, `devops`, `data_analyst`, `rl_judge`

#### `POST /api/tasks`

| Field | Value |
|---|---|
| Auth | None |
| Response | `202 Accepted` |
| Request body | `{task, context?, mode?}` |
| Response body | `{task_id: uuid, status: "pending", created_at: ISO8601}` |
| Behaviour | Fire-and-forget: returns immediately, pipeline runs in background via `asyncio.create_task()` |
| Storage | PostgreSQL (tasks, agents, messages) with Redis hot cache |

**Validation rules:**
- `task` is required, must be non-empty string
- `mode` must be one of `"best"`, `"balanced"`, `"cost_efficient"`; defaults to value from `graph.yaml` (balanced)
- Legacy fields (`num_workers`, `pm_model`, `worker_tier`) accepted but ignored by the dynamic pipeline
- Missing `task` → HTTP 422

#### `GET /api/tasks/{task_id}`

| Field | Value |
|---|---|
| Auth | None |
| Response | `200 OK` or `404 Not Found` |
| Response body | `TaskStatus` schema (see §5) |
| Status values | `pending → running → completed | failed` |
| `result` field | Null until `completed`; contains full `PMDelivery` |
| `error` field | Null unless `failed`; contains exception message |

#### `WS /api/ws/{task_id}`

| Field | Value |
|---|---|
| Protocol | WebSocket (RFC 6455) |
| URL | `wss://aifactory-4e5k.onrender.com/api/ws/{task_id}` |
| Auth | None |
| Direction | Server → Client only (client messages ignored, reserved for pause/resume) |
| Timeout | 30 s inactivity closes connection |
| Error handling | Unknown `task_id` → sends `{type:"error"}` then closes |

**Event types pushed by server (in order):**

```jsonc
// 1. Pipeline starts
{"type": "status", "status": "running", "progress": 0.05}

// 2. Agent assembly
{"type": "agent_hierarchy", "message": "Assembling agent hierarchy...", "progress": 0.1}

// 3a. Success
{"type": "completed", "progress": 1.0, "result": { /* PMDelivery */ }}

// 3b. Failure
{"type": "error", "message": "<exception message>"}
```

### 2.2 Pipeline — `dynamic_pipeline.py`

Entry point: `aifactory.graphs.dynamic_pipeline.build_dynamic_pipeline(mode)`

```
build_dynamic_pipeline(mode="balanced")
  → Loads YAML configs: config/agents/*.yaml + config/model_registry.yaml + config/graph.yaml
  → Creates 5 named agents: master, judge, analyser, researcher, execution_lead
  → Each agent's model selected via 10-ability dot-product scoring × mode profile
  → Compiles 8-node LangGraph with 3 conditional edges

Pipeline execution flow:
  1. master_receive    — acknowledges request, seeds shared context
  2. judge_brief       — defines success criteria
  3. analyse           — breaks request into tasks (or returns direct_answer)
     └─ conditional: direct_answer → master_respond (skip all workers)
     └─ conditional: research_for_analysis → researcher → reanalyse
     └─ conditional: research_needed → researcher
  4. researcher        — gathers research (if needed)
  5. reanalyse         — refines tasks after research (if needed)
  6. execution_lead    — plans worker dispatch, spawns N workers:
     └─ Per-task model selection based on ability requirements
     └─ Workers support NOT_NEEDED self-skip
     └─ Workers support QUESTION: interrupts (LangGraph interrupt())
     └─ Workers support SHARE: protocol for context broadcasting
     └─ Consolidates all worker results
  7. judge_review      — evaluates each agent, produces quality metrics
  8. master_respond    — synthesises final response for human
```

**Agent roles (YAML-configured):**
- `master` — Coordinator/synthesiser (always uses best_model)
- `judge` — Quality reviewer
- `analyser` — Task breakdown / direct-answer detection
- `researcher` — Pre-analysis and execution research
- `execution_lead` — Worker dispatcher + consolidator
- `worker` — Template for per-task dynamic workers

### 2.2.1 Legacy Pipeline — `single_team.py`

The original 4-node PM → TL → Workers pipeline is preserved for backward
compatibility but is no longer used by `_run_pipeline()`. It can still be
invoked directly via `SingleTeamPipeline`.

### 2.3 Base Agent (`base_agent.py`)

All agents inherit from `BaseAgent`:

| Capability | Detail |
|---|---|
| LLM interface | OpenAI SDK `AsyncOpenAI` — `chat.completions.create()` |
| Token budget | Hard cap per agent; raises `AgentBudgetError` when exceeded |
| Iteration cap | Configurable `max_iterations` per agent |
| Memory | `MemoryRouter` injected; store/retrieve per turn |
| Observability | `@traceable(name="agent_think", run_type="llm")` — traces to LangSmith |
| Retry logic | `tenacity` with exponential backoff on LLM errors |
| Structured output | Pydantic model parsing; fallback to raw text merge on failure |

### 2.4 Memory System (`memory_router.py`)

Three-layer Qdrant vector store with LlamaIndex as the query engine layer:

| Layer | Scope | Promoted from |
|---|---|---|
| `agent` | Private per-agent episodic memory | — |
| `team` | Shared per-team working memory | agent layer |
| `global` | Cross-project long-term memory | team layer |

**Architecture:** LlamaIndex `VectorStoreIndex` sits on top of Qdrant for advanced
retrieval (query engines, metadata filtering, reranking). Raw Qdrant is still used
for fast store/retrieve; LlamaIndex is used for `get_query_engine()` and `query()`.

**Embedding (LlamaIndex-managed):**
- Default: HuggingFace `BAAI/bge-small-en-v1.5` (384-dim, local, free via `llama-index-embeddings-huggingface`)
- Optional: OpenAI `text-embedding-3-small` (1536-dim, paid via `llama-index-embeddings-openai`)
- Tests: SHA-256 hash stub (`_stub_embed`) — deterministic, no model download

**LlamaIndex components:**
- `VectorStoreIndex` — one per collection, lazily cached
- `QdrantVectorStore` — bridges LlamaIndex to Qdrant collections
- `MetadataFilters` / `ExactMatchFilter` — agent_id, team_id scoping
- `FlagEmbeddingReranker` (optional) — BGE reranker for retrieval quality

**Key methods:**

| Method | Purpose |
|---|---|
| `store(entry)` | Upsert to Qdrant (raw, fast, deduplication via content hash) |
| `retrieve(query, levels, ...)` | Semantic search across layers via Qdrant |
| `get_query_engine(layer, ...)` | **NEW** — returns LlamaIndex query engine with filters + reranker |
| `query(query_text, layer, ...)` | **NEW** — high-level LlamaIndex query, returns synthesised text |
| `promote(agent_id, ...)` | Summarise and copy from lower to upper layer |
| `get_stats()` | Point counts per collection |
| `should_promote(agent_id)` | Check if N unpromoted entries ≥ threshold |
| `clear_agent(agent_id)` | Delete all entries for an agent |

**Shared singleton:** The API uses a single `MemoryRouter` instance (via `_get_memory_router()`)
so data persists across HTTP requests. In-memory Qdrant fallback when remote is unreachable.

**Promotion:** every `AIFACTORY_MEMORY_PROMOTION_INTERVAL` turns (default: 10),
agent-layer summaries are written to the team layer.

**Tracing:** `store()`, `retrieve()`, and `promote()` are all decorated with `@traceable`
and appear in LangSmith traces when tracing is enabled.

**Qdrant mode:**
- Dev: in-memory (no Docker needed, auto-fallback)
- Production: Qdrant Cloud via `QDRANT_URL` + `QDRANT_API_KEY`

**Dependencies (pyproject.toml):**
- `llama-index-core>=0.14.20`
- `llama-index-vector-stores-qdrant>=0.10.0`
- `llama-index-embeddings-huggingface>=0.7.0`
- `llama-index-embeddings-openai>=0.6.0`
- `llama-index-postprocessor-flag-embedding-reranker>=0.5.0`
- `llama-index-readers-file>=0.6.0`

### 2.5 RL Sidecar Schemas (defined, not yet executed)

`AgentPerformanceRecord` schema is fully defined in `schemas.py`.
The composite score formula is implemented as a `@property`:

| Metric | Weight |
|---|---|
| Correctness | 30% |
| Completeness | 20% |
| Edge case handling | 15% |
| Efficiency (tokens vs quality) | 15% |
| Communication quality | 10% |
| Peer judge avg (bonus) | 10% |

The scoring engine, model routing based on scores, and promotion/demotion logic
are Phase 3 work.

### 2.6 Structured A2A Schemas (all defined, used internally)

All defined in `core/schemas.py`:

| Schema | Purpose |
|---|---|
| `TaskAssignment` | PM/TL → subordinate: assign work |
| `TaskResult` | Worker/TL → supervisor: return deliverables |
| `StatusUpdate` | Any agent → supervisor: progress ping |
| `Question` | Agent → supervisor/human: blocking or non-blocking question |
| `Answer` | Response to a `Question` |
| `ReviewRequest` | → Judge: submit work for quality review |
| `ReviewResult` | Judge → requester: scored feedback |
| `Escalation` | Agent → hierarchy: critical issue bubbling up |
| `Artifact` | A single deliverable (code, doc, analysis) |
| `PMDelivery` | Final PM output: artifacts + metrics + assessment |
| `AgentPerformanceRecord` | RL sidecar input per agent per task |
| `MemoryEntry` | Single entry in any memory layer |

### 2.7 CORS Policy

Allowed origins:
- `http://localhost:3000` (local frontend dev)
- `http://localhost:3001`
- `http://frontend:3000` (Docker Compose internal)
- `https://*.vercel.app` (all Vercel preview deployments)
- `https://aifactory-ui.vercel.app` (production frontend)

---

## 3. Memory Bridge — Phase 2.9 (Built)

**File:** `src/aifactory/core/memory_bridge.py`

The `MemoryBridge` is the fail-safe adapter between `PipelineAgent` and the 3-layer `MemoryRouter`. Before Phase 2.9, the memory system existed but was completely disconnected from the production pipeline — agents started cold every run. Phase 2.9 wires them together.

### 3.1 Architecture

```
PipelineAgent.think(user_msg)
  ├── [1] memory_bridge.retrieve_context(agent_id, user_msg)
  │       └── MemoryRouter.retrieve(levels=[AGENT, TEAM], agent_id=, team_id=run_id)
  │           → formatted "## Relevant Memory\n..." block
  ├── [2] _build_system(shared_ctx, memory_context)
  │       └── system_prompt + ## Shared Context + ## Relevant Memory
  ├── [3] litellm.acompletion(...)  ← LLM sees memory context
  ├── [4] memory_bridge.store_output(agent_id, reply, turn_number)
  │       └── MemoryRouter.store(MemoryEntry(level=AGENT, agent_id=, team_id=run_id))
  └── [5] memory_bridge.maybe_promote(agent_id, summary)
          └── MemoryRouter.promote(AGENT → TEAM) if turn_count >= threshold

SHARE: updates (in researcher, execution_lead nodes):
  extract_share(raw) → updates dict
  bridge.store_share_updates(agent_name, updates)
  └── MemoryRouter.store(MemoryEntry(level=TEAM, entry_type="share_update"))
```

### 3.2 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Fail-safe at two levels | Both `MemoryBridge` methods AND `think()` wrap memory ops in try/except. A Qdrant outage must never kill the pipeline. |
| `run_id` as `team_id` | Isolates team-layer memories per run. Workers in run A don't see SHARE: updates from run B. |
| Only search AGENT + TEAM on retrieve | Global layer is too broad for per-call injection. Global is for curated long-term knowledge only. |
| Context capped at `memory_context_max_chars` | Prevents system prompt bloat. Default 800 chars ≈ 200 tokens. |
| Memory disabled = `MemoryBridge(router=None)` | All methods become noops. No branches inside `think()`. |
| Auto-create bridge per run | `build_dynamic_pipeline()` creates `MemoryRouter()` + `MemoryBridge(run_id=run_id)` automatically when `memory_pipeline_enabled=True`. Tests pass `MemoryBridge(router=None)` to skip. |

### 3.3 Settings (all prefixed `AIFACTORY_`)

| Setting | Default | Description |
|---------|---------|-------------|
| `MEMORY_PIPELINE_ENABLED` | `true` | Wire memory into pipeline agents |
| `MEMORY_CONTEXT_TOP_K` | `3` | Max entries retrieved per think() call |
| `MEMORY_CONTEXT_MAX_CHARS` | `800` | Max chars of memory context injected per think() |
| `MEMORY_PROMOTION_INTERVAL` | `10` | Promote agent → team layer every N turns |

### 3.4 CLI Commands

```bash
aifactory memory stats                          # Layer counts (agent/team/global)
aifactory memory stats --verbose                # Per-collection Qdrant details
aifactory memory search "FastAPI auth"          # Semantic search (all layers)
aifactory memory search "query" --layer team    # Filter by layer
aifactory memory search "query" --agent master --top-k 3
aifactory memory clear --layer agent --yes      # Clear agent layer (no prompt)
aifactory memory clear --agent Worker-task_1    # Clear specific agent
aifactory memory clear --layer all --yes        # Wipe all layers
```

### 3.5 Test Coverage

`tests/test_pipeline_memory.py` — 51 tests covering:
- `MemoryBridge` noop when `router=None` (8 cases)
- `retrieve_context`: empty store, populated store, layer label, cap, failure isolation, format edge cases (9 cases)
- `store_output`: agent layer, empty-content guard, entry type, failure isolation, accumulation (5 cases)
- `store_share_update/updates`: team layer, empty-value guard, content format, multiple updates, failure isolation (6 cases)
- `maybe_promote`: below threshold, at threshold, empty summary guard, failure isolation (4 cases)
- `PipelineAgent` integration: context injection, store called, failure isolation, no-memory no-regression, accumulation, budget-first ordering, promote called (7 cases)
- `build_dynamic_pipeline` injection: named agents get bridge, explicit-None bridge, settings toggle (3 cases)
- researcher SHARE: team persistence: with + without SHARE: (2 cases)
- CLI `cmd_memory`: stats, stats-failure, search-no-results, search-with-results, invalid-layer, clear-confirmation, clear-yes (7 cases)

---

## 4. ComplexityGate v2 — Phase 2.8 (Built)

**File:** `src/aifactory/core/complexity_gate.py`  
**Config:** `src/aifactory/config/gate_patterns.yaml`

### 4.1 4-Tier Routing

| Tier | Trigger | Cost | Latency |
|------|---------|------|---------|
| **bypass** | Pure arithmetic or static fact | 0 tokens | <1ms |
| **fast_llm** | Simple Q&A (strong QA signal, no tool signal) | ~300 tokens | ~1–3s |
| **full pipeline** | Tool use, multi-step, research, coding | ~3000–8000 tokens | ~15–45s |

The gate runs in both `aifactory new` and `aifactory run` before any LLM call. Bypass and fast_llm paths still write a run dump so they appear in `aifactory status`.

### 4.2 YAML Pattern Configuration

Patterns in `config/gate_patterns.yaml` have a `weight` (0.0–1.0) and a `label`.  
Scoring: `min(max_weight + 0.05 * (n_matches - 1), 1.0)`.  
`fast_llm_threshold` (default 0.7) controls when strong QA confidence routes to fast path.

### 4.3 Test Coverage

`tests/test_complexity_gate.py` — covers math bypass, NL math, static facts, fast_llm routing, full pipeline routing, YAML loading, fallback patterns, edge cases.

---

## 5. Frontend — Phase 1 (Built)

### 3.1 Stack

| Technology | Version | Purpose |
|---|---|---|
| Next.js | 16 | React framework, App Router |
| React | 19 | UI components |
| Tailwind CSS | 4 | Utility styling |
| next-themes | 0.4.6 | Dark/light mode |
| lucide-react | latest | Icons |

### 3.2 API Service Layer (`src/services/api.ts`)

All backend communication is centralised here. No component calls `fetch()` or `WebSocket()` directly.

| Export | Purpose |
|---|---|
| `checkHealth()` | `GET /api/health` — connectivity check |
| `getModels()` | `GET /api/models` — model registry for UI picker |
| `submitTask(req)` | `POST /api/tasks` — submit and get `task_id` |
| `getTask(id)` | `GET /api/tasks/{id}` — poll status + result |
| `connectTaskStream(id, onEvent)` | `WS /api/ws/{id}` — returns cleanup fn |

`connectTaskStream` includes a 20 s keep-alive interval (sends `ping`) and auto-reconnect is the caller's responsibility.

### 3.3 Views / Pages

All views are rendered inside `AppShell.tsx` (SPA — no page navigation).
Chat is the **default and primary view**. Other views are accessible via the top navbar.

#### Chat View (`ChatView.tsx`) — PRIMARY

- Conversational interface: user message bubbles + assistant response cards
- On submit: calls `submitTask()` → opens WebSocket via `connectTaskStream()`
- Live events rendered as a collapsible **Agent Stream Panel** inside the assistant card
- On `completed`: renders a **Result Card** (title, summary, quality, artifacts, metrics)
- **Model config panel** (collapsible): PM model dropdown (from `/api/models`) + worker tier select
- Keyboard shortcut: ⌘↵ to send
- Empty state with example prompts
- **Status:** Fully wired to real backend.

#### Dashboard View (`DashboardView.tsx`)
- Hero card: system status badge, active project count, memory health %
- Floating animated agent node decorations
- 3-column bento grid:
  - **Active Projects**: progress bars, status badges
  - **Memory Health**: SVG sparkline chart (7-day trend)
  - **Hibernated Teams**: idle team list with resume button
- **Recent Assessments**: full-width sortable table
- **Status:** All data hardcoded (mock). No backend calls.

#### Projects View (`ProjectsView.tsx`)
- Stats strip: Active / Reviewing / Completed / Issues counts
- Projects table: name, team members, progress bar, agent count, status badge
- **Status:** All data hardcoded (mock).

#### Agent Tree View (`AgentTree.tsx`)
- Hierarchical card tree: PM → TL(s) → Workers
- Per-node: role badge, model label, status dot, RL score, token count
- Collapsible subtrees
- **Status:** Demo tree hardcoded. No backend calls.

#### Memory View (`MemoryView.tsx`)
- 3-column layout: AGENT (cyan) · TEAM (purple) · GLOBAL (green)
- Memory entry cards: type badge, content, timestamp, agent/team source
- "PROMOTED" badge for promoted entries
- **Status:** Demo entries hardcoded.

#### Hibernated Teams View (`HibernatedView.tsx`)
- Team cards: name, idle duration, agent roles, memory health, token usage
- Resume / Delete action buttons (UI only, no backend)
- **Status:** Demo data hardcoded.

### 3.3 Layout Components

| Component | Description |
|---|---|
| `Navbar.tsx` | Top nav: logo, 6 view links (Chat first), search bar, theme toggle, notifications, avatar |
| `Header.tsx` | Metrics strip: token count, cost, elapsed time (simulated increments) |
| `AppShell.tsx` | Root layout: view router, Chat is default, chat bypasses `app-main` wrapper |

### 3.4 Task Input (`TaskInput.tsx`)

- 4-row textarea with "TASK INGESTION TERMINAL" label
- 4 quick-example prompt chips
- Model badges: PM: Claude 3.7 Sonnet · Workers: Gemini 2.5 Flash
- Keyboard shortcut: Cmd/Ctrl+Enter to submit
- Status badge: READY ↔ EXECUTING
- **Status:** `onSubmit()` fires but `AppShell` only sets `isRunning=true` for 8 s.
  No `POST /api/tasks` call yet.

### 3.5 Live Components (simulated, not wired to backend)

| Component | Simulation |
|---|---|
| `MessageStream.tsx` | Cycles through 8 hardcoded A2A messages every 4 s; pause/filter UI |
| `BudgetTracker.tsx` | Increments tokens/cost/iterations every 2.5 s |
| `Header.tsx` | Increments tokens/cost every 2 s |

### 3.6 Metrics Panel (`MetricsPanel.tsx`)

- RL composite scores per agent (hardcoded demo: Researcher-01, Coder-01)
- 6-metric breakdown with weight labels
- Bar chart visualization per metric
- **Status:** Hardcoded demo data.

### 3.7 Theme System

- **Provider:** `next-themes` with `attribute="class"`, `defaultTheme="dark"`, `enableSystem=false`
- **Toggle:** `ThemeToggle.tsx` — Sun/Moon icon in navbar; mounted-guard for SSR
- **CSS variables:** `html.dark` / `html.light` blocks in `globals.css`
- **Affected elements:** `html`, `body`, navbar, cards, inputs, all components via `var(--bg)` etc.
- Both `html` and `body` carry `background-color: var(--bg)` for full-viewport coverage

### 3.8 Backend Environment Variables

| Variable | Example | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `https://aifactory-4e5k.onrender.com` | Base URL for `fetch()` calls |
| `NEXT_PUBLIC_WS_URL` | `wss://aifactory-4e5k.onrender.com` | Base URL for WebSocket |

Set in `vercel.json` for production, `.env.local` for local dev.
Read by `src/services/api.ts` — all HTTP and WebSocket calls use these at runtime.

---

## 4. Deployment & Infrastructure (Built)

### 4.1 Backend — Render

| Setting | Value |
|---|---|
| Service | Web Service |
| Plan | Starter (always-on) |
| Runtime | Docker (Python 3.12-slim + uv) |
| Port | 8000 |
| Start command | `uvicorn aifactory.api.main:app --host 0.0.0.0 --port 8000 --workers 1` |
| Health check | `GET /api/health` |
| Live URL | `https://aifactory-4e5k.onrender.com` |

**Required env vars on Render:**

| Var | Notes |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (LLM calls + embeddings) |
| `DATABASE_URL` | Render managed PostgreSQL internal URL |
| `REDIS_URL` | Render managed Redis internal URL |
| `QDRANT_URL` | Qdrant Cloud cluster endpoint |
| `QDRANT_API_KEY` | Qdrant Cloud API key |
| `LANGCHAIN_API_KEY` | LangSmith API key (observability) |
| `LANGCHAIN_TRACING_V2` | `true` to enable tracing |
| `LANGCHAIN_PROJECT` | `aifactory` (LangSmith project name) |

### 4.2 Frontend — Vercel

| Setting | Value |
|---|---|
| Framework | Next.js |
| Build command | `npm run build` |
| Output | Default Next.js (not standalone — standalone is Docker-only via `DOCKER_BUILD=1`) |
| Live URL | Vercel-assigned (check dashboard) |

**Env vars in `vercel.json`:**

| Var | Value |
|---|---|
| `NEXT_PUBLIC_API_URL` | `https://aifactory-4e5k.onrender.com` |
| `NEXT_PUBLIC_WS_URL` | `wss://aifactory-4e5k.onrender.com` |

### 4.3 Local Docker Compose

Two compose files in `AiFactory/`:

| File | Purpose |
|---|---|
| `docker-compose.dbs.yml` | Local DBs only: Qdrant (6333), Redis (6379), Postgres (5432) |
| `docker-compose.yml` | Full stack: backend + frontend + all DBs |

**Usage:**
```bash
# DBs only (recommended for local dev)
docker compose -f docker-compose.dbs.yml up -d

# Full stack
cp .env.docker.example .env.docker   # fill in API keys
docker compose up --build
```

### 4.4 CI/CD — GitLab CI (`.gitlab-ci.yml`)

Three stages:

| Stage | Jobs | Passes when |
|---|---|---|
| lint | `ruff check`, `ruff format --check` | Zero lint + format violations |
| test | `pytest --cov` | All 216 tests pass |
| build-package | `uv build` | sdist + wheel build clean |

`UV_CACHE_DIR` is set to `$CI_PROJECT_DIR/.cache/uv` for GitLab caching.
`.cache/` is excluded from sdist via `pyproject.toml` hatchling config.

---

## 5. Data Contracts

### TaskRequest (HTTP request body)

```typescript
{
  task: string           // required, min 1 char
  context?: string       // default ""
  num_workers?: number   // default 3, range 1-3
  pm_model?: string      // optional: model ID from /api/models allowed list
  worker_tier?: string   // "premium" | "standard" | "budget", default "budget"
}
```

### TaskResponse (HTTP 202 body)

```typescript
{
  task_id: string       // UUID
  status: "pending"
  created_at: string    // ISO 8601 UTC
}
```

### TaskStatus (GET poll response)

```typescript
{
  task_id: string
  status: "pending" | "running" | "completed" | "failed"
  created_at: string
  updated_at: string
  progress: number      // 0.0 – 1.0
  agents: object[]      // reserved, always [] in Phase 1
  messages: object[]    // reserved, always [] in Phase 1
  result: PMDelivery | null
  error: string | null
  metrics: RunMetrics
}
```

### PMDelivery (inside result)

```typescript
{
  title: string
  summary: string
  artifacts: Artifact[]
  quality_assessment: string
  open_issues: string[]
  recommendations: string[]
  metrics: RunMetrics
}
```

### Artifact

```typescript
{
  name: string            // e.g. "main.py"
  content_type: string    // "python" | "markdown" | "json" | "text" | ...
  content: string         // full file content
  summary: string         // one-liner
}
```

### RunMetrics

```typescript
{
  total_iterations: number
  total_tokens: number
  total_teams: number
  total_agents: number
}
```

### WebSocket Events

```typescript
// Server → Client only

type WsEvent =
  | { type: "status";          status: "running"; progress: number }
  | { type: "agent_hierarchy"; message: string;   progress: number }
  | { type: "completed";       progress: 1.0;     result: PMDelivery }
  | { type: "error";           message: string }
```

---

## 6. Configuration Reference

All via environment variables (or `.env` file). Prefix: `AIFACTORY_` (except LLM keys).

### LLM Keys (no prefix)

| Var | Notes |
|---|---|
| `OPENROUTER_API_KEY` | Preferred — routes to any provider |
| `ANTHROPIC_API_KEY` | Direct Anthropic access (optional) |
| `OPENAI_API_KEY` | Direct OpenAI access (optional) |

### Model Routing

| Var | Default | Used by |
|---|---|---|
| `AIFACTORY_PREMIUM_MODEL` | `openrouter/anthropic/claude-sonnet-4-5` | PM, Assessors |
| `AIFACTORY_STANDARD_MODEL` | `openrouter/google/gemini-2.0-flash-001` | Team Leads |
| `AIFACTORY_BUDGET_MODEL` | `openrouter/google/gemini-2.0-flash-001` | Workers |
| `AIFACTORY_RL_JUDGE_MODEL` | `openrouter/google/gemini-2.0-flash-001` | RL Judge (Phase 3) |

### Memory

| Var | Default |
|---|---|
| `QDRANT_URL` | `http://localhost:6333` |
| `QDRANT_API_KEY` | _(empty)_ |
| `AIFACTORY_MEMORY_PROMOTION_INTERVAL` | `10` |
| `AIFACTORY_MEMORY_MAX_AGENT_ENTRIES` | `200` |
| `AIFACTORY_MEMORY_RETRIEVAL_TOP_K` | `5` |

### LlamaIndex Memory Engine

| Var | Default | Notes |
|---|---|---|
| `AIFACTORY_LLAMAINDEX_EMBEDDING_PROVIDER` | `huggingface` | `huggingface` (local, free) or `openai` (API, paid) |
| `AIFACTORY_LLAMAINDEX_HF_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model (384-dim) |
| `AIFACTORY_LLAMAINDEX_HF_DIMENSIONS` | `384` | Embedding vector dimensions |
| `AIFACTORY_LLAMAINDEX_RERANK_ENABLED` | `false` | Enable BGE reranker for retrieval quality |
| `AIFACTORY_LLAMAINDEX_RERANK_MODEL` | `BAAI/bge-reranker-base` | Reranker model name |
| `AIFACTORY_LLAMAINDEX_RERANK_TOP_N` | `3` | Final top-N after reranking |

### Hierarchy & Budgets

| Var | Default |
|---|---|
| `AIFACTORY_MAX_SUBORDINATES_PER_LEAD` | `3` |
| `AIFACTORY_MAX_PROJECT_MANAGERS` | `5` |
| `AIFACTORY_MAX_BUDGET_USD` | `5.00` |
| `AIFACTORY_MAX_ITERATIONS` | `50` |
| `AIFACTORY_MAX_TOTAL_TOKENS` | `500000` |

### LangSmith Observability

| Var | Default | Notes |
|---|---|---|
| `LANGCHAIN_TRACING_V2` | `false` | Set `true` to enable tracing |
| `LANGCHAIN_API_KEY` | _(empty)_ | LangSmith API key from smith.langchain.com |
| `LANGCHAIN_PROJECT` | `aifactory` | LangSmith project name |
| `LANGCHAIN_ENDPOINT` | `https://api.smith.langchain.com` | LangSmith API endpoint |

### Rate Limiting

| Var | Default | Notes |
|---|---|---|
| `AIFACTORY_RATE_LIMIT_ENABLED` | `true` | Enable/disable all rate limiting |
| `AIFACTORY_RATE_LIMIT_TASKS` | `10/minute` | POST /api/tasks — pipeline submissions |
| `AIFACTORY_RATE_LIMIT_READ` | `60/minute` | GET endpoints (health, models, tasks, observability) |
| `AIFACTORY_RATE_LIMIT_MEMORY` | `30/minute` | Memory endpoints (store, retrieve, promote, clear) |
| `AIFACTORY_RATE_LIMIT_WS` | `5/minute` | WebSocket connections |

When a client exceeds their limit, the API returns **HTTP 429 Too Many Requests** with:
- JSON body: `{"error": "Rate limit exceeded", "detail": "...", "retry_after": 60}`
- Header: `Retry-After: 60`

Rate limits are per client IP address. In-memory storage by default (single-worker);
can be switched to Redis URI for multi-worker deployments.

---

## 7. Observability (Built — Phase 2.0)

### 7.1 LangSmith Integration

Every `BaseAgent.think()` call is decorated with `@traceable`, which automatically logs to LangSmith when `LANGCHAIN_API_KEY` is set. When the key is absent, the decorator is a zero-cost passthrough.

**Traced operations:**
- Every LLM call (model, tokens in/out, cost, latency, prompt, response)
- LangGraph node transitions (via LangChain callbacks)
- Memory store/retrieve operations

**Module:** `core/observability.py`
- `setup_langsmith()` — call once at startup; reads settings, sets env vars
- `@traceable(name, run_type)` — decorator for any function
- `get_recent_runs(limit)` — fetch recent root runs from LangSmith
- `get_run_detail(run_id)` — fetch single run with trace tree

### 7.2 Observability API Endpoints

#### `GET /api/observability/status`

| Field | Value |
|---|---|
| Auth | None |
| Response | `200 OK` |
| Body | `{enabled: bool, project: str|null, endpoint: str|null}` |
| Used by | UI Traces view to check if LangSmith is configured |

#### `POST /api/memory/store`

| Field | Value |
|---|---|
| Auth | None |
| Response | `200 OK` |
| Request body | `{level, content, agent_id?, team_id?, entry_type?, turn_number?, metadata?}` |
| Response body | `{status: "stored", point_id: str, level: str}` |
| Behaviour | Upserts to Qdrant with content-hash deduplication. Same content+agent+turn = no duplicate. |
| Used by | Direct memory injection, testing, agent pipeline |

#### `POST /api/memory/retrieve`

| Field | Value |
|---|---|
| Auth | None |
| Response | `200 OK` |
| Request body | `{query, levels?, agent_id?, team_id?, top_k?, entry_type?}` |
| Response body | `{status: "ok", count: int, results: MemoryEntry[]}` |
| Behaviour | Semantic similarity search across specified layers with metadata filtering |
| Used by | Memory exploration, debugging, agent context injection |

#### `POST /api/memory/promote`

| Field | Value |
|---|---|
| Auth | None |
| Response | `200 OK` |
| Request body | `{agent_id, team_id, summary, from_level?, to_level?}` |
| Response body | `{status: "promoted", entry_id: str, from: str, to: str}` |
| Behaviour | Stores summary at target layer, marks source entries as promoted |
| Used by | Memory curation, agent auto-promotion pipeline |

#### `DELETE /api/memory/agent/{agent_id}`

| Field | Value |
|---|---|
| Auth | None |
| Response | `200 OK` |
| Response body | `{status: "cleared", agent_id: str}` |
| Behaviour | Deletes all memory entries for the agent across all layers |
| Used by | Agent cleanup, testing |

#### `GET /api/observability/runs`

| Field | Value |
|---|---|
| Auth | None |
| Query params | `limit` (1-100, default 20) |
| Response | `200 OK` |
| Body | `{runs: RunSummary[], enabled: bool, total: int}` |
| Graceful fallback | Returns `{runs: [], enabled: false}` when no API key |

`RunSummary`: `{run_id, name, run_type, status, start_time, end_time, latency_ms, total_tokens, prompt_tokens, completion_tokens, total_cost, error, tags}`

#### `GET /api/observability/runs/{run_id}`

| Field | Value |
|---|---|
| Auth | None |
| Response | `200 OK` or `503` (no key) or `404` (not found) |
| Body | `RunDetail` with nested `children[]` trace tree |

#### `GET /api/tasks/{task_id}/traces`

| Field | Value |
|---|---|
| Auth | None |
| Query params | `limit` (1-100, default 20) |
| Response | `200 OK` or `404` (task not found) |
| Body | `{task_id, traces: TraceSummary[], enabled: bool, total: int}` |
| Behaviour | Searches LangSmith for runs tagged with the `task_id`. Returns empty `traces` when LangSmith is not configured. |
| Graceful fallback | Returns `{traces: [], enabled: false}` when no API key |

`TraceSummary`: `{run_id, trace_id, name, run_type, status, start_time, end_time, latency_ms, total_tokens, prompt_tokens, completion_tokens, total_cost, error, tags}`

### 7.3 Traced Operations

All traced operations appear in LangSmith when `LANGCHAIN_API_KEY` is set:

| Operation | Decorator | `run_type` |
|---|---|---|
| `BaseAgent.think()` | `@traceable(name="agent_think")` | `llm` |
| `MemoryRouter.store()` | `@traceable(name="memory_store")` | `tool` |
| `MemoryRouter.retrieve()` | `@traceable(name="memory_retrieve")` | `retriever` |
| `MemoryRouter.promote()` | `@traceable(name="memory_promote")` | `tool` |

Pipeline runs are tagged with the `task_id` in LangSmith metadata, enabling
the `/api/tasks/{task_id}/traces` endpoint to link traces to specific tasks.

### 7.4 Frontend Traces View

`ObservabilityView.tsx` — accessible via "Traces" nav item:
- Shows aggregate stats (total runs, tokens, cost, avg latency)
- Runs table with expandable trace tree per run
- Graceful "LangSmith Not Configured" state with setup instructions
- Link to open full LangSmith dashboard

---

## 8. Rate Limiting (Built — Phase 3.5)

### 8.1 Architecture

All API endpoints are rate-limited per client IP using **slowapi** (built on the `limits` library).
Rate limits are enforced in-process (no external dependency). For multi-worker deployments,
the storage backend can be switched to Redis via `storage_uri` in `rate_limiter.py`.

**Module:** `core/rate_limiter.py`
- `limiter` — singleton `slowapi.Limiter` instance, attached to the FastAPI app
- `rate_limit_exceeded_handler()` — custom 429 handler returning JSON
- `get_rate_limit_info()` — returns current config (surfaced in `/api/health`)

### 8.2 Rate Limit Tiers

| Tier | Limit | Endpoints |
|---|---|---|
| **Tasks** | 10/minute | `POST /api/tasks` |
| **Read** | 60/minute | `GET /api/health`, `/api/models`, `/api/tasks`, `/api/tasks/{id}`, `/api/tasks/{id}/agents`, `/api/tasks/{id}/traces`, `/api/observability/*` |
| **Memory** | 30/minute | `POST /api/memory/store`, `POST /api/memory/retrieve`, `POST /api/memory/promote`, `DELETE /api/memory/agent/{id}`, `GET /api/memory/stats` |
| **WebSocket** | 5/minute | `WS /api/ws/{id}` |

### 8.3 Response Format (HTTP 429)

```json
{
  "error": "Rate limit exceeded",
  "detail": "Rate limit exceeded: 10 per 1 minute",
  "retry_after": 60
}
```

Headers: `Retry-After: 60`

### 8.4 Health Endpoint Integration

`GET /api/health` now includes a `rate_limiting` section:

```json
{
  "status": "ok",
  "version": "0.3.0",
  "rate_limiting": {
    "enabled": true,
    "limits": {
      "tasks_submit": "10/minute",
      "read": "60/minute",
      "memory": "30/minute",
      "websocket": "5/minute"
    }
  }
}
```

### 8.5 Configuration

All configurable via `AIFACTORY_RATE_LIMIT_*` environment variables (see §6).
Rate limiting can be disabled entirely by setting `AIFACTORY_RATE_LIMIT_ENABLED=false`.

### 8.6 Tests

17 tests in `tests/test_rate_limiter.py`:
- Module configuration (default values, singleton, info endpoint)
- Health endpoint includes rate limit info
- 429 enforcement (JSON body, Retry-After header, separate endpoint limits)
- Enable/disable toggle via env var
- Memory endpoint rate limits

---

## 9. Roadmap — Future Phases

### Phase 3.5 — Token Efficiency (✅ BUILT — 2026-04-11)

Three layered optimisations in `graphs/dynamic_pipeline.py`:

| Optimisation | Mechanism | Savings | Setting |
|---|---|---|---|
| **Anthropic prompt caching** | `cache_control: ephemeral` on system messages for `claude-*` and `openrouter/anthropic/claude-*` | 60–90% on prompt tokens for repeat calls | `AIFACTORY_PROMPT_CACHE_ENABLED` / `AIFACTORY_PROMPT_CACHE_OPENROUTER` |
| **LiteLLM response cache** | `litellm.cache = Cache(ttl=3600)` at pipeline start; all providers/tiers | 100% on identical (model, messages) pairs | `AIFACTORY_RESPONSE_CACHE_ENABLED` / `AIFACTORY_RESPONSE_CACHE_TTL` |
| **Smart judge skip** | Synthetic assessment for single-task runs with no research | ~800–1200 tokens per simple query | `AIFACTORY_JUDGE_SKIP_SIMPLE_RUNS` |
| **Shared context pruning** | `_prune_shared_context(ctx, max_keys=25)` after analyse/researcher/execution_lead | Prevents prompt bloat in multi-turn runs | `AIFACTORY_SHARED_CONTEXT_MAX_KEYS` |

OpenAI models also benefit from **automatic prompt caching** (native, no code needed) — OpenAI caches the first 1024+ token prefix of prompts for GPT-4o/mini automatically.

**52 new tests** in `tests/test_token_efficiency.py` covering all four optimisations.

### Phase 2.9 — Memory Bridge (✅ BUILT — 2026-04-11)

See [Section 3](#3-memory-bridge--phase-29-built) for full spec.

### Phase 2.8 — ComplexityGate v2 (✅ BUILT — 2026-04-10)

See [Section 4](#4-complexitygate-v2--phase-28-built) for full spec.

### Phase 3 — RL-lite Scoring (Not Yet Built)

After each run, store `(model, task_type, quality_score)` tuples from judge metrics to Global memory. `model_for_agent()` reads last-N performance records to adjust ability weights, enabling adaptive routing without a full RL sidecar.

### Phase 4 — Hardening (Not Yet Built)

Circuit breakers, budget caps, Redis task store (replace MemorySaver with Redis-backed checkpointer).

---

## 10. Legacy Roadmap — Phases 2.1–4

### Phase 3.5 — LlamaIndex Memory Engine (✅ BUILT)

**What was done:** Replaced raw OpenAI embeddings with LlamaIndex-powered memory query engine.

| Component | Detail |
|---|---|
| Embedding provider | HuggingFace `BAAI/bge-small-en-v1.5` (384-dim, local, free) via LlamaIndex |
| Optional fallback | OpenAI `text-embedding-3-small` (1536-dim) via `llamaindex_embedding_provider=openai` |
| Query engine | LlamaIndex `VectorStoreIndex.as_query_engine()` with `MetadataFilters` |
| Reranking | Optional `FlagEmbeddingReranker` (`BAAI/bge-reranker-base`) |
| API endpoints | `POST /api/memory/store`, `POST /api/memory/retrieve`, `POST /api/memory/promote`, `DELETE /api/memory/agent/{id}` |
| Shared singleton | `_get_memory_router()` in API — data persists across HTTP requests |
| Qdrant fallback | Auto in-memory mode when remote server unreachable |
| Tests | 403 passing (39 memory + 64 sandbox + 17 rate-limiter + 283 others) |
| Dependencies | 6 LlamaIndex packages in `pyproject.toml` |

### Phase 3.5 — Frontend Docs + Wiring (✅ BUILT)

| Component | Detail |
|---|---|
| Docs site | 20-page documentation site at `/docs` route (e2b-style, black+purple theme) |
| MessageStream | Wired to real API (removed DEMO_MESSAGES) |
| MetricsPanel | Wired to real API (removed DEMO_AGENTS) |
| SiteFooter | Updated to v0.3.0, removed dead links |
| Navbar | Added Docs link |

### Phase 3.5 — Code Execution Sandbox + Quality Gates (✅ BUILT)

| Component | Detail |
|---|---|
| `core/sandbox.py` | Dual-backend: local subprocess + E2B cloud micro-VM (auto-detected via `E2B_API_KEY`) |
| E2B SDK | `e2b-code-interpreter>=2.6.0` — `Sandbox.create(api_key=..., timeout=...)`, supports Python/JS/Bash |
| Backend selection | `_get_backend()`: E2B if `E2B_API_KEY` set (env var → settings/.env fallback), else local subprocess |
| Sandbox lifecycle | Create → execute → kill per code block (fresh isolated micro-VM, no shared state) |
| Code extraction | `extract_code_blocks()`: regex for fenced markdown + `CODE:lang\ncode\nCODE_END` protocol |
| Pipeline wiring | `dynamic_pipeline.py` lines 618-636: `execute_code_blocks(raw, timeout=30)` after each worker |
| Worker prompts | CODING EXCELLENCE (optimal DSA, Big-O in comments), ISOLATED SANDBOX (self-contained code) |
| Analyser rule | CODING RULE: write+test+run+show = 1 task (prevents sandbox isolation errors) |
| Execution lead | TASK MERGING: collapses dependent coding tasks the analyser over-splits |
| Master rule | Always includes code in final response when user explicitly asked for it |
| Worker self-correction | `_validate_and_fix()` in `worker.py` — sandbox validation + LLM retry (2x max) |
| PM quality review | `_assess_quality()` in `project_manager.py` — real LLM review gate (replaced pass-through) |
| Tests | 64 sandbox tests (real execution, zero mocks) + `@skip_no_e2b` for CI without API key |
| Live tests | `scripts/test_e2b_live.py` — 10 integration tests for both backends |

### Phase 3.5 — API Rate Limiting (✅ BUILT)

| Component | Detail |
|---|---|
| Library | `slowapi>=0.1.9` (built on `limits`) |
| Module | `core/rate_limiter.py` — Limiter singleton, 4 tiers, custom 429 handler |
| Tiers | Tasks: 10/min, Read: 60/min, Memory: 30/min, WS: 5/min |
| Per-IP | Limits tracked per client IP address |
| Health integration | `GET /api/health` now includes `rate_limiting` section |
| 429 format | JSON `{error, detail, retry_after}` + `Retry-After` header |
| Configurable | `AIFACTORY_RATE_LIMIT_*` env vars, disable with `ENABLED=false` |
| Tests | 17 rate-limiter tests passing |

### Phase 3.6 — Assessment Team

**Trigger:** task complexity exceeds single-team threshold
**New agents:**
- `MasterResearcher` — deep-dives the problem domain before planning
- `Assessor` (×2) — evaluate feasibility, scope, risk
- `DevilsAdvocate` — challenges assumptions
- `HierarchyBuilder` — constructs the full PM/TL/Worker tree dynamically

**New APIs needed:**
- `GET /api/tasks/{id}/agents` — live hierarchy tree
- WebSocket event: `{type: "agent_spawned", agent: AgentConfig}`
- WebSocket event: `{type: "agent_message", message: A2AMessage}`

### Phase 4 — RL Sidecar + Model Router

**What it does:** scores every agent after each task, promotes/demotes models
**Schema already defined:** `AgentPerformanceRecord` with 6 metrics + composite
**New components:**
- `RLSidecar` — calls judge agents, writes `AgentPerformanceRecord` to Qdrant
- `ModelRouter` — selects model tier based on agent's historical composite score
- Agents-as-Judge endpoint: `POST /api/judge`

**New APIs needed:**
- `GET /api/agents/{agent_id}/performance` — historical RL scores
- `GET /api/agents/{agent_id}/performance/history` — time-series

### Phase 5 — Hardening

- Circuit breakers on LLM calls (fail-fast after N consecutive errors)
- Redis-backed rate limiter storage (for multi-worker deployments)
- Budget enforcement middleware (hard-stop at `MAX_BUDGET_USD`)
- Observability: structured JSON logs, OpenTelemetry traces

### Frontend Integration

**Done (Phase 1.5 + 3.5):**
- `ChatView.tsx` → real `POST /api/tasks` + WebSocket streaming (fully wired)
- `src/services/api.ts` → typed API layer for all HTTP + WebSocket calls
- Model picker in ChatView → real `GET /api/models` response
- `MessageStream.tsx` → wired to real API (removed DEMO_MESSAGES)
- `MetricsPanel.tsx` → wired to real API (removed DEMO_AGENTS)
- 20-page documentation site at `/docs` route

**Remaining:**

| Component | Change needed |
|---|---|
| `AgentTree.tsx` | Replace demo tree with live `ws event: agent_hierarchy` data |
| `BudgetTracker.tsx` | Replace `setInterval` mock with real `ws event: metrics_update` |
| `Header.tsx` | Replace `setInterval` mock with real token/cost from task metrics |
| `DashboardView.tsx` | Replace hardcoded project list with `GET /api/tasks` (paginated) |
| `MemoryView.tsx` | Wire to new `POST /api/memory/retrieve` endpoint |
