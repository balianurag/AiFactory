# AiFactory — Feature Specification

**Version:** 0.3.0
**Phase:** 3 — Persistent Storage + Real Embeddings + Observability
**Last updated:** 2026-04-05

This document is the authoritative spec for what has been built. Use it as the
baseline for spec-driven development of future phases. Each new feature should
be specced here before implementation begins.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Backend — Phase 1 (Built)](#2-backend--phase-1-built)
3. [Frontend — Phase 1 (Built)](#3-frontend--phase-1-built)
4. [Deployment & Infrastructure (Built)](#4-deployment--infrastructure-built)
5. [Data Contracts](#5-data-contracts)
6. [Configuration Reference](#6-configuration-reference)
7. [Roadmap — Phases 2–4 (Not Yet Built)](#7-roadmap--phases-24-not-yet-built)

---

## 1. System Overview

AiFactory is a self-assembling, self-optimising multi-agent orchestration system
that accepts a raw human task description and returns production-grade output by
dynamically constructing a PM → Team Lead → Worker agent hierarchy.

```
Human Task
  → POST /api/tasks
      → Project Manager (premium LLM)
          → Team Lead (standard LLM)
              → Worker(s): CODER / RESEARCHER / WRITER  (budget LLM)
          → Consolidation
      → PMDelivery (artifacts + metrics)
  ← result via GET /api/tasks/{id} or WS /api/ws/{id}
```

**Key properties:**
- Hierarchy is constructed at runtime (not pre-defined)
- Complexity-based routing: trivial tasks skip PM/TL overhead
- All inter-agent communication is typed Pydantic (no free-form text)
- 3-layer vector memory: Agent → Team → Global (Qdrant with real OpenAI embeddings)
- LLM calls via OpenAI SDK (`openai.AsyncOpenAI`)
- PostgreSQL persistence for tasks/agents/messages; Redis hot cache
- LangSmith observability: every LLM call + memory op traced
- RL sidecar scoring schema ready; execution engine in Phase 4

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
| Request body | `{task, context?, num_workers?, pm_model?, worker_tier?}` |
| Response body | `{task_id: uuid, status: "pending", created_at: ISO8601}` |
| Behaviour | Fire-and-forget: returns immediately, pipeline runs in background via `asyncio.create_task()` |
| Storage | PostgreSQL (tasks, agents, messages) with Redis hot cache |

**Validation rules:**
- `task` is required, must be non-empty string
- `num_workers` defaults to 3, maximum 3 (capped to available specialties)
- `pm_model` must be a model ID present in `config/models.yaml` if provided; HTTP 422 otherwise
- `worker_tier` must be one of `"premium"`, `"standard"`, `"budget"`; defaults to `"budget"`
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

### 2.2 Pipeline — `single_team.py`

Entry point: `aifactory.graphs.single_team.run_single_team()`

```
run_single_team(task, context, num_workers, worker_specialties, pm_model, worker_model)
  → ProjectManager.run(task)   # uses pm_model (premium tier default)
      → LLM call → PMPlan (1-3 team objectives)
      → TeamLead.run(objective)
          → LLM call → WorkerTaskSpec list
          → Worker.run(spec) × N  (sequential)
              → LLM call → TaskResult + Artifact list
          → LLM call → consolidate worker results → TaskResult
      → LLM call → consolidate team results → PMDelivery
  → returns PMDelivery
```

**Worker specialties available:** `coder`, `researcher`, `writer`, `reviewer`,
`tester`, `architect`, `data_analyst`, `devops`
**Phase 1 uses:** first N of `[coder, researcher, writer]`

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

Three-layer Qdrant vector store:

| Layer | Scope | Promoted from |
|---|---|---|
| `agent` | Private per-agent episodic memory | — |
| `team` | Shared per-team working memory | agent layer |
| `global` | Cross-project long-term memory | team layer |

**Promotion:** every `AIFACTORY_MEMORY_PROMOTION_INTERVAL` turns (default: 10),
agent-layer summaries are written to the team layer.

**Embedding:** OpenAI `text-embedding-3-small` by default (real semantic search).
Falls back to SHA-256 hash stub only when `OPENAI_API_KEY` is absent.

**Tracing:** `store()`, `retrieve()`, and `promote()` are all decorated with `@traceable`
and appear in LangSmith traces when tracing is enabled.

**Qdrant mode:**
- Dev: in-memory (no Docker needed)
- Production: Qdrant Cloud via `QDRANT_URL` + `QDRANT_API_KEY`

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

## 3. Frontend — Phase 1 (Built)

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
| test | `pytest --cov` | All 187 tests pass |
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

## 8. Roadmap — Phases 2.1–4 (Not Yet Built)

### Phase 2.1 — LlamaIndex Memory Migration

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

### Phase 3 — RL Sidecar + Model Router

**What it does:** scores every agent after each task, promotes/demotes models
**Schema already defined:** `AgentPerformanceRecord` with 6 metrics + composite
**New components:**
- `RLSidecar` — calls judge agents, writes `AgentPerformanceRecord` to Qdrant
- `ModelRouter` — selects model tier based on agent's historical composite score
- Agents-as-Judge endpoint: `POST /api/judge`

**New APIs needed:**
- `GET /api/agents/{agent_id}/performance` — historical RL scores
- `GET /api/agents/{agent_id}/performance/history` — time-series

### Phase 4 — Hardening

- Circuit breakers on LLM calls (fail-fast after N consecutive errors)
- Redis-backed task store (replace in-memory dict — survives redeploys)
- Budget enforcement middleware (hard-stop at `MAX_BUDGET_USD`)
- Observability: structured JSON logs, OpenTelemetry traces
- `GET /api/tasks` — list all tasks with pagination

### Frontend Integration

**Done (Phase 1.5):**
- `ChatView.tsx` → real `POST /api/tasks` + WebSocket streaming (fully wired)
- `src/services/api.ts` → typed API layer for all HTTP + WebSocket calls
- Model picker in ChatView → real `GET /api/models` response

**Remaining (Phase 2 integration):**

| Component | Change needed |
|---|---|
| `AgentTree.tsx` | Replace demo tree with live `ws event: agent_hierarchy` data |
| `MessageStream.tsx` | Replace `setInterval` mock with real `ws event: agent_message` events |
| `BudgetTracker.tsx` | Replace `setInterval` mock with real `ws event: metrics_update` |
| `Header.tsx` | Replace `setInterval` mock with real token/cost from task metrics |
| `DashboardView.tsx` | Replace hardcoded project list with `GET /api/tasks` (paginated, Phase 4) |
