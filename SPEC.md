# AiFactory — Feature Specification

**Version:** 0.1.0
**Phase:** 1 — Single Team MVP
**Last updated:** 2026-03-30

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
- All inter-agent communication is typed Pydantic (no free-form text)
- 3-layer vector memory: Agent → Team → Global (Qdrant)
- Unified LLM interface via LiteLLM (any provider)
- RL sidecar scoring schema ready; execution engine in Phase 3

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

#### `POST /api/tasks`

| Field | Value |
|---|---|
| Auth | None |
| Response | `202 Accepted` |
| Request body | `{task: string, context?: string, num_workers?: 1-3}` |
| Response body | `{task_id: uuid, status: "pending", created_at: ISO8601}` |
| Behaviour | Fire-and-forget: returns immediately, pipeline runs in background via `asyncio.create_task()` |
| Storage | In-memory Python dict (resets on redeploy — Redis planned for Phase 4) |

**Validation rules:**
- `task` is required, must be non-empty string
- `num_workers` defaults to 3, maximum 3 (capped to available specialties)
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
run_single_team(task, context, num_workers, worker_specialties)
  → ProjectManager.run(task)
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
| LLM interface | LiteLLM `acompletion()` — any provider |
| Token budget | Hard cap per agent; raises `AgentBudgetError` when exceeded |
| Iteration cap | Configurable `max_iterations` per agent |
| Memory | `MemoryRouter` injected; store/retrieve per turn |
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

**Embedding:** SHA-256 hash-based pseudo-embedding in dev (no model needed).
Pass a real `embedding_fn` for production semantic search.

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

### 3.2 Views / Pages

All views are rendered inside `AppShell.tsx` (SPA — no page navigation).

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
| `Navbar.tsx` | Top nav: logo, 5 view links, search bar, theme toggle, notifications, avatar |
| `Header.tsx` | Metrics strip: token count, cost, elapsed time (simulated increments) |
| `AppShell.tsx` | Root layout: view router, task submit handler (mock 8s timer) |

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

### 3.8 Backend Environment Variables (defined, not yet used in code)

| Variable | Example | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `https://aifactory-4e5k.onrender.com` | Base URL for `fetch()` calls |
| `NEXT_PUBLIC_WS_URL` | `wss://aifactory-4e5k.onrender.com` | Base URL for WebSocket |

These are set in `vercel.json` for production and `.env.local` for local dev.
No component currently reads them — wiring is the next integration task.

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
| `OPENROUTER_API_KEY` | Set manually in dashboard |
| `QDRANT_URL` | Qdrant Cloud cluster endpoint |
| `QDRANT_API_KEY` | Qdrant Cloud API key |

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
  task: string          // required, min 1 char
  context?: string      // default ""
  num_workers?: number  // default 3, range 1-3
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

---

## 7. Roadmap — Phases 2–4 (Not Yet Built)

### Phase 2 — Assessment Team + Dynamic Hierarchy

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

### Frontend Integration (next immediate task)

Wire `TaskInput` → real backend. Components to update:

| Component | Change needed |
|---|---|
| `AppShell.tsx` | Replace 8 s mock with `fetch(POST /api/tasks)` + store `task_id` |
| `AppShell.tsx` | Open `WebSocket(/api/ws/{task_id})`, dispatch events to state |
| `AgentTree.tsx` | Replace demo tree with `ws event: agent_hierarchy` data |
| `MessageStream.tsx` | Replace `setInterval` mock with real `ws event: agent_message` |
| `BudgetTracker.tsx` | Replace `setInterval` mock with real `ws event: metrics_update` |
| `Header.tsx` | Replace `setInterval` mock with real token/cost from task metrics |
