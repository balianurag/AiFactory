# AiFactory — Sprint Plan

**Created:** 2 April 2026
**Methodology:** Spec-driven, story-based sprints
**Sprint Duration:** ~3-5 days each

---

## Sprint 1: LangSmith Observability Integration

**Goal:** Every LLM call and graph transition is traced and visible in LangSmith dashboard. Frontend has an Observability view.

**Duration:** 1-2 days

### Story 1.1: LangSmith Backend Wiring
**As a** developer
**I want** all LLM calls in BaseAgent.think() to be automatically traced to LangSmith
**So that** I can see the full execution tree, token costs, and latencies for every task run

**Acceptance Criteria:**
- [ ] `langsmith` package added to `pyproject.toml`
- [ ] `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` env vars added to `.env` and `settings.py`
- [ ] `core/observability.py` created with LangSmith setup, `@traceable` helpers, and cost aggregation
- [ ] `BaseAgent.think()` decorated with `@traceable` — every LLM call appears in LangSmith
- [ ] LangGraph pipeline runs appear as nested trace trees
- [ ] All 187+ existing tests still pass (LangSmith tracing is no-op when key is absent)
- [ ] Manual test: run a task, see full trace tree in LangSmith dashboard

### Story 1.2: Observability API Endpoints
**As a** frontend developer
**I want** REST endpoints to fetch LangSmith trace data
**So that** the UI can display run history, costs, and trace details

**Acceptance Criteria:**
- [ ] `GET /api/observability/runs` — list recent runs with cost, duration, status
- [ ] `GET /api/observability/runs/{run_id}` — single run detail with full trace tree
- [ ] `GET /api/tasks/{task_id}/traces` — traces linked to a specific task
- [ ] Graceful fallback when LangSmith key is not configured (returns empty data, not 500)
- [ ] API tests added

### Story 1.3: Frontend Observability View
**As a** user
**I want** a "Traces" view in the UI showing LangSmith run data
**So that** I can monitor agent performance, costs, and debug issues

**Acceptance Criteria:**
- [ ] New `ObservabilityView.tsx` component added
- [ ] Navbar updated with "Traces" link (between Memory and Hibernated)
- [ ] View shows: run list (time, task, cost, tokens, status), click to expand trace tree
- [ ] LangSmith dashboard iframe embed option for detailed exploration
- [ ] AppShell routes to ObservabilityView on `"traces"` activeView
- [ ] Gracefully shows "LangSmith not configured" when backend returns no data

### Story 1.4: Environment & Config Update
**As a** developer
**I want** `.env`, `settings.py`, `CLAUDE.md`, and `SPEC.md` updated
**So that** all docs reflect the new LangSmith integration

**Acceptance Criteria:**
- [ ] `.env` has LangSmith section with placeholder vars
- [ ] `settings.py` has `langchain_api_key`, `langchain_project`, `langchain_tracing` fields
- [ ] `CLAUDE.md` documents the LangSmith integration
- [ ] `SPEC.md` documents the new `/api/observability/*` endpoints

---

## Sprint 2: LlamaIndex Memory Migration

**Goal:** Replace the hash-based MemoryRouter with LlamaIndex-powered semantic memory backed by Qdrant.

**Duration:** 2-3 days

### Story 2.1: LlamaIndex Memory Router
**As a** developer
**I want** MemoryRouter to use LlamaIndex VectorStoreIndex + real embeddings
**So that** agents can do true semantic retrieval instead of hash-based matching

**Acceptance Criteria:**
- [ ] `llama-index-core`, `llama-index-vector-stores-qdrant`, `llama-index-embeddings-openai` added to `pyproject.toml`
- [ ] `MemoryRouter` rewritten: same public API (`store()`, `retrieve()`, `promote()`), LlamaIndex under the hood
- [ ] Real embeddings via `text-embedding-3-small` (configurable)
- [ ] Fallback to hash embeddings when no embedding API key available (dev mode)
- [ ] Three `VectorStoreIndex` instances: agent, team, global
- [ ] `QueryEngine` per layer for semantic retrieval
- [ ] All existing memory tests pass
- [ ] New tests for semantic retrieval accuracy

### Story 2.2: Cross-Layer Query Pipeline
**As an** agent
**I want** to query all three memory layers in a single call
**So that** I get the best context from private memory + team knowledge + global patterns

**Acceptance Criteria:**
- [ ] `retrieve()` queries agent → team → global and merges results
- [ ] Deduplication by semantic similarity (threshold configurable)
- [ ] Results ranked by relevance, recency, and layer priority
- [ ] `top_k` configurable per layer

### Story 2.3: Memory Promotion with Summarisation
**As a** system
**I want** agent memory to be summarised before promotion to team layer
**So that** only distilled knowledge propagates up

**Acceptance Criteria:**
- [ ] Promotion uses LLM-based summarisation (via LlamaIndex `SummaryIndex`)
- [ ] Summarisation is traceable in LangSmith
- [ ] Promotion interval configurable via `AIFACTORY_MEMORY_PROMOTION_INTERVAL`
- [ ] Only entries above RL score threshold promoted to global (when RL is active)

---

## Sprint 3: Assessment Team Agents

**Goal:** Build the four new agent types that form the Assessment Team.

**Duration:** 2-3 days

### Story 3.1: MasterResearcher Agent
**As a** system
**I want** a MasterResearcher agent that deep-dives the problem domain
**So that** the Assessment Team has rich context for planning

**Acceptance Criteria:**
- [ ] `agents/master_researcher.py` created
- [ ] Takes raw task → produces `ResearchReport` (market, tech, risks, SWOT)
- [ ] Uses premium model (t=0.4)
- [ ] System prompt in `prompts/master_researcher.md`
- [ ] Stores findings in team memory
- [ ] Unit tests with mocked LLM

### Story 3.2: Assessor Agent
**As a** system
**I want** Assessor agents that evaluate feasibility, scope, and effort
**So that** we know what resources the task needs

**Acceptance Criteria:**
- [ ] `agents/assessor.py` created (extends existing prompt)
- [ ] Takes `ResearchReport` → produces `AssessmentResult` (complexity, risk, effort)
- [ ] Semi-deterministic (t=0.3), structured Pydantic output
- [ ] Two assessors run independently, results compared
- [ ] Unit tests

### Story 3.3: DevilsAdvocate Agent
**As a** system
**I want** a Devil's Advocate that challenges all assumptions
**So that** we catch blind spots before committing resources

**Acceptance Criteria:**
- [ ] `agents/devils_advocate.py` created
- [ ] Takes `ResearchReport` + `AssessmentResult` → produces `ChallengeReport`
- [ ] Higher temperature (t=0.7) for creative objections
- [ ] System prompt in `prompts/devils_advocate.md`
- [ ] Unit tests

### Story 3.4: HierarchyBuilder Agent
**As a** system
**I want** a HierarchyBuilder that outputs the exact team structure needed
**So that** we can dynamically compile the execution graph

**Acceptance Criteria:**
- [ ] `agents/hierarchy_builder.py` created
- [ ] Takes all assessment inputs → produces `HierarchySpec`
- [ ] `HierarchySpec` schema: `{num_pms, tls_per_pm, workers_per_tl, specialties[], rationale}`
- [ ] Fully deterministic (t=0.2), strict Pydantic validation
- [ ] Respects 1:3 ratio constraints
- [ ] Unit tests

### Story 3.5: Assessment Schemas
**As a** developer
**I want** Pydantic schemas for all assessment data
**So that** the assessment flow is fully typed

**Acceptance Criteria:**
- [ ] `ResearchReport` schema added to `schemas.py`
- [ ] `AssessmentResult` schema added
- [ ] `ChallengeReport` schema added
- [ ] `HierarchySpec` schema added
- [ ] All schemas have validators, defaults, docstrings
- [ ] Tests for schema creation and round-trip serialisation

---

## Sprint 4: Dynamic Pipeline Graph

**Goal:** Build the LangGraph that wires Assessment Team → dynamic hierarchy → execution.

**Duration:** 2-3 days

### Story 4.1: Assessment Subgraph
**As a** system
**I want** a LangGraph subgraph that runs the full assessment flow
**So that** task analysis happens as a discrete, traceable phase

**Acceptance Criteria:**
- [ ] `graphs/assessment.py` created
- [ ] Graph: `research → assess_parallel → challenge → build_hierarchy`
- [ ] State typed: `AssessmentState(TypedDict)`
- [ ] Outputs `HierarchySpec`
- [ ] Full LangSmith trace for assessment phase
- [ ] Tests with mocked agents

### Story 4.2: Dynamic Graph Compiler
**As a** system
**I want** a function that takes `HierarchySpec` and compiles a LangGraph
**So that** the execution structure matches the assessed needs

**Acceptance Criteria:**
- [ ] `compile_execution_graph(spec: HierarchySpec) → CompiledGraph`
- [ ] Dynamically creates N PM nodes, M TL nodes, K Worker nodes
- [ ] Each PM gets its own subgraph with TLs and Workers
- [ ] Respects 1:3 ratio constraints
- [ ] Tests: compile with various specs, verify graph structure

### Story 4.3: Full Dynamic Pipeline
**As a** system
**I want** a top-level graph: `assess → compile → execute → consolidate`
**So that** simple tasks still use single_team and complex tasks use dynamic hierarchy

**Acceptance Criteria:**
- [ ] `graphs/dynamic_pipeline.py` created
- [ ] Routes: simple tasks → `single_team.py`, complex → dynamic hierarchy
- [ ] Complexity threshold configurable
- [ ] `run_dynamic_pipeline()` convenience function
- [ ] Full LangSmith trace tree (assessment + execution)
- [ ] Integration tests

### Story 4.4: API Integration
**As a** frontend developer
**I want** `POST /api/tasks` to use the dynamic pipeline
**So that** users get the full assessment + execution flow

**Acceptance Criteria:**
- [ ] `/api/tasks` routes to `run_dynamic_pipeline()` instead of `run_single_team()`
- [ ] Backward compatible: simple tasks still work exactly as before
- [ ] New WebSocket events: `agent_spawned`, `agent_message`, `metrics_update`
- [ ] `GET /api/tasks/{id}/agents` returns the dynamically-built hierarchy
- [ ] Frontend receives and renders new event types

---

## Sprint 5: RL Sidecar + Agents-as-Judge (Phase 3)

**Goal:** Score every agent after each task, use scores for model routing.

**Duration:** 3-4 days (deferred to after Sprint 4 is stable)

### Story 5.1: RL Scoring Engine
### Story 5.2: Agents-as-Judge Pipeline
### Story 5.3: Score-Based Model Router
### Story 5.4: Performance Dashboard View

*(Detailed stories to be written when Sprint 4 is complete)*

---

## Sprint 6: Hardening (Phase 4)

**Goal:** Production-readiness improvements.

**Duration:** 3-5 days (deferred)

### Story 6.1: Circuit Breakers
### Story 6.2: Redis Task Store
### Story 6.3: Budget Enforcement Middleware
### Story 6.4: OpenTelemetry Integration

*(Detailed stories to be written when Sprint 5 is complete)*

---

## Dependency Graph

```
Sprint 1 (LangSmith) ←── Sprint 2 (LlamaIndex Memory)
       ↓                          ↓
Sprint 3 (Assessment Agents) ←────┘
       ↓
Sprint 4 (Dynamic Pipeline)
       ↓
Sprint 5 (RL Sidecar)
       ↓
Sprint 6 (Hardening)
```

Sprint 1 is independent — can start immediately.
Sprint 2 is independent — can run in parallel with Sprint 1.
Sprint 3 depends on Sprint 2 (agents need real memory).
Sprint 4 depends on Sprint 3 (pipeline needs assessment agents).
