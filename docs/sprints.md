# AiFactory — Sprint Log

**Methodology:** Spec-driven, story-based sprints
**Last updated:** April 2026

> **Note:** Sprints 1–2 were executed as planned. Sprints 3–4 (Assessment Team + HierarchyBuilder) were redesigned before implementation — see "Architecture Pivot" below. The actual production architecture is the 8-node dynamic pipeline documented in `docs/architecture-v2.md`.

---

## Sprint 1: LangSmith Observability Integration ✅

**Goal:** Every LLM call and graph transition traced in LangSmith.
**Completed:** 2 April 2026

### Story 1.1: LangSmith Backend Wiring ✅
- `core/observability.py` created with LangSmith setup and `@traceable` helpers
- `BaseAgent.think()` decorated — every LLM call appears in LangSmith
- Zero-cost when `LANGCHAIN_API_KEY` absent (direct passthrough)

### Story 1.2: Observability API Endpoints ✅
- `GET /api/observability/runs` — list recent runs
- `GET /api/observability/runs/{run_id}` — single run detail
- Graceful fallback when key not configured

### Story 1.3: Frontend Observability View ✅
- `ObservabilityView.tsx` component with trace timeline
- Navbar "Traces" link added

### Story 1.4: Environment & Config Update ✅
- `settings.py` has `langchain_api_key`, `langchain_project`, `langchain_tracing` fields

---

## Sprint 2: LlamaIndex Memory Migration ✅

**Goal:** Real semantic embeddings via LlamaIndex + Qdrant.
**Completed:** ~5 April 2026

### Story 2.1: LlamaIndex Memory Router ✅
- `MemoryRouter` with `VectorStoreIndex` per layer (agent / team / global)
- Real embeddings via `text-embedding-3-small` (configurable)
- Stub embeddings (SHA-256 hash, 1536-dim) for offline dev
- Same public API: `store()`, `retrieve()`, `promote()`

### Story 2.2: Cross-Layer Query Pipeline ✅
- `retrieve()` queries agent → team → global, merges results
- Deduplication by semantic similarity (threshold configurable)
- `top_k` configurable per layer

### Story 2.3: Memory Promotion with Summarisation ✅
- LLM-based summarisation on promotion
- Promotion interval configurable via `AIFACTORY_MEMORY_PROMOTION_INTERVAL`

---

## Architecture Pivot (Sprints 3–4 Redesigned)

**Date:** ~6 April 2026

Sprints 3 and 4 were originally spec'd to build an "Assessment Team" — a separate subgraph of MasterResearcher, Assessor ×2, DevilsAdvocate, and HierarchyBuilder agents that would produce a `HierarchySpec` and compile a dynamic PM→TL→Worker graph at runtime.

**This approach was redesigned before implementation** for the following reasons:

1. **Complexity for marginal gain** — A 5-agent assessment phase before every task adds latency (~15-30s) and cost for benefit achievable with a single well-prompted analyser agent.
2. **Static hierarchy is the wrong model** — Having a fixed PM→TL→Worker structure even if dynamically sized misses the point. What matters is task decomposition, not management layers.
3. **Config-driven agents are simpler** — A single `PipelineAgent` class with YAML configs is more maintainable than 5+ separate agent classes.

**What was built instead (Sprints 3–4 equivalent):**

| Original plan | What was actually built |
|---|---|
| MasterResearcher agent | `researcher` node in 8-node pipeline (conditional, config-driven) |
| Assessor ×2 agents | `analyse` node — decomposes task into atomic tasks |
| DevilsAdvocate agent | `judge_brief` + `judge_review` nodes — quality criteria + evaluation |
| HierarchyBuilder + HierarchySpec | `execution_lead` node — dynamically spawns N workers per task breakdown |
| `compile_execution_graph()` | Not needed — workers spawned inline, no graph recompilation |
| PM→TL→Worker hierarchy | Flat: Execution Lead → N Workers (parallel waves) |
| Fixed 1:3 ratios | Dynamic: N workers = N tasks from analyser |

The result is a simpler, faster, and more capable architecture. Real runs show 2–4 workers per task dynamically spawned by the analyser's task decomposition.

---

## Sprint 3 (Actual): 8-Node Dynamic Pipeline ✅

**Completed:** ~7 April 2026

### What was built

- `graphs/dynamic_pipeline.py` — 8-node LangGraph: `master_receive → judge_brief → analyse → [researcher → reanalyse]? → execution_lead → judge_review → master_respond`
- `core/agent_factory.py` — YAML agent loader, SHARE: protocol, AgentRegistry, package registry injection
- `config/agents/*.yaml` — All agent role configs (master, judge, analyser, researcher, execution_lead, worker + 4 specialist variants)
- `core/model_selector.py` — 10-ability dot-product scoring
- `config/model_registry.yaml` — ~20+ models with ability scores and pricing
- Worker specialisation: `worker_coder.yaml`, `worker_researcher.yaml`, `worker_writer.yaml`, `worker_analyst.yaml`
- Parallel wave execution via `asyncio.gather()` with dependency topological sort
- Self-correction loop (VALIDATED / REVISION:) on non-tool workers
- Human-in-the-loop via LangGraph `interrupt()` + `MemorySaver` checkpointing

---

## Sprint 4 (Actual): ComplexityGate + Memory Bridge + Token Efficiency ✅

**Completed:** ~9 April 2026

### What was built

- `core/complexity_gate.py` — 4-tier routing (math bypass, unit conversion, static fact, fast_llm, full pipeline)
- `config/gate_patterns.yaml` — YAML-driven pattern matching
- `core/memory_bridge.py` — Fail-safe memory adapter wired into every PipelineAgent
- Token efficiency: Anthropic prompt caching, LiteLLM response cache, smart judge skip
- Docker MCP sandbox warm pool (`core/sandbox_pool.py`, `mcp/sandbox_server.py`)
- RL-lite adaptive model scoring (`core/rl_scorer.py`)
- MCP Catalog provider + DEPLOY_MODE routing
- Modular CLI (`cli/cmd_*.py`) — new, run, status, inspect, ask, info, gate, memory, sandbox, rl, cost, connect, agents, models, tools

---

## Sprint 5: RL Sidecar + Agents-as-Judge (Phase 3) — Partially Complete

**RL-lite scoring:** ✅ Done (`core/rl_scorer.py` — records quality after runs, blends into model selection)

**Remaining:**
- Agents-as-Judge (peer review step — agents score each other's outputs) — not yet built
- Adaptive promotion policy (entropy-based trigger vs fixed N-turn interval) — not yet built
- Evaluation datasets in LangSmith for quality regression testing — not yet built

---

## Sprint 6: Hardening (Phase 5) ⬜ Not started

- Circuit breakers (pipeline halts gracefully instead of full crash)
- Hard budget caps per project with mid-run enforcement
- Redis task store (replace in-memory state)
- Evaluation datasets for agent quality regression

---

## Dependency Graph (Actual)

```
Sprint 1 (LangSmith) ─── Sprint 2 (LlamaIndex Memory)
                                  │
                          Sprint 3 (8-node pipeline + YAML agents)
                                  │
                          Sprint 4 (Gate + Memory Bridge + Docker + RL)
                                  │
                          Sprint 5 (RL complete + Agents-as-Judge)
                                  │
                          Sprint 6 (Hardening)
```
