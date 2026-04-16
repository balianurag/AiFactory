# Progress Tracker

## Phase Overview

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 0** | Foundation — scaffolding, schemas, base agent, memory, config | ✅ Complete |
| **Phase 1** | Single-Team Pipeline — PM → TL → Workers end-to-end (legacy) | ✅ Complete |
| **Phase 1.5** | HTTP/WebSocket API — all views wired to real backend | ✅ Complete |
| **Phase 2.0** | LangSmith observability — traces, costs, UI Traces view | ✅ Complete |
| **Phase 2.1** | LiteLLM migration — replaced OpenAI SDK with cross-provider abstraction | ✅ Complete |
| **Phase 2.2** | Ability-based model selection — 10-ability dot-product scoring engine | ✅ Complete |
| **Phase 2.3** | YAML config-driven agents — one `PipelineAgent` class, all roles in YAML | ✅ Complete |
| **Phase 2.4** | 8-node dynamic pipeline — production architecture | ✅ Complete |
| **Phase 2.5** | Pipeline wired to API, rate limiting, code execution sandbox | ✅ Complete |
| **Phase 2.6** | E2B cloud sandbox integration — dual-backend code execution | ✅ Complete |
| **Phase 2.7** | Code optimisation flow — DSA excellence, task merging, sandbox awareness | ✅ Complete |
| **Phase 2.8** | ComplexityGate v2 — 4-tier routing (math bypass, static facts, fast_llm, full) | ✅ Complete |
| **Phase 2.9** | Memory Bridge — MemoryBridge wired into every PipelineAgent | ✅ Complete |
| **Phase 3.5** | Token Efficiency — Anthropic prompt caching, LiteLLM response cache, smart judge | ✅ Complete |
| **Phase 4.0** | Docker MCP Sandbox — warm container pool, FastMCP server, CLI commands | ✅ Complete |
| **Phase 3** | RL-lite adaptive model scoring — quality → Qdrant, RL signal in selection | ✅ Complete |
| **Phase 4.1** | MCP Catalog provider — MCPProvider + 6 configured servers | ✅ Complete |
| **Phase 4.2** | DEPLOY_MODE routing — local stdio vs cloud HTTP session pool | ✅ Complete |
| **Phase 4.3** | Parallel workers + role specialisation + self-correction + dependency waves | ✅ Complete |
| **Phase 5** | Hardening — circuit breakers, budget caps, Redis task store | ⬜ Not started |

---

## Phase 0: Foundation ✅

**Completed: 27 March 2026**

| Component | File | Description |
|-----------|------|-------------|
| A2A Schemas | `core/schemas.py` | 25+ Pydantic models — every message type, agent configs, RL performance records, memory entries |
| Base Agent | `core/base_agent.py` | Abstract agent with LiteLLM calls, memory integration, token tracking, budget enforcement |
| Memory Router | `core/memory_router.py` | 3-layer Qdrant memory — store/retrieve/promote/dedup across Agent/Team/Global layers |
| Settings | `config/settings.py` | Pydantic Settings from `.env` — API keys, model tiers, memory configs, budget limits |
| Model Config | `config/models.yaml` | Model pricing, context windows, tier mappings |
| Project setup | `pyproject.toml` | Python 3.13+, uv-managed, all dependencies declared |

---

## Phase 1: Single-Team Pipeline ✅ (Legacy — kept for backward compat)

**Completed: 27 March 2026**

The original pipeline architecture: PM → TeamLead → Workers via `graphs/single_team.py`. This was the v1 execution model. It is **superseded** by the 8-node dynamic pipeline (Phase 2.4) and kept only for backward compatibility. New work uses `graphs/dynamic_pipeline.py` exclusively.

| Component | File | Status |
|-----------|------|--------|
| Worker Agent | `agents/worker.py` | Legacy |
| Team Lead Agent | `agents/team_lead.py` | Legacy |
| Project Manager Agent | `agents/project_manager.py` | Legacy |
| LangGraph Pipeline | `graphs/single_team.py` | Legacy |

---

## Phase 1.5: HTTP/WebSocket API ✅

**Completed: ~30 March 2026**

FastAPI app wired to the pipeline with real-time WebSocket streaming.

| Component | File | Description |
|-----------|------|-------------|
| API server | `api/main.py` | `/api/health`, `/api/models`, `/api/tasks`, `/api/observability/*`, WS `/api/ws/{id}` |
| Rate limiting | `core/rate_limiter.py` | slowapi, 4 tiers |
| DB persistence | `api/main.py` | PostgreSQL tasks/agents/messages |
| Redis cache | `api/main.py` | Hot task cache |

---

## Phase 2.0: LangSmith Observability ✅

**Completed: 2 April 2026**

| Component | File | Description |
|-----------|------|-------------|
| Observability module | `core/observability.py` | LangSmith setup, `@traceable` decorator (zero-cost passthrough when key absent) |
| BaseAgent tracing | `core/base_agent.py` | `think()` decorated with `@traceable` |
| API endpoints | `api/main.py` | `GET /api/observability/runs`, `/runs/{id}` |

**Key design:** `@traceable` is a direct passthrough when `LANGCHAIN_API_KEY` is absent — zero overhead.

---

## Phase 2.1: LiteLLM Migration ✅

Replaced all direct OpenAI SDK calls with `litellm.acompletion()`. Single string change to switch providers. Supports OpenAI, Anthropic, Google, OpenRouter, Bedrock, local proxies.

---

## Phase 2.2: Ability-Based Model Selection ✅

**File:** `core/model_selector.py`, `config/model_registry.yaml`

Each model rated 0–3 on 10 abilities: reasoning, coding, math, writing, instruction_following, tool_use, knowledge, speed, cost_efficiency, context_handling. Agent YAML files declare required abilities. Dot-product scoring selects best model per role + mode. Three modes: `best`, `balanced`, `cost_efficient`.

> **Note:** This phase was originally labelled "Assessment Team + Dynamic Hierarchy Builder" in early planning docs. That concept was redesigned — the assessment function was distributed across `judge`, `analyser`, and `execution_lead` nodes in the 8-node pipeline. Ability-based model selection is what was actually built in this slot.

---

## Phase 2.3: YAML Config-Driven Agents ✅

**Files:** `core/agent_factory.py`, `config/agents/*.yaml`

Single `PipelineAgent` class handles all roles. Behaviour comes entirely from YAML configs. Agent roles: master, judge, analyser, researcher, execution_lead, worker (+ 4 specialist variants).

---

## Phase 2.4: 8-Node Dynamic Pipeline ✅

**File:** `graphs/dynamic_pipeline.py`

The production architecture. Replaced the Phase 1 PM→TL→Worker structure.

```
master_receive → judge_brief → analyse
  → [researcher → reanalyse]?   (optional, conditional)
  → execution_lead (spawns N workers dynamically)
  → judge_review → master_respond
```

- Analyser decomposes task into N atomic tasks (count varies per run based on complexity)
- Execution Lead spawns exactly N workers — one per task
- Workers are independent, parallel, role-specialised agents (not a hierarchy)
- Direct-answer bypass for trivial queries (no workers spawned)

---

## Phase 2.5–2.7: API Wiring, Sandbox, Code Optimisation ✅

- Pipeline wired to `POST /api/tasks`
- Code execution sandbox: local subprocess
- Worker self-correction loop
- E2B cloud sandbox (dual-backend)
- DSA-optimised coding prompts, task merging, sandbox isolation awareness

---

## Phase 2.8: ComplexityGate v2 ✅

**File:** `core/complexity_gate.py`, `config/gate_patterns.yaml`

4-tier routing before the pipeline runs:
1. **Pure math bypass** — AST eval, 0 tokens, <1ms (`2 + 2 = 4`)
2. **Unit conversion bypass** — regex + lookup, 0 tokens (`100°C to °F`)
3. **Static fact bypass** — dict lookup, 0 tokens (`capital of France`)
4. **Fast LLM** — single master call, ~1-3s (simple Q&A)
5. **Full pipeline** — 8-node multi-agent (complex tasks)

Gate fires on both `aifactory new` and `aifactory run`.

---

## Phase 2.9: Memory Bridge ✅

**File:** `core/memory_bridge.py`

Fail-safe adapter between `PipelineAgent` and `MemoryRouter`. All ops try/except-wrapped at both bridge and `think()` level — memory failures never break the pipeline. Retrieval before `think()`, storage after. SHARE: updates persisted to team layer. Auto-promotion at N-turn threshold.

---

## Phase 3.5: Token Efficiency ✅

Three layered optimisations:
- **Anthropic prompt caching** — `cache_control: ephemeral` on system messages for Claude models. 60–90% cost reduction on repeat calls.
- **LiteLLM response cache** — in-memory cache (TTL configurable). All providers benefit from identical `(model, messages)` hits returning 0-token cached responses.
- **Smart judge** — skip judge LLM call for single-task runs (~1000 tokens saved). Synthetic assessment from run metadata.
- **Shared context pruning** — cap at 25 keys after each node.

---

## Phase 4.0: Docker MCP Sandbox ✅

**Files:** `core/sandbox_pool.py`, `mcp/sandbox_server.py`, `mcp/client.py`

- Warm Docker container pool (min_idle=4, max_size=16), 10–50ms exec vs 1–2s cold start
- Security: `--network=none --cap-drop=ALL --read-only --tmpfs /workspace --pids-limit=64 --memory=512m` + custom seccomp profile. gVisor optional.
- FastMCP server exposes `execute_python/bash/javascript + pool_status`
- Two client modes: HTTP (prod) or stdio/in-process (dev)
- Backend priority: docker > e2b > local subprocess
- Package registry (`config/package_registry.yaml`) injected into worker prompts

---

## Phase 3: RL-lite Adaptive Model Scoring ✅

**File:** `core/rl_scorer.py`

- Stores `(model_id, task_type) → quality_score` in Qdrant collection `aifactory_rl_performance`
- After each run, `judge_review` calls `record_pipeline_performances()` — normalises judge quality (0–3 → 0–10) per worker, persists to Qdrant
- Before worker model selection, `create_worker_config()` calls `get_model_rl_scores(task_type)` and passes results to `select_model()`
- RL signal applied as multiplicative factor: `adj = 1 + rl_weight × (rl_score/5 − 1)`, capped at ±20%
- Only models with ≥ `AIFACTORY_RL_MIN_SAMPLES` records (default: 3) are adjusted
- CLI: `aifactory rl stats [--type coding]` / `aifactory rl reset`

---

## Phase 4.1: MCP Catalog Provider ✅

**Files:** `integrations/providers/mcp_provider.py`, `config/integrations/mcp_catalog.yaml`

`MCPProvider` alongside `ComposioProvider`. 6 configured servers: GitHub (official Docker), Atlassian/Jira+Confluence (uvx), Notion (npx), Linear (npx), Google Workspace (disabled — token vault pending), Slack (disabled). `AsyncExitStack` session management. `server::tool_name` slug format for `ACTION:mcp` blocks.

---

## Phase 4.2: DEPLOY_MODE Routing ✅

`AIFACTORY_DEPLOY_MODE=local` uses stdio subprocess per MCP server. `cloud` uses async HTTP session pool (N persistent sessions per server, `asyncio.Queue`-backed acquire/release).

---

## Phase 4.3: Parallel Workers + Role Specialisation + Self-Correction ✅

- **Parallel execution:** `asyncio.gather()` wave-based dispatch. `_build_dependency_waves()` topological sort → `list[list[task]]`. Net context deltas returned per worker, merged sequentially after gather.
- **Role specialisation:** 4 specialist YAML configs (`worker_coder/researcher/writer/analyst.yaml`). Task type detected by keyword. Specialist persona prepended before task instruction.
- **Self-correction:** After tool rounds, non-tool workers do one `VALIDATED / REVISION:` self-check. One revision max — no loop.
- **Dependency enforcement:** `dependencies_map` extracted from execution lead JSON, used for topological wave ordering.

---

## Phase 5: Hardening ⬜ Not started

Planned:
- Circuit breakers (pipeline can detect "structure isn't working" and fail gracefully)
- Hard budget caps per project with mid-run enforcement
- Redis task store (replace in-memory state)
- Evaluation datasets (LangSmith) for agent quality regression testing
