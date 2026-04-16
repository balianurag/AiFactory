# AiFactory — Architecture v2

**Version:** 2.1 (updated April 2026 to reflect production implementation)
**Status:** Current — describes the system as actually built

---

## 1. System Overview

AiFactory is a self-assembling, self-optimising multi-agent orchestration system.
A raw human idea goes in; production-grade output comes out.

### Core Principle

The system **assembles its own workforce at runtime**. A complexity gate first decides whether a task needs the full pipeline at all. If it does, an analyser decomposes the task into atomic sub-tasks, and an execution lead spawns exactly the right number of specialist workers — each with the best model for their specific task type.

### Production Pipeline (8 nodes)

```
Human Input
     │
     ▼
ComplexityGate (zero-LLM pre-filter)
  ├─ pure math / unit conversion / static fact → instant answer, 0 tokens
  ├─ simple Q&A → fast_llm (single master call, ~1-3s)
  └─ complex task → 8-node pipeline
          │
          ▼
    master_receive         — acknowledges request, seeds shared context
          │
          ▼
    judge_brief            — defines success criteria
          │
          ▼
    analyse                — decomposes into N atomic tasks
          │                  (or direct_answer for trivial queries)
          │
          ├─[research needed?]──▶ researcher ──▶ reanalyse
          │
          ▼
    execution_lead         — builds worker dispatch plan
          │                  creates N workers dynamically
          │                  assigns per-task model + specialist role
          │                  runs workers in parallel dependency waves
          │
          ▼
    judge_review           — evaluates each agent's output
          │                  records RL performance scores
          │
          ▼
    master_respond         — synthesises final response
```

### What "self-assembling" means in practice

- The **number of workers** varies per run based on how the analyser decomposes the task
- The **model per worker** is selected by ability-dot-product scoring against the task type
- The **role per worker** (coder / researcher / writer / analyst) is detected from task keywords
- Workers in the same dependency wave **run in parallel** via `asyncio.gather()`
- There are no fixed ratios, no management layers — just N independent specialist agents

Real examples from production runs:
- "Write a Python function to find top 3 words, test it, show results" → 2 workers (one run) or 4 workers (another run), depending on analyser decomposition
- "Build a FastAPI todo app with auth" → typically 4–6 workers

---

## 2. Technology Stack

| Layer | Technology | Role |
|---|---|---|
| **Orchestration** | LangGraph | Stateful 8-node graph, checkpointing, `interrupt()` for human-in-the-loop |
| **LLM Calls** | LiteLLM | Unified interface to 300+ models — OpenAI, Anthropic, Google, Bedrock, OpenRouter |
| **Memory & RAG** | LlamaIndex + Qdrant | 3-layer semantic memory (agent / team / global) |
| **Observability** | LangSmith | Full trace tree per LLM call, cost tracking, `@traceable` decorator |
| **Schemas** | Pydantic v2 | All A2A communication — structured JSON, no free-form text between agents |
| **API** | FastAPI + WebSocket | HTTP REST + real-time streaming |
| **Code Execution** | Docker MCP / E2B / subprocess | 3-tier sandbox fallback with warm container pool |
| **Integrations** | Composio + MCP Catalog | 1000+ tools across GitHub, Jira, Slack, Notion, etc. |
| **CLI** | Typer + Rich | Full developer CLI with 15+ commands |
| **Vector DB** | Qdrant | Memory store + RL performance records |

---

## 3. Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LangSmith (Observability)                       │
│  Every LLM call, graph node, memory op → traced automatically      │
│  Zero-cost when LANGCHAIN_API_KEY absent (@traceable passthrough)   │
└────────────────────────────┬────────────────────────────────────────┘
                             │ traces
┌────────────────────────────▼────────────────────────────────────────┐
│                    LangGraph (Orchestration Core)                    │
│                                                                      │
│  8-node pipeline:                                                    │
│    master_receive → judge_brief → analyse                            │
│    → [researcher → reanalyse]? (conditional)                        │
│    → execution_lead → [N workers in parallel waves]                 │
│    → judge_review → master_respond                                   │
│                                                                      │
│  Each agent: PipelineAgent (YAML-configured, LiteLLM-backed)        │
│  Human-in-the-loop: interrupt() + MemorySaver checkpointing         │
│  SHARE: protocol: agents broadcast context via text blocks           │
│                                                                      │
└──────────────┬──────────────────────────┬───────────────────────────┘
               │ store / retrieve          │ tool calls
┌──────────────▼────────────┐  ┌──────────▼───────────────────────────┐
│  LlamaIndex + Qdrant       │  │  Integrations                        │
│  (Memory & RAG)            │  │                                      │
│                            │  │  ComposioProvider                    │
│  Agent Index (per agent)   │  │    → GitHub, GitLab, Jira, Notion,  │
│  Team Index (per run_id)   │  │      Slack, Gmail, Linear, Drive     │
│  Global Index              │  │                                      │
│                            │  │  MCPProvider (catalog)               │
│  Stub embeddings (dev)     │  │    → GitHub official Docker image   │
│  Real embeddings (prod)    │  │    → Atlassian/Jira uvx              │
│                            │  │    → Notion/Linear npx               │
└────────────────────────────┘  │                                      │
                                │  Docker MCP Sandbox                  │
┌───────────────────────────┐   │    → warm pool (4-16 containers)    │
│  RL Performance Store     │   │    → execute_python/bash/javascript  │
│  (Qdrant collection)      │   │    → security: cap-drop, seccomp,   │
│                            │   │      read-only FS, no network       │
│  (model, task_type)        │   └──────────────────────────────────────┘
│    → avg quality score     │
│  Read before model select  │
│  Write after judge_review  │
└───────────────────────────┘
```

---

## 4. Agent Roles

### 4.1 Pipeline Nodes

All agents are instances of `PipelineAgent` (single class). Behaviour comes from `config/agents/*.yaml`.

| Node | YAML Config | Role |
|---|---|---|
| `master_receive` | `master.yaml` | Acknowledges request, seeds `shared_context.objective` |
| `judge_brief` | `judge.yaml` | Defines success criteria, what will be evaluated |
| `analyse` | `analyser.yaml` | Decomposes task into N atomic tasks OR returns `direct_answer` |
| `researcher` | `researcher.yaml` | Gathers research (optional, conditional on analyser output) |
| `reanalyse` | `analyser.yaml` | Refines task breakdown post-research (optional) |
| `execution_lead` | `execution_lead.yaml` | Plans worker dispatch, spawns N workers, consolidates |
| `judge_review` | `judge.yaml` | Evaluates outputs, produces quality metrics, records RL scores |
| `master_respond` | `master.yaml` | Synthesises final user-facing response |

### 4.2 Worker Specialist Roles

Workers are plain `PipelineAgent` instances. Task type is inferred from keywords, and the matching specialist YAML config is loaded:

| Specialist | YAML Config | Used when task involves |
|---|---|---|
| Coder | `worker_coder.yaml` | code, implement, function, script, algorithm, debug, test |
| Researcher | `worker_researcher.yaml` | research, investigate, find, summarise, compare, analyse |
| Writer | `worker_writer.yaml` | write, draft, document, explain, describe, create content |
| Analyst | `worker_analyst.yaml` | analyse, evaluate, assess, review, breakdown, recommend |
| Generic | `worker.yaml` | fallback for anything else |

### 4.3 Worker Lifecycle

```
execution_lead decides:
  task_type → load specialist YAML
  required_abilities → select best model (ability dot-product + RL bias)
  dependencies → assign to dependency wave

worker runs:
  1. system_prompt = specialist_persona + task + context
  2. tool rounds (if ACTION: blocks in output)
  3. self-correction check: "VALIDATED or REVISION: <note>\n<corrected>"
     → one revision max, then done
  4. return (result_entry, ctx_delta)

parallel wave:
  asyncio.gather(*wave_workers)
  ctx_deltas merged sequentially after gather
```

---

## 5. Memory Architecture (LlamaIndex + Qdrant)

### 5.1 Three-Layer System

| Layer | Qdrant Collection | Scope | TTL |
|---|---|---|---|
| **Agent** | `aifactory_agent_{agent_id}` | Private per agent — raw thoughts, tool calls | Session |
| **Team** | `aifactory_team_{run_id}` | Shared per pipeline run — decisions, SHARE: updates | Project |
| **Global** | `aifactory_global` | Cross-run patterns and learnings | Permanent |

Each run uses its `run_id` as the `team_id` for Qdrant isolation — parallel runs don't contaminate each other.

### 5.2 Data Flow

```
Agent thinks → store in Agent layer
  │
  │ every N turns (AIFACTORY_MEMORY_PROMOTION_INTERVAL, default 10)
  ▼
LLM-summarised → promote to Team layer
  │
  │ on run completion (if RL quality score > threshold)
  ▼
High-quality learnings → promote to Global layer
```

SHARE: updates (`SHARE: key: value` in agent output) are also persisted to team layer immediately — lightweight cross-agent context without full memory write overhead.

### 5.3 Query Pipeline

```
Before each agent think():
  1. Query own Agent layer (top_k configurable)
  2. Query Team layer (shared run context)
  3. Query Global layer (cross-run patterns)
  4. Merge + inject into system prompt as extra_context

After think():
  SHARE: blocks extracted and merged into shared_context dict
  Memory stored to Agent layer
```

### 5.4 MemoryBridge

`core/memory_bridge.py` wraps all memory operations. Every operation is try/except-wrapped — memory failures never break the pipeline. Wired into every `PipelineAgent` by `build_dynamic_pipeline()` using the run's `run_id` as `team_id`.

---

## 6. Model Selection

### 6.1 10-Ability Scoring

Each model in `config/model_registry.yaml` is rated 0–3 on:

| Ability | What it measures |
|---|---|
| reasoning | Multi-step logical reasoning |
| coding | Code generation and debugging |
| math | Mathematical computation and proof |
| writing | Prose quality and creativity |
| instruction_following | Adherence to structured instructions |
| tool_use | Accurate tool/function calling |
| knowledge | Factual accuracy and breadth |
| speed | Response latency |
| cost_efficiency | Cost per token relative to quality |
| context_handling | Long-context coherence |

### 6.2 Selection Algorithm

```
score(model, agent) = dot_product(model.abilities, agent.required_abilities)
                    × mode_bias(mode)   # best / balanced / cost_efficient
                    × rl_bias(model, task_type)  # ±20% from RL history

best_model = argmax(scores)
```

RL bias applies only when ≥ `AIFACTORY_RL_MIN_SAMPLES` records exist for that `(model, task_type)` pair.

### 6.3 Pipeline Modes

| Mode | Bias | Typical model choices |
|---|---|---|
| `best` | Quality-first | Claude Opus, GPT-5 for premium roles |
| `balanced` | Default | Claude Sonnet, GPT-4o mixed |
| `cost_efficient` | Cost-first | Claude Haiku, Gemini Flash for most roles |

---

## 7. ComplexityGate

`core/complexity_gate.py` runs before any LLM call on every `new` and `run` invocation.

```
Input task
  │
  ├─ pure math (AST eval)          → bypass: instant answer, 0 tokens
  ├─ unit conversion (regex)       → bypass: instant answer, 0 tokens
  ├─ static fact (dict lookup)     → bypass: instant answer, 0 tokens
  ├─ compound bypass query         → bypass: answer each part, 0 tokens
  ├─ simple Q&A (pattern score)    → fast_llm: single master call
  └─ complex task (pattern score)  → full 8-node pipeline
```

Patterns loaded from `config/gate_patterns.yaml` — no code changes needed to tune routing.
CLI: `aifactory gate "<task>"` — preview the decision without running anything.

---

## 8. Observability

### 8.1 What Gets Traced

| Component | Trace Type | Data Captured |
|---|---|---|
| `BaseAgent.think()` | LLM call | Model, tokens in/out, cost, latency, prompt, response |
| LangGraph node transitions | Chain step | Node name, state in/out, duration |
| Memory ops | Tool call | Layer, query, result count, latency |
| Pipeline run | Root trace | Full task → response tree, total cost, total time |

### 8.2 Zero-Cost When Disabled

`@traceable` is a direct passthrough when `LANGCHAIN_API_KEY` is absent. No try/catch overhead, no network calls, no import-time cost.

### 8.3 Action Audit Log

Every agent maintains `_actions: list` — timestamped log of every tool call, LLM call input/output, and SHARE: update. Used by `judge_review` for quality evaluation.

---

## 9. File Structure (Current)

```
src/aifactory/
├── core/
│   ├── base_agent.py          # LiteLLM agent, memory, token tracking, @traceable
│   ├── agent_factory.py       # YAML loader, SHARE: protocol, AgentRegistry
│   ├── model_selector.py      # 10-ability dot-product scoring
│   ├── schemas.py             # All Pydantic models
│   ├── memory_router.py       # 3-layer Qdrant memory
│   ├── memory_bridge.py       # Fail-safe memory adapter for PipelineAgent
│   ├── complexity_gate.py     # 4-tier pre-pipeline routing
│   ├── rl_scorer.py           # RL-lite adaptive model scoring
│   ├── sandbox.py             # 3-tier code execution (docker > e2b > local)
│   ├── sandbox_pool.py        # Async warm Docker container pool
│   ├── assessor.py            # Task complexity classifier (used by complexity_gate)
│   ├── observability.py       # LangSmith setup, @traceable
│   └── rate_limiter.py        # slowapi rate limiting
├── agents/                    # Legacy Phase 1 agents (backward compat only)
│   ├── worker.py
│   ├── team_lead.py
│   └── project_manager.py
├── graphs/
│   ├── dynamic_pipeline.py    # ★ Production: 8-node LangGraph pipeline
│   └── single_team.py         # Legacy Phase 1 pipeline (backward compat)
├── mcp/
│   ├── sandbox_server.py      # FastMCP server: execute_python/bash/js + pool_status
│   └── client.py              # MCP client factory (HTTP or stdio)
├── api/
│   └── main.py                # FastAPI: REST + WebSocket
├── integrations/
│   ├── base.py
│   ├── registry.py
│   └── providers/
│       ├── composio.py        # Composio integration
│       └── mcp_provider.py    # MCP catalog provider
├── cli/
│   ├── __init__.py            # App root + command registration
│   ├── _helpers.py            # Shared console, dump I/O, formatting
│   ├── cmd_new.py             # new / resume
│   ├── cmd_run.py             # run (advanced)
│   ├── cmd_status.py          # status / inspect
│   ├── cmd_connect.py         # connect add/list/remove / tools
│   ├── cmd_ask.py             # ask
│   ├── cmd_agents.py          # agents list/roles
│   ├── cmd_cost.py            # cost today/week/all/run/estimate
│   ├── cmd_info.py            # info / gate / models list/tiers
│   ├── cmd_memory.py          # memory stats/search/clear
│   ├── cmd_sandbox.py         # sandbox build/status/test/pull
│   └── cmd_rl.py              # rl stats/reset
└── config/
    ├── settings.py
    ├── model_registry.yaml    # ~20+ models with 10-ability scores + pricing
    ├── graph.yaml             # Runtime config (mode, fallback model)
    ├── gate_patterns.yaml     # ComplexityGate YAML-driven patterns
    ├── package_registry.yaml  # Pre-installed sandbox packages
    ├── tool_catalog.yaml      # Toolkit descriptions for execution lead
    └── agents/                # Per-role YAML configs
        ├── master.yaml
        ├── judge.yaml
        ├── analyser.yaml
        ├── researcher.yaml
        ├── execution_lead.yaml
        ├── worker.yaml
        ├── worker_coder.yaml
        ├── worker_researcher.yaml
        ├── worker_writer.yaml
        └── worker_analyst.yaml
```

---

## 10. Deployment Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Vercel         │     │   Render          │     │   Qdrant Cloud  │
│   (Frontend)     │────▶│   (Backend)       │────▶│   (Vector DB)   │
│   Next.js        │     │   FastAPI         │     │                 │
│   React          │     │   LangGraph       │     │   aifactory_    │
│   Tailwind       │     │   LiteLLM         │     │   agent/team/   │
│                  │     │   LlamaIndex      │     │   global/rl     │
└─────────────────┘     └────────┬──────────┘     └─────────────────┘
                                 │
                        ┌────────▼──────────┐     ┌─────────────────┐
                        │   LangSmith        │     │   Docker        │
                        │   (Observability)  │     │   (Sandbox pool)│
                        └───────────────────┘     └─────────────────┘
```

### DEPLOY_MODE

| Mode | MCP client | Use case |
|---|---|---|
| `local` (default) | stdio subprocess per server | Dev, CI, local runs |
| `cloud` | async HTTP session pool (N sessions/server, asyncio.Queue) | Production Render deployment |
