# AiFactory — Architecture v2

**Version:** 2.0
**Date:** 2 April 2026
**Status:** Approved — implementation in progress

---

## 1. System Overview

AiFactory is a self-assembling, self-optimising multi-agent orchestration system.
A raw human idea goes in; production-grade output comes out.

### Core Principle

The system **designs its own workforce** at runtime. An Assessment Team analyses
the task, decides how many PMs, TLs, and Workers are needed, then the framework
dynamically compiles a LangGraph to execute that exact structure.

---

## 2. Technology Stack (Finalised)

| Layer | Technology | Role in System |
|---|---|---|
| **Orchestration** | LangGraph | Stateful graph engine — dynamic subgraphs, checkpointing, cycles |
| **LLM Calls** | LiteLLM | Unified interface to 300+ models via a single string |
| **Memory & RAG** | LlamaIndex + Qdrant | Real semantic embeddings, vector indexing, query engines per memory layer |
| **Observability** | LangSmith | Full trace tree for every LLM call, graph transition, cost tracking |
| **Schemas** | Pydantic v2 | All A2A communication — zero free-form text between agents |
| **API** | FastAPI + WebSocket | HTTP REST + real-time streaming to frontend |
| **Frontend** | Next.js 16 + React 19 | Chat-first UI with live agent stream, dashboard, LangSmith embed |
| **Vector DB** | Qdrant | 3-layer memory store (agent / team / global) |
| **CLI** | Typer + Rich | Developer interface for task execution, config, memory inspection |

### Why This Stack

| Decision | Rationale |
|---|---|
| **LangGraph** (not CrewAI/AutoGen) | Only framework with native runtime subgraph compilation, typed state, and checkpointing. Essential for dynamic hierarchy. |
| **LlamaIndex** (not raw Qdrant client) | Best-in-class indexing, chunking, and query pipelines. Turns our hash-based pseudo-embeddings into real semantic retrieval. |
| **LangSmith** (not Langfuse/custom) | First-party integration with LangGraph. Automatic trace trees, cost attribution, evaluation datasets. |
| **LiteLLM** (stays) | Already works. Unified interface to any provider. No reason to change. |
| **Qdrant** (stays) | Already works. LlamaIndex has a first-class `QdrantVectorStore` adapter. |

---

## 3. Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LangSmith (Observability)                       │
│  Every LLM call, graph node, memory op → traced automatically      │
│  Dashboard: https://smith.langchain.com                             │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ traces (via LangChain callbacks)
┌────────────────────────────────▼────────────────────────────────────┐
│                    LangGraph (Orchestration Core)                    │
│                                                                      │
│  Phase 1 (current):                                                  │
│    [START] → setup_agents → execute_project → [END]                  │
│    (1 PM → 1 TL → 3 Workers)                                        │
│                                                                      │
│  Phase 2 (next):                                                     │
│    [START] → assess_task ──→ build_hierarchy ──→ execute_teams ──→  │
│              (subgraph)      (dynamic spawn)     (N subgraphs)       │
│              │                                        │              │
│              ▼                                        ▼              │
│         MasterResearcher              PM(s) → TL(s) → Worker(s)     │
│         Assessor ×2                   (compiled at runtime)          │
│         DevilsAdvocate                                               │
│         HierarchyBuilder                                             │
│                                                                      │
│  Each agent node:                                                    │
│    BaseAgent.think() → LiteLLM → structured Pydantic response       │
│                                                                      │
└──────────────────┬──────────────────────────────────────────────────┘
                   │ store / retrieve
┌──────────────────▼──────────────────────────────────────────────────┐
│              LlamaIndex (Memory & RAG Layer)                         │
│                                                                      │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐       │
│  │ Agent Index    │  │ Team Index    │  │ Global Index       │       │
│  │ (per agent_id) │  │ (per team_id) │  │ (cross-project)    │       │
│  │               │  │               │  │                    │       │
│  │ VectorStore   │  │ VectorStore   │  │ VectorStore        │       │
│  │ Index +       │  │ Index +       │  │ Index +            │       │
│  │ QueryEngine   │  │ QueryEngine   │  │ QueryEngine        │       │
│  └───────┬───────┘  └───────┬───────┘  └────────┬──────────┘       │
│          │                  │                    │                   │
│          └──────────────────┼────────────────────┘                   │
│                             │                                        │
│                    Qdrant Vector DB                                   │
│             (in-memory dev / cloud prod)                             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Agent Hierarchy (Phase 2 Target)

### 4.1 Assessment Flow

```
Human Task
  │
  ▼
┌─────────────────────────────────────┐
│ Assessment Team Subgraph            │
│                                     │
│  MasterResearcher (premium, t=0.4)  │
│    → domain research, tech scan,    │
│      risk analysis, SWOT            │
│         │                           │
│         ▼                           │
│  Assessor ×2 (premium, t=0.3)      │
│    → feasibility, scope, effort     │
│    → semi-deterministic (schemas)   │
│         │                           │
│         ▼                           │
│  DevilsAdvocate (premium, t=0.7)   │
│    → challenge all assumptions      │
│    → adversarial review             │
│         │                           │
│         ▼                           │
│  HierarchyBuilder (premium, t=0.2) │
│    → outputs HierarchySpec:         │
│      { num_pms, tls_per_pm,         │
│        workers_per_tl, specialties } │
└──────────────┬──────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ Dynamic Graph Compilation            │
│                                      │
│  HierarchySpec → compile_graph()     │
│    → N PM subgraphs                  │
│    → M TL subgraphs per PM          │
│    → K Worker nodes per TL          │
│                                      │
│  Simple task:  1 PM, 1 TL, 2 workers│
│  Complex task: 3 PM, 9 TL, 27 work. │
└──────────────┬───────────────────────┘
               │
               ▼
         [Execute Teams]
               │
               ▼
         [Consolidation]
               │
               ▼
          PMDelivery
```

### 4.2 Agent Determinism Modes

| Agent | Temperature | Mode | Rationale |
|---|---|---|---|
| MasterResearcher | 0.4 | Semi-creative | Needs creative exploration of problem space |
| Assessor | 0.3 | Semi-deterministic | Structured evaluation, some flexibility |
| DevilsAdvocate | 0.7 | Creative | Must generate novel objections |
| HierarchyBuilder | 0.2 | Deterministic | Must output valid HierarchySpec |
| ProjectManager | 0.3 | Semi-deterministic | Strategic but structured planning |
| TeamLead | 0.3 | Semi-deterministic | Tactical decomposition |
| Worker (Coder) | 0.1 | Fully deterministic | Code must be precise |
| Worker (Researcher) | 0.3 | Semi-deterministic | Research needs some exploration |
| Worker (Writer) | 0.4 | Semi-creative | Writing benefits from variation |

### 4.3 Hierarchy Ratios

```
Assessment Team: always exactly 5 agents (researcher + 2 assessors + DA + builder)

Execution Teams:
  PM : TL    = 1 : 3 (max)
  TL : Worker = 1 : 3 (max)

  Sr. PM spawned when PM count > 2 (with Validator agents)

Minimum: 1 PM + 1 TL + 1 Worker = 3 agents
Maximum: 1 Sr.PM + 3 PM + 9 TL + 27 Workers = 40 agents
```

---

## 5. Memory Architecture (LlamaIndex + Qdrant)

### 5.1 Three-Layer System

| Layer | Qdrant Collection | LlamaIndex Index | Scope | TTL |
|---|---|---|---|---|
| **Agent** | `aifactory_agent_{agent_id}` | `VectorStoreIndex` per agent | Private episodic memory, raw thoughts, tool calls | Session |
| **Team** | `aifactory_team_{team_id}` | `VectorStoreIndex` per team | Shared decisions, contracts, intermediate artifacts | Project |
| **Global** | `aifactory_global` | `VectorStoreIndex` (single) | Cross-project patterns, reusable components, learnings | Permanent |

### 5.2 Data Flow

```
Agent thinks → stores in Agent layer (VectorStoreIndex.insert)
  │
  │ every N turns (promotion interval)
  ▼
Agent summaries → promoted to Team layer (with summarisation)
  │
  │ on project completion (if RL score > threshold)
  ▼
High-quality learnings → promoted to Global LTM
```

### 5.3 Query Pipeline

```
Agent needs context:
  1. Query own Agent index (top_k=5) → private recent memory
  2. Query Team index (top_k=3) → shared team knowledge
  3. Query Global index (top_k=2) → cross-project patterns
  4. Merge + deduplicate → inject into LLM prompt as extra_context
```

### 5.4 Migration from Current MemoryRouter

| Aspect | Current (v1) | Target (v2) |
|---|---|---|
| Embeddings | SHA-256 hash (pseudo) | Real embeddings via `text-embedding-3-small` |
| Indexing | Raw Qdrant `upsert` | LlamaIndex `VectorStoreIndex` |
| Retrieval | Qdrant `search` | LlamaIndex `QueryEngine` with semantic ranking |
| Chunking | None | LlamaIndex `SentenceSplitter` |
| Dedup | Hash-based | Semantic similarity threshold |
| API | Custom `store()`/`retrieve()` | Same API, LlamaIndex under the hood |

---

## 6. Observability (LangSmith)

### 6.1 What Gets Traced

| Component | Trace Type | Data Captured |
|---|---|---|
| `BaseAgent.think()` | LLM call | Model, tokens in/out, cost, latency, prompt, response |
| LangGraph node transitions | Chain step | Node name, input state, output state, duration |
| Memory store/retrieve | Tool call | Layer, query, results count, latency |
| Pipeline run | Root trace | Full task → delivery tree, total cost, total time |

### 6.2 Integration Method

```python
# In BaseAgent.__init__ or think():
from langsmith import traceable
from langsmith.run_trees import RunTree

# Option 1: Decorator-based (simplest)
@traceable(name="agent_think", run_type="llm")
async def think(self, ...):
    ...

# Option 2: Callback-based (for LangGraph integration)
from langchain_core.callbacks import LangChainTracer
tracer = LangChainTracer(project_name="aifactory")
```

### 6.3 Environment Variables

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_xxxxx
LANGCHAIN_PROJECT=aifactory
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### 6.4 Frontend Integration

LangSmith dashboard is embedded in the UI via:
1. Backend proxy endpoint `GET /api/observability/traces/{task_id}`
2. Frontend `ObservabilityView` renders trace data as a timeline
3. Optional: iframe embed of LangSmith public share links

---

## 7. API Layer (Extended)

### 7.1 New Endpoints (Phase 2)

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/tasks/{id}/agents` | Live agent hierarchy tree |
| `GET` | `/api/tasks/{id}/traces` | LangSmith trace summary for a task |
| `GET` | `/api/memory/stats` | Memory layer statistics |
| `GET` | `/api/observability/runs` | Recent LangSmith runs with costs |
| `GET` | `/api/observability/runs/{run_id}` | Single run detail |

### 7.2 New WebSocket Events (Phase 2)

```jsonc
// Agent spawned during dynamic hierarchy building
{"type": "agent_spawned", "agent": { "agent_id": "...", "name": "...", "role": "...", "model": "..." }}

// Real A2A message between agents
{"type": "agent_message", "message": { "from": "PM-Main", "to": "TL-Alpha", "type": "TaskAssignment", ... }}

// Live metrics update
{"type": "metrics_update", "metrics": { "total_tokens": 1234, "total_cost": 0.0042, "agents_active": 5 }}
```

---

## 8. File Structure (Target)

```
src/aifactory/
├── core/
│   ├── base_agent.py           # + LangSmith @traceable decorator
│   ├── memory_router.py        # REWRITE → LlamaIndex VectorStoreIndex wrapper
│   ├── schemas.py              # + AssessmentResult, HierarchySpec, TraceSummary
│   └── observability.py        # NEW — LangSmith setup, trace helpers, cost aggregation
├── agents/
│   ├── worker.py               # Unchanged
│   ├── team_lead.py            # Minor: accept dynamic worker count
│   ├── project_manager.py      # Minor: accept dynamic team specs
│   ├── master_researcher.py    # NEW — deep domain research agent
│   ├── assessor.py             # NEW — feasibility/scope/risk evaluator
│   ├── devils_advocate.py      # NEW — challenges assumptions
│   └── hierarchy_builder.py    # NEW — outputs dynamic team spec
├── graphs/
│   ├── single_team.py          # Keep as fallback for simple tasks
│   ├── assessment.py           # NEW — assessment team subgraph
│   └── dynamic_pipeline.py     # NEW — main entry: assess → build → execute
├── api/
│   └── main.py                 # + /api/observability/* endpoints
├── config/
│   └── settings.py             # + LANGCHAIN_* settings, embedding config
├── prompts/
│   ├── master_researcher.md    # NEW
│   ├── assessor.md             # Exists
│   ├── devils_advocate.md      # NEW
│   └── hierarchy_builder.md    # NEW
└── cli.py                      # + `aifactory traces` command
```

---

## 9. Deployment Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Vercel         │     │   Render          │     │   Qdrant Cloud  │
│   (Frontend)     │────▶│   (Backend)       │────▶│   (Vector DB)   │
│                  │     │                   │     │                 │
│   Next.js 16     │     │   FastAPI         │     │   3 collections │
│   React 19       │     │   LangGraph       │     │   agent/team/   │
│   Tailwind 4     │     │   LiteLLM         │     │   global        │
│                  │     │   LlamaIndex      │     │                 │
└─────────────────┘     └────────┬──────────┘     └─────────────────┘
                                 │
                        ┌────────▼──────────┐
                        │   LangSmith       │
                        │   (Observability)  │
                        │                   │
                        │   Traces, costs,  │
                        │   evaluation      │
                        └───────────────────┘
```
