# Architecture

## The Idea (Original Brief)

Build a **self-assembling, self-optimising "AI Factory"** that takes a raw human idea and produces production-grade output with almost zero manual coordination.

### Problem it Solves

Current AI workflows (Copilot, Claude, ChatGPT) suffer from:
- **Serial bottlenecks** — one agent doing everything sequentially
- **Context fragmentation** — context lost between sessions/tools
- **User as project manager** — human has to coordinate, split tasks, merge results manually

### Solution: Dynamic Multi-Agent Hierarchy

```
Human Idea
   ↓
Master Agent (Auto-Researcher)
   → builds docs, market analysis, tech landscape, risks, opportunity map
   ↓
Project Assessment Team (2 agents + 1 Devil's Advocate)
   → evaluates research, designs hierarchy, chooses architecture type
   ↓
Project Manager(s)  [1:3 ratio with Team Leads]
   ↓
   ├─ Team Lead(s)   [1:3 ratio with Workers]
   │     └─ Worker Agents (coder, researcher, writer, tester, etc.)
   │
   └─ (when >2 PMs) → Sr. Project Manager + Validator agents
         ↓
Consolidation → Master Agent (final delivery + human handoff)
```

## Key Architecture Differentiators

### 1. Dynamic Runtime Hierarchy (Not Pre-defined Crews)
Unlike CrewAI/AutoGen which use static agent definitions, our hierarchy is **constructed at runtime** by the Assessment Team based on the actual task complexity. Simple task? 1 PM, 1 TL, 2 workers. Complex multi-domain project? 3 PMs, 9 teams, 27 workers.

### 2. Three-Layer Memory System

```
Agent Layer (Qdrant, per agent_id)
  → Private STM + episodic memory (raw conversation, tool calls, thoughts)
  → Promotes every N turns via summarisation

Team Layer (Qdrant, per team_id)
  → Shared working memory, contracts, decisions, intermediate artifacts
  → What a Team Lead uses to keep workers in sync

Global LTM Layer (Qdrant → Milvus at scale)
  → Stable patterns, reusable components, cross-project learnings
  → What Master Agent / Sr. PMs query for historical context
```

**Memory Router** — every agent calls `store()` / `retrieve()` through a unified router that handles layer selection, promotion, deduplication, and cross-layer search.

### 3. Structured A2A Communication
All inter-agent messages use **Pydantic schemas** (not free-form text):
- 5-10x fewer tokens per message vs chatty communication
- Zero parsing errors
- Enforced contracts between agents

Message types: `TaskAssignment`, `TaskResult`, `StatusUpdate`, `Question`, `Answer`, `ReviewRequest`, `ReviewResult`, `Escalation`

### 4. RL Sidecar + Agents-as-Judge
Every agent is scored after each task on 6 metrics:
- Correctness (30%)
- Completeness (20%)
- Edge Case Handling (15%)
- Communication Quality (10%)
- Efficiency (15%)
- Peer Judge Scores (10%)

These scores feed into model routing (which LLM to use per agent) and memory promotion (only high-quality learnings reach Global LTM).

### 5. Model Routing by Role
| Role | Tier | Default Model |
|------|------|---------------|
| Master Researcher, PMs, Assessment | Premium | Claude 3.7 Sonnet |
| Team Leads | Standard | Gemini 2.5 Flash |
| Workers | Budget | Gemini 2.5 Flash |

## Tech Stack

- **Python 3.12+** — project managed with `uv` + `pyproject.toml`
- **LangGraph** — stateful graph orchestration for agent pipelines
- **LiteLLM** — unified LLM interface (switch providers with a string)
- **Pydantic v2** — structured schemas for all communication
- **Qdrant** — vector DB for 3-layer memory (in-memory mode for dev)
- **Rich + Typer** — CLI interface
- **pytest** — testing

## Token & Cost Optimisation Strategy

| Technique | Savings | Status |
|-----------|---------|--------|
| Strict Pydantic JSON schemas for A2A | 5-8x | ✅ Implemented |
| Hierarchical summarisation gates | 3-5x | ✅ Implemented |
| Model routing (cheap for workers) | 40-70% cost | ✅ Implemented |
| Prompt caching | 70-90% on repeated calls | 🔜 Phase 3 |
| Context compression | 2-4x | 🔜 Phase 4 |
| Semantic deduplication in Memory Router | 30-50% | ✅ Implemented |

## DB Architecture Roadmap

```
Phase 1 – MVP (current)
├── Qdrant (single cluster, in-memory for dev)
│   ├── agent_id namespace → STM + raw episodic (TTL)
│   ├── team_id namespace  → shared working mem + contracts
│   └── global namespace   → curated LTM patterns
└── (Postgres + Redis planned for Phase 2+)

Phase 2 – Scale (after 10-20 real projects)
├── Qdrant stays for Agent + Team
├── Milvus for global LTM (cross-project search)
├── Postgres for task graph state, agent registry
└── Redis for pub/sub promotion events + cache
```
