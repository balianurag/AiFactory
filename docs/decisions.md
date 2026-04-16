# Technical Decisions Log

## Decision 1: Python 3.12+ with uv

**Date:** 27 March 2026
**Decision:** Use Python 3.12+ managed by `uv` (not pip/poetry/conda).
**Rationale:**
- `uv` is 10-100x faster than pip for dependency resolution and installation
- Native `pyproject.toml` support, no extra config files
- Lock file (`uv.lock`) for reproducible builds
- Built-in venv management

## Decision 2: LangGraph for Orchestration

**Date:** 27 March 2026
**Decision:** Use LangGraph (not CrewAI, AutoGen, or custom orchestration).
**Rationale:**
- Stateful graph with typed state — perfect for our dynamic hierarchy
- Native async support
- Built-in checkpointing (can resume failed runs)
- Active development by LangChain team, strong 2026 ecosystem
- Supports dynamic graph construction at runtime

**Trade-off:** Slightly more boilerplate than CrewAI, but much more flexible for our dynamic hierarchy pattern.

## Decision 3: LiteLLM for Model Abstraction

**Date:** 27 March 2026
**Decision:** Use LiteLLM (not direct OpenAI/Anthropic/Google SDKs).
**Rationale:**
- Unified interface across 100+ LLM providers with a single string change
- Built-in retry, fallback, and caching support
- Essential for our model routing strategy (premium/standard/budget tiers)
- Cost tracking per call

## Decision 4: Qdrant for All Memory Layers (MVP)

**Date:** 27 March 2026
**Decision:** Use Qdrant for all 3 memory layers initially, with in-memory mode for development.
**Rationale:**
- In-memory mode means zero infrastructure to get started (no Docker)
- Sub-10ms writes, fast enough for agent-level memory
- Collection + payload filters provide namespace isolation without multiple DBs
- Migration path: move Global LTM to Milvus when cross-project search dominates (Phase 2+)

**Trade-off:** In-memory mode loses data on restart. Acceptable for MVP; production will use persistent Qdrant.

## Decision 5: Pydantic v2 for All A2A Communication

**Date:** 27 March 2026
**Decision:** Every inter-agent message must be a Pydantic model. No free-form text between agents.
**Rationale:**
- 5-10x token savings vs chatty natural language
- Zero parsing errors — structured JSON enforced at the schema level
- Self-documenting contracts between agent roles
- Works with LiteLLM's `response_format` for structured output from LLMs

## Decision 6: 1:3 Hierarchy Ratio ~~(Superseded)~~

**Date:** 27 March 2026
**Superseded:** ~6 April 2026

This decision assumed a PM→TL→Worker hierarchy where a HierarchyBuilder would size the teams. That architecture was redesigned before implementation.

**What replaced it:** The `execution_lead` node in the 8-node pipeline spawns workers dynamically based on the task count returned by the `analyse` node. There are no management layers (no PM, no TL) — just a flat Execution Lead → N Workers structure. Worker count is determined by task complexity, not by a ratio formula. Real runs show 2–5 workers per task depending on analyser output.

## Decision 7: Stub Embeddings for Dev, Real Embeddings for Prod

**Date:** 27 March 2026
**Decision:** Memory Router uses a deterministic hash-based pseudo-embedding for offline development. Real embedding model plugged in for production.
**Rationale:**
- Allows full memory system testing without API calls or embedding model
- Deduplication and basic retrieval work with hash-based approach
- Semantic similarity only works properly with real embeddings (production)

## Decision 8: LangSmith for Observability

**Date:** 2 April 2026
**Decision:** Use LangSmith (not Langfuse, custom logging, or OpenTelemetry alone) for LLM observability.
**Rationale:**
- First-party integration with LangGraph — automatic trace trees for graph nodes
- Zero-cost when disabled — `@traceable` decorator is a passthrough without API key
- Cost attribution per LLM call (critical for our budget tracking)
- Evaluation datasets for regression testing agent quality (Phase 3)
- Single dashboard for: traces, costs, latencies, errors, prompt debugging

**Trade-off:** Vendor lock-in to LangChain ecosystem. Acceptable because we already use LangGraph. Can layer OpenTelemetry for infrastructure-level traces later.

## Decision 9: LangGraph + LlamaIndex + LangSmith Stack

**Date:** 2 April 2026
**Decision:** Use three complementary tools — LangGraph (orchestration), LlamaIndex (memory/RAG), LangSmith (observability). Not a single monolithic framework.
**Rationale:**
- LangGraph: only framework with runtime dynamic subgraph compilation and checkpointing
- LlamaIndex: best-in-class document indexing and semantic retrieval over Qdrant
- LangSmith: best observability for LangGraph workloads
- Each tool does one thing well — no compromises trying to make one tool do everything
- LiteLLM stays as the LLM call layer (unchanged from Phase 1)

## Decision 10: Venv Layout

**Date:** 2 April 2026
**Decision:** Single `aifactory_venv/` directory parallel to `AiFactory/` and `AiFactory-ui/`, managed by `uv`.
**Rationale:**
- Shared venv for all Python work (backend, scripts, notebooks)
- `uv` for all package management (10-100x faster than pip)
- Parallel directory avoids polluting the git repo with venv artifacts

## Decision 11: Config-Driven Agents (YAML over Subclasses)

**Date:** ~6 April 2026
**Decision:** All agent roles defined in YAML (`config/agents/*.yaml`). Single `PipelineAgent` class handles every role — master, judge, analyser, researcher, execution_lead, worker, and all specialist variants.
**Rationale:**
- Adding a new agent role requires zero Python changes
- Role behaviour (system prompt, abilities, model constraints, temperature) is readable config, not code
- Dramatically reduces class hierarchy and inheritance complexity
- Makes agent experiments cheap: edit YAML, re-run, no deploy

## Decision 12: Flat Execution Structure (No PM/TL Layers)

**Date:** ~6 April 2026
**Decision:** The production pipeline uses Execution Lead → N Workers directly. No Project Manager or Team Lead layers.
**Rationale:**
- Management layers add latency and tokens without improving output quality
- Task decomposition (what the analyser does) already handles the "what to delegate" problem
- Workers are independently capable with specialist roles — they don't need leads
- Real runs confirm: 2–5 workers per task is the natural decomposition range, which a single Execution Lead handles cleanly

**Trade-off:** For genuinely large tasks (>10 sub-tasks), a single Execution Lead managing many workers could become a bottleneck. Accepted for now — this is Phase 5 territory.

## Decision 13: ComplexityGate Before Pipeline Entry

**Date:** ~8 April 2026
**Decision:** A zero-LLM routing layer (`core/complexity_gate.py`) sits in front of both `aifactory new` and `aifactory run`. Pure math, unit conversions, and static facts are answered without any LLM call.
**Rationale:**
- "What is 2+2?" should not consume 1000+ tokens across 8 nodes
- AST-based math eval is deterministic and safe (no `eval()` with arbitrary expressions)
- Static facts (capitals, planet counts, etc.) are small enough to embed in code
- `fast_llm` tier (single master call) handles simple Q&A at ~1/10 the cost of the full pipeline

## Decision 14: SHARE: Protocol for Inter-Agent Context

**Date:** ~7 April 2026
**Decision:** Agents broadcast discovered context by appending `SHARE: key: value` lines to their output. These are extracted and merged into `shared_context`, which all downstream agents can read.
**Rationale:**
- Lighter than passing full message objects between agents
- Naturally fits LLM text output — no extra tool calls needed
- Phase 2.9: SHARE: updates are also persisted to team-layer memory for cross-run retrieval

## Decision 15: Docker MCP Sandbox as Primary Code Execution Backend

**Date:** ~9 April 2026
**Decision:** Priority order for code execution: Docker MCP (warm pool) > E2B cloud > local subprocess.
**Rationale:**
- Warm pool (4–16 pre-warmed containers) gives 10–50ms execution vs 1–2s cold start
- Full isolation: `--network=none --cap-drop=ALL --read-only --tmpfs /workspace`
- MCP protocol means the sandbox is a proper tool the agent calls, not a subprocess hack
- E2B fallback for cloud environments where Docker is unavailable
- Local subprocess for CI and dev with no Docker

## Decision 16: RL Signal as Multiplicative Bias (Not Replacement)

**Date:** ~10 April 2026
**Decision:** RL scores adjust model selection by a multiplicative factor (±20% max) rather than overriding static ability scores.
**Rationale:**
- Static ability scores are carefully hand-tuned; RL data from a handful of runs shouldn't fully override them
- Multiplicative factor means a model with 3 RL records can't displace a well-scored model that has 0 records yet
- `AIFACTORY_RL_MIN_SAMPLES=3` ensures adjustment only fires when statistically meaningful
- Keeps selection predictable and debuggable

## Decision 17: GitLab CI (Not GitHub Actions)

**Date:** April 2026
**Decision:** CI/CD runs on GitLab CI (`.gitlab-ci.yml`). The repository is hosted on GitLab.
**Rationale:** Repository is on GitLab. GitHub Actions workflows are not used.
**Stages:** test (pytest + coverage + smoke) → build (uv build).
