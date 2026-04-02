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

## Decision 6: 1:3 Hierarchy Ratio

**Date:** 27 March 2026
**Decision:** Max 3 subordinates per lead (TL has max 3 workers, PM has max 3 TLs).
**Rationale:**
- Prevents context explosion (each lead only manages 3 summaries, not 10+)
- Natural decomposition boundary — most tasks break into 2-3 sub-tasks
- Configurable per project via Settings (can be adjusted by RL sidecar in Phase 3)

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
