# CLAUDE.md

> **Keep this file updated** after every significant change. It is read by Claude Code at session start and directly shapes response quality, token efficiency, and agentic team behaviour.

---

## Claude Working Rules

These rules govern how Claude Code should operate in this repository. They are not suggestions — apply them on every task.

### 1. Token Efficiency

- **Read before writing.** Always grep or read the relevant file section before editing. Never load a 500-line file to change 3 lines.
- **Grep first.** Use `grep -n` to locate the exact line, then `Read` with `offset`+`limit` to pull only that section.
- **No redundant reads.** If a file was read this session and unchanged, do not re-read it to verify an edit — the Edit tool guarantees the write succeeded.
- **Batch parallel tool calls.** When multiple files need reading or multiple independent commands need running, issue them in one response block — never sequentially when they are independent.
- **Skip broad searches when scope is known.** If the user references a specific file or symbol, go there directly. Only spawn an Explore agent when scope is genuinely unknown.
- **Summarise, don't echo.** Do not repeat file contents back to the user. State the finding in one sentence and move on.
- **No filler updates.** Avoid "Let me check…", "Now I'll…", "As you can see…". State what was found or changed, nothing else.

### 2. Code Changes

- **Smallest correct change.** Fix the bug, add the feature — nothing else. No surrounding cleanup, no refactors, no future-proofing unless asked.
- **No comments by default.** Only add a comment when the WHY is non-obvious: a hidden constraint, a Zscaler workaround, a subtle invariant. Never describe what the code does.
- **No backwards-compat shims.** If something is unused, delete it. No `# removed`, no re-exports for dead code.
- **Trust the framework.** LangGraph, LiteLLM, Pydantic v2 — don't add defensive checks for things the framework guarantees. Validate only at system boundaries.
- **Async-native.** All new async code uses `async/await`. No `asyncio.run()` inside an async context. No blocking I/O in async paths.
- **Never touch `.env` values without asking** unless the task explicitly requires it. API keys and model overrides are live credentials.

### 3. Agentic Team Design

When designing or modifying agent workflows in this codebase, follow these principles:

**Model assignment by task type:**
| Task type | Proxy model to use | Why |
|---|---|---|
| Reasoning / planning (master, analyser) | `us.anthropic.claude-sonnet-4-20250514-v1:0` | Best instruction-following + structured output |
| Research / synthesis (researcher, writer workers) | `gemini-2.5-flash` | Fast, large context, good at summarisation |
| Coding / math (coder workers) | `gpt-4.1` or `gpt-4o` | Strong code generation, reliable tool use |
| Cheap routing / classification (gate, judge) | `gemini-2.0-flash` or `gpt-4.1-mini` | Near-zero cost, adequate for classification |
| Deep eval / quality judge | `us.anthropic.claude-sonnet-4-20250514-v1:0` | Calibrated scoring, follows rubrics precisely |

**Worker count discipline:**
- Analyser should produce **2–5 tasks** for most requests. More than 5 tasks is a smell — merge related work.
- Single-worker runs should always go through the `fast_llm` gate tier, not the full pipeline.
- Never spawn a worker for a task that takes <50 tokens to answer — the gate should catch these.

**Context passing rules:**
- Use `SHARE: key: value` for facts that downstream agents need. Keep values under 200 chars.
- Use `FINDINGS:` blocks (not full prose) when passing results between waves — 80-90% token reduction.
- Use `REMEMBER:` only for facts that should survive across runs. Don't pollute global memory with ephemeral task state.
- Never inject the full prior-task output into the next worker's prompt. Use `FINDINGS:` summaries.

**Self-correction discipline:**
- The REVISION loop runs once. If a worker emits REVISION: on its first check, that's the final answer — no third pass.
- Workers must emit `VALIDATED` or `REVISION:` — never a vague "I've reviewed my work" paragraph.

### 4. Memory Layer Usage

- **Agent layer** — per-run scratch memory. Cleared between runs. Use for intra-run coordination.
- **Team layer** — per-run shared context promoted from agent layer at N-turn intervals. Use for cross-agent facts within a run.
- **Global layer** — persistent across runs. Use only for high-value, stable facts (user preferences, domain knowledge, project constants). Never write ephemeral state here.
- **FactsStore** (`~/.aifactory/memories/facts.json`) — survives Qdrant restarts. For `REMEMBER:` facts that must outlast infrastructure.
- **ResearchCache** (`~/.aifactory/memories/research.json`) — saves researcher LLM calls. Check it before running researcher on a query that was researched recently.

### 5. Complexity Gate Discipline

Before adding a new gate pattern to `config/gate_patterns.yaml`, run `aifactory gate "<query>"` to verify the current routing. Add patterns only when a real query class is being routed to the wrong tier. Never add patterns to force expensive queries through cheap tiers.

Gate tier decision guide:
- `bypass` — pure arithmetic, no LLM needed
- `static_facts` — dictionary lookup, no LLM needed
- `fast_llm` — single master call, no workers
- `research_cached` — fast_llm + injected cached research
- `full` — 8-node multi-agent pipeline

If unsure, let it go to `full`. Over-routing to `fast_llm` produces shallow answers.

### 6. Environment — Expedia Internal Dev

This project runs inside the Expedia corporate network during development. When shipping externally, these settings are removed.

**SSL:** All outbound HTTPS goes through Zscaler. The combined CA bundle lives at `certs/combined-ca-bundle.pem` (gitignored). Regenerate with `make certs` after a macOS keychain update or machine rebuild.

**LiteLLM proxy:** `https://genai-proxy.rcp.us-east-1.data.test.exp-aws.net` — internal model gateway giving access to Bedrock, Azure OpenAI, Vertex AI, and more. Set `LITELLM_API_BASE` + `LITELLM_API_KEY` in `.env`. When these are set, all LiteLLM calls are prefixed with `openai/` and routed through the proxy automatically.

**External shipping checklist** (when productising for external customers):
- [ ] Remove `LITELLM_API_BASE` from `.env` — app falls back to direct provider keys
- [ ] Remove `SSL_CERT_FILE` from `.env`
- [ ] Remove `certs/` directory from deployment artefact
- [ ] Rotate `LITELLM_API_KEY` — it stays Expedia-internal
- [ ] Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` for the external deployment

### 7. Git Identity

Personal projects (this repo and all repos under `/Users/anbali/Desktop/personal/`) must use personal git identity:
```
git config user.email "anuragbali10596@gmail.com"
git config user.name "Anurag Bali"
```
The global git config is set to the Expedia corporate email. Always verify repo-level identity before committing. Push personal repos to `github` remote (`balianurag`), not `origin` (GitLab).

### 8. After Every Task

1. Update this file and `SPEC.md` if architecture, commands, or design decisions changed.
2. Run `uv run aifactory info` with the proxy env vars to verify the system still starts cleanly.
3. If model registry or gate patterns changed, run one smoke test: `uv run aifactory run "what is 2+2" --mode cost_efficient` — should hit bypass or fast_llm tier in under 2 seconds.

---

## Project Overview

**AI Factory** — A self-assembling, self-optimising multi-agent system that turns a raw human idea into production-grade output with minimal manual coordination.

### Architecture

```
Human Idea → Master (coordinator)
           → Judge (briefs quality criteria)
           → Analyser (task breakdown / direct-answer bypass)
           → [Researcher → Re-analyse]?   (optional pre-analysis research loop)
           → [Researcher]?                 (optional execution research)
           → Execution Lead (spawns N Workers dynamically, per-task model selection)
           → Judge (reviews pipeline run quality)
           → Master (synthesises final response)
```

Key features:
- **Config-driven agents** — all roles defined in YAML (`config/agents/*.yaml`)
- **10-ability model scoring** — each model rated on reasoning, coding, math, writing, etc. (`config/model_registry.yaml`)
- **3 mode profiles** — `best`, `balanced`, `cost_efficient` (alpha-blended quality vs cost)
- **LiteLLM** — cross-provider LLM abstraction (OpenAI, Anthropic, Bedrock, etc.)
- **SHARE: protocol** — lightweight inter-agent context broadcasting
- **Dynamic worker spawning** — Execution Lead creates per-task workers with ability-scored models
- **Direct-answer bypass** — trivial queries skip workers entirely (Analyser detects)
- **Docker MCP sandbox** — warm Docker container pool via MCP (10-50ms, gVisor+seccomp+cap-drop security). Falls back to E2B or local subprocess automatically
- **Package registry** — `config/package_registry.yaml` injected into worker prompts so LLMs never waste tokens on `pip install`
- **DSA-optimised coding** — workers use optimal algorithms, state Big-O complexity, prefer stdlib tools
- **3-layer memory**: Agent → Team → Global LTM (all via Qdrant)
- **Human-in-the-loop** — LangGraph `interrupt()` for worker QUESTION: prompts
- **MemorySaver checkpointing** — graph state persisted for resume after interrupts

### Tech Stack

- **Python 3.13+**, managed with `uv` + `pyproject.toml` (hatchling build)
- **LangGraph** — stateful graph orchestration
- **LlamaIndex** — memory management and retrieval
- **LiteLLM** — unified LLM interface (any provider)
- **Pydantic v2** — structured schemas for all A2A communication
- **Qdrant** — vector DB for 3-layer memory (in-memory mode for local dev)
- **Rich + Typer** — CLI interface
- **pytest** — testing

### Commands

All dev and app commands are available via `manage.py` at the repo root:

```bash
# One-time environment setup (Python check, uv, venv, deps, pre-commit)
python manage.py setup

# Dev tooling
python manage.py install          # uv sync --dev
python manage.py dev              # FastAPI dev server with auto-reload (port 8000)
python manage.py test             # pytest tests/
python manage.py test -k "name"   # Filter by test name
python manage.py test --cov       # With HTML coverage report
python manage.py lint             # ruff check src/ tests/
python manage.py lint --fix       # Auto-fix violations
python manage.py format           # ruff format src/ tests/
python manage.py format --check   # CI mode (no writes)
python manage.py typecheck        # mypy src/
python manage.py build            # uv build → dist/

# Application commands (delegates to aifactory CLI)
python manage.py run "task"                     # Run the agent pipeline
python manage.py run --task-file task.md        # Task from file
python manage.py info                           # Show config + API key status
python manage.py memory-stats                   # Memory layer stats
python manage.py models list                    # All roles + allowed models
python manage.py models list --role coder       # Filter by role
python manage.py models tiers                   # Premium/standard/budget assignments
python manage.py agents list                    # All roles, tiers, default models

# Memory commands (Phase 2.9 / 5.0)
uv run aifactory memory health                  # ★ Check Qdrant + embedding provider + Redis + collections
uv run aifactory memory stats                   # Layer counts (agent/team/global)
uv run aifactory memory search "FastAPI auth"   # Semantic search across memory
uv run aifactory memory search "query" --layer team --agent analyser
uv run aifactory memory clear --layer agent --yes   # Clear agent layer
uv run aifactory memory clear --agent Worker-task_1 # Clear specific agent

# API server (Phase 5.0)
uv run aifactory serve                          # Start persistent API server (localhost:8080)
uv run aifactory serve --port 9000              # Custom port
uv run aifactory serve --workers 4             # Production (multi-process)
uv run aifactory serve --reload --log-level debug  # Hot-reload dev mode

# RL-lite adaptive model scoring commands (Phase 3)
uv run aifactory rl stats                       # Show model performance by task type
uv run aifactory rl stats --type coding         # Filter to a single task type
uv run aifactory rl reset --yes                 # Clear all RL records (fresh start)
uv run aifactory rl reset --type coding         # Clear records for one task type

# Docker MCP Sandbox commands (Phase 4.0)
uv run aifactory sandbox build                  # Build aifactory-sandbox + aifactory-mcp-server images
uv run aifactory sandbox build --no-cache       # Force full rebuild
uv run aifactory sandbox pull                   # Pull base images (python:3.12-slim, socket-proxy)
uv run aifactory sandbox status                 # Pool health metrics (ready/in_use/total)
uv run aifactory sandbox status --url http://localhost:8100  # Override MCP server URL
uv run aifactory sandbox test                   # Smoke test: Python + numpy/pandas in container
uv run aifactory sandbox test --language bash   # Test bash execution
uv run aifactory sandbox test --language javascript  # Test Node.js execution
```

Raw `uv run aifactory <cmd>` also works for all application commands.

### Project Structure

```
src/aifactory/
├── core/
│   ├── base_agent.py           # Base agent with LiteLLM, memory, token tracking, @traceable
│   ├── assessor.py             # Task complexity classifier (trivial/moderate/complex)
│   ├── model_selector.py       # 10-ability dot-product model scoring engine
│   ├── agent_factory.py        # YAML agent loader, SHARE: protocol, AgentRegistry, pkg registry
│   ├── schemas.py              # ALL Pydantic models (A2A messages, configs, RL)
│   ├── memory_router.py        # 3-layer Qdrant memory (store/retrieve/promote)
│   ├── memory_bridge.py        # Fail-safe adapter between PipelineAgent + MemoryRouter
│   ├── facts_store.py          # ★ JSON-backed REMEMBER: fact persistence (survives Qdrant restarts)
│   ├── research_cache.py       # ★ Persistent researcher-output cache (exact/cosine/word-overlap lookup)
│   ├── semantic_classifier.py  # ★ Cosine-similarity intent classifier for gate ambiguous queries
│   ├── output_validator.py     # Worker output validation pre-consolidation
│   ├── rate_limiter.py         # slowapi rate limiting (4 tiers, Redis-backed in prod)
│   ├── database.py             # PostgreSQL asyncio adapter (used by API layer)
│   ├── redis_client.py         # Redis async client wrapper (used by API layer)
│   ├── sandbox.py              # ★ 3-backend code execution (docker > e2b > local)
│   ├── sandbox_pool.py         # ★ Async warm Docker container pool (Phase 4.0)
│   ├── rl_scorer.py            # ★ RL-lite adaptive model scoring (Phase 3)
│   ├── model_calibrator.py     # ★ RL-driven ability score patching for model_registry.yaml
│   ├── run_store.py            # ★ Pipeline run persistence (Qdrant + JSONL fallback)
│   ├── embed_provider.py       # ★ Auto-select embedding provider + Redis embedding cache (Phase 5.0)
│   ├── semantic_cache.py       # ★ Qdrant-backed semantic query deduplication (Phase 5.0)
│   ├── budget_tracker.py       # Per-run USD hard cap enforcement
│   ├── retry.py                # Exponential backoff + circuit breaker (per-model failure tracking)
│   ├── otel.py                 # ★ OpenTelemetry distributed tracing (span(), @trace_async, @trace_sync)
│   └── observability.py        # LangSmith setup, @traceable decorator, run fetchers
├── agents/
│   ├── code_security_processor.py  # ★ CODE_ANALYSIS: signal interception — prevents local code leakage to LLM APIs
│   ├── worker.py               # Legacy worker — kept for backward compat
│   ├── team_lead.py            # Legacy team lead — kept for backward compat
│   └── project_manager.py      # Legacy PM — kept for backward compat
├── graphs/
│   ├── dynamic_pipeline.py     # ★ 8-node LangGraph: the new config-driven pipeline
│   └── single_team.py          # Legacy 4-node pipeline — kept for backward compat
├── api/
│   └── main.py                 # FastAPI app: /api/health, /api/models, /api/tasks, /api/observability/*, WS /api/ws/{id}
├── prompts/                    # System prompts per role (.md files)
├── mcp/                        # ★ MCP sandbox integration (Phase 4.0)
│   ├── __init__.py
│   ├── sandbox_server.py       #   FastMCP server: execute_python/bash/js + pool_status tools
│   └── client.py               #   MCP client factory (HTTP mode or in-process stdio)
├── config/
│   ├── settings.py             # Pydantic Settings (env vars + .env, incl. LangSmith config)
│   ├── models.yaml             # Legacy model registry for /api/models UI endpoint
│   ├── model_registry.yaml     # ★ ~20+ models with 10-ability scores + pricing
│   ├── graph.yaml              # ★ Graph runtime config (mode, fallback model)
│   ├── gate_patterns.yaml      # ★ ComplexityGate YAML pattern library
│   ├── tool_catalog.yaml       # Integration tool catalog
│   ├── package_registry.yaml   # ★ Pre-installed sandbox packages (injected into worker prompts)
│   └── agents/                 # ★ YAML configs per agent role:
│       ├── master.yaml         #   - Coordinator / synthesiser
│       ├── judge.yaml          #   - Quality reviewer
│       ├── analyser.yaml       #   - Task breakdown / direct-answer detection
│       ├── researcher.yaml     #   - Pre-analysis and execution research
│       ├── execution_lead.yaml #   - Worker dispatcher + consolidator
│       ├── worker.yaml         #   - Template for per-task workers
│       ├── worker_coder.yaml   # ★ Specialist: software engineer (coding/math tasks)
│       ├── worker_researcher.yaml # ★ Specialist: research & synthesis (research tasks)
│       ├── worker_writer.yaml  # ★ Specialist: writer & communicator (writing tasks)
│       ├── worker_analyst.yaml # ★ Specialist: analyst & reasoner (analysis tasks)
│       └── meta.yaml           # ★ Meta-agent for self-improvement reviews
└── cli/                        # ★ Modular Typer CLI
    ├── __init__.py             #   - App root + command registration
    ├── _helpers.py             #   - Shared console, dump loading, formatting
    ├── cmd_run.py              #   - `aifactory run` / `aifactory resume` (complexity gate → fast/full path)
    ├── cmd_status.py           #   - `aifactory status` / `aifactory inspect` / `aifactory gate`
    ├── cmd_connect.py          #   - `aifactory connect` / `aifactory tools`
    ├── cmd_ask.py              #   - `aifactory ask`
    ├── cmd_agents.py           #   - `aifactory agents list`
    ├── cmd_cost.py             #   - `aifactory cost`
    ├── cmd_info.py             #   - `aifactory info` / `aifactory models`
    ├── cmd_memory.py           #   ★ `aifactory memory stats/search/clear/health` — also queries FactsStore
    ├── cmd_sandbox.py          #   ★ `aifactory sandbox build/status/test/pull` (Phase 4.0)
    ├── cmd_rl.py               #   ★ `aifactory rl stats/reset` (Phase 3)
    ├── cmd_batch.py            #   ★ `aifactory batch run/status` — bulk JSONL pipeline execution
    ├── cmd_runs.py             #   ★ `aifactory runs list/show/search` — run history browser
    ├── cmd_meta.py             #   ★ `aifactory meta review/calibrate` — self-improvement + model calibration
    └── cmd_serve.py            #   ★ `aifactory serve` — persistent API server with pre-warming (Phase 5.0)
```

### Implementation Status

- [x] Phase 0: Foundation (schemas, base agent, memory router, settings)
- [x] Phase 1: Single-team pipeline (PM → TL → Workers graph) — legacy, still in codebase
- [x] Phase 1.5: HTTP/WebSocket API (`/api/tasks`, `/api/models`, WS streaming) — deployed on Render
- [x] Phase 1.5: Curated model registry (`models.yaml`) — UI model picker endpoint
- [x] Phase 1.5: CLI model/agent commands (`models list`, `models tiers`, `agents list`)
- [x] Phase 2.0: LangSmith observability — `@traceable` on BaseAgent.think(), API endpoints, UI Traces view
- [x] Phase 2.1: LiteLLM migration — replaced OpenAI SDK with `litellm.acompletion()` everywhere
- [x] Phase 2.2: Ability-based model selection (`model_selector.py` + `model_registry.yaml`)
- [x] Phase 2.3: YAML config-driven agents (`agent_factory.py` + `config/agents/*.yaml`)
- [x] Phase 2.4: 8-node dynamic pipeline (`dynamic_pipeline.py`) — the production architecture
- [x] Phase 2.5: Pipeline wired to API (`_run_pipeline()` uses `build_dynamic_pipeline()`)
- [x] Phase 2.5: API rate limiting (slowapi, 4 tiers)
- [x] Phase 2.5: Code execution sandbox (local subprocess)
- [x] Phase 2.5: Worker self-correction loop
- [x] Phase 2.6: E2B cloud sandbox integration — dual-backend code execution for workers
- [x] Phase 2.7: Code optimisation flow — DSA excellence, concise prompts, task merging, isolated sandbox awareness
- [x] Phase 2.8: ComplexityGate v2 — 4-tier routing (bypass math, static facts, fast_llm, full pipeline), YAML-driven patterns
- [x] Phase 2.9: Memory Bridge (Phase 2.1) — `MemoryBridge` wired into every `PipelineAgent`; retrieval before think(), storage after; SHARE: updates persisted to team layer; promotion at N-turn threshold; CLI `memory stats/search/clear`
- [x] Phase 3.5: Token Efficiency — Anthropic prompt caching (direct + OpenRouter Claude, `cache_control: ephemeral`); LiteLLM in-memory response cache for ALL providers/tiers; smart judge (skip LLM for single-task runs, ~1000 token saving); shared context pruning (cap at 25 keys); 3 new settings
- [x] Phase 4.0: Docker MCP Sandbox — warm container pool (`sandbox_pool.py`), FastMCP server (`mcp/sandbox_server.py`), auto-detect client (`mcp/client.py`), `sandbox.py` updated (docker > e2b > local priority), package registry injected into worker prompts, CLI `aifactory sandbox build/status/test/pull`; 62 new tests
- [x] Phase 3: RL-lite scoring — `core/rl_scorer.py`; `(model, task_type) → avg_quality` stored in Qdrant `aifactory_rl_performance`; `select_model()` blends static ability scores + RL signal (±20% multiplicative at default rl_weight=0.2); judge_review records per-worker performance after each run; `create_worker_config()` fetches RL scores before model selection; CLI `aifactory rl stats/reset`; 3 new settings (`AIFACTORY_RL_ENABLED/WEIGHT/MIN_SAMPLES`); 22 new tests
- [x] Phase 4.1: MCP Catalog provider — `MCPProvider` alongside `ComposioProvider`; config-driven server catalog (`config/integrations/mcp_catalog.yaml`); 6 servers: GitHub (official Docker), Atlassian/Jira+Confluence (uvx), Notion (npx), Google Workspace (disabled until token vault), Linear (npx), Slack (disabled); `AsyncExitStack` session management; `server::tool_name` slug format for `ACTION:mcp` blocks; `_build_toolkit_catalog()` prefers MCP for covered toolkits, falls back to Composio for the rest; `AIFACTORY_MCP_CATALOG_ENABLED=false` feature flag; 63 new tests
- [x] Phase 4.2: DEPLOY_MODE routing — `AIFACTORY_DEPLOY_MODE=local|cloud`; `local` uses stdio subprocess (dev); `cloud` maintains async HTTP session pool per MCP server (N persistent sessions, acquire/release via `asyncio.Queue`); `url_env_var` + `pool_size` fields in `mcp_catalog.yaml`; `AIFACTORY_MCP_HTTP_POOL_SIZE` setting; 12 new tests (76 total in `test_mcp_provider.py`)
- [x] Phase 4.3: Parallel worker execution + role specialization + self-correction + dependency enforcement — `asyncio.gather()` wave-based dispatch; 4 specialist YAML configs (`worker_coder/researcher/writer/analyst.yaml`); REVISION: self-check loop; `dependencies_map` extracted from plan JSON; `_build_dependency_waves()` topological sort; 20 new tests
- [x] Phase 5.0: Production hardening — token streaming (`on_token` callback, `_think_stream()`, `AIFACTORY_STREAM_TOKENS_ENABLED`); Redis task store (run lifecycle tracked: running → completed, sorted-set index `aifactory:runs:all`, 7d TTL); semantic cache (`core/semantic_cache.py`, Qdrant collection `aifactory_semantic_cache`, cosine similarity dedup at query-intent level, `AIFACTORY_SEMANTIC_CACHE_*` settings); judge deep eval (post-metrics LLM scoring on relevance/accuracy/completeness/actionability, `AIFACTORY_JUDGE_DEEP_EVAL_ENABLED`); auto-embedding provider (`core/embed_provider.py`, priority: OpenAI → Voyage → Cohere → HuggingFace → stub, Redis embedding cache, clear stub warnings); `aifactory serve` command (`cmd_serve.py`, pre-warms all connections); `aifactory memory health` subcommand; ComplexityGate pattern expansion (7 new `simple_qa` patterns, narrowed research routing); budget warn threshold with `on_warn` callback (80% cap warning); 134 new tests (1024 total)
- [x] Phase 5.1 (partial): Experiment integrations — 5 features ported from experiment branch:
  1. **max_quality cap + _EFFICIENCY_ABILITIES** (`model_selector.py`) — `max_quality` per mode (best=3, balanced=2, cost_efficient=1) caps model quality ability scores so balanced/cost_efficient modes don't over-spend on expensive high-quality models; speed/cost_efficiency are efficiency abilities that bypass the cap;
  2. **FINDINGS: protocol** (`agent_factory.py`, `dynamic_pipeline.py`) — workers append compact `FINDINGS:\n- fact` blocks; extracted per worker and injected as `## Prior Task Findings` into subsequent waves (80-90% token reduction vs full prose prior-results);
  3. **RESEARCH: inline signal** (`dynamic_pipeline.py`) — workers emit `RESEARCH: <query>` mid-task; researcher called inline, result injected back so worker completes without pipeline restart; `researcher` param threaded from `build_dynamic_pipeline` → `make_execution_lead` → `_worker_kwargs`;
  4. **REMEMBER: cross-run fact seeding** (`agent_factory.py`, `memory_bridge.py`, `dynamic_pipeline.py`) — workers append `REMEMBER:\n- fact` blocks; stored to global Qdrant layer via `bridge.store_remember_facts()`; auto-retrieved on next run via existing `retrieve_global_context()`;
  5. **Pipeline visualization** (`cmd_status.py`) — `inspect` command now renders a Rich Tree hierarchy showing every pipeline node (master_receive → judge_brief → analyse → researcher? → execution_lead → [workers] → judge_review → master_respond) with per-worker model, result preview, and FINDINGS facts inline; bypass/fast_llm gate runs shown separately.
- [x] Phase 5.1 (remaining — now complete): Hardening
  - **Circuit breakers** (`core/retry.py`) — `is_circuit_open(model_id, threshold)` tracks consecutive failures per model in Qdrant `aifactory_circuit_breaker`; wired into `PipelineAgent.think()` at line ~604 of `dynamic_pipeline.py`; `reset_circuit()` for manual recovery; `AIFACTORY_CIRCUIT_BREAKER_THRESHOLD` setting
  - **Budget hard caps** (`core/budget_tracker.py`) — `BudgetTracker` accumulates per-run USD cost; `check()` raises `BudgetExceededError` before any call that would exceed `AIFACTORY_MAX_BUDGET_USD`; `on_warn` callback at 80% threshold; wired into every `PipelineAgent` via `_budget_tracker` in `build_dynamic_pipeline()`
  - **Batch pipeline** (`cli/cmd_batch.py`) — `aifactory batch run tasks.jsonl [--concurrency N] [--mode balanced] [--output out.jsonl]`; JSONL input/output; concurrency-limited via `asyncio.Semaphore`; progress bar via Rich; `aifactory batch status <batch-id>`
- [x] Phase 5.2: Undocumented features now on the books
  - **`aifactory batch`** — see Phase 5.1 above
  - **`aifactory runs list/show/search`** (`cli/cmd_runs.py`) — browse and semantically search past pipeline runs stored in `RunStore` (Qdrant or JSONL fallback)
  - **`aifactory meta review`** (`cli/cmd_meta.py`) — meta-agent LLM reviews last N pipeline runs and produces system improvement recommendations
  - **`aifactory meta calibrate`** (`cli/cmd_meta.py` + `core/model_calibrator.py`) — applies RL-derived ability score patches to `model_registry.yaml`; produces `_empirical_scores` per model; `--apply` flag to write changes
  - **`aifactory gate <query>`** (`cli/cmd_status.py`) — test any query against ComplexityGate, see which tier it routes to and why
  - **`aifactory cost`** (`cli/cmd_cost.py`) — cost tracking and reporting for past runs
  - **`aifactory connect` / `aifactory tools`** (`cli/cmd_connect.py`) — integration and tool connection management
  - **OpenTelemetry** (`core/otel.py`) — distributed tracing exporter; toggle `AIFACTORY_OTEL_ENABLED`; endpoint `AIFACTORY_OTEL_ENDPOINT`; service name `AIFACTORY_OTEL_SERVICE_NAME`
  - **`RunStore`** (`core/run_store.py`) — persistent record of every pipeline run; Qdrant-backed with JSONL fallback; powers `aifactory runs` and meta-agent
  - **`ModelCalibrator`** (`core/model_calibrator.py`) — reads RL history, computes empirical ability score deltas, patches `model_registry.yaml` with `_empirical_scores` section
  - **API auth** (`api/main.py`, `config/settings.py`) — `AIFACTORY_API_KEY` shared key via `X-API-Key` header on POST/DELETE endpoints; empty = dev mode (no auth)
  - **Redis-backed rate limiter** (`core/rate_limiter.py`) — uses `REDIS_URL` as `storage_uri` when set; falls back to in-memory for single-process dev
  - **SHA-256 point IDs** (`core/semantic_cache.py`, `core/run_store.py`) — replaced MD5 with SHA-256 for Qdrant point ID hashing
  - **`CodeSecurityProcessor`** (`agents/code_security_processor.py`) — intercepts `CODE_ANALYSIS:` signals from workers; strips proprietary local code before any LLM API call; B2B security feature preventing code leakage to external providers
  - **`FactsStore`** (`core/facts_store.py`) — JSON-backed REMEMBER: fact persistence at `~/.aifactory/memories/facts.json`; survives Qdrant in-memory restarts; used by `memory_bridge.py` to seed global layer and persist cross-run facts
  - **`ResearchCache`** (`core/research_cache.py`) — persistent researcher-output cache at `~/.aifactory/memories/research.json`; three-tier lookup (exact → cosine ≥ 0.82 → word-overlap ≥ 0.72); wired into `make_researcher()` and checked before every LLM call; embed_fn passed for cosine tier when available
  - **`SemanticClassifier`** (`core/semantic_classifier.py`) — cosine-similarity classifier over 24 labelled anchor examples (bypass/fast_llm/full); called in Stage 6 of ComplexityGate for ambiguous keyword scores (0.2–0.7 confidence band)
  - **`OutputValidator`** (`core/output_validator.py`) — validates worker output before consolidation in `dynamic_pipeline.py`
  - **`database.py`** (`core/database.py`) — PostgreSQL asyncio adapter used by the API layer
  - **`redis_client.py`** (`core/redis_client.py`) — Redis async client wrapper used by the API layer
- [x] Phase 5.3: Opportunity features
  - **5th gate tier: research_cached** (`core/complexity_gate.py`) — Stage 7 in `ComplexityGate.evaluate()`: after routing to "full", checks `ResearchCache` word-overlap; if hit → `GateDecision(fast_llm=True, research_cached=True, research_context=...)` routes query to fast_llm path with injected cached research (skips full 8-node pipeline and researcher agent). `cmd_run.py` `_run_fast_llm()` accepts `research_context` and injects it into master prompt.
  - **Semantic researcher cache lookup** (`graphs/dynamic_pipeline.py`) — `make_researcher()` now accepts `embed_fn`; `ResearchCache.lookup(query, embedding=embed_fn(query))` enables Tier 2 cosine-similarity matching (previously only word-overlap Tier 3 was used). `build_dynamic_pipeline()` loads `embed_fn` from `embed_provider` and passes it through.
  - **FactsStore surfaced in memory CLI** (`cli/cmd_memory.py`) — `memory stats` shows Persistent Facts row; `memory search --layer global/all` queries `FactsStore.load_all()` with word-overlap and appends matching facts to results table; `memory health` shows facts path + count.
  - **Auto-calibration hook** (`graphs/dynamic_pipeline.py`, `config/settings.py`) — `judge_review` fires a background `asyncio.create_task()` after `record_pipeline_performances()` when `AIFACTORY_CALIBRATE_EVERY_N_RUNS > 0` and `run_count % N == 0`. Calls `get_calibration_report()` then `apply_patches(dry_run=False)` — fully non-blocking. Default: 0 (disabled).
- [x] Phase W1–W6: Adaptive Mesh Pipeline — self-modifying graph architecture
  - **W1 — GraphCompiler + GraphSpec** (`graphs/graph_compiler.py`) — `TopologyShape` (MINIMAL/STANDARD/FULL/EXTENDED), `GraphSpec` with `effective_topology` resolver (STANDARD+research→FULL, FULL+no_judge→STANDARD), `GraphCompiler.compile()` delegates to topology-specific builders; `build_dynamic_pipeline` delegates to `GraphCompiler` when spec is present; analyser emits `graph_spec` JSON block.
  - **W2 — KnowledgeGraph entity store** (`core/knowledge_graph.py`) — in-process entity/relation store (EntityNode: file/function/class/import, RelationEdge: contains/imports/calls); snapshot-based diff (`diff_since()`); `ingest_codebase()` + `ingest_structural_index()` (parses CSP text output); `has_integration_code()`, `stats()`, `to_summary()`; MemoryRouter KG bridge methods (`kg_query_functions/classes`, `kg_has_integration_code`, `attach_knowledge_graph`); KG populated from `CodeSecurityProcessor.build_structural_index()` at pipeline start.
  - **W3 — WaveGovernor node + EXTENDED topology** (`graphs/wave_governor.py`) — post-wave quality controller; `_compute_quality()` structural heuristic (error→0.0, REVISION→0.3, ok→1.0); mode escalation map (cost_efficient→balanced→best); `make_wave_governor()` LangGraph node; `route_after_wave_governor()` conditional edge; EXTENDED topology auto-selected when `gate_complexity_score > EXTENDED_THRESHOLD`; wired between `execution_lead` and `judge_review` in `_compile_full(extended=True)`.
  - **W4 — Worker Auction** (`core/worker_auction.py`) — second-price sealed-bid auction; `WorkerBid`, `AuctionResult` (with `to_shared_ctx_update()` that stores `auction_floor`/`auction_winner`/`auction_winner_model`); `bids_for_task()` generates keyword-signal bids for 4 specialist roles; RL history adjusts confidences multiplicatively; auction runs in `execution_lead` and writes to `shared_ctx`; WaveGovernor reads `auction_floor` and escalates if quality falls below it.
  - **W5 — End-to-end wiring** (`graphs/graph_compiler.py`, `graphs/dynamic_pipeline.py`) — `GraphCompiler._kg_upgrade_spec()`: KG integration code → FULL topology; >200 entities → STANDARD; called at start of `compile()`; `judge_review` W5 block feeds auction winner model + task_type quality back into `RLScorer` so future auctions use RL-informed confidences.
  - **W6 — Self-calibrating gate** (`core/gate_calibrator.py`) — `compute_patches()` groups `composite_quality` by topology from run history, nudges LITE/EXTENDED thresholds toward topology choices that produced higher quality; `save_thresholds()` persists to `~/.aifactory/gate_thresholds.json`; `build_dynamic_pipeline` loads calibrated thresholds at startup; `judge_review` fires background calibration task every `AIFACTORY_GATE_CALIBRATE_EVERY_N_RUNS` runs (default 0); CLI: `aifactory gate-calibrate [--apply]`.

**Current test count: 1505 passing**

### Key Design Decisions

1. **Config-driven agents** — Agent behaviour comes from YAML files, not Python subclasses. One `PipelineAgent` class handles all roles.
2. **10-ability model scoring** — Each model rated 0-3 on reasoning, coding, math, writing, instruction_following, tool_use, knowledge, speed, cost_efficiency, context_handling. Agent YAML files declare required abilities. Dot-product scoring selects the best model per role+mode.
3. **3 execution modes** — `best` (quality-first), `balanced` (default), `cost_efficient`. Each mode has an alpha weight that blends quality vs cost/speed.
4. **SHARE: protocol** — Agents append `SHARE: key: value` lines to their output. These are extracted and merged into a shared context dict visible to downstream agents. Phase 2.9: SHARE: updates are also persisted to team-layer memory for cross-run retrieval.
5. **LiteLLM for model abstraction** — switch providers with a string change. Supports OpenAI, Anthropic, Bedrock, etc.
6. **Memory promotion every N turns** — agent layer summarised → team layer. Interval set via `AIFACTORY_MEMORY_PROMOTION_INTERVAL` (default: 10).
7. **In-memory Qdrant for dev** — no Docker needed. Falls back automatically if `QDRANT_URL` is unreachable.
8. **Token budgets per agent** — hard caps enforced in `BaseAgent.think()`. Raises `AgentBudgetError` when exceeded.
9. **uv for package management** — 10-100x faster than pip. Venv: `.venv` inside the project root.
10. **Stub embeddings in dev** — `MemoryRouter` uses a SHA-256 hash-based pseudo-embedding (1536-dim). Pass a real `embedding_fn` for production semantic search.
11. **Env var naming** — LLM API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) have **no prefix**. All other settings use the `AIFACTORY_` prefix. LangSmith uses `LANGCHAIN_` prefix.
12. **Tests are async-native** — `asyncio_mode = "auto"` in `pyproject.toml`; no `@pytest.mark.asyncio` needed.
13. **LangSmith tracing is zero-cost when disabled** — `@traceable` decorator is a passthrough when `LANGCHAIN_API_KEY` is absent.
14. **Stack: LangGraph + LlamaIndex + LangSmith + LiteLLM** — LangGraph for orchestration, LlamaIndex for memory/RAG, LangSmith for observability, LiteLLM for cross-provider LLM calls.
15. **Project-level `pip.conf`** — `pip.conf` at the repo root pins PyPI as the index.
16. **E2B sandbox** — `Sandbox.create(api_key=..., timeout=...)` for SDK v2.6+. Auto-detected via `E2B_API_KEY` env var → settings/.env fallback. Each worker's code runs in a fresh isolated micro-VM — no shared state between workers. Now third priority after Docker MCP.
22. **Docker MCP Sandbox (Phase 4.0)** — `core/sandbox_pool.py` + `mcp/sandbox_server.py` + `mcp/client.py`. Warm Docker container pool (min_idle=4, max_size=16). Every container runs with: `--network=none --cap-drop=ALL --read-only --tmpfs /workspace --pids-limit=64 --memory=512m` + custom seccomp profile. gVisor (`runsc`) runtime optional. FastMCP server exposes `execute_python/bash/javascript + pool_status`. Two client modes: HTTP (prod, bearer auth) or stdio/in-process (dev, zero config). Auto-selected via `AIFACTORY_MCP_SANDBOX_URL`. Backend priority: docker > e2b > local. Toggle with `AIFACTORY_SANDBOX_DOCKER_ENABLED=true`.
23. **Package registry injection** — `config/package_registry.yaml` loaded once (LRU-cached) by `agent_factory.py`. Its `prompt_snippet` field is appended to every worker system prompt so LLMs never attempt to `pip install` packages that are already pre-installed in the sandbox image (numpy, pandas, requests, sklearn, etc.).
17. **DSA-optimised coding prompts** — CODING EXCELLENCE instructions in worker.yaml + agent_factory.py. Concise: Big-O in code comments, not separate prose. Execution lead merges dependent coding tasks the analyser over-splits.
18. **Analyser CODING RULE** — write+test+run+show = 1 task. Prevents sandbox isolation errors from splitting tightly-coupled code across workers.
19. **MemoryBridge (Phase 2.9)** — `core/memory_bridge.py` is the fail-safe adapter between `PipelineAgent` and `MemoryRouter`. All ops are try/except-wrapped at both bridge and `think()` level — memory failures never break the pipeline. `build_dynamic_pipeline()` auto-creates one bridge per run (using `run_id` as `team_id` for isolation). Disable with `AIFACTORY_MEMORY_PIPELINE_ENABLED=false`. Tune injection with `AIFACTORY_MEMORY_CONTEXT_TOP_K` and `AIFACTORY_MEMORY_CONTEXT_MAX_CHARS`.
20. **ComplexityGate v2** — `core/complexity_gate.py`. 4-tier: pure math bypass (AST eval, 0 tokens) → static facts (dict lookup, 0 tokens) → fast_llm (single master call, ~1s) → full 8-node pipeline. YAML-driven patterns in `config/gate_patterns.yaml`. Wired into both `aifactory new` and `aifactory run`.
24. **RL-lite Adaptive Model Selection (Phase 3)** — `core/rl_scorer.py` stores `(model_id, task_type) → quality_score` in a dedicated Qdrant collection (`aifactory_rl_performance`). After each pipeline run, `judge_review` calls `record_pipeline_performances()` which normalises judge quality (0–3 → 0–10) per worker and persists to Qdrant. Before worker model selection, `create_worker_config()` calls `get_model_rl_scores(task_type)` and passes results to `select_model()`. The RL signal is applied as a multiplicative factor: `adj = 1 + rl_weight × (rl_score/5 − 1)`, capped at ±rl_weight (default ±20%). Only models with ≥ `AIFACTORY_RL_MIN_SAMPLES` records (default: 3) are adjusted — prevents a single run from skewing selection. Task type inferred by keyword matching (7 types). Falls back to in-memory Qdrant if remote unavailable. Toggle: `AIFACTORY_RL_ENABLED`. CLI: `aifactory rl stats [--type coding]` / `aifactory rl reset`.
25. **MCP Catalog Strategy (investigated April 2026)** — `docs/mcp-catalog-feasibility.md`. Decision: hybrid approach. Full Composio replacement is premature (token vault / per-tenant OAuth management is 3–4 weeks of infra). Phase 1: add `MCPProvider` class behind feature flag. Phase 2: migrate GitHub, Jira, Notion, Google Workspace to direct MCP servers (data residency + audit wins for B2B enterprise). Phase 3: full catalog + token vault.
26. **DEPLOY_MODE MCP routing (Phase 4.2)** — `AIFACTORY_DEPLOY_MODE=local` (default) uses stdio subprocess per MCP server; `cloud` uses an async HTTP session pool. Pool is `asyncio.Queue`-backed, N sessions per server (default 3, per-server `pool_size` override in `mcp_catalog.yaml`). Tool calls in `cloud` mode: `queue.get(timeout=30)` → call → `queue.put_nowait`. Pool cleanup in `disconnect()` drains stacks before closing stdio.
27. **Parallel worker execution (Phase 4.3)** — Workers execute in dependency-ordered waves. `_build_dependency_waves(tasks, deps_map)` topological sort → `list[list[task]]`. Single-task waves run directly (preserves `interrupt()` semantics). Multi-task waves: `asyncio.gather(*coros, return_exceptions=True)`. Each worker returns `(result_entry, ctx_delta)` — only net context changes are returned so parallel workers don't clobber shared_ctx. Deltas merged sequentially after gather.
28. **Worker role specialization (Phase 4.3)** — `create_worker_config()` detects task type (coding/math/research/writing/analysis) by keyword. Loads matching specialist YAML (`worker_coder/researcher/writer/analyst.yaml`). Specialist `task` text prepended before the execution lead's instruction, so the LLM assumes the role persona before reading the task. Falls back to generic `worker.yaml` for unrecognised types.
29. **Self-correction loop (Phase 4.3)** — After tool rounds complete, non-tool workers do one self-check `think()`: "Is this complete and accurate? Reply VALIDATED or REVISION: <note>\n<corrected>". `REVISION:` triggers a single corrected response that replaces the original result. Skipped for tool tasks (quality determined by tool success) and `NOT_NEEDED` outputs. One revision max — no infinite loop.
30. **Token streaming (Phase 5.0)** — `PipelineAgent.on_token` callback wired into `think()`. When set, `_think_stream()` uses `litellm.acompletion(..., stream=True)` and fires the callback for each delta token. Falls back to non-streaming automatically on pre-first-token errors. CLI `cmd_run.py` emits a `"token"` event that writes raw tokens to stdout. Toggle: `AIFACTORY_STREAM_TOKENS_ENABLED`.
31. **Auto-embedding provider (Phase 5.0)** — `core/embed_provider.py` auto-selects the best embedding model from available API keys: OpenAI `text-embedding-3-small` → Voyage `voyage-3-lite` → Cohere `embed-english-light-v3.0` → HuggingFace `all-MiniLM-L6-v2` (local) → SHA-256 stub. Stub mode emits a prominent warning panel and clearly instructs the fix. Redis embedding cache wraps raw embed calls (key: `sha256[:24]`, TTL: 7d). `MemoryRouter` calls `build_embed_fn()` instead of the old hardcoded model.
32. **Semantic cache (Phase 5.0)** — `core/semantic_cache.py`. Collection: `aifactory_semantic_cache`. `lookup(query)` → embed → kNN (limit=1, score_threshold) → TTL check → return cached `final_response` if hit. `store(query, final_response)` → embed → upsert. Wired into `master_receive` (check) and `master_respond` (store). Settings: `AIFACTORY_SEMANTIC_CACHE_ENABLED`, `AIFACTORY_SEMANTIC_CACHE_THRESHOLD` (default 0.92), `AIFACTORY_SEMANTIC_CACHE_TTL_DAYS`, `AIFACTORY_SEMANTIC_CACHE_MAX_SIZE`.
33. **Judge deep eval (Phase 5.0)** — After synthetic/LLM judge metrics, an additional focused LLM call scores `final_response` on relevance (0.30), accuracy (0.30), completeness (0.25), actionability (0.15) producing `composite_quality` and `eval_*` fields in judge metrics. Toggle: `AIFACTORY_JUDGE_DEEP_EVAL_ENABLED`. Model: `AIFACTORY_JUDGE_EVAL_MODEL` (default `openai/gpt-4o-mini`).
34. **`aifactory serve` (Phase 5.0)** — `cmd_serve.py`. Starts `aifactory.api.main:app` via uvicorn. Pre-warms Redis, Qdrant, embedding provider, LiteLLM cache, model registry, and Composio before accepting requests. Options: `--host`, `--port`, `--workers`, `--reload`, `--log-level`, `--no-warmup`. `serve`, `new`, and `run` are independent: `serve` is a persistent multi-request API server; `new`/`run` are one-shot CLI commands that spin up their own pipeline process.
35. **Redis task store (Phase 5.0)** — `cmd_run.py`. At run start: `SET aifactory:run:{rid}` (status=running) + `ZADD aifactory:runs:all`. At completion: updates status=completed with elapsed_s, total_tokens, final_response_preview, judge_status. TTL: 7d. Falls back silently if `REDIS_URL` is not set. Consumed by `api/main.py` and `aifactory runs` CLI.
36. **5th ComplexityGate tier: research_cached (Phase 5.3)** — Stage 7 of `ComplexityGate.evaluate()` fires after all keyword/semantic stages fail to return fast_llm. Checks `ResearchCache` with word-overlap (no embed required at gate time). A hit returns `GateDecision(fast_llm=True, research_cached=True, research_context=<summary>)`. `cmd_run.py _run_fast_llm()` detects `research_context` and injects it into the master prompt — full pipeline + researcher agent entirely skipped.
37. **Semantic researcher lookup (Phase 5.3)** — `make_researcher()` accepts `embed_fn`; `ResearchCache.lookup(query, embedding=vec)` now uses Tier 2 cosine similarity (threshold 0.82) instead of word-overlap fallback only. `build_dynamic_pipeline()` calls `embed_provider.build_embed_fn()` once and passes the result through. Falls back gracefully when embed provider is stub.
38. **FactsStore in memory CLI (Phase 5.3)** — `memory stats` shows a "Persistent Facts" row counting `FactsStore` entries. `memory search --layer global/all` queries `FactsStore.load_all()` with word-overlap and shows matching facts in the results table under the `facts` layer label. `memory health` reports path and count of stored facts.
39. **Auto-calibration (Phase 5.3)** — `AIFACTORY_CALIBRATE_EVERY_N_RUNS` (default 0 = disabled). When set to N, `judge_review` counts total stored runs after `record_pipeline_performances()`; if `count % N == 0`, fires `asyncio.create_task(_auto_calibrate())` — non-blocking background task that calls `get_calibration_report()` + `apply_patches(dry_run=False)` to self-update `model_registry.yaml` ability scores. Recommended: 20.
40. **Lite mode (`AIFACTORY_LITE_MODE`)** — When enabled, `build_dynamic_pipeline()` flattens the graph to `master_receive → analyse → execution_lead → master_respond`. Skips: `judge_brief` (no quality criteria), `reanalyse` node, and the judge LLM call (replaced with synthetic metrics). Saves ~800–1500 tokens per run. The researcher router switches to `route_after_researcher_lite`. Suitable for high-volume batch runs where quality pre-checks aren't needed.
41. **CodeSecurityProcessor** (`agents/code_security_processor.py`) — intercepts `CODE_ANALYSIS:` signals in worker outputs. When a worker emits `CODE_ANALYSIS: <path>`, the processor reads the file locally and passes only safe abstractions (function signatures, class names, docstrings) to the LLM — never raw source. This prevents proprietary local code from leaking to external LLM APIs (OpenAI, Anthropic). Initialised in `build_dynamic_pipeline()` and passed to each worker via `_worker_kwargs`.
21. **Token Efficiency (Phase 3.5)** — Three layered optimisations in `dynamic_pipeline.py`:
    - **Anthropic prompt caching** — `cache_control: ephemeral` on system messages for `claude-*`, `anthropic/claude-*`, and `openrouter/anthropic/claude-*`. 60–90% cost reduction on repeat calls with the same system prompt. Off by default when routing through a local LiteLLM proxy. Toggle with `AIFACTORY_PROMPT_CACHE_ENABLED` and `AIFACTORY_PROMPT_CACHE_OPENROUTER`.
    - **LiteLLM response cache** — `litellm.cache = Cache(ttl=3600)` initialised once per pipeline. All providers/tiers benefit: identical (model, messages) pairs return 0-token cached responses. Useful for QUESTION/ANSWER retries and repeated runs. `AIFACTORY_RESPONSE_CACHE_ENABLED` / `AIFACTORY_RESPONSE_CACHE_TTL`.
    - **Smart judge** — skip judge LLM call for single-task runs with no research (~800–1200 tokens saved). Synthetic assessment generated from run metadata. Toggle with `AIFACTORY_JUDGE_SKIP_SIMPLE_RUNS`.
    - **Shared context pruning** — `_prune_shared_context(ctx, max_keys=25)` called after analyse, researcher, and execution_lead nodes. Priority keys (objective, tasks_count, etc.) always kept. `AIFACTORY_SHARED_CONTEXT_MAX_KEYS`.

### Agent Execution Pattern

The 8-node dynamic pipeline follows this flow:

1. **master_receive** — Acknowledges the request, seeds shared context with the objective
2. **judge_brief** — Defines success criteria and what will be evaluated
3. **analyse** — Breaks request into atomic tasks (or returns a direct answer for trivial queries)
4. **researcher** (conditional) — Gathers research for pre-analysis or execution
5. **reanalyse** (conditional) — Refines task breakdown after research
6. **execution_lead** — Plans worker dispatch with per-task ability requirements and dependency graph, spawns N workers dynamically with per-task model + specialist role selection, runs workers in parallel dependency waves, consolidates results
7. **judge_review** — Evaluates each agent's performance, produces quality metrics
8. **master_respond** — Synthesises final response for the human (no mention of internal process)

Workers support:
- **NOT_NEEDED** self-skip — if a worker determines its task is already covered
- **QUESTION:** interrupts — worker can pause and ask the human a question (via LangGraph interrupt())
- **SHARE:** protocol — share discovered context with downstream agents
- **Specialist roles** — coding/math → Coder persona; research → Researcher persona; writing → Writer persona; analysis → Analyst persona
- **Self-correction** — one VALIDATED/REVISION: self-check after initial response (non-tool, non-skipped tasks)
- **Parallel execution** — tasks in the same dependency wave run concurrently via asyncio.gather()

### Documentation

Project docs live in `docs/`:
- `docs/architecture-v2.md` — full system architecture (LangGraph + LlamaIndex + LangSmith)
- `docs/sprints.md` — sprint plan with stories and acceptance criteria
- `docs/progress.md` — what’s built, current status, what’s next
- `docs/decisions.md` — key technical decisions and rationale
