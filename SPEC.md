# AI Factory — Technical Specification

**Version:** 1.1.0
**Phase:** 5.2
**Last updated:** 2026-04-16
**Tests passing:** 1053

This document is the authoritative spec for what has been built. Use it as the baseline for spec-driven development of future phases.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Agent Configuration](#3-agent-configuration)
4. [Inter-Agent Protocols](#4-inter-agent-protocols)
5. [Memory Architecture](#5-memory-architecture)
6. [Code Execution Sandbox](#6-code-execution-sandbox)
7. [RL-Lite Adaptive Model Scoring](#7-rl-lite-adaptive-model-scoring)
8. [Token Efficiency](#8-token-efficiency)
9. [Integrations](#9-integrations)
10. [API](#10-api)
11. [CLI](#11-cli)
12. [Security](#12-security)
13. [Observability](#13-observability)
14. [Configuration Reference](#14-configuration-reference)
15. [Roadmap — Remaining Work](#15-roadmap--remaining-work)

---

## 1. System Overview

AI Factory is an 8-node LangGraph pipeline that routes every request through a 4-tier complexity gate, assembles specialist agents at runtime from YAML configs, runs workers in parallel dependency waves, and returns a synthesised response.

### Core Properties

| Property | Detail |
|---|---|
| Config-driven | Agent behaviour from YAML — no Python subclasses per role |
| Model scoring | 10-ability dot-product scoring selects the best model per role and mode |
| 3 execution modes | `best`, `balanced` (default), `cost_efficient` |
| Complexity routing | 4-tier gate before pipeline (math bypass → static facts → fast_llm → full) |
| Parallel execution | Workers run in dependency-ordered waves via `asyncio.gather()` |
| 3-layer memory | Agent → Team → Global (Qdrant), auto-promoted every N turns |
| LLM interface | LiteLLM — any provider (OpenAI, Anthropic, Bedrock, Google) |

---

## 2. Pipeline Architecture

### 2.1 ComplexityGate — Pre-Pipeline Routing

Every request passes through `ComplexityGate` before any LLM call:

| Tier | Trigger | Handler | Cost |
|---|---|---|---|
| `bypass_math` | Pure arithmetic expression | Python `ast.literal_eval` | 0 tokens |
| `bypass_static` | Static fact lookup (capitals, conversions, etc.) | Dict lookup | 0 tokens |
| `fast_llm` | Simple factual / single-hop question | Single `master` LLM call | ~200 tokens |
| `full_pipeline` | Multi-step, research-needed, tool-use, coding | Full 8-node LangGraph | Full cost |

Patterns are YAML-driven (`config/gate_patterns.yaml`). Each pattern has a `weight` (0.0–1.0). Scoring: `min(max_weight + 0.05 × (n_matches - 1), 1.0)`. The `fast_llm_threshold` (default: 0.7) controls when strong QA confidence routes to the fast path.

Bypass and fast_llm paths write a run dump so they appear in `aifactory status`.

### 2.2 8-Node Pipeline

```
master_receive
  → judge_brief
  → analyse
    ├── (direct_answer) → master_respond
    ├── (research_for_analysis) → researcher → reanalyse → execution_lead
    ├── (research_needed) → researcher → execution_lead
    └── (no research) → execution_lead
  → judge_review
  → master_respond
```

| Node | Agent | Responsibility |
|---|---|---|
| `master_receive` | master | Acknowledge request; seed `shared_ctx`; check semantic cache |
| `judge_brief` | judge | Define success criteria and quality bar |
| `analyse` | analyser | Break request into atomic tasks; detect direct-answer bypass |
| `researcher` | researcher | Gather research (pre-analysis or execution phase) |
| `reanalyse` | analyser | Refine task breakdown after research |
| `execution_lead` | execution_lead | Plan + spawn workers; run dependency waves; consolidate results |
| `judge_review` | judge | Score per-worker quality; record RL data; produce judge metrics |
| `master_respond` | master | Synthesise final response; store to semantic cache |

### 2.3 Worker Execution Inside `execution_lead`

1. `execution_lead` parses the task plan and extracts `dependencies_map`.
2. `_build_dependency_waves()` topological sort → `list[list[Task]]`.
3. Single-task waves run directly (preserves `interrupt()` for QUESTION: prompts).
4. Multi-task waves: `asyncio.gather(*coros, return_exceptions=True)`.
5. Each worker returns `(result_entry, ctx_delta)` — only net context delta merged into `shared_ctx`.
6. Deltas merged sequentially after gather to prevent clobber.

### 2.4 Worker Self-Correction

After completing initial output, non-tool workers do one self-check `think()`:

> "Is this complete and accurate? Reply VALIDATED or REVISION: <note>\n<corrected>"

- `REVISION:` triggers one corrected response that replaces the original.
- Skipped for tool tasks and `NOT_NEEDED` outputs.
- One revision max — no infinite loop.

### 2.5 State — `GraphState`

LangGraph state dict passed between nodes:

| Key | Type | Description |
|---|---|---|
| `task` | `str` | Original user request |
| `context` | `str` | Additional user context |
| `mode` | `str` | Execution mode: best/balanced/cost_efficient |
| `run_id` | `str` | UUID for this run (team memory isolation) |
| `shared_ctx` | `dict` | Live shared context dict (updated by SHARE: protocol) |
| `tasks` | `list` | Parsed task list from analyser |
| `worker_results` | `list` | Per-worker result entries |
| `judge_metrics` | `dict` | Judge quality scores |
| `final_response` | `str` | Synthesised response from master |
| `semantic_cache_hit` | `bool` | Whether master_receive found a cache hit |

---

## 3. Agent Configuration

### 3.1 YAML Structure (`config/agents/*.yaml`)

```yaml
role: analyser
tier: standard
system_prompt: |
  You are the Analyser agent...
abilities:
  - reasoning
  - instruction_following
token_budget: 8000
temperature: 0.3
```

| Field | Description |
|---|---|
| `role` | Maps to a `PipelineAgent` instance |
| `tier` | `premium`, `standard`, or `budget` — influences model pool |
| `abilities` | Required abilities from the 10-ability registry |
| `token_budget` | Hard cap enforced in `BaseAgent.think()`; raises `AgentBudgetError` if exceeded |
| `system_prompt` | Full system prompt with Jinja-compatible interpolation |

### 3.2 Agent Roles

| File | Role | Purpose |
|---|---|---|
| `master.yaml` | master | Coordinator + synthesiser |
| `judge.yaml` | judge | Quality reviewer + RL recorder |
| `analyser.yaml` | analyser | Task breakdown + direct-answer detection |
| `researcher.yaml` | researcher | Research gathering |
| `execution_lead.yaml` | execution_lead | Worker dispatcher + consolidator |
| `worker.yaml` | worker | Generic worker template |
| `worker_coder.yaml` | worker_coder | Specialist: Software Engineer |
| `worker_researcher.yaml` | worker_researcher | Specialist: Research Analyst |
| `worker_writer.yaml` | worker_writer | Specialist: Technical Writer |
| `worker_analyst.yaml` | worker_analyst | Specialist: Data/Strategy Analyst |

### 3.3 10-Ability Model Scoring

Each model in `config/model_registry.yaml` is rated 0–3 on:

```
reasoning, coding, math, writing, instruction_following,
tool_use, knowledge, speed, cost_efficiency, context_handling
```

Selection process in `model_selector.py`:

1. Filter to allowed models for the given tier/mode.
2. Compute dot-product: `Σ(model_ability[i] × required_weight[i])`.
3. Apply mode `max_quality` cap: `best=3`, `balanced=2`, `cost_efficient=1`.
4. Blend RL signal: `adj = 1 + rl_weight × (rl_score/5 − 1)`, capped at ±rl_weight.
5. Return top-scoring model.

Speed and cost_efficiency are `_EFFICIENCY_ABILITIES` — they bypass the quality cap.

### 3.4 Worker Role Specialisation

`create_worker_config()` detects task type by keyword:

| Keywords | Specialist YAML |
|---|---|
| code, implement, build, script, algorithm, function, debug | `worker_coder.yaml` |
| research, find, investigate, discover, survey | `worker_researcher.yaml` |
| write, draft, document, explain, describe, compose | `worker_writer.yaml` |
| analyse, evaluate, compare, assess, review, audit | `worker_analyst.yaml` |
| _(no match)_ | `worker.yaml` |

---

## 4. Inter-Agent Protocols

All protocols are parsed from raw LLM text. No structured API between agents.

### 4.1 SHARE: — Context Broadcasting

```
SHARE: key: value
```

- Any agent can emit SHARE: lines.
- `agent_factory.py` extracts and merges into `shared_ctx`.
- Also persisted to **team memory layer** for cross-run retrieval.

### 4.2 FINDINGS: — Compact Fact Passing

```
FINDINGS:
- fact one
- fact two
```

- Workers append after completing tasks.
- Extracted per worker, injected as `## Prior Task Findings` into subsequent waves.
- 80–90% token reduction vs injecting full prose results.

### 4.3 RESEARCH: — Inline Research

```
RESEARCH: <search query>
```

- Worker emits mid-task when external info is needed.
- Pipeline intercepts, calls `researcher` inline, injects result back.
- Worker completes without pipeline restart.
- `researcher` param threaded: `build_dynamic_pipeline` → `make_execution_lead` → `_worker_kwargs`.

### 4.4 REMEMBER: — Cross-Run Fact Seeding

```
REMEMBER:
- important fact
```

- Workers append facts worth preserving across runs.
- Stored to **global Qdrant layer** via `MemoryBridge.store_remember_facts()`.
- Auto-retrieved on future runs via `retrieve_global_context()`.

### 4.5 CODE_ANALYSIS: — Secure Codebase Analysis

```
CODE_ANALYSIS: /absolute/path/to/project
```

- Worker emits to request codebase analysis.
- Pipeline calls `CodeSecurityProcessor` locally:
  1. Reads files from disk (`read_from_path()`).
  2. LLM reasons over full source code **in-process** — raw code never leaves.
  3. Returns only: plain-English summary, file metadata (names, not bodies), file count, languages.
- Safe data injected back into worker context.
- `_sanitize_params()` provides a second line of defence before external tool calls.

### 4.6 ACTION: — External Tool Calls

```
ACTION:composio
{"tool": "GITHUB_CREATE_ISSUE", "params": {"title": "...", "body": "..."}}
ACTION_END

ACTION:mcp
{"tool": "github::create_issue", "params": {...}}
ACTION_END
```

- Parsed by `extract_action_blocks()` in `executor.py`.
- Dispatched to `ComposioProvider` or `MCPProvider`.
- `_sanitize_params()` scans all param values — redacts code-looking strings (≥300 chars, ≥3 code signals).

### 4.7 QUESTION: — Human-in-the-Loop

```
QUESTION: <question for the human>
```

- Worker pauses via LangGraph `interrupt()`.
- Human answers via CLI prompt or API.
- `aifactory resume <run-id>` resumes.

### 4.8 NOT_NEEDED

```
NOT_NEEDED
```

- Worker self-skips if task already covered by prior workers.
- No self-correction applied.

---

## 5. Memory Architecture

### 5.1 Three Layers

| Layer | Scope | Namespace | Use |
|---|---|---|---|
| Agent | Per-agent episodic | `agent:{agent_id}` | STM, raw outputs |
| Team | Per-run shared | `team:{run_id}` | SHARE: updates, team context |
| Global | Cross-run | `global` | REMEMBER: facts, promoted summaries |

Backend: Qdrant. In-memory Qdrant used in dev (auto-fallback when no `QDRANT_URL`).

### 5.2 MemoryBridge

`core/memory_bridge.py` — fail-safe adapter between `PipelineAgent` and `MemoryRouter`.

```
PipelineAgent.think(user_msg)
  [1] bridge.retrieve_context(agent_id, user_msg)
      → MemoryRouter.retrieve(levels=[AGENT, TEAM])
      → "## Relevant Memory\n..." prefix
  [2] _build_system(shared_ctx, memory_context)
  [3] litellm.acompletion(...)
  [4] bridge.store_output(agent_id, reply, turn_number)
  [5] bridge.maybe_promote(agent_id, summary)  ← if turn_count >= threshold
```

- All ops are try/except-wrapped — memory failures never kill the pipeline.
- `run_id` used as `team_id` — isolates team layer per run.
- `MemoryBridge(router=None)` → all methods become noops (used in tests).
- Disable: `AIFACTORY_MEMORY_PIPELINE_ENABLED=false`.

### 5.3 Auto-Embedding Provider

`core/embed_provider.py` — selects the best available embedding model:

```
OpenAI text-embedding-3-small (1536-dim)
→ Voyage voyage-3-lite
→ Cohere embed-english-light-v3.0
→ HuggingFace all-MiniLM-L6-v2 (local)
→ SHA-256 stub (always available, emits warning panel)
```

Redis embedding cache (key: `sha256[:24]`, TTL: 7d) wraps raw embed calls.

### 5.4 Memory Settings

| Setting | Default | Description |
|---|---|---|
| `AIFACTORY_MEMORY_PIPELINE_ENABLED` | `true` | Wire MemoryBridge into pipeline |
| `AIFACTORY_MEMORY_CONTEXT_TOP_K` | `3` | Max entries retrieved per think() |
| `AIFACTORY_MEMORY_CONTEXT_MAX_CHARS` | `800` | Max chars of memory injected per think() |
| `AIFACTORY_MEMORY_PROMOTION_INTERVAL` | `10` | Promote agent → team every N turns |

---

## 6. Code Execution Sandbox

### 6.1 Backend Priority

```
Docker MCP → E2B cloud → local subprocess
```

Auto-detected by `sandbox.py` based on `AIFACTORY_SANDBOX_DOCKER_ENABLED` and `E2B_API_KEY`.

### 6.2 Docker MCP Sandbox

Container pool managed by `core/sandbox_pool.py`:

- `min_idle=4`, `max_size=16`
- Security flags: `--network=none --cap-drop=ALL --read-only --tmpfs /workspace --pids-limit=64 --memory=512m` + custom seccomp profile
- gVisor (`runsc`) runtime optional (kernel-level isolation)

FastMCP server (`mcp/sandbox_server.py`) exposes:
- `execute_python` — run Python code
- `execute_bash` — run shell commands
- `execute_javascript` — run Node.js code
- `pool_status` — container pool health

Two client modes (`mcp/client.py`):
- `stdio` / in-process: dev, zero config
- HTTP: prod, bearer auth, auto-detected via `AIFACTORY_MCP_SANDBOX_URL`

### 6.3 Package Registry

`config/package_registry.yaml` lists all pre-installed sandbox packages.

`prompt_snippet` is appended to every worker system prompt — LLMs never `pip install` packages that are already available (numpy, pandas, requests, sklearn, etc.).

---

## 7. RL-Lite Adaptive Model Scoring

### 7.1 Data Model

Qdrant collection: `aifactory_rl_performance`

Each record: `(model_id, task_type, quality_score, timestamp)`

Task types: `coding`, `math`, `research`, `writing`, `analysis`, `general`, `other`

### 7.2 Score Flow

After each pipeline run:

1. `judge_review` calls `record_pipeline_performances()`.
2. Normalises judge quality (0–3 → 0–10) per worker.
3. Persists records to Qdrant.

Before worker model selection:

1. `create_worker_config()` calls `get_model_rl_scores(task_type)`.
2. `select_model()` blends: `adj = 1 + rl_weight × (rl_score/5 − 1)`.
3. Only models with ≥ `AIFACTORY_RL_MIN_SAMPLES` records are adjusted.

### 7.3 Settings

| Setting | Default | Description |
|---|---|---|
| `AIFACTORY_RL_ENABLED` | `true` | Enable RL signal in model selection |
| `AIFACTORY_RL_WEIGHT` | `0.2` | Max RL adjustment (±20%) |
| `AIFACTORY_RL_MIN_SAMPLES` | `3` | Min records before RL is applied |

---

## 8. Token Efficiency

### 8.1 Anthropic Prompt Caching

- `cache_control: ephemeral` on system messages for `claude-*`, `anthropic/claude-*`, `openrouter/anthropic/claude-*`
- 60–90% cost reduction on repeat calls with the same system prompt
- Toggle: `AIFACTORY_PROMPT_CACHE_ENABLED`

### 8.2 LiteLLM Response Cache

- `litellm.cache = Cache(ttl=3600)` initialised once per pipeline
- Identical `(model, messages)` pairs return 0-token cached responses
- Toggle: `AIFACTORY_RESPONSE_CACHE_ENABLED` / TTL: `AIFACTORY_RESPONSE_CACHE_TTL`

### 8.3 Semantic Cache

- Qdrant collection: `aifactory_semantic_cache`
- `lookup(query)` → embed → kNN (limit=1, score_threshold) → TTL check → return `final_response`
- Wired into `master_receive` (check) and `master_respond` (store)
- Toggle: `AIFACTORY_SEMANTIC_CACHE_ENABLED`
- Threshold: `AIFACTORY_SEMANTIC_CACHE_THRESHOLD` (default: 0.92)
- TTL: `AIFACTORY_SEMANTIC_CACHE_TTL_DAYS` (default: 7)

### 8.4 Smart Judge

- Single-task runs with no research skip the judge LLM call (~800–1200 tokens saved)
- Synthetic assessment built from run metadata
- Toggle: `AIFACTORY_JUDGE_SKIP_SIMPLE_RUNS`

### 8.5 FINDINGS: Protocol

Workers emit compact `FINDINGS:` blocks instead of full prose — 80–90% token reduction when injecting prior results into subsequent waves.

### 8.6 Shared Context Pruning

`_prune_shared_context(ctx, max_keys=25)` after analyse, researcher, and execution_lead nodes. Priority keys always preserved. Setting: `AIFACTORY_SHARED_CONTEXT_MAX_KEYS`.

---

## 9. Integrations

### 9.1 Composio

Primary integration provider. Workers emit `ACTION:composio` blocks. `executor.py` dispatches via `ComposioProvider`.

### 9.2 MCP Catalog

Config: `config/integrations/mcp_catalog.yaml`

| Server | Status | Transport |
|---|---|---|
| GitHub (official Docker image) | Enabled | Docker |
| Atlassian (Jira + Confluence) | Enabled | uvx |
| Notion | Enabled | npx |
| Linear | Enabled | npx |
| Google Workspace | Disabled (pending token vault) | — |
| Slack | Disabled (pending token vault) | — |

Feature flag: `AIFACTORY_MCP_CATALOG_ENABLED=false` (default off).

Tool slug format: `server::tool_name` (e.g. `github::create_issue`)

### 9.3 DEPLOY_MODE

| Mode | Transport | Config |
|---|---|---|
| `local` (default) | stdio subprocess per server | Zero config for dev |
| `cloud` | Async HTTP session pool | `AIFACTORY_MCP_HTTP_POOL_SIZE` sessions per server |

Pool: `asyncio.Queue`-backed. Per-server `pool_size` override in `mcp_catalog.yaml`.

---

## 10. API

### 10.1 Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check with rate limiting info |
| `GET` | `/api/models` | Model registry for the UI |
| `POST` | `/api/tasks` | Submit task (SSE streaming response) |
| `GET` | `/api/tasks/{run_id}` | Poll run status + result |
| `GET` | `/api/observability/runs` | List LangSmith traced runs |
| `GET` | `/api/observability/runs/{run_id}` | Trace detail |
| `GET` | `/api/tasks/{id}/traces` | LangSmith traces for a task |
| `POST` | `/api/memory/store` | Store memory entry |
| `POST` | `/api/memory/retrieve` | Semantic memory search |
| `POST` | `/api/memory/promote` | Promote memory to upper layer |
| `DELETE` | `/api/memory/agent/{id}` | Clear agent memory |
| `GET` | `/api/memory/stats` | Layer entry counts |
| `WS` | `/api/ws/{run_id}` | WebSocket live output |

### 10.2 Rate Limiting (slowapi)

| Tier | Limit | Endpoints |
|---|---|---|
| Tasks | 10/min | `POST /api/tasks` |
| Read | 60/min | All `GET` endpoints |
| Memory | 30/min | All `/api/memory/*` |
| WebSocket | 5/min | `WS /api/ws/*` |

429 response: `{error, detail, retry_after}` + `Retry-After` header.

Disable: `AIFACTORY_RATE_LIMIT_ENABLED=false`.

### 10.3 WebSocket Events

```jsonc
{"type": "status",        "status": "running",  "progress": 0.05}
{"type": "worker_start",  "worker_id": "...",   "task": "..."}
{"type": "worker_done",   "worker_id": "...",   "preview": "..."}
{"type": "token",         "delta": "..."}       // streaming tokens
{"type": "completed",     "progress": 1.0,      "result": {...}}
{"type": "error",         "message": "..."}
```

---

## 11. CLI

### 11.1 Entry Points

| Command | Description |
|---|---|
| `aifactory run "task"` | Main entry: full pipeline via ComplexityGate |
| `aifactory run --task-file task.md` | Read task from file |
| `aifactory run --mode best\|balanced\|cheap\|fast` | Mode aliases |
| `aifactory run -c "context"` | Pass additional context |
| `aifactory run --no-chat` | Skip post-pipeline chat |
| `aifactory resume <run-id>` | Resume a paused (QUESTION:) run |
| `aifactory serve` | Start persistent API server (pre-warms connections) |
| `aifactory ask "question"` | Single-agent quick answer |
| `aifactory status` | List recent runs |
| `aifactory inspect <run-id>` | Rich tree view of pipeline run |

### 11.2 Streaming in `run`

`_stream_event()` callback handles events:

| Event | Action |
|---|---|
| `worker_start` | Print worker task |
| `worker_complete` | Print worker result preview |
| `budget_warning` | Print yellow warning panel |
| `qdrant_in_memory` | Print dim note |
| `token` | Write raw delta token to stdout |

---

## 12. Security

### 12.1 Code Leakage Prevention — `CodeSecurityProcessor`

`agents/code_security_processor.py` — two-phase local analysis:

1. **Phase 1 (local):** Build index over codebase; LLM reasons over full source in-process.
2. **Phase 2 (safe):** Generate plain-English summary; validate and strip any code that slips through via `_validate_summary()`.

Safe output: `{safe_summary, metadata (names only), file_count, total_bytes, languages, warning}`.

Raw code never leaves the processor.

### 12.2 Param Sanitizer — `_sanitize_params()`

`integrations/executor.py` — scans all ACTION: params before dispatch.

`_looks_like_code(text)` detects code by counting signals (def/class/import/return/const/let/function/```). Returns `True` if string ≥300 chars AND ≥3 signals.

Redacted params contain: `"[CODE REDACTED — use CODE_ANALYSIS: protocol instead]"`.

### 12.3 Docker Sandbox Security

Container flags: `--network=none --cap-drop=ALL --read-only --tmpfs /workspace --pids-limit=64 --memory=512m` + custom seccomp profile + optional gVisor.

### 12.4 Pre-Commit Hooks

| Hook | Purpose |
|---|---|
| `ruff` (with `S` rules) | Lint + security rules (eval, SQL injection, insecure hash) |
| `ruff-format` | Code formatting |
| `mypy` | Type checking over `src/aifactory/` |
| `detect-secrets` | Baseline-anchored secret scanning |
| `detect-private-key` | Private key detection |
| `check-yaml/json/toml` | Config file validation |
| `check-added-large-files` | Block files > 500 KB |
| `check-merge-conflict` | Prevent committed merge markers |
| `pytest-fast` | Fast unit tests on Python file changes |

---

## 13. Observability

### 13.1 LangSmith

- `@traceable` on all `BaseAgent.think()` calls
- Zero-cost passthrough when `LANGCHAIN_API_KEY` absent
- Traces tagged with `run_id` — linkable via `/api/tasks/{id}/traces`

### 13.2 Judge Deep Eval

After judge metrics, an additional LLM call scores the final response:

| Dimension | Weight |
|---|---|
| Relevance | 30% |
| Accuracy | 30% |
| Completeness | 25% |
| Actionability | 15% |

Produces `composite_quality` and `eval_*` fields in judge metrics.

Toggle: `AIFACTORY_JUDGE_DEEP_EVAL_ENABLED`. Model: `AIFACTORY_JUDGE_EVAL_MODEL` (default: `openai/gpt-4o-mini`).

### 13.3 Redis Task Store

Each run tracked in Redis:

- `SET aifactory:run:{rid}` — status, elapsed_s, total_tokens, final_response_preview, judge_status
- `ZADD aifactory:runs:all` — sorted-set run index
- TTL: 7 days
- Silent fallback when `REDIS_URL` not set

---

## 14. Configuration Reference

All via environment variables (or `.env` file). LLM API keys have no prefix. All other settings use `AIFACTORY_`. LangSmith uses `LANGCHAIN_`.

### LLM Keys

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic direct access |
| `OPENAI_API_KEY` | OpenAI direct access |
| `GEMINI_API_KEY` | Google Gemini direct access |
| `COHERE_API_KEY` | Cohere API |
| `OPENROUTER_API_KEY` | OpenRouter (multi-provider routing) |

### LangSmith

| Variable | Default | Description |
|---|---|---|
| `LANGCHAIN_TRACING_V2` | `false` | Enable tracing |
| `LANGCHAIN_API_KEY` | — | LangSmith API key |
| `LANGCHAIN_PROJECT` | `aifactory` | Project name |
| `LANGCHAIN_ENDPOINT` | `https://api.smith.langchain.com` | API endpoint |

### Infrastructure

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | in-memory | Qdrant cluster URL |
| `QDRANT_API_KEY` | — | Qdrant API key |
| `REDIS_URL` | — | Redis URL (task store + embedding cache) |

### Pipeline

| Variable | Default | Description |
|---|---|---|
| `AIFACTORY_MODE` | `balanced` | Default execution mode |
| `AIFACTORY_MAX_BUDGET_USD` | `5.00` | Budget warn threshold |
| `AIFACTORY_MAX_ITERATIONS` | `50` | Max pipeline iterations |
| `AIFACTORY_MAX_TOTAL_TOKENS` | `500000` | Max tokens per run |
| `AIFACTORY_SHARED_CONTEXT_MAX_KEYS` | `25` | Shared context pruning cap |

### Memory

| Variable | Default | Description |
|---|---|---|
| `AIFACTORY_MEMORY_PIPELINE_ENABLED` | `true` | Wire MemoryBridge |
| `AIFACTORY_MEMORY_CONTEXT_TOP_K` | `3` | Entries retrieved per think() |
| `AIFACTORY_MEMORY_CONTEXT_MAX_CHARS` | `800` | Max memory chars per think() |
| `AIFACTORY_MEMORY_PROMOTION_INTERVAL` | `10` | Agent → team promotion every N turns |

### Token Efficiency

| Variable | Default | Description |
|---|---|---|
| `AIFACTORY_PROMPT_CACHE_ENABLED` | `true` | Anthropic prompt caching |
| `AIFACTORY_PROMPT_CACHE_OPENROUTER` | `false` | Include OpenRouter Claude paths |
| `AIFACTORY_RESPONSE_CACHE_ENABLED` | `true` | LiteLLM response cache |
| `AIFACTORY_RESPONSE_CACHE_TTL` | `3600` | Response cache TTL (seconds) |
| `AIFACTORY_JUDGE_SKIP_SIMPLE_RUNS` | `true` | Skip judge LLM for single-task runs |
| `AIFACTORY_STREAM_TOKENS_ENABLED` | `true` | Token streaming to CLI/API |

### Semantic Cache

| Variable | Default | Description |
|---|---|---|
| `AIFACTORY_SEMANTIC_CACHE_ENABLED` | `false` | Enable semantic dedup |
| `AIFACTORY_SEMANTIC_CACHE_THRESHOLD` | `0.92` | Cosine similarity threshold |
| `AIFACTORY_SEMANTIC_CACHE_TTL_DAYS` | `7` | Cache entry TTL |
| `AIFACTORY_SEMANTIC_CACHE_MAX_SIZE` | `10000` | Max cached entries |

### Sandbox

| Variable | Default | Description |
|---|---|---|
| `AIFACTORY_SANDBOX_DOCKER_ENABLED` | `false` | Enable Docker MCP sandbox |
| `AIFACTORY_MCP_SANDBOX_URL` | — | MCP server URL (auto-selects HTTP vs stdio) |
| `E2B_API_KEY` | — | E2B cloud sandbox API key |

### RL Scoring

| Variable | Default | Description |
|---|---|---|
| `AIFACTORY_RL_ENABLED` | `true` | Enable RL signal |
| `AIFACTORY_RL_WEIGHT` | `0.2` | Max RL adjustment factor |
| `AIFACTORY_RL_MIN_SAMPLES` | `3` | Min samples before RL adjusts |

### Integrations

| Variable | Default | Description |
|---|---|---|
| `AIFACTORY_MCP_CATALOG_ENABLED` | `false` | Enable MCP tool catalog |
| `AIFACTORY_DEPLOY_MODE` | `local` | `local` (stdio) or `cloud` (HTTP pool) |
| `AIFACTORY_MCP_HTTP_POOL_SIZE` | `3` | HTTP sessions per MCP server |
| `COMPOSIO_API_KEY` | — | Composio API key |

### Judge Deep Eval

| Variable | Default | Description |
|---|---|---|
| `AIFACTORY_JUDGE_DEEP_EVAL_ENABLED` | `false` | Enable LLM deep eval |
| `AIFACTORY_JUDGE_EVAL_MODEL` | `openai/gpt-4o-mini` | Model for deep eval |

### Rate Limiting

| Variable | Default | Description |
|---|---|---|
| `AIFACTORY_RATE_LIMIT_ENABLED` | `true` | Enable rate limiting |
| `AIFACTORY_RATE_LIMIT_TASKS` | `10/minute` | POST /api/tasks |
| `AIFACTORY_RATE_LIMIT_READ` | `60/minute` | GET endpoints |
| `AIFACTORY_RATE_LIMIT_MEMORY` | `30/minute` | Memory endpoints |
| `AIFACTORY_RATE_LIMIT_WS` | `5/minute` | WebSocket connections |

---

## 15. Roadmap — Remaining Work

### Phase 5.1 (remaining)

- Circuit breakers on LLM calls (fail-fast after N consecutive errors)
- Budget hard caps enforcement (stop run when `MAX_BUDGET_USD` exceeded)
- Batch pipeline (submit N tasks, run in parallel, return N results)

### Phase 5.2 (planned)

- Frontend: full wiring to real backend (AgentTree, BudgetTracker live data)
- Multi-tenant auth (API keys per user, per-user rate limits)
- Postgres task graph state (replace in-memory LangGraph checkpointer)
- OpenTelemetry traces export alongside LangSmith
