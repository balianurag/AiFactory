# Experiments — Claude Context

Self-contained multi-agent experiment. Entry point: `uv run python experiments/main.py`

---

## Architecture

```
Human
  └── Leader               best_model(), mode-independent — primary human I/O
        ├── Judge           quality review after pipeline; flags issues to human
        └── Program Manager plan tasks + execute workers
                  ├── Researcher   if research_for_analysis=true → pm_replan
                  ├── Researcher   if research_needed=true
                  └── Worker × N   sequential, one per task, per-task model selection
```

LangGraph flow:
```
START → leader_receive → pm_plan
      → [researcher → pm_replan]?
      → [researcher]?
      → pm_execute → judge_review → leader_respond → END
```

---

## Key Files

| Path | Purpose |
|------|---------|
| `experiments/main.py` | Everything — agents, graph, nodes, routing, chat loop |
| `experiments/config/graph.yaml` | Fallback model, mode |
| `experiments/config/agents/*.yaml` | Per-agent role, max_tokens, ability profile, system_prompt |
| `experiments/config/models.yaml` | ~40 models rated 0–3 on 10 abilities |
| `experiments/config/tools.yaml` | Worker tools (python_exec, read_file, write_file, edit_file, glob_search, grep_search) |
| `experiments/config/researcher_tools.yaml` | Researcher-only tools (web_search, web_crawl) |
| `experiments/config/mcp_servers.yaml` | MCP server configs (empty by default — opt-in) |
| `experiments/.env` | `LITELLM_API_BASE` + `LITELLM_API_KEY` (gitignored) |
| `experiments/dumps/runs/` | JSON dump per run |
| `experiments/dumps/judge/` | Plain-text judge report per run |
| `experiments/dumps/graphs/` | Pipeline graph visual per run |
| `experiments/memories/facts.json` | LTM: persistent cross-run facts (REMEMBER: protocol) |
| `experiments/memories/research.json` | LTM: research cache (researcher-private) |

---

## Memory Architecture

```
Memory
  ├── stm: ShortTermMemory    in-memory, current run only
  │         seeded from LTM facts at startup; updated via SHARE: broadcasts
  └── ltm: LongTermMemory     persisted via MemoryBackend
            ├── facts          key-value facts — visible to all agents
            └── research       query→summary cache — researcher-private only
```

- **Backend**: `FilesystemBackend` (JSON files). Swap to Qdrant/Redis: `Memory(backend=QdrantBackend(...))`
- **Facts** accessible to: PM, workers, researcher, leader
- **Research cache** accessible to: researcher only, via `_do_research()` — the sole entry point
- **Research cache lookup order**: exact key → cosine similarity ≥0.82 → word-overlap ≥0.72

---

## Tool System

Three registries, two dispatch paths:

| Registry | File | Who uses it | Assignment |
|----------|------|-------------|------------|
| Worker tools | `config/tools.yaml` | Workers | PM assigns per-task tool IDs in pm_plan output |
| Researcher tools | `config/researcher_tools.yaml` | Researcher only | Always available — no assignment needed |
| MCP servers | `config/mcp_servers.yaml` | Workers (opt-in) | PM assigns `mcp__{server}__{tool}` IDs like any built-in tool |

**Worker tools**: `python_exec`, `read_file`, `write_file`, `edit_file`, `glob_search`, `grep_search`
**Researcher tools**: `web_search` (DuckDuckGo), `web_crawl` (httpx + BeautifulSoup)
**MCP tools**: any tool from any connected MCP server — prefixed `mcp__{server_id}__{tool_name}`

`ToolExecutor` builds LiteLLM function schemas and dispatches calls. Built-in tool names → Python functions. `mcp__*` names → `McpClient.call()`. `Agent.think()` runs the tool loop internally, always returns a string.

`McpClient` lifecycle: `await connect()` at startup (discovers all tools), `await close()` in `finally`. Servers configured in `mcp_servers.yaml` — servers with unresolved `${VAR}` tokens are skipped with a warning (safe to leave GitHub/GitLab entries when tokens aren't set).

### GitHub + GitLab setup

Add to `experiments/.env`:
```
GITHUB_TOKEN=ghp_...          # scopes: repo, pull_requests (read:org for org repos)
GITLAB_TOKEN=glpat-...        # scopes: api (read_repository + write_repository)
GITLAB_URL=https://gitlab.com # optional — omit for gitlab.com
```

Both servers are pre-configured in `mcp_servers.yaml`. They activate automatically when their token is present. Workers can then use tools like `mcp__github__create_pull_request`, `mcp__github__get_pull_request`, `mcp__gitlab__create_merge_request`, `mcp__gitlab__list_merge_requests`, etc.

---

## Model Selection

- **Leader**: always `best_model()` — top average quality, never affected by mode
- **Named agents** (judge, PM, researcher): `select_model(abilities, models, mode)` from YAML profile
- **Workers**: per-task ability requirements from PM plan output

Scoring: `score = Σ [ ((1-alpha)*task_req + alpha*mode_profile) * min(model_ability, max_quality) ]`

| Mode | alpha | max_quality | Effect |
|------|-------|-------------|--------|
| `best` | 1.0 | 3 | Always most capable |
| `balanced` | 0.35 | 2 | Task reqs dominate; cap equalises top models, cost/speed breaks tie |
| `cost_efficient` | 0.75 | 3 | Speed/cost bias; quality barely contributes |

---

## PM Output Format

```json
{"direct_answer": "",
 "facts": {},
 "research_for_analysis": false, "research_for_analysis_query": "",
 "research_needed": false, "research_query": "",
 "tasks": [{"id": "task_1", "description": "<actionable goal>",
             "reasoning": 2, "coding": 3, "math": 0, "writing": 1,
             "instruction_following": 2, "tool_use": 1, "knowledge": 1,
             "speed": 0, "cost_efficiency": 0, "context_handling": 2,
             "tools": [], "prompt": "<worker instructions>"}]}
```

- `direct_answer`: set for trivial/conversational queries — skips workers entirely
- `facts`: new facts PM extracted from user statements (skips keys already in `## Shared`)
- `research_needed`: set when tasks require current/external knowledge workers can't produce alone
- `tools`: worker tool IDs from `config/tools.yaml`; researcher tools are not assignable

---

## Worker Signals

Workers emit these on a single line in their output:

| Signal | Effect |
|--------|--------|
| `RESEARCH: <query>` | PM fetches inline via `_do_research()` — no user interaction |
| `QUESTION: <text>` | PM calls `interrupt()`, graph pauses, user answers, stored as `qa_N` in shared_context |
| `NOT_NEEDED` | Worker self-skips; excluded from task results |
| `REMEMBER:\nkey: value` | Facts saved to LTM and merged into STM + shared_context |
| `SHARE:\nkey: value` | Ephemeral facts merged into shared_context for downstream agents |

---

## Judge

- Runs after every pipeline (including direct-answer flows)
- Assesses each agent: `status` (successful/unsuccessful/not_required), `quality` (0–3 or null), `tokens`, `notes`
- **Critical rule**: quality score must be based strictly on actual output — empty output = quality 0 + unsuccessful
- `human_message`: non-empty if quality issue worth surfacing; shown before leader's final response
- Writes plain-text report to `dumps/judge/<run_id>.txt`

---

## PM Execute — Interrupt Resume

LangGraph re-runs the full node on every `Command(resume=value)`. Guards:

- `pm._exec_cache`: stores `{key, workers, completed}` keyed on task fingerprint — survives node replay
- `workers` dict: Agent objects by task ID; QUESTION-interrupted workers resume from cached output
- `completed` dict: finished workers skipped on resume

Interrupt detection: use `result.get("__interrupt__")` from `ainvoke` — NOT `snap.next` (unreliable in LangGraph 1.1.x).

---

## LiteLLM Proxy

- `litellm.ssl_verify = False`
- `_proxy_model(id)` — prepends `openai/` when `LITELLM_API_BASE` is set

---

## Optimisation Direction

**Cost and output quality. Speed is not a goal.**

1. Fewer tokens — fewer LLM calls, research cache hits, NOT_NEEDED skips, no consolidation step
2. Better quality — right model per task, sequential workers building on each other, judge flagging failures

---

## Potential Next Steps

- [ ] Judge retry — re-dispatch unsuccessful workers with corrective context
- [ ] Persistent memory (Qdrant backend for LTM)
- [ ] Multi-team support (multiple PMs for independent workstreams)
- [ ] Follow-up chat: leader detects "new task" vs "clarification" — new tasks re-invoke full pipeline
