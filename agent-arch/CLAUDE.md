# Experiments — Claude Context

Self-contained multi-agent experiment. Entry point: `uv run python experiments/main.py`

---

## Architecture

```
Human
  └── Master Agent          always uses best_model(), mode-independent, hand-written prompt
        └── Judge            monitors pipeline, reviews every agent post-run
              └── Analyser   breaks query into atomic tasks, flags research needs
                    ├── Researcher (pre-analysis)   if research_for_analysis=true
                    │     └── Reanalyse             second analyser pass with research
                    │           └── Researcher (execution)   if research_needed=true after reanalyse
                    ├── Researcher (execution)       if research_needed=true (no pre-analysis needed)
                    └── Execution Lead              plans worker dispatch, selects models, consolidates
                          └── Worker-task_N × N     one per atomic task, model selected per task
```

LangGraph flow:
```
START → master_receive → judge_brief → analyse
      → [researcher → reanalyse]?   (pre-analysis research loop)
      → [researcher]?               (execution research)
      → execution_lead
      → judge_review
      → master_respond → END
```

---

## Key Files

| Path | Purpose |
|------|---------|
| `experiments/main.py` | Everything — agents, graph, nodes, routing, chat loop |
| `experiments/config/graph.yaml` | Runtime config: fallback model, mode, mode_alpha |
| `experiments/config/agents/*.yaml` | Per-agent role, task, max_tokens, ability profile |
| `experiments/config/models.yaml` | ~40 models rated 0–3 on 10 abilities |
| `experiments/.env` | `LITELLM_API_BASE` + `LITELLM_API_KEY` (corporate proxy, gitignored) |
| `experiments/dumps/runs/` | JSON dump per run |
| `experiments/dumps/judge/` | Plain-text judge report per run |

---

## Config

### `config/graph.yaml`
```yaml
model: us.anthropic.claude-3-haiku-20240307-v1:0   # fallback if model selection fails
mode: balanced                                       # best | balanced | cost_efficient
# mode_alpha removed — alpha is baked into each mode profile in MODE_PROFILES
```

### `config/models.yaml`
~40 models (Claude, GPT, Gemini, DeepSeek, Llama, Mistral, Qwen, Perplexity, etc.) each rated on 10 abilities using a **semantic 0–3 scale**:
- `0` = bad at this / not applicable
- `1` = can do it
- `2` = good at it
- `3` = will crush it

Abilities: `reasoning, coding, math, writing, instruction_following, tool_use, knowledge, speed, cost_efficiency, context_handling`

To add a model: append a new entry with all 10 ability scores (0–3).

---

## Model Selection Algorithm

- **Master agent**: always `best_model()` — highest average quality score across 8 quality abilities (speed/cost excluded). Never affected by mode.
- **All other named agents** (judge, analyser, researcher, execution_lead): `select_model(agent_abilities, models, mode)` using the agent's own ability profile from its YAML.
- **Workers**: execution lead's planning call outputs per-task ability requirements. Python scores all models and picks highest. Also affected by mode.
- **Alpha** is no longer a separate parameter — it is baked into `MODE_PROFILES`: `best=1.0`, `balanced=0.35`, `cost_efficient=0.75`.

**Scoring formula:**
```
score = Σ [ ((1 - alpha) × task_requirement_i  +  alpha × mode_profile_i) × model_ability_i ]
```

---

## Structured Output Format

All structured agent output uses **JSON**. Models are asked to output JSON only (no prose, no markdown fences). Code fences are stripped before parsing:
```python
clean = re.sub(r"```[a-z]*\n?", "", raw).replace("```", "").strip()
parsed = json.loads(clean)
```

### Agent output formats

**Analyser:**
```json
{"research_for_analysis": false, "research_for_analysis_query": "", "research_needed": false, "research_query": "", "tasks": [{"id": "task_1", "description": "<actionable goal>"}]}
```

**Execution Lead (plan + prompts):**
```json
{"plan": [{"id": "task_1", "reasoning": 2, "coding": 3, "math": 0, "writing": 1, "instruction_following": 2, "tool_use": 1, "knowledge": 1, "speed": 0, "cost_efficiency": 0, "context_handling": 2}], "prompts": {"task_1": "<complete worker spec>"}}
```

**Judge:**
```json
{"assessments": [{"agent": "master", "status": "successful", "quality": 3, "tokens": 151, "notes": "clear routing"}], "metrics": {"overall_status": "successful", "total_tokens": 437, "pipeline_efficiency": "high", "cost_rating": 2, "recommendations": "none"}}
```

---

## Agent Prompts

- `master.yaml` — hand-written `system_prompt`.
- All named agents (judge, analyser, researcher, execution_lead) — static `system_prompt` field in their YAML. No LLM generation at startup.
- Workers — prompts generated per-task by the execution lead's JSON plan (`prompts` key). If the plan parse fails, workers fall back to a minimal prompt that still includes the QUESTION instruction.

**Key constraints embedded in static prompts:**
- Analyser: task descriptions must be **actionable goals**, never completed facts or methods. `direct_answer` field used for conversational/simple queries.
- Execution Lead: worker prompts must ask for plain prose output — never JSON.
- Judge: outputs JSON only with exact format including assessments + metrics.

---

## LiteLLM Proxy

All LLM calls go through a corporate LiteLLM proxy (`LITELLM_API_BASE` in `.env`).

- `litellm.ssl_verify = False` — internal CA not trusted by default
- `_proxy_model(model_id)` — prepends `openai/` to all model IDs when proxy is configured, forcing `/v1/chat/completions` routing regardless of model name format

---

## Shared Context (`shared_context` dict)

Injected into every agent's system prompt as `## Shared\nkey: value\n...`

Any agent can broadcast facts by ending their response with:
```
SHARE:
key: value
```
Parsed as TOON, merged into shared_context, visible to all subsequent agents.

---

## Research Flows

| Flag | When | What it does |
|------|------|-------------|
| `research_for_analysis` | Analyser needs context to decompose properly | researcher → reanalyse (second analyser pass) |
| `research_needed` | Workers need external knowledge | researcher → execution_lead |

Both can be true. Reanalyse clears `research_for_analysis` and may set `research_needed`.

---

## Worker Questions

Workers output `QUESTION: <text>` (single line) if they need clarification.
Execution lead detects this, calls `interrupt()`, pauses the graph.
Main loop surfaces the question, reads user answer, resumes with `Command(resume=answer)`.
Answer stored in shared_context as `qa_N: Q:...|A:...` — subsequent workers see it.

---

## Judge

Two graph nodes:

- **judge_brief**: short internal call before pipeline — receives actual agent names from registry, states expected agents and success criteria. Passing real names prevents hallucination of fictional agents.
- **judge_review**: after execution_lead — receives run summary (per-agent tokens, elapsed, task results, research flags), outputs TOON assessments + metrics.

Dumps a plain-text hierarchical report to `dumps/judge/<run_id>.txt` with columns:
`agent | model | tokens | time | status | quality`

Status symbols: `✓` = successful, `✗` = unsuccessful, `—` = not required / unassessed.

---

## Timing

Each `Agent.think()` call records `elapsed_s` per action (measured before/after `litellm.acompletion`).
`_elapsed(agent)` sums all per-action durations — correct even for single-call agents.

---

## Dump Locations

| What | Where |
|------|-------|
| Full JSON run dump | `dumps/runs/<run_id>.json` |
| Judge plain-text report | `dumps/judge/<run_id>.txt` |

---

## Chat Mode

Graph runs once (full pipeline). After master_respond completes, main() enters a direct chat loop with the master agent (which retains its private `_messages` context). Type `quit / exit / done / bye / q` to end.

---

## What's Working

- [x] Full pipeline end-to-end (master → judge → analyser → researcher → execution_lead → workers → judge review → master respond)
- [x] JSON structured output throughout — code fences stripped before `json.loads()`, fallback on parse failure
- [x] Static system prompts in agent YAMLs — no LLM generation at startup, zero extra cost per run
- [x] Analyser task descriptions enforced as actionable goals, not completed facts or methods; `direct_answer` field bypasses pipeline for conversational queries
- [x] Analyser YAML forbids "ask user" tasks — workers ask via QUESTION: themselves when they discover missing info
- [x] Execution lead YAML instructs worker prompts to use QUESTION: for user-input tasks and NOT_NEEDED for conditional skip
- [x] Worker fallback prompt includes QUESTION and NOT_NEEDED instructions (appended in Python unconditionally)
- [x] Judge brief receives real agent names — no hallucinated fictional agents
- [x] Execution lead consolidation returns plain text (not JSON)
- [x] Dynamic model selection per agent and per worker task (dot-product scoring, 0–3 semantic ability scale)
- [x] Mode bias unified — alpha baked into each mode profile (`best=1.0`, `balanced=0.35`, `cost_efficient=0.75`); `mode_alpha` config removed
- [x] Pre-analysis research loop (analyser flags → researcher → reanalyse)
- [x] Shared context + SHARE: broadcast mechanism
- [x] Worker QUESTION: interrupt flow with user answer injected into shared_context as `qa_N`
- [x] Plan cache on execution lead agent object (`lead._plan_cache`) — survives LangGraph node replay on resume; prevents duplicate planning call
- [x] Worker cache (`workers_cache`) — agents reused across resume; QUESTION-interrupted workers resume from cached output, not re-called
- [x] Completed task cache (`completed` dict in plan cache) — finished workers skipped on resume
- [x] NOT_NEEDED self-skip — workers output exactly `NOT_NEEDED` for conditional tasks; excluded from consolidation
- [x] Full prior task results injected into each worker (no truncation) — conditional workers can self-gate correctly
- [x] Partial dump written on early quit — `app.get_state()` called before exit
- [x] Agent Roster printed after model selection with actual model names (not fallback)
- [x] Empty final_response guarded — warning shown instead of empty panel
- [x] Judge brief + review with per-agent assessments + pipeline metrics
- [x] Hierarchical plain-text judge report with status/quality columns
- [x] Per-action elapsed time tracking (correct for single-call agents)
- [x] Separate dump directories for runs vs judge reports
- [x] LiteLLM proxy integration (ssl_verify=False, openai/ prefix routing)
- [x] Interactive chat with master after pipeline completes

## Model Selection

- **`mode_alpha` removed from `config/graph.yaml`** — alpha is now baked into each mode profile in `MODE_PROFILES`:
  - `best`: alpha=1.0 — always picks the most capable model regardless of task requirements
  - `balanced`: alpha=0.35 — task requirements dominate, mode nudges toward quality
  - `cost_efficient`: alpha=0.75 — strongly biases toward speed/cheap models
- Scoring: `score = Σ [ ((1-alpha)*task_req + alpha*mode_profile) * model_ability ]`

## Execution Lead — Duplicate Prevention

LangGraph re-runs the entire node on every `Command(resume=value)`. To avoid re-work:

- **Plan cache**: `lead._plan_cache` on the Agent object stores the planning result keyed on `tuple((t["id"], t["description"]) for t in tasks)`. On replay, if key matches, the plan is reused — no LLM call.
- **Worker cache**: `workers_cache` stores Agent objects by task ID. If a cached worker's last action was a QUESTION, the cached output is used directly on resume (no re-call).
- **Completed cache**: `completed` dict tracks finished tasks. Already-done workers are skipped with a log line.

## Potential Next Steps

- [ ] Judge retry logic — if a worker is marked unsuccessful, re-dispatch
- [ ] Parallel worker execution (currently sequential)
- [ ] Persistent shared memory across runs (Qdrant / LlamaIndex)
- [ ] Multi-team support (multiple execution leads)
- [ ] Judge assessment fed back into next run as context
- [ ] Structured A2A worker output (STATUS/FINDINGS envelope) for richer inter-worker communication
