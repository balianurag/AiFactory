# CHANGES

## Branch: `feat/reducers-checkpointing-milvus` — Phase 6.1 production hardening

Three additive, feature-flagged upgrades to the pipeline runtime. Every
change defaults **off** so the existing test suite passes unchanged and
current deployments see zero behavioural drift until a flag is flipped.

### What's in this branch

| Change | Files (new) | Files (touched) | Default |
|---|---|---|---|
| Postgres LangGraph checkpointer | `src/aifactory/graphs/checkpointer.py`, `tests/test_postgres_checkpoint.py` | `src/aifactory/graphs/graph_compiler.py`, `src/aifactory/config/settings.py` | off (`memory`) |
| Reducer state + Send-API fan-out | `src/aifactory/graphs/state_v2.py`, `src/aifactory/graphs/worker_node.py`, `src/aifactory/graphs/planner_node.py`, `tests/test_reducers_state.py` | `src/aifactory/graphs/graph_compiler.py`, `src/aifactory/config/settings.py` | off |
| Tiered vector memory (Qdrant/Milvus) | `src/aifactory/core/milvus_store.py`, `tests/test_milvus_tier.py` | `src/aifactory/core/memory_router.py`, `src/aifactory/config/settings.py` | off |

New deps: `langgraph-checkpoint-postgres`, `pymilvus`, `milvus-lite`
(the last two are under the `milvus` optional extra).

### Phased rollout (staging)

Enable one flag at a time; each is independent.

**Phase A — Postgres checkpointer**
```bash
export AIFACTORY_PIPELINE_CHECKPOINTER=postgres
export DATABASE_URL=postgresql://user:pass@host:5432/aifactory  # pragma: allowlist secret
make dev  # or: uv run uvicorn aifactory.api.main:app
```
Pipelines survive process restarts. Falls back to `MemorySaver` + a warning if
the DB is unreachable — no crash.

**Phase B — Reducer state + Send-API fan-out**
```bash
export AIFACTORY_STATE_REDUCERS_ENABLED=true
make run TASK="…" MODE=balanced
```
STANDARD / FULL / EXTENDED topologies swap `execution_lead` for the
`planner` + `worker` pair. Parallel worker writes merge via reducers
instead of last-writer-wins. MINIMAL topology is unaffected.

**Phase C — Tiered vector memory**
```bash
export AIFACTORY_VECTOR_TIERED_ENABLED=true
export AIFACTORY_MILVUS_URI=./milvus.db          # or milvus://host:19530
export AIFACTORY_MILVUS_COLD_LEVELS=global       # optional; default 'global'
make memory-stats
```
Cold-tier operations route to Milvus; hot tiers stay on Qdrant. Every
Milvus call is fail-soft: any error logs a warning and the router falls
back to Qdrant for the rest of the process.

### Rollback

Unset the env var (or set it back to its default) and restart the
process. There is nothing to un-migrate at the code level; existing
Qdrant data and legacy state schemas remain valid.
