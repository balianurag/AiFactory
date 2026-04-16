"""
Global settings for the AI Factory.

Loads from environment variables and .env file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # AiFactory/


class Settings(BaseSettings):
    """Central configuration — all values can be overridden via env vars."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_prefix="AIFACTORY_",
        extra="ignore",
    )

    # ── LLM API Keys ──────────────────────────────────────────────────
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    xai_api_key: str = Field(default="", alias="XAI_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    # OpenRouter — single key, access to 300+ models via openrouter.ai
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")

    # ── Model Tiers (OpenAI model strings) ─────────────────────────────
    premium_model: str = "gpt-4o"
    standard_model: str = "gpt-4o-mini"
    budget_model: str = "gpt-4o-mini"

    # ── PostgreSQL ─────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql://localhost:5432/aifactory",
        alias="DATABASE_URL",
    )

    # ── Redis ──────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # ── Qdrant ────────────────────────────────────────────────────────
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection_prefix: str = "aifactory"

    # ── Memory Settings ───────────────────────────────────────────────
    memory_promotion_interval: int = Field(
        default=10,
        description="Promote agent memory to team layer every N turns",
    )
    memory_max_agent_entries: int = 200
    memory_max_team_entries: int = 500
    memory_retrieval_top_k: int = 5
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # ── Token Efficiency (Phase 3.5) ─────────────────────────────────
    prompt_cache_enabled: bool = Field(
        default=True,
        description=(
            "Inject cache_control: ephemeral on system messages for Anthropic Claude models. "
            "60-90% cost reduction on repeat calls with the same system prompt. "
            "No-op for non-Anthropic models. "
            "Env: AIFACTORY_PROMPT_CACHE_ENABLED"
        ),
    )
    judge_skip_simple_runs: bool = Field(
        default=True,
        description=(
            "Skip the judge LLM call for single-task runs with no research. "
            "Saves ~800-1200 tokens on simple queries. "
            "A synthetic assessment is generated from run metadata instead. "
            "Env: AIFACTORY_JUDGE_SKIP_SIMPLE_RUNS"
        ),
    )
    shared_context_max_keys: int = Field(
        default=25,
        description=(
            "Cap shared_context dict at this many keys to prevent prompt bloat. "
            "Priority keys (objective, tasks_count, etc.) are always kept. "
            "Env: AIFACTORY_SHARED_CONTEXT_MAX_KEYS"
        ),
    )
    response_cache_enabled: bool = Field(
        default=True,
        description=(
            "Enable LiteLLM in-memory response cache for ALL providers and model tiers. "
            "Identical (model, messages) pairs are served from cache at 0 tokens — "
            "useful for retries, QUESTION/ANSWER loops, and repeated runs. "
            "Env: AIFACTORY_RESPONSE_CACHE_ENABLED"
        ),
    )
    response_cache_ttl: int = Field(
        default=3600,
        description=(
            "TTL in seconds for LiteLLM response cache entries (default 1 hour). "
            "Set to 0 to disable expiry. "
            "Env: AIFACTORY_RESPONSE_CACHE_TTL"
        ),
    )
    prompt_cache_openrouter: bool = Field(
        default=True,
        description=(
            "Extend Anthropic cache_control to OpenRouter-routed Claude models "
            "(e.g. openrouter/anthropic/claude-*). OpenRouter passes cache_control "
            "through to Anthropic, so savings still apply. "
            "Env: AIFACTORY_PROMPT_CACHE_OPENROUTER"
        ),
    )

    # ── Pipeline Memory (Phase 2.1) ───────────────────────────────────
    memory_pipeline_enabled: bool = Field(
        default=True,
        description=(
            "Wire 3-layer MemoryRouter into pipeline agents. "
            "Set false to disable (useful for tests or minimal deployments). "
            "Env: AIFACTORY_MEMORY_PIPELINE_ENABLED"
        ),
    )
    memory_context_top_k: int = Field(
        default=3,
        description=(
            "Max memory entries retrieved per agent think() call. "
            "Lower = faster + cheaper; higher = more context. "
            "Env: AIFACTORY_MEMORY_CONTEXT_TOP_K"
        ),
    )
    memory_context_max_chars: int = Field(
        default=800,
        description=(
            "Max chars of memory context injected into each think() system prompt. "
            "~800 chars ≈ 200 tokens. Increase for context-heavy agents. "
            "Env: AIFACTORY_MEMORY_CONTEXT_MAX_CHARS"
        ),
    )

    # ── LlamaIndex Memory Engine ──────────────────────────────────────
    llamaindex_embedding_provider: str = Field(
        default="litellm",
        description=(
            "'litellm' (default, routes via LiteLLM proxy — same path as all LLM calls), "
            "'huggingface' (local, no network), "
            "'openai' (direct OpenAI SDK, legacy), "
            "'stub' (hash-based, offline/dev)"
        ),
    )
    llamaindex_hf_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace model for local embeddings (384-dim)",
    )
    llamaindex_hf_dimensions: int = Field(
        default=384,
        description="Dimensions for the HuggingFace embedding model",
    )
    llamaindex_rerank_enabled: bool = Field(
        default=False,
        description="Enable BGE reranker for better retrieval quality",
    )
    llamaindex_rerank_model: str = Field(
        default="BAAI/bge-reranker-base",
        description="Reranker model name",
    )
    llamaindex_rerank_top_n: int = Field(
        default=3,
        description="Final top-N after reranking",
    )

    # ── Hierarchy ─────────────────────────────────────────────────────
    max_subordinates_per_lead: int = Field(
        default=3,
        description="The 1:N ratio — default 1:3",
    )
    max_project_managers: int = 5
    sr_pm_threshold: int = Field(
        default=2,
        description="Spawn Sr. PM when PM count exceeds this",
    )

    # ── Budget & Limits ───────────────────────────────────────────────
    max_budget_usd: float = Field(
        default=5.00,
        description=(
            "Hard USD cap per pipeline run. BudgetTracker raises BudgetExceededError "
            "when accumulated LLM cost reaches this value. "
            "Set to 0.0 to disable. Env: AIFACTORY_MAX_BUDGET_USD"
        ),
    )
    max_iterations: int = 50
    max_tokens_per_agent_turn: int = 2048
    max_total_tokens: int = 500_000

    # ── Retry & Circuit Breaker ───────────────────────────────────────
    llm_retry_max_attempts: int = Field(
        default=3,
        description=(
            "Max LLM call attempts before trying fallback model. "
            "Env: AIFACTORY_LLM_RETRY_MAX_ATTEMPTS"
        ),
    )
    llm_retry_base_delay_s: float = Field(
        default=1.0,
        description=(
            "Base delay (seconds) for exponential backoff on LLM failures. "
            "Delays: base, base×3, base×9. "
            "Env: AIFACTORY_LLM_RETRY_BASE_DELAY_S"
        ),
    )
    circuit_breaker_threshold: int = Field(
        default=3,
        description=(
            "Consecutive failures before circuit breaker opens for a model. "
            "When open, calls are rerouted to fallback_model immediately. "
            "Env: AIFACTORY_CIRCUIT_BREAKER_THRESHOLD"
        ),
    )
    tool_retry_max_attempts: int = Field(
        default=3,
        description=(
            "Max retry attempts for Composio and MCP tool calls on 429 or failure. "
            "Env: AIFACTORY_TOOL_RETRY_MAX_ATTEMPTS"
        ),
    )
    circuit_breaker_ttl_s: int = Field(
        default=3600,
        description=(
            "Seconds before a tripped circuit breaker auto-resets. "
            "Persisted to ~/.aifactory/circuit_state.json so failures survive "
            "process restarts. Set 0 to never auto-reset. "
            "Env: AIFACTORY_CIRCUIT_BREAKER_TTL_S"
        ),
    )
    sandbox_timeout_coding_s: int = Field(
        default=90,
        description=(
            "Sandbox execution timeout for coding/math tasks (seconds). "
            "Longer than the default (30s) to handle compile+run cycles. "
            "Env: AIFACTORY_SANDBOX_TIMEOUT_CODING_S"
        ),
    )

    # ── LangSmith Observability ─────────────────────────────────────
    langchain_api_key: str = Field(default="", alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="aifactory", alias="LANGCHAIN_PROJECT")
    langchain_tracing: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com", alias="LANGCHAIN_ENDPOINT"
    )

    # ── Logging ───────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Rate Limiting ─────────────────────────────────────────────────
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable/disable API rate limiting",
    )
    rate_limit_tasks: str = Field(
        default="10/minute",
        description="Rate limit for POST /api/tasks (pipeline submissions)",
    )
    rate_limit_read: str = Field(
        default="60/minute",
        description="Rate limit for GET endpoints (health, models, tasks, observability)",
    )
    rate_limit_memory: str = Field(
        default="30/minute",
        description="Rate limit for memory endpoints (store, retrieve, promote, clear)",
    )
    rate_limit_ws: str = Field(
        default="5/minute",
        description="Rate limit for WebSocket connections",
    )

    # ── E2B Sandbox ──────────────────────────────────────────────────
    e2b_api_key: str = Field(default="", alias="E2B_API_KEY")
    sandbox_timeout: int = Field(
        default=30,
        description="Default code execution timeout in seconds",
    )
    sandbox_memory_mb: int = Field(
        default=256,
        description="Default memory limit for local sandbox in MB",
    )
    sandbox_backend: str = Field(
        default="auto",
        description="'docker' | 'e2b' | 'local' | 'auto' (docker > e2b > local priority)",
    )

    # ── Docker MCP Sandbox ────────────────────────────────────────────────
    sandbox_docker_enabled: bool = Field(
        default=False,
        description=(
            "Enable Docker container pool as the primary sandbox backend. "
            "Requires Docker daemon + aifactory-sandbox image. "
            "Env: AIFACTORY_SANDBOX_DOCKER_ENABLED"
        ),
    )
    sandbox_docker_image: str = Field(
        default="aifactory-sandbox:latest",
        description="Docker image for sandbox containers. Env: AIFACTORY_SANDBOX_DOCKER_IMAGE",
    )
    sandbox_docker_memory: str = Field(
        default="512m",
        description="Memory limit per container (512m, 1g). Env: AIFACTORY_SANDBOX_DOCKER_MEMORY",
    )
    sandbox_docker_cpus: float = Field(
        default=0.5,
        description="CPU quota per container (0.5 = half a core). Env: AIFACTORY_SANDBOX_DOCKER_CPUS",  # noqa: E501
    )
    sandbox_docker_pids_limit: int = Field(
        default=64,
        description="Max PIDs per container (prevents fork bombs). Env: AIFACTORY_SANDBOX_DOCKER_PIDS_LIMIT",  # noqa: E501
    )
    sandbox_gvisor_enabled: bool = Field(
        default=False,
        description=(
            "Use gVisor (runsc) runtime — kernel-level isolation via syscall interception. "
            "Requires gVisor installed on host. Falls back to runc silently. "
            "Env: AIFACTORY_SANDBOX_GVISOR_ENABLED"
        ),
    )
    sandbox_pool_min_idle: int = Field(
        default=4,
        description="Warm containers kept ready in pool. Env: AIFACTORY_SANDBOX_POOL_MIN_IDLE",
    )
    sandbox_pool_max_size: int = Field(
        default=16,
        description="Max total containers (ready + in-use). Env: AIFACTORY_SANDBOX_POOL_MAX_SIZE",
    )
    sandbox_pool_max_reuse: int = Field(
        default=50,
        description="Retire container after N executions. Env: AIFACTORY_SANDBOX_POOL_MAX_REUSE",
    )
    sandbox_pool_acquire_timeout: float = Field(
        default=10.0,
        description="Seconds to wait for a free container. Env: AIFACTORY_SANDBOX_POOL_ACQUIRE_TIMEOUT",  # noqa: E501
    )
    sandbox_max_output_bytes: int = Field(
        default=50_000,
        description="Max stdout+stderr bytes per execution. Env: AIFACTORY_SANDBOX_MAX_OUTPUT_BYTES",  # noqa: E501
    )
    sandbox_max_executions_per_worker: int = Field(
        default=10,
        description="Hard cap on code executions per worker per run. Env: AIFACTORY_SANDBOX_MAX_EXECUTIONS_PER_WORKER",  # noqa: E501
    )
    sandbox_seccomp_profile: str = Field(
        default="docker/sandbox/seccomp.json",
        description="Path to seccomp JSON profile. Empty = Docker default. Env: AIFACTORY_SANDBOX_SECCOMP_PROFILE",  # noqa: E501
    )
    sandbox_mcp_url: str = Field(
        default="",
        description=(
            "HTTP URL of MCP sandbox server root (e.g. http://localhost:8100). "
            "The /mcp endpoint is appended automatically. "
            "Empty = stdio subprocess mode. Env: AIFACTORY_MCP_SANDBOX_URL"
        ),
    )
    sandbox_mcp_auth_token: str = Field(
        default="",
        alias="MCP_AUTH_TOKEN",
        description="Bearer token for MCP sandbox HTTP auth. Env: MCP_AUTH_TOKEN",
    )

    # ── Integrations (Composio) ──────────────────────────────────────────
    composio_api_key: str = Field(default="", alias="COMPOSIO_API_KEY")
    composio_default_user_id: str = Field(
        default="aifactory-deployer",
        description="Default Composio user_id for tool execution",
    )
    composio_enabled: bool = Field(
        default=True,
        description="Auto-connect Composio on startup when API key is set",
    )

    # ── RL Sidecar ────────────────────────────────────────────────────────────
    rl_min_score_for_promotion: float = Field(
        default=6.0,
        description="Minimum composite score to promote memory to global LTM",
    )
    rl_judge_model: str = "gpt-4o-mini"

    # ── Deployment Mode ───────────────────────────────────────────────────────
    deploy_mode: str = Field(
        default="local",
        description=(
            "Deployment context: 'local' or 'cloud'. "
            "Controls MCP server transport strategy: "
            "'local' = stdio subprocess (Docker/uvx/npx commands from mcp_catalog.yaml); "
            "'cloud' = HTTP session pool (persistent MCP server services, zero subprocess overhead). "  # noqa: E501
            "Env: AIFACTORY_DEPLOY_MODE"
        ),
    )

    # ── MCP Catalog ───────────────────────────────────────────────────────────
    mcp_catalog_enabled: bool = Field(
        default=False,
        description="Enable MCP catalog provider (direct MCP server connections) alongside Composio",  # noqa: E501
    )
    mcp_http_pool_size: int = Field(
        default=3,
        description=(
            "Number of persistent HTTP sessions per MCP server in cloud mode. "
            "Higher = more concurrent pipeline runs supported, more memory used. "
            "Only active when AIFACTORY_DEPLOY_MODE=cloud. "
            "Env: AIFACTORY_MCP_HTTP_POOL_SIZE"
        ),
    )

    # ── OpenTelemetry (D2) ───────────────────────────────────────────────────
    otel_enabled: bool = Field(
        default=False,
        description=(
            "Enable OpenTelemetry distributed tracing. "
            "Requires opentelemetry-sdk + opentelemetry-exporter-otlp-proto-http. "
            "Env: AIFACTORY_OTEL_ENABLED"
        ),
    )
    otel_endpoint: str = Field(
        default="",
        description=(
            "OTLP HTTP endpoint for span export (e.g. http://localhost:4318). "
            "Empty = no export (spans recorded in memory only). "
            "Env: AIFACTORY_OTEL_ENDPOINT"
        ),
    )
    otel_service_name: str = Field(
        default="aifactory",
        description="Service name reported in OTel traces. Env: AIFACTORY_OTEL_SERVICE_NAME",
    )

    # ── RL-lite Adaptive Model Selection ─────────────────────────────────────
    rl_enabled: bool = Field(
        default=True,
        description="Enable RL-lite adaptive model selection from historical scores",
    )
    rl_weight: float = Field(
        default=0.2,
        description=(
            "Blend factor for RL signal in model scoring. "
            "0.0 = pure static ability scores, 1.0 = RL-dominated. "
            "0.2 means ±20% adjustment over the static score."
        ),
    )
    rl_min_samples: int = Field(
        default=3,
        description="Minimum performance records before RL influences model selection",
    )

    # ── Embedding Cache (Redis-backed, reduces repeat embedding API costs) ─────
    embed_cache_enabled: bool = Field(
        default=True,
        description=(
            "Cache embedding vectors in Redis to avoid repeat API calls. "
            "Same text+model pair served at 0 cost from cache. "
            "No-op when REDIS_URL is not set. "
            "Env: AIFACTORY_EMBED_CACHE_ENABLED"
        ),
    )
    embed_cache_ttl: int = Field(
        default=604_800,  # 7 days
        description=(
            "TTL in seconds for cached embedding vectors (default 7 days). "
            "Set 0 for no expiry. "
            "Env: AIFACTORY_EMBED_CACHE_TTL"
        ),
    )

    # ── Semantic Query Cache (Qdrant-backed, intent-level deduplication) ───────
    semantic_cache_enabled: bool = Field(
        default=True,
        description=(
            "Enable semantic similarity cache for pipeline queries. "
            "Queries with the same intent return cached results at 0 LLM cost. "
            "Requires Qdrant (in-memory fallback works but doesn't persist). "
            "Env: AIFACTORY_SEMANTIC_CACHE_ENABLED"
        ),
    )
    semantic_cache_threshold: float = Field(
        default=0.92,
        description=(
            "Minimum cosine similarity (0–1) to accept a cached result. "
            "Higher = stricter matching, fewer false positives. "
            "0.92 = near-identical phrasing; 0.85 = paraphrases. "
            "Env: AIFACTORY_SEMANTIC_CACHE_THRESHOLD"
        ),
    )
    semantic_cache_ttl_days: int = Field(
        default=7,
        description=(
            "Days before a semantic cache entry expires. "
            "Set 0 for no expiry (cache grows unbounded). "
            "Env: AIFACTORY_SEMANTIC_CACHE_TTL_DAYS"
        ),
    )
    semantic_cache_max_size: int = Field(
        default=10_000,
        description=(
            "Max number of entries in the semantic cache. "
            "Oldest entries are pruned when this is exceeded. "
            "Env: AIFACTORY_SEMANTIC_CACHE_MAX_SIZE"
        ),
    )

    # ── Judge Deep Evaluation (LLM-as-judge for output quality) ───────────────
    judge_deep_eval_enabled: bool = Field(
        default=False,
        description=(
            "Run an additional LLM-as-judge call to score the actual final_response "
            "on relevance, accuracy, completeness, and actionability (0–10 each). "
            "Produces a more accurate composite_quality than metadata-only scoring. "
            "Costs ~200–400 tokens extra per run. "
            "Env: AIFACTORY_JUDGE_DEEP_EVAL_ENABLED"
        ),
    )
    judge_eval_model: str = Field(
        default="",
        description=(
            "Model to use for deep output evaluation (default: cost_efficient model). "
            "Recommend a cheap/fast model: gpt-4o-mini, claude-haiku-4-5-20251001. "
            "Empty = auto-select cheapest available. "
            "Env: AIFACTORY_JUDGE_EVAL_MODEL"
        ),
    )

    # ── Token Streaming ────────────────────────────────────────────────────────
    stream_tokens_enabled: bool = Field(
        default=True,
        description=(
            "Stream individual LLM tokens to the CLI as they are generated, "
            "instead of waiting for the full node response. "
            "Env: AIFACTORY_STREAM_TOKENS_ENABLED"
        ),
    )

    # ── Redis Task Store ───────────────────────────────────────────────────────
    task_store_enabled: bool = Field(
        default=True,
        description=(
            "Write run status and results to Redis for cross-process aifactory status. "
            "No-op when REDIS_URL is not set. "
            "Env: AIFACTORY_TASK_STORE_ENABLED"
        ),
    )
    task_store_ttl_days: int = Field(
        default=7,
        description=("Days to retain run records in Redis. Env: AIFACTORY_TASK_STORE_TTL_DAYS"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings loader."""
    return Settings()
