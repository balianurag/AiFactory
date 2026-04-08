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

    # ── LlamaIndex Memory Engine ──────────────────────────────────────
    llamaindex_embedding_provider: str = Field(
        default="huggingface",
        description="'huggingface' (local, free) or 'openai' (API, paid)",
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
    max_budget_usd: float = 5.00
    max_iterations: int = 50
    max_tokens_per_agent_turn: int = 2048
    max_total_tokens: int = 500_000

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
        description="'e2b' | 'local' | 'auto' (e2b if key set, else local)",
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings loader."""
    return Settings()
