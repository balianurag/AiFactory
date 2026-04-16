"""
Embedding Provider — Auto-selection, Redis caching, and graceful fallback.

Selection priority (checked at init time via env vars, no network probe):
  1. OPENAI_API_KEY         → text-embedding-3-small  (1536-dim, $0.02/1M)
  2. VOYAGE_API_KEY         → voyage/voyage-3          (1024-dim, best quality)
  3. COHERE_API_KEY         → cohere/embed-english-v3.0 (1024-dim)
  4. LITELLM_API_BASE set   → settings.embedding_model via custom proxy
  5. HuggingFace available  → BAAI/bge-small-en-v1.5   (384-dim, local, free)
  6. Stub (SHA-256 hash)    → 1536-dim, logs a loud warning, NO semantic search

Redis embedding cache:
  Key:  aifactory:emb:{sha256(model_id + ":" + text)[:24]}
  TTL:  AIFACTORY_EMBED_CACHE_TTL (default 7 days = 604800 s)
  Cost: cached embeddings cost $0 — repeated pipeline runs on same content are free.

Usage:
    from aifactory.core.embed_provider import build_embed_fn
    embed_fn, dimensions = build_embed_fn()     # callable(text) → list[float]
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Module-level cache so provider detection only runs once per process
_RESOLVED: tuple[str, str, int] | None = None
_REDIS_CLIENT: Any = None
_REDIS_PROBED: bool = False


# ─── Provider Resolution ──────────────────────────────────────────────────────


def resolve_provider() -> tuple[str, str, int]:
    """
    Determine the best embedding provider from available API keys.
    Returns (provider_name, model_id, dimensions).

    Checks env vars without making any network calls — deterministic and instant.
    Honour explicit AIFACTORY_LLAMAINDEX_EMBEDDING_PROVIDER overrides first.
    """
    global _RESOLVED
    if _RESOLVED is not None:
        return _RESOLVED

    try:
        from aifactory.config.settings import get_settings

        s = get_settings()
    except Exception:
        _RESOLVED = ("stub", "stub", 1536)
        return _RESOLVED

    explicit = s.llamaindex_embedding_provider

    # ── Explicit non-auto provider ─────────────────────────────────────
    if explicit not in ("litellm", "auto"):
        if explicit == "huggingface":
            _RESOLVED = ("huggingface", s.llamaindex_hf_model, s.llamaindex_hf_dimensions)
        elif explicit == "stub":
            _RESOLVED = ("stub", "stub", 1536)
        else:
            # openai / custom litellm
            _RESOLVED = ("litellm", s.embedding_model, s.embedding_dimensions)
        logger.debug("Embed provider: explicit override → %s (%s)", _RESOLVED[0], _RESOLVED[1])
        return _RESOLVED

    # ── Custom LiteLLM proxy — assume it supports embeddings ──────────
    if os.getenv("LITELLM_API_BASE", "").strip():
        _RESOLVED = ("litellm", s.embedding_model, s.embedding_dimensions)
        logger.debug("Embed provider: custom LiteLLM proxy → %s", s.embedding_model)
        return _RESOLVED

    # ── OpenAI key (most common, excellent default) ───────────────────
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key and not openai_key.startswith(("sk-your", "your-key", "CHANGE")):
        _RESOLVED = ("litellm", "text-embedding-3-small", 1536)
        logger.debug("Embed provider: text-embedding-3-small (OpenAI key detected)")
        return _RESOLVED

    # ── Voyage AI (high quality, Anthropic-owned) ─────────────────────
    voyage_key = os.getenv("VOYAGE_API_KEY", "").strip()
    if voyage_key:
        _RESOLVED = ("litellm", "voyage/voyage-3", 1024)
        logger.debug("Embed provider: voyage/voyage-3 (Voyage key detected)")
        return _RESOLVED

    # ── Cohere ────────────────────────────────────────────────────────
    cohere_key = os.getenv("COHERE_API_KEY", "").strip()
    if cohere_key:
        _RESOLVED = ("litellm", "cohere/embed-english-v3.0", 1024)
        logger.debug("Embed provider: cohere/embed-english-v3.0 (Cohere key detected)")
        return _RESOLVED

    # ── Local HuggingFace (no API key needed) ─────────────────────────
    try:
        import llama_index.embeddings.huggingface  # noqa: F401

        _RESOLVED = ("huggingface", s.llamaindex_hf_model, s.llamaindex_hf_dimensions)
        logger.info(
            "Embed provider: HuggingFace %s (local, free, no API key)",
            s.llamaindex_hf_model,
        )
        return _RESOLVED
    except ImportError:
        pass

    # ── Stub: last resort — NO semantic search ─────────────────────────
    logger.warning(
        "\n"
        "╔══════════════════════════════════════════════════════════════════╗\n"
        "║  ⚠  STUB EMBEDDINGS ACTIVE — memory search is NON-SEMANTIC      ║\n"
        "║  Stored memories will NOT be retrieved by relevance.             ║\n"
        "║  Set one of these to enable real semantic search:                ║\n"
        "║    OPENAI_API_KEY   → text-embedding-3-small (recommended)       ║\n"
        "║    VOYAGE_API_KEY   → voyage/voyage-3 (high quality)             ║\n"
        "║    COHERE_API_KEY   → embed-english-v3.0                         ║\n"
        "║  Or: AIFACTORY_LLAMAINDEX_EMBEDDING_PROVIDER=huggingface (free)  ║\n"
        "╚══════════════════════════════════════════════════════════════════╝"
    )
    _RESOLVED = ("stub", "stub", 1536)
    return _RESOLVED


def clear_provider_cache() -> None:
    """Clear the cached provider resolution (useful in tests to force re-detection)."""
    global _RESOLVED
    _RESOLVED = None


# ─── Redis Embedding Cache ────────────────────────────────────────────────────


def _get_redis() -> Any:
    """Return a sync Redis client for embedding cache, or None if unavailable."""
    global _REDIS_CLIENT, _REDIS_PROBED
    if _REDIS_PROBED:
        return _REDIS_CLIENT
    _REDIS_PROBED = True

    try:
        from aifactory.config.settings import get_settings

        if not get_settings().embed_cache_enabled:
            return None
    except Exception:
        pass

    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        return None

    try:
        import redis as _redis_sync

        client = _redis_sync.from_url(
            redis_url,
            decode_responses=False,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        client.ping()
        _REDIS_CLIENT = client
        logger.debug("Embedding Redis cache: connected to %s", redis_url)
    except Exception as exc:
        logger.debug("Embedding Redis cache: unavailable (%s)", exc)

    return _REDIS_CLIENT


def _cache_key(model_id: str, text: str) -> str:
    h = hashlib.sha256(f"{model_id}\x00{text}".encode()).hexdigest()[:24]
    return f"aifactory:emb:{h}"


def _get_embed_cache_ttl() -> int:
    try:
        from aifactory.config.settings import get_settings

        return get_settings().embed_cache_ttl
    except Exception:
        return 604_800  # 7 days


def get_cached_embedding(model_id: str, text: str) -> list[float] | None:
    """Return cached embedding vector or None on miss."""
    client = _get_redis()
    if client is None:
        return None
    try:
        raw = client.get(_cache_key(model_id, text))
        return json.loads(raw) if raw else None
    except Exception:
        return None


def set_cached_embedding(model_id: str, text: str, vector: list[float]) -> None:
    """Cache an embedding vector. Silently no-ops if Redis is unavailable."""
    client = _get_redis()
    if client is None:
        return
    ttl = _get_embed_cache_ttl()
    with contextlib.suppress(Exception):
        client.set(_cache_key(model_id, text), json.dumps(vector), ex=ttl if ttl > 0 else None)


def _wrap_with_cache(
    model_id: str, raw_fn: Callable[[str], list[float]]
) -> Callable[[str], list[float]]:
    """Return a cached wrapper around raw_fn. No-ops cleanly if Redis is absent."""

    def _cached(text: str) -> list[float]:
        cached = get_cached_embedding(model_id, text)
        if cached is not None:
            return cached
        vec = raw_fn(text)
        set_cached_embedding(model_id, text, vec)
        return vec

    return _cached


# ─── Stub Embedding ───────────────────────────────────────────────────────────


def _stub_embed(text: str, dimensions: int = 1536) -> list[float]:
    """SHA-256 deterministic pseudo-embedding (no semantic meaning)."""
    digest = hashlib.sha256(text.encode()).digest()
    vec = []
    for i in range(dimensions):
        byte = digest[i % len(digest)]
        bit = (byte >> (i % 8)) & 1
        vec.append(float(bit) * 2 - 1.0)
    return vec


# ─── LiteLLM Embedding ────────────────────────────────────────────────────────

# Substrings that indicate a transient network/connectivity failure vs an auth
# or quota error.  Only connection failures silently fall back to stub.
_CONNECTION_ERROR_MARKERS = (
    "connection error",
    "connection refused",
    "connection timeout",
    "network is unreachable",
    "name resolution failed",
    "failed to connect",
    "socket timeout",
    "readtimeout",
    "connecttimeout",
    "max retries exceeded",
    "remotedisconnected",
)


def _is_connection_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _CONNECTION_ERROR_MARKERS)


def _make_litellm_embed(model_id: str, fallback_dims: int = 1536) -> Callable[[str], list[float]]:
    """Return a litellm-backed embedding callable.

    Includes its own Redis cache layer (only real vectors are cached — stub
    fallback results are never stored so the cache isn't contaminated).

    On a connection/network error the function logs a warning and permanently
    falls back to stub embeddings for the lifetime of the process.  Auth errors
    and quota errors propagate so the user can see and fix them.
    """
    _use_stub = [False]  # mutable cell — flipped once on first network failure

    def _embed(text: str) -> list[float]:
        # Always check cache first — contains only real (non-stub) vectors
        cached = get_cached_embedding(model_id, text)
        if cached is not None:
            return cached

        if _use_stub[0]:
            # Network is down; return stub without caching
            return _stub_embed(text, fallback_dims)

        try:
            import litellm

            resp = litellm.embedding(model=model_id, input=[text])
            vec = resp.data[0]["embedding"]
            set_cached_embedding(model_id, text, vec)  # only real vectors cached
            return vec
        except Exception as exc:
            if _is_connection_error(exc):
                logger.warning(
                    "Embedding provider '%s' unreachable — falling back to stub embeddings.\n"
                    "  Memory search will be non-semantic until the provider is restored.\n"
                    "  Fix: check your network / API endpoint, then restart.\n"
                    "  Run `aifactory memory health` to diagnose.",
                    model_id,
                )
                _use_stub[0] = True
                return _stub_embed(text, fallback_dims)
            # Auth errors, quota errors — propagate so the user can fix them
            raise

    return _embed


# ─── HuggingFace Embedding ────────────────────────────────────────────────────


def _make_hf_embed(model_name: str) -> Callable[[str], list[float]]:
    """Return a HuggingFace-backed embedding callable (loaded once, cached)."""
    _model: list[Any] = []  # lazy singleton via mutable cell

    def _embed(text: str) -> list[float]:
        if not _model:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            _model.append(HuggingFaceEmbedding(model_name=model_name))
        return _model[0].get_text_embedding(text)

    return _embed


# ─── Public API ───────────────────────────────────────────────────────────────


def build_embed_fn() -> tuple[Callable[[str], list[float]], int]:
    """
    Build the best available embedding function with Redis caching.

    Returns (embed_fn, dimensions) where embed_fn(text) → list[float].

    This is the primary entrypoint for MemoryRouter and SemanticCache.
    """
    provider, model_id, dims = resolve_provider()

    if provider == "stub":
        # Stub: don't bother with Redis cache (hash is free)
        return (lambda text: _stub_embed(text, dims)), dims

    if provider == "huggingface":
        raw = _make_hf_embed(model_id)
        cached = _wrap_with_cache(f"hf/{model_id}", raw)
        return cached, dims

    # litellm / openai — caching + connection-error fallback are built-in
    return _make_litellm_embed(model_id, fallback_dims=dims), dims
