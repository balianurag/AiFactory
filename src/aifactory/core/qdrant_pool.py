"""
Shared Qdrant client singleton.

All modules that need a QdrantClient should call get_qdrant() instead of
constructing their own instance.  This reduces connection pool overhead from
4–5 independent clients to a single shared pool.

Fallback: when the configured remote Qdrant is unreachable, falls back to an
in-memory QdrantClient (same behaviour as before, but shared across all
consumers).  The fallback decision is made once at startup — subsequent calls
to get_qdrant() always return the same client.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_client: Any = None       # QdrantClient — module-level singleton
_is_in_memory: bool = False


def get_qdrant() -> Any:
    """Return the shared QdrantClient, building it on first call."""
    global _client
    if _client is None:
        _client = _build_client()
    return _client


def is_in_memory() -> bool:
    """Return True when the singleton is using an in-process store (no persistence)."""
    get_qdrant()  # ensure singleton is built
    return _is_in_memory


def reset_qdrant_client() -> None:
    """Reset the singleton — used in tests to swap in a mock client."""
    global _client, _is_in_memory
    _client = None
    _is_in_memory = False


def _build_client() -> Any:
    """
    Build the singleton QdrantClient.

    Priority:
      1. Remote Qdrant at QDRANT_URL (when URL starts with 'http')
      2. In-memory QdrantClient (":memory:") as graceful fallback
    """
    global _is_in_memory

    from qdrant_client import QdrantClient

    from aifactory.config.settings import get_settings

    settings = get_settings()

    if settings.qdrant_url.startswith("http"):
        try:
            client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )
            client.get_collections()  # connectivity probe
            _is_in_memory = False
            logger.info(
                "Qdrant singleton: connected to %s",
                settings.qdrant_url.split("@")[-1] if "@" in settings.qdrant_url else settings.qdrant_url,
            )
            return client
        except Exception as exc:
            logger.warning(
                "Qdrant at %s unreachable (%s) — falling back to in-memory mode. "
                "Vector data will NOT persist between restarts. "
                "Set QDRANT_URL to a reachable Qdrant server to enable persistence.",
                settings.qdrant_url,
                exc,
            )

    _is_in_memory = True
    client = QdrantClient(":memory:")
    logger.info("Qdrant singleton: using in-memory mode")
    return client
