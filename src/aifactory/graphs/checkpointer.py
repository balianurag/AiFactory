"""
LangGraph checkpointer factory — MemorySaver (default) or PostgresSaver (opt-in).

Feature-flagged via `settings.aifactory_pipeline_checkpointer`:
  - "memory"   : MemorySaver (current behaviour, no persistence across processes)
  - "postgres" : PostgresSaver backed by settings.database_url

Fail-soft: if postgres is unreachable at first use, we fall back to MemorySaver
and log a loud warning so the pipeline keeps running.

Thread-safe process singleton, mirroring the qdrant_pool pattern.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

_checkpointer: Any = None
_checkpointer_lock = threading.Lock()
_fell_back_to_memory: bool = False


def reset_checkpointer() -> None:
    """Reset the singleton — used in tests to swap in a fresh instance."""
    global _checkpointer, _fell_back_to_memory
    with _checkpointer_lock:
        _cp = _checkpointer
        _checkpointer = None
        _fell_back_to_memory = False
    # Best-effort close of the underlying pool for PostgresSaver
    try:
        pool = getattr(_cp, "conn", None)
        if pool is not None and hasattr(pool, "close"):
            pool.close()
    except Exception as _exc:
        logger.debug("reset_checkpointer: pool close raised (%s)", _exc)


def build_checkpointer(settings: Any | None = None) -> Any:
    """
    Return a LangGraph checkpointer instance based on
    ``settings.aifactory_pipeline_checkpointer``.

      - "memory"   (default): :class:`MemorySaver` — current behaviour.
      - "postgres"           : :class:`PostgresSaver` connected to
        ``settings.database_url``; on any failure we fall back to
        :class:`MemorySaver` and log a loud warning.

    Idempotent bootstrap: on first postgres use we call ``saver.setup()`` to
    create the required tables. Thread-safe process singleton.
    """
    global _checkpointer, _fell_back_to_memory
    if _checkpointer is not None:
        return _checkpointer

    with _checkpointer_lock:
        if _checkpointer is not None:
            return _checkpointer

        if settings is None:
            try:
                from aifactory.config.settings import get_settings

                settings = get_settings()
            except Exception as exc:  # pragma: no cover — extreme edge
                logger.warning(
                    "build_checkpointer: settings unavailable (%s) — using MemorySaver",
                    exc,
                )
                _checkpointer = MemorySaver()
                _fell_back_to_memory = True
                return _checkpointer

        backend = str(getattr(settings, "aifactory_pipeline_checkpointer", "memory")).lower()

        if backend != "postgres":
            _checkpointer = MemorySaver()
            _fell_back_to_memory = False
            return _checkpointer

        # ── Postgres path ────────────────────────────────────────────────
        db_url = getattr(settings, "database_url", "") or ""
        min_size = int(getattr(settings, "aifactory_checkpoint_pool_min_size", 2))
        max_size = int(getattr(settings, "aifactory_checkpoint_pool_max_size", 10))

        try:
            saver = _build_postgres_saver(db_url, min_size=min_size, max_size=max_size)
            _checkpointer = saver
            _fell_back_to_memory = False
            logger.info(
                "Checkpointer: PostgresSaver active (pool min=%d max=%d)",
                min_size,
                max_size,
            )
            return _checkpointer
        except Exception as exc:
            logger.warning(
                "PostgresSaver unavailable (%s) — falling back to MemorySaver. "
                "Pipeline runs will NOT survive process restarts. "
                "Fix: check DATABASE_URL and that postgres is reachable.",
                exc,
            )
            _checkpointer = MemorySaver()
            _fell_back_to_memory = True
            return _checkpointer


def _build_postgres_saver(db_url: str, *, min_size: int, max_size: int) -> Any:
    """
    Build & bootstrap a PostgresSaver. Raises on any failure so the caller can
    apply its fail-soft policy.
    """
    if not db_url:
        raise ValueError("database_url is empty")

    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        conninfo=db_url,
        min_size=min_size,
        max_size=max_size,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False,
    )
    pool.open(wait=True, timeout=5.0)

    saver = PostgresSaver(pool)  # type: ignore[arg-type]
    # Idempotent bootstrap — creates checkpointer tables on first use.
    saver.setup()
    return saver


def is_fell_back_to_memory() -> bool:
    """Test helper: True when the postgres path silently degraded to MemorySaver."""
    return _fell_back_to_memory
