"""
Tests for the Phase 6.1 Postgres-backed LangGraph checkpointer.

The default flag path returns an in-memory saver so existing pipelines
keep working. The postgres path builds a PostgresSaver when reachable
and falls back to MemorySaver + a loud warning when it isn't.

A live-database resume-after-crash test runs when TEST_POSTGRES_URL is
set; otherwise it's skipped.
"""

from __future__ import annotations

import os

import pytest
from langgraph.checkpoint.memory import MemorySaver

from aifactory.config.settings import get_settings
from aifactory.graphs import checkpointer as cp_mod
from aifactory.graphs.checkpointer import build_checkpointer, reset_checkpointer


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    reset_checkpointer()
    yield
    reset_checkpointer()


def test_default_flag_returns_memory_saver() -> None:
    """With no override the singleton is a MemorySaver — legacy behaviour."""
    saver = build_checkpointer(get_settings())
    assert isinstance(saver, MemorySaver)
    # Idempotent — same instance on second call.
    assert build_checkpointer(get_settings()) is saver


def test_postgres_flag_unreachable_falls_back(monkeypatch, caplog) -> None:
    """Setting flag=postgres with an unreachable DB → MemorySaver + warning."""
    monkeypatch.setenv("AIFACTORY_PIPELINE_CHECKPOINTER", "postgres")
    monkeypatch.setenv("DATABASE_URL", "postgresql://nope:5432/does-not-exist")
    get_settings.cache_clear()

    with caplog.at_level("WARNING"):
        saver = build_checkpointer(get_settings())

    assert isinstance(saver, MemorySaver)
    assert cp_mod.is_fell_back_to_memory() is True
    assert any(
        "PostgresSaver" in rec.getMessage() or "postgres" in rec.getMessage().lower()
        for rec in caplog.records
    )


def test_postgres_flag_calls_setup_when_reachable(monkeypatch) -> None:
    """flag=postgres + a reachable pool → PostgresSaver and setup() ran."""
    from aifactory.graphs import checkpointer as _cp

    monkeypatch.setenv("AIFACTORY_PIPELINE_CHECKPOINTER", "postgres")
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake/fake")
    get_settings.cache_clear()

    class _FakeSaver:
        def __init__(self) -> None:
            self.setup_called = 0

        def setup(self) -> None:
            self.setup_called += 1

    fake = _FakeSaver()

    def _fake_build(db_url: str, *, min_size: int, max_size: int):  # type: ignore[no-untyped-def]
        assert db_url == "postgresql://fake/fake"
        assert min_size == 2
        assert max_size == 10
        fake.setup()  # simulate the real bootstrap
        return fake

    monkeypatch.setattr(_cp, "_build_postgres_saver", _fake_build)

    saver = build_checkpointer(get_settings())
    assert saver is fake
    assert fake.setup_called >= 1
    assert cp_mod.is_fell_back_to_memory() is False


@pytest.mark.skipif(
    not os.getenv("TEST_POSTGRES_URL"),
    reason="TEST_POSTGRES_URL not set — skipping live postgres checkpoint smoke test",
)
def test_resume_after_crash_smoke(monkeypatch) -> None:
    """Live smoke: run two nodes, kill session, re-open, resume from checkpoint."""
    monkeypatch.setenv("AIFACTORY_PIPELINE_CHECKPOINTER", "postgres")
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_POSTGRES_URL"])
    get_settings.cache_clear()

    from langgraph.graph import END, START, StateGraph

    saver = build_checkpointer(get_settings())

    class _State(dict):  # type: ignore[type-arg]
        pass

    def _node_a(state: dict) -> dict:  # type: ignore[type-arg]
        return {"a": (state.get("a", 0) + 1)}

    def _node_b(state: dict) -> dict:  # type: ignore[type-arg]
        return {"b": (state.get("b", 0) + 10)}

    g = StateGraph(dict)
    g.add_node("a", _node_a)
    g.add_node("b", _node_b)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    graph = g.compile(checkpointer=saver)

    thread_id = "resume-smoke-1"
    cfg = {"configurable": {"thread_id": thread_id}}

    state_1 = graph.invoke({}, cfg)
    assert state_1["a"] == 1 and state_1["b"] == 10

    # Simulate a fresh process — new singleton, same DB → resumes state.
    reset_checkpointer()
    saver2 = build_checkpointer(get_settings())
    graph2 = g.compile(checkpointer=saver2)
    resumed = graph2.get_state(cfg)
    assert resumed.values.get("a") == 1
    assert resumed.values.get("b") == 10
