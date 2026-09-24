"""
Benchmark: parallel per-level fan-out in ``MemoryRouter.retrieve()``.

The previous perf pass (commit a651f2d, Opt #1) already parallelised the
run-start I/O in ``master_receive``. Opt #2, shipped here, parallelises the
per-level Qdrant queries inside ``retrieve()`` itself.

Baseline (serial): for each of ``len(MemoryLevel)`` (=3) levels, call
    self.client.query_points(...) sequentially.

Optimized (parallel): fan out the N per-level query_points calls onto a
module-level ThreadPoolExecutor. QdrantClient is thread-safe. Wall-clock
latency collapses from ``sum(per_level_latency)`` to
``max(per_level_latency)``.

Simulated per-level latency: 20 ms per query × 3 levels
    serial   ≈ 60 ms
    parallel ≈ 20 ms + fan-out overhead
    → theoretical speedup ≈ 66%.

We assert ≥ 40% speedup to leave headroom for thread-pool overhead and CI
jitter. All I/O is mocked with ``time.sleep`` — no Qdrant, no network.
"""

from __future__ import annotations

import logging
import statistics
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from aifactory.core.memory_router import (
    MemoryRouter,
    _close_retrieve_executor,
    _stub_embed,
)
from aifactory.core.schemas import MemoryEntry, MemoryLevel

# ───────────────────────── I/O latency simulation ─────────────────────────

_PER_LEVEL_SLEEP_S = 0.020  # 20 ms per Qdrant query
_WARMUP = 10
_ITERS = 100


@pytest.fixture(autouse=True)
def _reset_executor() -> Any:
    """Ensure a clean executor before + after each test."""
    _close_retrieve_executor()
    yield
    _close_retrieve_executor()


def _make_router_with_mock_client(
    query_points_side_effect: Any = None,
) -> MemoryRouter:
    """
    Build a MemoryRouter whose ``client.query_points`` is a MagicMock,
    but whose collection-init path is entirely stubbed out (no Qdrant).
    """
    client = MagicMock()
    client.get_collections.return_value = SimpleNamespace(collections=[])
    client.create_collection.return_value = None
    client.create_payload_index.return_value = None
    router = MemoryRouter(qdrant_client=client, embedding_fn=_stub_embed)
    if query_points_side_effect is not None:
        router.client.query_points = MagicMock(side_effect=query_points_side_effect)
    return router


def _fake_query_points_slow(*_args: Any, **kwargs: Any) -> Any:
    """Simulate a per-level Qdrant query with fixed latency + one hit."""
    time.sleep(_PER_LEVEL_SLEEP_S)
    collection = kwargs.get("collection_name", "")
    if collection.endswith("_agent"):
        level = "agent"
    elif collection.endswith("_team"):
        level = "team"
    else:
        level = "global"
    payload = {
        "level": level,
        "content": f"hit from {collection}",
        "agent_id": "a1",
        "team_id": "t1",
        "entry_type": "observation",
        "turn_number": 1,
    }
    # Score varies per level so we can assert score-desc sort order.
    score = {"agent": 0.9, "team": 0.5, "global": 0.7}[level]
    point = SimpleNamespace(score=score, payload=payload)
    return SimpleNamespace(points=[point])


# ───────────────────────── benchmark baseline ─────────────────────────


def _serial_retrieve_baseline(router: MemoryRouter, query: str) -> list[MemoryEntry]:
    """
    Replicates the PRE-optimization serial code path directly against the
    mocked client — used as the baseline for the speedup assertion so both
    branches run in the same process against the same mock.
    """
    levels = list(MemoryLevel)
    k = router.top_k
    query_vector = router._embed(query)
    all_results: list[tuple[float, MemoryEntry]] = []
    for level in levels:
        collection = router._collection_name(level)
        try:
            results = router.client.query_points(
                collection_name=collection,
                query=query_vector,
                query_filter=None,
                limit=k,
                with_payload=True,
            )
            for point in results.points:
                score = point.score if point.score is not None else 0.0
                entry_obj = MemoryEntry(**point.payload)
                all_results.append((score, entry_obj))
        except Exception:
            pass
    all_results.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in all_results[:k]]


def _measure(fn: Any, iters: int) -> list[float]:
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


# ───────────────────────── tests ─────────────────────────


@pytest.mark.perf
def test_retrieve_parallel_beats_serial() -> None:
    """
    Assert per-level fan-out is measurably faster than serial iteration
    against a mocked QdrantClient with 20 ms per-level latency.

        serial   ≈ 3 × 20 ms = 60 ms
        parallel ≈ 20 ms + thread-pool overhead
        → theoretical speedup ≈ 66%. Assert ≥ 40% to allow CI jitter.
    """
    router = _make_router_with_mock_client(query_points_side_effect=_fake_query_points_slow)

    _measure(lambda: _serial_retrieve_baseline(router, "hello"), _WARMUP)
    _measure(lambda: router.retrieve("hello"), _WARMUP)

    baseline = _measure(lambda: _serial_retrieve_baseline(router, "hello"), _ITERS)
    optimized = _measure(lambda: router.retrieve("hello"), _ITERS)

    base_mean = statistics.mean(baseline)
    opt_mean = statistics.mean(optimized)
    speedup = (base_mean - opt_mean) / base_mean * 100.0

    print(
        "\n[perf] MemoryRouter.retrieve per-level fan-out\n"
        f"  levels               : {len(list(MemoryLevel))}\n"
        f"  per-level sleep      : {_PER_LEVEL_SLEEP_S * 1000:.1f} ms\n"
        f"  iterations           : {_ITERS} (warmup {_WARMUP})\n"
        f"  baseline serial mean : {base_mean * 1000:.2f} ms\n"
        f"  baseline serial p50  : {statistics.median(baseline) * 1000:.2f} ms\n"
        f"  optimized par. mean  : {opt_mean * 1000:.2f} ms\n"
        f"  optimized par. p50   : {statistics.median(optimized) * 1000:.2f} ms\n"
        f"  speedup              : {speedup:.1f}%\n"
    )

    assert opt_mean < base_mean, (
        f"Parallel impl not faster: {opt_mean * 1000:.2f}ms vs {base_mean * 1000:.2f}ms baseline"
    )
    assert speedup >= 40.0, (
        f"Expected ≥40% speedup, got {speedup:.1f}% "
        f"(baseline {base_mean * 1000:.2f}ms, optimized {opt_mean * 1000:.2f}ms)"
    )


def test_retrieve_per_level_error_containment(caplog: pytest.LogCaptureFixture) -> None:
    """
    A single per-level query raising must not poison the other levels.
    ``retrieve()`` must still return the successful results and log a
    warning naming the failing collection.
    """

    def side_effect(*_args: Any, **kwargs: Any) -> Any:
        collection = kwargs.get("collection_name", "")
        if collection.endswith("_team"):
            raise RuntimeError("simulated Qdrant outage on team collection")
        return _fake_query_points_slow(**kwargs)

    router = _make_router_with_mock_client(query_points_side_effect=side_effect)

    with caplog.at_level(logging.WARNING, logger="aifactory.core.memory_router"):
        results = router.retrieve("hello", top_k=10)

    assert len(results) >= 1, "successful levels should still return entries"
    contents = {r.content for r in results}
    # agent + global succeeded; team failed
    assert any(c.endswith("_agent") for c in contents), contents
    assert any(c.endswith("_global") for c in contents), contents
    assert not any(c.endswith("_team") for c in contents), contents

    assert any(
        "Memory retrieval failed" in rec.getMessage() and "team" in rec.getMessage()
        for rec in caplog.records
    ), "per-level failure must emit a warning naming the failing collection"


def test_retrieve_results_score_desc_sorted() -> None:
    """
    After fan-out, results must be sorted by score descending — identical
    to the pre-optimization ordering guarantee.
    """
    router = _make_router_with_mock_client(query_points_side_effect=_fake_query_points_slow)
    # Ask for the full slice so we can inspect ordering across all 3 levels.
    results = router.retrieve("hello", top_k=10)

    # scores: agent=0.9, global=0.7, team=0.5 → expected order.
    assert len(results) == 3
    ordered_contents = [r.content for r in results]
    assert ordered_contents[0].endswith("_agent"), ordered_contents
    assert ordered_contents[1].endswith("_global"), ordered_contents
    assert ordered_contents[2].endswith("_team"), ordered_contents

    ordered_levels = [r.level.value for r in results]
    assert ordered_levels == ["agent", "global", "team"]
