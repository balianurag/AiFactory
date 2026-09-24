"""
Benchmark + correctness tests: ``best_model()`` result cache.

Prior perf commits (Opt #1 ``a651f2d``, Opt #2 ``0210ac2``) parallelised
run-start I/O and per-level Qdrant retrieval. Opt #3 shipped here caches
the ``best_model()`` result so the 10-ability dot-product across the ~20
model registry entries is not recomputed on every worker dispatch.

Baseline (cold): 100 fresh calls that each invalidate the cache first, so
each call re-runs the full ``sum(...) / len(QUALITY_ABILITIES)`` loop
across the model list.

Optimized (warm): 100 calls with the same ``models`` list under a stable
version counter — every call after the first is an O(1) dict lookup.

Cache invalidation is deterministic: ``clear_best_model_cache()`` bumps a
version counter that is part of every cache key, so any mutation to model
abilities (calibrator patches, ``load_models.cache_clear()``, direct
registry edits) forces a recompute on the next call.
"""

from __future__ import annotations

import statistics
import threading
import time
from typing import Any

import pytest

from aifactory.core.model_selector import (
    best_model,
    clear_best_model_cache,
)

_WARMUP = 10
_ITERS = 100


# A realistic-sized registry: 20 models × 10 abilities, deterministic values.
def _make_registry() -> list[dict[str, Any]]:
    registry: list[dict[str, Any]] = []
    for i in range(20):
        registry.append(
            {
                "id": f"model-{i:02d}",
                "reasoning": (i % 4),
                "coding": ((i + 1) % 4),
                "math": ((i + 2) % 4),
                "writing": ((i + 3) % 4),
                "instruction_following": ((i + 1) % 4),
                "tool_use": ((i + 2) % 4),
                "knowledge": ((i + 3) % 4),
                "speed": (3 - (i % 4)),
                "cost_efficiency": (3 - ((i + 1) % 4)),
                "context_handling": ((i + 2) % 4),
            }
        )
    # Guarantee a unique winner so cache/no-cache results are byte-identical.
    for a in (
        "reasoning",
        "coding",
        "math",
        "writing",
        "instruction_following",
        "tool_use",
        "knowledge",
        "context_handling",
    ):
        registry[7][a] = 3
    return registry


@pytest.fixture(autouse=True)
def _reset_cache() -> Any:
    clear_best_model_cache()
    yield
    clear_best_model_cache()


def _measure(fn: Any, iters: int) -> list[float]:
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


@pytest.mark.perf
def test_best_model_cache_beats_cold_recompute() -> None:
    """
    Assert warm cache hits are at least 20× faster than cold recomputes.

    Cold path: ~20 models × 8 quality abilities × dict lookups → tens of µs.
    Warm path: a single dict.get() on the module-level cache → sub-µs.
    """
    registry = _make_registry()

    def cold_call() -> str:
        clear_best_model_cache()
        return best_model(models=registry)

    def warm_call() -> str:
        return best_model(models=registry)

    # Warm-up JIT / interpreter caches and prime the warm path once.
    _measure(cold_call, _WARMUP)
    best_model(models=registry)  # prime cache
    _measure(warm_call, _WARMUP)

    cold = _measure(cold_call, _ITERS)
    # Re-prime because the cold measurements above cleared the cache.
    best_model(models=registry)
    warm = _measure(warm_call, _ITERS)

    cold_mean = statistics.mean(cold)
    warm_mean = statistics.mean(warm)
    speedup = cold_mean / warm_mean if warm_mean > 0 else float("inf")

    print(
        "\n[perf] best_model() result cache\n"
        f"  registry size    : {len(registry)} models\n"
        f"  iterations       : {_ITERS} (warmup {_WARMUP})\n"
        f"  cold mean        : {cold_mean * 1e6:.2f} µs\n"
        f"  cold p50         : {statistics.median(cold) * 1e6:.2f} µs\n"
        f"  warm mean        : {warm_mean * 1e6:.2f} µs\n"
        f"  warm p50         : {statistics.median(warm) * 1e6:.2f} µs\n"
        f"  speedup          : {speedup:.1f}×\n"
    )

    assert warm_mean < cold_mean, (
        f"Warm cache not faster: {warm_mean * 1e6:.2f}µs vs {cold_mean * 1e6:.2f}µs cold"
    )
    assert speedup >= 20.0, (
        f"Expected ≥20× speedup, got {speedup:.1f}× "
        f"(cold {cold_mean * 1e6:.2f}µs, warm {warm_mean * 1e6:.2f}µs)"
    )


def test_clear_cache_forces_recompute_on_ability_change() -> None:
    """
    After ``clear_best_model_cache()``, a mutation to model abilities must be
    reflected in the returned model_id — proving the cache is not silently stale.
    """
    registry = _make_registry()

    first = best_model(models=registry)
    # Same call, cache hit, identical result.
    assert best_model(models=registry) == first

    # Mutate abilities in place so a *different* model becomes the winner.
    # Zero out the current winner's quality scores and boost model-00 to max.
    for m in registry:
        if m["id"] == first:
            for a in (
                "reasoning",
                "coding",
                "math",
                "writing",
                "instruction_following",
                "tool_use",
                "knowledge",
                "context_handling",
            ):
                m[a] = 0
        if m["id"] == "model-00":
            for a in (
                "reasoning",
                "coding",
                "math",
                "writing",
                "instruction_following",
                "tool_use",
                "knowledge",
                "context_handling",
            ):
                m[a] = 3

    # Without invalidation the cache would still return `first` — proving the
    # need for explicit invalidation. Confirm the stale hit first…
    assert best_model(models=registry) == first, (
        "cache should return the stale value until invalidated (that's the point of the cache)"
    )

    # …then invalidate and confirm the recompute picks the new winner.
    clear_best_model_cache()
    second = best_model(models=registry)
    assert second == "model-00"
    assert second != first


def test_cache_hits_are_byte_identical_to_cold_calls() -> None:
    """100 warm hits must return exactly the same model_id as 100 cold calls."""
    registry = _make_registry()

    cold_results: list[str] = []
    for _ in range(100):
        clear_best_model_cache()
        cold_results.append(best_model(models=registry))

    clear_best_model_cache()
    best_model(models=registry)  # prime
    warm_results = [best_model(models=registry) for _ in range(100)]

    assert len(set(cold_results)) == 1, "cold path must be deterministic"
    assert len(set(warm_results)) == 1, "warm path must be deterministic"
    assert cold_results[0] == warm_results[0], (
        f"warm result {warm_results[0]!r} diverges from cold result {cold_results[0]!r}"
    )


def test_cache_is_thread_safe_under_concurrent_hits() -> None:
    """
    10 threads hammering ``best_model()`` on the same key must all return the
    same result with no exceptions. Catches sloppy locking around the cache dict.
    """
    registry = _make_registry()
    clear_best_model_cache()

    results: list[str] = []
    errors: list[BaseException] = []
    results_lock = threading.Lock()
    start_gate = threading.Event()

    def worker() -> None:
        try:
            start_gate.wait()
            # Mix cold + warm within each thread to exercise the write path.
            local: list[str] = []
            for _ in range(50):
                local.append(best_model(models=registry))
            with results_lock:
                results.extend(local)
        except BaseException as exc:  # noqa: BLE001 - surface any failure
            with results_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    start_gate.set()
    for t in threads:
        t.join()

    assert not errors, f"concurrent best_model() calls raised: {errors!r}"
    assert len(results) == 10 * 50
    assert len(set(results)) == 1, (
        f"concurrent callers disagreed on the winner: {set(results)!r}"
    )
