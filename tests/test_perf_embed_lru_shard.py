"""
Benchmark + correctness tests: in-process LRU shard in front of the Redis
embed cache (Opt #7).

Prior perf commits (Opt #1 ``a651f2d`` .. Opt #6 ``d14fc95``) parallelised
I/O, added result caches, and stabilised prompt-cache prefixes. Opt #7 shipped
here caches embedding vectors in a 512-entry process-local LRU so repeated
embed queries for the same ``(model_id, text)`` skip the ~1-3 ms Redis
round-trip entirely.

Baseline (cold): every call hits the underlying provider (simulated with a
    2 ms sleep — the sum of a Redis round-trip plus a token of network jitter).

Optimized (warm): after the first call, all subsequent same-key calls are
    served from the in-process LRU in sub-microsecond time.
"""

from __future__ import annotations

import statistics
import threading
import time
from typing import Any

import pytest

import aifactory.core.embed_provider as ep
from aifactory.core.embed_provider import (
    _clear_embed_lru,
    _wrap_with_process_lru,
)

_WARMUP = 5
_ITERS = 100
_SIMULATED_ROUND_TRIP_S = 0.002  # 2 ms — Redis + jitter


@pytest.fixture(autouse=True)
def _reset_lru() -> Any:
    _clear_embed_lru()
    # Ensure Redis is never actually consulted in these tests
    prev_probed, prev_client = ep._REDIS_PROBED, ep._REDIS_CLIENT
    ep._REDIS_PROBED = True
    ep._REDIS_CLIENT = None
    yield
    _clear_embed_lru()
    ep._REDIS_PROBED, ep._REDIS_CLIENT = prev_probed, prev_client


def _make_slow_fn(call_counter: list[int]) -> Any:
    def _slow(text: str) -> list[float]:
        call_counter[0] += 1
        time.sleep(_SIMULATED_ROUND_TRIP_S)
        # Deterministic-per-text so byte-identity assertions are meaningful.
        seed = sum(ord(c) for c in text)
        return [float((seed + i) % 7) * 0.1 for i in range(8)]

    return _slow


def _measure(fn: Any, iters: int) -> list[float]:
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


@pytest.mark.perf
def test_lru_shard_beats_cold_redis_roundtrip() -> None:
    """
    Warm LRU hits must be at least 100× faster than the simulated 2 ms
    cold path. In practice the ratio is closer to 10,000× (2 ms → sub-µs).
    """
    calls: list[int] = [0]
    slow = _make_slow_fn(calls)
    text = "the quick brown fox jumps over the lazy dog"

    def time_cold_once() -> float:
        # Clear cache OUTSIDE the timed region so setup cost doesn't skew
        # the measurement; we're benchmarking the embed() call itself.
        _clear_embed_lru()
        embed = _wrap_with_process_lru("bench-model", slow)
        t0 = time.perf_counter()
        embed(text)
        return time.perf_counter() - t0

    _clear_embed_lru()
    embed_warm = _wrap_with_process_lru("bench-model", slow)
    embed_warm(text)  # prime

    def time_warm_once() -> float:
        t0 = time.perf_counter()
        embed_warm(text)
        return time.perf_counter() - t0

    # Warm-up
    for _ in range(_WARMUP):
        time_cold_once()
    for _ in range(_WARMUP):
        time_warm_once()

    cold = [time_cold_once() for _ in range(_ITERS)]
    # Re-prime warm cache after the cold loop cleared it.
    _clear_embed_lru()
    embed_warm = _wrap_with_process_lru("bench-model", slow)
    embed_warm(text)
    warm = [time_warm_once() for _ in range(_ITERS)]

    cold_mean = statistics.mean(cold)
    warm_mean = statistics.mean(warm)
    speedup = cold_mean / warm_mean if warm_mean > 0 else float("inf")

    print(
        "\n[perf] embed_provider in-process LRU shard\n"
        f"  simulated RTT    : {_SIMULATED_ROUND_TRIP_S * 1e3:.1f} ms per cold call\n"
        f"  iterations       : {_ITERS} (warmup {_WARMUP})\n"
        f"  cold mean        : {cold_mean * 1e6:.2f} µs\n"
        f"  cold p50         : {statistics.median(cold) * 1e6:.2f} µs\n"
        f"  warm mean        : {warm_mean * 1e6:.2f} µs\n"
        f"  warm p50         : {statistics.median(warm) * 1e6:.2f} µs\n"
        f"  speedup          : {speedup:.1f}×\n"
    )

    assert warm_mean < cold_mean, (
        f"Warm LRU not faster: {warm_mean * 1e6:.2f}µs vs {cold_mean * 1e6:.2f}µs cold"
    )
    assert speedup >= 100.0, (
        f"Expected ≥100× speedup, got {speedup:.1f}× "
        f"(cold {cold_mean * 1e6:.2f}µs, warm {warm_mean * 1e6:.2f}µs)"
    )


def test_lru_hits_are_byte_identical_to_cold_calls() -> None:
    """Cached vectors must exactly equal a fresh underlying-fn result."""
    calls: list[int] = [0]
    slow = _make_slow_fn(calls)
    embed = _wrap_with_process_lru("model-x", slow)

    text = "byte identity check"
    cold = embed(text)  # first call: underlying fn is invoked
    baseline_calls = calls[0]

    # 100 warm hits — should not invoke the underlying fn again
    warms = [embed(text) for _ in range(100)]

    assert calls[0] == baseline_calls, (
        f"underlying fn called {calls[0] - baseline_calls} times on warm hits"
    )
    for i, w in enumerate(warms):
        assert w == cold, f"warm hit #{i} diverged from cold result"

    # Each returned list must be a fresh list — caller mutation must not
    # corrupt future cache reads.
    warms[0].append(999.0)
    fresh = embed(text)
    assert fresh == cold, "caller-side mutation of returned list corrupted the cache"


def test_lru_is_thread_safe_under_concurrent_hits() -> None:
    """
    10 threads calling the same key concurrently must all get the identical
    vector with no exceptions. Catches sloppy locking around the shard.
    """
    calls: list[int] = [0]
    slow = _make_slow_fn(calls)
    embed = _wrap_with_process_lru("thread-model", slow)

    text = "concurrent access"
    results: list[list[float]] = []
    errors: list[BaseException] = []
    results_lock = threading.Lock()
    start_gate = threading.Event()

    def worker() -> None:
        try:
            start_gate.wait()
            local = [embed(text) for _ in range(20)]
            with results_lock:
                results.extend(local)
        except BaseException as exc:  # noqa: BLE001
            with results_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    start_gate.set()
    for t in threads:
        t.join()

    assert not errors, f"concurrent embed() calls raised: {errors!r}"
    assert len(results) == 10 * 20
    first = results[0]
    for i, r in enumerate(results):
        assert r == first, f"thread result #{i} diverged from first"


def test_lru_isolates_by_model_id() -> None:
    """
    Changing model_id between calls must force a fresh miss — the LRU must
    never return a vector computed for a different model.
    """
    calls_a: list[int] = [0]
    calls_b: list[int] = [0]

    def fn_a(text: str) -> list[float]:
        calls_a[0] += 1
        return [1.0, 1.0, 1.0]

    def fn_b(text: str) -> list[float]:
        calls_b[0] += 1
        return [2.0, 2.0, 2.0]

    embed_a = _wrap_with_process_lru("model-A", fn_a)
    embed_b = _wrap_with_process_lru("model-B", fn_b)

    text = "same text, different models"
    va1 = embed_a(text)
    vb1 = embed_b(text)
    assert va1 == [1.0, 1.0, 1.0]
    assert vb1 == [2.0, 2.0, 2.0]
    assert calls_a[0] == 1 and calls_b[0] == 1

    # Warm hits per model — no extra underlying calls.
    for _ in range(20):
        assert embed_a(text) == [1.0, 1.0, 1.0]
        assert embed_b(text) == [2.0, 2.0, 2.0]
    assert calls_a[0] == 1, "model-A LRU leaked into model-B key"
    assert calls_b[0] == 1, "model-B LRU leaked into model-A key"
