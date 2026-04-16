"""
Tests for core/research_cache.py — semantic research cache.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from aifactory.core.research_cache import (
    ResearchCache,
    _cosine,
    _norm_query,
    _word_overlap,
)

# =============================================================================
# Helpers
# =============================================================================


class TestNormQuery:
    def test_lowercases(self) -> None:
        assert _norm_query("Hello WORLD") == "hello world"

    def test_collapses_whitespace(self) -> None:
        assert _norm_query("  foo   bar  ") == "foo bar"

    def test_strips(self) -> None:
        assert _norm_query("\tleading") == "leading"


class TestCosine:
    def test_identical_vectors(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_empty_vectors(self) -> None:
        assert _cosine([], []) == 0.0

    def test_mismatched_length_returns_zero(self) -> None:
        assert _cosine([1.0, 2.0], [1.0]) == 0.0


class TestWordOverlap:
    def test_identical_strings(self) -> None:
        assert _word_overlap("hello world", "hello world") == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        assert _word_overlap("foo bar", "baz qux") == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        # {"a", "b"} ∩ {"b", "c"} = {"b"}; union = {"a","b","c"} → 1/3
        assert _word_overlap("a b", "b c") == pytest.approx(1 / 3)

    def test_empty_strings(self) -> None:
        assert _word_overlap("", "foo") == 0.0
        assert _word_overlap("foo", "") == 0.0


# =============================================================================
# ResearchCache
# =============================================================================


class TestResearchCache:
    @pytest.fixture
    def cache(self, tmp_path: Path) -> ResearchCache:
        return ResearchCache(cache_dir=tmp_path)

    def test_empty_cache_returns_none(self, cache: ResearchCache) -> None:
        assert cache.lookup("anything") is None

    def test_store_and_exact_lookup(self, cache: ResearchCache) -> None:
        cache.store("how does TCP work", "TCP summary here", run_id="r1")
        hit = cache.lookup("how does TCP work")
        assert hit is not None
        assert hit["summary"] == "TCP summary here"
        assert hit["run_id"] == "r1"

    def test_normalised_key_lookup(self, cache: ResearchCache) -> None:
        cache.store("Bhuvan WMS API", "Bhuvan summary", run_id="r1")
        # Different casing and spacing should still hit
        hit = cache.lookup("bhuvan wms api")
        assert hit is not None
        assert hit["summary"] == "Bhuvan summary"

    def test_word_overlap_fallback(self, cache: ResearchCache) -> None:
        cache.store("how to use FastAPI with PostgreSQL", "FastAPI summary", run_id="r1")
        # High word overlap (>= 0.72) should hit
        hit = cache.lookup("FastAPI with PostgreSQL how to use")
        assert hit is not None
        assert "FastAPI" in hit["summary"]

    def test_no_word_overlap_miss(self, cache: ResearchCache) -> None:
        cache.store("python asyncio event loop", "asyncio summary", run_id="r1")
        # Completely different query
        hit = cache.lookup("completely different topic altogether")
        assert hit is None

    def test_cosine_similarity_hit(self, cache: ResearchCache) -> None:
        # Create two similar unit vectors (cosine > 0.82)
        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = [0.95, 0.31, 0.0, 0.0]  # cos ≈ 0.95
        # Normalise v2
        mag = math.sqrt(sum(x**2 for x in v2))
        v2_norm = [x / mag for x in v2]

        cache.store("query A", "summary A", run_id="r1", embedding=v1)
        hit = cache.lookup("query B", embedding=v2_norm)
        assert hit is not None
        assert hit["summary"] == "summary A"

    def test_cosine_similarity_miss(self, cache: ResearchCache) -> None:
        # Orthogonal vectors → cosine = 0.0 → miss
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        cache.store("query A", "summary A", run_id="r1", embedding=v1)
        hit = cache.lookup("different query", embedding=v2)
        assert hit is None

    def test_size(self, cache: ResearchCache) -> None:
        assert cache.size() == 0
        cache.store("q1", "s1", run_id="r1")
        cache.store("q2", "s2", run_id="r1")
        assert cache.size() == 2

    def test_clear(self, cache: ResearchCache) -> None:
        cache.store("q1", "s1", run_id="r1")
        cache.clear()
        assert cache.size() == 0
        assert cache.lookup("q1") is None

    def test_persists_to_disk(self, tmp_path: Path) -> None:
        c1 = ResearchCache(cache_dir=tmp_path)
        c1.store("persistent query", "persistent summary", run_id="r1")

        # New instance reads from same directory
        c2 = ResearchCache(cache_dir=tmp_path)
        hit = c2.lookup("persistent query")
        assert hit is not None
        assert hit["summary"] == "persistent summary"

    def test_embedding_stored_in_entry(self, cache: ResearchCache) -> None:
        emb = [0.1, 0.2, 0.3]
        cache.store("with embedding", "summary", run_id="r1", embedding=emb)
        hit = cache.lookup("with embedding")
        assert hit is not None
        assert hit["embedding"] == emb

    def test_no_embedding_stored_without_it(self, cache: ResearchCache) -> None:
        cache.store("no embed", "summary", run_id="r1")
        hit = cache.lookup("no embed")
        assert hit is not None
        assert "embedding" not in hit
