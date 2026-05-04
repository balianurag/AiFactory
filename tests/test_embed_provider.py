"""
Tests for core/embed_provider.py — auto-selection, stub fallback, Redis cache.

Covers:
  - resolve_provider() with various env-var combinations
  - clear_provider_cache() resets the module-level singleton
  - _stub_embed() determinism and dimension
  - _cache_key() uniqueness
  - get_cached_embedding() / set_cached_embedding() with mock Redis
  - _wrap_with_cache() caches and short-circuits on second call
  - _make_litellm_embed() — hit, miss, connection fallback, auth-error propagation
  - build_embed_fn() returns (callable, int) for each provider path
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aifactory.core.embed_provider import (
    _cache_key,
    _is_connection_error,
    _make_litellm_embed,
    _stub_embed,
    _wrap_with_cache,
    build_embed_fn,
    clear_provider_cache,
    get_cached_embedding,
    resolve_provider,
    set_cached_embedding,
)

# =============================================================================
# Helpers
# =============================================================================


def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip provider API keys so tests start from a clean slate."""
    for var in (
        "OPENAI_API_KEY",
        "VOYAGE_API_KEY",
        "COHERE_API_KEY",
        "LITELLM_API_BASE",
        "REDIS_URL",
        "AIFACTORY_LLAMAINDEX_EMBEDDING_PROVIDER",
    ):
        monkeypatch.delenv(var, raising=False)


# =============================================================================
# Provider resolution
# =============================================================================


class TestResolveProvider:
    def setup_method(self) -> None:
        clear_provider_cache()

    def test_openai_key_selects_text_embedding_3_small(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clean_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key-12345")
        provider, model, dims = resolve_provider()
        assert provider == "litellm"
        assert model == "text-embedding-3-small"
        assert dims == 1536

    def test_placeholder_openai_key_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clean_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-your-key-here")
        # Voyage key not set either — should fall through to stub
        provider, _model, _dims = resolve_provider()
        assert provider == "stub"

    def test_voyage_key_selects_voyage_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clean_env(monkeypatch)
        monkeypatch.setenv("VOYAGE_API_KEY", "voyage-real-key")
        provider, model, dims = resolve_provider()
        assert provider == "litellm"
        assert "voyage" in model
        assert dims == 1024

    def test_cohere_key_selects_cohere_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clean_env(monkeypatch)
        monkeypatch.setenv("COHERE_API_KEY", "cohere-real-key")
        provider, model, dims = resolve_provider()
        assert provider == "litellm"
        assert "cohere" in model
        assert dims == 1024

    def test_litellm_api_base_selects_litellm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clean_env(monkeypatch)
        monkeypatch.setenv("LITELLM_API_BASE", "http://localhost:8080")
        provider, _model, _dims = resolve_provider()
        assert provider == "litellm"

    def test_no_keys_falls_back_to_stub(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clean_env(monkeypatch)
        # Ensure HuggingFace import also fails
        with patch.dict("sys.modules", {"llama_index.embeddings.huggingface": None}):
            provider, model, dims = resolve_provider()
        assert provider == "stub"
        assert model == "stub"
        assert dims == 1536

    def test_explicit_stub_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clean_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key-12345")
        monkeypatch.setenv("AIFACTORY_LLAMAINDEX_EMBEDDING_PROVIDER", "stub")
        provider, _model, dims = resolve_provider()
        assert provider == "stub"
        assert dims == 1536

    def test_caches_result_on_second_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clean_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
        first = resolve_provider()
        # Mutate env — cached result should not change
        monkeypatch.delenv("OPENAI_API_KEY")
        second = resolve_provider()
        assert first == second

    def test_clear_cache_forces_re_detection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clean_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
        first = resolve_provider()
        assert first[0] == "litellm"
        clear_provider_cache()
        monkeypatch.delenv("OPENAI_API_KEY")
        # No keys → stub
        with patch.dict("sys.modules", {"llama_index.embeddings.huggingface": None}):
            second = resolve_provider()
        assert second[0] == "stub"


# =============================================================================
# Stub embedding
# =============================================================================


class TestStubEmbed:
    def test_returns_correct_dimension(self) -> None:
        vec = _stub_embed("hello", dimensions=1536)
        assert len(vec) == 1536

    def test_custom_dimension(self) -> None:
        vec = _stub_embed("hello", dimensions=384)
        assert len(vec) == 384

    def test_deterministic(self) -> None:
        assert _stub_embed("foo") == _stub_embed("foo")

    def test_different_texts_differ(self) -> None:
        a = _stub_embed("hello world")
        b = _stub_embed("goodbye world")
        assert a != b

    def test_values_are_plus_minus_one(self) -> None:
        vec = _stub_embed("test")
        assert all(v in (-1.0, 1.0) for v in vec)


# =============================================================================
# Cache key
# =============================================================================


class TestCacheKey:
    def test_same_input_same_key(self) -> None:
        assert _cache_key("model", "text") == _cache_key("model", "text")

    def test_different_model_different_key(self) -> None:
        assert _cache_key("model-a", "text") != _cache_key("model-b", "text")

    def test_different_text_different_key(self) -> None:
        assert _cache_key("model", "foo") != _cache_key("model", "bar")

    def test_key_format(self) -> None:
        key = _cache_key("text-embedding-3-small", "hello")
        assert key.startswith("aifactory:emb:")


# =============================================================================
# Redis cache operations
# =============================================================================


class TestRedisCache:
    def _make_redis_mock(self) -> MagicMock:
        client = MagicMock()
        client.ping.return_value = True
        client.get.return_value = None
        return client

    def test_get_returns_none_when_no_redis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("REDIS_URL", raising=False)
        import aifactory.core.embed_provider as ep

        # Reset probe state
        ep._REDIS_PROBED = False
        ep._REDIS_CLIENT = None
        result = get_cached_embedding("model", "text")
        assert result is None

    def test_set_and_get_with_mock_redis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import aifactory.core.embed_provider as ep

        mock_client = self._make_redis_mock()
        vec = [0.1, 0.2, 0.3]
        mock_client.get.return_value = json.dumps(vec).encode()

        ep._REDIS_PROBED = True
        ep._REDIS_CLIENT = mock_client

        result = get_cached_embedding("model", "text")
        assert result == vec

        ep._REDIS_PROBED = False
        ep._REDIS_CLIENT = None

    def test_set_cached_embedding_calls_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import aifactory.core.embed_provider as ep

        mock_client = self._make_redis_mock()
        ep._REDIS_PROBED = True
        ep._REDIS_CLIENT = mock_client

        set_cached_embedding("model", "text", [0.5, 0.6])
        assert mock_client.set.called

        ep._REDIS_PROBED = False
        ep._REDIS_CLIENT = None

    def test_set_noop_when_redis_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import aifactory.core.embed_provider as ep

        ep._REDIS_PROBED = True
        ep._REDIS_CLIENT = None
        # Should not raise
        set_cached_embedding("model", "text", [1.0])


# =============================================================================
# Wrap-with-cache
# =============================================================================


class TestWrapWithCache:
    def test_cached_result_returned_without_calling_raw_fn(self) -> None:
        import aifactory.core.embed_provider as ep

        mock_client = MagicMock()
        vec = [0.1, 0.2]
        mock_client.get.return_value = json.dumps(vec).encode()

        ep._REDIS_PROBED = True
        ep._REDIS_CLIENT = mock_client

        call_count = [0]

        def raw_fn(text: str) -> list[float]:
            call_count[0] += 1
            return [9.9]

        cached = _wrap_with_cache("model", raw_fn)
        result = cached("hello")
        assert result == vec
        assert call_count[0] == 0  # raw_fn never called

        ep._REDIS_PROBED = False
        ep._REDIS_CLIENT = None

    def test_cache_miss_calls_raw_fn(self) -> None:
        import aifactory.core.embed_provider as ep

        ep._REDIS_PROBED = True
        ep._REDIS_CLIENT = None  # No Redis → always miss

        raw_called = [False]

        def raw_fn(text: str) -> list[float]:
            raw_called[0] = True
            return [1.0, 2.0]

        cached = _wrap_with_cache("model", raw_fn)
        result = cached("hello")
        assert result == [1.0, 2.0]
        assert raw_called[0]

        ep._REDIS_PROBED = False


# =============================================================================
# Connection error detection
# =============================================================================


class TestIsConnectionError:
    def test_connection_refused(self) -> None:
        assert _is_connection_error(Exception("connection refused"))

    def test_network_unreachable(self) -> None:
        assert _is_connection_error(Exception("Network is unreachable"))

    def test_auth_error_is_not_connection_error(self) -> None:
        assert not _is_connection_error(Exception("401 Unauthorized"))

    def test_quota_error_is_not_connection_error(self) -> None:
        assert not _is_connection_error(Exception("quota exceeded"))


# =============================================================================
# LiteLLM embed — connection fallback
# =============================================================================


class TestMakeLitellmEmbed:
    def test_fallback_to_stub_on_connection_error(self) -> None:
        import aifactory.core.embed_provider as ep

        ep._REDIS_PROBED = True
        ep._REDIS_CLIENT = None

        def _raise(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise Exception("connection refused")

        with patch("litellm.embedding", side_effect=_raise):
            fn = _make_litellm_embed("text-embedding-3-small", fallback_dims=1536)
            result = fn("hello")

        assert len(result) == 1536
        # Subsequent calls also return stub (flag flipped)
        result2 = fn("world")
        assert len(result2) == 1536

        ep._REDIS_PROBED = False

    def test_auth_error_propagates(self) -> None:
        import aifactory.core.embed_provider as ep

        ep._REDIS_PROBED = True
        ep._REDIS_CLIENT = None

        def _raise(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise Exception("401 Invalid API key")

        with patch("litellm.embedding", side_effect=_raise):
            fn = _make_litellm_embed("text-embedding-3-small", fallback_dims=1536)
            with pytest.raises(Exception, match="401"):
                fn("hello")

        ep._REDIS_PROBED = False

    def test_successful_call_returns_vector(self) -> None:
        import aifactory.core.embed_provider as ep

        ep._REDIS_PROBED = True
        ep._REDIS_CLIENT = None

        mock_resp = MagicMock()
        mock_resp.data = [{"embedding": [0.1, 0.2, 0.3]}]

        with patch("litellm.embedding", return_value=mock_resp):
            fn = _make_litellm_embed("text-embedding-3-small", fallback_dims=1536)
            result = fn("hello")

        assert result == [0.1, 0.2, 0.3]

        ep._REDIS_PROBED = False


# =============================================================================
# build_embed_fn
# =============================================================================


class TestBuildEmbedFn:
    def setup_method(self) -> None:
        clear_provider_cache()

    def test_stub_provider_returns_callable_and_dims(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clean_env(monkeypatch)
        with patch.dict("sys.modules", {"llama_index.embeddings.huggingface": None}):
            fn, dims = build_embed_fn()
        assert callable(fn)
        assert dims == 1536
        result = fn("hello")
        assert isinstance(result, list)
        assert len(result) == 1536

    def test_litellm_provider_returns_callable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clean_env(monkeypatch)
        clear_provider_cache()
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key-12345")
        fn, dims = build_embed_fn()
        assert callable(fn)
        assert dims == 1536
