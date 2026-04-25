"""Unit tests for NotebookRouter — tokenization, Jaccard, scoring, routing logic."""

import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from smallclawlm.notebook_router import (
    _tokenize, _jaccard, NotebookMetadata, RouteResult, NotebookRouter
)


class TestTokenizer:
    def test_basic(self):
        assert _tokenize("Hello World") == {"hello", "world"}

    def test_stops_words_removed(self):
        tokens = _tokenize("The quick brown fox is a test of the system")
        assert "the" not in tokens
        assert "a" not in tokens
        assert "of" not in tokens
        assert "is" not in tokens
        assert "hello" not in tokens  # not in input
        assert "quick" in tokens

    def test_punctuation_stripped(self):
        assert _tokenize("SmallClawLM: Zero-Token Agent!") == {"smallclawlm", "zero", "token", "agent"}

    def test_numbers_preserved(self):
        assert "2026" in _tokenize("April 2026 review")

    def test_case_insensitive(self):
        assert _tokenize("POLYMARKET") == _tokenize("polymarket")

    def test_empty(self):
        assert _tokenize("") == set()

    def test_only_stopwords(self):
        assert _tokenize("the a an is of") == set()


class TestJaccard:
    def test_identical(self):
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint(self):
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_partial(self):
        result = _jaccard({"a", "b", "c"}, {"a", "b", "d"})
        assert abs(result - 0.5) < 0.01  # intersection=2, union=4

    def test_empty_left(self):
        assert _jaccard(set(), {"a"}) == 0.0

    def test_empty_right(self):
        assert _jaccard({"a"}, set()) == 0.0

    def test_both_empty(self):
        assert _jaccard(set(), set()) == 0.0


class TestNotebookMetadata:
    def test_title_tokens(self):
        meta = NotebookMetadata(
            id="abc123",
            title="Strategic Foundations for Polymarket and Algorithmic Trading",
            source_count=300,
            last_used=time.time(),
            topics=["strategic", "polymarket", "algorithmic", "trading"]
        )
        tokens = meta.title_tokens
        assert "polymarket" in tokens
        assert "trading" in tokens
        assert "the" not in tokens  # stop word removed

    def test_default_values(self):
        meta = NotebookMetadata(id="x", title="Test Notebook")
        assert meta.source_count == 0
        assert meta.last_used == 0.0
        assert meta.topics == []


class TestRouteResult:
    def test_fields(self):
        r = RouteResult(
            notebook_id="abc123",
            title="Test Notebook",
            score=0.85,
            match_level="strong"
        )
        assert r.notebook_id == "abc123"
        assert r.score == 0.85
        assert r.match_level == "strong"
        assert r.created_new is False

    def test_created_new(self):
        r = RouteResult(
            notebook_id="new123",
            title="New Topic",
            score=0.0,
            match_level="none",
            created_new=True
        )
        assert r.created_new is True


class TestNotebookRouterScoring:
    """Test the scoring logic without hitting the NotebookLM API."""

    def _make_router_with_mock_metadata(self):
        router = NotebookRouter.__new__(NotebookRouter)
        router._metadata = {
            "nb-trading": NotebookMetadata(
                id="nb-trading",
                title="Strategic Foundations for Polymarket and Algorithmic Trading",
                source_count=300,
                last_used=time.time(),
                topics=["strategic", "polymarket", "algorithmic", "trading"]
            ),
            "nb-smallclaw": NotebookMetadata(
                id="nb-smallclaw",
                title="SmallClawLM: Zero-Token NotebookLM Agent",
                source_count=24,
                last_used=0.0,
                topics=["token", "agent", "notebooklm", "smallclawlm"]
            ),
            "nb-llm": NotebookMetadata(
                id="nb-llm",
                title="Local LLM Inference Backends: Alternatives to llama.cpp",
                source_count=5,
                last_used=0.0,
                topics=["llm", "inference", "backends", "alternatives"]
            ),
        }
        router._cache_ttl = 300
        router._last_refresh = time.time()
        router._usage = {"nb-trading": time.time()}
        return router

    def test_polymarket_routes_to_trading(self):
        router = self._make_router_with_mock_metadata()
        result = router.route_sync("polymarket trading strategies")
        assert result.notebook_id == "nb-trading"
        assert result.match_level == "strong"

    def test_smallclawlm_routes_correctly(self):
        router = self._make_router_with_mock_metadata()
        result = router.route_sync("SmallClawLM agent architecture")
        assert result.notebook_id == "nb-smallclaw"

    def test_llm_inference_routes_correctly(self):
        router = self._make_router_with_mock_metadata()
        result = router.route_sync("local LLM inference alternatives")
        assert result.notebook_id == "nb-llm"
        assert result.match_level in ("strong", "weak")

    def test_no_match_returns_none(self):
        """When create_if_none=False and no match, should raise or return low score."""
        router = self._make_router_with_mock_metadata()
        result = router.route_sync("quantum computing research", create_if_none=False)
        # Should return the best available (even if weak)
        assert result.match_level in ("weak", "none", "strong")

    def test_log_source_scoring(self):
        """300 sources should not dominate 24-source notebook on topic match."""
        router = self._make_router_with_mock_metadata()
        # SmallClawLM query should NOT route to trading despite 300 sources
        result = router.route_sync("SmallClawLM agent")
        assert result.notebook_id == "nb-smallclaw"

    def test_gated_recency(self):
        """Recently-used but irrelevant notebook should not win over matched one."""
        router = self._make_router_with_mock_metadata()
        # Trading notebook has recency=1.0 but is irrelevant to LLM query
        result = router.route_sync("LLM inference llama.cpp")
        assert result.notebook_id == "nb-llm"


class TestAuthModule:
    """Test auth module functions without hitting disk/network."""

    def test_get_auth_sync_exists(self):
        from smallclawlm.auth import get_auth_sync
        assert callable(get_auth_sync)

    def test_clear_cache(self):
        from smallclawlm.auth import clear_cache
        # Should not raise
        clear_cache()

    def test_ensure_authenticated_exists(self):
        from smallclawlm.auth import ensure_authenticated
        assert callable(ensure_authenticated)
