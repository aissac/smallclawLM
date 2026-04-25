"""NotebookRouter — Intelligently select the best NotebookLM notebook for a query.

Problem: SmallClawLM has 10+ notebooks but no way to know which one to use.
Every command requires manual --notebook-id. The router fixes this.

Scoring formula:
  score = (0.6 * title_match) + (0.3 * source_relevance) + (0.1 * recency)

  - title_match: Jaccard similarity between query tokens and notebook title tokens
  - source_relevance: number of sources normalized by max (more sources = richer notebook)
  - recency: decays from 1.0 (just used) to 0.0 over 7 days

Thresholds:
  - score >= 0.5  -> strong match (use this notebook)
  - score >= 0.3  -> weak match (use but log warning)
  - score < 0.3   -> no match (create new notebook)

The router caches notebook metadata locally so it doesn't hit the API every time.
Cache expires after 5 minutes or on explicit refresh.
"""

import asyncio
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".smallclawlm" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
METADATA_CACHE_FILE = CACHE_DIR / "notebook_metadata.json"
USAGE_TRACKER_FILE = CACHE_DIR / "notebook_usage.json"
CACHE_TTL = 300  # 5 minutes


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return set(text.split()) - {"", "a", "an", "the", "is", "of", "for", "and", "or", "in", "on", "to", "with"}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a or not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / len(union)


@dataclass
class NotebookMetadata:
    """Cached metadata for a single notebook."""
    id: str
    title: str
    source_count: int = 0
    last_used: float = 0.0  # Unix timestamp
    topics: list[str] = field(default_factory=list)  # Extracted topic keywords

    @property
    def title_tokens(self) -> set[str]:
        return _tokenize(self.title)


@dataclass
class RouteResult:
    """Result of notebook routing."""
    notebook_id: str
    title: str
    score: float
    match_level: str  # "strong", "weak", "none"
    created_new: bool = False


class NotebookRouter:
    """Select the best notebook for a query using title similarity, source richness, and recency.

    Usage:
        router = NotebookRouter()
        result = await router.route("research fusion energy")
        print(result.notebook_id, result.score, result.match_level)

        # Or sync:
        result = router.route_sync("research fusion energy")
    """

    def __init__(self, cache_ttl: int = CACHE_TTL):
        self._metadata: dict[str, NotebookMetadata] = {}
        self._cache_ttl = cache_ttl
        self._last_refresh: float = 0.0
        self._usage: dict[str, float] = {}  # notebook_id -> last_used timestamp
        self._load_usage()

    def _load_usage(self):
        """Load usage tracker from disk."""
        if USAGE_TRACKER_FILE.exists():
            try:
                data = json.loads(USAGE_TRACKER_FILE.read_text())
                self._usage = {k: v for k, v in data.items() if isinstance(v, (int, float))}
            except Exception:
                self._usage = {}

    def _save_usage(self):
        """Persist usage tracker to disk."""
        USAGE_TRACKER_FILE.write_text(json.dumps(self._usage, indent=2))

    def record_usage(self, notebook_id: str):
        """Mark a notebook as recently used."""
        now = time.time()
        self._usage[notebook_id] = now
        self._save_usage()

    async def _fetch_metadata(self) -> dict[str, NotebookMetadata]:
        """Fetch notebook metadata from NotebookLM API."""
        from smallclawlm.auth import get_auth
        from notebooklm import NotebookLMClient

        auth = await get_auth()
        async with NotebookLMClient(auth) as client:
            notebooks = await client.notebooks.list()
            metadata = {}
            for nb in notebooks:
                try:
                    sources = await client.sources.list(nb.id)
                    source_count = len(sources) if sources else 0
                except Exception:
                    source_count = 0

                # Extract topic keywords from title
                title_tokens = _tokenize(nb.title)
                # Remove generic words, keep domain-specific ones
                generic = {"notebook", "lm", "deep", "dive", "review", "apr", "2026", "2025", "new"}
                topics = list(title_tokens - generic)

                metadata[nb.id] = NotebookMetadata(
                    id=nb.id,
                    title=nb.title,
                    source_count=source_count,
                    last_used=self._usage.get(nb.id, 0.0),
                    topics=topics,
                )
            return metadata

    async def refresh(self, force: bool = False) -> None:
        """Refresh cached metadata. Skips if cache is fresh unless force=True."""
        now = time.time()
        if not force and self._metadata and (now - self._last_refresh) < self._cache_ttl:
            return
        self._metadata = await self._fetch_metadata()
        self._last_refresh = now
        # Save cache
        data = {
            nb_id: {
                "id": meta.id,
                "title": meta.title,
                "source_count": meta.source_count,
                "last_used": meta.last_used,
                "topics": meta.topics,
            }
            for nb_id, meta in self._metadata.items()
        }
        METADATA_CACHE_FILE.write_text(json.dumps(data, indent=2))
        logger.info(f"Refreshed notebook metadata: {len(self._metadata)} notebooks")

    async def route(self, query: str, create_if_none: bool = True) -> RouteResult:
        """Select the best notebook for a query.

        Args:
            query: User's task or question.
            create_if_none: If no notebook matches (score < 0.3), create a new one.

        Returns:
            RouteResult with notebook_id, score, and match_level.
        """
        await self.refresh()

        if not self._metadata:
            if create_if_none:
                return await self._create_notebook(query)
            raise RuntimeError("No notebooks available and create_if_none=False")

        query_tokens = _tokenize(query)
        max_sources = max((m.source_count for m in self._metadata.values()), default=1) or 1
        now = time.time()

        scored: list[tuple[str, NotebookMetadata, float]] = []
        for nb_id, meta in self._metadata.items():
            # Title match: Jaccard similarity
            title_score = _jaccard(query_tokens, meta.title_tokens)

            # Also match against extracted topics
            topics_tokens = set()
            for t in meta.topics:
                topics_tokens |= _tokenize(t)
            topic_score = _jaccard(query_tokens, topics_tokens) if topics_tokens else 0.0

            # Take the better of title vs topic match
            match_score = max(title_score, topic_score)

            # Source relevance: logarithmic scale (diminishing returns)
            # log(1 + count) / log(1 + max) normalizes 0..1 with compression
            source_score = (math.log(1 + meta.source_count) / math.log(1 + max_sources)
                            if max_sources > 0 else 0.0)

            # Recency: exponential decay over 7 days (604800 seconds)
            age = now - meta.last_used if meta.last_used else 604800 * 2  # never used = old
            recency = max(0.0, 1.0 - (age / 604800))

            # Weighted combination — recency gated by match_score
            # A recently-used but unrelated notebook shouldn't beat a well-matched one
            recency_boost = recency if match_score > 0.1 else recency * 0.1
            total = (0.65 * match_score) + (0.20 * source_score) + (0.15 * recency_boost)
            scored.append((nb_id, meta, total))

        # Sort by score descending
        scored.sort(key=lambda x: x[2], reverse=True)
        best_id, best_meta, best_score = scored[0]

        # Determine match level
        if best_score >= 0.5:
            match_level = "strong"
        elif best_score >= 0.3:
            match_level = "weak"
            logger.info(f"Weak notebook match for '{query}': '{best_meta.title}' (score={best_score:.2f})")
        else:
            match_level = "none"
            if create_if_none:
                logger.info(f"No good notebook match for '{query}' (best={best_score:.2f}). Creating new notebook.")
                return await self._create_notebook(query)

        # Record usage
        self.record_usage(best_id)

        return RouteResult(
            notebook_id=best_id,
            title=best_meta.title,
            score=best_score,
            match_level=match_level,
        )

    async def _create_notebook(self, query: str) -> RouteResult:
        """Create a new notebook for a query that has no good match."""
        from smallclawlm.auth import get_auth
        from notebooklm import NotebookLMClient

        # Generate a short title from the query
        title_words = query.split()[:6]
        title = " ".join(title_words)
        if len(title) > 60:
            title = title[:57] + "..."

        auth = await get_auth()
        async with NotebookLMClient(auth) as client:
            nb = await client.notebooks.create(title)

        # Add to metadata cache
        now = time.time()
        self._metadata[nb.id] = NotebookMetadata(
            id=nb.id,
            title=title,
            source_count=0,
            last_used=now,
            topics=list(_tokenize(title)),
        )
        self.record_usage(nb.id)

        logger.info(f"Created new notebook '{title}' (id: {nb.id}) for query '{query}'")
        return RouteResult(
            notebook_id=nb.id,
            title=title,
            score=0.0,
            match_level="none",
            created_new=True,
        )

    # --- Sync wrappers ---

    def route_sync(self, query: str, create_if_none: bool = True) -> RouteResult:
        """Synchronous wrapper for route()."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(lambda: asyncio.run(self.route(query, create_if_none))).result()
        else:
            return asyncio.run(self.route(query, create_if_none))

    def refresh_sync(self, force: bool = False) -> None:
        """Synchronous wrapper for refresh()."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(lambda: asyncio.run(self.refresh(force))).result()
        else:
            asyncio.run(self.refresh(force))