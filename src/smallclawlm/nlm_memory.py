"""NLMMemory — NotebookLM as the agent's persistent knowledge system.

Every agent session is backed by a NotebookLM notebook. Facts, research results,
and decisions are automatically synced as notebook sources. The agent doesn't
"call" NotebookLM — it *lives* in NotebookLM.

Three-layer architecture:
  Layer 1: REFLEX  — SmolLM3 local model (10.7 tok/s, tool selection)
  Layer 2: COGNITION — NotebookLM chat (reasoning, synthesis, citations)
  Layer 3: MEMORY — NotebookLM notebooks (persistent, growing knowledge)

Key insight: AgentMemory was a sliding window that forgot everything.
NLMMemory is a notebook that remembers everything and grows smarter over time.
"""

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MEMORY_DIR = Path.home() / ".smallclawlm" / "memory"


class NLMMemory:
    """NotebookLM-backed persistent memory that grows with every session.

    Unlike AgentMemory (sliding window, ~2K chars, forgets old facts),
    NLMMemory stores everything in a NotebookLM notebook. Facts become
    sources. Research becomes context. The notebook IS the memory.
    """

    def __init__(
        self,
        notebook_id: str | None = None,
        notebook_title: str = "SmallClawLM Memory",
        auto_create: bool = True,
        max_local_cache: int = 2000,
    ):
        self.notebook_id = notebook_id
        self.notebook_title = notebook_title
        self._auto_create = auto_create
        self._client = None
        self._auth = None
        self._loop = None
        self._thread = None
        self._local_cache: list[str] = []
        self._max_local_cache = max_local_cache
        self._state_file = MEMORY_DIR / "notebook_map.json"
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    def _ensure_loop(self):
        if self._loop is None or not self._loop.is_running():
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._thread.start()
        return self._loop

    def _run(self, coro):
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=120)

    async def _get_client(self):
        if self._client is None:
            from smallclawlm.auth import get_auth
            from notebooklm.client import NotebookLMClient
            self._auth = await get_auth()
            self._client = NotebookLMClient(self._auth)
            await self._client.__aenter__()
        return self._client

    async def _ensure_notebook(self):
        if self.notebook_id:
            return self.notebook_id
        if self._state_file.exists():
            try:
                mapping = json.loads(self._state_file.read_text())
                if mapping.get("default_notebook"):
                    self.notebook_id = mapping["default_notebook"]
                    return self.notebook_id
            except Exception:
                pass
        if not self._auto_create:
            raise RuntimeError("No notebook ID and auto_create=False")
        client = await self._get_client()
        nb = await client.notebooks.create(self.notebook_title)
        self.notebook_id = nb.id
        self._state_file.write_text(json.dumps({
            "default_notebook": self.notebook_id,
            "title": self.notebook_title,
            "created": time.strftime("%Y-%m-%d %H:%M"),
        }, indent=2))
        logger.info(f"Created memory notebook: {self.notebook_id}")
        return self.notebook_id

    # ─── Sync public API ───

    def add(self, fact: str) -> None:
        """Add a fact. Auto-syncs to NotebookLM as a text source."""
        timestamp = time.strftime("%H:%M")
        entry = f"[{timestamp}] {fact}"
        self._local_cache.append(entry)
        self._prune_local()
        try:
            self._run(self._add_source_async(entry))
        except Exception as e:
            logger.debug(f"Failed to sync fact: {e}")

    def add_observation(self, tool: str, result: str, max_len: int = 500) -> None:
        truncated = result[:max_len] + "..." if len(result) > max_len else result
        self.add(f"{tool} -> {truncated}")

    def add_decision(self, thought: str, action: str) -> None:
        short = thought[:120] + "..." if len(thought) > 120 else thought
        self.add(f"Decided: {short} -> {action}")

    def add_research(self, query: str, report: str) -> None:
        """Add a full research report as a source (not truncated)."""
        try:
            self._run(self._add_research_async(query, report))
        except Exception as e:
            logger.warning(f"Failed to sync research: {e}")
        summary = report[:200] + "..." if len(report) > 200 else report
        self.add(f"Research on '{query}': {summary}")

    def query(self, question: str) -> str:
        """Query the notebook using NotebookLM chat — the cognition layer."""
        return self._run(self._query_async(question))

    def render(self) -> str:
        if not self._local_cache:
            return ""
        return "\n".join(self._local_cache)

    @property
    def facts(self) -> list[str]:
        return self._local_cache

    # ─── Async implementations ───

    async def _add_source_async(self, text: str):
        try:
            nb_id = await self._ensure_notebook()
            client = await self._get_client()
            await client.sources.add_text(
                nb_id,
                title=f"Memory {time.strftime('%H:%M:%S')}",
                content=text,
            )
        except Exception as e:
            logger.debug(f"Sync failed: {e}")

    async def _add_research_async(self, query: str, report: str):
        try:
            nb_id = await self._ensure_notebook()
            client = await self._get_client()
            await client.sources.add_text(
                nb_id,
                title=f"Research: {query}",
                content=report,
            )
        except Exception as e:
            logger.warning(f"Research sync failed: {e}")

    async def _query_async(self, question: str) -> str:
        try:
            nb_id = await self._ensure_notebook()
            client = await self._get_client()
            result = await client.chat.ask(nb_id, question)
            answer = result.answer if hasattr(result, "answer") else str(result)
            self.add(f"Q: {question[:80]} -> A: {answer[:80]}")
            return answer
        except Exception as e:
            return f"Memory query failed: {e}"

    def _prune_local(self):
        while len(self._local_cache) > 50:
            self._local_cache.pop(0)
        while len(self.render()) > self._max_local_cache and len(self._local_cache) > 5:
            self._local_cache.pop(0)

    def load_session(self, session_id: str):
        return self._run(self._load_session_async(session_id))

    async def _load_session_async(self, session_id: str):
        mapping_file = MEMORY_DIR / f"session_{session_id}.json"
        if mapping_file.exists():
            try:
                data = json.loads(mapping_file.read_text())
                self.notebook_id = data["notebook_id"]
                self.notebook_title = data.get("title", self.notebook_title)
                logger.info(f"Resumed session notebook: {self.notebook_id}")
                return
            except Exception:
                pass
        client = await self._get_client()
        nb = await client.notebooks.create(f"SmallClawLM Session: {session_id}")
        self.notebook_id = nb.id
        mapping_file.write_text(json.dumps({
            "notebook_id": nb.id,
            "title": f"SmallClawLM Session: {session_id}",
            "created": time.strftime("%Y-%m-%d %H:%M"),
        }, indent=2))

    async def close(self):
        if self._client:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass
            self._client = None
