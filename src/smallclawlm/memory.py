"""Memory — Persistent agent memory backed by NotebookLM notebooks.

Stores agent conversation history and learned facts as NotebookLM notes,
providing cross-session persistence without any external database.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from smallclawlm.auth import get_auth

logger = logging.getLogger(__name__)


class NLMMemory:
    """Agent memory stored as NotebookLM notes.

    Instead of requiring a separate vector DB or memory service,
    SmallClawLM uses NotebookLM notebooks as the memory backend.
    Each "memory" is a note in the notebook, queryable via NLM chat.
    """

    def __init__(self, notebook_id: str | None = None):
        self.notebook_id = notebook_id
        self._client = None

    async def _ensure(self):
        from notebooklm import NotebookLMClient
        if self._client is None:
            auth = await get_auth()
            self._client = NotebookLMClient(auth)

        if self.notebook_id is None:
            notebooks = await self._client.notebooks.list()
            title_match = [nb for nb in notebooks if "SmallClawLM Memory" in nb.title]
            if title_match:
                self.notebook_id = title_match[0].id
            else:
                nb = await self._client.notebooks.create("SmallClawLM Memory")
                self.notebook_id = nb.id

    async def save(self, key: str, value: str) -> str:
        """Save a memory as a NotebookLM note."""
        await self._ensure()
        note = await self._client.notes.create(
            self.notebook_id,
            title=key,
            content=f"[{datetime.now().isoformat()}] {value}",
        )
        return note.id

    async def recall(self, query: str) -> str:
        """Recall memories by asking NotebookLM chat."""
        await self._ensure()
        result = await self._client.chat.ask(
            self.notebook_id,
            f"From my saved notes, recall: {query}",
        )
        return result.answer if hasattr(result, "answer") else str(result)

    async def list_memories(self) -> list[dict]:
        """List all saved memory notes."""
        await self._ensure()
        notes = await self._client.notes.list(self.notebook_id)
        return [{"id": n.id, "title": n.title} for n in notes]

    def save_sync(self, key: str, value: str) -> str:
        return asyncio.run(self.save(key, value))

    def recall_sync(self, query: str) -> str:
        return asyncio.run(self.recall(query))
