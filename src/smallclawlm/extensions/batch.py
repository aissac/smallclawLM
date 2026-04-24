"""BatchProcessor — Process multiple notebooks or tasks in parallel."""

import asyncio
import logging
from typing import Any

from smallclawlm.auth import get_auth

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process multiple NotebookLM tasks in parallel.

    Example:
        batch = BatchProcessor()
        batch.add_research("quantum computing", mode="fast")
        batch.add_research("fusion energy", mode="fast")
        batch.add_research("mRNA vaccines", mode="fast")
        results = batch.execute()
    """

    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self._tasks: list[dict] = []

    def add_research(self, query: str, mode: str = "deep", title: str | None = None) -> "BatchProcessor":
        self._tasks.append({"type": "research", "query": query, "mode": mode, "title": title or query})
        return self

    def add_source(self, url: str, notebook_id: str | None = None) -> "BatchProcessor":
        self._tasks.append({"type": "add_source", "url": url, "notebook_id": notebook_id})
        return self

    async def execute_async(self) -> list[dict]:
        from notebooklm import NotebookLMClient

        auth = await get_auth()
        results = []

        async with NotebookLMClient(auth) as client:
            sem = asyncio.Semaphore(self.max_concurrent)

            async def _run_task(task):
                async with sem:
                    return await self._process_task(client, task)

            coros = [_run_task(t) for t in self._tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        return [
            {"task": t, "result": r if not isinstance(r, Exception) else None, "error": str(r) if isinstance(r, Exception) else None}
            for t, r in zip(self._tasks, results)
        ]

    def execute(self) -> list[dict]:
        return asyncio.run(self.execute_async())

    async def _process_task(self, client, task: dict) -> dict:
        if task["type"] == "research":
            nb = await client.notebooks.create(task["title"])
            await client.research.start(
                notebook_id=nb.id, query=task["query"], source="web", mode=task["mode"]
            )
            poll = await client.research.poll(nb.id)
            return {"notebook_id": nb.id, "report": poll.get("report", "")}

        elif task["type"] == "add_source":
            nb_id = task.get("notebook_id")
            if not nb_id:
                notebooks = await client.notebooks.list()
                nb_id = notebooks[0].id if notebooks else None
            if not nb_id:
                nb = await client.notebooks.create("Batch Source Session")
                nb_id = nb.id
            src = await client.sources.add_url(nb_id, task["url"])
            return {"source_id": src.id, "status": src.status}

        else:
            raise ValueError(f"Unknown task type: {task['type']}")
