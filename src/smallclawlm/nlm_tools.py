"""NLMTools — smolagents Tool wrappers around NotebookLM operations.

Each tool maps to a NotebookLM API operation, allowing the agent to act
on the user's behalf: research topics, add sources, generate artifacts,
manage notebooks, and more.
"""

import asyncio
import logging
from typing import Any

from smolagents import Tool

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine from sync smolagents context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an existing event loop — create a new one in a thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


class NotebookLMToolBase(Tool):
    """Base class for all NotebookLM tools with shared client management."""

    _client = None
    _auth = None
    _notebook_id: str | None = None

    async def _get_client(self):
        from notebooklm import NotebookLMClient
        from smallclawlm.auth import get_auth

        if self._client is None:
            self._auth = await get_auth()
            self._client = NotebookLMClient(self._auth)
        return self._client

    async def _get_notebook_id(self):
        if self._notebook_id is None:
            client = await self._get_client()
            notebooks = await client.notebooks.list()
            if notebooks:
                self._notebook_id = notebooks[0].id
            else:
                nb = await client.notebooks.create("SmallClawLM Session")
                self._notebook_id = nb.id
        return self._notebook_id


class DeepResearchTool(NotebookLMToolBase):
    """Run deep web research on a topic using NotebookLM."""
    name = "deep_research"
    description = "Run deep web research on a topic. Returns a comprehensive report with cited sources."
    inputs = {
        "query": {"type": "string", "description": "Research question or topic"},
        "mode": {"type": "string", "description": "Research depth: 'fast' or 'deep' (default: deep)"},
    }
    output_type = "string"

    def forward(self, query: str, mode: str = "deep") -> str:
        async def _do():
            client = await self._get_client()
            nb_id = await self._get_notebook_id()
            result = await client.research.start(
                notebook_id=nb_id,
                query=query,
                source="web",
                mode=mode,
            )
            # Poll until done
            poll = await client.research.poll(nb_id)
            return poll.get("report", "Research completed but no report generated.")
        return _run_async(_do())


class GeneratePodcastTool(NotebookLMToolBase):
    """Generate an audio overview podcast from notebook sources."""
    name = "generate_podcast"
    description = "Generate an audio overview podcast from the notebook's sources."
    inputs = {
        "instructions": {"type": "string", "description": "Custom instructions for the podcast (optional)", "nullable": True},
    }
    output_type = "string"

    def forward(self, instructions: str | None = None) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = await self._get_notebook_id()
            result = await client.artifacts.generate_audio(
                nb_id, instructions=instructions or ""
            )
            return f"Podcast generation started. Task ID: {result.task_id}"
        return _run_async(_do())


class GenerateVideoTool(NotebookLMToolBase):
    """Generate a video explainer from notebook sources."""
    name = "generate_video"
    description = "Generate a video explainer from the notebook's sources."
    inputs = {
        "style": {"type": "string", "description": "Video style: 'whiteboard' or 'animated' (default: whiteboard)"},
    }
    output_type = "string"

    def forward(self, style: str = "whiteboard") -> str:
        async def _do():
            client = await self._get_client()
            nb_id = await self._get_notebook_id()
            result = await client.artifacts.generate_video(
                nb_id, style=style
            )
            return f"Video generation started. Task ID: {result.task_id}"
        return _run_async(_do())


class GenerateQuizTool(NotebookLMToolBase):
    """Generate a quiz from notebook sources."""
    name = "generate_quiz"
    description = "Generate a quiz from the notebook's sources."
    inputs = {
        "difficulty": {"type": "string", "description": "Quiz difficulty: 'easy', 'medium', or 'hard' (default: medium)"},
    }
    output_type = "string"

    def forward(self, difficulty: str = "medium") -> str:
        async def _do():
            client = await self._get_client()
            nb_id = await self._get_notebook_id()
            result = await client.artifacts.generate_quiz(
                nb_id, difficulty=difficulty
            )
            return f"Quiz generation started. Task ID: {result.task_id}"
        return _run_async(_do())


class GenerateFlashcardsTool(NotebookLMToolBase):
    """Generate study flashcards from notebook sources."""
    name = "generate_flashcards"
    description = "Generate study flashcards from the notebook's sources."
    inputs = {
        "quantity": {"type": "string", "description": "Number of flashcards: 'few', 'more', or 'many' (default: more)"},
    }
    output_type = "string"

    def forward(self, quantity: str = "more") -> str:
        async def _do():
            client = await self._get_client()
            nb_id = await self._get_notebook_id()
            result = await client.artifacts.generate_flashcards(
                nb_id, quantity=quantity
            )
            return f"Flashcard generation started. Task ID: {result.task_id}"
        return _run_async(_do())


class GenerateMindMapTool(NotebookLMToolBase):
    """Generate a mind map from notebook sources."""
    name = "generate_mind_map"
    description = "Generate a visual mind map from the notebook's sources."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = await self._get_notebook_id()
            result = await client.artifacts.generate_mind_map(nb_id)
            return str(result)
        return _run_async(_do())


class GenerateReportTool(NotebookLMToolBase):
    """Generate a structured report from notebook sources."""
    name = "generate_report"
    description = "Generate a structured report from the notebook's sources."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = await self._get_notebook_id()
            result = await client.artifacts.generate_report(nb_id)
            return str(result)
        return _run_async(_do())


class AddSourceTool(NotebookLMToolBase):
    """Add a source (URL, YouTube, PDF) to the notebook."""
    name = "add_source"
    description = "Add a source to the notebook. Supports URLs, YouTube videos, and PDFs."
    inputs = {
        "url": {"type": "string", "description": "URL of the source to add"},
        "title": {"type": "string", "description": "Optional title for the source", "nullable": True},
    }
    output_type = "string"

    def forward(self, url: str, title: str | None = None) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = await self._get_notebook_id()
            source = await client.sources.add_url(nb_id, url, title=title)
            return f"Added source: {source.id} (status: {source.status})"
        return _run_async(_do())


class ListSourcesTool(NotebookLMToolBase):
    """List all sources in the current notebook."""
    name = "list_sources"
    description = "List all sources in the current notebook."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = await self._get_notebook_id()
            sources = await client.sources.list(nb_id)
            lines = [f"{s.id} | {s.title} | {s.status}" for s in sources]
            return "\n".join(lines) if lines else "No sources found."
        return _run_async(_do())


class CreateNotebookTool(NotebookLMToolBase):
    """Create a new NotebookLM notebook."""
    name = "create_notebook"
    description = "Create a new NotebookLM notebook for research and content generation."
    inputs = {
        "title": {"type": "string", "description": "Title for the new notebook"},
    }
    output_type = "string"

    def forward(self, title: str) -> str:
        async def _do():
            client = await self._get_client()
            nb = await client.notebooks.create(title)
            self._notebook_id = nb.id
            return f"Created notebook: {nb.id} ({nb.title})"
        return _run_async(_do())


class DedupSourcesTool(NotebookLMToolBase):
    """Remove duplicate sources from a notebook."""
    name = "dedup_sources"
    description = "Find and remove duplicate sources from the notebook."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = await self._get_notebook_id()
            sources = await client.sources.list(nb_id)

            seen = {}
            dup_ids = []
            for s in sources:
                key = getattr(s, "url", None) or s.title or s.id
                if key in seen:
                    dup_ids.append(s.id)
                else:
                    seen[key] = s.id

            if not dup_ids:
                return "No duplicates found."

            for did in dup_ids:
                await client.sources.delete(nb_id, did)

            return f"Removed {len(dup_ids)} duplicate sources."
        return _run_async(_do())
