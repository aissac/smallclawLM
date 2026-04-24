"""NLMTools - smolagents Tool wrappers around NotebookLM operations.

Architecture:
- Uses daemon thread event loop via run_coroutine_threadsafe
- forward() is sync (smolagents requirement), internally calls async APIs
- Errors caught and returned as strings so agent can adapt its plan
"""

import asyncio
import logging
import threading
from smolagents import Tool
from smallclawlm.auth import get_auth

logger = logging.getLogger(__name__)


class NotebookLMToolBase(Tool):
    """Base class for all NotebookLM tools. Uses daemon event loop for async/sync bridge."""
    _client = None
    _auth = None
    _notebook_id: str | None = None
    _loop: asyncio.AbstractEventLoop | None = None
    _thread = None

    def _run_async(self, coro):
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._thread.start()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def _get_client(self):
        from notebooklm import NotebookLMClient
        if self._client is None:
            self._auth = await get_auth()
            self._client = NotebookLMClient(self._auth)
            await self._client.__aenter__()
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
    name = "deep_research"
    description = "Run deep web research on a topic. Returns a comprehensive report with cited sources."
    inputs = {
        "query": {"type": "string", "description": "Research question or topic"},
        "mode": {"type": "string", "description": "Research depth: fast or deep (default: deep)"},
    }
    output_type = "string"

    def forward(self, query: str, mode: str = "deep") -> str:
        async def _do():
            client = await self._get_client()
            nb_id = await self._get_notebook_id()
            await client.research.start(notebook_id=nb_id, query=query, source="web", mode=mode)
            poll = await client.research.poll(nb_id)
            return poll.get("report", "Research completed but no report generated.")
        try:
            return self._run_async(_do())
        except Exception as e:
            return f"Research error: {e}"


class GeneratePodcastTool(NotebookLMToolBase):
    name = "generate_podcast"
    description = "Generate an audio overview podcast from the notebook sources."
    inputs = {"instructions": {"type": "string", "description": "Custom instructions (optional)", "nullable": True}}
    output_type = "string"

    def forward(self, instructions: str | None = None) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = await self._get_notebook_id()
            result = await client.artifacts.generate_audio(nb_id, instructions=instructions or "")
            return f"Podcast generation started. Task ID: {result.task_id}"
        try:
            return self._run_async(_do())
        except Exception as e:
            return f"Generation error: {e}"


class GenerateVideoTool(NotebookLMToolBase):
    name = "generate_video"
    description = "Generate a video explainer from the notebook sources."
    inputs = {"style": {"type": "string", "description": "Video style: whiteboard or animated (default: whiteboard)"}}
    output_type = "string"

    def forward(self, style: str = "whiteboard") -> str:
        async def _do():
            client = await self._get_client()
            nb_id = await self._get_notebook_id()
            result = await client.artifacts.generate_video(nb_id, style=style)
            return f"Video generation started. Task ID: {result.task_id}"
        try:
            return self._run_async(_do())
        except Exception as e:
            return f"Generation error: {e}"


class GenerateQuizTool(NotebookLMToolBase):
    name = "generate_quiz"
    description = "Generate a quiz from the notebook sources."
    inputs = {"difficulty": {"type": "string", "description": "Quiz difficulty: easy, medium, or hard (default: medium)"}}
    output_type = "string"

    def forward(self, difficulty: str = "medium") -> str:
        async def _do():
            client = await self._get_client()
            nb_id = await self._get_notebook_id()
            result = await client.artifacts.generate_quiz(nb_id, difficulty=difficulty)
            return f"Quiz generation started. Task ID: {result.task_id}"
        try:
            return self._run_async(_do())
        except Exception as e:
            return f"Generation error: {e}"


class GenerateMindMapTool(NotebookLMToolBase):
    name = "generate_mind_map"
    description = "Generate a visual mind map from the notebook sources. Returns immediately."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = await self._get_notebook_id()
            result = await client.artifacts.generate_mind_map(nb_id)
            return str(result)
        try:
            return self._run_async(_do())
        except Exception as e:
            return f"Generation error: {e}"


class GenerateReportTool(NotebookLMToolBase):
    name = "generate_report"
    description = "Generate a structured report from the notebook sources."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = await self._get_notebook_id()
            result = await client.artifacts.generate_report(nb_id)
            return str(result)
        try:
            return self._run_async(_do())
        except Exception as e:
            return f"Generation error: {e}"


class AddSourceTool(NotebookLMToolBase):
    name = "add_source"
    description = "Add a source (URL, YouTube, PDF) to the notebook."
    inputs = {
        "url": {"type": "string", "description": "URL of the source to add"},
        "title": {"type": "string", "description": "Optional title", "nullable": True},
    }
    output_type = "string"

    def forward(self, url: str, title: str | None = None) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = await self._get_notebook_id()
            source = await client.sources.add_url(nb_id, url, title=title)
            return f"Added source: {source.id} (status: {source.status})"
        try:
            return self._run_async(_do())
        except Exception as e:
            return f"Source error: {e}"


class ListSourcesTool(NotebookLMToolBase):
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
        try:
            return self._run_async(_do())
        except Exception as e:
            return f"List error: {e}"


class CreateNotebookTool(NotebookLMToolBase):
    name = "create_notebook"
    description = "Create a new NotebookLM notebook."
    inputs = {"title": {"type": "string", "description": "Title for the new notebook"}}
    output_type = "string"

    def forward(self, title: str) -> str:
        async def _do():
            client = await self._get_client()
            nb = await client.notebooks.create(title)
            self._notebook_id = nb.id
            return f"Created notebook: {nb.id} ({nb.title})"
        try:
            return self._run_async(_do())
        except Exception as e:
            return f"Create error: {e}"


class DedupSourcesTool(NotebookLMToolBase):
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
        try:
            return self._run_async(_do())
        except Exception as e:
            return f"Dedup error: {e}"


class AskNotebookTool(NotebookLMToolBase):
    name = "ask_notebook"
    description = "Ask a question about the notebook sources. Returns a grounded answer with citations."
    inputs = {"question": {"type": "string", "description": "Question to ask about the sources"}}
    output_type = "string"

    def forward(self, question: str) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = await self._get_notebook_id()
            result = await client.chat.ask(nb_id, question)
            return result.answer if hasattr(result, "answer") else str(result)
        try:
            return self._run_async(_do())
        except Exception as e:
            return f"Ask error: {e}"


DEFAULT_TOOLS = [
    DeepResearchTool, GeneratePodcastTool, GenerateVideoTool,
    GenerateQuizTool, GenerateMindMapTool, GenerateReportTool,
    AddSourceTool, ListSourcesTool, CreateNotebookTool,
    DedupSourcesTool, AskNotebookTool,
]
