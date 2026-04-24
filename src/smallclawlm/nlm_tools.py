"""NLMTools — smolagents Tool wrappers around NotebookLM operations.

One agent = one notebook. These tools operate on the agent's single notebook.
All tools use daemon event loop for async/sync bridging.

Key insight from NotebookLM analysis:
- Source operations MUST use wait=True to block until processing finishes
- Auth errors trigger refresh_auth() before retrying
- Errors are returned as strings so CodeAgent can self-correct
"""

import asyncio
import logging
import threading
from smolagents import Tool

from smallclawlm.auth import get_auth

logger = logging.getLogger(__name__)


class _SharedLoop:
    """Singleton daemon event loop shared by all tools."""
    _loop: asyncio.AbstractEventLoop | None = None
    _thread: threading.Thread | None = None

    @classmethod
    def get_loop(cls) -> asyncio.AbstractEventLoop:
        if cls._loop is None or not cls._loop.is_running():
            cls._loop = asyncio.new_event_loop()
            cls._thread = threading.Thread(target=cls._loop.run_forever, daemon=True)
            cls._thread.start()
        return cls._loop

    @classmethod
    def run(cls, coro):
        loop = cls.get_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()


class _ClientMixin:
    """Mixin providing lazy NotebookLM client initialization."""
    _client = None
    _auth = None

    async def _get_client(self):
        if self._client is None:
            self._auth = await get_auth()
            from notebooklm import NotebookLMClient
            self._client = NotebookLMClient(self._auth)
            await self._client.__aenter__()
        return self._client


class DeepResearchTool(Tool, _ClientMixin):
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
            nb_id = getattr(self, '_notebook_id', None)
            if not nb_id:
                notebooks = await client.notebooks.list()
                nb_id = notebooks[0].id if notebooks else (await client.notebooks.create("Research")).id
            await client.research.start(notebook_id=nb_id, query=query, source="web", mode=mode)
            poll = await client.research.poll(nb_id)
            return poll.get("report", "Research completed but no report generated.")
        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Research error: {e}"


class AskNotebookTool(Tool, _ClientMixin):
    name = "ask_notebook"
    description = "Ask a question about the notebook sources. Returns a grounded answer with citations."
    inputs = {
        "question": {"type": "string", "description": "Question to ask about the sources"},
    }
    output_type = "string"

    def forward(self, question: str) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = getattr(self, '_notebook_id', None)
            if not nb_id:
                notebooks = await client.notebooks.list()
                nb_id = notebooks[0].id if notebooks else (await client.notebooks.create("SmallClawLM Session")).id
            result = await client.chat.ask(nb_id, question, conversation_id=None)
            return result.answer if hasattr(result, "answer") else str(result)
        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Ask error: {e}"


class GeneratePodcastTool(Tool, _ClientMixin):
    name = "generate_podcast"
    description = "Generate an audio overview podcast from the notebook sources."
    inputs = {
        "instructions": {"type": "string", "description": "Custom instructions for the podcast (optional)", "nullable": True},
    }
    output_type = "string"

    def forward(self, instructions: str | None = None) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = getattr(self, '_notebook_id', None)
            if not nb_id:
                notebooks = await client.notebooks.list()
                nb_id = notebooks[0].id if notebooks else (await client.notebooks.create("Podcast")).id
            result = await client.artifacts.generate_audio(nb_id, instructions=instructions or "")
            return f"Podcast generation started. Task ID: {result.task_id}"
        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Generation error: {e}"


class GenerateVideoTool(Tool, _ClientMixin):
    name = "generate_video"
    description = "Generate a video explainer from the notebook sources."
    inputs = {
        "style": {"type": "string", "description": "Video style: whiteboard or animated (default: whiteboard)"},
    }
    output_type = "string"

    def forward(self, style: str = "whiteboard") -> str:
        async def _do():
            client = await self._get_client()
            nb_id = getattr(self, '_notebook_id', None)
            if not nb_id:
                notebooks = await client.notebooks.list()
                nb_id = notebooks[0].id if notebooks else (await client.notebooks.create("Video")).id
            result = await client.artifacts.generate_video(nb_id, style=style)
            return f"Video generation started. Task ID: {result.task_id}"
        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Generation error: {e}"


class GenerateQuizTool(Tool, _ClientMixin):
    name = "generate_quiz"
    description = "Generate a quiz from the notebook sources."
    inputs = {
        "difficulty": {"type": "string", "description": "Quiz difficulty: easy, medium, or hard (default: medium)"},
    }
    output_type = "string"

    def forward(self, difficulty: str = "medium") -> str:
        async def _do():
            client = await self._get_client()
            nb_id = getattr(self, '_notebook_id', None)
            if not nb_id:
                notebooks = await client.notebooks.list()
                nb_id = notebooks[0].id if notebooks else (await client.notebooks.create("Quiz")).id
            result = await client.artifacts.generate_quiz(nb_id, difficulty=difficulty)
            return f"Quiz generation started. Task ID: {result.task_id}"
        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Generation error: {e}"


class GenerateMindMapTool(Tool, _ClientMixin):
    name = "generate_mind_map"
    description = "Generate a visual mind map from the notebook sources."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = getattr(self, '_notebook_id', None)
            if not nb_id:
                notebooks = await client.notebooks.list()
                nb_id = notebooks[0].id if notebooks else (await client.notebooks.create("MindMap")).id
            result = await client.artifacts.generate_mind_map(nb_id)
            return str(result)
        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Generation error: {e}"


class GenerateReportTool(Tool, _ClientMixin):
    name = "generate_report"
    description = "Generate a structured report from the notebook sources."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = getattr(self, '_notebook_id', None)
            if not nb_id:
                notebooks = await client.notebooks.list()
                nb_id = notebooks[0].id if notebooks else (await client.notebooks.create("Report")).id
            result = await client.artifacts.generate_report(nb_id)
            return str(result)
        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Generation error: {e}"


class AddSourceTool(Tool, _ClientMixin):
    name = "add_source"
    description = "Add a source (URL, YouTube, PDF) to the notebook. Waits for processing to complete."
    inputs = {
        "url": {"type": "string", "description": "URL of the source to add"},
        "title": {"type": "string", "description": "Optional title for the source", "nullable": True},
    }
    output_type = "string"

    def forward(self, url: str, title: str | None = None) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = getattr(self, '_notebook_id', None)
            if not nb_id:
                notebooks = await client.notebooks.list()
                nb_id = notebooks[0].id if notebooks else (await client.notebooks.create("Session")).id
            # MUST wait for source to finish processing before next chat call
            source = await client.sources.add_url(nb_id, url, title=title, wait=True)
            return f"Added source: {source.id} (title: {getattr(source, 'title', 'N/A')})"
        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Source error: {e}"


class ListSourcesTool(Tool, _ClientMixin):
    name = "list_sources"
    description = "List all sources in the current notebook."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = getattr(self, '_notebook_id', None)
            if not nb_id:
                notebooks = await client.notebooks.list()
                nb_id = notebooks[0].id if notebooks else (await client.notebooks.create("Session")).id
            sources = await client.sources.list(nb_id)
            lines = [f"{s.id} | {getattr(s, 'title', 'N/A')} | {s.status}" for s in sources]
            return "\n".join(lines) if lines else "No sources found."
        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"List error: {e}"


class CreateNotebookTool(Tool, _ClientMixin):
    name = "create_notebook"
    description = "Create a new NotebookLM notebook."
    inputs = {
        "title": {"type": "string", "description": "Title for the new notebook"},
    }
    output_type = "string"

    def forward(self, title: str) -> str:
        async def _do():
            client = await self._get_client()
            nb = await client.notebooks.create(title)
            return f"Created notebook: {nb.id} ({nb.title})"
        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Create error: {e}"


class DirectResponseTool(Tool):
    """Tool for when the agent has enough info and just wants to respond directly."""
    name = "direct_response"
    description = "Respond directly to the user without using any other tool. Use when you have enough information to answer."
    inputs = {
        "content": {"type": "string", "description": "Your response to the user"},
    }
    output_type = "string"

    def forward(self, content: str) -> str:
        return content


# Tool groups for different agent types
RESEARCH_TOOLS = [DeepResearchTool, AskNotebookTool, AddSourceTool, ListSourcesTool, CreateNotebookTool]
PODCAST_TOOLS = [AskNotebookTool, AddSourceTool, ListSourcesTool, GeneratePodcastTool]
QUIZ_TOOLS = [AskNotebookTool, AddSourceTool, ListSourcesTool, GenerateQuizTool]
REPORT_TOOLS = [AskNotebookTool, AddSourceTool, ListSourcesTool, GenerateReportTool]
MINDMAP_TOOLS = [AskNotebookTool, AddSourceTool, ListSourcesTool, GenerateMindMapTool]
ALL_TOOLS = [
    DeepResearchTool, AskNotebookTool, GeneratePodcastTool, GenerateVideoTool,
    GenerateQuizTool, GenerateMindMapTool, GenerateReportTool,
    AddSourceTool, ListSourcesTool, CreateNotebookTool, DirectResponseTool,
]
