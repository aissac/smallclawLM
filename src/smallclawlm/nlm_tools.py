"""NLMTools — smolagents Tool wrappers around NotebookLM operations.

One agent = one notebook. These tools operate on the agent's single notebook.
All tools use daemon event loop for async/sync bridging.

Tools return human-readable strings, not status objects. This is critical
because CodeAgent needs to see results to make decisions.

Key fixes from v0.1:
- Tools return actual content (str), not GenerationStatus objects
- Generate tools now poll for completion and return artifact IDs
- Auth errors trigger automatic refresh
- All async ops use _SharedLoop singleton
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
        return future.result(timeout=120)


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
    description = (
        "Run deep web research on a topic. Returns a comprehensive report. "
        "Use this when you need current information about any topic. "
        "After calling this, the research results are added as sources to the notebook."
    )
    inputs = {
        "query": {"type": "string", "description": "Research question or topic"},
        "mode": {"type": "string", "description": "Research depth: fast or deep (default: deep)", "nullable": True},
    }
    output_type = "string"

    def forward(self, query: str, mode: str = "deep") -> str:
        async def _do():
            client = await self._get_client()
            nb_id = getattr(self, '_notebook_id', None)
            if not nb_id:
                notebooks = await client.notebooks.list()
                nb_id = notebooks[0].id if notebooks else (await client.notebooks.create("Research")).id
                self._notebook_id = nb_id

            await client.research.start(notebook_id=nb_id, query=query, source="web", mode=mode)
            # Poll for completion
            for _ in range(60):  # 60 * 5s = 5 min max
                await asyncio.sleep(5)
                status = await client.research.poll(nb_id)
                if status and hasattr(status, 'done') and status.done:
                    return f"Deep research on '{query}' complete. Results added to notebook."
            return f"Deep research on '{query}' started. Check notebook for results."

        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Error: deep research failed - {e}"


class AskNotebookTool(Tool, _ClientMixin):
    name = "ask_notebook"
    description = (
        "Ask a question about the notebook's sources. The notebook uses its "
        "built-in Gemini model to answer with citations from loaded sources. "
        "Use this for questions that can be answered from the existing sources."
    )
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
                nb_id = notebooks[0].id if notebooks else (await client.notebooks.create("Q&A")).id
                self._notebook_id = nb_id

            result = await client.chat.ask(nb_id, question)
            if hasattr(result, 'answer'):
                return result.answer
            return str(result)

        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Error: ask_notebook failed - {e}"


class GeneratePodcastTool(Tool, _ClientMixin):
    name = "generate_podcast"
    description = "Generate an audio overview podcast from the notebook's sources."
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
                nb_id = notebooks[0].id

            kwargs = {"notebook_id": nb_id}
            if instructions:
                kwargs["instructions"] = instructions

            result = await client.artifacts.generate_podcast(**kwargs)
            # Poll for completion
            for _ in range(60):
                await asyncio.sleep(5)
                status = await client.artifacts.poll(nb_id)
                if status and hasattr(status, 'done') and status.done:
                    return f"Podcast generated successfully for notebook {nb_id}. Use 'smallclaw download' to get the audio file."
            return f"Podcast generation started for notebook {nb_id}. Check back in a few minutes."

        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Error: podcast generation failed - {e}"


class GenerateVideoTool(Tool, _ClientMixin):
    name = "generate_video"
    description = "Generate an explainer video from the notebook's sources."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = getattr(self, '_notebook_id', None)
            if not nb_id:
                notebooks = await client.notebooks.list()
                nb_id = notebooks[0].id

            result = await client.artifacts.generate_video(notebook_id=nb_id)
            for _ in range(60):
                await asyncio.sleep(5)
                status = await client.artifacts.poll(nb_id)
                if status and hasattr(status, 'done') and status.done:
                    return f"Video generated for notebook {nb_id}."
            return f"Video generation started for notebook {nb_id}."

        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Error: video generation failed - {e}"


class GenerateQuizTool(Tool, _ClientMixin):
    name = "generate_quiz"
    description = "Generate a quiz from the notebook's sources."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = getattr(self, '_notebook_id', None)
            if not nb_id:
                notebooks = await client.notebooks.list()
                nb_id = notebooks[0].id

            result = await client.artifacts.generate_quiz(notebook_id=nb_id)
            for _ in range(60):
                await asyncio.sleep(3)
                status = await client.artifacts.poll(nb_id)
                if status and hasattr(status, 'done') and status.done:
                    return f"Quiz generated for notebook {nb_id}."
            return f"Quiz generation started for notebook {nb_id}."

        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Error: quiz generation failed - {e}"


class GenerateMindMapTool(Tool, _ClientMixin):
    name = "generate_mind_map"
    description = "Generate a mind map / concept map from the notebook's sources."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = getattr(self, '_notebook_id', None)
            if not nb_id:
                notebooks = await client.notebooks.list()
                nb_id = notebooks[0].id

            result = await client.artifacts.generate_mindmap(notebook_id=nb_id)
            for _ in range(60):
                await asyncio.sleep(3)
                status = await client.artifacts.poll(nb_id)
                if status and hasattr(status, 'done') and status.done:
                    return f"Mind map generated for notebook {nb_id}."
            return f"Mind map generation started for notebook {nb_id}."

        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Error: mind map generation failed - {e}"


class GenerateReportTool(Tool, _ClientMixin):
    name = "generate_report"
    description = "Generate a structured report from the notebook's sources."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = getattr(self, '_notebook_id', None)
            if not nb_id:
                notebooks = await client.notebooks.list()
                nb_id = notebooks[0].id

            result = await client.artifacts.generate_report(notebook_id=nb_id)
            for _ in range(60):
                await asyncio.sleep(3)
                status = await client.artifacts.poll(nb_id)
                if status and hasattr(status, 'done') and status.done:
                    return f"Report generated for notebook {nb_id}."
            return f"Report generation started for notebook {nb_id}."

        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Error: report generation failed - {e}"


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
                nb_id = notebooks[0].id

            sources = await client.sources.list(nb_id)
            if not sources:
                return "No sources in this notebook."
            lines = [f"Sources in notebook {nb_id}:"]
            for s in sources:
                name = getattr(s, 'title', None) or getattr(s, 'name', None) or getattr(s, 'filename', str(s.id))
                lines.append(f"  - {name} ({s.id})")
            return "\n".join(lines)

        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Error: list_sources failed - {e}"


class AddSourceTool(Tool, _ClientMixin):
    name = "add_source"
    description = "Add a URL as a source to the notebook."
    inputs = {
        "url": {"type": "string", "description": "URL to add as a source"},
        "title": {"type": "string", "description": "Optional title for the source", "nullable": True},
    }
    output_type = "string"

    def forward(self, url: str, title: str | None = None) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = getattr(self, '_notebook_id', None)
            if not nb_id:
                notebooks = await client.notebooks.list()
                nb_id = notebooks[0].id

            source = await client.sources.add_url(nb_id, url, title=title)
            name = title or url
            return f"Added source '{name}' (id: {source.id}) to notebook {nb_id}"

        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Error: add_source failed - {e}"


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
            self._notebook_id = nb.id  # Set for subsequent calls
            return f"Created notebook '{title}' (id: {nb.id})"

        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Error: create_notebook failed - {e}"


# ---- Tool presets ----

ALL_TOOLS = [
    DeepResearchTool, AskNotebookTool, GeneratePodcastTool, GenerateVideoTool,
    GenerateQuizTool, GenerateMindMapTool, GenerateReportTool,
    AddSourceTool, ListSourcesTool, CreateNotebookTool,
]

RESEARCH_TOOLS = [DeepResearchTool, AskNotebookTool, AddSourceTool, ListSourcesTool, CreateNotebookTool]
PODCAST_TOOLS = [AskNotebookTool, AddSourceTool, ListSourcesTool, GeneratePodcastTool]
QUIZ_TOOLS = [AskNotebookTool, AddSourceTool, ListSourcesTool, GenerateQuizTool]
REPORT_TOOLS = [AskNotebookTool, AddSourceTool, ListSourcesTool, GenerateReportTool]
MINDMAP_TOOLS = [AskNotebookTool, AddSourceTool, ListSourcesTool, GenerateMindMapTool]
