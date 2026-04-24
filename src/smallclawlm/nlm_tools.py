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
    description = (
        "Run deep web research on a topic using Google NotebookLM's built-in web research. "
        "Returns a comprehensive report with cited sources. Use this when you need current, "
        "detailed information about any topic. The mode parameter controls depth: 'fast' for "
        "quick overviews, 'deep' (default) for thorough analysis. After calling this, the "
        "research results are added as sources to the notebook."
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
            await client.research.start(notebook_id=nb_id, query=query, source="web", mode=mode)
            poll = await client.research.poll(nb_id)
            return poll.get("report", "Research completed but no report generated.")
        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Research error: {e}. Try: 1) simplify your query, 2) use mode='fast' instead, or 3) try ask_notebook() with a more specific question."


class AskNotebookTool(Tool, _ClientMixin):
    name = "ask_notebook"
    description = (
        "Ask a question about the content currently loaded in the notebook. The answer is "
        "grounded in the notebook's sources with citations. Use this when you need information "
        "that should already be in the notebook's sources, or when deep_research fails. The "
        "question should be specific and clear."
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
                nb_id = notebooks[0].id if notebooks else (await client.notebooks.create("SmallClawLM Session")).id
            result = await client.chat.ask(nb_id, question, conversation_id=None)
            return result.answer if hasattr(result, "answer") else str(result)
        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Ask error: {e}. Try: 1) check if sources are loaded with list_sources(), 2) make your question more specific, or 3) add more sources with add_source()."


class GeneratePodcastTool(Tool, _ClientMixin):
    name = "generate_podcast"
    description = (
        "Generate an audio podcast (Audio Overview) from the notebook's sources. Two AI hosts "
        "discuss the material in a conversational tone. You can customize the focus and style "
        "using the instructions parameter. The audio file is generated asynchronously - the tool "
        "returns a task ID you can reference."
    )
    inputs = {
        "instructions": {"type": "string", "description": "Custom instructions for the podcast", "nullable": True},
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
            return f"Generation error: {e}. Try: 1) make sure sources are loaded, 2) check source status, or 3) add more sources first."


class GenerateVideoTool(Tool, _ClientMixin):
    name = "generate_video"
    description = (
        "Generate a video explainer from the notebook's sources. Style can be 'whiteboard' "
        "(default, hand-drawn animation) or 'animated'. The video is generated asynchronously - "
        "the tool returns a task ID."
    )
    inputs = {
        "style": {"type": "string", "description": "Video style: whiteboard or animated (default: whiteboard)", "nullable": True},
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
            return f"Generation error: {e}. Try: 1) make sure sources are loaded, 2) check source status, or 3) add more sources first."


class GenerateQuizTool(Tool, _ClientMixin):
    name = "generate_quiz"
    description = (
        "Generate a quiz from the notebook's sources. Difficulty levels: 'easy' (basic recall), "
        "'medium' (default, understanding), 'hard' (analysis and synthesis). Great for study "
        "aids and self-assessment."
    )
    inputs = {
        "difficulty": {"type": "string", "description": "Quiz difficulty: easy, medium, or hard (default: medium)", "nullable": True},
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
            return f"Generation error: {e}. Try: 1) make sure sources are loaded, 2) check source status, or 3) add more sources first."


class GenerateMindMapTool(Tool, _ClientMixin):
    name = "generate_mind_map"
    description = (
        "Generate a visual mind map (connections diagram) from the notebook's sources. Shows "
        "how concepts and topics relate to each other. No parameters needed - it uses all "
        "sources in the notebook. Returns immediately with the mind map data."
    )
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
            return f"Generation error: {e}. Try: 1) make sure sources are loaded, 2) check source status, or 3) add more sources first."


class GenerateReportTool(Tool, _ClientMixin):
    name = "generate_report"
    description = (
        "Generate a structured report from the notebook's sources. Automatically organizes "
        "the content into sections with headers, key points, and citations. No parameters "
        "needed - it synthesizes all loaded sources."
    )
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
            return f"Generation error: {e}. Try: 1) make sure sources are loaded, 2) check source status, or 3) add more sources first."


class AddSourceTool(Tool, _ClientMixin):
    name = "add_source"
    description = (
        "Add a source to the notebook by URL. Supported formats: web pages, YouTube videos, "
        "PDFs, Google Docs, and Google Slides. IMPORTANT: This tool WAITS until the source is "
        "fully processed and ready before returning. You can then query it with ask_notebook."
    )
    inputs = {
        "url": {"type": "string", "description": "URL of the source to add"},
        "title": {"type": "string", "description": "Title for the source", "nullable": True},
    }
    output_type = "string"

    def forward(self, url: str, title: str | None = None) -> str:
        async def _do():
            client = await self._get_client()
            nb_id = getattr(self, '_notebook_id', None)
            if not nb_id:
                notebooks = await client.notebooks.list()
                nb_id = notebooks[0].id if notebooks else (await client.notebooks.create("Session")).id
            source = await client.sources.add_url(nb_id, url, title=title, wait=True)
            return f"Added source: {source.id} (title: {getattr(source, 'title', 'N/A')})"
        try:
            return _SharedLoop.run(_do())
        except Exception as e:
            return f"Source error: {e}. Try: 1) verify the URL is correct, 2) use a different URL format, or 3) check if the URL requires authentication."


class ListSourcesTool(Tool, _ClientMixin):
    name = "list_sources"
    description = (
        "List all sources currently loaded in the notebook. Returns each source's ID, title, "
        "and processing status (PENDING, READY, or FAILED). Use this to verify that sources "
        "have been processed before querying them."
    )
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
            return f"List error: {e}. The notebook may not exist or auth may have expired."


class CreateNotebookTool(Tool, _ClientMixin):
    name = "create_notebook"
    description = (
        "Create a new empty NotebookLM notebook with the given title. Returns the notebook ID "
        "which you can use for subsequent operations. Use this when you need a fresh notebook "
        "for a different topic."
    )
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
            return f"Create error: {e}. Try: 1) check if your title is valid, or 2) re-authenticate if you see auth errors."


class FinalAnswerTool(Tool):
    """Tool for when the agent has enough info and just wants to respond directly."""
    name = "final_answer"
    description = (
        "Return a final answer to the user. This is the ONLY way to end the agent's execution "
        "loop. You MUST call this when you have finished your task. The content parameter "
        "should contain your complete, well-organized answer to the user's original question."
    )
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
    AddSourceTool, ListSourcesTool, CreateNotebookTool, FinalAnswerTool,
]
