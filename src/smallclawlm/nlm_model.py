"""NLMModel — Tool-routing model for smolagents CodeAgent.

Does NOT call chat.ask() directly. Instead, generates code-block
instructions telling the agent which tool to call. The actual
NotebookLM interaction happens inside the tools, which use
simple prompts that the parser can handle reliably.

Architecture:
  User task → NLMModel.generate() → "<code>tool_call()</code>" → CodeAgent executes tool → NotebookLM
"""

import asyncio
import logging
import threading
from typing import Any

from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import TokenUsage

logger = logging.getLogger(__name__)

# Tool routing rules — maps task intent to tool calls
_ROUTING_RULES = [
    {
        "keywords": ["list sources", "list the sources", "what sources", "show sources", "loaded sources"],
        "tool": "list_sources",
        "code": "<code>\nlist_sources()\n</code>",
    },
    {
        "keywords": ["add source", "add a source", "add url", "load url", "import source"],
        "tool": "add_source",
        "code": '<code>\nadd_source(url="<URL>")\n</code>',
    },
    {
        "keywords": ["create notebook", "new notebook", "make notebook"],
        "tool": "create_notebook",
        "code": '<code>\ncreate_notebook(title="<TITLE>")\n</code>',
    },
    {
        "keywords": ["research", "deep research", "investigate", "look up", "find out"],
        "tool": "deep_research",
        "code": '<code>\ndeep_research(query="<QUERY>")\n</code>',
    },
    {
        "keywords": ["podcast", "audio overview", "generate audio"],
        "tool": "generate_podcast",
        "code": "<code>\ngenerate_podcast()\n</code>",
    },
    {
        "keywords": ["video", "generate video", "explainer video"],
        "tool": "generate_video",
        "code": "<code>\ngenerate_video()\n</code>",
    },
    {
        "keywords": ["quiz", "test me", "questions", "generate quiz"],
        "tool": "generate_quiz",
        "code": "<code>\ngenerate_quiz()\n</code>",
    },
    {
        "keywords": ["mind map", "mindmap", "connections", "concept map"],
        "tool": "generate_mind_map",
        "code": "<code>\ngenerate_mind_map()\n</code>",
    },
    {
        "keywords": ["report", "summary report", "generate report"],
        "tool": "generate_report",
        "code": "<code>\ngenerate_report()\n</code>",
    },
]


class NLMModel(Model):
    """smolagents Model that routes tasks to NotebookLM tools.

    Instead of calling chat.ask() (which has parser issues),
    this model analyzes the task and returns a <code> block
    telling CodeAgent which tool to invoke. The actual NotebookLM
    interaction happens inside the tools.
    """

    def __init__(
        self,
        notebook_id: str | None = None,
        notebook_title: str | None = None,
        auto_create: bool = True,
        model_id: str = "notebooklm-router",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._notebook_id = notebook_id
        self._notebook_title = notebook_title or "SmallClawLM Agent"
        self._auto_create = auto_create
        self.model_id = model_id

        # Daemon event loop for async/sync bridging (used by tools)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        self._client = None
        self._auth = None
        self._step_count = 0

    def _run_async(self, coro):
        """Submit async coroutine to daemon loop, blocking until done."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def _ensure_client(self):
        """Get or create the NotebookLM client."""
        if self._client is not None:
            return
        from notebooklm import NotebookLMClient
        from smallclawlm.auth import get_auth
        self._auth = await get_auth()
        self._client = NotebookLMClient(self._auth)
        await self._client.__aenter__()

    async def _ensure_notebook(self):
        """Ensure we have a notebook ID."""
        await self._ensure_client()
        if self._notebook_id is not None:
            return
        if not self._auto_create:
            raise ValueError("No notebook_id provided and auto_create=False.")
        nb = await self._client.notebooks.create(self._notebook_title)
        self._notebook_id = nb.id
        logger.info(f"Auto-created notebook: {nb.id} ({nb.title})")

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """Route the task to the appropriate tool via <code> blocks.

        Step 1: Route task → tool call code block
        Step 2: After tool returns, call final_answer with results
        """
        self._step_count += 1
        task = self._extract_task(messages)

        # Step 2+: If we have tool output, deliver final answer
        last_obs = self._get_last_observation(messages)
        if last_obs and self._step_count > 1:
            content = '<code>\nfinal_answer(answer="' + self._escape(last_obs) + '")\n</code>'
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                token_usage=TokenUsage(input_tokens=0, output_tokens=0),
            )

        # Step 1: Route task to tool
        routed = self._route(task)
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=routed,
            token_usage=TokenUsage(input_tokens=0, output_tokens=0),
        )

    def _extract_task(self, messages: list[ChatMessage]) -> str:
        """Extract the user's task from the message history.

        smolagents passes USER content as a list of dicts:
        [{"type": "text", "text": "New task:\nActual task"}]
        """
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                c = self._unwrap_content(msg.content)
                if c.startswith("New task:\n"):
                    c = c[len("New task:\n"):]
                return c.strip()
        return ""

    @staticmethod
    def _unwrap_content(content) -> str:
        """Unwrap smolagents message content (may be str or list of dicts)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # [{"type": "text", "text": "..."}]
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif isinstance(item, str):
                    texts.append(item)
            return " ".join(texts)
        return str(content)

    def _get_last_observation(self, messages: list[ChatMessage]) -> str:
        """Get the last tool response from the history."""
        for msg in reversed(messages):
            if msg.role == MessageRole.TOOL_RESPONSE:
                return self._unwrap_content(msg.content)
        return ""

    def _route(self, task: str) -> str:
        """Determine which tool to call based on the task text.

        Uses set-of-words matching: all words in a keyword must appear in the task.
        Falls back to ask_notebook for unmatched tasks.
        """
        task_words = set(task.lower().split())

        for rule in _ROUTING_RULES:
            for kw in rule["keywords"]:
                kw_words = set(kw.split())
                if kw_words.issubset(task_words):
                    logger.info(f"Routed to {rule['tool']} (matched '{kw}')")
                    return rule["code"]

        # Default: ask_notebook for general questions
        escaped = self._escape(task)
        logger.info(f"Routed to ask_notebook (no keyword match)")
        return f'<code>\nask_notebook(question="{escaped}")\n</code>'

    @staticmethod
    def _escape(text: str) -> str:
        """Escape text for inclusion in a Python string literal."""
        return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")[:2000]
