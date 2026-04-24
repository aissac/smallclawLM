"""NLMModel - smolagents Model backed by NotebookLM chat API.

The core of SmallClawLM: routes all LLM calls through NotebookLM's built-in Gemini.
Zero external API tokens needed.

Architecture (per smolagents integration analysis):
- Daemon thread event loop via run_coroutine_threadsafe (NOT asyncio.run)
- Never reuses conversation_id - each generate() call is stateless
- Handles stop_sequences (truncate before Observation: markers)
- CodeAgent ONLY - NotebookLM can't return structured JSON tool calls
- Forces coding persona so NotebookLM outputs Python code blocks
- Returns dummy TokenUsage (NotebookLM doesn't expose token counts)
"""

import asyncio
import logging
import threading
from typing import Any

from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import TokenUsage

from smallclawlm.auth import get_auth

logger = logging.getLogger(__name__)

_CODING_PERSONA_PREFIX = (
    "You are an autonomous coding agent. When asked to act, you MUST output "
    "Python code inside ```python ... ``` code blocks. "
    "Do not just summarize or explain - write executable code. "
    "If you need to use a tool, write Python code that calls it. "
)


class NLMModel(Model):
    """smolagents Model using NotebookLM's chat API as the LLM backend.

    All agent reasoning powered by Gemini inside Google's NotebookLM.
    No external LLM API keys required.

    Uses shared daemon event loop for async/sync bridging.
    Each generate() call is stateless (no conversation_id reuse).
    """

    def __init__(
        self,
        notebook_id: str | None = None,
        notebook_title: str | None = None,
        auto_create: bool = True,
        inject_coding_persona: bool = True,
        model_id: str = "notebooklm-gemini",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._notebook_id = notebook_id
        self._notebook_title = notebook_title or "SmallClawLM Session"
        self._auto_create = auto_create
        self._inject_coding_persona = inject_coding_persona
        self.model_id = model_id

        # Daemon event loop for async/sync bridging
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        self._client = None
        self._auth = None

    def _run_async(self, coro):
        """Submit async coroutine to daemon loop, blocking until done."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def _ensure_client(self):
        if self._client is not None:
            return
        from notebooklm import NotebookLMClient
        self._auth = await get_auth()
        self._client = NotebookLMClient(self._auth)
        await self._client.__aenter__()

    async def _ensure_notebook(self):
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
        """Generate a response using NotebookLM chat API. Stateless per call."""
        return self._run_async(self._agenerate(messages, stop_sequences, **kwargs))

    async def _agenerate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        **kwargs,
    ) -> ChatMessage:
        await self._ensure_notebook()

        # Flatten smolagents history into single prompt (NO conversation_id)
        prompt = self._messages_to_prompt(messages)

        if self._inject_coding_persona:
            prompt = _CODING_PERSONA_PREFIX + prompt

        try:
            result = await self._client.chat.ask(self._notebook_id, prompt)
            content = result.answer if hasattr(result, "answer") else str(result)
        except Exception as e:
            logger.warning(f"NotebookLM chat error: {e}")
            content = f"NotebookLM Error: {e}"

        if stop_sequences:
            from smolagents.models import remove_content_after_stop_sequences
            content = remove_content_after_stop_sequences(content, stop_sequences)

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            token_usage=TokenUsage(input_tokens=0, output_tokens=0),
        )

    def _messages_to_prompt(self, messages: list[ChatMessage]) -> str:
        role_labels = {
            MessageRole.SYSTEM: "System",
            MessageRole.USER: "User",
            MessageRole.ASSISTANT: "Assistant",
            MessageRole.TOOL_CALL: "Tool Call",
            MessageRole.TOOL_RESPONSE: "Tool Result",
        }
        parts = []
        for msg in messages:
            label = role_labels.get(msg.role, str(msg.role))
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            parts.append(f"[{label}]: {content}")
        return "\n\n".join(parts)
