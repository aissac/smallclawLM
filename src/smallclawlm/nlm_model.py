"""NLMModel — smolagents Model backed by NotebookLM chat API.

This is the core of SmallClawLM: a smolagents-compatible Model that routes
all LLM calls through NotebookLM's built-in Gemini, requiring zero external
API tokens for the agent's "thinking."
"""

import asyncio
import logging
from typing import Any

from smolagents.models import Model, ChatMessage, MessageRole

from smallclawlm.auth import get_auth, ensure_authenticated

logger = logging.getLogger(__name__)


class NLMModel(Model):
    """smolagents Model that uses NotebookLM's chat API as the LLM backend.

    All agent reasoning is powered by Gemini inside Google's NotebookLM.
    No external LLM API keys required.

    Args:
        notebook_id: NotebookLM notebook ID to use as context.
            If None, uses the most recent notebook or creates one.
        notebook_title: Title for auto-created notebooks.
        auto_create: Whether to auto-create notebooks when needed.
        language: Response language (default: "en").
        model_id: Display name for smolagents (default: "notebooklm-gemini").
    """

    def __init__(
        self,
        notebook_id: str | None = None,
        notebook_title: str | None = None,
        auto_create: bool = True,
        language: str = "en",
        model_id: str = "notebooklm-gemini",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._notebook_id = notebook_id
        self._notebook_title = notebook_title or "SmallClawLM Session"
        self._auto_create = auto_create
        self._language = language
        self.model_id = model_id
        self._client = None
        self._auth = None

    async def _ensure_client(self):
        """Lazily initialize the NotebookLM client."""
        if self._client is not None:
            return

        from notebooklm import NotebookLMClient

        self._auth = await get_auth()
        self._client = NotebookLMClient(self._auth)

    async def _ensure_notebook(self):
        """Ensure we have a notebook ID, creating one if needed."""
        await self._ensure_client()

        if self._notebook_id is not None:
            return

        if not self._auto_create:
            raise ValueError(
                "No notebook_id provided and auto_create=False. "
                "Either pass a notebook_id or set auto_create=True."
            )

        nb = await self._client.notebooks.create(self._notebook_title)
        self._notebook_id = nb.id
        logger.info(f"Auto-created notebook: {nb.id} ({nb.title})")

    def generate(
        self,
        messages: list[ChatMessage],
        tools: list[Any] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """Generate a response using NotebookLM's chat API.

        Converts smolagents message history into a single prompt for NotebookLM.
        The agent's entire reasoning loop runs through Gemini inside Google.
        """
        return asyncio.get_event_loop().run_until_complete(
            self._agenerate(messages, tools, **kwargs)
        )

    async def _agenerate(
        self,
        messages: list[ChatMessage],
        tools: list[Any] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """Async implementation of generate."""
        await self._ensure_notebook()

        # Convert smolagents messages to a single prompt
        prompt = self._messages_to_prompt(messages)

        # Query NotebookLM — Gemini inside Google does the thinking
        result = await self._client.chat.ask(
            self._notebook_id,
            prompt,
        )

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=result.answer if hasattr(result, "answer") else str(result),
        )

    def _messages_to_prompt(self, messages: list[ChatMessage]) -> str:
        """Convert smolagents message history into a NotebookLM-compatible prompt."""
        parts = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                parts.append(f"[System]: {msg.content}")
            elif msg.role == MessageRole.USER:
                parts.append(f"[User]: {msg.content}")
            elif msg.role == MessageRole.ASSISTANT:
                parts.append(f"[Assistant]: {msg.content}")
            elif msg.role == MessageRole.TOOL_CALL:
                parts.append(f"[Tool Call]: {msg.content}")
            elif msg.role == MessageRole.TOOL_RESPONSE:
                parts.append(f"[Tool Result]: {msg.content}")
        return "\n\n".join(parts)
