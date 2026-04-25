"""NLMModel - NotebookLM chat-backed smolagents Model for the slow path.

When the router classifies input as needing reasoning (conversational,
analytical, multi-step), NLMAgent uses this model. It sends the full
smolagents message history to NotebookLM chat API and parses the
response for tool calls or final answers.

This is the "brain" of the slow path. The fast path bypasses this entirely,
going straight to Pipeline / direct tool calls.

Architecture:
  Fast path:  user input -> router -> Pipeline -> direct API call -> result
  Slow path:  user input -> router -> NLMAgent -> NLMModel.generate() -> chat.ask()
              -> parse response -> CodeAgent executes tools -> result

Key design decisions:
- Stateless: no conversation_id reuse (avoids sync drift)
- Daemon thread event loop for async/sync bridge
- Error -> string conversion so agent can self-correct
- Handles list content in smolagents messages (step 2+)
- Truncates long prompts to 10K chars for NotebookLM limits
"""

import asyncio
import logging
import threading
from typing import Any

from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import TokenUsage

from smallclawlm.auth import get_auth

logger = logging.getLogger(__name__)


class NLMModel(Model):
    """smolagents Model backed by NotebookLM chat API.

    Used only in the slow path (NLMAgent). The fast path calls
    NotebookLM tools directly via Pipeline - no model involved.
    """

    def __init__(
        self,
        notebook_id: str | None = None,
        notebook_title: str | None = None,
        auto_create: bool = True,
        model_id: str = "notebooklm-chat",
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self._notebook_id = notebook_id
        self._notebook_title = notebook_title or "SmallClawLM Agent"
        self._auto_create = auto_create
        self._client = None
        self._loop = None
        self._thread = None

    def _ensure_loop(self):
        """Get or create the daemon event loop for async operations."""
        if self._loop is None or not self._loop.is_running():
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._thread.start()
        return self._loop

    async def _ensure_notebook(self):
        """Ensure we have a notebook ID, creating one if needed."""
        if self._notebook_id:
            return self._notebook_id

        if not self._auto_create:
            raise RuntimeError("No notebook ID provided and auto_create=False")

        client = await self._get_client()
        nb = await client.notebooks.create(self._notebook_title)
        self._notebook_id = nb.id
        logger.info(f"Created notebook: {self._notebook_id}")
        return self._notebook_id

    async def _get_client(self):
        """Get or create the NotebookLM client."""
        if self._client is None:
            auth = await get_auth()
            from notebooklm import NotebookLMClient
            self._client = NotebookLMClient(auth)
            await self._client.__aenter__()
        return self._client

    async def _chat(self, prompt: str) -> str:
        """Send a chat message to NotebookLM and return the response."""
        nb_id = await self._ensure_notebook()
        client = await self._get_client()

        try:
            result = await client.chat.ask(nb_id, prompt)
            return result.answer if hasattr(result, "answer") else str(result)
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"[ERROR] {e}"

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        tools_to_call_from: list | None = None,
        **kwargs,
    ) -> ChatMessage:
        """Generate a response using NotebookLM chat.

        Flattens smolagents message history into a single prompt,
        sends it to NotebookLM, and returns the response.
        The CodeAgent will parse the response for tool calls or final answers.
        """
        # Flatten message history into a single prompt
        # smolagents messages can have content as str or list[dict]
        prompt_parts = []
        for msg in messages:
            content = msg.content
            # Handle list content (smolagents sends dicts for tool results)
            if isinstance(content, list):
                content = " ".join(
                    str(item) if isinstance(item, str) else str(item)
                    for item in content
                )
            content = str(content) if content else ""

            if not content:
                continue

            if msg.role == MessageRole.SYSTEM:
                prompt_parts.append(f"[System] {content}")
            elif msg.role == MessageRole.USER:
                prompt_parts.append(f"[User] {content}")
            elif msg.role == MessageRole.ASSISTANT:
                # Truncate long assistant messages to avoid token bloat
                truncated = content[:500] if len(content) > 500 else content
                prompt_parts.append(f"[Assistant] {truncated}")
            else:
                prompt_parts.append(content)

        if not prompt_parts:
            full_prompt = "[User] Please respond."
        else:
            full_prompt = "\n\n".join(prompt_parts)

        # Truncate total prompt to avoid NotebookLM token limits
        if len(full_prompt) > 10000:
            full_prompt = full_prompt[:10000] + "\n[...truncated]"

        # Run async chat in the daemon loop
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(self._chat(full_prompt), loop)
        try:
            response = future.result(timeout=60)
        except Exception as e:
            logger.error(f"Generate error: {e}")
            response = 'final_answer("Error: ' + str(e) + '")'

        # If NotebookLM returned empty/error, return a fallback
        if not response or response.strip() == "" or response.startswith("[ERROR]"):
            response = 'final_answer("I could not get a response from NotebookLM. Please try rephrasing your request.")'

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=response,
            token_usage=TokenUsage(input_tokens=0, output_tokens=0),
        )
