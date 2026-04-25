"""NLMModel — smolagents Model backed by a single NotebookLM notebook.

One agent = one notebook = one specialty brain.
The model calls chat.ask() on its notebook for every generate() call.

Key design decisions (validated with NotebookLM):
- Stateless chat: conversation_id=None (smolagents manages its own history)
- ChatMode.CONCISE: shorter responses = faster agent loops
- Exponential backoff: handle ChatError rate limits gracefully
- Auth refresh: auto-recover from expired tokens via refresh_auth()
- Source wait: tools must wait for source processing before next generate()
"""

import asyncio
import logging
import threading
import time
from typing import Any

from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import TokenUsage

from smallclawlm.auth import get_auth

logger = logging.getLogger(__name__)

# Retry config for rate limits
_MAX_RETRIES = 3
_RETRY_DELAYS = (2, 4, 8)


class NLMModel(Model):
    """smolagents Model using a single NotebookLM notebook as the LLM backend.

    Each agent gets one notebook with its own domain sources.
    All reasoning powered by Gemini inside Google's NotebookLM.
    No external LLM API keys required.
    """

    def __init__(
        self,
        notebook_id: str | None = None,
        notebook_title: str | None = None,
        auto_create: bool = True,
        concise: bool = True,
        model_id: str = "notebooklm-gemini",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._notebook_id = notebook_id
        self._notebook_title = notebook_title or "SmallClawLM Agent"
        self._auto_create = auto_create
        self._concise = concise
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
        """Get or create the NotebookLM client and notebook."""
        if self._client is not None:
            return

        from notebooklm import NotebookLMClient
        self._auth = await get_auth()
        self._client = NotebookLMClient(self._auth)
        await self._client.__aenter__()



    async def _ensure_notebook(self):
        """Ensure we have a notebook ID and chat mode is set."""
        await self._ensure_client()
        if self._notebook_id is not None:
            return
        if not self._auto_create:
            raise ValueError("No notebook_id provided and auto_create=False.")
        nb = await self._client.notebooks.create(self._notebook_title)
        self._notebook_id = nb.id
        logger.info(f"Auto-created notebook: {nb.id} ({nb.title})")

        # NOTE: ChatMode.CONCISE breaks chat.ask() (timeout/parse failure)
        # Using DEFAULT mode which works reliably.

    async def _chat_with_retry(self, question: str) -> str:
        """Call chat.ask() with exponential backoff for rate limits."""
        await self._ensure_notebook()

        for attempt in range(_MAX_RETRIES):
            try:
                result = await self._client.chat.ask(
                    self._notebook_id,
                    question,
                    conversation_id=None,  # Always stateless
                )
                answer = result.answer if hasattr(result, "answer") else str(result)
                if not answer or not answer.strip():
                    logger.warning(f"Empty answer from NotebookLM for prompt: {question[:200]}")
                    return ""
                return answer

            except Exception as e:
                error_msg = str(e).lower()
                is_rate_limit = "rate limit" in error_msg or "429" in error_msg
                is_auth_error = "csrf" in error_msg or "session" in error_msg or "401" in error_msg
                is_network_error = "timeout" in error_msg or "connection" in error_msg

                if is_auth_error and attempt == 0:
                    # Try refreshing auth once
                    logger.warning(f"Auth error, refreshing tokens: {e}")
                    try:
                        self._auth = await get_auth(force_refresh=True)
                        self._client = NotebookLMClient(self._auth)
                        await self._client.__aenter__()
                        continue
                    except Exception:
                        pass

                if is_rate_limit and attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                    logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt+1})")
                    await asyncio.sleep(delay)
                    continue

                if is_network_error and attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                    logger.warning(f"Network error, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                    continue

                # Unrecoverable or exhausted retries
                logger.error(f"chat.ask() failed after {attempt+1} attempts: {e}")
                return f"Error: {e}"

        return "Error: Max retries exceeded"

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """Generate a response using NotebookLM chat API.

        Smolagents passes its full message history. We flatten it into
        a compact prompt string rather than passing conversation_id,
        keeping NotebookLM stateless on its side.
        """
        return self._run_async(self._agenerate(messages, stop_sequences, **kwargs))

    async def _agenerate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        **kwargs,
    ) -> ChatMessage:
        prompt = self._messages_to_prompt(messages)

        content = await self._chat_with_retry(prompt)

        if stop_sequences:
            from smolagents.models import remove_content_after_stop_sequences
            content = remove_content_after_stop_sequences(content, stop_sequences)

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            token_usage=TokenUsage(input_tokens=0, output_tokens=0),
        )

    _CODING_INSTRUCTION = (
        "IMPORTANT: When you need to use a tool, output a Python code block using this format:\n"
        "<code>\n"
        "tool_name(argument=value)\n"
        "</code>\n"
        "When you have the final answer, use: <code>final_answer(content=\"your answer\")</code>\n"
        "Always wrap tool calls in <code> tags. Do not respond with plain text alone if a tool call is needed.\n"
    )

    def _messages_to_prompt(self, messages: list[ChatMessage]) -> str:
        """Flatten smolagents message history into a single prompt string.

        NotebookLM is stateless per call (no conversation_id).
        smolagents manages its own state — we flatten the full ReAct
        history into the question text.
        """
        role_labels = {
            MessageRole.SYSTEM: "System",
            MessageRole.USER: "User",
            MessageRole.ASSISTANT: "Assistant",
            MessageRole.TOOL_CALL: "Action",
            MessageRole.TOOL_RESPONSE: "Observation",
        }
        parts = [self._CODING_INSTRUCTION]
        for msg in messages:
            label = role_labels.get(msg.role, str(msg.role))
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            # Truncate very long observations to keep prompt compact
            if msg.role == MessageRole.TOOL_RESPONSE and len(content) > 1500:
                content = content[:1500] + "\n...[truncated]"
            parts.append(f"[{label}]: {content}")
        return "\n\n".join(parts)
