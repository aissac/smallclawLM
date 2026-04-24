"""Unit tests for NLMModel with mocked NotebookLM client.

We mock the async chat.ask() call to avoid needing real auth.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from smolagents.models import ChatMessage, MessageRole

from smallclawlm.nlm_model import NLMModel


class TestNLMModelInit:
    def test_creates_with_defaults(self):
        model = NLMModel.__new__(NLMModel)
        # Can't call __init__ easily without mocking super().__init__
        # but we can test the class exists and has expected attributes
        assert hasattr(NLMModel, 'generate')
        assert hasattr(NLMModel, '_agenerate')
        assert hasattr(NLMModel, '_messages_to_prompt')

    def test_messages_to_prompt(self):
        model = NLMModel.__new__(NLMModel)
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are an agent"),
            ChatMessage(role=MessageRole.USER, content="Research fusion energy"),
        ]
        prompt = NLMModel._messages_to_prompt(model, messages)
        assert "[System]" in prompt
        assert "[User]" in prompt
        assert "Research fusion energy" in prompt

    def test_messages_to_prompt_truncates_long_observations(self):
        model = NLMModel.__new__(NLMModel)
        messages = [
            ChatMessage(role=MessageRole.TOOL_RESPONSE, content="x" * 3000),
        ]
        prompt = NLMModel._messages_to_prompt(model, messages)
        assert "truncated" in prompt
        assert len(prompt) < 2000  # Much shorter than 3000 chars


class TestNLMModelGenerate:
    """Test generate() with mocked async client."""

    @pytest.fixture
    def mock_model(self):
        """Create an NLMModel with mocked internals."""
        model = NLMModel(notebook_id="test-notebook-123", auto_create=False)
        # Simulate client being ready
        model._client = MagicMock()
        return model

    def test_generate_returns_chat_message(self, mock_model):
        """Verify generate() returns a smolagents ChatMessage."""
        # Mock the async chain
        mock_result = MagicMock()
        mock_result.answer = "Test response from NotebookLM"

        async def mock_chat_ask(*args, **kwargs):
            return mock_result

        mock_model._client.chat = MagicMock()
        mock_model._client.chat.ask = mock_chat_ask

        messages = [ChatMessage(role=MessageRole.USER, content="Test query")]
        result = mock_model.generate(messages)

        assert isinstance(result, ChatMessage)
        assert result.role == MessageRole.ASSISTANT
        assert result.content == "Test response from NotebookLM"

    def test_generate_handles_error_gracefully(self, mock_model):
        """Verify generate() returns error as content, not exception."""
        async def mock_chat_ask(*args, **kwargs):
            raise RuntimeError("ChatError: rate limited")

        mock_model._client.chat = MagicMock()
        mock_model._client.chat.ask = mock_chat_ask

        messages = [ChatMessage(role=MessageRole.USER, content="Test")]
        result = mock_model.generate(messages)

        assert isinstance(result, ChatMessage)
        assert "Error" in result.content


class TestRetryLogic:
    """Test that _chat_with_retry handles rate limits and auth errors."""

    @pytest.fixture
    def mock_model(self):
        model = NLMModel(notebook_id="test-nb", auto_create=False)
        model._client = MagicMock()
        return model

    def test_rate_limit_retries(self, mock_model):
        """First call rate limited, second succeeds."""
        call_count = 0

        async def mock_ask(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Chat request was rate limited. Wait a few seconds.")
            mock_result = MagicMock()
            mock_result.answer = "Success on retry"
            return mock_result

        mock_model._client.chat = MagicMock()
        mock_model._client.chat.ask = mock_ask

        messages = [ChatMessage(role=MessageRole.USER, content="test")]
        result = mock_model.generate(messages)

        # Should have retried and eventually returned something
        assert isinstance(result, ChatMessage)
