"""Tests for NLMModel."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from smallclawlm.nlm_model import NLMModel


class TestNLMModel:
    """Test NLMModel initialization and message conversion."""

    def test_init_defaults(self):
        """NLMModel initializes with default values."""
        model = NLMModel()
        assert model.model_id == "notebooklm-gemini"
        assert model._notebook_id is None
        assert model._auto_create is True
        assert model._language == "en"

    def test_init_custom(self):
        """NLMModel accepts custom parameters."""
        model = NLMModel(
            notebook_id="test-nb-123",
            notebook_title="Test Notebook",
            model_id="custom-model",
        )
        assert model._notebook_id == "test-nb-123"
        assert model._notebook_title == "Test Notebook"
        assert model.model_id == "custom-model"

    def test_messages_to_prompt_system(self):
        """System messages are prefixed correctly."""
        from smolagents.models import ChatMessage, MessageRole

        model = NLMModel()
        messages = [ChatMessage(role=MessageRole.SYSTEM, content="You are helpful.")]
        prompt = model._messages_to_prompt(messages)
        assert "[System]: You are helpful." in prompt

    def test_messages_to_prompt_user(self):
        """User messages are prefixed correctly."""
        from smolagents.models import ChatMessage, MessageRole

        model = NLMModel()
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        prompt = model._messages_to_prompt(messages)
        assert "[User]: Hello" in prompt

    def test_messages_to_prompt_mixed(self):
        """Mixed messages are combined with double newlines."""
        from smolagents.models import ChatMessage, MessageRole

        model = NLMModel()
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="Be helpful."),
            ChatMessage(role=MessageRole.USER, content="Hi"),
        ]
        prompt = model._messages_to_prompt(messages)
        assert "[System]: Be helpful." in prompt
        assert "[User]: Hi" in prompt
        assert "\n\n" in prompt
