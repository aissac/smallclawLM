"""Tests for NLMModel initialization and routing logic."""

import pytest
from smolagents.models import ChatMessage, MessageRole

from smallclawlm.nlm_model import NLMModel


class TestNLMModel:
    """Test NLMModel initialization and core methods."""

    def test_init_defaults(self):
        model = NLMModel()
        assert model.model_id == "notebooklm-router"
        assert model._notebook_id is None
        assert model._auto_create is True

    def test_init_custom(self):
        model = NLMModel(
            notebook_id="test-nb-123",
            notebook_title="Test Notebook",
            model_id="custom-model",
        )
        assert model._notebook_id == "test-nb-123"
        assert model._notebook_title == "Test Notebook"
        assert model.model_id == "custom-model"

    def test_route_returns_code_block(self):
        model = NLMModel()
        result = model._route("research something")
        assert result.startswith("<code>")
        assert result.strip().endswith("</code>")

    def test_escape_handles_special_chars(self):
        escaped = NLMModel._escape('He said "hello"\nnew line\\backslash')
        assert '\\"' in escaped
        assert "\\n" in escaped
        assert "\\\\" in escaped
