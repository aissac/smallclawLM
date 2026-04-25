"""Unit tests for NLMModel — tool-routing model."""

import pytest
from smolagents.models import ChatMessage, MessageRole

from smallclawlm.nlm_model import NLMModel


class TestNLMModelInit:
    def test_has_generate(self):
        assert hasattr(NLMModel, "generate")

    def test_has_route(self):
        assert hasattr(NLMModel, "_route")

    def test_has_extract_task(self):
        assert hasattr(NLMModel, "_extract_task")


class TestNLMModelRouting:
    """Test that the model routes tasks to the correct tool calls."""

    @pytest.fixture
    def model(self):
        return NLMModel(notebook_id="test-nb")

    def test_route_research(self, model):
        result = model._route("research fusion energy")
        assert "deep_research" in result
        assert "<code>" in result

    def test_route_list_sources(self, model):
        result = model._route("list the sources in this notebook")
        assert "list_sources" in result
        assert "<code>" in result

    def test_route_podcast(self, model):
        result = model._route("generate a podcast about AI")
        assert "generate_podcast" in result

    def test_route_quiz(self, model):
        result = model._route("create a quiz on this topic")
        assert "generate_quiz" in result

    def test_route_mindmap(self, model):
        result = model._route("create a mind map of the concepts")
        assert "generate_mind_map" in result

    def test_route_report(self, model):
        result = model._route("generate a summary report")
        assert "generate_report" in result

    def test_route_video(self, model):
        result = model._route("make a video explainer")
        assert "generate_video" in result

    def test_route_add_source(self, model):
        result = model._route("add this url as a source")
        assert "add_source" in result

    def test_route_create_notebook(self, model):
        result = model._route("create a new notebook")
        assert "create_notebook" in result

    def test_route_default_ask(self, model):
        """Unknown tasks route to ask_notebook."""
        result = model._route("What is the meaning of life?")
        assert "ask_notebook" in result


class TestNLMModelGenerate:
    """Test generate() returns proper ChatMessage with code blocks."""

    @pytest.fixture
    def model(self):
        return NLMModel(notebook_id="test-nb")

    def test_generate_returns_chat_message(self, model):
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are an agent"),
            ChatMessage(role=MessageRole.USER, content="New task:\nResearch AI"),
        ]
        result = model.generate(messages)
        assert isinstance(result, ChatMessage)
        assert result.role == MessageRole.ASSISTANT
        assert "<code>" in result.content
        assert "deep_research" in result.content

    def test_generate_routes_general_question(self, model):
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are an agent"),
            ChatMessage(role=MessageRole.USER, content="New task:\nWhat is this about?"),
        ]
        result = model.generate(messages)
        assert "ask_notebook" in result.content

    def test_generate_final_answer_after_observation(self, model):
        """After a tool returns results, the model should call final_answer."""
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are an agent"),
            ChatMessage(role=MessageRole.USER, content="New task:\nList sources"),
            ChatMessage(role=MessageRole.ASSISTANT, content="<code>list_sources()</code>"),
            ChatMessage(role=MessageRole.TOOL_CALL, content="list_sources()"),
            ChatMessage(role=MessageRole.TOOL_RESPONSE, content="source1.md | READY\nsource2.py | READY"),
        ]
        model._step_count = 1  # Simulate being in step 2+
        result = model.generate(messages)
        assert "final_answer" in result.content

    def test_extract_task_strips_prefix(self, model):
        messages = [
            ChatMessage(role=MessageRole.USER, content="New task:\nResearch fusion"),
        ]
        task = model._extract_task(messages)
        assert task == "Research fusion"

    def test_extract_task_no_prefix(self, model):
        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello there"),
        ]
        task = model._extract_task(messages)
        assert task == "Hello there"
