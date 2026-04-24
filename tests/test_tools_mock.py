"""Unit tests for NLM tools with mocked NotebookLM client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from smallclawlm.nlm_tools import (
    DeepResearchTool, AskNotebookTool, GeneratePodcastTool,
    AddSourceTool, ListSourcesTool, CreateNotebookTool,
    DirectResponseTool, ALL_TOOLS, RESEARCH_TOOLS, PODCAST_TOOLS,
    _SharedLoop, _ClientMixin,
)


class TestToolPresets:
    def test_all_tools_count(self):
        assert len(ALL_TOOLS) == 11

    def test_research_tools_count(self):
        assert len(RESEARCH_TOOLS) == 5

    def test_podcast_tools_count(self):
        assert len(PODCAST_TOOLS) == 4

    def test_direct_response_in_all(self):
        tool_classes = ALL_TOOLS
        assert DirectResponseTool in tool_classes

    def test_research_has_no_podcast(self):
        assert GeneratePodcastTool not in RESEARCH_TOOLS


class TestDirectResponseTool:
    """The only tool we can test without mocking — no async API calls."""

    def test_forward_returns_content(self):
        tool = DirectResponseTool()
        result = tool.forward(content="Hello world")
        assert result == "Hello world"

    def test_forward_empty_content(self):
        tool = DirectResponseTool()
        result = tool.forward(content="")
        assert result == ""

    def test_tool_name(self):
        assert DirectResponseTool.name == "direct_response"


class TestToolSMolagentsInterface:
    """Verify all tools satisfy smolagents Tool interface."""

    def test_all_tools_have_name(self):
        for tool_cls in ALL_TOOLS:
            assert hasattr(tool_cls, 'name')
            assert isinstance(tool_cls.name, str)
            assert len(tool_cls.name) > 0

    def test_all_tools_have_description(self):
        for tool_cls in ALL_TOOLS:
            assert hasattr(tool_cls, 'description')
            assert isinstance(tool_cls.description, str)
            assert len(tool_cls.description) > 0

    def test_all_tools_have_inputs(self):
        for tool_cls in ALL_TOOLS:
            assert hasattr(tool_cls, 'inputs')
            assert isinstance(tool_cls.inputs, dict)

    def test_all_tools_have_output_type(self):
        for tool_cls in ALL_TOOLS:
            assert hasattr(tool_cls, 'output_type')
            assert tool_cls.output_type == "string"

    def test_all_tools_instantiable(self):
        for tool_cls in ALL_TOOLS:
            tool = tool_cls()
            assert tool is not None
            assert hasattr(tool, 'forward')


class TestSharedLoop:
    def test_run_executes_coroutine(self):
        async def dummy():
            return 42
        result = _SharedLoop.run(dummy())
        assert result == 42

    def test_loop_reuse(self):
        """SharedLoop should reuse the same event loop."""
        loop1 = _SharedLoop.get_loop()
        loop2 = _SharedLoop.get_loop()
        assert loop1 is loop2
        assert loop1.is_running()
