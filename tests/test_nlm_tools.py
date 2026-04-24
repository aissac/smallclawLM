"""Tests for NLMTools."""

import pytest
from smallclawlm.nlm_tools import (
    DeepResearchTool,
    GeneratePodcastTool,
    AddSourceTool,
    CreateNotebookTool,
    ListSourcesTool,
)


class TestToolDefinitions:
    """Test that tools have correct metadata for smolagents."""

    def test_deep_research_tool(self):
        tool = DeepResearchTool()
        assert tool.name == "deep_research"
        assert "query" in tool.inputs
        assert tool.output_type == "string"

    def test_generate_podcast_tool(self):
        tool = GeneratePodcastTool()
        assert tool.name == "generate_podcast"

    def test_add_source_tool(self):
        tool = AddSourceTool()
        assert tool.name == "add_source"
        assert "url" in tool.inputs

    def test_create_notebook_tool(self):
        tool = CreateNotebookTool()
        assert tool.name == "create_notebook"
        assert "title" in tool.inputs

    def test_list_sources_tool(self):
        tool = ListSourcesTool()
        assert tool.name == "list_sources"
