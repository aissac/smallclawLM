"""Tests for Pipeline construction."""
import pytest
from smallclawlm.extensions.pipeline import Pipeline, ArtifactType, PipelineStep


class TestPipelineConstruction:
    """Test Pipeline step construction (no API calls)."""

    def test_add_source(self):
        pipe = Pipeline()
        result = pipe.add_source("https://example.com")
        assert result is pipe  # Fluent API
        assert len(pipe.steps) == 1
        assert pipe.steps[0].action == "add_source"

    def test_research(self):
        pipe = Pipeline()
        result = pipe.research("fusion energy", mode="fast")
        assert result is pipe
        assert len(pipe.steps) == 1
        assert pipe.steps[0].action == "research"
        assert pipe.steps[0].params["query"] == "fusion energy"
        assert pipe.steps[0].params["mode"] == "fast"

    def test_generate_podcast(self):
        pipe = Pipeline()
        result = pipe.generate(ArtifactType.PODCAST)
        assert result is pipe
        assert len(pipe.steps) == 1
        assert pipe.steps[0].action == "generate"
        assert pipe.steps[0].params["artifact_type"] == "podcast"

    def test_ask(self):
        pipe = Pipeline()
        result = pipe.ask("What is SmallClawLM?")
        assert result is pipe
        assert len(pipe.steps) == 1
        assert pipe.steps[0].action == "ask"

    def test_chained_steps(self):
        pipe = Pipeline()
        pipe.add_source("https://example.com").research("test").generate(ArtifactType.REPORT)
        assert len(pipe.steps) == 3
        assert pipe.steps[0].action == "add_source"
        assert pipe.steps[1].action == "research"
        assert pipe.steps[2].action == "generate"


class TestArtifactType:
    """Test ArtifactType enum."""

    def test_all_types(self):
        expected = {"podcast", "video", "quiz", "flashcards", "mindmap", 
                     "report", "datatable", "infographic", "slidedeck"}
        actual = {e.value for e in ArtifactType}
        assert actual == expected
