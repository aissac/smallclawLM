"""Tests for Pipeline."""

import pytest
from unittest.mock import AsyncMock, patch

from smallclawlm.extensions.pipeline import Pipeline, PipelineStep, ArtifactType


class TestPipeline:
    """Test Pipeline step composition."""

    def test_add_source(self):
        pipeline = Pipeline()
        pipeline.add_source("https://example.com")
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].action == "add_source"
        assert pipeline.steps[0].params["url"] == "https://example.com"

    def test_research(self):
        pipeline = Pipeline()
        pipeline.research("quantum computing")
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].action == "research"
        assert pipeline.steps[0].params["query"] == "quantum computing"

    def test_generate(self):
        pipeline = Pipeline()
        pipeline.generate(ArtifactType.PODCAST)
        assert pipeline.steps[0].params["artifact_type"] == "podcast"

    def test_chaining(self):
        pipeline = (
            Pipeline()
            .add_source("https://example.com")
            .research("AI safety")
            .generate(ArtifactType.PODCAST)
        )
        assert len(pipeline.steps) == 3
        assert pipeline.steps[0].action == "add_source"
        assert pipeline.steps[1].action == "research"
        assert pipeline.steps[2].action == "generate"
