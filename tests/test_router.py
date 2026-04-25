"""Tests for the hybrid fast/slow path router."""
import pytest
from smallclawlm.router import route, Path, RouteResult


class TestFastPath:
    """Fast path: direct tool calls with no LLM reasoning needed."""

    @pytest.mark.parametrize("input_text,expected_intent", [
        ("generate a podcast", "generate_podcast"),
        ("podcast", "generate_podcast"),
        ("audio overview", "generate_podcast"),
        ("generate video", "generate_video"),
        ("explainer video", "generate_video"),
        ("generate report", "generate_report"),
        ("summary report", "generate_report"),
        ("report", "generate_report"),
        ("generate quiz", "generate_quiz"),
        ("quiz", "generate_quiz"),
        ("test me", "generate_quiz"),
        ("mind map", "generate_mind_map"),
        ("concept map", "generate_mind_map"),
        ("deep research", "deep_research"),
        ("research on fusion", "deep_research"),
        ("research about vaccines", "deep_research"),
        ("research", "deep_research"),
        ("list sources", "list_sources"),
        ("list the sources", "list_sources"),
        ("show sources", "list_sources"),
        ("show the sources", "list_sources"),
        ("what sources", "list_sources"),
        ("what are the sources", "list_sources"),
        ("add source", "add_source"),
        ("add a source", "add_source"),
        ("load url", "add_source"),
        ("create notebook", "create_notebook"),
        ("create a notebook", "create_notebook"),
        ("new notebook", "create_notebook"),
    ])
    def test_fast_path_routing(self, input_text, expected_intent):
        result = route(input_text)
        assert result.path == Path.FAST, f"Expected FAST, got {result.path} for '{input_text}'"
        assert result.intent == expected_intent, f"Expected {expected_intent}, got {result.intent}"

    def test_fast_path_high_confidence(self):
        result = route("generate a podcast")
        assert result.confidence >= 0.8, f"Fast path should have high confidence, got {result.confidence}"


class TestSlowPath:
    """Slow path: needs reasoning."""

    @pytest.mark.parametrize("input_text", [
        "Why is cold fusion difficult?",
        "Explain quantum computing",
        "How does photosynthesis work?",
        "What are the key findings?",
        "Can you summarize the main points?",
        "Tell me about machine learning",
        "Describe the architecture",
        "Compare React vs Vue",
        "Analyze the data",
        "Evaluate the performance",
        "help me understand",
    ])
    def test_slow_path_routing(self, input_text):
        result = route(input_text)
        assert result.path == Path.SLOW, f"Expected SLOW, got {result.path} for '{input_text}'"
        assert result.intent == "ask"


class TestQueryExtraction:
    """Test query parameter extraction for research/ask intents."""

    def test_research_query_extraction(self):
        result = route("research on fusion energy")
        assert result.params.get("query") is not None

    def test_default_query(self):
        result = route("hello world")
        assert result.params.get("query") == "hello world"


class TestDefaultRouting:
    """Unknown input defaults to slow path."""

    def test_unknown_input_defaults_to_slow(self):
        result = route("hello")
        assert result.path == Path.SLOW

    def test_empty_like_input(self):
        result = route("something random")
        assert result.path == Path.SLOW
