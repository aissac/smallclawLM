"""Unit tests for the fuzzy parser — SmallClawLM's most critical component."""

import pytest
from smallclawlm.parser import parse_nlm_response, ParsedAction, _fuzzy_json_parse, format_smolagents_output


class TestFuzzyJsonParse:
    """Test the fuzzy JSON parser that handles NotebookLM quirks."""

    def test_valid_json(self):
        assert _fuzzy_json_parse('{"tool": "deep_research", "params": {"query": "fusion"}}') == {
            "tool": "deep_research", "params": {"query": "fusion"}
        }

    def test_url_in_value(self):
        result = _fuzzy_json_parse('{"url": "https://example.com/paper"}')
        assert result == {"url": "https://example.com/paper"}

    def test_unquoted_keys(self):
        result = _fuzzy_json_parse('{query: "what is fusion?", mode: deep}')
        assert result["query"] == "what is fusion?"
        assert result["mode"] == "deep"

    def test_single_quotes(self):
        result = _fuzzy_json_parse("{'key': 'value'}")
        assert result == {"key": "value"}

    def test_trailing_comma(self):
        result = _fuzzy_json_parse('{"query": "fusion",}')
        assert result["query"] == "fusion"

    def test_python_bools(self):
        result = _fuzzy_json_parse('{verbose: True, fast: False, empty: None}')
        assert result["verbose"] == "true"
        assert result["fast"] == "false"

    def test_empty_string(self):
        assert _fuzzy_json_parse("") == {}

    def test_broken_json_returns_raw(self):
        result = _fuzzy_json_parse("this is not json at all")
        assert "raw" in result

    def test_kwargs_style(self):
        result = _fuzzy_json_parse('query="fusion energy", mode="deep"')
        assert result["query"] == "fusion energy"
        assert result["mode"] == "deep"


class TestParseNLMResponse:
    """Test the 3-tier response parser."""

    def test_thought_action_blocks(self):
        text = "THOUGHT: Need to research fusion\nACTION: deep_research(query=\"fusion\", mode=\"deep\")"
        r = parse_nlm_response(text)
        assert r.tool_name == "deep_research"
        assert r.params["query"] == "fusion"
        assert r.params["mode"] == "deep"
        assert "research" in r.thought.lower()
        assert r.is_action

    def test_python_code_block(self):
        text = "I should research this.\n\n```python\ndeep_research(query=\"fusion energy\", mode=\"deep\")\n```"
        r = parse_nlm_response(text)
        assert r.code == 'deep_research(query="fusion energy", mode="deep")'
        assert r.is_code_block
        assert r.is_action

    def test_json_tool_params(self):
        text = '```json\n{"tool": "add_source", "params": {"url": "https://example.com"}}\n```'
        r = parse_nlm_response(text)
        assert r.tool_name == "add_source"
        assert r.params["url"] == "https://example.com"

    def test_json_thought_code(self):
        text = '```json\n{"thought": "Need to add source", "code": "add_source(url=\\"https://x.com\\")"}\n```'
        r = parse_nlm_response(text)
        assert r.code == 'add_source(url="https://x.com")'

    def test_prose_fallback(self):
        text = "Just a plain text response with no structure at all."
        r = parse_nlm_response(text)
        assert not r.is_action
        assert "plain text" in r.raw

    def test_empty_input(self):
        r = parse_nlm_response("")
        assert not r.is_action
        r2 = parse_nlm_response("  ")
        assert not r2.is_action

    def test_multi_line_thought(self):
        text = "THOUGHT: I need to first check\nwhat sources are available\nbefore deciding\nACTION: list_sources()"
        r = parse_nlm_response(text)
        assert r.tool_name == "list_sources"
        assert "check" in r.thought

    def test_action_after_explanatory_text(self):
        text = "The best approach is to search for recent papers.\n\n```python\nask_notebook(question=\"latest fusion papers\")\n```"
        r = parse_nlm_response(text)
        assert r.code == 'ask_notebook(question="latest fusion papers")'

    def test_conversational_wrapper_stripped(self):
        text = "Sure! Let me help with that.\n\nTHOUGHT: Need research\nACTION: deep_research(query=\"fusion\")"
        r = parse_nlm_response(text)
        assert r.tool_name == "deep_research"

    def test_malformed_action_falls_through(self):
        text = "Maybe I should try something like deep_research but I'm not sure."
        r = parse_nlm_response(text)
        assert not r.is_action  # No code block, no ACTION: prefix → fallback


class TestParsedAction:
    """Test ParsedAction properties and helpers."""

    def test_is_action_with_tool(self):
        a = ParsedAction(tool_name="test", params={})
        assert a.is_action
        assert not a.is_code_block

    def test_is_action_with_code(self):
        a = ParsedAction(code="print('hi')")
        assert a.is_action
        assert a.is_code_block

    def test_not_action_when_empty(self):
        a = ParsedAction(raw="just text")
        assert not a.is_action

    def test_to_code_string_from_tool(self):
        a = ParsedAction(tool_name="deep_research", params={"query": "fusion", "mode": "deep"})
        assert a.to_code_string() == "deep_research(query='fusion', mode='deep')"

    def test_to_code_string_from_code(self):
        a = ParsedAction(code="result = deep_research(query='fusion')")
        assert a.to_code_string() == "result = deep_research(query='fusion')"


class TestFormatSmolagentsOutput:
    """Test the output formatter for smolagents CodeAgent."""

    def test_code_block_passthrough(self):
        a = ParsedAction(code="deep_research(query='fusion')")
        assert format_smolagents_output(a) == "deep_research(query='fusion')"

    def test_tool_to_code(self):
        a = ParsedAction(tool_name="ask_notebook", params={"question": "what?"})
        assert format_smolagents_output(a) == "ask_notebook(question='what?')"

    def test_raw_text_wrapped(self):
        a = ParsedAction(raw="Some observation text here")
        result = format_smolagents_output(a)
        assert "Observation" in result
