"""Parser — Extract structured actions from NotebookLM free-form responses.

Three-tier fallback strategy:
1. Structured: THOUGHT/ACTION blocks or JSON code blocks
2. Python code blocks: Extract smolagents-compatible tool calls
3. Fallback: Return raw text as a direct observation

NotebookLM naturally outputs Markdown code blocks when prompted with
source-based format examples. This parser handles the common imperfections:
missing quotes, trailing commas, mixed formats, conversational wrappers.
"""

import json
import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ParsedAction:
    """A parsed structured action from NotebookLM response."""
    thought: str = ""
    tool_name: str = ""
    params: dict = field(default_factory=dict)
    code: str = ""  # Raw Python code if this is a code-block action
    raw: str = ""   # Original text if parse failed

    @property
    def is_action(self) -> bool:
        """Whether this represents a callable action (not just text)."""
        return bool(self.tool_name) or bool(self.code)

    @property
    def is_code_block(self) -> bool:
        """Whether this is a Python code block (smolagents CodeAgent style)."""
        return bool(self.code) and not self.tool_name

    def to_code_string(self) -> str:
        """Convert to Python code string for smolagents CodeAgent execution."""
        if self.code:
            return self.code
        if self.tool_name:
            # Convert tool call to Python function call
            args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
            return f"{self.tool_name}({args})"
        return self.raw


def parse_nlm_response(text: str) -> ParsedAction:
    """Parse a NotebookLM response into a structured action.

    Tries three strategies in order:
    1. THOUGHT/ACTION blocks or inline JSON
    2. Python code blocks (```python ... ```)
    3. Fallback: entire response as observation text
    """
    if not text or not text.strip():
        return ParsedAction(raw=text or "")

    # Strategy 1a: THOUGHT: ... ACTION: ... blocks
    parsed = _parse_thought_action(text)
    if parsed and parsed.is_action:
        logger.debug(f"Parsed THOUGHT/ACTION: {parsed.tool_name or 'code'}")
        return parsed

    # Strategy 1b: JSON code blocks with tool/params schema
    parsed = _parse_json_action(text)
    if parsed and parsed.is_action:
        logger.debug(f"Parsed JSON action: {parsed.tool_name}")
        return parsed

    # Strategy 2: Python code blocks
    parsed = _parse_code_block(text)
    if parsed and parsed.is_action:
        logger.debug(f"Parsed code block: {parsed.code[:80]}...")
        return parsed

    # Strategy 3: Fallback — entire response as observation
    logger.debug("No structured action found, using raw text fallback")
    return ParsedAction(raw=text.strip())


def _parse_thought_action(text: str) -> ParsedAction | None:
    """Extract THOUGHT and ACTION blocks from text.

    Handles formats like:
        THOUGHT: I need to research this
        ACTION: ask_notebook({"notebook_id": "abc", "query": "..."})

    Also handles inline format:
        THOUGHT: ... | ACTION: tool(params)
    """
    # Try multi-line THOUGHT/ACTION
    thought_m = re.search(
        r'THOUGHT\s*:\s*(.+?)(?=\n\s*ACTION\s*:|\n\s*```|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    action_m = re.search(
        r'ACTION\s*:\s*(\w+)\s*\((.*?)\)',
        text, re.DOTALL | re.IGNORECASE
    )

    if action_m:
        tool_name = action_m.group(1)
        params_raw = action_m.group(2).strip()
        params = _fuzzy_json_parse(params_raw)
        thought = thought_m.group(1).strip() if thought_m else ""
        return ParsedAction(thought=thought, tool_name=tool_name, params=params)

    # Try ACTION as Python code block after THOUGHT
    action_block = re.search(
        r'ACTION\s*:\s*\n*\s*```python\s*\n(.*?)\n\s*```',
        text, re.DOTALL | re.IGNORECASE
    )
    if action_block:
        code = action_block.group(1).strip()
        thought = thought_m.group(1).strip() if thought_m else ""
        return ParsedAction(thought=thought, code=code)

    return None


def _parse_json_action(text: str) -> ParsedAction | None:
    """Extract JSON actions from code blocks.

    Handles:
        ```json
        {"tool": "deep_research", "params": {"query": "..."}}
        ```

    Also handles:
        ```json
        {"thought": "...", "code": "deep_research(query='...')"}
        ```
    """
    # Try JSON code blocks
    json_blocks = re.findall(
        r'```(?:json)?\s*\n(.*?)\n\s*```',
        text, re.DOTALL
    )

    for block in json_blocks:
        data = _fuzzy_json_parse(block)
        if isinstance(data, dict):
            # Schema 1: {"tool": "...", "params": {...}}
            if "tool" in data:
                return ParsedAction(
                    thought=data.get("thought", ""),
                    tool_name=data["tool"],
                    params=data.get("params", {}),
                )
            # Schema 2: {"thought": "...", "code": "..."}
            if "code" in data:
                return ParsedAction(
                    thought=data.get("thought", ""),
                    code=data["code"],
                )

    return None


def _parse_code_block(text: str) -> ParsedAction | None:
    """Extract Python code blocks for smolagents CodeAgent.

    Handles:
        ```python
        deep_research(query="fusion energy", mode="deep")
        ```

    Also handles mixed responses where the code block comes after
    explanatory text.
    """
    # Find all Python code blocks
    code_blocks = re.findall(
        r'```python\s*\n(.*?)\n\s*```',
        text, re.DOTALL
    )

    if not code_blocks:
        # Try without language tag
        code_blocks = re.findall(
            r'```\s*\n(.*?)\n\s*```',
            text, re.DOTALL
        )

    if code_blocks:
        # Use the last (most complete) code block
        code = code_blocks[-1].strip()
        # Extract any preceding thought text
        thought = _extract_thought_before_code(text, code)
        return ParsedAction(thought=thought, code=code)

    return None


def _extract_thought_before_code(text: str, code: str) -> str:
    """Extract the reasoning text that appears before a code block."""
    code_idx = text.find(code)
    if code_idx <= 0:
        return ""

    pre_text = text[:code_idx].strip()
    # Remove any markdown code fence prefix
    pre_text = re.sub(r'```\w*\s*$', '', pre_text).strip()

    # Take last 500 chars of reasoning (keep it compact)
    if len(pre_text) > 500:
        pre_text = "..." + pre_text[-500:]

    return pre_text


def _fuzzy_json_parse(s: str) -> dict:
    """Parse JSON with common NotebookLM quirks.

    Handles:
    - Unquoted keys: {query: "foo"}
    - Single quotes: {'query': 'foo'}
    - Trailing commas: {"query": "foo",}
    - Python-style booleans: {deep: True}
    - None values: {mode: None}
    - URL values that contain colons
    """
    s = s.strip()
    if not s:
        return {}

    # Remove wrapping quotes if present
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]

    # Try strict JSON first (fast path — handles well-formed input including URLs)
    try:
        result = json.loads(s)
        return result if isinstance(result, dict) else {"raw": s}
    except json.JSONDecodeError:
        pass

    # Fuzzy parsing for NotebookLM quirks
    fixed = s
    # Fix single quotes to double quotes
    fixed = fixed.replace("'", '"')
    # Fix trailing commas before closing braces/brackets
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    # Fix Python booleans/None (only as standalone values, not inside strings)
    fixed = re.sub(r'\bTrue\b', 'true', fixed)
    fixed = re.sub(r'\bFalse\b', 'false', fixed)
    fixed = re.sub(r'\bNone\b', 'null', fixed)
    # Fix unquoted keys: only match word chars followed by colon,
    # but NOT if preceded by a quote (already quoted)
    fixed = re.sub(r'(?<!")(\b\w+)\s*:', r'"\1":', fixed)
    # Remove double-quoting that may have occurred on already-quoted keys
    fixed = re.sub(r'""(\w+)"":', r'"\1":', fixed)
    # Fix unquoted string values: "key": value  → "key": "value"
    # Only match values that are bare words (not numbers, not already quoted, not bools/null)
    fixed = re.sub(
        r':\s*([a-zA-Z_][a-zA-Z0-9_-]*)(\s*[,}])',
        r': "\1"\2',
        fixed
    )

    try:
        result = json.loads(fixed)
        return result if isinstance(result, dict) else {"raw": s}
    except json.JSONDecodeError:
        pass

    # Last resort: try to extract key=value pairs (for Python-style function args)
    pairs = re.findall(r'(\w+)\s*=\s*["\']([^"\']*)["\']', s)
    if pairs:
        return {k: v for k, v in pairs}

    # Final fallback: try key=value without quotes
    pairs = re.findall(r'(\w+)\s*=\s*([^,\s}]+)', s)
    if pairs:
        return {k: v.strip('"').strip("'") for k, v in pairs}

    return {"raw": s}


def format_smolagents_output(parsed: ParsedAction) -> str:
    """Format a ParsedAction into output suitable for smolagents CodeAgent.

    CodeAgent expects Python code in ```python ... ``` blocks.
    If we have a tool_name with params, we convert it to a function call.
    If we have raw code, we pass it through.
    If we have raw text, we wrap it as a string observation.
    """
    if parsed.is_code_block:
        # Already have Python code — just return it
        return parsed.code

    if parsed.tool_name:
        # Convert tool call to Python function call
        args = ", ".join(f"{k}={v!r}" for k, v in parsed.params.items())
        return f"{parsed.tool_name}({args})"

    # Raw text — return as string comment/observation
    return f"# Observation: {parsed.raw[:200]}"
