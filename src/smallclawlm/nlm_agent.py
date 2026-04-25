"""NLMAgent — Slow-path agent backed by NotebookLM chat API.

Used when the router classifies input as needing reasoning (conversational,
analytical, multi-step). The fast path bypasses this entirely, going
straight to Pipeline / direct tool calls.

Architecture:
  Fast path:  user input → router → Pipeline → direct API call → result
  Slow path:  user input → router → NLMAgent → NLMModel.generate() → chat.ask()
              → CodeAgent executes tools → result

One agent = one notebook. The notebook's sources define its specialty.
No external LLM API keys needed — all reasoning runs through Gemini inside NotebookLM.
"""

import logging
from smolagents import CodeAgent

from smallclawlm.nlm_model import NLMModel
from smallclawlm.nlm_tools import (
    ALL_TOOLS, RESEARCH_TOOLS, PODCAST_TOOLS, QUIZ_TOOLS,
    REPORT_TOOLS, MINDMAP_TOOLS,
)

logger = logging.getLogger(__name__)

TOOL_PRESETS = {
    "research": RESEARCH_TOOLS,
    "podcast": PODCAST_TOOLS,
    "quiz": QUIZ_TOOLS,
    "report": REPORT_TOOLS,
    "mindmap": MINDMAP_TOOLS,
    "all": ALL_TOOLS,
}

DEFAULT_INSTRUCTIONS = """You are a SmallClawLM agent powered by Google NotebookLM.
You have access to notebook tools for research, content generation, and analysis.

IMPORTANT RULES:
1. Use tools to gather information before answering.
2. When you have enough information, call final_answer() with your complete response.
3. If a tool returns an error, read the suggested fix and try again.
4. Always call final_answer() when done — it is the ONLY way to end execution.
5. Keep your code blocks simple — one tool call per block.
"""


class NLMAgent:
    """One agent, one notebook, one specialty.

    Uses NotebookLM's built-in Gemini as the reasoning engine.
    No external LLM API keys required.
    """

    def __init__(
        self,
        notebook_id: str | None = None,
        notebook_title: str | None = None,
        tools: str | list[type] = "all",
        instructions: str | None = None,
        additional_authorized_imports: list[str] | None = None,
        planning_interval: int | None = None,
        max_steps: int = 10,
        verbosity_level: int = 1,
        upload_sources: list[str] | None = None,
    ):
        self._notebook_id = notebook_id
        self._upload_sources = upload_sources or []

        # Resolve tool preset
        if isinstance(tools, str):
            tool_classes = TOOL_PRESETS.get(tools, ALL_TOOLS)
        else:
            tool_classes = tools

        tool_instances = [cls() for cls in tool_classes]

        # Create model pointing to this notebook
        model = NLMModel(
            notebook_id=notebook_id,
            notebook_title=notebook_title or "SmallClawLM Agent",
            auto_create=True,
        )

        self.agent = CodeAgent(
            model=model,
            tools=tool_instances,
            instructions=instructions or DEFAULT_INSTRUCTIONS,
            max_steps=max_steps,
            verbosity_level=verbosity_level,
            additional_authorized_imports=additional_authorized_imports or [],
            planning_interval=planning_interval,
        )

    def run(self, task: str) -> str:
        """Run a task and return the result."""
        return self.agent.run(task)


def create_agent(specialty: str = "all", **kwargs) -> NLMAgent:
    """Convenience factory for creating agents by specialty."""
    return NLMAgent(tools=specialty, **kwargs)
