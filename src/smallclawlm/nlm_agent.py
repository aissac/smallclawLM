"""NLMAgent - Pre-configured smolagents CodeAgent with NotebookLM as the brain.

IMPORTANT: Must use CodeAgent, NOT ToolCallingAgent.
NotebookLM cannot return structured JSON tool calls - it outputs Python code.
CodeAgent parses Python code blocks from the model output.
"""

import logging
from typing import Any
from smolagents import CodeAgent
from smallclawlm.nlm_model import NLMModel
from smallclawlm.nlm_tools import DEFAULT_TOOLS

logger = logging.getLogger(__name__)


class NLMAgent:
    """Zero-token AI agent powered by Google NotebookLM.

    Uses NotebookLM Gemini as reasoning engine + NotebookLM APIs as actions.
    No external LLM API keys needed.

    Uses CodeAgent (not ToolCallingAgent) because NotebookLM outputs
    Python code blocks, not structured JSON tool calls.
    """

    def __init__(
        self,
        model: Any | None = None,
        notebook_id: str | None = None,
        notebook_title: str | None = None,
        tools: list[type] | None = None,
        additional_tools: list[type] | None = None,
        max_steps: int = 20,
        verbosity_level: int = 1,
    ):
        if model is None:
            model = NLMModel(
                notebook_id=notebook_id,
                notebook_title=notebook_title,
                auto_create=True,
            )

        tool_classes = tools or DEFAULT_TOOLS
        tool_instances = [cls() for cls in tool_classes]

        if additional_tools:
            tool_instances.extend(cls() for cls in additional_tools)

        self.agent = CodeAgent(
            model=model,
            tools=tool_instances,
            max_steps=max_steps,
            verbosity_level=verbosity_level,
        )

    def run(self, task: str, **kwargs) -> str:
        """Run the agent on a task."""
        logger.info(f"Starting NLMAgent with task: {task[:100]}...")
        return self.agent.run(task, **kwargs)
