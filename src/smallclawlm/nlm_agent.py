"""NLMAgent — Pre-configured smolagents CodeAgent with NotebookLM as the brain.

This is the main entry point: a ready-to-use agent that thinks using
NotebookLM's Gemini and acts using NotebookLM's APIs.
"""

import logging
from typing import Any

from smolagents import CodeAgent

from smallclawlm.nlm_model import NLMModel
from smallclawlm.nlm_tools import (
    DeepResearchTool,
    GeneratePodcastTool,
    GenerateVideoTool,
    GenerateQuizTool,
    GenerateFlashcardsTool,
    GenerateMindMapTool,
    GenerateReportTool,
    AddSourceTool,
    ListSourcesTool,
    CreateNotebookTool,
    DedupSourcesTool,
)

logger = logging.getLogger(__name__)

# Default set of tools for the agent
DEFAULT_TOOLS = [
    DeepResearchTool,
    GeneratePodcastTool,
    GenerateVideoTool,
    GenerateQuizTool,
    GenerateFlashcardsTool,
    GenerateMindMapTool,
    GenerateReportTool,
    AddSourceTool,
    ListSourcesTool,
    CreateNotebookTool,
    DedupSourcesTool,
]


class NLMAgent:
    """Zero-token AI agent powered by Google NotebookLM.

    Uses NotebookLM's Gemini as the reasoning engine and NotebookLM's
    APIs as actions. No external LLM API keys needed.

    Args:
        model: smolagents Model instance. Defaults to NLMModel.
        notebook_id: NotebookLM notebook ID to use.
        notebook_title: Title for auto-created notebooks.
        tools: List of Tool classes to provide. Defaults to all NLM tools.
        additional_tools: Extra tools to add beyond defaults.
        max_steps: Maximum agent reasoning steps.
        verbosity_level: Logging verbosity (0-2).
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
        # Default to NLMModel (zero external tokens!)
        if model is None:
            model = NLMModel(
                notebook_id=notebook_id,
                notebook_title=notebook_title,
                auto_create=True,
            )

        # Instantiate tool classes
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
        """Run the agent on a task.

        Args:
            task: Natural language task description.

        Returns:
            Agent's final answer.
        """
        logger.info(f"Starting NLMAgent with task: {task[:100]}...")
        return self.agent.run(task, **kwargs)
