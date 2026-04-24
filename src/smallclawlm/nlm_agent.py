"""NLMAgent — One agent, one notebook, one specialty.

Each agent owns exactly one NotebookLM notebook. The notebook's sources
define its specialty. No cross-brain routing, no multi-brain orchestration.

Usage:
    # Research agent
    agent = NLMAgent(notebook_id="abc123", tools="research")
    result = agent.run("What are the latest fusion energy breakthroughs?")

    # Podcast agent (different notebook, different tools)
    agent = NLMAgent(notebook_id="xyz789", tools="podcast")
    result = agent.run("Create a podcast about climate change")

    # Custom tool set
    agent = NLMAgent(notebook_id="abc123", tools=[AskNotebookTool, AddSourceTool])
"""

import logging
from pathlib import Path
from typing import Any

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


class NLMAgent:
    """One agent, one notebook, one specialty.

    Uses NotebookLM's built-in Gemini as the reasoning engine.
    No external LLM API keys required.

    Uses CodeAgent because NotebookLM outputs Python code blocks
    naturally — no need to force structured JSON tool calls.
    """

    def __init__(
        self,
        notebook_id: str | None = None,
        notebook_title: str | None = None,
        tools: str | list[type] = "all",
        additional_authorized_imports: list[str] | None = None,
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
            concise=True,
        )

        self.agent = CodeAgent(
            model=model,
            tools=tool_instances,
            max_steps=max_steps,
            verbosity_level=verbosity_level,
            additional_authorized_imports=additional_authorized_imports or [],
        )

        # Share notebook_id with tools so they operate on the same notebook
        self._model = model
        self._share_notebook_id()

    def _share_notebook_id(self):
        """Propagate notebook_id to all tool instances after model creates notebook."""
        # Ensure model has initialized its notebook
        if self._notebook_id:
            self._model._notebook_id = self._notebook_id

        # Set notebook_id on tool instances that support it
        for tool in self.agent.tools:
            if hasattr(tool, '_notebook_id'):
                tool._notebook_id = self._notebook_id

    def run(self, task: str, **kwargs) -> str:
        """Run the agent on a task.

        First call triggers lazy notebook creation and source upload.
        """
        # Ensure notebook exists before running
        self._model._run_async(self._model._ensure_notebook())

        # Propagate the (possibly auto-created) notebook_id
        self._notebook_id = self._model._notebook_id
        self._share_notebook_id()

        # Upload any initial sources
        if self._upload_sources:
            self._upload_initial_sources()

        logger.info(f"Starting NLMAgent with task: {task[:100]}...")
        return self.agent.run(task, **kwargs)

    async def _upload_initial_sources(self):
        """Upload any initial source URLs to the notebook."""
        if not self._upload_sources or not self._notebook_id:
            return
        client = await self._model._ensure_client()
        for url in self._upload_sources:
            try:
                source = await client.sources.add_url(
                    self._notebook_id, url, wait=True
                )
                logger.info(f"Uploaded source: {source.id}")
            except Exception as e:
                logger.warning(f"Failed to upload source {url}: {e}")


def create_agent(
    specialty: str = "research",
    notebook_id: str | None = None,
    **kwargs,
) -> NLMAgent:
    """Factory function to create a specialized agent.

    Args:
        specialty: One of "research", "podcast", "quiz", "report", "mindmap"
        notebook_id:_existing notebook ID (auto-creates if None)
        **kwargs: Additional args passed to NLMAgent

    Returns:
        Configured NLMAgent ready to run
    """
    return NLMAgent(
        notebook_id=notebook_id,
        tools=specialty,
        **kwargs,
    )
