"""NLMAgent — One agent, one notebook, one specialty.

Each agent owns exactly one NotebookLM notebook. The notebook's sources
define its specialty. No cross-brain routing, no multi-brain orchestration.

Usage:
    # Research agent
    agent = NLMAgent(notebook_id="abc123", tools="research")
    result = agent.run("What are the latest fusion energy breakthroughs?")

    # Custom instructions
    agent = NLMAgent(
        notebook_id="abc123",
        instructions="You are a marine biology expert. Focus on ocean ecosystems.",
    )
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
4. Always call final_answer() when done — it is the ONLY way to end execution."""


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
            concise=True,
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

        self._model = model
        self._share_notebook_id()

    def _share_notebook_id(self):
        """Propagate notebook_id to all tool instances."""
        if self._notebook_id:
            self._model._notebook_id = self._notebook_id
        for tool in self.agent.tools:
            if hasattr(tool, '_notebook_id'):
                tool._notebook_id = self._notebook_id

    def run(self, task: str, **kwargs) -> str:
        """Run the agent on a task."""
        self._model._run_async(self._model._ensure_notebook())
        self._notebook_id = self._model._notebook_id
        self._share_notebook_id()

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
    """Factory function to create a specialized agent."""
    return NLMAgent(
        notebook_id=notebook_id,
        tools=specialty,
        **kwargs,
    )
