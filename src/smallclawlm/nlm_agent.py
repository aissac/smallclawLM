"""NLMAgent — Agent where NotebookLM IS the brain.

Uses OrchestratorModel (3-layer: reflex/cognition/memory) as the model
and NLMTools as the hands. Every generate() call automatically:
  1. Injects memory context from the notebook
  2. Falls back to NotebookLM chat for unknowns
  3. Auto-syncs results back to the notebook

The agent doesn't "decide" to use NotebookLM — it IS NotebookLM.
"""

import logging
from smolagents import CodeAgent

from smallclawlm.orchestrator import OrchestratorModel
from smallclawlm.nlm_memory import NLMMemory
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

SYSTEM_INSTRUCTIONS = """You are SmallClawLM — an AI agent that IS NotebookLM.
You don't just USE NotebookLM, you LIVE in it. Every fact, research result, and
decision is automatically stored in your notebook memory.

Your memory grows with every session. When you're uncertain, you automatically
query your accumulated knowledge. When you research, results become permanent.

IMPORTANT RULES:
1. Use tools to gather information before answering.
2. When you have enough information, call final_answer() with your complete response.
3. If a tool returns an error, read the suggested fix and try again.
4. Always call final_answer() when done — it's the ONLY way to end execution.
5. Keep code blocks simple — one tool call per block.
6. Use exact parameter names from tool descriptions.
"""


def create_agent(
    notebook_id: str | None = None,
    notebook_title: str = "SmallClawLM Agent",
    tools: str | list[type] = "all",
    model_path: str | None = None,
    n_threads: int = 4,
    n_ctx: int = 2048,
    **kwargs,
) -> "NLMAgent":
    """Factory function to create an agent with sensible defaults."""
    agent = NLMAgent(
        notebook_id=notebook_id,
        notebook_title=notebook_title,
        tools=tools,
        model_path=model_path,
        n_threads=n_threads,
        n_ctx=n_ctx,
        **kwargs,
    )
    return agent


class NLMAgent:
    """One agent, one notebook, one brain.

    The OrchestratorModel handles the 3-layer architecture automatically.
    The agent just calls tools and returns answers — memory and cognition
    are built into every generate() call.

    Args:
        notebook_id: NotebookLM notebook ID. Auto-creates one if None.
        notebook_title: Title for auto-created notebook.
        tools: Tool preset name ("all", "research", etc.) or list of tool classes.
        model_path: Path to GGUF model file (smollm backend only).
        n_threads: Number of threads for SmolLM3 (optimal: 4 for dual-channel DDR4).
        n_ctx: Context window size for SmolLM3.
    """

    def __init__(
        self,
        notebook_id: str | None = None,
        notebook_title: str | None = None,
        tools: str | list[type] = "all",
        instructions: str | None = None,
        additional_authorized_imports: list[str] | None = None,
        planning_interval: int | None = None,
        model_path: str | None = None,
        n_threads: int = 4,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        max_steps: int = 10,
    ):
        self._notebook_id = notebook_id
        self._notebook_title = notebook_title or "SmallClawLM Agent"

        # Resolve tool preset
        if isinstance(tools, str):
            tool_classes = TOOL_PRESETS.get(tools, ALL_TOOLS)
        else:
            tool_classes = tools

        # Instantiate tools
        self._tool_instances = [cls() for cls in tool_classes]

        # Set notebook_id on all tools
        if notebook_id:
            for tool in self._tool_instances:
                tool._notebook_id = notebook_id

        # Create OrchestratorModel (3-layer brain)
        self._memory = NLMMemory(
            notebook_id=notebook_id,
            notebook_title=self._notebook_title,
        )
        self._model = OrchestratorModel(
            memory=self._memory,
            notebook_id=notebook_id,
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
        )

        # Build the CodeAgent
        self._agent = CodeAgent(
            model=self._model,
            tools=self._tool_instances,
            instructions=instructions or SYSTEM_INSTRUCTIONS,
            additional_authorized_imports=additional_authorized_imports or [],
            planning_interval=planning_interval,
        )
        self._max_steps = max_steps

    @property
    def memory(self) -> NLMMemory:
        """Access the agent's persistent memory."""
        return self._memory

    @property
    def model(self) -> OrchestratorModel:
        """Access the orchestrator model."""
        return self._model

    @property
    def notebook_id(self) -> str | None:
        """Get the notebook ID (resolves lazily)."""
        return self._memory.notebook_id

    def run(self, task: str, **kwargs):
        """Run a task through the agent.

        Every call automatically:
        1. Injects memory context
        2. Uses SmolLM3 for tool selection (reflex)
        3. Escalates to NotebookLM for unknowns (cognition)
        4. Auto-syncs results to memory
        """
        try:
            result = self._agent.run(task, max_steps=self._max_steps, **kwargs)
            # Auto-sync the result to memory
            if result:
                self._memory.add_observation("agent_result", str(result)[:500])
            return result
        except Exception as e:
            self._memory.add(f"Agent error: {e}")
            raise
