"""SmallClawLM - Zero-token AI agent powered by Google NotebookLM.

Architecture: Hybrid Fast/Slow Path
  - Fast Path (Pipeline): Direct tool calls for known intents (podcast, report, quiz, etc.)
  - Slow Path (Agent): Full CodeAgent reasoning with NLMModel for conversational/analytical tasks

No external LLM API keys needed. All reasoning runs through Google's Gemini
inside NotebookLM.
"""

__version__ = "0.3.0"

from smallclawlm.nlm_model import NLMModel
from smallclawlm.nlm_agent import NLMAgent, create_agent
from smallclawlm.nlm_tools import (
    DeepResearchTool,
    AskNotebookTool,
    GeneratePodcastTool,
    GenerateVideoTool,
    GenerateQuizTool,
    GenerateMindMapTool,
    GenerateReportTool,
    AddSourceTool,
    ListSourcesTool,
    CreateNotebookTool,
    ALL_TOOLS,
    RESEARCH_TOOLS,
    PODCAST_TOOLS,
    QUIZ_TOOLS,
    REPORT_TOOLS,
    MINDMAP_TOOLS,
)
from smallclawlm.router import route, Path as RoutePath, RouteResult
from smallclawlm.memory import AgentMemory

__all__ = [
    "NLMModel", "NLMAgent", "create_agent",
    "DeepResearchTool", "AskNotebookTool", "GeneratePodcastTool",
    "GenerateVideoTool", "GenerateQuizTool", "GenerateMindMapTool",
    "GenerateReportTool", "AddSourceTool", "ListSourcesTool",
    "CreateNotebookTool",
    "ALL_TOOLS", "RESEARCH_TOOLS", "PODCAST_TOOLS",
    "QUIZ_TOOLS", "REPORT_TOOLS", "MINDMAP_TOOLS",
    "route", "RoutePath", "RouteResult",
    "AgentMemory",
]
