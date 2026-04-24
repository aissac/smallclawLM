"""SmallClawLM - Zero-token AI agent powered by Google NotebookLM.

One agent, one notebook, one specialty.
Each agent owns exactly one NotebookLM notebook. The notebook sources define
its domain expertise. No external LLM API keys needed.
"""

__version__ = "0.2.0"

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
    DirectResponseTool,
    ALL_TOOLS,
    RESEARCH_TOOLS,
    PODCAST_TOOLS,
    QUIZ_TOOLS,
    REPORT_TOOLS,
    MINDMAP_TOOLS,
)
from smallclawlm.memory import AgentMemory
from smallclawlm.parser import parse_nlm_response, ParsedAction

__all__ = [
    "NLMModel", "NLMAgent", "create_agent",
    "DeepResearchTool", "AskNotebookTool", "GeneratePodcastTool",
    "GenerateVideoTool", "GenerateQuizTool", "GenerateMindMapTool",
    "GenerateReportTool", "AddSourceTool", "ListSourcesTool",
    "CreateNotebookTool", "DirectResponseTool",
    "ALL_TOOLS", "RESEARCH_TOOLS", "PODCAST_TOOLS",
    "QUIZ_TOOLS", "REPORT_TOOLS", "MINDMAP_TOOLS",
    "AgentMemory", "parse_nlm_response", "ParsedAction",
]
