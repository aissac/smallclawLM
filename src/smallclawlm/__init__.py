"""SmallClawLM - Zero-token AI agent powered by Google NotebookLM."""

__version__ = "0.1.0"

from smallclawlm.nlm_model import NLMModel
from smallclawlm.nlm_agent import NLMAgent
from smallclawlm.nlm_tools import (
    DeepResearchTool,
    GeneratePodcastTool,
    GenerateVideoTool,
    GenerateQuizTool,
    GenerateMindMapTool,
    GenerateReportTool,
    AddSourceTool,
    ListSourcesTool,
    CreateNotebookTool,
    DedupSourcesTool,
    AskNotebookTool,
)
from smallclawlm.extensions.pipeline import Pipeline
from smallclawlm.extensions.batch import BatchProcessor
from smallclawlm.extensions.templates import NotebookTemplate

__all__ = [
    "NLMModel", "NLMAgent",
    "DeepResearchTool", "GeneratePodcastTool", "GenerateVideoTool",
    "GenerateQuizTool", "GenerateMindMapTool", "GenerateReportTool",
    "AddSourceTool", "ListSourcesTool", "CreateNotebookTool",
    "DedupSourcesTool", "AskNotebookTool",
    "Pipeline", "BatchProcessor", "NotebookTemplate",
]
