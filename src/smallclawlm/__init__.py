"""SmallClawLM — Zero-token AI agent where NotebookLM IS the brain.

NotebookLM's built-in Gemini handles all reasoning and tool calling.
No local LLM needed. No external API keys required.

Architecture:
  - NLMModel wraps NotebookLM chat API as a smolagents Model
  - CodeAgent uses NLMModel for structured <code> tool calls
  - NotebookRouter auto-selects the best notebook per query
  - NLMMemory persists facts, research, and decisions across sessions
"""

__version__ = "0.7.0"

from smallclawlm.nlm_model import NLMModel
from smallclawlm.nlm_memory import NLMMemory
from smallclawlm.nlm_agent import NLMAgent, create_agent
from smallclawlm.notebook_router import NotebookRouter, RouteResult

__all__ = [
    "NLMModel", "NLMMemory",
    "NLMAgent", "create_agent",
    "NotebookRouter", "RouteResult",
]
