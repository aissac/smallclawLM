"""SmallClawLM — Zero-token AI agent powered by SmolLM3 + NotebookLM."""

__version__ = "0.4.0"

from smallclawlm.router import route, Path, RouteResult
from smallclawlm.nlm_agent import NLMAgent, create_agent
from smallclawlm.smollm_model import SmolLMModel
from smallclawlm.nlm_model import NLMModel

__all__ = [
    "route", "RouteResult", "Path",
    "NLMAgent", "create_agent",
    "SmolLMModel", "NLMModel",
]
