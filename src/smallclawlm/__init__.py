"""SmallClawLM — Zero-token AI agent where NotebookLM IS the brain.

Three-layer architecture:
  Layer 1: REFLEX  — SmolLM3 local model (10.7 tok/s, tool selection)
  Layer 2: COGNITION — NotebookLM chat (reasoning, synthesis, citations)
  Layer 3: MEMORY — NLMMemory (persistent notebook, auto-sync, cross-session)

The agent doesn't "call" NotebookLM — it lives in NotebookLM.
Every thought, decision, and research result flows through the notebook automatically.
"""

__version__ = "0.6.0"

from smallclawlm.orchestrator import OrchestratorModel
from smallclawlm.nlm_memory import NLMMemory
from smallclawlm.nlm_agent import NLMAgent, create_agent
from smallclawlm.smollm_model import SmolLMModel
from smallclawlm.nlm_model import NLMModel
from smallclawlm.notebook_router import NotebookRouter, RouteResult

__all__ = [
    "OrchestratorModel", "NLMMemory",
    "NLMAgent", "create_agent",
    "SmolLMModel", "NLMModel",
    "NotebookRouter", "RouteResult",
]
