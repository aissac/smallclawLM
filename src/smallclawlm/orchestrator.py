"""Orchestrator — The three-layer brain that IS NotebookLM.

This replaces the simple fast/slow router with a true three-layer architecture:

  Layer 1: REFLEX  — SmolLM3 local (10.7 tok/s, <code> tool calls, fast responses)
  Layer 2: COGNITION — NotebookLM chat (reasoning, synthesis, citations from sources)
  Layer 3: MEMORY — NLMMemory (persistent notebook, auto-sync, cross-session)

The orchestrator decides which layer handles each generate() call:
  - Simple tool calls → Layer 1 only (fast, zero latency)
  - Questions needing context → Layer 1 + Layer 2 (SmolLM3 calls NLM query tool)
  - Multi-step reasoning → All three layers (memory feeds cognition feeds reflex)

The key insight: The agent doesn't "call" NotebookLM as an external service.
NotebookLM IS its memory and reasoning. Every generate() has memory context
injected. Every answer can escalate to cognition. Every result auto-syncs.

NotebookRouter integration:
  When no notebook_id is provided, the Orchestrator uses NotebookRouter
  to automatically select the best notebook based on the query topic.
  This means `smallclaw run "research fusion"` Just Works — no manual -n flag.
"""

import logging
import re
from typing import Optional

from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import TokenUsage

from smallclawlm.nlm_memory import NLMMemory
from smallclawlm.smollm_model import SmolLMModel
from smallclawlm.notebook_router import NotebookRouter

logger = logging.getLogger(__name__)


class OrchestratorModel(Model):
    """Three-layer model that seamlessly blends SmolLM3 + NotebookLM.

    From the smolagents perspective, this IS a Model subclass. But internally,
    it routes between local inference and NotebookLM cognition based on query
    complexity, injecting memory context automatically.

    The agent never "decides" to use NotebookLM — the orchestrator handles
    escalation transparently. SmolLM3 sees memory context in every prompt.

    If no notebook_id is given, NotebookRouter picks the best one automatically.
    """

    def __init__(
        self,
        memory: NLMMemory | None = None,
        notebook_id: str | None = None,
        model_path: str | None = None,
        n_ctx: int = 2048,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
        model_id: str = "smallclawlm-orchestrator",
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)

        # Layer 1: REFLEX (local SmolLM3)
        self._reflex = SmolLMModel(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
        )

        # NotebookRouter for automatic notebook selection
        self._router = NotebookRouter()
        self._routed_notebook_id: str | None = None

        # Layer 3: MEMORY (NotebookLM-backed persistence)
        if memory is not None:
            self._memory = memory
            self._notebook_id = memory.notebook_id or notebook_id
        else:
            self._notebook_id = notebook_id
            self._memory = NLMMemory(notebook_id=self._notebook_id)

    def _resolve_notebook_id(self, query: str | None = None) -> str | None:
        """Resolve notebook_id, using router if none provided.

        If notebook_id is already set, use it directly.
        If not, use NotebookRouter to pick the best one based on the query.
        """
        if self._notebook_id:
            return self._notebook_id

        if query:
            result = self._router.route_sync(query)
            logger.info(f"Router selected notebook '{result.title}' (score={result.score:.2f}, {result.match_level})")
            self._notebook_id = result.notebook_id
            self._routed_notebook_id = result.notebook_id
            # Update memory to use the routed notebook
            self._memory.notebook_id = result.notebook_id
            return result.notebook_id

        return None

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        tools_to_call_from: list | None = None,
        **kwargs,
    ) -> ChatMessage:
        """Generate with automatic memory injection and cognition escalation.

        Flow:
        1. Resolve notebook_id via router if needed
        2. Inject memory context into the prompt (Layer 3 → Layer 1)
        3. SmolLM3 generates response (Layer 1: REFLEX)
        4. If response indicates knowledge gap, auto-escalate to Layer 2
        5. Auto-sync result to memory (Layer 1 → Layer 3)
        """
        # Step 0: Auto-resolve notebook if not set
        query = self._extract_query(messages)
        self._resolve_notebook_id(query)

        # Step 1: Inject memory context
        enriched_messages = self._inject_memory(messages)

        # Step 2: Generate via SmolLM3 (fast, local)
        response = self._reflex.generate(
            enriched_messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )

        # Step 3: Check for knowledge gaps → escalate to cognition
        content = response.content if isinstance(response.content, str) else str(response.content)
        if self._needs_cognition(content):
            logger.debug("Escalating to cognition layer (NotebookLM chat)")
            answer = self._memory.query(self._extract_query(messages))
            # Feed cognition result back to SmolLM3 for tool-call formatting
            cognized = self._reflex.generate(
                self._inject_cognition(enriched_messages, answer),
                stop_sequences=stop_sequences,
                tools_to_call_from=tools_to_call_from,
                **kwargs,
            )
            content = cognized.content if isinstance(cognized.content, str) else str(cognized.content)

        # Step 4: Auto-sync to memory
        self._auto_sync(messages, content)

        return ChatMessage(role=MessageRole.ASSISTANT, content=content)

    def _inject_memory(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Inject memory context into the system prompt.

        Every generate() call gets the agent's accumulated knowledge prepended,
        so SmolLM3 always has context from past sessions, research, and decisions.
        """
        memory_context = self._memory.render()
        if not memory_context:
            return messages

        enriched = list(messages)
        if enriched and enriched[0].role == MessageRole.SYSTEM:
            # Merge memory into existing system prompt
            original = enriched[0].content if isinstance(enriched[0].content, str) else str(enriched[0].content)
            enriched[0] = ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"{original}\n\n--- Agent Memory ---\n{memory_context}",
            )
        else:
            # Prepend memory as system context
            enriched.insert(0, ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"Agent Memory:\n{memory_context}",
            ))

        return enriched

    def _needs_cognition(self, response: str) -> bool:
        """Detect if SmolLM3's response indicates a knowledge gap.

        Triggers: "I don't know", "I'm not sure", hallucination markers,
        or when the response is too short (likely uncertain).
        """
        uncertainty_patterns = [
            r"\bI don'?t know\b",
            r"\bI'?m not sure\b",
            r"\bI cannot\b",
            r"\bI can'?t (?:answer|determine|find|recall|remember)\b",
            r"\bno (?:information|data|knowledge|record)s?\b",
            r"\bunknown\b",
        ]
        for pattern in uncertainty_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return True

        # Very short responses to reasoning questions suggest uncertainty
        if len(response.strip()) < 20:
            return True

        return False

    def _extract_query(self, messages: list[ChatMessage]) -> str:
        """Extract the core question from the message history for cognition."""
        # Use the last user message as the query
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                return content[:500]  # NotebookLM limit
        return "Please help with the current task"

    def _inject_cognition(
        self, messages: list[ChatMessage], cognition_answer: str
    ) -> list[ChatMessage]:
        """Feed the cognition layer's answer back as context for SmolLM3."""
        enriched = list(messages)
        enriched.append(ChatMessage(
            role=MessageRole.SYSTEM,
            content=f"NotebookLM knowledge retrieval result:\n{cognition_answer[:3000]}\n\nUse this information to answer the user's question or call the appropriate tool.",
        ))
        return enriched

    def _auto_sync(self, messages: list[ChatMessage], response: str):
        """Auto-sync significant interactions to memory."""
        # Extract the task/query
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if len(content) > 20:  # Only sync meaningful interactions
                    self._memory.add_observation("agent_response", response, max_len=300)
                break

    def __repr__(self):
        nb = self._notebook_id or "auto"
        return f"OrchestratorModel(reflex=SmolLM3, memory=NLMMemory(nb={nb}))"
