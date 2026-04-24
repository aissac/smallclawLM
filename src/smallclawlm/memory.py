"""Memory — Compact persistent memory that travels with every chat query.

Key insight: NotebookLM chat has limited effective input length.
We do NOT dump full smolagents message history into the chat window.
Instead, we carry a small sliding-window memory (~2K chars) that
summarizes what the agent knows so far.

This memory is prepended to every orchestrator query, keeping
context compact while preserving critical facts across steps.
"""

import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 2000
DEFAULT_MAX_FACTS = 30
MEMORY_DIR = Path.home() / ".smallclawlm" / "memory"


@dataclass
class AgentMemory:
    """Sliding-window compact memory for the orchestrator brain.

    Keeps the most recent and important facts within a character budget.
    Prepended to every chat.ask() call so the orchestrator has context
    without needing full conversation history.
    """

    max_chars: int = DEFAULT_MAX_CHARS
    max_facts: int = DEFAULT_MAX_FACTS
    facts: list[str] = field(default_factory=list)
    _persist_path: Path | None = None

    def add(self, fact: str) -> None:
        """Add a fact to memory, pruning oldest if over budget."""
        timestamp = time.strftime("%H:%M")
        entry = f"[{timestamp}] {fact}"
        self.facts.append(entry)
        self._prune()
        logger.debug(f"Memory add: {fact[:60]}... ({len(self.facts)} facts, {len(self.render())} chars)")

    def add_observation(self, tool: str, result: str, max_len: int = 200) -> None:
        """Add a tool observation to memory (auto-truncated)."""
        truncated = result[:max_len] + "..." if len(result) > max_len else result
        self.add(f"{tool} → {truncated}")

    def add_decision(self, thought: str, action: str) -> None:
        """Add an agent decision (thought + action) to memory."""
        short_thought = thought[:80] + "..." if len(thought) > 80 else thought
        self.add(f"Decided: {short_thought} → {action}")

    def _prune(self) -> None:
        """Remove oldest facts until within character budget."""
        # First, drop oldest facts if over max_facts
        while len(self.facts) > self.max_facts:
            self.facts.pop(0)

        # Then, drop oldest facts until within char budget
        while len(self.render()) > self.max_chars and len(self.facts) > 3:
            self.facts.pop(0)

    def render(self) -> str:
        """Render memory as compact text for prepending to prompts."""
        if not self.facts:
            return ""
        header = "[AGENT MEMORY]\n"
        body = "\n".join(self.facts)
        footer = "\n[END MEMORY]"
        return f"{header}{body}{footer}"

    def as_prefix(self) -> str:
        """Render as prefix for chat queries. Empty string if no facts."""
        rendered = self.render()
        if rendered:
            return rendered + "\n\n"
        return ""

    def clear(self) -> None:
        """Clear all facts."""
        self.facts.clear()

    def save(self, path: Path | str | None = None) -> None:
        """Persist memory to disk."""
        p = Path(path) if path else self._persist_path
        if p is None:
            p = MEMORY_DIR / "default.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"facts": self.facts}, indent=2))
        logger.debug(f"Memory saved to {p}")

    def load(self, path: Path | str | None = None) -> None:
        """Load memory from disk."""
        p = Path(path) if path else self._persist_path
        if p is None:
            p = MEMORY_DIR / "default.json"
        if p.exists():
            data = json.loads(p.read_text())
            self.facts = data.get("facts", [])
            logger.debug(f"Memory loaded from {p} ({len(self.facts)} facts)")

    def summary(self) -> dict:
        """Return a summary of memory state (for debugging)."""
        return {
            "fact_count": len(self.facts),
            "total_chars": len(self.render()),
            "max_chars": self.max_chars,
            "max_facts": self.max_facts,
            "latest": self.facts[-1] if self.facts else None,
        }

    def __repr__(self) -> str:
        return f"AgentMemory(facts={len(self.facts)}, chars={len(self.render())}/{self.max_chars})"


class BrainMemory:
    """Per-brain memory — tracks what each specialist brain has been asked.

    Prevents the orchestrator from asking the same specialist the same
    question twice, and stores brief summaries of previous brain responses.
    """

    def __init__(self, max_per_brain: int = 10):
        self.max_per_brain = max_per_brain
        self._brain_histories: dict[str, list[str]] = {}

    def record(self, brain_id: str, query: str, response_summary: str) -> None:
        """Record a brain interaction."""
        if brain_id not in self._brain_histories:
            self._brain_histories[brain_id] = []

        entry = f"Q: {query[:60]} → A: {response_summary[:80]}"
        self._brain_histories[brain_id].append(entry)

        # Prune oldest if over limit
        while len(self._brain_histories[brain_id]) > self.max_per_brain:
            self._brain_histories[brain_id].pop(0)

    def get_history(self, brain_id: str) -> list[str]:
        """Get interaction history for a brain."""
        return self._brain_histories.get(brain_id, [])

    def render_for_brain(self, brain_id: str) -> str:
        """Render history for a specific brain as context prefix."""
        history = self.get_history(brain_id)
        if not history:
            return ""
        lines = [f"[PREVIOUS QUERIES TO THIS BRAIN]"]
        for h in history[-5:]:  # Last 5 interactions
            lines.append(h)
        lines.append("[END PREVIOUS QUERIES]")
        return "\n".join(lines) + "\n\n"

    def clear(self, brain_id: str | None = None) -> None:
        """Clear history for a specific brain or all brains."""
        if brain_id:
            self._brain_histories.pop(brain_id, None)
        else:
            self._brain_histories.clear()
