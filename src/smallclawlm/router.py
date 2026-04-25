"""Router - Intent classification for fast path vs slow path.

Fast path: Direct tool calls without LLM reasoning (pipelines).
Slow path: Full CodeAgent with NLMModel reasoning (conversational).

The router classifies user input using simple keyword matching.
If intent is clear -> fast path (direct Pipeline call).
If ambiguous or conversational -> slow path (NLMAgent with NLMModel brain).

This avoids burning NotebookLM rate limits on "thinking" calls
when the user just wants a podcast generated.
"""

import re
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Path(Enum):
    FAST = "fast"
    SLOW = "slow"


@dataclass
class RouteResult:
    path: Path
    intent: str
    confidence: float
    params: dict


_INTENTS = [
    (r"\bgenerate\s+podcast\b", Path.FAST, "generate_podcast", {}),
    (r"\bpodcast\b", Path.FAST, "generate_podcast", {}),
    (r"\baudio\s+overview\b", Path.FAST, "generate_podcast", {}),
    (r"\bgenerate\s+video\b", Path.FAST, "generate_video", {}),
    (r"\bexplainer\s+video\b", Path.FAST, "generate_video", {}),
    (r"\bgenerate\s+report\b", Path.FAST, "generate_report", {}),
    (r"\bsummary\s+report\b", Path.FAST, "generate_report", {}),
    (r"\breport\b", Path.FAST, "generate_report", {}),
    (r"\bgenerate\s+quiz\b", Path.FAST, "generate_quiz", {}),
    (r"\bquiz\b", Path.FAST, "generate_quiz", {}),
    (r"\btest\s+me\b", Path.FAST, "generate_quiz", {}),
    (r"\bmind\s*map\b", Path.FAST, "generate_mind_map", {}),
    (r"\bconcept\s+map\b", Path.FAST, "generate_mind_map", {}),
    (r"\bdeep\s+research\b", Path.FAST, "deep_research", {}),
    (r"\bresearch\s+on\b", Path.FAST, "deep_research", {}),
    (r"\bresearch\s+about\b", Path.FAST, "deep_research", {}),
    (r"\bresearch\b", Path.FAST, "deep_research", {}),
    (r"\blist\s+(?:the\s+)?sources\b", Path.FAST, "list_sources", {}),
    (r"\bshow\s+(?:the\s+)?sources\b", Path.FAST, "list_sources", {}),
    (r"\bwhat\s+(?:are\s+the\s+)?sources\b", Path.FAST, "list_sources", {}),
    (r"\badd\s+(?:a\s+)?source\b", Path.FAST, "add_source", {}),
    (r"\bload\s+(?:a\s+)?url\b", Path.FAST, "add_source", {}),
    (r"\bcreate\s+(?:a\s+)?notebook\b", Path.FAST, "create_notebook", {}),
    (r"\bnew\s+notebook\b", Path.FAST, "create_notebook", {}),
    (r"\bwhy\b", Path.SLOW, "ask", {}),
    (r"\bhow\b", Path.SLOW, "ask", {}),
    (r"\bexplain\b", Path.SLOW, "ask", {}),
    (r"\bcompare\b", Path.SLOW, "ask", {}),
    (r"\banalyze\b", Path.SLOW, "ask", {}),
    (r"\bevaluate\b", Path.SLOW, "ask", {}),
    (r"\bsummarize\b", Path.SLOW, "ask", {}),
    (r"\bsummarise\b", Path.SLOW, "ask", {}),
    (r"\btell\s+me\b", Path.SLOW, "ask", {}),
    (r"\bdescribe\b", Path.SLOW, "ask", {}),
    (r"\bwhat\s+is\b", Path.SLOW, "ask", {}),
    (r"\bwhat\s+are\b", Path.SLOW, "ask", {}),
    (r"\bcan\s+you\b", Path.SLOW, "ask", {}),
    (r"\bhelp\s+me\b", Path.SLOW, "ask", {}),
]


def route(user_input: str) -> RouteResult:
    text = user_input.lower().strip()
    for pattern, path, intent, params in _INTENTS:
        if re.search(pattern, text):
            confidence = 0.9 if path == Path.FAST else 0.7
            extracted_params = dict(params)
            if intent in ("deep_research", "ask"):
                for prefix in ("research on ", "research about ", "deep research ", "research "):
                    if prefix in text:
                        extracted_params["query"] = text.split(prefix, 1)[1].strip()
                        break
                if "query" not in extracted_params:
                    extracted_params["query"] = user_input
            logger.info(f"{'Fast' if path == Path.FAST else 'Slow'} path: {intent} (confidence={confidence})")
            return RouteResult(path=path, intent=intent, confidence=confidence, params=extracted_params)
    logger.info("Slow path: ask (default, no pattern match)")
    return RouteResult(path=Path.SLOW, intent="ask", confidence=0.5, params={"query": user_input})
