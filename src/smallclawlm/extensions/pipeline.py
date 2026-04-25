"""Pipeline — Fast path for SmallClawLM.

Chains NotebookLM operations without an agent. No LLM reasoning needed —
direct API calls for known intents like generate podcast, research, etc.

This is the core of the fast path. When the router determines that a user
request maps to a known intent (e.g. "generate podcast", "deep research"),
it creates a Pipeline and executes it directly, bypassing the agent loop
entirely. This saves NotebookLM rate limits for actual reasoning tasks.

Example:
    pipe = Pipeline(notebook_id="abc123")
    pipe.research("fusion energy")
    pipe.generate(ArtifactType.PODCAST)
    result = pipe.execute()
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from smallclawlm.auth import get_auth

logger = logging.getLogger(__name__)


class ArtifactType(Enum):
    PODCAST = "podcast"
    VIDEO = "video"
    QUIZ = "quiz"
    FLASHCARDS = "flashcards"
    MINDMAP = "mindmap"
    REPORT = "report"
    DATATABLE = "datatable"
    INFOGRAPHIC = "infographic"
    SLIDEDECK = "slidedeck"


@dataclass
class PipelineStep:
    """A single step in a pipeline."""
    action: str  # "research", "add_source", "generate", "download", "ask"
    params: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""
    success: bool
    outputs: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def __str__(self):
        if self.errors:
            return f"Pipeline failed: {'; '.join(self.errors)}"
        if self.outputs:
            lines = ["Pipeline completed:"]
            for k, v in self.outputs.items():
                val_str = str(v)[:200]
                lines.append(f"  {k}: {val_str}")
            return "\n".join(lines)
        return "Pipeline completed (no outputs)"


class Pipeline:
    """Chain NotebookLM operations into a pipeline.

    Example:
        pipeline = Pipeline()
        pipeline.add_source("https://arxiv.org/abs/2401.12345")
        pipeline.research("mRNA vaccine technology")
        pipeline.generate(ArtifactType.PODCAST)
        result = pipeline.execute()
    """

    def __init__(self, notebook_id: str | None = None, notebook_title: str = "Pipeline Session"):
        self.notebook_id = notebook_id
        self.notebook_title = notebook_title
        self.steps: list[PipelineStep] = []
        self._client = None

    def add_source(self, url: str, title: str | None = None) -> "Pipeline":
        self.steps.append(PipelineStep("add_source", {"url": url, "title": title}))
        return self

    def research(self, query: str, mode: str = "deep") -> "Pipeline":
        self.steps.append(PipelineStep("research", {"query": query, "mode": mode}))
        return self

    def generate(self, artifact_type: str) -> "Pipeline":
        """Add a generate step. artifact_type can be ArtifactType enum or string."""
        at = artifact_type.value if isinstance(artifact_type, ArtifactType) else artifact_type
        self.steps.append(PipelineStep("generate", {"artifact_type": at}))
        return self

    def ask(self, question: str) -> "Pipeline":
        self.steps.append(PipelineStep("ask", {"question": question}))
        return self

    def download(self, path: str) -> "Pipeline":
        self.steps.append(PipelineStep("download", {"path": path}))
        return self

    async def execute_async(self) -> PipelineResult:
        from notebooklm import NotebookLMClient

        auth = await get_auth()
        result = PipelineResult(success=True)

        async with NotebookLMClient(auth) as client:
            # Ensure notebook
            if self.notebook_id is None:
                nb = await client.notebooks.create(self.notebook_title)
                self.notebook_id = nb.id
                result.outputs["notebook_id"] = nb.id

            for i, step in enumerate(self.steps):
                try:
                    output = await self._execute_step(client, step)
                    result.outputs[f"step_{i}_{step.action}"] = output
                except Exception as e:
                    logger.error(f"Pipeline step {i} ({step.action}) failed: {e}")
                    result.errors.append(f"{step.action}: {e}")
                    result.success = False

        return result

    async def _execute_step(self, client, step: PipelineStep) -> str:
        if step.action == "add_source":
            url = step.params["url"]
            title = step.params.get("title")
            src = await client.sources.add_url(self.notebook_id, url, title=title)
            return f"Added source: {url} (id: {src.id})"

        elif step.action == "research":
            query = step.params["query"]
            mode = step.params.get("mode", "deep")
            await client.research.start(notebook_id=self.notebook_id, query=query, source="web", mode=mode)
            return f"Research started: {query}"

        elif step.action == "generate":
            at = step.params["artifact_type"]
            if at == "podcast":
                await client.artifacts.generate_podcast(notebook_id=self.notebook_id)
                return "Podcast generation started"
            elif at == "video":
                await client.artifacts.generate_video(notebook_id=self.notebook_id)
                return "Video generation started"
            elif at == "quiz":
                await client.artifacts.generate_quiz(notebook_id=self.notebook_id)
                return "Quiz generation started"
            elif at == "mindmap":
                await client.artifacts.generate_mindmap(notebook_id=self.notebook_id)
                return "Mind map generation started"
            elif at == "report":
                await client.artifacts.generate_report(notebook_id=self.notebook_id)
                return "Report generation started"
            else:
                return f"Unknown artifact type: {at}"

        elif step.action == "ask":
            question = step.params["question"]
            answer = await client.chat.ask(self.notebook_id, question)
            return answer.answer if hasattr(answer, "answer") else str(answer)

        elif step.action == "download":
            # TODO: implement artifact download
            return f"Download to {step.params['path']} (not yet implemented)"

        else:
            raise ValueError(f"Unknown pipeline action: {step.action}")

    def execute(self) -> PipelineResult:
        """Synchronous execute wrapper."""
        return asyncio.run(self.execute_async())
