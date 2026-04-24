"""Pipeline — Chain NotebookLM operations without an agent.

Pipelines let you compose research → generate → download sequences
as a simple declarative workflow, no agent reasoning needed.
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

    def generate(self, artifact_type: ArtifactType) -> "Pipeline":
        self.steps.append(PipelineStep("generate", {"artifact_type": artifact_type.value}))
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
                    output = await self._execute_step(client, step, result)
                    result.outputs[f"step_{i}"] = output
                except Exception as e:
                    result.success = False
                    result.errors.append(f"Step {i} ({step.action}): {e}")
                    logger.error(f"Pipeline step {i} failed: {e}")

        return result

    def execute(self) -> PipelineResult:
        return asyncio.run(self.execute_async())

    async def _execute_step(self, client, step: PipelineStep, result: PipelineResult) -> Any:
        if step.action == "add_source":
            src = await client.sources.add_url(
                self.notebook_id,
                step.params["url"],
                title=step.params.get("title"),
            )
            return {"source_id": src.id, "status": src.status}

        elif step.action == "research":
            r = await client.research.start(
                notebook_id=self.notebook_id,
                query=step.params["query"],
                source="web",
                mode=step.params.get("mode", "deep"),
            )
            poll = await client.research.poll(self.notebook_id)
            return {"report": poll.get("report", "")}

        elif step.action == "generate":
            artifact_type = step.params["artifact_type"]
            gen_map = {
                "podcast": client.artifacts.generate_audio,
                "video": client.artifacts.generate_video,
                "quiz": client.artifacts.generate_quiz,
                "flashcards": client.artifacts.generate_flashcards,
                "mindmap": client.artifacts.generate_mind_map,
                "report": client.artifacts.generate_report,
            }
            gen_fn = gen_map.get(artifact_type)
            if gen_fn is None:
                raise ValueError(f"Unknown artifact type: {artifact_type}")
            gen_result = await gen_fn(self.notebook_id)
            return {"task_id": getattr(gen_result, "task_id", "completed")}

        elif step.action == "ask":
            chat_result = await client.chat.ask(
                self.notebook_id,
                step.params["question"],
            )
            return {"answer": chat_result.answer if hasattr(chat_result, "answer") else str(chat_result)}

        else:
            raise ValueError(f"Unknown pipeline action: {step.action}")
