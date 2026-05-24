from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from models import RetrievedChunk
from rag.citations import CitationReference
from rag.orchestration import OrchestrationMetrics


@dataclass
class WorkflowState:
    query: str
    session_id: str | None = None
    message_id: str | None = None
    workflow_execution_id: str | None = None
    workflow_path: str | None = None
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    conversation_history: str = ""
    prompt: str | None = None
    generated_answer: str | None = None
    citations: list[CitationReference] = field(default_factory=list)
    orchestration_metrics: OrchestrationMetrics | None = None
    evaluation_results: dict[str, Any] = field(default_factory=dict)
    prompt_template_used: str | None = None
    prompt_version: str | None = None
    workflow_duration: float | None = None
    node_timings: dict[str, float] = field(default_factory=dict)
    node_failures: dict[str, str] = field(default_factory=dict)


__all__ = ["WorkflowState"]
