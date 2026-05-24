from __future__ import annotations

import time
from typing import Any

from config.workflow import WorkflowConfig
from rag.orchestration import OrchestrationMetrics
from rag.memory import SQLConversationMemory
from services.generation_service import GenerationService
from services.retrieval_service import RetrievalService

from workflow.nodes import (
    citation_node,
    evaluation_node,
    generation_node,
    memory_node,
    prompt_selection_node,
    retrieval_node,
    _format_retrieved_context,
)
from workflow.state import WorkflowState


class SafeEvaluationNode:
    """Evaluation node that respects config flags."""

    def __init__(self, config: WorkflowConfig) -> None:
        self.config = config

    def run(self, state: WorkflowState) -> WorkflowState:
        if not self.config.workflow_evaluation_enabled:
            state.evaluation_results = {}
            return state
        return evaluation_node(state)


class WorkflowGraph:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        generation_service: GenerationService,
        memory: SQLConversationMemory | None = None,
        config: WorkflowConfig | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.generation_service = generation_service
        self.memory = memory or SQLConversationMemory()
        self.config = config or WorkflowConfig.disabled()
        self.name = "workflow_graph"
        self.safe_evaluation = SafeEvaluationNode(self.config)

    def run(self, state: WorkflowState) -> WorkflowState:
        overall_start = time.perf_counter()
        state.node_timings = {}
        state.node_failures = {}

        steps: list[tuple[str, Any]] = [
            ("retrieval", lambda current: retrieval_node(current, self.retrieval_service)),
            ("memory", lambda current: memory_node(current, self.memory)),
            ("prompt_selection", prompt_selection_node),
            ("generation", lambda current: generation_node(current, self.generation_service)),
            ("citation", citation_node),
            ("evaluation", self.safe_evaluation.run),
        ]

        for node_name, node_fn in steps:
            start = time.perf_counter()
            try:
                state = node_fn(state)
            except Exception as exc:
                state.node_failures[node_name] = str(exc)
                state.node_timings[node_name] = time.perf_counter() - start
                break
            else:
                state.node_timings[node_name] = time.perf_counter() - start

        if state.generated_answer is None:
            state.generated_answer = (
                "The workflow could not complete the request. "
                "Please try again later or use the standard answer method."
            )

        state.workflow_duration = time.perf_counter() - overall_start
        citation_count = len(state.citations)
        retrieval_scores = tuple(chunk.score for chunk in state.retrieved_chunks)
        context_size_estimate = len(_format_retrieved_context(state.retrieved_chunks))
        history_count = len(state.conversation_history.splitlines()) if state.conversation_history else 0

        state.orchestration_metrics = OrchestrationMetrics(
            memory_load_time=state.node_timings.get("memory", 0.0),
            prompt_assembly_time=state.node_timings.get("prompt_selection", 0.0),
            generation_time=state.node_timings.get("generation", 0.0),
            total_time=state.workflow_duration or 0.0,
            history_count=history_count,
            retrieved_chunk_count=len(state.retrieved_chunks),
            citation_count=citation_count,
            prompt_name=state.prompt_template_used or "",
            prompt_version=state.prompt_version or "",
            prompt_template_used=state.prompt_template_used or "",
            retrieval_scores=retrieval_scores,
            context_size_estimate=context_size_estimate,
        )
        return state


def build_workflow_graph(
    retrieval_service: RetrievalService,
    generation_service: GenerationService,
    memory: SQLConversationMemory | None = None,
    config: WorkflowConfig | None = None,
) -> WorkflowGraph:
    return WorkflowGraph(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
        memory=memory,
        config=config,
    )
