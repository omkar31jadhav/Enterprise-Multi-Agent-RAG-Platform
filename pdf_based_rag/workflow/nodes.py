from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from models import RetrievedChunk
from ml.evaluation import (
    evaluate_answer_grounding,
    evaluate_citation_coverage,
    evaluate_retrieval_relevance,
    evaluate_response_completeness,
)
from rag.citations import CitationReference, extract_citation_references, format_citation_references
from rag.memory import SQLConversationMemory
from rag.prompts import format_prompt, PromptTemplate
from services.generation_service import GenerationService
from services.retrieval_service import RetrievalService

from workflow.routing import select_prompt_template
from workflow.state import WorkflowState


def retrieval_node(
    state: WorkflowState,
    retrieval_service: RetrievalService,
    filters: Any | None = None,
) -> WorkflowState:
    state.retrieved_chunks = retrieval_service.retrieve(
        state.query,
        filters=filters,
        session_id=state.session_id,
        message_id=state.message_id,
    )
    return state


def memory_node(state: WorkflowState, memory: SQLConversationMemory) -> WorkflowState:
    if state.session_id:
        state.conversation_history = memory.prompt_ready_history(state.session_id)
    else:
        state.conversation_history = ""
    return state


def _build_prompt(
    template: PromptTemplate,
    query: str,
    history: str,
    retrieved_chunks: Iterable[RetrievedChunk],
) -> str:
    context = _format_retrieved_context(retrieved_chunks)
    if not context:
        context = "No relevant document content was found."

    if history:
        return format_prompt(
            template,
            history=history,
            context=context,
            question=query,
        )

    return format_prompt(
        template,
        context=context,
        question=query,
    )


def prompt_selection_node(state: WorkflowState) -> WorkflowState:
    prompt_template = select_prompt_template(state.conversation_history)
    state.prompt_template_used = prompt_template.name
    state.prompt_version = prompt_template.version
    state.prompt = _build_prompt(
        template=prompt_template,
        query=state.query,
        history=state.conversation_history,
        retrieved_chunks=state.retrieved_chunks,
    )
    return state


def generation_node(state: WorkflowState, generation_service: GenerationService) -> WorkflowState:
    if not state.prompt or not state.prompt.strip():
        state.generated_answer = "No prompt available for generation."
        return state

    state.generated_answer = generation_service.generate_from_prompt(state.prompt)
    return state


def _attach_citations(answer: str, citations: list[CitationReference]) -> str:
    if not citations:
        return answer
    citation_block = format_citation_references(citations)
    return f"{answer}\n\nReferences:\n{citation_block}"


def citation_node(state: WorkflowState) -> WorkflowState:
    state.citations = extract_citation_references(state.retrieved_chunks)
    if state.generated_answer is not None:
        state.generated_answer = _attach_citations(state.generated_answer, state.citations)
    return state


def evaluation_node(state: WorkflowState) -> WorkflowState:
    state.evaluation_results = {
        "retrieval": evaluate_retrieval_relevance(state.retrieved_chunks),
        "citation_coverage": evaluate_citation_coverage(state.citations),
        "answer_grounding": evaluate_answer_grounding(
            state.generated_answer or "",
            state.retrieved_chunks,
        ),
        "response_completeness": evaluate_response_completeness(state.generated_answer or ""),
    }
    return state


def _format_retrieved_context(retrieved_chunks: Iterable[RetrievedChunk]) -> str:
    context_parts: list[str] = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata: list[str] = [f"Source: {chunk.source}"]
        if chunk.page_number is not None:
            metadata.append(f"Page: {chunk.page_number}")
        if chunk.section_title:
            metadata.append(f"Section: {chunk.section_title}")
        if chunk.score is not None:
            metadata.append(f"Score: {chunk.score:.4f}")
        header = f"[{index}] {' | '.join(metadata)}"
        context_parts.append(f"{header}\n{chunk.content.strip()}")
    return "\n\n".join(context_parts)
