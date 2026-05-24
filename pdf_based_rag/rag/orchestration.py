from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from ml.tracking import NoOpTrackingAdapter, TrackingAdapter
from models import RetrievedChunk
from rag.citations import CitationReference, extract_citation_references, format_citation_references
from rag.memory import SQLConversationMemory
from rag.prompts import (
    CITATION_AWARE_QA_PROMPT,
    CONVERSATIONAL_QA_PROMPT,
    format_prompt,
    PromptTemplate,
)
from utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class OrchestrationMetrics:
    memory_load_time: float
    prompt_assembly_time: float
    generation_time: float
    total_time: float
    history_count: int
    retrieved_chunk_count: int
    citation_count: int
    prompt_name: str
    prompt_version: str
    prompt_template_used: str
    retrieval_scores: tuple[float | None, ...]
    context_size_estimate: int


@dataclass(frozen=True)
class OrchestratedResponse:
    answer: str
    prompt: str
    citations: list[CitationReference]
    retrieved_chunks: list[RetrievedChunk]
    metrics: OrchestrationMetrics


# Preserve backward compatibility for earlier tests and callers.
OrchestrationResult = OrchestratedResponse


class RagOrchestrator:
    """Lightweight orchestration for conversational retrieval chains."""

    MAX_CONTEXT_CHARS = 3200

    def __init__(
        self,
        memory: SQLConversationMemory | None = None,
        tracking_adapter: TrackingAdapter | None = None,
    ) -> None:
        self.memory = memory or SQLConversationMemory()
        self.tracking_adapter = tracking_adapter or NoOpTrackingAdapter()
        self.name = "rag_orchestrator"

    def run(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        session_id: str | None = None,
        message_id: str | None = None,
        generation_function: Callable[[str], str] | None = None,
    ) -> OrchestratedResponse:
        if generation_function is None:
            raise ValueError("A generation function must be provided to run orchestration.")

        retrieved_chunks = retrieved_chunks or []
        retrieved_chunk_count = len(retrieved_chunks)
        start = time.perf_counter()

        history = self._load_history(session_id)
        history_count = len(history.splitlines()) if history else 0
        history_truncated = self._truncate_history(history)
        memory_load_time = time.perf_counter() - start

        prompt_template = self._select_prompt_template(history_truncated)
        prompt_start = time.perf_counter()
        prompt = self._build_prompt(
            prompt_template=prompt_template,
            query=query,
            history=history_truncated,
            retrieved_chunks=retrieved_chunks,
        )
        prompt_assembly_time = time.perf_counter() - prompt_start

        logger.info(
            "Orchestrator prompt assembled prompt_name=%s session_id=%s message_id=%s retrieved_chunks=%s history_count=%s",
            prompt_template.name,
            session_id,
            message_id,
            retrieved_chunk_count,
            history_count,
        )

        generation_start = time.perf_counter()
        answer = self._safe_generate(generation_function, prompt)
        generation_time = time.perf_counter() - generation_start

        references = self._safe_extract_references(retrieved_chunks)
        citation_count = len(references)
        answer_with_citations = self._attach_citations(answer, references)

        total_time = time.perf_counter() - start
        retrieval_scores = tuple(chunk.score for chunk in retrieved_chunks)
        context = self._format_retrieved_context(retrieved_chunks)
        context_size_estimate = len(context)
        metrics = OrchestrationMetrics(
            memory_load_time=memory_load_time,
            prompt_assembly_time=prompt_assembly_time,
            generation_time=generation_time,
            total_time=total_time,
            history_count=history_count,
            retrieved_chunk_count=retrieved_chunk_count,
            citation_count=citation_count,
            prompt_name=prompt_template.name,
            prompt_version=prompt_template.version,
            prompt_template_used=prompt_template.name,
            retrieval_scores=retrieval_scores,
            context_size_estimate=context_size_estimate,
        )

        self._log_tracking_metrics(prompt_template, metrics, session_id, message_id)

        logger.info(
            "Completed orchestration session_id=%s message_id=%s total_time=%.4f citations=%s",
            session_id,
            message_id,
            metrics.total_time,
            citation_count,
        )

        return OrchestratedResponse(
            answer=answer_with_citations,
            prompt=prompt,
            citations=references,
            retrieved_chunks=retrieved_chunks,
            metrics=metrics,
        )

    def _log_tracking_metrics(
        self,
        prompt_template: PromptTemplate,
        metrics: OrchestrationMetrics,
        session_id: str | None,
        message_id: str | None,
    ) -> None:
        self.tracking_adapter.log_params(
            {
                "orchestration_path": self.name,
                "prompt_template_used": prompt_template.name,
                "prompt_version": prompt_template.version,
                "prompt_name": prompt_template.name,
                "session_id": session_id or "none",
                "message_id": message_id or "none",
            }
        )
        self.tracking_adapter.log_metrics(
            {
                "memory_load_time": metrics.memory_load_time,
                "prompt_assembly_time": metrics.prompt_assembly_time,
                "generation_time": metrics.generation_time,
                "orchestration_total_time": metrics.total_time,
                "retrieved_chunk_count": metrics.retrieved_chunk_count,
                "citation_count": metrics.citation_count,
                "history_message_count": metrics.history_count,
                "context_size_estimate": metrics.context_size_estimate,
            }
        )
        self.tracking_adapter.log_params(
            {"retrieval_scores": str(metrics.retrieval_scores)}
        )

    def _load_history(self, session_id: str | None) -> str:
        if not session_id:
            return ""
        try:
            return self.memory.prompt_ready_history(session_id)
        except Exception:
            logger.exception("Failed to load conversation history for session_id=%s", session_id)
            return "No prior conversation history."

    def _build_prompt(
        self,
        prompt_template: PromptTemplate,
        query: str,
        history: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> str:
        context = self._format_retrieved_context(retrieved_chunks)
        if not context:
            context = "No relevant document content was found."

        if history:
            return format_prompt(
                prompt_template,
                history=history,
                context=context,
                question=query,
            )

        return format_prompt(
            prompt_template,
            context=context,
            question=query,
        )

    def _format_retrieved_context(self, retrieved_chunks: list[RetrievedChunk]) -> str:
        if not retrieved_chunks:
            return ""

        context_parts: list[str] = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            metadata_parts = self._build_metadata_parts(chunk)
            header = f"[{index}] {' | '.join(metadata_parts)}"
            content = chunk.content.strip() if chunk.content else ""
            context_parts.append(f"{header}\n{content}")

        context = "\n\n".join(context_parts)
        return self._truncate_context(context)

    def _build_metadata_parts(self, chunk: RetrievedChunk) -> list[str]:
        parts: list[str] = [f"Source: {chunk.source or 'unknown'}"]
        if chunk.page_number is not None:
            parts.append(f"Page: {chunk.page_number}")
        if chunk.section_title:
            parts.append(f"Section: {chunk.section_title}")
        if chunk.chunk_id:
            parts.append(f"Chunk ID: {chunk.chunk_id}")
        if chunk.score is not None:
            parts.append(f"Score: {chunk.score:.4f}")
        return parts

    def _truncate_context(self, context: str) -> str:
        if len(context) <= self.MAX_CONTEXT_CHARS:
            return context
        truncated = context[: self.MAX_CONTEXT_CHARS].rstrip()
        logger.warning(
            "Truncated retrieved context from %s to %s characters",
            len(context),
            len(truncated),
        )
        return f"{truncated}\n\n[Truncated additional context]"

    def _truncate_history(self, history: str) -> str:
        max_chars = self.memory.max_history_chars
        if len(history) <= max_chars:
            return history
        truncated = history[-max_chars :].lstrip()
        logger.warning(
            "Truncated conversation history from %s to %s characters",
            len(history),
            len(truncated),
        )
        return f"...{truncated}"

    def _safe_generate(self, generation_function: Callable[[str], str], prompt: str) -> str:
        try:
            return generation_function(prompt)
        except Exception:
            logger.exception("Generation failed for prompt. Returning fallback message.")
            return "The system could not generate an answer at this time. Please try again."

    @staticmethod
    def _safe_extract_references(chunks: list[RetrievedChunk]) -> list[CitationReference]:
        try:
            return extract_citation_references(chunks)
        except Exception:
            logger.exception("Citation extraction failed. Continuing without citations.")
            return []

    @staticmethod
    def _attach_citations(answer: str, references: list[CitationReference]) -> str:
        if not references:
            return answer
        citation_block = format_citation_references(references)
        return f"{answer}\n\nReferences:\n{citation_block}"

    def _select_prompt_template(self, history: str) -> PromptTemplate:
        return CONVERSATIONAL_QA_PROMPT if history else CITATION_AWARE_QA_PROMPT
