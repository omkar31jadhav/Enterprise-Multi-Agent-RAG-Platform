from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from models import RetrievedChunk
from rag.citations import CitationReference


@dataclass(frozen=True)
class RetrievalEvaluation:
    relevant_count: int
    total_count: int
    score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CitationCoverageEvaluation:
    matched_references: int
    expected_references: int
    coverage: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnswerGroundingEvaluation:
    grounded_references: int
    total_references: int
    grounding_score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResponseCompletenessEvaluation:
    completeness_score: float
    details: dict[str, Any] = field(default_factory=dict)


def evaluate_retrieval_relevance(
    retrieved_chunks: Sequence[RetrievedChunk],
    min_score: float = 0.0,
) -> RetrievalEvaluation:
    total = len(retrieved_chunks)
    relevant = sum(1 for chunk in retrieved_chunks if chunk.score is not None and chunk.score >= min_score)
    score = relevant / total if total else 0.0
    return RetrievalEvaluation(
        relevant_count=relevant,
        total_count=total,
        score=score,
        details={"min_score": min_score},
    )


def evaluate_citation_coverage(
    citations: Sequence[CitationReference],
    expected_sources: Sequence[str] | None = None,
) -> CitationCoverageEvaluation:
    expected_set = set(expected_sources or [])
    matched = 0
    for citation in citations:
        if citation.source and citation.source in expected_set:
            matched += 1
    expected_total = len(expected_set) or len(citations)
    coverage = matched / expected_total if expected_total else 0.0
    return CitationCoverageEvaluation(
        matched_references=matched,
        expected_references=expected_total,
        coverage=coverage,
        details={"expected_sources": list(expected_set)},
    )


def evaluate_answer_grounding(
    answer: str,
    retrieved_chunks: Sequence[RetrievedChunk],
) -> AnswerGroundingEvaluation:
    total_references = len(retrieved_chunks)
    grounded = 0
    for chunk in retrieved_chunks:
        if chunk.source and chunk.source in answer:
            grounded += 1
    grounding_score = grounded / total_references if total_references else 0.0
    return AnswerGroundingEvaluation(
        grounded_references=grounded,
        total_references=total_references,
        grounding_score=grounding_score,
        details={"sources_found": grounded},
    )


def evaluate_response_completeness(answer: str) -> ResponseCompletenessEvaluation:
    length = len(answer.strip())
    completeness_score = min(1.0, length / 200.0)
    return ResponseCompletenessEvaluation(
        completeness_score=completeness_score,
        details={"length": length},
    )
