from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from models import RetrievedChunk


@dataclass(frozen=True)
class CitationReference:
    source: str
    page_number: int | None = None
    section_title: str | None = None
    chunk_id: str | None = None
    score: float | None = None

    def format(self) -> str:
        parts = [f"Source: {self.source}"]
        if self.page_number is not None:
            parts.append(f"Page: {self.page_number}")
        if self.section_title:
            parts.append(f"Section: {self.section_title}")
        if self.chunk_id:
            parts.append(f"Chunk ID: {self.chunk_id}")
        if self.score is not None:
            parts.append(f"Score: {self.score:.4f}")
        return " | ".join(parts)


def extract_citation_references(chunks: Iterable[RetrievedChunk]) -> list[CitationReference]:
    references: list[CitationReference] = []
    for chunk in chunks:
        references.append(
            CitationReference(
                source=chunk.source,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                chunk_id=chunk.chunk_id,
                score=chunk.score,
            )
        )
    return references


def format_citation_references(references: Iterable[CitationReference]) -> str:
    formatted = [reference.format() for reference in references]
    return "\n".join(formatted)


def render_citation_block(references: Iterable[CitationReference]) -> str:
    header = "References:" if any(True for _ in references) else "No references available."
    lines = [reference.format() for reference in references]
    return f"{header}\n" + "\n".join(lines) if lines else header
