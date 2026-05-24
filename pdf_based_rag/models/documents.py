from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class DocumentMetadata:
    filename: str
    file_type: str
    total_pages: int | None = None
    ingestion_timestamp: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedSection:
    content: str
    document_name: str
    page_number: int | None = None
    section_title: str | None = None
    table_detected: bool = False
    hierarchy: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedDocument:
    metadata: DocumentMetadata
    sections: list[ExtractedSection]


@dataclass(frozen=True)
class DocumentChunk:
    content: str
    id: str | None = None
    document_name: str = ""
    source: str | None = None
    page_number: int | None = None
    section_title: str | None = None
    chunk_index: int | None = None
    table_detected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.document_name and self.source:
            object.__setattr__(self, "document_name", self.source)
        if self.source is None and self.document_name:
            object.__setattr__(self, "source", self.document_name)
        if self.id:
            return
        identity = "|".join(
            [
                self.document_name,
                str(self.page_number if self.page_number is not None else ""),
                self.section_title or "",
                str(self.chunk_index if self.chunk_index is not None else ""),
                self.content,
            ]
        )
        object.__setattr__(self, "id", sha256(identity.encode("utf-8")).hexdigest()[:16])


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    source: str
    score: float | None = None
    page_number: int | None = None
    section_title: str | None = None
    chunk_id: str | None = None
    table_detected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_context(self) -> str:
        reference_parts = [f"Source: {self.source}"]
        if self.page_number is not None:
            reference_parts.append(f"Page: {self.page_number}")
        if self.section_title:
            reference_parts.append(f"Section: {self.section_title}")
        if self.score is not None:
            reference_parts.append(f"Score: {self.score:.4f}")
        return f"{' | '.join(reference_parts)}\n{self.content}"
