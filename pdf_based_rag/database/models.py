from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from models import utc_now


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    filename: str
    file_type: str
    ingestion_timestamp: datetime
    total_pages: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunkRecord:
    chunk_id: str
    document_id: str
    content: str
    page_number: int | None
    section_title: str | None
    table_detected: bool
    metadata: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class ChatSessionRecord:
    session_id: str
    created_at: datetime = field(default_factory=utc_now)
    user_identifier: str | None = None


@dataclass(frozen=True)
class ChatMessageRecord:
    message_id: str
    session_id: str
    role: str
    content: str
    timestamp: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalLogRecord:
    retrieval_id: str
    query: str
    timestamp: datetime
    session_id: str | None = None
    message_id: str | None = None
    chunk_id: str | None = None
    document_name: str | None = None
    page_number: int | None = None
    section_title: str | None = None
    similarity_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeedbackRecord:
    feedback_id: str
    session_id: str
    rating: int
    created_at: datetime
    message_id: str | None = None
    comment: str | None = None


@dataclass(frozen=True)
class DatabaseStats:
    document_count: int
    chunk_count: int
    chat_session_count: int
    message_count: int
    retrieval_count: int
    feedback_count: int
