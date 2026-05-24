from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
import json
import sqlite3
from typing import Any
from uuid import uuid4

from database.models import (
    ChatMessageRecord,
    ChatSessionRecord,
    DatabaseStats,
    DocumentChunkRecord,
    DocumentRecord,
    FeedbackRecord,
    RetrievalLogRecord,
)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def to_json(data: dict[str, Any] | None) -> str:
    return json.dumps(data or {}, sort_keys=True)


def from_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    return json.loads(value)


def to_iso(value: datetime) -> str:
    return value.isoformat()


def from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


class DocumentRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create_document(self, record: DocumentRecord) -> DocumentRecord:
        self.connection.execute(
            """
            INSERT INTO documents (id, filename, file_type, ingestion_timestamp, total_pages, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.filename,
                record.file_type,
                to_iso(record.ingestion_timestamp),
                record.total_pages,
                to_json(record.metadata),
            ),
        )
        return record

    def upsert_chunks(self, chunks: Sequence[DocumentChunkRecord]) -> int:
        self.connection.executemany(
            """
            INSERT INTO document_chunks (
                chunk_id, document_id, content, page_number, section_title,
                table_detected, metadata, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                document_id = excluded.document_id,
                content = excluded.content,
                page_number = excluded.page_number,
                section_title = excluded.section_title,
                table_detected = excluded.table_detected,
                metadata = excluded.metadata,
                created_at = excluded.created_at
            """,
            [
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.content,
                    chunk.page_number,
                    chunk.section_title,
                    int(chunk.table_detected),
                    to_json(chunk.metadata),
                    to_iso(chunk.created_at),
                )
                for chunk in chunks
            ],
        )
        return len(chunks)

    def get_document(self, document_id: str) -> DocumentRecord | None:
        row = self.connection.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return _document_from_row(row) if row else None

    def list_documents(self, limit: int = 50) -> list[DocumentRecord]:
        rows = self.connection.execute(
            "SELECT * FROM documents ORDER BY ingestion_timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_document_from_row(row) for row in rows]


class ChatRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create_session(self, record: ChatSessionRecord) -> ChatSessionRecord:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO chat_sessions (session_id, created_at, user_identifier)
            VALUES (?, ?, ?)
            """,
            (record.session_id, to_iso(record.created_at), record.user_identifier),
        )
        return record

    def add_message(self, record: ChatMessageRecord) -> ChatMessageRecord:
        self.connection.execute(
            """
            INSERT INTO messages (message_id, session_id, role, content, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.message_id,
                record.session_id,
                record.role,
                record.content,
                to_iso(record.timestamp),
                to_json(record.metadata),
            ),
        )
        return record

    def list_messages(self, session_id: str, limit: int = 100) -> list[ChatMessageRecord]:
        rows = self.connection.execute(
            """
            SELECT * FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return [_message_from_row(row) for row in rows]

    def list_sessions(self, limit: int = 20) -> list[ChatSessionRecord]:
        rows = self.connection.execute(
            "SELECT * FROM chat_sessions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_session_from_row(row) for row in rows]


class RetrievalLogRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def add_logs(self, logs: Sequence[RetrievalLogRecord]) -> int:
        self.connection.executemany(
            """
            INSERT INTO retrieval_logs (
                retrieval_id, session_id, message_id, query, chunk_id, document_name,
                page_number, section_title, similarity_score, metadata, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    log.retrieval_id,
                    log.session_id,
                    log.message_id,
                    log.query,
                    log.chunk_id,
                    log.document_name,
                    log.page_number,
                    log.section_title,
                    log.similarity_score,
                    to_json(log.metadata),
                    to_iso(log.timestamp),
                )
                for log in logs
            ],
        )
        return len(logs)

    def list_recent(self, limit: int = 20, session_id: str | None = None) -> list[RetrievalLogRecord]:
        if session_id:
            rows = self.connection.execute(
                """
                SELECT * FROM retrieval_logs
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM retrieval_logs ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_retrieval_log_from_row(row) for row in rows]


class FeedbackRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def add_feedback(self, record: FeedbackRecord) -> FeedbackRecord:
        self.connection.execute(
            """
            INSERT INTO feedback (feedback_id, session_id, message_id, rating, comment, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.feedback_id,
                record.session_id,
                record.message_id,
                record.rating,
                record.comment,
                to_iso(record.created_at),
            ),
        )
        return record


class StatsRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def get_stats(self) -> DatabaseStats:
        return DatabaseStats(
            document_count=self._count("documents"),
            chunk_count=self._count("document_chunks"),
            chat_session_count=self._count("chat_sessions"),
            message_count=self._count("messages"),
            retrieval_count=self._count("retrieval_logs"),
            feedback_count=self._count("feedback"),
        )

    def _count(self, table_name: str) -> int:
        row = self.connection.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
        return int(row["count"])


def _document_from_row(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"],
        filename=row["filename"],
        file_type=row["file_type"],
        ingestion_timestamp=from_iso(row["ingestion_timestamp"]),
        total_pages=row["total_pages"],
        metadata=from_json(row["metadata"]),
    )


def _session_from_row(row: sqlite3.Row) -> ChatSessionRecord:
    return ChatSessionRecord(
        session_id=row["session_id"],
        created_at=from_iso(row["created_at"]),
        user_identifier=row["user_identifier"],
    )


def _message_from_row(row: sqlite3.Row) -> ChatMessageRecord:
    return ChatMessageRecord(
        message_id=row["message_id"],
        session_id=row["session_id"],
        role=row["role"],
        content=row["content"],
        timestamp=from_iso(row["timestamp"]),
        metadata=from_json(row["metadata"]),
    )


def _retrieval_log_from_row(row: sqlite3.Row) -> RetrievalLogRecord:
    return RetrievalLogRecord(
        retrieval_id=row["retrieval_id"],
        session_id=row["session_id"],
        message_id=row["message_id"],
        query=row["query"],
        chunk_id=row["chunk_id"],
        document_name=row["document_name"],
        page_number=row["page_number"],
        section_title=row["section_title"],
        similarity_score=row["similarity_score"],
        metadata=from_json(row["metadata"]),
        timestamp=from_iso(row["timestamp"]),
    )
