from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256

from database.connection import Database
from database.models import (
    ChatMessageRecord,
    ChatSessionRecord,
    DatabaseStats,
    DocumentChunkRecord,
    DocumentRecord,
    FeedbackRecord,
    RetrievalLogRecord,
)
from database.repositories import (
    ChatRepository,
    DocumentRepository,
    FeedbackRepository,
    RetrievalLogRepository,
    StatsRepository,
    new_id,
)
from models import DocumentChunk, RetrievedChunk, utc_now
from utils import get_logger


logger = get_logger(__name__)


class PersistenceService:
    def __init__(self, database: Database | None = None, initialize: bool = True) -> None:
        self.database = database or Database()
        if initialize:
            self.database.initialize()

    def record_ingestion(
        self,
        filename: str,
        file_type: str,
        chunks: Sequence[DocumentChunk],
        total_pages: int | None = None,
        metadata: dict | None = None,
    ) -> DocumentRecord:
        document = DocumentRecord(
            id=_document_id(filename, utc_now().isoformat()),
            filename=filename,
            file_type=file_type,
            ingestion_timestamp=utc_now(),
            total_pages=total_pages,
            metadata=metadata or {},
        )
        chunk_records = [
            DocumentChunkRecord(
                chunk_id=str(chunk.id),
                document_id=document.id,
                content=chunk.content,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                table_detected=chunk.table_detected,
                metadata=dict(chunk.metadata),
                created_at=chunk.created_at,
            )
            for chunk in chunks
        ]

        with self.database.transaction() as connection:
            documents = DocumentRepository(connection)
            documents.create_document(document)
            documents.upsert_chunks(chunk_records)

        logger.info(
            "Recorded ingestion document=%s chunks=%s database=%s",
            filename,
            len(chunk_records),
            self.database.sqlite_path,
        )
        return document

    def create_chat_session(self, user_identifier: str | None = None, session_id: str | None = None) -> ChatSessionRecord:
        session = ChatSessionRecord(
            session_id=session_id or new_id("session"),
            created_at=utc_now(),
            user_identifier=user_identifier,
        )
        with self.database.transaction() as connection:
            ChatRepository(connection).create_session(session)
        return session

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
        message_id: str | None = None,
    ) -> ChatMessageRecord:
        message = ChatMessageRecord(
            message_id=message_id or new_id("msg"),
            session_id=session_id,
            role=role,
            content=content,
            timestamp=utc_now(),
            metadata=metadata or {},
        )
        with self.database.transaction() as connection:
            ChatRepository(connection).add_message(message)
        return message

    def log_retrievals(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        session_id: str | None = None,
        message_id: str | None = None,
    ) -> int:
        logs = [
            RetrievalLogRecord(
                retrieval_id=new_id("retrieval"),
                session_id=session_id,
                message_id=message_id,
                query=query,
                chunk_id=chunk.chunk_id,
                document_name=chunk.source,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                similarity_score=chunk.score,
                metadata=dict(chunk.metadata),
                timestamp=utc_now(),
            )
            for chunk in chunks
        ]
        if not logs:
            return 0
        with self.database.transaction() as connection:
            RetrievalLogRepository(connection).add_logs(logs)
        logger.info("Recorded retrieval logs count=%s session_id=%s", len(logs), session_id)
        return len(logs)

    def add_feedback(
        self,
        session_id: str,
        rating: int,
        message_id: str | None = None,
        comment: str | None = None,
    ) -> FeedbackRecord:
        feedback = FeedbackRecord(
            feedback_id=new_id("feedback"),
            session_id=session_id,
            message_id=message_id,
            rating=rating,
            comment=comment,
            created_at=utc_now(),
        )
        with self.database.transaction() as connection:
            FeedbackRepository(connection).add_feedback(feedback)
        return feedback

    def list_chat_messages(self, session_id: str, limit: int = 100) -> list[ChatMessageRecord]:
        with self.database.connect() as connection:
            return ChatRepository(connection).list_messages(session_id=session_id, limit=limit)

    def list_chat_sessions(self, limit: int = 20) -> list[ChatSessionRecord]:
        with self.database.connect() as connection:
            return ChatRepository(connection).list_sessions(limit=limit)

    def list_documents(self, limit: int = 50) -> list[DocumentRecord]:
        with self.database.connect() as connection:
            return DocumentRepository(connection).list_documents(limit=limit)

    def list_recent_retrievals(
        self,
        limit: int = 20,
        session_id: str | None = None,
    ) -> list[RetrievalLogRecord]:
        with self.database.connect() as connection:
            return RetrievalLogRepository(connection).list_recent(limit=limit, session_id=session_id)

    def get_stats(self) -> DatabaseStats:
        with self.database.connect() as connection:
            return StatsRepository(connection).get_stats()


def _document_id(filename: str, ingestion_timestamp: str) -> str:
    digest = sha256(f"{filename}|{ingestion_timestamp}".encode("utf-8")).hexdigest()[:20]
    return f"doc_{digest}"
