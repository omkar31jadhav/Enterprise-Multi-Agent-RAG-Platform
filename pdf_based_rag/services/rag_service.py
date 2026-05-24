from __future__ import annotations

from database.models import ChatMessageRecord
from database.services import PersistenceService
from services.generation_service import GenerationService
from services.retrieval_service import RetrievalService
from utils import get_logger


logger = get_logger(__name__)


class RagService:
    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        generation_service: GenerationService | None = None,
        persistence_service: PersistenceService | None = None,
    ) -> None:
        self.persistence_service = persistence_service
        if retrieval_service is None:
            self.persistence_service = self.persistence_service or PersistenceService()
            retrieval_service = RetrievalService(persistence_service=self.persistence_service)
        self.retrieval_service = retrieval_service
        self.generation_service = generation_service or GenerationService()

    def answer_question(self, query: str) -> str:
        retrieved_chunks = self.retrieval_service.retrieve(query)
        logger.info("Answering query with %s retrieved chunks", len(retrieved_chunks))
        return self.generation_service.generate_answer(query, retrieved_chunks)

    def answer_chat(
        self,
        query: str,
        session_id: str,
    ) -> tuple[str, ChatMessageRecord, ChatMessageRecord]:
        if self.persistence_service is None:
            self.persistence_service = PersistenceService()
        user_message = self.persistence_service.add_message(
            session_id=session_id,
            role="user",
            content=query,
        )
        retrieved_chunks = self.retrieval_service.retrieve(
            query,
            session_id=session_id,
            message_id=user_message.message_id,
        )
        logger.info("Answering chat query with %s retrieved chunks", len(retrieved_chunks))
        answer = self.generation_service.generate_answer(query, retrieved_chunks)
        assistant_message = self.persistence_service.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
            metadata={
                "retrieved_chunk_ids": [
                    chunk.chunk_id for chunk in retrieved_chunks if chunk.chunk_id
                ]
            },
        )
        return answer, user_message, assistant_message
