from __future__ import annotations

import time

from database.models import ChatMessageRecord
from database.services import PersistenceService
from rag.orchestration import RagOrchestrator
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
        orchestrator: RagOrchestrator | None = None,
    ) -> None:
        self.persistence_service = persistence_service
        if retrieval_service is None:
            self.persistence_service = self.persistence_service or PersistenceService()
            retrieval_service = RetrievalService(persistence_service=self.persistence_service)
        self.retrieval_service = retrieval_service
        self.generation_service = generation_service or GenerationService()
        self.orchestrator = orchestrator or RagOrchestrator()

    def answer_question(self, query: str) -> str:
        retrieval_start = time.perf_counter()
        retrieved_chunks = self.retrieval_service.retrieve(query)
        retrieval_time = time.perf_counter() - retrieval_start
        logger.info(
            "Answering query with %s retrieved chunks retrieval_time=%.4f",
            len(retrieved_chunks),
            retrieval_time,
        )

        orchestration_result = self.orchestrator.run(
            query=query,
            retrieved_chunks=retrieved_chunks,
            session_id=None,
            message_id=None,
            generation_function=self._generate_from_prompt,
        )
        logger.info(
            "Orchestration completed for query=%s prompt_name=%s citations=%s prompt_time=%.4f generation_time=%.4f total_time=%.4f",
            query,
            orchestration_result.metrics.prompt_name,
            orchestration_result.metrics.citation_count,
            orchestration_result.metrics.prompt_assembly_time,
            orchestration_result.metrics.generation_time,
            orchestration_result.metrics.total_time,
        )
        return orchestration_result.answer

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
        retrieval_start = time.perf_counter()
        retrieved_chunks = self.retrieval_service.retrieve(
            query,
            session_id=session_id,
            message_id=user_message.message_id,
        )
        retrieval_time = time.perf_counter() - retrieval_start
        logger.info(
            "Answering chat query with %s retrieved chunks retrieval_time=%.4f session_id=%s",
            len(retrieved_chunks),
            retrieval_time,
            session_id,
        )

        orchestration_result = self.orchestrator.run(
            query=query,
            retrieved_chunks=retrieved_chunks,
            session_id=session_id,
            message_id=user_message.message_id,
            generation_function=self._generate_from_prompt,
        )

        answer = orchestration_result.answer
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

        logger.info(
            "Orchestration completed for chat session_id=%s message_id=%s prompt_name=%s citations=%s total_time=%.4f",
            session_id,
            user_message.message_id,
            orchestration_result.metrics.prompt_name,
            orchestration_result.metrics.citation_count,
            orchestration_result.metrics.total_time,
        )
        return answer, user_message, assistant_message

    def _generate_from_prompt(self, prompt: str) -> str:
        return self.generation_service.generate_from_prompt(prompt)
