from __future__ import annotations

import time

from database.models import ChatMessageRecord
from database.services import PersistenceService
from ml.tracking import NoOpTrackingAdapter, TrackingAdapter
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
        tracking_adapter: TrackingAdapter | None = None,
    ) -> None:
        self.persistence_service = persistence_service
        if retrieval_service is None:
            self.persistence_service = self.persistence_service or PersistenceService()
            retrieval_service = RetrievalService(persistence_service=self.persistence_service)
        self.retrieval_service = retrieval_service
        self.generation_service = generation_service or GenerationService()
        self.tracking_adapter = tracking_adapter or NoOpTrackingAdapter()
        self.orchestrator = orchestrator or RagOrchestrator(tracking_adapter=self.tracking_adapter)

    def answer_question(self, query: str) -> str:
        retrieval_start = time.perf_counter()
        retrieved_chunks = self.retrieval_service.retrieve(query)
        retrieval_time = time.perf_counter() - retrieval_start
        logger.info(
            "Answering query with %s retrieved chunks retrieval_time=%.4f",
            len(retrieved_chunks),
            retrieval_time,
        )
        self.tracking_adapter.log_metrics({"retrieval_time": retrieval_time})
        self.tracking_adapter.log_params(
            {
                "query": query,
                "execution_path": "answer_question",
                "session_id": "none",
            }
        )

        orchestration_result = self.orchestrator.run(
            query=query,
            retrieved_chunks=retrieved_chunks,
            session_id=None,
            message_id=None,
            generation_function=self._generate_from_prompt,
        )
        self.tracking_adapter.log_params(
            {
                "prompt_name": orchestration_result.metrics.prompt_name,
                "prompt_version": orchestration_result.metrics.prompt_version,
                "prompt_template_used": orchestration_result.metrics.prompt_template_used,
                "orchestration_path": "answer_question",
            }
        )
        self.tracking_adapter.log_metrics(
            {
                "generation_time": orchestration_result.metrics.generation_time,
                "orchestration_total_time": orchestration_result.metrics.total_time,
                "retrieved_chunk_count": orchestration_result.metrics.retrieved_chunk_count,
                "citation_count": orchestration_result.metrics.citation_count,
                "history_message_count": orchestration_result.metrics.history_count,
                "context_size_estimate": orchestration_result.metrics.context_size_estimate,
            }
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
        self.tracking_adapter.log_metrics({"retrieval_time": retrieval_time})
        self.tracking_adapter.log_params(
            {
                "query": query,
                "execution_path": "answer_chat",
                "session_id": session_id,
                "message_id": user_message.message_id,
            }
        )

        orchestration_result = self.orchestrator.run(
            query=query,
            retrieved_chunks=retrieved_chunks,
            session_id=session_id,
            message_id=user_message.message_id,
            generation_function=self._generate_from_prompt,
        )

        answer = orchestration_result.answer
        self.tracking_adapter.log_params(
            {
                "prompt_name": orchestration_result.metrics.prompt_name,
                "prompt_version": orchestration_result.metrics.prompt_version,
                "prompt_template_used": orchestration_result.metrics.prompt_template_used,
                "orchestration_path": "answer_chat",
            }
        )
        self.tracking_adapter.log_metrics(
            {
                "generation_time": orchestration_result.metrics.generation_time,
                "orchestration_total_time": orchestration_result.metrics.total_time,
                "retrieved_chunk_count": orchestration_result.metrics.retrieved_chunk_count,
                "citation_count": orchestration_result.metrics.citation_count,
                "history_message_count": orchestration_result.metrics.history_count,
                "context_size_estimate": orchestration_result.metrics.context_size_estimate,
            }
        )
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
