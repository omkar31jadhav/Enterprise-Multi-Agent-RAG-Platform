from __future__ import annotations

from typing import Any, Iterable

from models import RetrievedChunk
from services.retrieval_service import RetrievalService
from utils import get_logger

logger = get_logger(__name__)


class LangChainRetrieverAdapter:
    """Adapter that exposes the existing RetrievalService in a LangChain-friendly form."""

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        top_k: int | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service or RetrievalService()
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, str | int | float | bool] | None = None,
        session_id: str | None = None,
        message_id: str | None = None,
    ) -> list[RetrievedChunk]:
        actual_top_k = top_k or self.top_k
        logger.info(
            "LangChainRetrieverAdapter.retrieve query=%s top_k=%s filters=%s session_id=%s",
            query,
            actual_top_k,
            filters,
            session_id,
        )
        return self.retrieval_service.retrieve(
            query=query,
            top_k=actual_top_k,
            filters=filters,
            session_id=session_id,
            message_id=message_id,
        )

    def get_relevant_documents(self, query: str) -> list[str]:
        retrieved = self.retrieve(query)
        return [chunk.content for chunk in retrieved]

    def get_relevant_chunks(self, query: str) -> list[RetrievedChunk]:
        return self.retrieve(query)

    def get_relevant_documents_with_metadata(self, query: str) -> list[dict[str, Any]]:
        retrieved = self.retrieve(query)
        return [
            {
                "content": chunk.content,
                "source": chunk.source,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "chunk_id": chunk.chunk_id,
                "score": chunk.score,
                "metadata": chunk.metadata,
            }
            for chunk in retrieved
        ]
