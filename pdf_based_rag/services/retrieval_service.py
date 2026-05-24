from __future__ import annotations

from config import Config
from database.services import PersistenceService
from models import RetrievedChunk
from utils import get_logger
from vector_db import get_vector_store
from vectorstores import BaseVectorStore, MetadataFilter


logger = get_logger(__name__)


class RetrievalService:
    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        top_k: int = Config.TOP_K,
        persistence_service: PersistenceService | None = None,
        persist_retrievals: bool = True,
    ) -> None:
        self.vector_store = vector_store or get_vector_store()
        self.top_k = top_k
        self.persistence_service = persistence_service if persistence_service is not None else (
            PersistenceService() if persist_retrievals else None
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: MetadataFilter | None = None,
        session_id: str | None = None,
        message_id: str | None = None,
        log_retrieval: bool = True,
    ) -> list[RetrievedChunk]:
        k = top_k or self.top_k
        results = self.vector_store.similarity_search(query, k=k, filters=filters)
        if log_retrieval and self.persistence_service is not None:
            self.persistence_service.log_retrievals(
                query=query,
                chunks=results,
                session_id=session_id,
                message_id=message_id,
            )
        logger.info("Retrieved %s chunks for query filters=%s", len(results), filters)
        return results

    def retrieve_texts(
        self,
        query: str,
        top_k: int | None = None,
        filters: MetadataFilter | None = None,
    ) -> list[str]:
        return [
            chunk.content
            for chunk in self.retrieve(
                query,
                top_k=top_k,
                filters=filters,
                log_retrieval=False,
            )
        ]
