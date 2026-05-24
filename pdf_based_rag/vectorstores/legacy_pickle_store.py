from __future__ import annotations

from collections.abc import Iterable
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import Config
from embeddings import EmbeddingService
from models import DocumentChunk, RetrievedChunk
from utils import get_logger
from vectorstores.base import BaseVectorStore, CollectionStats, MetadataFilter


logger = get_logger(__name__)


class LegacyPickleVectorStore(BaseVectorStore):
    def __init__(
        self,
        store_path: Path | None = None,
        embedding_service: EmbeddingService | None = None,
        score_floor: float = Config.RETRIEVAL_SCORE_FLOOR,
    ) -> None:
        self.store_path = store_path or Config.VECTOR_STORE_PATH
        self.embedding_service = embedding_service or EmbeddingService()
        self.score_floor = score_floor
        self._records: list[dict[str, Any]] = []
        self.load()

    @property
    def records(self) -> list[dict[str, Any]]:
        return self._records

    def clear(self) -> None:
        self._records = []
        if self.store_path.exists():
            self.store_path.unlink()
        logger.info("Cleared legacy pickle vector store path=%s", self.store_path)

    def load(self) -> None:
        if not self.store_path.exists():
            self._records = []
            return

        with self.store_path.open("rb") as file:
            self._records = pickle.load(file)
        logger.info("Loaded legacy pickle vector records path=%s count=%s", self.store_path, len(self._records))

    def save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with self.store_path.open("wb") as file:
            pickle.dump(self._records, file)

    def index_chunks(self, chunks: Iterable[DocumentChunk]) -> int:
        cleaned_chunks = [chunk for chunk in chunks if chunk.content and chunk.content.strip()]
        if not cleaned_chunks:
            return 0

        contents = [chunk.content.strip() for chunk in cleaned_chunks]
        vectors = self.embedding_service.embed_documents(contents)
        for chunk, content, vector in zip(cleaned_chunks, contents, vectors, strict=True):
            self._records.append(
                {
                    "chunk": content,
                    "vector": np.asarray(vector, dtype=np.float32),
                    "source": chunk.document_name,
                    "document_name": chunk.document_name,
                    "chunk_id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "table_detected": chunk.table_detected,
                    "created_at": chunk.created_at.isoformat(),
                    "metadata": dict(chunk.metadata),
                }
            )

        self.save()
        logger.info("Indexed chunks into legacy pickle store path=%s count=%s", self.store_path, len(cleaned_chunks))
        return len(cleaned_chunks)

    def similarity_search(
        self,
        query: str,
        k: int = Config.TOP_K,
        filters: MetadataFilter | None = None,
    ) -> list[RetrievedChunk]:
        matching_records = [record for record in self._records if self._matches_filters(record, filters)]
        if not matching_records:
            return []

        query_vec = np.asarray(self.embedding_service.embed_query(query), dtype=np.float32)
        matrix = np.vstack([record["vector"] for record in matching_records])
        similarities = cosine_similarity([query_vec], matrix)[0]
        result_count = min(k, len(matching_records))
        top_indices = np.argsort(similarities)[-result_count:][::-1]

        results: list[RetrievedChunk] = []
        seen: set[str] = set()
        for index in top_indices:
            record = matching_records[index]
            score = float(similarities[index])
            if score < self.score_floor:
                continue
            chunk_id = str(record.get("chunk_id") or "")
            dedupe_key = chunk_id or f"{record.get('source')}:{record.get('page_number')}:{record.get('chunk')}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            metadata = dict(record.get("metadata") or {})
            if record.get("chunk_index") is not None:
                metadata.setdefault("chunk_index", record["chunk_index"])
            if record.get("created_at") is not None:
                metadata.setdefault("created_at", record["created_at"])
            results.append(
                RetrievedChunk(
                    content=str(record["chunk"]),
                    source=str(record.get("document_name") or record.get("source", "unknown")),
                    score=score,
                    page_number=record.get("page_number"),
                    section_title=record.get("section_title"),
                    chunk_id=chunk_id or None,
                    table_detected=bool(record.get("table_detected", False)),
                    metadata=metadata,
                )
            )

        logger.info("Legacy pickle retrieval requested_k=%s returned=%s filters=%s", k, len(results), filters)
        return results

    def collection_stats(self) -> CollectionStats:
        return CollectionStats(
            backend="legacy_pickle",
            collection_name=self.store_path.name,
            record_count=len(self._records),
            persist_path=str(self.store_path),
        )

    @staticmethod
    def _matches_filters(record: dict[str, Any], filters: MetadataFilter | None) -> bool:
        if not filters:
            return True
        metadata = dict(record.get("metadata") or {})
        metadata.update(
            {
                "document_name": record.get("document_name") or record.get("source"),
                "source": record.get("source"),
                "page_number": record.get("page_number"),
                "section_title": record.get("section_title"),
                "table_detected": record.get("table_detected"),
                "chunk_id": record.get("chunk_id"),
            }
        )
        return all(metadata.get(key) == value for key, value in filters.items())
