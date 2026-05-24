from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

from chromadb import PersistentClient

from config import Config
from embeddings import EmbeddingService
from models import DocumentChunk, RetrievedChunk
from utils import get_logger
from vectorstores.base import BaseVectorStore, CollectionStats, MetadataFilter


logger = get_logger(__name__)


class ChromaVectorStore(BaseVectorStore):
    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str = Config.CHROMA_COLLECTION_NAME,
        embedding_service: EmbeddingService | None = None,
        score_floor: float = Config.RETRIEVAL_SCORE_FLOOR,
    ) -> None:
        self.persist_dir = Path(persist_dir or Config.CHROMA_PERSIST_DIR)
        self.collection_name = collection_name
        self.embedding_service = embedding_service or EmbeddingService()
        self.score_floor = score_floor
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"embedding_model": self.embedding_service.model_name},
        )
        logger.info(
            "Initialized Chroma collection name=%s path=%s count=%s",
            self.collection_name,
            self.persist_dir,
            self.collection.count(),
        )

    @property
    def records(self) -> list[dict[str, Any]]:
        result = self.collection.get(include=["documents", "metadatas"])
        return [
            {
                "chunk": document,
                "metadata": metadata or {},
                "source": (metadata or {}).get("document_name", "unknown"),
            }
            for document, metadata in zip(
                result.get("documents") or [],
                result.get("metadatas") or [],
                strict=False,
            )
        ]

    def clear(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            logger.debug("Chroma collection did not exist during clear: %s", self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"embedding_model": self.embedding_service.model_name},
        )
        logger.info("Cleared Chroma collection name=%s", self.collection_name)

    def index_chunks(self, chunks: Iterable[DocumentChunk]) -> int:
        cleaned_chunks = [chunk for chunk in chunks if chunk.content and chunk.content.strip()]
        if not cleaned_chunks:
            return 0

        ids = [str(chunk.id) for chunk in cleaned_chunks]
        documents = [chunk.content.strip() for chunk in cleaned_chunks]
        metadatas = [self._metadata_from_chunk(chunk) for chunk in cleaned_chunks]
        embeddings = self.embedding_service.embed_documents(documents)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info(
            "Indexed chunks into Chroma collection=%s count=%s total=%s",
            self.collection_name,
            len(cleaned_chunks),
            self.collection.count(),
        )
        return len(cleaned_chunks)

    def add_preembedded_records(self, records: Iterable[dict[str, Any]]) -> int:
        normalized_records = [record for record in records if record.get("chunk") and record.get("vector") is not None]
        if not normalized_records:
            return 0

        ids = [str(record.get("chunk_id") or self._legacy_record_id(record, index)) for index, record in enumerate(normalized_records)]
        documents = [str(record["chunk"]).strip() for record in normalized_records]
        metadatas = [self._metadata_from_record(record) for record in normalized_records]
        embeddings = [self._vector_to_list(record["vector"]) for record in normalized_records]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info(
            "Migrated pre-embedded records into Chroma collection=%s count=%s total=%s",
            self.collection_name,
            len(normalized_records),
            self.collection.count(),
        )
        return len(normalized_records)

    def similarity_search(
        self,
        query: str,
        k: int = Config.TOP_K,
        filters: MetadataFilter | None = None,
    ) -> list[RetrievedChunk]:
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedding_service.embed_query(query)
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, self.collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            query_kwargs["where"] = filters

        result = self.collection.query(**query_kwargs)
        retrieved = self._result_to_chunks(result)
        deduped = self._deduplicate(retrieved)
        filtered = [chunk for chunk in deduped if chunk.score is None or chunk.score >= self.score_floor]
        logger.info(
            "Chroma retrieval collection=%s requested_k=%s returned=%s filters=%s",
            self.collection_name,
            k,
            len(filtered),
            filters,
        )
        return filtered

    def collection_stats(self) -> CollectionStats:
        return CollectionStats(
            backend="chroma",
            collection_name=self.collection_name,
            record_count=self.collection.count(),
            persist_path=str(self.persist_dir),
        )

    @staticmethod
    def _metadata_from_chunk(chunk: DocumentChunk) -> dict[str, Any]:
        metadata = _flatten_metadata(chunk.metadata)
        metadata.update(
            {
                "chunk_id": chunk.id or "",
                "document_name": chunk.document_name,
                "source": chunk.source or chunk.document_name,
                "chunk_index": chunk.chunk_index if chunk.chunk_index is not None else -1,
                "page_number": chunk.page_number if chunk.page_number is not None else -1,
                "section_title": chunk.section_title or "",
                "table_detected": bool(chunk.table_detected),
                "created_at": chunk.created_at.isoformat(),
            }
        )
        return metadata

    @staticmethod
    def _metadata_from_record(record: dict[str, Any]) -> dict[str, Any]:
        metadata = _flatten_metadata(record.get("metadata") or {})
        document_name = str(record.get("document_name") or record.get("source") or "unknown")
        metadata.update(
            {
                "chunk_id": str(record.get("chunk_id") or ""),
                "document_name": document_name,
                "source": document_name,
                "chunk_index": int(record.get("chunk_index") if record.get("chunk_index") is not None else -1),
                "page_number": int(record.get("page_number") if record.get("page_number") is not None else -1),
                "section_title": str(record.get("section_title") or ""),
                "table_detected": bool(record.get("table_detected", False)),
                "created_at": str(record.get("created_at") or ""),
                "legacy_migrated": True,
            }
        )
        return metadata

    @staticmethod
    def _legacy_record_id(record: dict[str, Any], index: int) -> str:
        source = str(record.get("document_name") or record.get("source") or "legacy")
        chunk = str(record.get("chunk") or "")
        digest = sha256(f"{source}|{index}|{chunk}".encode("utf-8")).hexdigest()[:24]
        return f"legacy-{digest}"

    @staticmethod
    def _vector_to_list(vector: Any) -> list[float]:
        if hasattr(vector, "tolist"):
            return [float(value) for value in vector.tolist()]
        return [float(value) for value in vector]

    @staticmethod
    def _deduplicate(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        seen: set[str] = set()
        deduped: list[RetrievedChunk] = []
        for chunk in chunks:
            key = chunk.chunk_id or f"{chunk.source}:{chunk.page_number}:{chunk.content}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(chunk)
        return deduped

    @staticmethod
    def _result_to_chunks(result: dict[str, Any]) -> list[RetrievedChunk]:
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances, strict=False):
            clean_metadata = dict(metadata or {})
            page_number = clean_metadata.get("page_number")
            if page_number == -1:
                page_number = None
            score = _distance_to_score(distance)
            chunks.append(
                RetrievedChunk(
                    content=str(document),
                    source=str(clean_metadata.get("document_name") or clean_metadata.get("source") or "unknown"),
                    score=score,
                    page_number=page_number,
                    section_title=str(clean_metadata.get("section_title") or "") or None,
                    chunk_id=str(clean_metadata.get("chunk_id") or "") or None,
                    table_detected=bool(clean_metadata.get("table_detected", False)),
                    metadata=clean_metadata,
                )
            )
        return chunks


def _distance_to_score(distance: float | int | None) -> float | None:
    if distance is None:
        return None
    # Chroma returns distance, where smaller is better. This bounded transform
    # gives the UI and logs a stable similarity-style confidence value.
    return 1.0 / (1.0 + max(float(distance), 0.0))


def _flatten_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    flattened: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            flattened[key] = value
        elif isinstance(value, (list, tuple)):
            flattened[key] = " > ".join(str(item) for item in value)
        else:
            flattened[key] = str(value)
    return flattened
