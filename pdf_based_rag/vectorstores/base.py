from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from config import Config
from models import DocumentChunk, RetrievedChunk


MetadataFilter = dict[str, str | int | float | bool]


@dataclass(frozen=True)
class CollectionStats:
    backend: str
    collection_name: str
    record_count: int
    persist_path: str | None = None


class BaseVectorStore(ABC):
    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def index_chunks(self, chunks: Iterable[DocumentChunk]) -> int:
        raise NotImplementedError

    def index_documents(self, chunks: Iterable[str], source: str) -> int:
        document_chunks = (
            DocumentChunk(content=chunk, document_name=source, chunk_index=index)
            for index, chunk in enumerate(chunks)
        )
        return self.index_chunks(document_chunks)

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int,
        filters: MetadataFilter | None = None,
    ) -> list[RetrievedChunk]:
        raise NotImplementedError

    def get_top_k_chunks(self, query: str, k: int = Config.TOP_K) -> list[str]:
        return [chunk.content for chunk in self.similarity_search(query, k=k)]

    @abstractmethod
    def collection_stats(self) -> CollectionStats:
        raise NotImplementedError

    @property
    @abstractmethod
    def records(self) -> list[dict[str, Any]]:
        raise NotImplementedError
