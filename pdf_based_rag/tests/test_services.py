from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from models import DocumentChunk, RetrievedChunk
from services.ingestion_service import IngestionService
from services.rag_service import RagService
from services.retrieval_service import RetrievalService


class FakeVectorStore:
    def __init__(self) -> None:
        self.indexed_sources: list[str] = []
        self.indexed_chunks: list[DocumentChunk] = []

    def index_documents(self, chunks: Iterable[str], source: str) -> int:
        chunk_list = list(chunks)
        self.indexed_sources.append(source)
        return len(chunk_list)

    def index_chunks(self, chunks: Iterable[DocumentChunk]) -> int:
        chunk_list = list(chunks)
        self.indexed_chunks.extend(chunk_list)
        self.indexed_sources.extend(chunk.document_name for chunk in chunk_list)
        return len(chunk_list)

    def similarity_search(self, query: str, k: int, filters: Any = None) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                content=f"Context for {query}",
                source="test.txt",
                score=0.9,
            )
        ][:k]


class FakeRetrievalService:
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: Any = None,
    ) -> list[RetrievedChunk]:
        return [RetrievedChunk(content=f"Retrieved: {query}", source="test.txt", score=0.8)]


class FakeGenerationService:
    def __init__(self) -> None:
        self.seen_prompt: str | None = None

    def generate_from_prompt(self, prompt: str) -> str:
        self.seen_prompt = prompt
        return "answer"


def test_ingestion_service_indexes_text_file(tmp_path: Path) -> None:
    document = tmp_path / "sample.txt"
    document.write_text("alpha beta gamma", encoding="utf-8")
    vector_store = FakeVectorStore()

    indexed_count = IngestionService(
        vector_store=vector_store,
        persist_metadata=False,
    ).ingest_file(document)

    assert indexed_count == 1
    assert vector_store.indexed_sources == ["sample.txt"]
    assert vector_store.indexed_chunks[0].document_name == "sample.txt"
    assert vector_store.indexed_chunks[0].page_number == 1


def test_retrieval_service_returns_typed_chunks() -> None:
    vector_store = FakeVectorStore()

    results = RetrievalService(
        vector_store=vector_store,
        top_k=1,
        persist_retrievals=False,
    ).retrieve("query")

    assert results == [RetrievedChunk(content="Context for query", source="test.txt", score=0.9)]


def test_rag_service_orchestrates_retrieval_and_generation() -> None:
    generation_service = FakeGenerationService()
    rag_service = RagService(
        retrieval_service=FakeRetrievalService(),
        generation_service=generation_service,
        persistence_service=None,
    )

    answer = rag_service.answer_question("What is covered?")

    assert answer.startswith("answer")
    assert "References:" in answer
    assert generation_service.seen_prompt is not None
    assert "What is covered?" in generation_service.seen_prompt
