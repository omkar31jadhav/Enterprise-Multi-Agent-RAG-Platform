import pickle

import numpy as np

from models import DocumentChunk
from vectorstores.chroma_store import ChromaVectorStore
from vectorstores.legacy_pickle_store import LegacyPickleVectorStore
from vectorstores.migration import migrate_pickle_to_chroma


class FakeEmbeddingService:
    model_name = "fake-embedding-model"

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, query):
        return self._embed(query)

    @staticmethod
    def _embed(text):
        lower = text.lower()
        if "battery" in lower:
            return [1.0, 0.0, 0.0]
        if "engine" in lower:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def test_chroma_store_persists_metadata_and_retrieves_across_instances(tmp_path) -> None:
    first_store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection_name="test_documents",
        embedding_service=FakeEmbeddingService(),
    )
    first_store.index_chunks(
        [
            DocumentChunk(
                content="battery safety instructions",
                document_name="manual.pdf",
                page_number=4,
                section_title="Battery Safety",
                chunk_index=0,
                table_detected=False,
                metadata={"hierarchy": ["Battery Safety"], "file_type": ".pdf"},
            ),
            DocumentChunk(
                content="engine calibration procedure",
                document_name="engine.pdf",
                page_number=2,
                section_title="Calibration",
                chunk_index=0,
                table_detected=True,
                metadata={"file_type": ".pdf"},
            ),
        ]
    )

    second_store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection_name="test_documents",
        embedding_service=FakeEmbeddingService(),
    )

    results = second_store.similarity_search("battery", k=1)

    assert second_store.collection_stats().record_count == 2
    assert len(results) == 1
    assert results[0].source == "manual.pdf"
    assert results[0].page_number == 4
    assert results[0].section_title == "Battery Safety"
    assert results[0].metadata["file_type"] == ".pdf"


def test_chroma_store_supports_metadata_filtering(tmp_path) -> None:
    store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection_name="filtered_documents",
        embedding_service=FakeEmbeddingService(),
    )
    store.index_chunks(
        [
            DocumentChunk(content="battery safety", document_name="manual.pdf", page_number=1),
            DocumentChunk(content="battery service", document_name="service.pdf", page_number=7),
        ]
    )

    results = store.similarity_search("battery", k=5, filters={"document_name": "service.pdf"})

    assert len(results) == 1
    assert results[0].source == "service.pdf"
    assert results[0].page_number == 7


def test_legacy_pickle_store_persists_and_returns_chunk_metadata(tmp_path) -> None:
    store = LegacyPickleVectorStore(
        store_path=tmp_path / "vectors.pkl",
        embedding_service=FakeEmbeddingService(),
    )
    store.index_chunks(
        [
            DocumentChunk(
                content="battery safety instructions",
                document_name="manual.pdf",
                page_number=4,
                section_title="Battery Safety",
                chunk_index=0,
                metadata={"file_type": ".pdf"},
            )
        ]
    )

    results = store.similarity_search("battery", k=1)

    assert len(results) == 1
    assert results[0].source == "manual.pdf"
    assert results[0].page_number == 4
    assert results[0].section_title == "Battery Safety"
    assert results[0].score == 1.0
    assert results[0].metadata["file_type"] == ".pdf"


def test_pickle_to_chroma_migration_preserves_precomputed_embeddings(tmp_path) -> None:
    pickle_path = tmp_path / "legacy.pkl"
    legacy_records = [
        {
            "chunk": "battery safety instructions",
            "vector": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "source": "legacy_manual.pdf",
            "page_number": 3,
            "section_title": "Legacy Battery Safety",
            "metadata": {"file_type": ".pdf"},
        }
    ]
    with pickle_path.open("wb") as file:
        pickle.dump(legacy_records, file)

    chroma_store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection_name="migrated_documents",
        embedding_service=FakeEmbeddingService(),
    )

    migrated_count = migrate_pickle_to_chroma(pickle_path, chroma_store=chroma_store)
    results = chroma_store.similarity_search("battery", k=1)

    assert migrated_count == 1
    assert chroma_store.collection_stats().record_count == 1
    assert results[0].source == "legacy_manual.pdf"
    assert results[0].page_number == 3
    assert results[0].metadata["legacy_migrated"] is True
