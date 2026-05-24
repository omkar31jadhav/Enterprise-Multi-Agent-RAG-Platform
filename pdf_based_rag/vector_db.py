from __future__ import annotations

from config import Config
from utils import get_logger
from vectorstores import BaseVectorStore, ChromaVectorStore, LegacyPickleVectorStore
from vectorstores.migration import migrate_pickle_to_chroma


logger = get_logger(__name__)

# Compatibility alias: new code should type against BaseVectorStore.
VectorStore = ChromaVectorStore

_VECTOR_STORE: BaseVectorStore | None = None


def get_vector_store() -> BaseVectorStore:
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = _create_vector_store()
    return _VECTOR_STORE


def _create_vector_store() -> BaseVectorStore:
    if Config.VECTOR_BACKEND == "legacy_pickle":
        logger.info("Using legacy pickle vector backend path=%s", Config.VECTOR_STORE_PATH)
        return LegacyPickleVectorStore(store_path=Config.VECTOR_STORE_PATH)

    if Config.VECTOR_BACKEND != "chroma":
        raise ValueError(f"Unsupported VECTOR_BACKEND: {Config.VECTOR_BACKEND}")

    store = ChromaVectorStore(
        persist_dir=Config.CHROMA_PERSIST_DIR,
        collection_name=Config.CHROMA_COLLECTION_NAME,
    )
    if (
        Config.AUTO_MIGRATE_LEGACY_VECTORS
        and store.collection_stats().record_count == 0
        and Config.VECTOR_STORE_PATH.exists()
    ):
        migrated_count = migrate_pickle_to_chroma(Config.VECTOR_STORE_PATH, chroma_store=store)
        logger.info("Auto-migrated legacy vectors into Chroma count=%s", migrated_count)
    return store


def reset_vector_store_singleton() -> None:
    global _VECTOR_STORE
    _VECTOR_STORE = None
