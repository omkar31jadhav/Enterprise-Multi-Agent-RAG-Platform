from __future__ import annotations

from pathlib import Path

from config import Config
from utils import get_logger
from vectorstores.chroma_store import ChromaVectorStore
from vectorstores.legacy_pickle_store import LegacyPickleVectorStore


logger = get_logger(__name__)


def migrate_pickle_to_chroma(
    pickle_path: str | Path = Config.VECTOR_STORE_PATH,
    chroma_store: ChromaVectorStore | None = None,
) -> int:
    legacy_store = LegacyPickleVectorStore(store_path=Path(pickle_path))
    if not legacy_store.records:
        logger.info("No legacy pickle records found for migration path=%s", pickle_path)
        return 0

    target_store = chroma_store or ChromaVectorStore()
    migrated_count = target_store.add_preembedded_records(legacy_store.records)
    logger.info(
        "Completed pickle to Chroma migration source=%s target_collection=%s count=%s",
        pickle_path,
        target_store.collection_name,
        migrated_count,
    )
    return migrated_count
