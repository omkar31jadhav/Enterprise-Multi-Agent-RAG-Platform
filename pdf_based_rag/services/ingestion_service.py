from __future__ import annotations

from pathlib import Path

from config import Config
from database.services import PersistenceService
from text_processor import process_document_with_metadata
from utils import get_logger
from vector_db import get_vector_store
from vectorstores import BaseVectorStore


logger = get_logger(__name__)

SUPPORTED_FORMATS = (".txt", ".pdf", ".docx")


class IngestionService:
    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        persistence_service: PersistenceService | None = None,
        persist_metadata: bool = True,
    ) -> None:
        self.vector_store = vector_store or get_vector_store()
        self.persistence_service = persistence_service if persistence_service is not None else (
            PersistenceService() if persist_metadata else None
        )

    def ingest_folder(self, doc_folder: str | Path | None = None, reset: bool = False) -> int:
        folder = Path(doc_folder) if doc_folder else Config.DOCUMENTS_DIR
        if not folder.exists():
            raise FileNotFoundError(
                f"Documents folder not found: {folder}. Create it and add PDF, DOCX, or TXT files."
            )

        if reset:
            self.vector_store.clear()

        total_chunks = 0
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in SUPPORTED_FORMATS:
                continue

            total_chunks += self.ingest_file(path)

        logger.info("Completed folder ingestion folder=%s chunks=%s", folder, total_chunks)
        return total_chunks

    def ingest_file(self, file_path: str | Path, source: str | None = None) -> int:
        path = Path(file_path)
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        logger.info("Ingesting document source=%s path=%s", source or path.name, path)
        document_name = source or path.name
        chunks = process_document_with_metadata(str(path), document_name=document_name)
        indexed_count = self.vector_store.index_chunks(chunks)
        if self.persistence_service is not None:
            total_pages = _total_pages_from_chunks(chunks)
            self.persistence_service.record_ingestion(
                filename=document_name,
                file_type=path.suffix.lower(),
                chunks=chunks,
                total_pages=total_pages,
                metadata={"source_path": str(path)},
            )
        return indexed_count


def _total_pages_from_chunks(chunks) -> int | None:
    for chunk in chunks:
        total_pages = chunk.metadata.get("total_pages")
        if isinstance(total_pages, int):
            return total_pages
    return None
