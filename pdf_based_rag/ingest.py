from pathlib import Path

from services.ingestion_service import IngestionService, SUPPORTED_FORMATS


def ingest_documents(doc_folder: str | Path | None = None, reset: bool = False) -> int:
    return IngestionService().ingest_folder(doc_folder=doc_folder, reset=reset)
