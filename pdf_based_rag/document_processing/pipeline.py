from __future__ import annotations

from pathlib import Path

from config import Config
from document_processing.chunker import StructuredChunker
from document_processing.extractors import get_extractor
from models import DocumentChunk, ExtractedDocument
from utils import get_logger


logger = get_logger(__name__)


class DocumentProcessingPipeline:
    def __init__(self, chunker: StructuredChunker | None = None) -> None:
        self.chunker = chunker or StructuredChunker(
            chunk_size=Config.CHUNK_SIZE,
            overlap=Config.CHUNK_OVERLAP,
        )

    def extract(self, file_path: str | Path, document_name: str | None = None) -> ExtractedDocument:
        extractor = get_extractor(file_path)
        return extractor.extract(file_path, document_name=document_name)

    def process(self, file_path: str | Path, document_name: str | None = None) -> list[DocumentChunk]:
        document = self.extract(file_path, document_name=document_name)
        chunks = self.chunker.chunk_document(document)
        logger.info(
            "Processed document filename=%s chunks=%s",
            document.metadata.filename,
            len(chunks),
        )
        return chunks


def process_document_structured(
    file_path: str | Path,
    document_name: str | None = None,
) -> list[DocumentChunk]:
    return DocumentProcessingPipeline().process(file_path, document_name=document_name)
