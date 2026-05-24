from __future__ import annotations

from collections.abc import Iterable

from config import Config
from models import DocumentChunk, ExtractedDocument, ExtractedSection
from utils import get_logger


logger = get_logger(__name__)


def split_text_into_word_chunks(
    text: str,
    chunk_size: int = Config.CHUNK_SIZE,
    overlap: int = Config.CHUNK_OVERLAP,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    words = text.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap

    return chunks


class StructuredChunker:
    def __init__(
        self,
        chunk_size: int = Config.CHUNK_SIZE,
        overlap: int = Config.CHUNK_OVERLAP,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document: ExtractedDocument) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for section_index, section in enumerate(document.sections):
            for content in self._chunk_section(section):
                chunks.append(
                    DocumentChunk(
                        content=content,
                        document_name=document.metadata.filename,
                        page_number=section.page_number,
                        section_title=section.section_title,
                        chunk_index=chunk_index,
                        table_detected=section.table_detected,
                        metadata={
                            **document.metadata.metadata,
                            **section.metadata,
                            "file_type": document.metadata.file_type,
                            "total_pages": document.metadata.total_pages,
                            "ingestion_timestamp": document.metadata.ingestion_timestamp.isoformat(),
                            "section_index": section_index,
                            "hierarchy": list(section.hierarchy),
                        },
                    )
                )
                chunk_index += 1

        logger.info(
            "Chunked document filename=%s sections=%s chunks=%s",
            document.metadata.filename,
            len(document.sections),
            len(chunks),
        )
        return chunks

    def _chunk_section(self, section: ExtractedSection) -> Iterable[str]:
        cleaned_text = " ".join(section.content.split())
        if not cleaned_text:
            return []
        return split_text_into_word_chunks(
            cleaned_text,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )
