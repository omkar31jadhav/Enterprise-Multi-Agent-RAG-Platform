from pathlib import Path

from document_processing.chunker import StructuredChunker
from document_processing.extractors import TxtExtractor
from models import DocumentMetadata, ExtractedDocument, ExtractedSection


def test_txt_extractor_preserves_section_metadata(tmp_path: Path) -> None:
    document = tmp_path / "manual.txt"
    document.write_text(
        "Overview\n"
        "This section explains the document.\n"
        "\n"
        "2. Details\n"
        "The details section has more content.\n",
        encoding="utf-8",
    )

    extracted = TxtExtractor().extract(document)

    assert extracted.metadata.filename == "manual.txt"
    assert extracted.metadata.file_type == ".txt"
    assert extracted.metadata.total_pages == 1
    assert [section.section_title for section in extracted.sections] == ["Overview", "2. Details"]
    assert all(section.page_number == 1 for section in extracted.sections)


def test_structured_chunker_preserves_page_section_and_table_metadata() -> None:
    extracted = ExtractedDocument(
        metadata=DocumentMetadata(filename="report.pdf", file_type=".pdf", total_pages=3),
        sections=[
            ExtractedSection(
                content="alpha beta gamma delta epsilon",
                document_name="report.pdf",
                page_number=2,
                section_title="Safety Summary",
                table_detected=True,
                hierarchy=("Safety Summary",),
                metadata={"extraction_backend": "test"},
            )
        ],
    )

    chunks = StructuredChunker(chunk_size=3, overlap=1).chunk_document(extracted)

    assert [chunk.content for chunk in chunks] == ["alpha beta gamma", "gamma delta epsilon", "epsilon"]
    assert {chunk.document_name for chunk in chunks} == {"report.pdf"}
    assert {chunk.page_number for chunk in chunks} == {2}
    assert {chunk.section_title for chunk in chunks} == {"Safety Summary"}
    assert all(chunk.table_detected for chunk in chunks)
    assert chunks[0].metadata["file_type"] == ".pdf"
    assert chunks[0].metadata["total_pages"] == 3
    assert chunks[0].metadata["hierarchy"] == ["Safety Summary"]
