from models import RetrievedChunk

from rag.citations import CitationReference, extract_citation_references, format_citation_references


def test_citation_reference_formats_full_reference() -> None:
    citation = CitationReference(
        source="policy.pdf",
        page_number=5,
        section_title="Executive Summary",
        chunk_id="abc123",
        score=0.9123,
    )

    formatted = citation.format()

    assert "Source: policy.pdf" in formatted
    assert "Page: 5" in formatted
    assert "Section: Executive Summary" in formatted
    assert "Chunk ID: abc123" in formatted
    assert "Score: 0.9123" in formatted


def test_extract_citation_references_from_retrieved_chunks() -> None:
    chunks = [
        RetrievedChunk(
            content="Example content.",
            source="policy.pdf",
            score=0.75,
            page_number=2,
            section_title="Scope",
            chunk_id="chunk-001",
        )
    ]

    references = extract_citation_references(chunks)

    assert len(references) == 1
    assert references[0].source == "policy.pdf"
    assert references[0].page_number == 2
    assert references[0].section_title == "Scope"
    assert references[0].chunk_id == "chunk-001"


def test_format_citation_references_returns_multiline_block() -> None:
    references = [
        CitationReference(source="policy.pdf", page_number=2, section_title="Scope"),
        CitationReference(source="guide.docx", page_number=1),
    ]

    result = format_citation_references(references)

    assert "Source: policy.pdf" in result
    assert "Section: Scope" in result
    assert "Source: guide.docx" in result
    assert result.count("Source:") == 2
