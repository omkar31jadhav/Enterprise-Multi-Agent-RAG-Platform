import os

import fitz
from docx import Document

from config import Config
from document_processing import process_document_structured, split_text_into_word_chunks
from models import DocumentChunk


def read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def read_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def read_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs)


def split_into_chunks(
    text: str, chunk_size: int = Config.CHUNK_SIZE, overlap: int = Config.CHUNK_OVERLAP
) -> list[str]:
    return split_text_into_word_chunks(text, chunk_size=chunk_size, overlap=overlap)


def process_document(file_path: str) -> list[str]:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        raw_text = read_txt(file_path)
    elif ext == ".pdf":
        raw_text = read_pdf(file_path)
    elif ext == ".docx":
        raw_text = read_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    cleaned_text = " ".join(raw_text.split())
    if not cleaned_text:
        return []

    return split_into_chunks(cleaned_text)


def process_document_with_metadata(file_path: str, document_name: str | None = None) -> list[DocumentChunk]:
    return process_document_structured(file_path, document_name=document_name)
