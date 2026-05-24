# PDF-Based RAG System

A lightweight Retrieval-Augmented Generation (RAG) project for asking questions over local PDF, DOCX, and TXT documents.

It includes:

- a CLI flow via `main.py`
- a Streamlit app via `app.py`
- local document chunking and embedding
- a persisted on-disk vector store in `data/vector_store.pkl`
- Groq-based answer generation using retrieved context

## Project Structure

```text
pdf_based_rag/
|- app.py                  # Streamlit entry point
|- main.py                 # CLI entry point
|- answer.py               # compatibility wrapper for question answering
|- ingest.py               # compatibility wrapper for ingestion
|- llm_generator.py        # compatibility wrapper for generation
|- text_processor.py       # current document readers and chunking
|- vector_db.py            # current local vector store implementation
|- embeddings/
|  |- embedding_service.py # reusable SentenceTransformers embedding service
|- vectorstores/
|  |- base.py              # vector store interface
|  |- chroma_store.py      # persistent ChromaDB backend
|  |- legacy_pickle_store.py
|  |- migration.py         # pickle-to-Chroma migration helper
|- database/
|  |- connection.py        # SQLite connection and transaction handling
|  |- models.py            # typed SQL persistence records
|  |- repositories.py      # repository pattern for SQL access
|  |- schema.sql           # normalized operational schema
|  |- services/
|     |- persistence_service.py
|- document_processing/
|  |- extractors.py        # PDF, DOCX, and TXT structure extraction
|  |- chunker.py           # metadata-aware chunking
|  |- pipeline.py          # extraction + chunking orchestration
|- config/
|  |- settings.py          # environment-driven application settings
|- models/
|  |- documents.py         # reusable document and retrieval dataclasses
|- services/
|  |- ingestion_service.py
|  |- retrieval_service.py
|  |- generation_service.py
|  |- rag_service.py
|- utils/
|  |- logging.py
|- tests/
|- documents/
|- data/
|- requirements.txt
|- requirements-dev.txt
|- pyproject.toml
```

## Requirements

- Python `3.11` or `3.12`
- A `GROQ_API_KEY`

Python `3.14` is not recommended for this repo because parts of the ML stack may not have stable wheels there yet. That mismatch is the main reason `uv` was awkward before.

## Quick Start With `uv`

1. Install Python `3.11` if you do not already have it:

```powershell
uv python install 3.11
```

2. Create the virtual environment:

```powershell
uv venv --python 3.11 .venv
```

3. Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
uv pip install -r requirements.txt
```

For local development and tests:

```powershell
uv pip install -r requirements-dev.txt
```

5. Create your environment file:

```powershell
Copy-Item .env.example .env
```

6. Add your API key to `.env`:

```env
GROQ_API_KEY=your_key_here
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

7. Put your files in `documents/`

8. Run the CLI:

```powershell
python main.py
```

9. Or run the Streamlit app:

```powershell
streamlit run app.py
```

## Quick Start With Standard `venv`

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

## How It Works

1. Documents are read from uploads or the local `documents/` folder.
2. PDF, DOCX, and TXT files are extracted into structured sections with document metadata, page references where available, section titles, hierarchy hints, and table detection.
3. Sections are split into overlapping metadata-aware chunks.
4. Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` by default for a lighter and faster local setup.
5. Embeddings, chunk text, and traceable metadata are saved in a persistent ChromaDB collection by default.
6. SQL persistence records documents, chunks, chat sessions, messages, retrieval traces, and feedback-ready records.
7. At query time, ChromaDB retrieves the most relevant chunks with similarity scores and source metadata, then sends them to Groq for answer generation.
8. Retrieval events are logged to SQLite so answers can be audited by query, chunk, source, score, and timestamp.

Phase 1 refactors the original prototype behind service boundaries while preserving the same CLI and Streamlit usage. The current services separate ingestion, retrieval, generation, and RAG orchestration so later phases can replace internals with ChromaDB, SQL persistence, MLflow, and multi-agent workflows without rewriting the entry points.

Phase 2 adds structured document intelligence while preserving the original RAG flow. The app still answers questions over retrieved text, but retrieved chunks now carry source, page, section, table, and hierarchy metadata for citation-ready responses in later phases.

Phase 3 replaces the prototype pickle retrieval path with a vector store abstraction and a persistent ChromaDB backend. The old pickle store remains available as `VECTOR_BACKEND=legacy_pickle`, and the default Chroma backend can auto-migrate existing `data/vector_store.pkl` records into the configured collection.

Phase 4 adds SQL persistence with SQLite. Chroma remains responsible for vector search; SQLite stores operational records for documents, chunks, chat sessions, messages, retrieval logs, and feedback. This keeps retrieval infrastructure independent from conversational traceability.

## Database Configuration

```env
DATABASE_URL=sqlite:///data/rag_platform.db
SQLITE_PATH=data/rag_platform.db
```

The current implementation supports SQLite. The connection and repository boundaries are intentionally separated so a PostgreSQL implementation can be introduced later without changing the RAG services.

## Vector Storage Configuration

```env
VECTOR_BACKEND=chroma
CHROMA_PERSIST_DIR=data/chroma
CHROMA_COLLECTION_NAME=enterprise_rag_documents
AUTO_MIGRATE_LEGACY_VECTORS=true
RETRIEVAL_SCORE_FLOOR=0.0
```

Use `VECTOR_BACKEND=legacy_pickle` only when validating old behavior or debugging migration.

## Notes

- The local vector store is persisted in `data/vector_store.pkl`.
- The CLI resets and rebuilds the index from `documents/` on startup.
- The Streamlit app keeps uploaded files in the vector store until you press `Reset Documents`.
- `documents/` and `data/` are included as folders, but generated data is ignored by git except for `.gitkeep`.

## Common Issues

### `uv` fails or resolves strangely

This repo now declares a supported Python range in `pyproject.toml`. Use Python `3.11` or `3.12`, then recreate the environment.

### Hugging Face model download fails

If the embedding model cannot download, check your proxy environment variables first. On this machine, `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` were set to `http://127.0.0.1:9`, which causes connection failures.

### No answers are generated

Make sure:

- your `.env` file exists
- `GROQ_API_KEY` is set
- you have indexed documents first

## Next Improvements

- add source-aware citations in answers
- deduplicate chunks across repeated uploads
- add tests for ingestion and retrieval
