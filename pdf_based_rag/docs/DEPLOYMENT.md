# Deployment Guide

Operational readiness for the PDF-based RAG platform. This document does not change orchestration architecture; it describes how to run, verify, and ship the existing service boundaries.

## Deployment architecture

```text
                    ┌─────────────────────────────────────┐
                    │  GitHub Actions (repo root CI)      │
                    │  pytest · mypy · compileall         │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │  pdf_based_rag (application root)   │
                    │  Streamlit app.py / CLI main.py     │
                    │  RagService → classic | workflow    │
                    └─────────────────┬───────────────────┘
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
   SQLite (metadata)           Chroma (vectors)            Groq API (generation)
   DATA_DIR / rag_platform.db  CHROMA_PERSIST_DIR          GROQ_API_KEY
```

- **State**: SQLite + Chroma persist under `DATA_DIR` (mount as a volume in Docker).
- **Documents**: Place files in `DOCUMENTS_DIR` or upload via Streamlit.
- **Secrets**: Only `GROQ_API_KEY` is required for live Q&A; supply via `.env` or orchestrator secrets — never commit `.env`.

## Runtime environment

Copy the sample file and edit locally:

```powershell
Copy-Item .env.example .env
```

| Variable | Purpose | Default |
|----------|---------|---------|
| `GROQ_API_KEY` | Groq chat API for generation | *(empty — required at runtime)* |
| `WORKFLOW_ENABLED` | Use workflow graph instead of classic orchestrator | `false` |
| `WORKFLOW_FALLBACK_ENABLED` | Fall back to classic path on workflow failure | `true` |
| `WORKFLOW_OBSERVABILITY_ENABLED` | Workflow observability hooks | `true` |
| `WORKFLOW_EVALUATION_ENABLED` | Run evaluation node | `true` |
| `DATA_DIR` | SQLite + Chroma persistence root | `data` |
| `CHROMA_PERSIST_DIR` | Chroma persistence path | `data/chroma` |
| `MLFLOW_ENABLED` | Enable MLflow adapter when installed | `false` |

Entry points (`app.py`, `main.py`) load `WorkflowConfig.from_env()`. With defaults in `.env.example`, behavior remains **classic orchestration**.

## Docker

### Build and run (Streamlit UI)

From `pdf_based_rag/`:

```powershell
Copy-Item .env.example .env
# Set GROQ_API_KEY in .env

docker compose up --build
```

Open http://localhost:8501

### Image only

```powershell
docker build -t pdf-based-rag:latest .
docker run --rm -p 8501:8501 --env-file .env -v pdf_rag_data:/app/data -v "${PWD}/documents:/app/documents" pdf-based-rag:latest
```

### Volumes

| Mount | Purpose |
|-------|---------|
| `rag_data` (compose) or `/app/data` | SQLite DB + Chroma index |
| `./documents` | Seed documents for ingestion |

First query may be slow while the embedding model downloads inside the container.

## Health check

The container and Compose service use `scripts/healthcheck.py`:

- Verifies imports and writable `DATA_DIR`
- Does **not** call Groq (no API spend)
- Warns if `GROQ_API_KEY` is unset

Run manually:

```powershell
python scripts/healthcheck.py
echo $LASTEXITCODE
```

Expect exit code `0` when the environment is importable and data directory is writable.

## Startup instructions

### Local development

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
Copy-Item .env.example .env
# Edit .env — set GROQ_API_KEY

python -m pytest tests/ -v
python -m mypy
streamlit run app.py
```

### CLI chat loop

```powershell
python main.py
```

### Ingest documents (before Q&A)

Place PDF/DOCX/TXT under `documents/`, then use Streamlit upload or:

```powershell
python ingest.py
```

## Minimal production notes

1. **Secrets**: Inject `GROQ_API_KEY` via your platform (Kubernetes secret, Compose env, CI vars). Keep `.env` out of git.
2. **Persistence**: Back up `DATA_DIR` (SQLite + Chroma). Without volumes, container removal loses indexed data.
3. **Workflow**: Enable `WORKFLOW_ENABLED=true` only after validating fallback metrics in staging. Keep `WORKFLOW_FALLBACK_ENABLED=true` for safer rollout.
4. **Resources**: Embedding model + Chroma benefit from ≥2 GB RAM; first start downloads model weights.
5. **Scaling**: Single-process Streamlit + SQLite suits demos and small teams. Multi-replica production needs shared vector/DB storage (out of scope for this phase).
6. **CI**: Push to `main`/`master` triggers `.github/workflows/ci.yml` at the repository root with `working-directory: pdf_based_rag`.

## CI commands (mirror GitHub Actions)

```powershell
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests/ -v
python -m mypy
python -m compileall -q -x "\.venv" .
```
