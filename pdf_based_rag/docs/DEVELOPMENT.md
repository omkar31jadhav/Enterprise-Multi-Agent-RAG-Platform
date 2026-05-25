# Developer Onboarding

## Prerequisites

- Python **3.11** (see `pyproject.toml`: `>=3.11,<3.13`)
- Git
- Optional: Docker for containerized runs

## First-time setup

```powershell
cd pdf_based_rag
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
Copy-Item .env.example .env
```

Set `GROQ_API_KEY` in `.env` before testing live generation.

## Daily commands

| Task | Command |
|------|---------|
| Run tests | `python -m pytest tests/ -v` |
| Type check | `python -m mypy` |
| Compile check | `python -m compileall -q -x ".venv" .` |
| Streamlit UI | `streamlit run app.py` |
| CLI chat | `python main.py` |
| Health check | `python scripts/healthcheck.py` |

Windows Make targets (requires `.venv`):

```powershell
make setup
make test
make mypy
```

## Project layout (where to change what)

| Area | Path | Notes |
|------|------|-------|
| Public RAG API | `services/rag_service.py` | Classic vs workflow routing |
| Classic orchestration | `rag/orchestration.py` | Do not bypass for workflow features |
| Workflow graph | `workflow/` | Additive nodes only |
| Runtime toggles | `config/workflow.py` | `WORKFLOW_*` env vars |
| App settings | `config/settings.py` | Paths, models, MLflow |
| Tests | `tests/` | Run before every PR |

## Pull request checklist

- [ ] `pytest`, `mypy`, and `compileall` pass locally
- [ ] No secrets or `.env` in the diff
- [ ] README / `docs/` updated if env or deploy steps change
- [ ] Orchestration behavior unchanged unless explicitly scoped

## CI

GitHub Actions runs at the monorepo root (`.github/workflows/ci.yml`) with `working-directory: pdf_based_rag`. Path filters limit runs to changes under `pdf_based_rag/` or workflow files.
