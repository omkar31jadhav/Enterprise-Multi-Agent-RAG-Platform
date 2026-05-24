from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppSettings:
    base_dir: Path
    documents_dir: Path
    data_dir: Path
    database_url: str
    sqlite_path: Path
    vector_store_path: Path
    vector_backend: str
    chroma_persist_dir: Path
    chroma_collection_name: str
    auto_migrate_legacy_vectors: bool
    embedding_model: str
    groq_api_key: str | None
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    retrieval_score_floor: float
    log_level: str
    mlflow_tracking_uri: str | None
    mlflow_experiment_name: str
    mlflow_enabled: bool


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


def get_settings() -> AppSettings:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = _path_from_env("DATA_DIR", base_dir / "data")

    return AppSettings(
        base_dir=base_dir,
        documents_dir=_path_from_env("DOCUMENTS_DIR", base_dir / "documents"),
        data_dir=data_dir,
        database_url=os.getenv("DATABASE_URL", f"sqlite:///{data_dir / 'rag_platform.db'}"),
        sqlite_path=_path_from_env("SQLITE_PATH", data_dir / "rag_platform.db"),
        vector_store_path=_path_from_env("VECTOR_STORE_PATH", data_dir / "vector_store.pkl"),
        vector_backend=os.getenv("VECTOR_BACKEND", "chroma").lower(),
        chroma_persist_dir=_path_from_env("CHROMA_PERSIST_DIR", data_dir / "chroma"),
        chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "enterprise_rag_documents"),
        auto_migrate_legacy_vectors=os.getenv("AUTO_MIGRATE_LEGACY_VECTORS", "true").lower()
        in {"1", "true", "yes", "y"},
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        llm_model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        top_k=int(os.getenv("TOP_K", "3")),
        retrieval_score_floor=float(os.getenv("RETRIEVAL_SCORE_FLOOR", "0.0")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "pdf_based_rag"),
        mlflow_enabled=os.getenv("MLFLOW_ENABLED", "false").lower() in {"1", "true", "yes", "y"},
    )


class Config:
    """Compatibility settings facade for existing imports.

    New code should prefer get_settings() when dependency injection is useful.
    """

    _settings = get_settings()

    BASE_DIR = _settings.base_dir
    DOCUMENTS_DIR = _settings.documents_dir
    DATA_DIR = _settings.data_dir
    DATABASE_URL = _settings.database_url
    SQLITE_PATH = _settings.sqlite_path
    VECTOR_STORE_PATH = _settings.vector_store_path
    VECTOR_BACKEND = _settings.vector_backend
    CHROMA_PERSIST_DIR = _settings.chroma_persist_dir
    CHROMA_COLLECTION_NAME = _settings.chroma_collection_name
    AUTO_MIGRATE_LEGACY_VECTORS = _settings.auto_migrate_legacy_vectors
    EMBEDDING_MODEL = _settings.embedding_model
    GROQ_API_KEY = _settings.groq_api_key
    LLM_MODEL = _settings.llm_model
    CHUNK_SIZE = _settings.chunk_size
    CHUNK_OVERLAP = _settings.chunk_overlap
    TOP_K = _settings.top_k
    RETRIEVAL_SCORE_FLOOR = _settings.retrieval_score_floor
    LOG_LEVEL = _settings.log_level
    MLFLOW_TRACKING_URI = _settings.mlflow_tracking_uri
    MLFLOW_EXPERIMENT_NAME = _settings.mlflow_experiment_name
    MLFLOW_ENABLED = _settings.mlflow_enabled
