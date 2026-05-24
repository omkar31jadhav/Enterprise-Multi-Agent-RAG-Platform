from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator
from urllib.parse import unquote, urlparse

from config import Config
from utils import get_logger


logger = get_logger(__name__)


class Database:
    def __init__(self, database_url: str | None = None, sqlite_path: str | Path | None = None) -> None:
        self.database_url = database_url or Config.DATABASE_URL
        if sqlite_path is None and database_url is None:
            sqlite_path = Config.SQLITE_PATH
        self.sqlite_path = self._resolve_sqlite_path(database_url=self.database_url, sqlite_path=sqlite_path)

    def connect(self) -> sqlite3.Connection:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.sqlite_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        with self.connect() as connection:
            connection.executescript(schema_path.read_text(encoding="utf-8"))
        logger.info("Initialized SQLite database path=%s", self.sqlite_path)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _resolve_sqlite_path(database_url: str, sqlite_path: str | Path | None) -> Path:
        if sqlite_path is not None:
            return Path(sqlite_path).expanduser()

        parsed = urlparse(database_url)
        if parsed.scheme in ("", "sqlite"):
            if parsed.scheme == "":
                return Path(database_url).expanduser()
            if parsed.netloc and parsed.netloc != "":
                raise ValueError("Only local SQLite database URLs are supported in Phase 4.")
            path = unquote(parsed.path)
            if path.startswith("/") and len(path) > 3 and path[2] == ":":
                path = path[1:]
            return Path(path).expanduser()

        raise ValueError(
            f"Unsupported DATABASE_URL scheme '{parsed.scheme}'. Phase 4 implements SQLite only."
        )
