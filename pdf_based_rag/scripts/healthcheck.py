"""Lightweight container/process health check (no external API calls)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    base_dir = Path(__file__).resolve().parents[1]
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    try:
        from config import get_settings
        from config.workflow import WorkflowConfig
        from services.rag_service import RagService
    except Exception as exc:
        print(f"healthcheck: import failed: {exc}")
        return 1

    settings = get_settings()
    data_dir = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(data_dir, os.W_OK):
        print(f"healthcheck: data directory not writable: {data_dir}")
        return 1

    _ = WorkflowConfig.from_env()
    _ = RagService(workflow_config=WorkflowConfig.disabled())

    if not settings.groq_api_key:
        print("healthcheck: warning — GROQ_API_KEY is not set; generation will fail at runtime")

    print("healthcheck: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
