from __future__ import annotations

import logging
import sys

from config import Config


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=(level or Config.LOG_LEVEL).upper(),
        format=LOG_FORMAT,
        stream=sys.stdout,
        force=False,
    )


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
