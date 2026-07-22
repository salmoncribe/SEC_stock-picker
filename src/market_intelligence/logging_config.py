"""Structured, JSON-compatible logging via structlog.

``configure_logging`` wires structlog through the stdlib logging root so that
both our loggers and third-party libraries emit the same structured records.
Output goes to stderr and, optionally, a rotating-free append log file.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog

_configured = False


def configure_logging(
    level: str = "INFO",
    *,
    log_file: str | Path | None = None,
    json_logs: bool = True,
    force: bool = False,
) -> None:
    """Configure process-wide structured logging (idempotent unless ``force``)."""
    global _configured
    if _configured and not force:
        return

    numeric_level = getattr(logging, str(level).upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        cache_logger_on_first_use=True,
    )

    renderer: Any = (
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.setLevel(numeric_level)
    _configured = True


def get_logger(name: str | None = None) -> Any:
    """Return a structlog logger. Safe to call before ``configure_logging``."""
    return structlog.get_logger(name)


def reset_logging() -> None:
    """Testing hook: allow reconfiguration."""
    global _configured
    _configured = False
