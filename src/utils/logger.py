"""Shared logger for the ML pipeline."""

import logging
from typing import Any, cast

import structlog
from structlog.stdlib import BoundLogger


def setup_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure structured logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_output: If True, output JSON logs; otherwise, human-readable.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            (
                structlog.processors.JSONRenderer()
                if json_output
                else structlog.dev.ConsoleRenderer()
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str, **kwargs: Any) -> BoundLogger:
    """Get a bound logger instance.

    Args:
        name: Logger name (typically module name).
        **kwargs: Additional context to bind to the logger.

    Returns:
        A bound logger instance.
    """
    return cast(BoundLogger, structlog.get_logger(name, **kwargs))
