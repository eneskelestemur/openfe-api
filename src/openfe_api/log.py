"""Logging configuration for ``openfe_api``.

Named ``log`` rather than ``logging`` so it can never shadow the standard library
module, which breaks as soon as the package directory lands on ``sys.path``.

Library modules only ever call :func:`get_logger`. Configuration is applied once by an
entry point (the CLI or the service) through :func:`configure_logging`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

__all__ = ["configure_logging", "get_logger"]

_ROOT_LOGGER_NAME = "openfe_api"
_FILE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return the logger for a module inside the package.

    Args:
        name: Module name, normally ``__name__``.

    Returns:
        A logger namespaced under ``openfe_api``.
    """
    if name == _ROOT_LOGGER_NAME or name.startswith(f"{_ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def configure_logging(
    level: int | str = logging.INFO,
    log_file: Path | None = None,
    console: Console | None = None,
) -> logging.Logger:
    """Configure package logging for an entry point.

    Replaces any handlers set by an earlier call, so it is safe to call more than once.

    Args:
        level: Minimum level for console output, as a level number or name.
        log_file: Optional file that receives plain-text records at ``DEBUG`` level.
            Parent directories are created if needed.
        console: Rich console to write to. Defaults to a console on ``stderr``, which
            keeps machine-readable command output on ``stdout`` clean.

    Returns:
        The configured ``openfe_api`` logger.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    rich_handler = RichHandler(
        console=console or Console(file=sys.stderr),
        show_path=False,
        rich_tracebacks=True,
        omit_repeated_times=False,
    )
    rich_handler.setLevel(level)
    rich_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    logger.addHandler(rich_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        logger.addHandler(file_handler)

    return logger
