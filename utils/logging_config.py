"""Structured local logging and crash hardening for HandPilot."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType

from app.config import get_default_config_dir

logger = logging.getLogger(__name__)


def get_default_log_dir() -> Path:
    """Return local directory for persistent logs."""
    return get_default_config_dir() / "logs"


def configure_logging(
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_dir: Path | None = None,
) -> None:
    """Configure root console and rotating file logging.

    Privacy Guarantee:
        Camera frames and biometric landmark coordinates must NEVER be logged.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)

    if log_to_file and not any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        target_dir = Path(log_dir) if log_dir is not None else get_default_log_dir()
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            log_file = target_dir / "handpilot.log"
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=2 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            root_logger.addHandler(file_handler)
            logger.info("Logging initialized (file: %s)", log_file)
        except Exception as exc:
            logger.warning("Failed to initialize rotating file logger: %s", exc)


def install_exception_hook(on_crash_callback: Callable[[], None] | None = None) -> None:
    """Install global sys.excepthook ensuring unhandled crashes are logged and cleaned up safely."""
    orig_excepthook = sys.excepthook

    def _custom_excepthook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            orig_excepthook(exc_type, exc_value, exc_traceback)
            return

        logging.critical(
            "Unhandled top-level application exception:",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        if on_crash_callback is not None:
            try:
                on_crash_callback()
            except Exception:
                logging.exception("Failed during crash recovery callback")

        orig_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = _custom_excepthook

