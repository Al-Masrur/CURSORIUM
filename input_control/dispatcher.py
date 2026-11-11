"""Serialized OS input dispatcher with safety gate and guaranteed release."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from input_control.backend import InputBackend

logger = logging.getLogger(__name__)


class InputDispatcher:
    """Serializes cursor commands and gates real OS execution behind an explicit flag."""

    def __init__(
        self,
        backend: InputBackend,
        os_input_enabled: bool = False,
        on_backend_error: Callable[[str], None] | None = None,
    ) -> None:
        self._backend = backend
        self._lock = threading.Lock()
        self._os_input_enabled = bool(os_input_enabled)
        self._on_backend_error = on_backend_error
        self._last_sent_position: tuple[int, int] | None = None

    @property
    def is_enabled(self) -> bool:
        with self._lock:
            return self._os_input_enabled

    @property
    def backend(self) -> InputBackend:
        return self._backend

    def screen_size(self) -> tuple[int, int]:
        return self._backend.screen_size()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable OS input execution. Disabling always calls release_all()."""
        with self._lock:
            enabled = bool(enabled)
            if self._os_input_enabled == enabled:
                return
            self._os_input_enabled = enabled
            self._last_sent_position = None
            if not enabled:
                self._safe_release_locked()
        logger.info("InputDispatcher OS input enabled=%s", enabled)

    def emergency_stop(self) -> None:
        """Immediately disable OS input and release any held backend state."""
        with self._lock:
            self._os_input_enabled = False
            self._last_sent_position = None
            self._safe_release_locked()
        logger.warning("InputDispatcher emergency stop triggered")

    def release_all(self) -> None:
        """Safely release any held input state without changing enablement."""
        with self._lock:
            self._last_sent_position = None
            self._safe_release_locked()

    def move_cursor(self, x: int, y: int) -> bool:
        """Send move_cursor to the backend only when OS input is explicitly enabled.

        Returns True if a backend move was executed, False if gated/skipped/errored.
        """
        error_msg: str | None = None
        with self._lock:
            if not self._os_input_enabled:
                return False
            pos = (int(x), int(y))
            if pos == self._last_sent_position:
                return False
            try:
                self._backend.move_cursor(pos[0], pos[1])
                self._last_sent_position = pos
                return True
            except Exception as exc:
                self._os_input_enabled = False
                self._last_sent_position = None
                self._safe_release_locked()
                error_msg = f"OS input error (control disabled safely): {exc}"
                logger.exception("Input backend move_cursor failed")

        if error_msg is not None and self._on_backend_error is not None:
            self._on_backend_error(error_msg)
        return False

    def close(self) -> None:
        """Disable OS input and close the underlying backend safely."""
        with self._lock:
            self._os_input_enabled = False
            self._last_sent_position = None
            try:
                self._backend.close()
            except Exception:
                logger.exception("Failed while closing input backend")

    def _safe_release_locked(self) -> None:
        try:
            self._backend.release_all()
        except Exception:
            logger.exception("Failed while releasing held input state")

