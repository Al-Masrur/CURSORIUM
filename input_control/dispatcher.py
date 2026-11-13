"""Serialized OS input dispatcher with safety gate and guaranteed release."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from input_control.backend import InputBackend

logger = logging.getLogger(__name__)

_ALLOWED_MOUSE_BUTTONS = frozenset({"left", "right"})


class InputDispatcher:
    """Serializes OS commands and gates real execution behind an explicit flag."""

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
        self._held_buttons: set[str] = set()

    @property
    def is_enabled(self) -> bool:
        with self._lock:
            return self._os_input_enabled

    @property
    def held_buttons(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._held_buttons)

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
                if not enabled:
                    self._safe_release_locked()
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
        """Send move_cursor to the backend only when OS input is explicitly enabled."""
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
                error_msg = self._handle_exception_locked("move_cursor", exc)

        if error_msg is not None and self._on_backend_error is not None:
            self._on_backend_error(error_msg)
        return False

    def click(self, button: str = "left", clicks: int = 1) -> bool:
        """Send a single or double click only when OS input is explicitly enabled."""
        if button not in _ALLOWED_MOUSE_BUTTONS:
            raise ValueError(f"Unsupported mouse button: {button!r}")
        if clicks not in (1, 2):
            raise ValueError("clicks must be 1 or 2")

        error_msg: str | None = None
        with self._lock:
            if not self._os_input_enabled:
                return False
            try:
                self._backend.click(button=button, clicks=clicks)
                return True
            except Exception as exc:
                error_msg = self._handle_exception_locked("click", exc)

        if error_msg is not None and self._on_backend_error is not None:
            self._on_backend_error(error_msg)
        return False

    def button_down(self, button: str = "left") -> bool:
        """Press and hold an allowed mouse button only when OS input is enabled."""
        if button not in _ALLOWED_MOUSE_BUTTONS:
            raise ValueError(f"Unsupported mouse button: {button!r}")

        error_msg: str | None = None
        with self._lock:
            if not self._os_input_enabled:
                return False
            if button in self._held_buttons:
                return False
            try:
                self._backend.button_down(button=button)
                self._held_buttons.add(button)
                return True
            except Exception as exc:
                error_msg = self._handle_exception_locked("button_down", exc)

        if error_msg is not None and self._on_backend_error is not None:
            self._on_backend_error(error_msg)
        return False

    def button_up(self, button: str = "left") -> bool:
        """Release a held mouse button. Safe even if OS input was just disabled."""
        if button not in _ALLOWED_MOUSE_BUTTONS:
            raise ValueError(f"Unsupported mouse button: {button!r}")

        error_msg: str | None = None
        with self._lock:
            if button not in self._held_buttons:
                return False
            try:
                self._backend.button_up(button=button)
                self._held_buttons.discard(button)
                return True
            except Exception as exc:
                error_msg = self._handle_exception_locked("button_up", exc)

        if error_msg is not None and self._on_backend_error is not None:
            self._on_backend_error(error_msg)
        return False

    def scroll(self, clicks: int) -> bool:
        """Send vertical scroll to the backend only when OS input is explicitly enabled."""
        if clicks == 0:
            return False

        error_msg: str | None = None
        with self._lock:
            if not self._os_input_enabled:
                return False
            try:
                self._backend.scroll(clicks=int(clicks))
                return True
            except Exception as exc:
                error_msg = self._handle_exception_locked("scroll", exc)

        if error_msg is not None and self._on_backend_error is not None:
            self._on_backend_error(error_msg)
        return False

    def close(self) -> None:
        """Disable OS input and close the underlying backend safely."""
        with self._lock:
            self._os_input_enabled = False
            self._last_sent_position = None
            self._held_buttons.clear()
            try:
                self._backend.close()
            except Exception:
                logger.exception("Failed while closing input backend")

    def _handle_exception_locked(self, action: str, exc: Exception) -> str:
        self._os_input_enabled = False
        self._last_sent_position = None
        self._safe_release_locked()
        logger.exception("Input backend %s failed", action)
        return f"OS input error (control disabled safely): {exc}"

    def _safe_release_locked(self) -> None:
        self._held_buttons.clear()
        try:
            self._backend.release_all()
        except Exception:
            logger.exception("Failed while releasing held input state")
