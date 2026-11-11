"""Desktop OS input backend using PyAutoGUI behind the InputBackend interface."""

from __future__ import annotations

import logging
import threading
from typing import Any

from input_control.backend import InputBackend

logger = logging.getLogger(__name__)


class DesktopInputBackend(InputBackend):
    """Executes real OS cursor movement via PyAutoGUI."""

    def __init__(self, pyautogui_module: Any | None = None) -> None:
        self._lock = threading.Lock()
        self._pyautogui = pyautogui_module
        self._configured = False
        self._held_buttons: set[str] = set()

    def _get_api(self) -> Any:
        if self._pyautogui is None:
            import pyautogui  # lazy import so tests without display can avoid importing it

            self._pyautogui = pyautogui

        if not self._configured:
            # Remove PyAutoGUI's default 100ms per-call sleep and corner failsafe;
            # HandPilot enforces its own Esc emergency stop and safe release.
            self._pyautogui.PAUSE = 0.0
            self._pyautogui.FAILSAFE = False
            self._configured = True
        return self._pyautogui

    def screen_size(self) -> tuple[int, int]:
        api = self._get_api()
        size = api.size()
        return (int(size.width), int(size.height))

    def move_cursor(self, x: int, y: int) -> None:
        with self._lock:
            api = self._get_api()
            size = api.size()
            max_x = max(0, int(size.width) - 1)
            max_y = max(0, int(size.height) - 1)
            clamped_x = max(0, min(max_x, int(x)))
            clamped_y = max(0, min(max_y, int(y)))
            api.moveTo(clamped_x, clamped_y, _pause=False)

    def release_all(self) -> None:
        with self._lock:
            if not self._held_buttons:
                return
            api = self._get_api()
            for button in list(self._held_buttons):
                try:
                    api.mouseUp(button=button, _pause=False)
                except Exception:
                    logger.exception("Failed releasing button '%s'", button)
            self._held_buttons.clear()

