"""Desktop OS input backend using PyAutoGUI behind the InputBackend interface."""

from __future__ import annotations

import logging
import threading
from typing import Any

from input_control.backend import InputBackend

logger = logging.getLogger(__name__)


class DesktopInputBackend(InputBackend):
    """Executes real OS cursor movement, clicks, and button holds via PyAutoGUI."""

    def __init__(self, pyautogui_module: Any | None = None) -> None:
        self._lock = threading.Lock()
        self._pyautogui = pyautogui_module
        self._configured = False
        self._held_buttons: set[str] = set()
        self._cached_screen_size: tuple[int, int] | None = None

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
        screen_dims = (int(size.width), int(size.height))
        self._cached_screen_size = screen_dims
        return screen_dims

    def move_cursor(self, x: int, y: int) -> None:
        with self._lock:
            api = self._get_api()
            if self._cached_screen_size is None:
                size = api.size()
                self._cached_screen_size = (int(size.width), int(size.height))
            width, height = self._cached_screen_size
            max_x = max(0, width - 1)
            max_y = max(0, height - 1)
            clamped_x = max(0, min(max_x, int(x)))
            clamped_y = max(0, min(max_y, int(y)))
            api.moveTo(clamped_x, clamped_y, _pause=False)

    def click(self, button: str = "left", clicks: int = 1) -> None:
        with self._lock:
            api = self._get_api()
            api.click(button=button, clicks=int(clicks), _pause=False)

    def button_down(self, button: str = "left") -> None:
        with self._lock:
            if button in self._held_buttons:
                return
            api = self._get_api()
            api.mouseDown(button=button, _pause=False)
            self._held_buttons.add(button)

    def button_up(self, button: str = "left") -> None:
        with self._lock:
            if button not in self._held_buttons:
                return
            api = self._get_api()
            try:
                api.mouseUp(button=button, _pause=False)
            finally:
                self._held_buttons.discard(button)

    def scroll(self, clicks: int) -> None:
        with self._lock:
            if clicks == 0:
                return
            api = self._get_api()
            api.scroll(clicks=int(clicks), _pause=False)

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
