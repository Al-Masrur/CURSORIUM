"""In-memory mock InputBackend for safe testing without OS mouse or click execution."""

from __future__ import annotations

import threading

from input_control.backend import InputBackend


class MockInputBackend(InputBackend):
    """Records cursor moves, clicks, button transitions, and release_all calls."""

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        fail_on_move: bool = False,
        fail_on_click: bool = False,
        fail_on_button: bool = False,
        fail_on_scroll: bool = False,
    ) -> None:
        if screen_width <= 0 or screen_height <= 0:
            raise ValueError("screen_width and screen_height must be positive")
        self._lock = threading.Lock()
        self._screen_width = int(screen_width)
        self._screen_height = int(screen_height)
        self.fail_on_move = fail_on_move
        self.fail_on_click = fail_on_click
        self.fail_on_button = fail_on_button
        self.fail_on_scroll = fail_on_scroll
        self.move_history: list[tuple[int, int]] = []
        self.click_history: list[tuple[str, int, tuple[int, int] | None]] = []
        self.button_events: list[tuple[str, str]] = []
        self.scroll_history: list[int] = []
        self.held_buttons: set[str] = set()
        self.last_position: tuple[int, int] | None = None
        self.release_all_count: int = 0
        self.closed: bool = False

    def screen_size(self) -> tuple[int, int]:
        return (self._screen_width, self._screen_height)

    def move_cursor(self, x: int, y: int) -> None:
        with self._lock:
            if self.fail_on_move:
                raise RuntimeError("Simulated input backend move failure")
            pos = (int(x), int(y))
            self.last_position = pos
            self.move_history.append(pos)

    def click(self, button: str = "left", clicks: int = 1) -> None:
        with self._lock:
            if self.fail_on_click:
                raise RuntimeError("Simulated input backend click failure")
            self.click_history.append((button, int(clicks), self.last_position))

    def button_down(self, button: str = "left") -> None:
        with self._lock:
            if self.fail_on_button:
                raise RuntimeError("Simulated input backend button_down failure")
            if button not in self.held_buttons:
                self.held_buttons.add(button)
                self.button_events.append(("down", button))

    def button_up(self, button: str = "left") -> None:
        with self._lock:
            if button in self.held_buttons:
                self.held_buttons.remove(button)
                self.button_events.append(("up", button))

    def scroll(self, clicks: int) -> None:
        with self._lock:
            if self.fail_on_scroll:
                raise RuntimeError("Simulated input backend scroll failure")
            self.scroll_history.append(int(clicks))

    def release_all(self) -> None:
        with self._lock:
            for button in sorted(self.held_buttons):
                self.button_events.append(("up", button))
            self.held_buttons.clear()
            self.release_all_count += 1

    def close(self) -> None:
        self.release_all()
        with self._lock:
            self.closed = True
