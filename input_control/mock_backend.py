"""In-memory mock InputBackend for safe testing without OS cursor movement."""

from __future__ import annotations

import threading

from input_control.backend import InputBackend


class MockInputBackend(InputBackend):
    """Records cursor moves and release_all calls without touching the OS."""

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        fail_on_move: bool = False,
    ) -> None:
        if screen_width <= 0 or screen_height <= 0:
            raise ValueError("screen_width and screen_height must be positive")
        self._lock = threading.Lock()
        self._screen_width = int(screen_width)
        self._screen_height = int(screen_height)
        self.fail_on_move = fail_on_move
        self.move_history: list[tuple[int, int]] = []
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

    def release_all(self) -> None:
        with self._lock:
            self.release_all_count += 1

    def close(self) -> None:
        self.release_all()
        with self._lock:
            self.closed = True

