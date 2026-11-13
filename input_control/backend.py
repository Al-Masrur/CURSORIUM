"""Abstract interface for OS-level input backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class InputBackend(ABC):
    """Replaceable input backend interface for cursor movement, clicks, and safe release."""

    @abstractmethod
    def screen_size(self) -> tuple[int, int]:
        """Return primary screen (width, height) in pixels."""

    @abstractmethod
    def move_cursor(self, x: int, y: int) -> None:
        """Move the cursor to absolute screen pixel coordinates (x, y)."""

    @abstractmethod
    def click(self, button: str = "left", clicks: int = 1) -> None:
        """Execute one or more mouse clicks for the given button ('left' or 'right')."""

    @abstractmethod
    def button_down(self, button: str = "left") -> None:
        """Press and hold a mouse button."""

    @abstractmethod
    def button_up(self, button: str = "left") -> None:
        """Release a held mouse button."""

    @abstractmethod
    def release_all(self) -> None:
        """Release any held input state safely. Must be safe to call repeatedly."""

    @abstractmethod
    def scroll(self, clicks: int) -> None:
        """Execute vertical mouse scroll by the specified number of wheel clicks (positive=up, negative=down)."""

    def close(self) -> None:
        """Release held inputs and clean up backend resources."""
        self.release_all()
