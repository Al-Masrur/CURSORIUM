"""Abstract interface for OS-level input backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class InputBackend(ABC):
    """Replaceable input backend interface for cursor movement and safe release."""

    @abstractmethod
    def screen_size(self) -> tuple[int, int]:
        """Return primary screen (width, height) in pixels."""

    @abstractmethod
    def move_cursor(self, x: int, y: int) -> None:
        """Move the cursor to absolute screen pixel coordinates (x, y)."""

    @abstractmethod
    def release_all(self) -> None:
        """Release any held input state safely. Must be safe to call repeatedly."""

    def close(self) -> None:
        """Release held inputs and clean up backend resources."""
        self.release_all()

