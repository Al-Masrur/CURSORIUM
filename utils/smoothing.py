"""Exponential moving average (EMA) smoothing helpers for cursor motion."""

from __future__ import annotations


def smoothing_to_alpha(smoothing: float) -> float:
    """Convert a UI smoothing factor in [0.0, 0.95] to an EMA alpha in [0.05, 1.0].

    Higher smoothing produces a smaller alpha (more damping, less jitter).
    """
    if not 0.0 <= smoothing <= 0.95:
        raise ValueError("smoothing must be between 0.0 and 0.95")
    return max(0.05, min(1.0, 1.0 - smoothing))


class ExponentialSmoother:
    """Applies exponential moving average smoothing to 2D coordinates."""

    def __init__(self, alpha: float = 0.65) -> None:
        self._alpha = self._validate_alpha(alpha)
        self._previous: tuple[float, float] | None = None

    @staticmethod
    def _validate_alpha(alpha: float) -> float:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0.0, 1.0]")
        return float(alpha)

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = self._validate_alpha(value)

    @property
    def has_state(self) -> bool:
        return self._previous is not None

    def reset(self) -> None:
        """Clear previous smoothed state (used on hand loss or calibration change)."""
        self._previous = None

    def update(self, x: float, y: float) -> tuple[float, float]:
        """Compute smoothed (x, y) using ``alpha * current + (1 - alpha) * previous``."""
        if self._previous is None or self._alpha == 1.0:
            self._previous = (float(x), float(y))
            return self._previous

        prev_x, prev_y = self._previous
        inv = 1.0 - self._alpha
        smoothed_x = self._alpha * x + inv * prev_x
        smoothed_y = self._alpha * y + inv * prev_y
        self._previous = (smoothed_x, smoothed_y)
        return self._previous

