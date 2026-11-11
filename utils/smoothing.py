"""Exponential and velocity-adaptive smoothing helpers for cursor motion."""

from __future__ import annotations

import math


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


class AdaptiveCursorSmoother(ExponentialSmoother):
    """Velocity-adaptive EMA with micro-jitter deadband for noisy laptop webcams.

    - Movements smaller than ``deadband_px`` are suppressed when smoothing is enabled,
      eliminating fingertip tremor while hovering over buttons.
    - Slow movements receive extra damping for precision.
    - Fast hand sweeps ramp alpha up for low-latency tracking.
    - When ``alpha == 1.0`` (smoothing = 0%), acts as an exact raw passthrough.
    """

    def __init__(
        self,
        alpha: float = 0.65,
        deadband_px: float = 1.5,
        min_alpha_scale: float = 0.35,
        max_alpha_scale: float = 1.5,
        velocity_ref_px: float = 35.0,
    ) -> None:
        super().__init__(alpha=alpha)
        if deadband_px < 0.0:
            raise ValueError("deadband_px must be >= 0.0")
        self._deadband_px = float(deadband_px)
        self._min_alpha_scale = float(min_alpha_scale)
        self._max_alpha_scale = float(max_alpha_scale)
        self._velocity_ref_px = max(1.0, float(velocity_ref_px))

    def update(self, x: float, y: float) -> tuple[float, float]:
        if self._previous is None or self._alpha >= 0.999:
            self._previous = (float(x), float(y))
            return self._previous

        prev_x, prev_y = self._previous
        dist = math.hypot(x - prev_x, y - prev_y)
        if dist < self._deadband_px:
            return self._previous

        speed_ratio = min(1.0, (dist - self._deadband_px) / self._velocity_ref_px)
        scale = self._min_alpha_scale + (self._max_alpha_scale - self._min_alpha_scale) * speed_ratio
        effective_alpha = max(0.05, min(1.0, self._alpha * scale))
        inv = 1.0 - effective_alpha
        smoothed_x = effective_alpha * x + inv * prev_x
        smoothed_y = effective_alpha * y + inv * prev_y
        self._previous = (smoothed_x, smoothed_y)
        return self._previous
