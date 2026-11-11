"""Camera-to-screen coordinate mapping, calibration region, clamping, and smoothing."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from app.events import NormalizedPoint
from utils.smoothing import ExponentialSmoother, smoothing_to_alpha


@dataclass(frozen=True, slots=True)
class ActiveRegion:
    """Normalized active control box inside camera space ([0, 1] x [0, 1])."""

    left: float = 0.15
    top: float = 0.15
    right: float = 0.85
    bottom: float = 0.85

    def __post_init__(self) -> None:
        if not (0.0 <= self.left < self.right <= 1.0):
            raise ValueError("ActiveRegion must satisfy 0.0 <= left < right <= 1.0")
        if not (0.0 <= self.top < self.bottom <= 1.0):
            raise ValueError("ActiveRegion must satisfy 0.0 <= top < bottom <= 1.0")

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @classmethod
    def from_margins(cls, margin_x: float, margin_y: float) -> ActiveRegion:
        """Create a symmetric centered region from horizontal and vertical margins."""
        if not (0.0 <= margin_x < 0.5) or not (0.0 <= margin_y < 0.5):
            raise ValueError("Margins must be in [0.0, 0.5)")
        return cls(
            left=margin_x,
            top=margin_y,
            right=1.0 - margin_x,
            bottom=1.0 - margin_y,
        )

    @classmethod
    def centered_around(
        cls,
        center: NormalizedPoint,
        width: float,
        height: float,
    ) -> ActiveRegion:
        """Center a box of size (width, height) around ``center``, clamped inside [0, 1]."""
        if not (0.05 <= width <= 1.0) or not (0.05 <= height <= 1.0):
            raise ValueError("Region width and height must be in [0.05, 1.0]")

        half_w = width / 2.0
        half_h = height / 2.0

        left = max(0.0, min(1.0 - width, center.x - half_w))
        top = max(0.0, min(1.0 - height, center.y - half_h))
        right = min(1.0, left + width)
        bottom = min(1.0, top + height)
        return cls(left=left, top=top, right=right, bottom=bottom)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class CoordinateMapper:
    """Maps normalized camera coordinates into smoothed, clamped screen pixels."""

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        active_region: ActiveRegion | None = None,
        sensitivity: float = 1.0,
        smoothing: float = 0.35,
        mirror_x: bool = False,
    ) -> None:
        if screen_width <= 0 or screen_height <= 0:
            raise ValueError("screen_width and screen_height must be positive")
        if not 0.2 <= sensitivity <= 3.0:
            raise ValueError("sensitivity must be between 0.2 and 3.0")

        self._lock = threading.Lock()
        self._screen_width = int(screen_width)
        self._screen_height = int(screen_height)
        self._active_region = active_region or ActiveRegion()
        self._sensitivity = float(sensitivity)
        self._smoothing = float(smoothing)
        self._mirror_x = bool(mirror_x)
        self._smoother = ExponentialSmoother(alpha=smoothing_to_alpha(self._smoothing))

    @property
    def active_region(self) -> ActiveRegion:
        with self._lock:
            return self._active_region

    @property
    def screen_size(self) -> tuple[int, int]:
        with self._lock:
            return (self._screen_width, self._screen_height)

    @property
    def sensitivity(self) -> float:
        with self._lock:
            return self._sensitivity

    @property
    def smoothing(self) -> float:
        with self._lock:
            return self._smoothing

    @property
    def mirror_x(self) -> bool:
        with self._lock:
            return self._mirror_x

    def reset_smoothing(self) -> None:
        """Reset EMA smoothing history (e.g., on hand loss or calibration change)."""
        with self._lock:
            self._smoother.reset()

    def set_screen_size(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("screen_width and screen_height must be positive")
        with self._lock:
            if (self._screen_width, self._screen_height) != (width, height):
                self._screen_width = int(width)
                self._screen_height = int(height)
                self._smoother.reset()

    def set_active_region(self, region: ActiveRegion) -> None:
        with self._lock:
            self._active_region = region
            self._smoother.reset()

    def set_sensitivity(self, sensitivity: float) -> None:
        if not 0.2 <= sensitivity <= 3.0:
            raise ValueError("sensitivity must be between 0.2 and 3.0")
        with self._lock:
            self._sensitivity = float(sensitivity)

    def set_smoothing(self, smoothing: float) -> None:
        alpha = smoothing_to_alpha(smoothing)
        with self._lock:
            self._smoothing = float(smoothing)
            self._smoother.alpha = alpha

    def set_mirror_x(self, mirror_x: bool) -> None:
        with self._lock:
            if self._mirror_x != bool(mirror_x):
                self._mirror_x = bool(mirror_x)
                self._smoother.reset()

    def map_point(self, point: NormalizedPoint) -> tuple[int, int]:
        """Map a normalized camera point into clamped, smoothed screen pixel coordinates."""
        with self._lock:
            x = 1.0 - point.x if self._mirror_x else point.x
            y = point.y

            region = self._active_region
            nx = (x - region.left) / region.width
            ny = (y - region.top) / region.height

            # Scale around the center of the active region according to sensitivity.
            nx = 0.5 + (nx - 0.5) * self._sensitivity
            ny = 0.5 + (ny - 0.5) * self._sensitivity

            nx = _clamp(nx, 0.0, 1.0)
            ny = _clamp(ny, 0.0, 1.0)

            max_x = max(0, self._screen_width - 1)
            max_y = max(0, self._screen_height - 1)

            raw_x = nx * max_x
            raw_y = ny * max_y

            smooth_x, smooth_y = self._smoother.update(raw_x, raw_y)

            clamped_x = int(_clamp(round(smooth_x), 0, max_x))
            clamped_y = int(_clamp(round(smooth_y), 0, max_y))
            return (clamped_x, clamped_y)

