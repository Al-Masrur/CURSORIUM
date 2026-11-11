"""Typed data structures shared between camera, vision, controller, and UI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class CameraStatus(str, Enum):
    """Lifecycle status of the camera capture pipeline."""

    STOPPED = "Stopped"
    STARTING = "Starting"
    RUNNING = "Running"
    ERROR = "Error"


@dataclass(frozen=True, slots=True)
class FramePacket:
    """Single captured video frame with a monotonic millisecond timestamp."""

    frame_bgr: np.ndarray
    timestamp_ms: int
    sequence_id: int


@dataclass(frozen=True, slots=True)
class NormalizedPoint:
    """Normalized 3D landmark coordinate in image space (x, y typically in [0, 1])."""

    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True, slots=True)
class HandDetection:
    """Single-hand 21-landmark detection result."""

    landmarks: tuple[NormalizedPoint, ...]
    handedness: str
    confidence: float


@dataclass(frozen=True, slots=True)
class TrackingResult:
    """Per-frame tracking output published from VisionWorker to the controller/UI."""

    timestamp_ms: int
    hand_detected: bool
    detection: HandDetection | None
    camera_fps: float
    tracking_fps: float
    error: str | None = None

