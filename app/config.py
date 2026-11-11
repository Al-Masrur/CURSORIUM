"""Validated configuration defaults for HandPilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "hand_landmarker.task"


@dataclass(slots=True)
class AppConfig:
    """Runtime settings for webcam capture, hand tracking, and cursor mapping."""

    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    mirror_preview: bool = True
    show_landmarks: bool = True
    model_path: Path = DEFAULT_MODEL_PATH
    num_hands: int = 1
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_consecutive_read_failures: int = 15

    # Phase 2: Virtual mouse movement settings (OS input stays disabled by default)
    os_input_enabled: bool = False
    active_region_left: float = 0.15
    active_region_top: float = 0.15
    active_region_right: float = 0.85
    active_region_bottom: float = 0.85
    cursor_sensitivity: float = 1.0
    cursor_smoothing: float = 0.35

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        if self.camera_index < 0:
            raise ValueError("camera_index must be >= 0")
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("frame_width and frame_height must be positive")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if self.num_hands != 1:
            raise ValueError("MVP supports tracking exactly 1 hand (num_hands=1)")
        for field_name in (
            "min_hand_detection_confidence",
            "min_hand_presence_confidence",
            "min_tracking_confidence",
        ):
            val = getattr(self, field_name)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0")
        if self.max_consecutive_read_failures <= 0:
            raise ValueError("max_consecutive_read_failures must be positive")
        if not (0.0 <= self.active_region_left < self.active_region_right <= 1.0):
            raise ValueError("Active horizontal region must satisfy 0.0 <= left < right <= 1.0")
        if not (0.0 <= self.active_region_top < self.active_region_bottom <= 1.0):
            raise ValueError("Active vertical region must satisfy 0.0 <= top < bottom <= 1.0")
        if not 0.2 <= self.cursor_sensitivity <= 3.0:
            raise ValueError("cursor_sensitivity must be between 0.2 and 3.0")
        if not 0.0 <= self.cursor_smoothing <= 0.95:
            raise ValueError("cursor_smoothing must be between 0.0 and 0.95")
