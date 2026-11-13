"""Validated configuration defaults for HandPilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "hand_landmarker.task"


@dataclass(slots=True)
class AppConfig:
    """Runtime settings for webcam capture, tracking, cursor mapping, and gestures."""

    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    mirror_preview: bool = True
    show_landmarks: bool = True
    low_light_boost: bool = True
    model_path: Path = DEFAULT_MODEL_PATH
    num_hands: int = 1
    min_hand_detection_confidence: float = 0.55
    min_hand_presence_confidence: float = 0.55
    min_tracking_confidence: float = 0.45
    max_consecutive_read_failures: int = 15
    stale_frame_timeout_ms: int = 250

    # Phase 2: Virtual mouse movement settings (OS input stays disabled by default)
    os_input_enabled: bool = False
    active_region_left: float = 0.15
    active_region_top: float = 0.15
    active_region_right: float = 0.85
    active_region_bottom: float = 0.85
    cursor_sensitivity: float = 1.0
    cursor_smoothing: float = 0.35
    adaptive_smoothing: bool = True
    cursor_deadband_px: float = 1.5

    # Phase 3: Click and drag gesture thresholds and timing
    pinch_activate_ratio: float = 0.32
    pinch_release_ratio: float = 0.45
    pinch_stable_ms: int = 50
    pinch_smoothing_alpha: float = 0.70
    min_pinch_confidence: float = 0.50
    drag_hold_ms: int = 350
    double_click_interval_ms: int = 320
    click_cooldown_ms: int = 200
    tracking_loss_grace_ms: int = 90

    # Phase 4: Scroll and Pause settings
    scroll_sensitivity: float = 1.0
    scroll_deadzone: float = 0.015
    scroll_stable_ms: int = 80
    pause_hold_ms: int = 500
    pause_cooldown_ms: int = 1000

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
            "min_pinch_confidence",
        ):
            val = getattr(self, field_name)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0")
        if self.max_consecutive_read_failures <= 0:
            raise ValueError("max_consecutive_read_failures must be positive")
        if self.stale_frame_timeout_ms <= 0:
            raise ValueError("stale_frame_timeout_ms must be positive")
        if not (0.0 <= self.active_region_left < self.active_region_right <= 1.0):
            raise ValueError("Active horizontal region must satisfy 0.0 <= left < right <= 1.0")
        if not (0.0 <= self.active_region_top < self.active_region_bottom <= 1.0):
            raise ValueError("Active vertical region must satisfy 0.0 <= top < bottom <= 1.0")
        if not 0.2 <= self.cursor_sensitivity <= 3.0:
            raise ValueError("cursor_sensitivity must be between 0.2 and 3.0")
        if not 0.0 <= self.cursor_smoothing <= 0.95:
            raise ValueError("cursor_smoothing must be between 0.0 and 0.95")
        if self.cursor_deadband_px < 0.0:
            raise ValueError("cursor_deadband_px must be >= 0.0")
        if not 0.10 <= self.pinch_activate_ratio <= 0.80:
            raise ValueError("pinch_activate_ratio must be between 0.10 and 0.80")
        if not (self.pinch_activate_ratio + 0.04 <= self.pinch_release_ratio <= 1.20):
            raise ValueError("pinch_release_ratio must exceed pinch_activate_ratio by at least 0.04")
        if self.pinch_stable_ms < 0:
            raise ValueError("pinch_stable_ms must be >= 0")
        if not 0.0 < self.pinch_smoothing_alpha <= 1.0:
            raise ValueError("pinch_smoothing_alpha must be in (0.0, 1.0]")
        if self.drag_hold_ms < 100:
            raise ValueError("drag_hold_ms must be >= 100")
        if self.double_click_interval_ms < 100:
            raise ValueError("double_click_interval_ms must be >= 100")
        if self.click_cooldown_ms < 0:
            raise ValueError("click_cooldown_ms must be >= 0")
        if self.tracking_loss_grace_ms < 0:
            raise ValueError("tracking_loss_grace_ms must be >= 0")
        if not 0.1 <= self.scroll_sensitivity <= 10.0:
            raise ValueError("scroll_sensitivity must be between 0.1 and 10.0")
        if not 0.0 <= self.scroll_deadzone <= 0.20:
            raise ValueError("scroll_deadzone must be between 0.0 and 0.20")
        if self.scroll_stable_ms < 0:
            raise ValueError("scroll_stable_ms must be >= 0")
        if self.pause_hold_ms < 100:
            raise ValueError("pause_hold_ms must be >= 100")
        if self.pause_cooldown_ms < 100:
            raise ValueError("pause_cooldown_ms must be >= 100")
