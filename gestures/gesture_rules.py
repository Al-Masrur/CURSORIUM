"""Typed gesture definitions, configurable thresholds, and pinch classification rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from vision.landmark_features import HandFeatures


class PinchType(str, Enum):
    """Normalized pinch pose recognized from palm-scaled fingertip distances."""

    NONE = "None"
    THUMB_INDEX = "Thumb-Index"
    THUMB_MIDDLE = "Thumb-Middle"


class GestureType(str, Enum):
    """High-level gesture events emitted by GestureStateManager."""

    NONE = "None"
    MOVE_CURSOR = "Move Cursor"
    PINCH_HOLD = "Pinch Hold"
    LEFT_CLICK = "Left Click"
    DOUBLE_CLICK = "Double-Click"
    RIGHT_CLICK = "Right Click"
    DRAG_START = "Drag Start"
    DRAG_MOVE = "Drag Move"
    DRAG_END = "Drag End"
    SCROLL_START = "Scroll Start"
    SCROLL_MOVE = "Scroll Move"
    SCROLL_END = "Scroll End"
    PAUSE_TOGGLED = "Pause Toggled"


@dataclass(frozen=True, slots=True)
class GestureEvent:
    """Single discrete or continuous gesture event with monotonic timestamp."""

    gesture: GestureType
    timestamp_ms: int
    reason: str = ""
    scroll_delta: int = 0


@dataclass(slots=True)
class GestureRules:
    """Validated thresholds and timing windows for click, double-click, drag, scroll, and pause."""

    pinch_activate_ratio: float = 0.32
    pinch_release_ratio: float = 0.45
    pinch_stable_ms: int = 45
    drag_hold_ms: int = 350
    double_click_interval_ms: int = 320
    click_cooldown_ms: int = 180
    pinch_smoothing_alpha: float = 1.0
    min_pinch_confidence: float = 0.0
    tracking_loss_grace_ms: int = 0
    scroll_stable_ms: int = 80
    scroll_sensitivity: float = 1.0
    scroll_deadzone: float = 0.015
    scroll_release_grace_ms: int = 0
    pause_hold_ms: int = 500
    pause_cooldown_ms: int = 1000

    def __post_init__(self) -> None:
        if not 0.10 <= self.pinch_activate_ratio <= 0.80:
            raise ValueError("pinch_activate_ratio must be between 0.10 and 0.80")
        if not (self.pinch_activate_ratio + 0.04 <= self.pinch_release_ratio <= 1.20):
            raise ValueError("pinch_release_ratio must exceed pinch_activate_ratio by >= 0.04")
        if self.pinch_stable_ms < 0:
            raise ValueError("pinch_stable_ms must be >= 0")
        if self.drag_hold_ms < 100:
            raise ValueError("drag_hold_ms must be >= 100")
        if self.double_click_interval_ms < 100:
            raise ValueError("double_click_interval_ms must be >= 100")
        if self.click_cooldown_ms < 0:
            raise ValueError("click_cooldown_ms must be >= 0")
        if not 0.05 <= self.pinch_smoothing_alpha <= 1.0:
            raise ValueError("pinch_smoothing_alpha must be between 0.05 and 1.0")
        if not 0.0 <= self.min_pinch_confidence <= 1.0:
            raise ValueError("min_pinch_confidence must be between 0.0 and 1.0")
        if self.tracking_loss_grace_ms < 0:
            raise ValueError("tracking_loss_grace_ms must be >= 0")
        if self.scroll_stable_ms < 0:
            raise ValueError("scroll_stable_ms must be >= 0")
        if not 0.1 <= self.scroll_sensitivity <= 10.0:
            raise ValueError("scroll_sensitivity must be between 0.1 and 10.0")
        if not 0.0 <= self.scroll_deadzone <= 0.20:
            raise ValueError("scroll_deadzone must be between 0.0 and 0.20")
        if self.scroll_release_grace_ms < 0:
            raise ValueError("scroll_release_grace_ms must be >= 0")
        if self.pause_hold_ms < 100:
            raise ValueError("pause_hold_ms must be >= 100")
        if self.pause_cooldown_ms < 100:
            raise ValueError("pause_cooldown_ms must be >= 100")


def classify_pinch_with_hysteresis(
    features: HandFeatures,
    active_pinch: PinchType,
    rules: GestureRules,
) -> PinchType:
    """Classify thumb-index or thumb-middle pinch using separate enter/exit thresholds.

    When neither pinch is currently active, simultaneous hits (such as a closed fist)
    are rejected as ambiguous to prevent accidental clicks.
    """
    if (
        not features.hand_detected
        or features.thumb_index_ratio is None
        or features.thumb_middle_ratio is None
    ):
        return PinchType.NONE

    if active_pinch == PinchType.THUMB_INDEX:
        if features.thumb_index_ratio < rules.pinch_release_ratio:
            return PinchType.THUMB_INDEX
        return PinchType.NONE

    if active_pinch == PinchType.THUMB_MIDDLE:
        if features.thumb_middle_ratio < rules.pinch_release_ratio:
            return PinchType.THUMB_MIDDLE
        return PinchType.NONE

    index_hit = features.thumb_index_ratio <= rules.pinch_activate_ratio
    middle_hit = features.thumb_middle_ratio <= rules.pinch_activate_ratio

    if index_hit and middle_hit:
        return PinchType.NONE
    if index_hit:
        return PinchType.THUMB_INDEX
    if middle_hit:
        return PinchType.THUMB_MIDDLE
    return PinchType.NONE

