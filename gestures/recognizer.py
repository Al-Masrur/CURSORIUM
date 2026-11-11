"""Temporal gesture recognizer with hysteresis and debounce."""

from __future__ import annotations

from dataclasses import replace

from gestures.gesture_rules import (
    GestureRules,
    PinchType,
    classify_pinch_with_hysteresis,
)
from vision.landmark_features import HandFeatures


class GestureRecognizer:
    """Converts per-frame HandFeatures into temporally validated PinchType states."""

    def __init__(self, rules: GestureRules | None = None) -> None:
        self._rules = rules or GestureRules()
        self._active_pinch: PinchType = PinchType.NONE
        self._candidate_pinch: PinchType = PinchType.NONE
        self._candidate_since_ms: int | None = None
        self._smoothed_index_ratio: float | None = None
        self._smoothed_middle_ratio: float | None = None

    @property
    def rules(self) -> GestureRules:
        return self._rules

    @property
    def active_pinch(self) -> PinchType:
        return self._active_pinch

    def update_rules(self, rules: GestureRules) -> None:
        self._rules = rules

    def reset(self) -> None:
        """Clear active and candidate pinch tracking."""
        self._active_pinch = PinchType.NONE
        self._candidate_pinch = PinchType.NONE
        self._candidate_since_ms = None
        self._smoothed_index_ratio = None
        self._smoothed_middle_ratio = None

    def _smooth_ratios(self, index_ratio: float, middle_ratio: float) -> tuple[float, float]:
        alpha = self._rules.pinch_smoothing_alpha
        if alpha >= 0.999 or self._smoothed_index_ratio is None or self._smoothed_middle_ratio is None:
            self._smoothed_index_ratio = index_ratio
            self._smoothed_middle_ratio = middle_ratio
        else:
            self._smoothed_index_ratio = alpha * index_ratio + (1.0 - alpha) * self._smoothed_index_ratio
            self._smoothed_middle_ratio = alpha * middle_ratio + (1.0 - alpha) * self._smoothed_middle_ratio
        return (self._smoothed_index_ratio, self._smoothed_middle_ratio)

    def update(self, features: HandFeatures, timestamp_ms: int) -> PinchType:
        """Return confirmed PinchType after hysteresis and temporal stability checks."""
        if (
            not features.hand_detected
            or features.thumb_index_ratio is None
            or features.thumb_middle_ratio is None
        ):
            self.reset()
            return PinchType.NONE

        if (
            self._active_pinch == PinchType.NONE
            and features.confidence < self._rules.min_pinch_confidence
        ):
            self._candidate_pinch = PinchType.NONE
            self._candidate_since_ms = None
            return PinchType.NONE

        smoothed_index, smoothed_middle = self._smooth_ratios(
            features.thumb_index_ratio,
            features.thumb_middle_ratio,
        )
        eval_features = replace(
            features,
            thumb_index_ratio=smoothed_index,
            thumb_middle_ratio=smoothed_middle,
        )

        if self._active_pinch != PinchType.NONE:
            next_pinch = classify_pinch_with_hysteresis(
                eval_features,
                self._active_pinch,
                self._rules,
            )
            if next_pinch == PinchType.NONE:
                self.reset()
                return PinchType.NONE
            return self._active_pinch

        candidate = classify_pinch_with_hysteresis(
            eval_features,
            PinchType.NONE,
            self._rules,
        )
        if candidate == PinchType.NONE:
            self._candidate_pinch = PinchType.NONE
            self._candidate_since_ms = None
            return PinchType.NONE

        if candidate != self._candidate_pinch:
            self._candidate_pinch = candidate
            self._candidate_since_ms = timestamp_ms

        assert self._candidate_since_ms is not None
        if timestamp_ms - self._candidate_since_ms >= self._rules.pinch_stable_ms:
            self._active_pinch = candidate
            self._candidate_pinch = PinchType.NONE
            self._candidate_since_ms = None
            return self._active_pinch

        return PinchType.NONE

