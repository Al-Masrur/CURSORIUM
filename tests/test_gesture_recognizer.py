"""Unit tests for GestureRecognizer hysteresis, temporal debounce, and false positives."""

from __future__ import annotations

from app.events import NormalizedPoint
from gestures.gesture_rules import GestureRules, PinchType
from gestures.recognizer import GestureRecognizer
from vision.landmark_features import HandFeatures


def _features(
    *,
    thumb_index_ratio: float | None = 0.80,
    thumb_middle_ratio: float | None = 0.80,
    hand_detected: bool = True,
    index_extended: bool = True,
    confidence: float | None = None,
) -> HandFeatures:
    resolved_confidence = confidence if confidence is not None else (0.95 if hand_detected else 0.0)
    return HandFeatures(
        hand_detected=hand_detected,
        index_tip=NormalizedPoint(0.5, 0.5) if hand_detected else None,
        index_extended=index_extended,
        confidence=resolved_confidence,
        palm_scale=0.2 if hand_detected else None,
        thumb_index_ratio=thumb_index_ratio if hand_detected else None,
        thumb_middle_ratio=thumb_middle_ratio if hand_detected else None,
    )


def test_hysteresis_and_temporal_validation() -> None:
    rules = GestureRules(
        pinch_activate_ratio=0.32,
        pinch_release_ratio=0.45,
        pinch_stable_ms=40,
    )
    rec = GestureRecognizer(rules=rules)

    # Initial frame below activate threshold must wait for temporal validation
    assert rec.update(_features(thumb_index_ratio=0.25), timestamp_ms=0) == PinchType.NONE
    assert rec.update(_features(thumb_index_ratio=0.25), timestamp_ms=25) == PinchType.NONE
    assert rec.update(_features(thumb_index_ratio=0.25), timestamp_ms=45) == PinchType.THUMB_INDEX

    # Inside hysteresis band [0.32, 0.45) stays active
    assert rec.update(_features(thumb_index_ratio=0.39), timestamp_ms=70) == PinchType.THUMB_INDEX

    # Rising past release threshold (>= 0.45) releases pinch
    assert rec.update(_features(thumb_index_ratio=0.46), timestamp_ms=95) == PinchType.NONE


def test_single_frame_spike_does_not_trigger_pinch() -> None:
    rules = GestureRules(pinch_stable_ms=45)
    rec = GestureRecognizer(rules=rules)

    assert rec.update(_features(thumb_index_ratio=0.20), timestamp_ms=10) == PinchType.NONE
    assert rec.update(_features(thumb_index_ratio=0.70), timestamp_ms=40) == PinchType.NONE
    assert rec.update(_features(thumb_index_ratio=0.20), timestamp_ms=60) == PinchType.NONE


def test_ambiguous_simultaneous_pinch_or_fist_rejected() -> None:
    rules = GestureRules(pinch_stable_ms=0)
    rec = GestureRecognizer(rules=rules)

    # Both thumb-index and thumb-middle pinched at the same time (e.g., fist) -> NONE
    assert (
        rec.update(_features(thumb_index_ratio=0.20, thumb_middle_ratio=0.22), timestamp_ms=10)
        == PinchType.NONE
    )


def test_hand_loss_resets_active_pinch() -> None:
    rules = GestureRules(pinch_stable_ms=0)
    rec = GestureRecognizer(rules=rules)

    assert rec.update(_features(thumb_index_ratio=0.20), timestamp_ms=10) == PinchType.THUMB_INDEX
    assert rec.update(_features(hand_detected=False), timestamp_ms=40) == PinchType.NONE
    assert rec.active_pinch == PinchType.NONE


def test_pinch_ema_and_confidence_gate_suppress_noisy_frames() -> None:
    rules = GestureRules(
        pinch_activate_ratio=0.32,
        pinch_release_ratio=0.45,
        pinch_stable_ms=0,
        pinch_smoothing_alpha=0.5,
        min_pinch_confidence=0.60,
    )
    rec = GestureRecognizer(rules=rules)

    # Low-confidence noisy frame is blocked from starting a pinch even at low ratio
    low_conf = _features(thumb_index_ratio=0.20, confidence=0.35)
    assert rec.update(low_conf, timestamp_ms=10) == PinchType.NONE

    # High-confidence open frame anchors EMA near 0.60, so one noisy 0.20 drop
    # smooths to 0.40 (> 0.32) and does not falsely trigger; sustained 0.20 activates.
    assert rec.update(_features(thumb_index_ratio=0.60, confidence=0.90), timestamp_ms=40) == PinchType.NONE
    assert rec.update(_features(thumb_index_ratio=0.20, confidence=0.90), timestamp_ms=70) == PinchType.NONE
    assert rec.update(_features(thumb_index_ratio=0.20, confidence=0.90), timestamp_ms=100) == PinchType.THUMB_INDEX

