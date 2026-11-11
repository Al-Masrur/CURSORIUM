"""Unit tests for landmark 8 extraction and index finger extension geometry."""

from __future__ import annotations

from app.events import HandDetection, NormalizedPoint
from vision.landmark_features import (
    INDEX_DIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_MCP,
    THUMB_TIP,
    WRIST,
    compute_palm_scale,
    extract_hand_features,
    extract_index_fingertip,
    is_index_finger_extended,
)


def _make_hand(
    p5: tuple[float, float],
    p6: tuple[float, float],
    p7: tuple[float, float],
    p8: tuple[float, float],
    *,
    confidence: float = 0.95,
) -> HandDetection:
    points = [NormalizedPoint(x=0.5, y=0.8) for _ in range(21)]
    points[INDEX_MCP] = NormalizedPoint(*p5)
    points[INDEX_PIP] = NormalizedPoint(*p6)
    points[INDEX_DIP] = NormalizedPoint(*p7)
    points[INDEX_TIP] = NormalizedPoint(*p8)
    return HandDetection(landmarks=tuple(points), handedness="Right", confidence=confidence)


def test_extract_index_fingertip_returns_landmark_8() -> None:
    hand = _make_hand((0.5, 0.7), (0.5, 0.6), (0.5, 0.5), (0.52, 0.38))
    tip = extract_index_fingertip(hand)
    assert tip == NormalizedPoint(0.52, 0.38)
    assert extract_index_fingertip(None) is None

    short_hand = HandDetection(
        landmarks=(NormalizedPoint(0.1, 0.2),),
        handedness="Right",
        confidence=0.9,
    )
    assert extract_index_fingertip(short_hand) is None


def test_index_finger_extended_when_straight_at_various_angles() -> None:
    vertical = _make_hand((0.5, 0.7), (0.5, 0.6), (0.5, 0.5), (0.5, 0.4))
    diagonal = _make_hand((0.4, 0.6), (0.46, 0.54), (0.52, 0.48), (0.58, 0.42))
    horizontal = _make_hand((0.3, 0.5), (0.4, 0.5), (0.5, 0.5), (0.6, 0.5))

    assert is_index_finger_extended(vertical) is True
    assert is_index_finger_extended(diagonal) is True
    assert is_index_finger_extended(horizontal) is True


def test_index_finger_folded_or_bent_returns_false() -> None:
    curled = _make_hand((0.5, 0.6), (0.5, 0.5), (0.5, 0.58), (0.5, 0.66))
    sharp_bend = _make_hand((0.5, 0.6), (0.5, 0.5), (0.6, 0.5), (0.7, 0.52))
    collapsed = _make_hand((0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5))

    assert is_index_finger_extended(curled) is False
    assert is_index_finger_extended(sharp_bend) is False
    assert is_index_finger_extended(collapsed) is False
    assert is_index_finger_extended(None) is False


def test_extract_hand_features_reports_extension_and_loss() -> None:
    extended_hand = _make_hand((0.5, 0.7), (0.5, 0.6), (0.5, 0.5), (0.5, 0.4), confidence=0.91)
    features = extract_hand_features(extended_hand)
    assert features.hand_detected is True
    assert features.index_extended is True
    assert features.index_tip == NormalizedPoint(0.5, 0.4)
    assert features.confidence == 0.91

    lost = extract_hand_features(None)
    assert lost.hand_detected is False
    assert lost.index_extended is False
    assert lost.index_tip is None
    assert lost.palm_scale is None


def test_pinch_ratios_are_palm_scale_invariant() -> None:
    def make_scaled_hand(scale: float) -> HandDetection:
        pts = [NormalizedPoint(0.5, 0.5) for _ in range(21)]
        pts[WRIST] = NormalizedPoint(0.5, 0.5 + 0.2 * scale)
        pts[MIDDLE_MCP] = NormalizedPoint(0.5, 0.5)
        pts[INDEX_MCP] = NormalizedPoint(0.5 - 0.08 * scale, 0.5)
        pts[PINKY_MCP] = NormalizedPoint(0.5 + 0.08 * scale, 0.5)
        pts[THUMB_TIP] = NormalizedPoint(0.5 - 0.02 * scale, 0.5 - 0.1 * scale)
        pts[INDEX_TIP] = NormalizedPoint(0.5 + 0.02 * scale, 0.5 - 0.1 * scale)
        pts[MIDDLE_TIP] = NormalizedPoint(0.5 + 0.06 * scale, 0.5 - 0.2 * scale)
        return HandDetection(landmarks=tuple(pts), handedness="Right", confidence=0.95)

    small_hand = make_scaled_hand(0.5)
    large_hand = make_scaled_hand(1.5)

    assert compute_palm_scale(small_hand) is not None
    small_feat = extract_hand_features(small_hand)
    large_feat = extract_hand_features(large_hand)

    assert small_feat.thumb_index_ratio is not None and large_feat.thumb_index_ratio is not None
    assert abs(small_feat.thumb_index_ratio - large_feat.thumb_index_ratio) < 1e-6
    assert small_feat.thumb_middle_ratio is not None and large_feat.thumb_middle_ratio is not None
    assert abs(small_feat.thumb_middle_ratio - large_feat.thumb_middle_ratio) < 1e-6

