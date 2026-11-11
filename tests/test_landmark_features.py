"""Unit tests for landmark 8 extraction and index finger extension geometry."""

from __future__ import annotations

from app.events import HandDetection, NormalizedPoint
from vision.landmark_features import (
    INDEX_DIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
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

