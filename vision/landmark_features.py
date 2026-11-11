"""Pure geometry helpers for extracting landmark features without a webcam."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.events import HandDetection, NormalizedPoint

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
EXPECTED_LANDMARK_COUNT = 21

_MIN_SEGMENT_LENGTH = 1e-4
_MIN_JOINT_COSINE = 0.5
_MIN_STRAIGHTNESS_RATIO = 0.80


@dataclass(frozen=True, slots=True)
class HandFeatures:
    """Extracted Phase 2 features from a single-hand detection."""

    hand_detected: bool
    index_tip: NormalizedPoint | None
    index_extended: bool
    confidence: float


def extract_index_fingertip(detection: HandDetection | None) -> NormalizedPoint | None:
    """Return landmark 8 (index fingertip) if a valid 21-landmark hand is present."""
    if detection is None or len(detection.landmarks) <= INDEX_TIP:
        return None
    return detection.landmarks[INDEX_TIP]


def is_index_finger_extended(detection: HandDetection | None) -> bool:
    """Validate whether the index finger is extended using landmarks 5, 6, 7, and 8.

    Checks three geometric properties in normalized 2D image space so detection is
    invariant to hand rotation:
    1. All three bone segments (5->6, 6->7, 7->8) have non-trivial length.
    2. Successive segments remain aligned (no sharp bend at PIP or DIP).
    3. Distance from MCP (5) grows monotonically along PIP (6) -> DIP (7) -> TIP (8),
       and the chord length (5->8) is at least 80% of the polyline length.
    """
    if detection is None or len(detection.landmarks) <= INDEX_TIP:
        return False

    mcp = detection.landmarks[INDEX_MCP]
    pip = detection.landmarks[INDEX_PIP]
    dip = detection.landmarks[INDEX_DIP]
    tip = detection.landmarks[INDEX_TIP]

    pts = np.array(
        [
            [mcp.x, mcp.y],
            [pip.x, pip.y],
            [dip.x, dip.y],
            [tip.x, tip.y],
        ],
        dtype=np.float64,
    )

    v1 = pts[1] - pts[0]  # 5 -> 6
    v2 = pts[2] - pts[1]  # 6 -> 7
    v3 = pts[3] - pts[2]  # 7 -> 8

    len1 = float(np.linalg.norm(v1))
    len2 = float(np.linalg.norm(v2))
    len3 = float(np.linalg.norm(v3))
    if min(len1, len2, len3) < _MIN_SEGMENT_LENGTH:
        return False

    cos_pip = float(np.dot(v1, v2) / (len1 * len2))
    cos_dip = float(np.dot(v2, v3) / (len2 * len3))
    if cos_pip < _MIN_JOINT_COSINE or cos_dip < _MIN_JOINT_COSINE:
        return False

    dist_mcp_pip = len1
    dist_mcp_dip = float(np.linalg.norm(pts[2] - pts[0]))
    dist_mcp_tip = float(np.linalg.norm(pts[3] - pts[0]))
    if not (dist_mcp_tip > dist_mcp_dip > dist_mcp_pip):
        return False

    polyline_length = len1 + len2 + len3
    straightness = dist_mcp_tip / polyline_length
    return straightness >= _MIN_STRAIGHTNESS_RATIO


def extract_hand_features(detection: HandDetection | None) -> HandFeatures:
    """Extract index fingertip position and extension state from a HandDetection."""
    if detection is None or len(detection.landmarks) < EXPECTED_LANDMARK_COUNT:
        return HandFeatures(
            hand_detected=False,
            index_tip=None,
            index_extended=False,
            confidence=0.0,
        )

    tip = extract_index_fingertip(detection)
    extended = is_index_finger_extended(detection)
    return HandFeatures(
        hand_detected=True,
        index_tip=tip,
        index_extended=extended,
        confidence=detection.confidence,
    )

