"""Pure geometry helpers for extracting landmark features without a webcam."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.events import HandDetection, NormalizedPoint

WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
PINKY_MCP = 17
EXPECTED_LANDMARK_COUNT = 21

_MIN_SEGMENT_LENGTH = 1e-4
_MIN_PALM_SCALE = 1e-4
_MIN_JOINT_COSINE = 0.5
_MIN_STRAIGHTNESS_RATIO = 0.80


@dataclass(frozen=True, slots=True)
class HandFeatures:
    """Extracted hand geometry features from a single-hand detection."""

    hand_detected: bool
    index_tip: NormalizedPoint | None
    index_extended: bool
    confidence: float
    thumb_tip: NormalizedPoint | None = None
    middle_tip: NormalizedPoint | None = None
    palm_scale: float | None = None
    thumb_index_ratio: float | None = None
    thumb_middle_ratio: float | None = None


def euclidean_distance_2d(a: NormalizedPoint, b: NormalizedPoint) -> float:
    """Compute 2D Euclidean distance between two normalized landmarks."""
    return math.hypot(a.x - b.x, a.y - b.y)


def extract_index_fingertip(detection: HandDetection | None) -> NormalizedPoint | None:
    """Return landmark 8 (index fingertip) if a valid 21-landmark hand is present."""
    if detection is None or len(detection.landmarks) <= INDEX_TIP:
        return None
    return detection.landmarks[INDEX_TIP]


def compute_palm_scale(detection: HandDetection | None) -> float | None:
    """Compute a stable palm-size reference from wrist->middle_mcp and palm width."""
    if detection is None or len(detection.landmarks) < EXPECTED_LANDMARK_COUNT:
        return None
    landmarks = detection.landmarks
    wrist_to_middle = euclidean_distance_2d(landmarks[WRIST], landmarks[MIDDLE_MCP])
    palm_width = euclidean_distance_2d(landmarks[INDEX_MCP], landmarks[PINKY_MCP])
    scale = max(wrist_to_middle, palm_width)
    if scale < _MIN_PALM_SCALE:
        return None
    return scale


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
    """Extract fingertip positions, palm-normalized pinch ratios, and index extension."""
    if detection is None or len(detection.landmarks) < EXPECTED_LANDMARK_COUNT:
        return HandFeatures(
            hand_detected=False,
            index_tip=None,
            index_extended=False,
            confidence=0.0,
        )

    landmarks = detection.landmarks
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_TIP]
    middle_tip = landmarks[MIDDLE_TIP]
    extended = is_index_finger_extended(detection)

    palm_scale = compute_palm_scale(detection)
    if palm_scale is not None:
        thumb_index_ratio = euclidean_distance_2d(thumb_tip, index_tip) / palm_scale
        thumb_middle_ratio = euclidean_distance_2d(thumb_tip, middle_tip) / palm_scale
    else:
        thumb_index_ratio = None
        thumb_middle_ratio = None

    return HandFeatures(
        hand_detected=True,
        index_tip=index_tip,
        index_extended=extended,
        confidence=detection.confidence,
        thumb_tip=thumb_tip,
        middle_tip=middle_tip,
        palm_scale=palm_scale,
        thumb_index_ratio=thumb_index_ratio,
        thumb_middle_ratio=thumb_middle_ratio,
    )
