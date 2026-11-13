"""Pure geometry helpers for extracting landmark features without a webcam."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.events import HandDetection, NormalizedPoint

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20
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
    ring_tip: NormalizedPoint | None = None
    pinky_tip: NormalizedPoint | None = None
    palm_scale: float | None = None
    thumb_index_ratio: float | None = None
    thumb_middle_ratio: float | None = None
    middle_extended: bool = False
    ring_extended: bool = False
    pinky_extended: bool = False
    thumb_extended: bool = False
    scroll_pose: bool = False
    open_palm: bool = False
    scroll_point: NormalizedPoint | None = None


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


def is_finger_extended_by_indices(
    detection: HandDetection | None,
    mcp_idx: int,
    pip_idx: int,
    dip_idx: int,
    tip_idx: int,
) -> bool:
    """Validate whether a 4-joint finger is extended using geometry checks in 2D space."""
    if detection is None or len(detection.landmarks) <= tip_idx:
        return False

    landmarks = detection.landmarks
    mcp = landmarks[mcp_idx]
    pip = landmarks[pip_idx]
    dip = landmarks[dip_idx]
    tip = landmarks[tip_idx]

    pts = np.array(
        [
            [mcp.x, mcp.y],
            [pip.x, pip.y],
            [dip.x, dip.y],
            [tip.x, tip.y],
        ],
        dtype=np.float64,
    )

    v1 = pts[1] - pts[0]  # MCP -> PIP
    v2 = pts[2] - pts[1]  # PIP -> DIP
    v3 = pts[3] - pts[2]  # DIP -> TIP

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


def is_index_finger_extended(detection: HandDetection | None) -> bool:
    """Validate whether the index finger is extended using landmarks 5, 6, 7, and 8."""
    return is_finger_extended_by_indices(
        detection, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP
    )


def is_middle_finger_extended(detection: HandDetection | None) -> bool:
    """Validate whether the middle finger is extended using landmarks 9, 10, 11, and 12."""
    return is_finger_extended_by_indices(
        detection, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP
    )


def is_ring_finger_extended(detection: HandDetection | None) -> bool:
    """Validate whether the ring finger is extended using landmarks 13, 14, 15, and 16."""
    return is_finger_extended_by_indices(
        detection, RING_MCP, RING_PIP, RING_DIP, RING_TIP
    )


def is_pinky_finger_extended(detection: HandDetection | None) -> bool:
    """Validate whether the pinky finger is extended using landmarks 17, 18, 19, and 20."""
    return is_finger_extended_by_indices(
        detection, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP
    )


def is_thumb_finger_extended(detection: HandDetection | None) -> bool:
    """Validate whether the thumb is extended away from the wrist/palm."""
    if detection is None or len(detection.landmarks) < EXPECTED_LANDMARK_COUNT:
        return False
    landmarks = detection.landmarks
    wrist = landmarks[WRIST]
    thumb_mcp = landmarks[THUMB_MCP]
    thumb_ip = landmarks[THUMB_IP]
    thumb_tip = landmarks[THUMB_TIP]

    dist_wrist_tip = euclidean_distance_2d(wrist, thumb_tip)
    dist_wrist_ip = euclidean_distance_2d(wrist, thumb_ip)
    dist_mcp_tip = euclidean_distance_2d(thumb_mcp, thumb_tip)
    dist_mcp_ip = euclidean_distance_2d(thumb_mcp, thumb_ip)

    return dist_wrist_tip > dist_wrist_ip and dist_mcp_tip > dist_mcp_ip


def extract_hand_features(detection: HandDetection | None) -> HandFeatures:
    """Extract fingertip positions, palm-normalized pinch ratios, and finger extension states."""
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
    ring_tip = landmarks[RING_TIP]
    pinky_tip = landmarks[PINKY_TIP]

    index_ext = is_index_finger_extended(detection)
    middle_ext = is_middle_finger_extended(detection)
    ring_ext = is_ring_finger_extended(detection)
    pinky_ext = is_pinky_finger_extended(detection)
    thumb_ext = is_thumb_finger_extended(detection)

    palm_scale = compute_palm_scale(detection)
    if palm_scale is not None:
        thumb_index_ratio = euclidean_distance_2d(thumb_tip, index_tip) / palm_scale
        thumb_middle_ratio = euclidean_distance_2d(thumb_tip, middle_tip) / palm_scale
    else:
        thumb_index_ratio = None
        thumb_middle_ratio = None

    # Pinch conflict guard: if thumb is actively pinching index or middle, it's not a pure scroll or open palm
    is_pinching = (
        (thumb_index_ratio is not None and thumb_index_ratio < 0.40)
        or (thumb_middle_ratio is not None and thumb_middle_ratio < 0.40)
    )

    # Scroll pose: Index + Middle extended, Ring + Pinky folded, not pinching
    scroll_pose = (
        index_ext
        and middle_ext
        and not ring_ext
        and not pinky_ext
        and not is_pinching
    )

    scroll_point: NormalizedPoint | None = None
    if scroll_pose:
        scroll_point = NormalizedPoint(
            x=(index_tip.x + middle_tip.x) / 2.0,
            y=(index_tip.y + middle_tip.y) / 2.0,
            z=(index_tip.z + middle_tip.z) / 2.0,
        )

    # Open palm: all 4 non-thumb fingers extended, thumb extended, not pinching
    open_palm = (
        index_ext
        and middle_ext
        and ring_ext
        and pinky_ext
        and thumb_ext
        and not is_pinching
    )

    return HandFeatures(
        hand_detected=True,
        index_tip=index_tip,
        index_extended=index_ext,
        confidence=detection.confidence,
        thumb_tip=thumb_tip,
        middle_tip=middle_tip,
        ring_tip=ring_tip,
        pinky_tip=pinky_tip,
        palm_scale=palm_scale,
        thumb_index_ratio=thumb_index_ratio,
        thumb_middle_ratio=thumb_middle_ratio,
        middle_extended=middle_ext,
        ring_extended=ring_ext,
        pinky_extended=pinky_ext,
        thumb_extended=thumb_ext,
        scroll_pose=scroll_pose,
        open_palm=open_palm,
        scroll_point=scroll_point,
    )
