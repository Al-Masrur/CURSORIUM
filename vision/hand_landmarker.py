"""MediaPipe Hand Landmarker Tasks API wrapper for single-hand tracking."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from app.config import AppConfig
from app.events import HandDetection, NormalizedPoint

logger = logging.getLogger(__name__)

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = tuple(
    (conn.start, conn.end) for conn in vision.HandLandmarksConnections.HAND_CONNECTIONS
)


class HandLandmarkerLike(Protocol):
    """Minimal protocol for MediaPipe HandLandmarker to allow test doubles."""

    def detect_for_video(self, image: Any, timestamp_ms: int) -> Any: ...
    def close(self) -> None: ...


class HandLandmarkerService:
    """Wraps MediaPipe's HandLandmarker Tasks API in VIDEO running mode."""

    def __init__(
        self,
        config: AppConfig,
        detector: HandLandmarkerLike | None = None,
    ) -> None:
        self._config = config
        self._detector = detector
        self._last_timestamp_ms: int = -1
        self._closed: bool = False

    def initialize(self) -> None:
        """Create the underlying MediaPipe HandLandmarker if not already created."""
        if self._detector is not None:
            return
        model_path = self._config.model_path
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Hand landmarker model not found at '{model_path}'. "
                "Place hand_landmarker.task there as documented in assets/models/README.md."
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self._config.num_hands,
            min_hand_detection_confidence=self._config.min_hand_detection_confidence,
            min_hand_presence_confidence=self._config.min_hand_presence_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._closed = False
        self._last_timestamp_ms = -1
        logger.info("Initialized MediaPipe HandLandmarker (VIDEO mode)")

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> HandDetection | None:
        """Detect up to one hand in an RGB frame and return 21 normalized landmarks."""
        if self._closed:
            raise RuntimeError("HandLandmarkerService has been closed")
        if self._detector is None:
            self.initialize()

        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        if frame_rgb.dtype == np.uint8 and frame_rgb.flags["C_CONTIGUOUS"]:
            contiguous_rgb = frame_rgb
        else:
            contiguous_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=contiguous_rgb)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)
        return self._extract_primary_hand(result)

    @staticmethod
    def _extract_primary_hand(result: Any) -> HandDetection | None:
        hand_landmarks_list = getattr(result, "hand_landmarks", None)
        if not hand_landmarks_list:
            return None

        raw_landmarks = hand_landmarks_list[0]
        points = tuple(
            NormalizedPoint(
                x=float(lm.x),
                y=float(lm.y),
                z=float(getattr(lm, "z", 0.0)),
            )
            for lm in raw_landmarks
        )

        handedness_label = "Unknown"
        confidence = 0.0
        handedness_list = getattr(result, "handedness", None)
        if handedness_list and handedness_list[0]:
            top_category = handedness_list[0][0]
            handedness_label = (
                getattr(top_category, "category_name", None)
                or getattr(top_category, "display_name", None)
                or "Unknown"
            )
            confidence = float(getattr(top_category, "score", 0.0))

        return HandDetection(
            landmarks=points,
            handedness=handedness_label,
            confidence=confidence,
        )

    def close(self) -> None:
        """Release MediaPipe HandLandmarker native resources."""
        if self._closed:
            return
        self._closed = True
        if self._detector is not None:
            try:
                self._detector.close()
            finally:
                self._detector = None
            logger.info("Closed MediaPipe HandLandmarker")

