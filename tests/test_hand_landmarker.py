"""Unit and smoke tests for MediaPipe HandLandmarkerService."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from app.config import DEFAULT_MODEL_PATH, AppConfig
from vision.hand_landmarker import HAND_CONNECTIONS, HandLandmarkerService


@dataclass
class _FakeLandmark:
    x: float
    y: float
    z: float = 0.0


@dataclass
class _FakeCategory:
    category_name: str
    score: float


@dataclass
class _FakeResult:
    hand_landmarks: list[list[_FakeLandmark]]
    handedness: list[list[_FakeCategory]]


class _FakeDetector:
    def __init__(self, return_hand: bool = True) -> None:
        self.return_hand = return_hand
        self.timestamps: list[int] = []
        self.closed = False

    def detect_for_video(self, image, timestamp_ms: int) -> _FakeResult:
        self.timestamps.append(timestamp_ms)
        if not self.return_hand:
            return _FakeResult(hand_landmarks=[], handedness=[])
        landmarks = [_FakeLandmark(x=i / 20.0, y=0.5, z=-0.01) for i in range(21)]
        return _FakeResult(
            hand_landmarks=[landmarks],
            handedness=[[_FakeCategory(category_name="Right", score=0.96)]],
        )

    def close(self) -> None:
        self.closed = True


def test_hand_connections_has_21_edges() -> None:
    assert len(HAND_CONNECTIONS) == 21


def test_missing_model_raises_file_not_found(tmp_path: Path) -> None:
    cfg = AppConfig(model_path=tmp_path / "missing.task")
    service = HandLandmarkerService(cfg)
    with pytest.raises(FileNotFoundError):
        service.initialize()


def test_detect_extracts_21_landmarks_and_enforces_monotonic_timestamps() -> None:
    detector = _FakeDetector(return_hand=True)
    service = HandLandmarkerService(AppConfig(), detector=detector)
    frame_rgb = np.zeros((64, 64, 3), dtype=np.uint8)

    det1 = service.detect(frame_rgb, 100)
    det2 = service.detect(frame_rgb, 100)  # non-increasing timestamp should bump to 101

    assert det1 is not None
    assert len(det1.landmarks) == 21
    assert det1.handedness == "Right"
    assert det1.confidence == pytest.approx(0.96)
    assert det2 is not None
    assert detector.timestamps == [100, 101]

    service.close()
    assert detector.closed is True


def test_detect_returns_none_when_no_hand_present() -> None:
    detector = _FakeDetector(return_hand=False)
    service = HandLandmarkerService(AppConfig(), detector=detector)
    frame_rgb = np.zeros((64, 64, 3), dtype=np.uint8)

    assert service.detect(frame_rgb, 10) is None
    service.close()


def test_real_mediapipe_model_smoke() -> None:
    assert DEFAULT_MODEL_PATH.is_file(), f"Missing model asset at {DEFAULT_MODEL_PATH}"
    service = HandLandmarkerService(AppConfig())
    service.initialize()
    try:
        blank_rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        result = service.detect(blank_rgb, 1)
        assert result is None
    finally:
        service.close()

