"""Unit tests for VisionWorker frame consumption, mirroring, and hand-loss updates."""

from __future__ import annotations

import threading
import time

import numpy as np

from app.config import AppConfig
from app.events import FramePacket, HandDetection, NormalizedPoint, TrackingResult
from camera.frame_buffer import LatestFrameBuffer
from vision.vision_worker import VisionWorker


class FakeLandmarkerService:
    """Stub HandLandmarkerService returning deterministic hand/no-hand sequences."""

    def __init__(self, fail_on_detect: bool = False) -> None:
        self.fail_on_detect = fail_on_detect
        self.initialized = False
        self.closed = False
        self.calls = 0

    def initialize(self) -> None:
        self.initialized = True

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> HandDetection | None:
        self.calls += 1
        if self.fail_on_detect:
            raise RuntimeError("simulated inference error")
        if self.calls == 1:
            points = tuple(NormalizedPoint(x=0.5, y=0.5) for _ in range(21))
            return HandDetection(landmarks=points, handedness="Right", confidence=0.92)
        return None

    def close(self) -> None:
        self.closed = True


def test_vision_worker_mirrors_and_reports_hand_detected_then_lost() -> None:
    buf = LatestFrameBuffer()
    fake_landmarker = FakeLandmarkerService()
    results: list[tuple[np.ndarray, TrackingResult]] = []
    done = threading.Event()

    def on_result(frame_rgb: np.ndarray, tracking: TrackingResult) -> None:
        results.append((frame_rgb, tracking))
        if len(results) >= 2:
            done.set()

    cfg = AppConfig(mirror_preview=True)
    worker = VisionWorker(
        config=cfg,
        frame_buffer=buf,
        landmarker_factory=lambda: fake_landmarker,  # type: ignore[return-value]
        camera_fps_provider=lambda: 30.0,
        on_result=on_result,
    )
    worker.start()

    # Left half black (0), right half red in BGR ([0, 0, 255]) -> mirrored becomes left half red in RGB ([255, 0, 0])
    frame_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    frame_bgr[:, 5:] = (0, 0, 255)
    buf.put(FramePacket(frame_bgr=frame_bgr, timestamp_ms=10, sequence_id=1))

    deadline = time.monotonic() + 1.0
    while fake_landmarker.calls < 1 and time.monotonic() < deadline:
        time.sleep(0.01)

    buf.put(FramePacket(frame_bgr=frame_bgr, timestamp_ms=43, sequence_id=2))
    assert done.wait(timeout=1.5)

    worker.stop()

    assert fake_landmarker.initialized is True
    assert fake_landmarker.closed is True
    assert len(results) == 2

    first_rgb, first_tracking = results[0]
    assert tuple(first_rgb[0, 0]) == (255, 0, 0)  # Mirrored + BGR->RGB verified
    assert first_tracking.hand_detected is True
    assert first_tracking.detection is not None
    assert len(first_tracking.detection.landmarks) == 21

    _, second_tracking = results[1]
    assert second_tracking.hand_detected is False
    assert second_tracking.detection is None


def test_vision_worker_reports_error_and_closes_landmarker() -> None:
    buf = LatestFrameBuffer()
    fake_landmarker = FakeLandmarkerService(fail_on_detect=True)
    errors: list[str] = []

    worker = VisionWorker(
        config=AppConfig(),
        frame_buffer=buf,
        landmarker_factory=lambda: fake_landmarker,  # type: ignore[return-value]
        on_error=errors.append,
    )
    worker.start()
    buf.put(FramePacket(frame_bgr=np.zeros((8, 8, 3), dtype=np.uint8), timestamp_ms=1, sequence_id=1))

    deadline = time.monotonic() + 1.5
    while worker.is_running and time.monotonic() < deadline:
        time.sleep(0.01)
    worker.stop()

    assert fake_landmarker.closed is True
    assert len(errors) == 1
    assert "simulated inference error" in errors[0]

