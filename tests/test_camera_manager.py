"""Unit tests for CameraManager lifecycle and error handling without hardware."""

from __future__ import annotations

import time

import numpy as np

from app.config import AppConfig
from app.events import CameraStatus
from camera.camera_manager import CameraManager
from camera.frame_buffer import LatestFrameBuffer


class FakeVideoCapture:
    """In-memory VideoCapture double for testing CameraManager."""

    def __init__(
        self,
        opened: bool = True,
        fail_after_reads: int | None = None,
    ) -> None:
        self._opened = opened
        self._fail_after_reads = fail_after_reads
        self.read_count = 0
        self.released = False
        self.props: dict[int, float] = {}

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_count += 1
        if self._fail_after_reads is not None and self.read_count > self._fail_after_reads:
            return False, None
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        time.sleep(0.005)
        return True, frame

    def set(self, prop_id: int, value: float) -> bool:
        self.props[prop_id] = value
        return True

    def release(self) -> None:
        self.released = True


def test_camera_open_failure_reports_error_and_releases() -> None:
    fake_cap = FakeVideoCapture(opened=False)
    statuses: list[CameraStatus] = []
    errors: list[str] = []
    buf = LatestFrameBuffer()

    manager = CameraManager(
        config=AppConfig(camera_index=2),
        frame_buffer=buf,
        capture_factory=lambda _idx: fake_cap,
        on_status=statuses.append,
        on_error=errors.append,
    )
    manager.start()

    deadline = time.monotonic() + 2.0
    while manager.is_running and time.monotonic() < deadline:
        time.sleep(0.01)
    manager.stop()

    assert fake_cap.released is True
    assert manager.status == CameraStatus.ERROR
    assert CameraStatus.ERROR in statuses
    assert len(errors) == 1
    assert "camera index 2" in errors[0].lower()


def test_camera_captures_frames_and_releases_cleanly_on_stop() -> None:
    fake_cap = FakeVideoCapture(opened=True)
    statuses: list[CameraStatus] = []
    errors: list[str] = []
    buf = LatestFrameBuffer()

    manager = CameraManager(
        config=AppConfig(),
        frame_buffer=buf,
        capture_factory=lambda _idx: fake_cap,
        on_status=statuses.append,
        on_error=errors.append,
    )
    manager.start()

    packet = buf.get_latest(last_sequence_id=0, timeout=1.0)
    assert packet is not None
    assert packet.frame_bgr.shape == (48, 64, 3)
    assert packet.sequence_id >= 1

    manager.stop()

    assert errors == []
    assert fake_cap.released is True
    assert not manager.is_running
    assert manager.status == CameraStatus.STOPPED
    assert CameraStatus.RUNNING in statuses


def test_camera_disconnect_triggers_error_and_cleanup() -> None:
    fake_cap = FakeVideoCapture(opened=True, fail_after_reads=2)
    errors: list[str] = []
    buf = LatestFrameBuffer()

    manager = CameraManager(
        config=AppConfig(max_consecutive_read_failures=3),
        frame_buffer=buf,
        capture_factory=lambda _idx: fake_cap,
        on_error=errors.append,
    )
    manager.start()

    deadline = time.monotonic() + 2.0
    while manager.is_running and time.monotonic() < deadline:
        time.sleep(0.01)
    manager.stop()

    assert fake_cap.released is True
    assert manager.status == CameraStatus.ERROR
    assert len(errors) == 1
    assert "disconnected" in errors[0].lower()

