"""Offscreen UI and AppController integration tests without physical hardware."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.app_controller import AppController
from app.config import AppConfig
from app.events import CameraStatus, HandDetection, NormalizedPoint
from ui.main_window import MainWindow


class _MockCapture:
    def __init__(self, opened: bool = True) -> None:
        self._opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray]:
        time.sleep(0.01)
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def set(self, _prop: int, _val: float) -> bool:
        return True

    def release(self) -> None:
        self.released = True


class _MockLandmarker:
    def __init__(self) -> None:
        self.closed = False
        self.calls = 0

    def initialize(self) -> None:
        pass

    def detect(self, _frame_rgb: np.ndarray, _ts: int) -> HandDetection | None:
        self.calls += 1
        if self.calls == 1:
            pts = tuple(NormalizedPoint(x=0.5, y=0.5) for _ in range(21))
            return HandDetection(landmarks=pts, handedness="Right", confidence=0.95)
        return None

    def close(self) -> None:
        self.closed = True


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_window_displays_preview_landmarks_and_stops_cleanly(
    qt_app: QApplication,
    tmp_path,
) -> None:
    model_file = tmp_path / "hand_landmarker.task"
    model_file.write_bytes(b"stub")

    cap = _MockCapture(opened=True)
    landmarker = _MockLandmarker()
    controller = AppController(
        config=AppConfig(model_path=model_file),
        capture_factory=lambda _idx: cap,
        landmarker_factory=lambda: landmarker,  # type: ignore[return-value]
    )
    window = MainWindow(controller=controller)

    assert "Disabled" in window.safety_badge.text()
    assert window.start_stop_button.text() == "Start Camera"

    window.start_stop_button.click()

    deadline = time.monotonic() + 2.0
    while landmarker.calls < 2 and time.monotonic() < deadline:
        qt_app.processEvents()
        time.sleep(0.01)
    qt_app.processEvents()

    assert controller.is_running is True
    assert window.start_stop_button.text() == "Stop Camera"
    assert "Running" in window.camera_status_label.text()

    # Exercise painting both with and without the 21-landmark overlay
    pixmap_with_landmarks = window.camera_view.grab()
    assert not pixmap_with_landmarks.isNull()
    window.landmarks_checkbox.setChecked(False)
    qt_app.processEvents()
    pixmap_without_landmarks = window.camera_view.grab()
    assert not pixmap_without_landmarks.isNull()

    window.close()
    qt_app.processEvents()

    assert controller.is_running is False
    assert cap.released is True
    assert landmarker.closed is True


def test_main_window_shows_error_when_camera_unavailable(
    qt_app: QApplication,
    tmp_path,
) -> None:
    model_file = tmp_path / "hand_landmarker.task"
    model_file.write_bytes(b"stub")

    cap = _MockCapture(opened=False)
    landmarker = _MockLandmarker()
    controller = AppController(
        config=AppConfig(model_path=model_file),
        capture_factory=lambda _idx: cap,
        landmarker_factory=lambda: landmarker,  # type: ignore[return-value]
    )
    window = MainWindow(controller=controller)

    window.start_stop_button.click()

    deadline = time.monotonic() + 2.0
    while controller.camera_status != CameraStatus.ERROR and time.monotonic() < deadline:
        qt_app.processEvents()
        time.sleep(0.01)
    # Process queued signals from AppController
    for _ in range(5):
        qt_app.processEvents()
        time.sleep(0.01)

    assert controller.is_running is False
    assert window.error_label.text() != ""
    assert "Unable to open camera" in window.error_label.text()
    assert window.start_stop_button.text() == "Start Camera"

    window.close()
    qt_app.processEvents()
