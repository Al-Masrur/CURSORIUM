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
from input_control.mock_backend import MockInputBackend
from ui.main_window import MainWindow
from vision.landmark_features import INDEX_DIP, INDEX_MCP, INDEX_PIP, INDEX_TIP


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


class _ControllableLandmarker:
    """Emits configurable hand poses across frames for Phase 2 integration tests."""

    def __init__(self) -> None:
        self.closed = False
        self.calls = 0
        self.mode = "extended"  # "extended" | "folded" | "lost"
        self.tip_x = 0.5
        self.tip_y = 0.4

    def initialize(self) -> None:
        pass

    def detect(self, _frame_rgb: np.ndarray, _ts: int) -> HandDetection | None:
        self.calls += 1
        if self.mode == "lost":
            return None
        pts = [NormalizedPoint(x=0.5, y=0.8) for _ in range(21)]
        if self.mode == "extended":
            pts[INDEX_MCP] = NormalizedPoint(x=self.tip_x, y=self.tip_y + 0.3)
            pts[INDEX_PIP] = NormalizedPoint(x=self.tip_x, y=self.tip_y + 0.2)
            pts[INDEX_DIP] = NormalizedPoint(x=self.tip_x, y=self.tip_y + 0.1)
            pts[INDEX_TIP] = NormalizedPoint(x=self.tip_x, y=self.tip_y)
        else:
            pts[INDEX_MCP] = NormalizedPoint(x=0.5, y=0.6)
            pts[INDEX_PIP] = NormalizedPoint(x=0.5, y=0.5)
            pts[INDEX_DIP] = NormalizedPoint(x=0.5, y=0.58)
            pts[INDEX_TIP] = NormalizedPoint(x=0.5, y=0.66)
        return HandDetection(landmarks=tuple(pts), handedness="Right", confidence=0.96)

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


def test_phase2_cursor_control_calibration_tracking_loss_and_esc_stop(
    qt_app: QApplication,
    tmp_path,
) -> None:
    model_file = tmp_path / "hand_landmarker.task"
    model_file.write_bytes(b"stub")

    cap = _MockCapture(opened=True)
    landmarker = _ControllableLandmarker()
    mock_backend = MockInputBackend(screen_width=1001, screen_height=1001)
    escape_pressed = {"flag": False}

    controller = AppController(
        config=AppConfig(model_path=model_file, cursor_smoothing=0.0),
        capture_factory=lambda _idx: cap,
        landmarker_factory=lambda: landmarker,  # type: ignore[return-value]
        input_backend=mock_backend,
        escape_key_checker=lambda: escape_pressed["flag"],
    )
    window = MainWindow(controller=controller)

    # OS input must be disabled by default
    assert controller.os_input_enabled is False
    assert window.os_control_button.isChecked() is False

    window.start_stop_button.click()
    start_calls = landmarker.calls
    deadline = time.monotonic() + 2.0
    while landmarker.calls < start_calls + 2 and time.monotonic() < deadline:
        qt_app.processEvents()
        time.sleep(0.01)
    qt_app.processEvents()

    # Extended finger computes simulated coordinates without moving real backend
    assert mock_backend.move_history == []
    assert "(sim)" in window.cursor_coords_label.text()

    # Calibration controls update active region
    window.margin_slider.setValue(25)
    window.sensitivity_slider.setValue(15)
    window.smoothing_slider.setValue(0)
    qt_app.processEvents()
    assert controller.active_region.left == pytest.approx(0.25)
    assert controller.coordinate_mapper.sensitivity == pytest.approx(1.5)

    assert window.calibrate_center_button.isEnabled()
    window.calibrate_center_button.click()
    qt_app.processEvents()
    window.reset_calibration_button.click()
    qt_app.processEvents()
    assert controller.active_region.left == pytest.approx(0.15)

    # Explicitly enable OS cursor control
    window.os_control_button.click()
    qt_app.processEvents()
    assert controller.os_input_enabled is True
    assert "OS CONTROL ACTIVE" in window.safety_badge.text()

    landmarker.tip_x = 0.5
    landmarker.tip_y = 0.5
    target_calls = landmarker.calls + 2
    deadline = time.monotonic() + 2.0
    while landmarker.calls < target_calls and time.monotonic() < deadline:
        qt_app.processEvents()
        time.sleep(0.01)
    qt_app.processEvents()

    assert len(mock_backend.move_history) >= 1
    assert mock_backend.last_position == (500, 500)

    # Tracking loss (folded finger / lost hand) stops cursor movement immediately
    landmarker.mode = "lost"
    moves_before_loss = len(mock_backend.move_history)
    target_calls = landmarker.calls + 2
    deadline = time.monotonic() + 2.0
    while landmarker.calls < target_calls and time.monotonic() < deadline:
        qt_app.processEvents()
        time.sleep(0.01)
    qt_app.processEvents()
    assert len(mock_backend.move_history) == moves_before_loss
    assert mock_backend.release_all_count >= 1
    assert controller.coordinate_mapper._smoother.has_state is False

    # Re-acquire hand, then trigger Esc emergency stop
    landmarker.mode = "extended"
    landmarker.tip_x = 0.65
    landmarker.tip_y = 0.5
    target_calls = landmarker.calls + 2
    deadline = time.monotonic() + 2.0
    while landmarker.calls < target_calls and time.monotonic() < deadline:
        qt_app.processEvents()
        time.sleep(0.01)
    qt_app.processEvents()

    escape_pressed["flag"] = True
    deadline = time.monotonic() + 1.5
    while controller.os_input_enabled and time.monotonic() < deadline:
        qt_app.processEvents()
        time.sleep(0.01)
    qt_app.processEvents()

    assert controller.os_input_enabled is False
    assert window.os_control_button.isChecked() is False
    assert "EMERGENCY STOP" in window.safety_badge.text()
    assert controller.is_running is True  # Camera stays running after Esc stop

    # Re-enable OS control, then verify Qt Esc shortcut also triggers emergency stop
    escape_pressed["flag"] = False
    window.os_control_button.click()
    qt_app.processEvents()
    assert controller.os_input_enabled is True
    window.escape_shortcut.activated.emit()
    qt_app.processEvents()
    assert controller.os_input_enabled is False
    assert "EMERGENCY STOP" in window.safety_badge.text()

    window.close()
    qt_app.processEvents()
    assert controller.is_running is False
    assert mock_backend.closed is True
