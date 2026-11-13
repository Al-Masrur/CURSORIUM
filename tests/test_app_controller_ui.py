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
from vision.landmark_features import (
    INDEX_DIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_MCP,
    THUMB_TIP,
    WRIST,
)


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
    """Emits configurable hand poses across frames for Phase 2 & 3 integration tests."""

    def __init__(self) -> None:
        self.closed = False
        self.calls = 0
        self.mode = "extended"  # "extended" | "folded" | "lost" | "left_pinch" | "right_pinch"
        self.tip_x = 0.5
        self.tip_y = 0.4

    def initialize(self) -> None:
        pass

    def detect(self, _frame_rgb: np.ndarray, _ts: int) -> HandDetection | None:
        self.calls += 1
        if self.mode == "lost":
            return None
        pts = [NormalizedPoint(x=0.5, y=0.75) for _ in range(21)]
        pts[WRIST] = NormalizedPoint(x=0.5, y=0.85)
        pts[MIDDLE_MCP] = NormalizedPoint(x=0.5, y=0.65)
        pts[PINKY_MCP] = NormalizedPoint(x=0.58, y=0.68)
        pts[THUMB_TIP] = NormalizedPoint(x=0.32, y=0.70)
        pts[MIDDLE_TIP] = NormalizedPoint(x=0.56, y=0.20)

        if self.mode == "extended":
            pts[INDEX_MCP] = NormalizedPoint(x=self.tip_x, y=self.tip_y + 0.3)
            pts[INDEX_PIP] = NormalizedPoint(x=self.tip_x, y=self.tip_y + 0.2)
            pts[INDEX_DIP] = NormalizedPoint(x=self.tip_x, y=self.tip_y + 0.1)
            pts[INDEX_TIP] = NormalizedPoint(x=self.tip_x, y=self.tip_y)
        elif self.mode == "left_pinch":
            pts[THUMB_TIP] = NormalizedPoint(x=self.tip_x, y=self.tip_y)
            pts[INDEX_MCP] = NormalizedPoint(x=0.45, y=0.65)
            pts[INDEX_PIP] = NormalizedPoint(x=0.46, y=0.55)
            pts[INDEX_DIP] = NormalizedPoint(x=self.tip_x, y=self.tip_y + 0.04)
            pts[INDEX_TIP] = NormalizedPoint(x=self.tip_x + 0.01, y=self.tip_y)
        elif self.mode == "right_pinch":
            pts[INDEX_MCP] = NormalizedPoint(x=0.45, y=0.65)
            pts[INDEX_PIP] = NormalizedPoint(x=0.45, y=0.55)
            pts[INDEX_DIP] = NormalizedPoint(x=0.45, y=0.45)
            pts[INDEX_TIP] = NormalizedPoint(x=0.45, y=0.35)
            pts[THUMB_TIP] = NormalizedPoint(x=0.50, y=0.55)
            pts[MIDDLE_TIP] = NormalizedPoint(x=0.51, y=0.55)
        elif self.mode == "scroll":
            pts[INDEX_MCP] = NormalizedPoint(x=0.45, y=self.tip_y + 0.3)
            pts[INDEX_PIP] = NormalizedPoint(x=0.45, y=self.tip_y + 0.2)
            pts[INDEX_DIP] = NormalizedPoint(x=0.45, y=self.tip_y + 0.1)
            pts[INDEX_TIP] = NormalizedPoint(x=0.45, y=self.tip_y)
            pts[9] = NormalizedPoint(x=0.50, y=self.tip_y + 0.3)
            pts[10] = NormalizedPoint(x=0.50, y=self.tip_y + 0.2)
            pts[11] = NormalizedPoint(x=0.50, y=self.tip_y + 0.1)
            pts[MIDDLE_TIP] = NormalizedPoint(x=0.50, y=self.tip_y)
            pts[13] = NormalizedPoint(x=0.55, y=0.60)
            pts[14] = NormalizedPoint(x=0.55, y=0.55)
            pts[15] = NormalizedPoint(x=0.55, y=0.62)
            pts[16] = NormalizedPoint(x=0.55, y=0.65)
            pts[17] = NormalizedPoint(x=0.60, y=0.60)
            pts[18] = NormalizedPoint(x=0.60, y=0.55)
            pts[19] = NormalizedPoint(x=0.60, y=0.62)
            pts[20] = NormalizedPoint(x=0.60, y=0.65)
            pts[THUMB_TIP] = NormalizedPoint(x=0.30, y=0.60)
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


def test_phase3_gestures_clicks_drag_safety_and_ui(
    qt_app: QApplication,
    tmp_path,
) -> None:
    model_file = tmp_path / "hand_landmarker.task"
    model_file.write_bytes(b"stub")

    cap = _MockCapture(opened=True)
    landmarker = _ControllableLandmarker()
    mock_backend = MockInputBackend(screen_width=1001, screen_height=1001)

    cfg = AppConfig(
        model_path=model_file,
        cursor_smoothing=0.0,
        pinch_stable_ms=10,
        drag_hold_ms=120,
        double_click_interval_ms=120,
        click_cooldown_ms=40,
    )
    controller = AppController(
        config=cfg,
        capture_factory=lambda _idx: cap,
        landmarker_factory=lambda: landmarker,  # type: ignore[return-value]
        input_backend=mock_backend,
        escape_key_checker=lambda: False,
    )
    window = MainWindow(controller=controller)

    def wait_frames(n: int) -> None:
        target = landmarker.calls + n
        deadline = time.monotonic() + 2.0
        while landmarker.calls < target and time.monotonic() < deadline:
            qt_app.processEvents()
            time.sleep(0.01)
        qt_app.processEvents()

    # Gesture sliders update rules cleanly
    window.pinch_activate_slider.setValue(30)
    window.pinch_release_slider.setValue(48)
    window.drag_hold_slider.setValue(180)
    window.double_click_slider.setValue(160)
    qt_app.processEvents()
    assert controller.config.pinch_activate_ratio == pytest.approx(0.30)
    assert controller.config.pinch_release_ratio == pytest.approx(0.48)
    assert controller.config.drag_hold_ms == 180
    assert controller.config.double_click_interval_ms == 160

    window.start_stop_button.click()
    wait_frames(3)

    # While OS input is disabled, pinches never execute OS clicks
    landmarker.mode = "left_pinch"
    wait_frames(4)
    landmarker.mode = "extended"
    wait_frames(20)
    assert mock_backend.click_history == []
    assert mock_backend.held_buttons == set()

    # Explicitly enable OS control
    window.os_control_button.click()
    qt_app.processEvents()
    assert controller.os_input_enabled is True

    # 1. Single left click (quick pinch then wait > 160ms)
    landmarker.mode = "left_pinch"
    wait_frames(3)
    landmarker.mode = "extended"
    time.sleep(0.20)
    wait_frames(3)
    assert any(btn == "left" and count == 1 for btn, count, _ in mock_backend.click_history)

    # 2. Double-click (two quick pinches within 160ms window)
    landmarker.mode = "left_pinch"
    wait_frames(3)
    landmarker.mode = "extended"
    wait_frames(2)
    landmarker.mode = "left_pinch"
    wait_frames(3)
    landmarker.mode = "extended"
    wait_frames(6)
    assert any(btn == "left" and count == 2 for btn, count, _ in mock_backend.click_history)

    # 3. Right click (thumb-middle pinch)
    landmarker.mode = "right_pinch"
    wait_frames(3)
    landmarker.mode = "extended"
    wait_frames(8)
    assert any(btn == "right" and count == 1 for btn, count, _ in mock_backend.click_history)

    # 4. Hold pinch -> drag start, move, then Esc emergency stop releases held button
    landmarker.mode = "left_pinch"
    time.sleep(0.22)
    wait_frames(4)
    assert mock_backend.held_buttons == {"left"}
    assert "Dragging" in window.tracking_mode_label.text() or "Dragging" in window.gesture_status_label.text()

    window.escape_shortcut.activated.emit()
    qt_app.processEvents()
    assert controller.os_input_enabled is False
    assert mock_backend.held_buttons == set()
    assert ("up", "left") in mock_backend.button_events

    # 5. Re-enable OS control, start drag again, and verify window close releases held button
    window.os_control_button.click()
    qt_app.processEvents()
    landmarker.mode = "left_pinch"
    time.sleep(0.22)
    wait_frames(4)
    assert mock_backend.held_buttons == {"left"}

    window.close()
    qt_app.processEvents()
    assert controller.is_running is False
    assert mock_backend.held_buttons == set()
    assert mock_backend.closed is True


def test_phase4_scroll_pause_controls_and_ui(qt_app: QApplication) -> None:
    config = AppConfig(
        os_input_enabled=True,
        pinch_activate_ratio=0.32,
        pinch_release_ratio=0.45,
        pinch_stable_ms=10,
        drag_hold_ms=150,
        scroll_stable_ms=20,
        scroll_deadzone=0.015,
        scroll_sensitivity=1.0,
    )
    mock_backend = MockInputBackend(screen_width=1920, screen_height=1080)
    capture = _MockCapture(opened=True)
    landmarker = _ControllableLandmarker()

    controller = AppController(
        config=config,
        capture_factory=lambda _idx: capture,
        landmarker_factory=lambda: landmarker,
        input_backend=mock_backend,
        escape_key_checker=lambda: False,
    )
    window = MainWindow(controller=controller)
    window.show()
    qt_app.processEvents()

    # 1. Verify sliders update settings
    window.scroll_sensitivity_slider.setValue(20)
    qt_app.processEvents()
    assert abs(controller.config.scroll_sensitivity - 2.0) < 1e-4
    assert window.scroll_sensitivity_value_label.text() == "2.0x"

    window.scroll_deadzone_slider.setValue(25)
    qt_app.processEvents()
    assert abs(controller.config.scroll_deadzone - 0.025) < 1e-4
    assert window.scroll_deadzone_value_label.text() == "0.025"

    # Start camera
    window.start_stop_button.click()
    qt_app.processEvents()
    time.sleep(0.05)
    qt_app.processEvents()

    # 2. Toggle pause via UI button
    assert controller.is_paused is False
    window.pause_resume_button.click()
    qt_app.processEvents()
    assert controller.is_paused is True
    assert window.pause_resume_button.isChecked() is True
    assert "Resume Tracking" in window.pause_resume_button.text()
    assert "Paused" in window.tracking_mode_label.text()

    # Toggle pause via shortcut P
    window.pause_shortcut.activated.emit()
    qt_app.processEvents()
    assert controller.is_paused is False
    assert window.pause_resume_button.isChecked() is False
    assert "Pause Tracking" in window.pause_resume_button.text()

    window.close()
    qt_app.processEvents()
