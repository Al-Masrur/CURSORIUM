"""Minimal Phase 1 desktop window for HandPilot."""

from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.app_controller import AppController
from app.events import CameraStatus, TrackingResult
from ui.camera_view import CameraView


class MainWindow(QMainWindow):
    """Minimal Phase 1 window displaying webcam feed, 21 landmarks, and status."""

    def __init__(
        self,
        controller: AppController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self.setWindowTitle("HandPilot — Phase 1 (Webcam & Hand Tracking)")
        self.resize(820, 640)

        self._build_ui()
        self._connect_signals()
        self._sync_initial_state()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        # Controls bar
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)

        self.start_stop_button = QPushButton("Start Camera", central)
        self.start_stop_button.setMinimumWidth(120)

        camera_label = QLabel("Camera:", central)
        self.camera_selector = QComboBox(central)
        for index in range(4):
            self.camera_selector.addItem(f"Camera {index}", index)

        self.mirror_checkbox = QCheckBox("Mirror Preview", central)
        self.landmarks_checkbox = QCheckBox("Show Landmarks", central)

        self.safety_badge = QLabel("OS Input: Disabled (Preview Only)", central)
        self.safety_badge.setStyleSheet(
            "padding: 4px 8px; border-radius: 4px; "
            "background-color: #1e293b; color: #93c5fd; font-weight: 600;"
        )

        controls_layout.addWidget(self.start_stop_button)
        controls_layout.addWidget(camera_label)
        controls_layout.addWidget(self.camera_selector)
        controls_layout.addWidget(self.mirror_checkbox)
        controls_layout.addWidget(self.landmarks_checkbox)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.safety_badge)

        # Live camera preview
        self.camera_view = CameraView(central)

        # Status bar area inside layout
        status_layout = QHBoxLayout()
        status_layout.setSpacing(16)

        self.camera_status_label = QLabel("Camera: Stopped", central)
        self.hand_status_label = QLabel("Hand: Not detected", central)
        self.fps_label = QLabel("FPS: --", central)

        status_layout.addWidget(self.camera_status_label)
        status_layout.addWidget(self.hand_status_label)
        status_layout.addStretch(1)
        status_layout.addWidget(self.fps_label)

        self.error_label = QLabel("", central)
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #f87171; font-weight: 500;")
        self.error_label.hide()

        root_layout.addLayout(controls_layout)
        root_layout.addWidget(self.camera_view, stretch=1)
        root_layout.addLayout(status_layout)
        root_layout.addWidget(self.error_label)

        self.setCentralWidget(central)

    def _connect_signals(self) -> None:
        self.start_stop_button.clicked.connect(self._on_start_stop_clicked)
        self.camera_selector.currentIndexChanged.connect(self._on_camera_changed)
        self.mirror_checkbox.toggled.connect(self._controller.set_mirror_preview)
        self.landmarks_checkbox.toggled.connect(self._on_landmarks_toggled)

        self._controller.preview_available.connect(self._on_preview_available)
        self._controller.status_changed.connect(self._on_status_changed)
        self._controller.error_occurred.connect(self._on_error_occurred)

    def _sync_initial_state(self) -> None:
        cfg = self._controller.config
        self.camera_selector.blockSignals(True)
        self.mirror_checkbox.blockSignals(True)
        self.landmarks_checkbox.blockSignals(True)
        try:
            idx = self.camera_selector.findData(cfg.camera_index)
            if idx >= 0:
                self.camera_selector.setCurrentIndex(idx)
            self.mirror_checkbox.setChecked(cfg.mirror_preview)
            self.landmarks_checkbox.setChecked(cfg.show_landmarks)
            self.camera_view.set_show_landmarks(cfg.show_landmarks)
        finally:
            self.camera_selector.blockSignals(False)
            self.mirror_checkbox.blockSignals(False)
            self.landmarks_checkbox.blockSignals(False)

    @Slot()
    def _on_start_stop_clicked(self) -> None:
        if self._controller.is_running:
            self._controller.stop()
        else:
            self.error_label.hide()
            self.error_label.setText("")
            self._controller.start()

    @Slot(int)
    def _on_camera_changed(self, combo_index: int) -> None:
        camera_index = self.camera_selector.itemData(combo_index)
        if isinstance(camera_index, int):
            self._controller.set_camera_index(camera_index)

    @Slot(bool)
    def _on_landmarks_toggled(self, enabled: bool) -> None:
        self._controller.set_show_landmarks(enabled)
        self.camera_view.set_show_landmarks(enabled)

    @Slot()
    def _on_preview_available(self) -> None:
        item = self._controller.pop_latest_preview()
        if item is None:
            return
        frame_rgb, result = item
        self._apply_tracking_result(frame_rgb, result)

    def _apply_tracking_result(self, frame_rgb, result: TrackingResult) -> None:
        self.camera_view.update_frame(frame_rgb, result.detection)

        if result.hand_detected and result.detection is not None:
            det = result.detection
            self.hand_status_label.setText(
                f"Hand: Detected ({det.handedness}, {det.confidence:.0%}, "
                f"{len(det.landmarks)} landmarks)"
            )
            self.hand_status_label.setStyleSheet("color: #4ade80; font-weight: 600;")
        else:
            self.hand_status_label.setText("Hand: Lost / not detected")
            self.hand_status_label.setStyleSheet("color: #cbd5e1;")

        self.fps_label.setText(
            f"Camera: {result.camera_fps:.1f} FPS | Tracking: {result.tracking_fps:.1f} FPS"
        )

    @Slot(object)
    def _on_status_changed(self, status: CameraStatus) -> None:
        self.camera_status_label.setText(f"Camera: {status.value}")
        if status in (CameraStatus.STARTING, CameraStatus.RUNNING):
            self.start_stop_button.setText("Stop Camera")
            if status == CameraStatus.STARTING:
                self.camera_view.clear("Starting camera...")
        else:
            self.start_stop_button.setText("Start Camera")
            self.hand_status_label.setText("Hand: Not detected")
            self.hand_status_label.setStyleSheet("")
            self.fps_label.setText("FPS: --")
            if status == CameraStatus.STOPPED:
                self.camera_view.clear(
                    "Camera is stopped.\nClick 'Start Camera' to preview webcam & 21 hand landmarks."
                )

    @Slot(str)
    def _on_error_occurred(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.show()
        self.camera_view.clear(f"Camera Error:\n{message}")

    def closeEvent(self, event: QCloseEvent) -> None:
        self._controller.shutdown()
        super().closeEvent(event)
