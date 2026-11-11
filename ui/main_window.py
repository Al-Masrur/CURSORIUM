"""Desktop window for HandPilot Phase 2 (webcam, tracking, calibration, and cursor control)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.app_controller import AppController
from app.events import CameraStatus, ControlMode, TrackingResult
from ui.camera_view import CameraView
from vision.coordinate_mapper import ActiveRegion


class MainWindow(QMainWindow):
    """Desktop interface for live webcam tracking, calibration, and safe cursor control."""

    def __init__(
        self,
        controller: AppController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self.setWindowTitle("HandPilot — Phase 2 (Virtual Mouse Movement)")
        self.resize(940, 740)

        self._build_ui()
        self._connect_signals()
        self._sync_initial_state()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        # Top controls bar
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)

        self.start_stop_button = QPushButton("Start Camera", central)
        self.start_stop_button.setMinimumWidth(120)

        self.os_control_button = QPushButton("Enable OS Cursor Control", central)
        self.os_control_button.setCheckable(True)
        self.os_control_button.setMinimumWidth(185)

        camera_label = QLabel("Camera:", central)
        self.camera_selector = QComboBox(central)
        for index in range(4):
            self.camera_selector.addItem(f"Camera {index}", index)

        self.mirror_checkbox = QCheckBox("Mirror Preview", central)
        self.landmarks_checkbox = QCheckBox("Show Landmarks", central)

        self.safety_badge = QLabel("OS Input: Disabled (Esc = Emergency Stop)", central)

        controls_layout.addWidget(self.start_stop_button)
        controls_layout.addWidget(self.os_control_button)
        controls_layout.addWidget(camera_label)
        controls_layout.addWidget(self.camera_selector)
        controls_layout.addWidget(self.mirror_checkbox)
        controls_layout.addWidget(self.landmarks_checkbox)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.safety_badge)

        # Live camera preview
        self.camera_view = CameraView(central)

        # Calibration & motion settings panel
        settings_group = QGroupBox("Calibration & Cursor Settings", central)
        settings_layout = QHBoxLayout(settings_group)
        settings_layout.setSpacing(16)

        # Sensitivity slider (0.4x .. 2.5x mapped from 4 .. 25)
        sens_box = QHBoxLayout()
        sens_title = QLabel("Sensitivity:", settings_group)
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal, settings_group)
        self.sensitivity_slider.setRange(4, 25)
        self.sensitivity_value_label = QLabel("1.0x", settings_group)
        self.sensitivity_value_label.setMinimumWidth(38)
        sens_box.addWidget(sens_title)
        sens_box.addWidget(self.sensitivity_slider)
        sens_box.addWidget(self.sensitivity_value_label)

        # Smoothing slider (0% .. 90%)
        smooth_box = QHBoxLayout()
        smooth_title = QLabel("Smoothing:", settings_group)
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal, settings_group)
        self.smoothing_slider.setRange(0, 90)
        self.smoothing_value_label = QLabel("35%", settings_group)
        self.smoothing_value_label.setMinimumWidth(36)
        smooth_box.addWidget(smooth_title)
        smooth_box.addWidget(self.smoothing_slider)
        smooth_box.addWidget(self.smoothing_value_label)

        # Active region margin slider (5% .. 35%)
        margin_box = QHBoxLayout()
        margin_title = QLabel("Region Margin:", settings_group)
        self.margin_slider = QSlider(Qt.Orientation.Horizontal, settings_group)
        self.margin_slider.setRange(5, 35)
        self.margin_value_label = QLabel("15%", settings_group)
        self.margin_value_label.setMinimumWidth(36)
        margin_box.addWidget(margin_title)
        margin_box.addWidget(self.margin_slider)
        margin_box.addWidget(self.margin_value_label)

        self.calibrate_center_button = QPushButton("Center on Hand", settings_group)
        self.reset_calibration_button = QPushButton("Reset Region", settings_group)

        settings_layout.addLayout(sens_box, stretch=1)
        settings_layout.addLayout(smooth_box, stretch=1)
        settings_layout.addLayout(margin_box, stretch=1)
        settings_layout.addWidget(self.calibrate_center_button)
        settings_layout.addWidget(self.reset_calibration_button)

        # Status bar area inside layout
        status_layout = QHBoxLayout()
        status_layout.setSpacing(16)

        self.camera_status_label = QLabel("Camera: Stopped", central)
        self.hand_status_label = QLabel("Hand: Not detected", central)
        self.tracking_mode_label = QLabel("Mode: Preview Only", central)
        self.cursor_coords_label = QLabel("Cursor: --", central)
        self.fps_label = QLabel("FPS: --", central)

        status_layout.addWidget(self.camera_status_label)
        status_layout.addWidget(self.hand_status_label)
        status_layout.addWidget(self.tracking_mode_label)
        status_layout.addWidget(self.cursor_coords_label)
        status_layout.addStretch(1)
        status_layout.addWidget(self.fps_label)

        self.notice_label = QLabel(
            "Tip: Extend your index finger to steer the cursor. Press Esc anytime to stop OS control.",
            central,
        )
        self.notice_label.setStyleSheet("color: #94a3b8;")

        self.error_label = QLabel("", central)
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #f87171; font-weight: 500;")
        self.error_label.hide()

        root_layout.addLayout(controls_layout)
        root_layout.addWidget(self.camera_view, stretch=1)
        root_layout.addWidget(settings_group)
        root_layout.addLayout(status_layout)
        root_layout.addWidget(self.notice_label)
        root_layout.addWidget(self.error_label)

        self.setCentralWidget(central)

        # In-window Esc shortcut complements global Windows Esc polling.
        self.escape_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)

    def _connect_signals(self) -> None:
        self.start_stop_button.clicked.connect(self._on_start_stop_clicked)
        self.os_control_button.toggled.connect(self._on_os_control_toggled)
        self.camera_selector.currentIndexChanged.connect(self._on_camera_changed)
        self.mirror_checkbox.toggled.connect(self._controller.set_mirror_preview)
        self.landmarks_checkbox.toggled.connect(self._on_landmarks_toggled)

        self.sensitivity_slider.valueChanged.connect(self._on_sensitivity_changed)
        self.smoothing_slider.valueChanged.connect(self._on_smoothing_changed)
        self.margin_slider.valueChanged.connect(self._on_margin_changed)
        self.calibrate_center_button.clicked.connect(self._on_calibrate_center_clicked)
        self.reset_calibration_button.clicked.connect(self._on_reset_calibration_clicked)

        self.escape_shortcut.activated.connect(self._controller.emergency_stop)

        self._controller.preview_available.connect(self._on_preview_available)
        self._controller.status_changed.connect(self._on_status_changed)
        self._controller.error_occurred.connect(self._on_error_occurred)
        self._controller.os_control_changed.connect(self._on_os_control_changed)
        self._controller.emergency_stopped.connect(self._on_emergency_stopped)
        self._controller.calibration_changed.connect(self._on_calibration_changed)

    def _sync_initial_state(self) -> None:
        cfg = self._controller.config
        widgets = (
            self.camera_selector,
            self.mirror_checkbox,
            self.landmarks_checkbox,
            self.os_control_button,
            self.sensitivity_slider,
            self.smoothing_slider,
            self.margin_slider,
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            idx = self.camera_selector.findData(cfg.camera_index)
            if idx >= 0:
                self.camera_selector.setCurrentIndex(idx)
            self.mirror_checkbox.setChecked(cfg.mirror_preview)
            self.landmarks_checkbox.setChecked(cfg.show_landmarks)
            self.camera_view.set_show_landmarks(cfg.show_landmarks)

            self.os_control_button.setChecked(cfg.os_input_enabled)
            self._update_os_control_indicator(cfg.os_input_enabled, emergency=False)

            sens_val = int(round(cfg.cursor_sensitivity * 10))
            self.sensitivity_slider.setValue(sens_val)
            self.sensitivity_value_label.setText(f"{cfg.cursor_sensitivity:.1f}x")

            smooth_pct = int(round(cfg.cursor_smoothing * 100))
            self.smoothing_slider.setValue(smooth_pct)
            self.smoothing_value_label.setText(f"{smooth_pct}%")

            margin_pct = int(round(cfg.active_region_left * 100))
            self.margin_slider.setValue(max(5, min(35, margin_pct)))
            self.margin_value_label.setText(f"{self.margin_slider.value()}%")

            self.camera_view.set_active_region(self._controller.active_region)
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    def _update_os_control_indicator(self, active: bool, *, emergency: bool) -> None:
        self.camera_view.set_os_control_active(active)
        if active:
            self.os_control_button.setText("Disable OS Cursor Control")
            self.safety_badge.setText("OS CONTROL ACTIVE — Press Esc to Stop")
            self.safety_badge.setStyleSheet(
                "padding: 4px 10px; border-radius: 4px; "
                "background-color: #991b1b; color: #fef2f2; font-weight: 700;"
            )
            self.tracking_mode_label.setText("Mode: OS Control Active")
        elif emergency:
            self.os_control_button.setText("Enable OS Cursor Control")
            self.safety_badge.setText("OS Input: Disabled — EMERGENCY STOP (Esc)")
            self.safety_badge.setStyleSheet(
                "padding: 4px 10px; border-radius: 4px; "
                "background-color: #92400e; color: #fef3c7; font-weight: 700;"
            )
            self.tracking_mode_label.setText("Mode: Emergency Stopped (Esc)")
        else:
            self.os_control_button.setText("Enable OS Cursor Control")
            self.safety_badge.setText("OS Input: Disabled (Esc = Emergency Stop)")
            self.safety_badge.setStyleSheet(
                "padding: 4px 8px; border-radius: 4px; "
                "background-color: #1e293b; color: #93c5fd; font-weight: 600;"
            )
            self.tracking_mode_label.setText("Mode: Preview Only")

    @Slot()
    def _on_start_stop_clicked(self) -> None:
        if self._controller.is_running:
            self._controller.stop()
        else:
            self.error_label.hide()
            self.error_label.setText("")
            self._controller.start()

    @Slot(bool)
    def _on_os_control_toggled(self, checked: bool) -> None:
        self._controller.set_os_input_enabled(checked)

    @Slot(int)
    def _on_camera_changed(self, combo_index: int) -> None:
        camera_index = self.camera_selector.itemData(combo_index)
        if isinstance(camera_index, int):
            self._controller.set_camera_index(camera_index)

    @Slot(bool)
    def _on_landmarks_toggled(self, enabled: bool) -> None:
        self._controller.set_show_landmarks(enabled)
        self.camera_view.set_show_landmarks(enabled)

    @Slot(int)
    def _on_sensitivity_changed(self, value: int) -> None:
        sensitivity = value / 10.0
        self.sensitivity_value_label.setText(f"{sensitivity:.1f}x")
        self._controller.set_sensitivity(sensitivity)

    @Slot(int)
    def _on_smoothing_changed(self, value: int) -> None:
        self.smoothing_value_label.setText(f"{value}%")
        self._controller.set_smoothing(value / 100.0)

    @Slot(int)
    def _on_margin_changed(self, value: int) -> None:
        self.margin_value_label.setText(f"{value}%")
        margin = value / 100.0
        self._controller.set_region_margins(margin, margin)

    @Slot()
    def _on_calibrate_center_clicked(self) -> None:
        if self._controller.calibrate_center_on_hand():
            self.notice_label.setText("Calibration updated: active region centered on hand.")
        else:
            self.notice_label.setText(
                "Calibration requires a detected hand in frame. Start camera and show your hand."
            )

    @Slot()
    def _on_reset_calibration_clicked(self) -> None:
        self.margin_slider.blockSignals(True)
        try:
            self.margin_slider.setValue(15)
            self.margin_value_label.setText("15%")
        finally:
            self.margin_slider.blockSignals(False)
        self._controller.reset_calibration()
        self.notice_label.setText("Calibration reset to default active region (15% margin).")

    @Slot()
    def _on_preview_available(self) -> None:
        item = self._controller.pop_latest_preview()
        if item is None:
            return
        frame_rgb, result = item
        self._apply_tracking_result(frame_rgb, result)

    def _apply_tracking_result(self, frame_rgb, result: TrackingResult) -> None:
        cursor = result.cursor_state
        index_extended = cursor.index_extended if cursor is not None else False
        os_active = cursor.os_input_active if cursor is not None else False

        self.camera_view.update_frame(
            frame_rgb,
            result.detection,
            active_region=self._controller.active_region,
            index_extended=index_extended,
            os_control_active=os_active,
        )

        if result.hand_detected and result.detection is not None:
            det = result.detection
            ext_text = "Index Extended" if index_extended else "Index Folded"
            self.hand_status_label.setText(
                f"Hand: {det.handedness} ({det.confidence:.0%}) · {ext_text}"
            )
            self.hand_status_label.setStyleSheet(
                "color: #4ade80; font-weight: 600;"
                if index_extended
                else "color: #fbbf24; font-weight: 600;"
            )
        else:
            self.hand_status_label.setText("Hand: Lost / not detected")
            self.hand_status_label.setStyleSheet("color: #cbd5e1;")

        if cursor is not None:
            self.tracking_mode_label.setText(f"Mode: {cursor.mode.value}")
            if cursor.mapped_position is not None:
                px, py = cursor.mapped_position
                suffix = "" if cursor.os_input_active else " (sim)"
                self.cursor_coords_label.setText(f"Cursor: ({px}, {py}){suffix}")
            else:
                self.cursor_coords_label.setText("Cursor: --")
        else:
            self.cursor_coords_label.setText("Cursor: --")

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
            self.cursor_coords_label.setText("Cursor: --")
            self.fps_label.setText("FPS: --")
            if status == CameraStatus.STOPPED:
                self.camera_view.clear(
                    "Camera is stopped.\nClick 'Start Camera' to preview webcam & 21 hand landmarks."
                )

    @Slot(bool)
    def _on_os_control_changed(self, enabled: bool) -> None:
        self.os_control_button.blockSignals(True)
        try:
            self.os_control_button.setChecked(enabled)
        finally:
            self.os_control_button.blockSignals(False)
        self._update_os_control_indicator(enabled, emergency=False)
        if enabled:
            self.notice_label.setText(
                "OS cursor control is LIVE. Extend index finger to steer; press Esc to stop."
            )

    @Slot()
    def _on_emergency_stopped(self) -> None:
        self.os_control_button.blockSignals(True)
        try:
            self.os_control_button.setChecked(False)
        finally:
            self.os_control_button.blockSignals(False)
        self._update_os_control_indicator(False, emergency=True)
        self.cursor_coords_label.setText("Cursor: --")
        self.notice_label.setText(
            "Emergency stop (Esc) triggered: OS cursor control disabled safely."
        )

    @Slot(object)
    def _on_calibration_changed(self, region: ActiveRegion) -> None:
        self.camera_view.set_active_region(region)

    @Slot(str)
    def _on_error_occurred(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.show()
        self.camera_view.clear(f"Camera Error:\n{message}")

    def closeEvent(self, event: QCloseEvent) -> None:
        self._controller.shutdown()
        super().closeEvent(event)
