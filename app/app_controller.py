"""Application controller coordinating camera, vision, gestures, mapping, and OS input safety."""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import replace

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal, Slot

from app.config import AppConfig, reset_config_to_defaults, save_config
from app.events import (
    CameraStatus,
    ControlMode,
    CursorState,
    NormalizedPoint,
    TrackingResult,
)
from camera.camera_manager import CameraManager, CaptureFactory
from camera.frame_buffer import LatestFrameBuffer
from gestures.gesture_rules import GestureRules, GestureType
from gestures.state_manager import GestureFSMState, GestureStateManager
from input_control.action_mapper import ActionMapper
from input_control.backend import InputBackend
from input_control.desktop_backend import DesktopInputBackend
from input_control.dispatcher import InputDispatcher
from vision.coordinate_mapper import ActiveRegion, CoordinateMapper
from vision.hand_landmarker import HandLandmarkerService
from vision.landmark_features import extract_hand_features
from vision.vision_worker import VisionWorker

logger = logging.getLogger(__name__)

_VK_ESCAPE = 0x1B
_ACTION_DISPLAY_TTL_MS = 1200


def is_windows_escape_pressed() -> bool:
    """Poll whether the physical Esc key is currently pressed on Windows."""
    if not sys.platform.startswith("win"):
        return False
    try:
        state = ctypes.windll.user32.GetAsyncKeyState(_VK_ESCAPE)
        return bool(state & 0x8000)
    except Exception:
        return False


class AppController(QObject):
    """Owns capture, tracking, gesture FSM, coordinate mapping, and safe OS input."""

    preview_available = Signal()
    status_changed = Signal(object)
    error_occurred = Signal(str)
    os_control_changed = Signal(bool)
    emergency_stopped = Signal()
    calibration_changed = Signal(object)
    pause_changed = Signal(bool)

    def __init__(
        self,
        config: AppConfig | None = None,
        config_path: Path | None = None,
        config_notice: str | None = None,
        capture_factory: CaptureFactory | None = None,
        landmarker_factory: Callable[[], HandLandmarkerService] | None = None,
        input_backend: InputBackend | None = None,
        escape_key_checker: Callable[[], bool] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config or AppConfig()
        self._config_path = config_path
        self._config_notice = config_notice
        self._frame_buffer = LatestFrameBuffer()
        self._escape_key_checker = escape_key_checker or is_windows_escape_pressed

        self._lifecycle_lock = threading.Lock()
        self._preview_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._gesture_lock = threading.Lock()
        self._pending_preview: tuple[np.ndarray, TrackingResult] | None = None
        self._preview_notified = False
        self._running = False
        self._emergency_stopped = False
        self._latest_fingertip: NormalizedPoint | None = None
        self._last_action_label: str = "None"
        self._last_action_until_ms: int = 0
        self._latest_cursor_state = CursorState(
            mode=ControlMode.DISABLED,
            index_extended=False,
            fingertip=None,
            mapped_position=None,
            os_input_active=False,
        )

        backend = input_backend or DesktopInputBackend()
        self._dispatcher = InputDispatcher(
            backend=backend,
            os_input_enabled=self._config.os_input_enabled,
            on_backend_error=self._handle_input_error,
        )
        self._action_mapper = ActionMapper(self._dispatcher)

        gesture_rules = self._build_gesture_rules()
        self._gesture_state_manager = GestureStateManager(rules=gesture_rules)

        screen_w, screen_h = self._safe_screen_size()
        initial_region = ActiveRegion(
            left=self._config.active_region_left,
            top=self._config.active_region_top,
            right=self._config.active_region_right,
            bottom=self._config.active_region_bottom,
        )
        self._mapper = CoordinateMapper(
            screen_width=screen_w,
            screen_height=screen_h,
            active_region=initial_region,
            sensitivity=self._config.cursor_sensitivity,
            smoothing=self._config.cursor_smoothing,
            mirror_x=False,
            adaptive_smoothing=self._config.adaptive_smoothing,
            deadband_px=self._config.cursor_deadband_px,
        )

        self._camera = CameraManager(
            config=self._config,
            frame_buffer=self._frame_buffer,
            capture_factory=capture_factory,
            on_status=self._handle_camera_status,
            on_error=self._handle_worker_error,
        )
        self._vision = VisionWorker(
            config=self._config,
            frame_buffer=self._frame_buffer,
            landmarker_factory=landmarker_factory,
            camera_fps_provider=lambda: self._camera.current_fps,
            on_result=self._handle_vision_result,
            on_error=self._handle_worker_error,
        )

        self._escape_timer = QTimer(self)
        self._escape_timer.setInterval(30)
        self._escape_timer.timeout.connect(self._poll_escape_key)
        self.os_control_changed.connect(self._sync_escape_timer)
        if self._config.os_input_enabled:
            self._escape_timer.start()

    def _safe_screen_size(self) -> tuple[int, int]:
        try:
            width, height = self._dispatcher.screen_size()
            if width > 0 and height > 0:
                return (width, height)
        except Exception:
            logger.warning("Unable to query screen size; falling back to 1920x1080")
        return (1920, 1080)

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def is_running(self) -> bool:
        with self._lifecycle_lock:
            return self._running

    @property
    def camera_status(self) -> CameraStatus:
        return self._camera.status

    @property
    def os_input_enabled(self) -> bool:
        return self._dispatcher.is_enabled

    @property
    def active_region(self) -> ActiveRegion:
        return self._mapper.active_region

    @property
    def input_dispatcher(self) -> InputDispatcher:
        return self._dispatcher

    @property
    def coordinate_mapper(self) -> CoordinateMapper:
        return self._mapper

    @property
    def gesture_state_manager(self) -> GestureStateManager:
        return self._gesture_state_manager

    @property
    def latest_cursor_state(self) -> CursorState:
        with self._state_lock:
            return self._latest_cursor_state

    @property
    def config_path(self) -> Path | None:
        return self._config_path

    @property
    def config_notice(self) -> str | None:
        return self._config_notice

    def save_settings(self, path: Path | None = None) -> bool:
        """Persist current configuration to local JSON."""
        target_path = path or self._config_path
        return save_config(self._config, target_path)

    def reset_settings_to_defaults(self) -> AppConfig:
        """Reset config to defaults and update live controller parameters."""
        new_cfg = reset_config_to_defaults(self._config_path)
        self._config = new_cfg
        self.set_sensitivity(new_cfg.cursor_sensitivity)
        self.set_smoothing(new_cfg.cursor_smoothing)
        self.set_pinch_thresholds(new_cfg.pinch_activate_ratio, new_cfg.pinch_release_ratio)
        self.set_drag_hold_ms(new_cfg.drag_hold_ms)
        self.set_double_click_interval_ms(new_cfg.double_click_interval_ms)
        self.set_scroll_sensitivity(new_cfg.scroll_sensitivity)
        self.set_scroll_deadzone(new_cfg.scroll_deadzone)
        self.reset_calibration()
        return new_cfg

    @property
    def is_paused(self) -> bool:
        with self._gesture_lock:
            return self._gesture_state_manager.is_paused

    def set_paused(self, paused: bool) -> None:
        """Explicitly pause or resume hand gesture tracking."""
        timestamp_ms = time.monotonic_ns() // 1_000_000
        with self._gesture_lock:
            events = self._gesture_state_manager.set_paused(paused, timestamp_ms=timestamp_ms)
        if paused:
            self._dispatcher.release_all()
        if events:
            dispatch_result = self._action_mapper.dispatch(events, None)
            if dispatch_result.action_label:
                with self._state_lock:
                    self._last_action_label = dispatch_result.action_label
                    self._last_action_until_ms = timestamp_ms + _ACTION_DISPLAY_TTL_MS
        self.pause_changed.emit(paused)

    def toggle_pause(self) -> None:
        """Toggle pause/resume tracking state."""
        self.set_paused(not self.is_paused)

    def start(self) -> None:
        """Start webcam capture and MediaPipe hand tracking."""
        with self._lifecycle_lock:
            if self._running:
                return
            if not self._config.model_path.is_file():
                msg = (
                    f"Hand landmarker model not found at '{self._config.model_path}'. "
                    "See assets/models/README.md for setup instructions."
                )
                self.status_changed.emit(CameraStatus.ERROR)
                self.error_occurred.emit(msg)
                return

            self._running = True
            self._clear_pending_preview()
            self._mapper.reset_smoothing()
            with self._gesture_lock:
                self._gesture_state_manager.reset(reason="start")
            self._camera.start()
            self._vision.start()
            logger.info("AppController started capture and tracking pipeline")

    def stop(self) -> None:
        """Stop webcam capture and vision processing and safely release OS input."""
        self.set_os_input_enabled(False)
        with self._lifecycle_lock:
            if not self._running and not self._camera.is_running and not self._vision.is_running:
                return
            self._running = False
        self._stop_workers()
        self._clear_pending_preview()
        with self._gesture_lock:
            self._gesture_state_manager.reset(reason="stop")
        self._dispatcher.release_all()
        with self._state_lock:
            self._latest_fingertip = None
            self._last_action_label = "None"
            self._last_action_until_ms = 0
            self._latest_cursor_state = CursorState(
                mode=ControlMode.DISABLED,
                index_extended=False,
                fingertip=None,
                mapped_position=None,
                os_input_active=False,
            )

    def shutdown(self) -> None:
        """Cleanly terminate workers and release all OS input state on exit."""
        logger.info("Shutting down AppController")
        self.stop()
        self._dispatcher.close()

    def set_camera_index(self, camera_index: int) -> None:
        """Update the selected camera index; restarts pipeline if currently running."""
        if camera_index < 0:
            raise ValueError("camera_index must be >= 0")
        was_running = self.is_running
        if was_running:
            self.stop()
        self._config.camera_index = camera_index
        if was_running:
            self.start()

    def set_mirror_preview(self, enabled: bool) -> None:
        """Toggle horizontal preview mirroring consistently with landmark detection."""
        self._config.mirror_preview = enabled
        self._vision.set_mirror_preview(enabled)
        self._mapper.reset_smoothing()

    def set_show_landmarks(self, enabled: bool) -> None:
        """Toggle landmark overlay visibility in configuration."""
        self._config.show_landmarks = enabled

    def set_os_input_enabled(self, enabled: bool) -> None:
        """Explicitly enable or disable real OS cursor and click execution."""
        enabled = bool(enabled)
        screen_w, screen_h = self._safe_screen_size()
        self._mapper.set_screen_size(screen_w, screen_h)
        self._mapper.reset_smoothing()

        with self._gesture_lock:
            self._gesture_state_manager.reset(
                timestamp_ms=time.monotonic_ns() // 1_000_000,
                reason="os_toggle",
            )

        changed = False
        with self._state_lock:
            if enabled:
                self._emergency_stopped = False
            if self._config.os_input_enabled != enabled or self._dispatcher.is_enabled != enabled:
                self._config.os_input_enabled = enabled
                self._dispatcher.set_enabled(enabled)
                changed = True
            elif not enabled:
                self._dispatcher.release_all()

        if changed:
            self.os_control_changed.emit(enabled)

    def emergency_stop(self) -> None:
        """Immediately halt OS control, reset gesture FSM, and release held buttons."""
        was_enabled = self._dispatcher.is_enabled or self._config.os_input_enabled
        with self._gesture_lock:
            self._gesture_state_manager.reset(
                timestamp_ms=time.monotonic_ns() // 1_000_000,
                reason="emergency_stop",
            )
        self._dispatcher.emergency_stop()
        self._mapper.reset_smoothing()
        with self._state_lock:
            self._config.os_input_enabled = False
            self._emergency_stopped = True
            self._last_action_label = "Emergency Stop (Buttons Released)"
            self._last_action_until_ms = 0
            self._latest_cursor_state = CursorState(
                mode=ControlMode.EMERGENCY_STOPPED,
                index_extended=False,
                fingertip=self._latest_fingertip,
                mapped_position=None,
                os_input_active=False,
                gesture_state="Emergency Stopped",
                last_action="Emergency Stop (Buttons Released)",
                dragging=False,
            )
        if was_enabled:
            self.os_control_changed.emit(False)
        self.emergency_stopped.emit()
        logger.warning("Emergency stop executed via Esc")

    def set_sensitivity(self, sensitivity: float) -> None:
        """Update cursor movement sensitivity."""
        self._mapper.set_sensitivity(sensitivity)
        self._config.cursor_sensitivity = sensitivity

    def set_smoothing(self, smoothing: float) -> None:
        """Update exponential moving average smoothing factor."""
        self._mapper.set_smoothing(smoothing)
        self._config.cursor_smoothing = smoothing

    def _build_gesture_rules(self) -> GestureRules:
        return GestureRules(
            pinch_activate_ratio=self._config.pinch_activate_ratio,
            pinch_release_ratio=self._config.pinch_release_ratio,
            pinch_stable_ms=self._config.pinch_stable_ms,
            drag_hold_ms=self._config.drag_hold_ms,
            double_click_interval_ms=self._config.double_click_interval_ms,
            click_cooldown_ms=self._config.click_cooldown_ms,
            pinch_smoothing_alpha=self._config.pinch_smoothing_alpha,
            min_pinch_confidence=self._config.min_pinch_confidence,
            tracking_loss_grace_ms=self._config.tracking_loss_grace_ms,
            scroll_stable_ms=self._config.scroll_stable_ms,
            scroll_sensitivity=self._config.scroll_sensitivity,
            scroll_deadzone=self._config.scroll_deadzone,
            scroll_release_grace_ms=self._config.scroll_release_grace_ms,
            pause_hold_ms=self._config.pause_hold_ms,
            pause_cooldown_ms=self._config.pause_cooldown_ms,
        )

    def set_scroll_sensitivity(self, sensitivity: float) -> None:
        """Update scroll displacement sensitivity."""
        new_rules = replace(
            self._build_gesture_rules(),
            scroll_sensitivity=sensitivity,
        )
        self._config.scroll_sensitivity = sensitivity
        with self._gesture_lock:
            self._gesture_state_manager.update_rules(new_rules)

    def set_scroll_deadzone(self, deadzone: float) -> None:
        """Update scroll vertical deadzone threshold."""
        new_rules = replace(
            self._build_gesture_rules(),
            scroll_deadzone=deadzone,
        )
        self._config.scroll_deadzone = deadzone
        with self._gesture_lock:
            self._gesture_state_manager.update_rules(new_rules)

    def set_pinch_thresholds(self, activate_ratio: float, release_ratio: float) -> None:
        """Update palm-normalized pinch activation and release thresholds."""
        new_rules = replace(
            self._build_gesture_rules(),
            pinch_activate_ratio=activate_ratio,
            pinch_release_ratio=release_ratio,
        )
        self._config.pinch_activate_ratio = activate_ratio
        self._config.pinch_release_ratio = release_ratio
        with self._gesture_lock:
            self._gesture_state_manager.update_rules(new_rules)

    def set_drag_hold_ms(self, drag_hold_ms: int) -> None:
        """Update pinch hold duration before drag activates."""
        new_rules = replace(
            self._build_gesture_rules(),
            drag_hold_ms=drag_hold_ms,
        )
        self._config.drag_hold_ms = drag_hold_ms
        with self._gesture_lock:
            self._gesture_state_manager.update_rules(new_rules)

    def set_double_click_interval_ms(self, interval_ms: int) -> None:
        """Update double-click timing window in milliseconds."""
        new_rules = replace(
            self._build_gesture_rules(),
            double_click_interval_ms=interval_ms,
        )
        self._config.double_click_interval_ms = interval_ms
        with self._gesture_lock:
            self._gesture_state_manager.update_rules(new_rules)

    def set_region_margins(self, margin_x: float, margin_y: float) -> None:
        """Set symmetric active movement region margins."""
        region = ActiveRegion.from_margins(margin_x, margin_y)
        self._apply_active_region(region)

    def calibrate_center_on_hand(self) -> bool:
        """Center the active movement box around the currently detected index fingertip."""
        with self._state_lock:
            tip = self._latest_fingertip
        if tip is None:
            return False
        current = self._mapper.active_region
        centered = ActiveRegion.centered_around(
            center=tip,
            width=current.width,
            height=current.height,
        )
        self._apply_active_region(centered)
        return True

    def reset_calibration(self) -> None:
        """Restore the default active movement region."""
        self._apply_active_region(ActiveRegion())

    def _apply_active_region(self, region: ActiveRegion) -> None:
        self._mapper.set_active_region(region)
        self._config.active_region_left = region.left
        self._config.active_region_top = region.top
        self._config.active_region_right = region.right
        self._config.active_region_bottom = region.bottom
        self.calibration_changed.emit(region)

    def pop_latest_preview(self) -> tuple[np.ndarray, TrackingResult] | None:
        """Pop the newest preview frame and tracking result on the Qt UI thread."""
        with self._preview_lock:
            item = self._pending_preview
            self._pending_preview = None
            self._preview_notified = False
            return item

    def _clear_pending_preview(self) -> None:
        with self._preview_lock:
            self._pending_preview = None
            self._preview_notified = False

    def _stop_workers(self) -> None:
        self._frame_buffer.close()
        self._vision.stop()
        self._camera.stop()

    @Slot(bool)
    def _sync_escape_timer(self, enabled: bool) -> None:
        if enabled:
            if not self._escape_timer.isActive():
                self._escape_timer.start()
        elif self._escape_timer.isActive():
            self._escape_timer.stop()

    @Slot()
    def _poll_escape_key(self) -> None:
        if self._dispatcher.is_enabled and self._escape_key_checker():
            self.emergency_stop()

    def _process_cursor_tracking(self, result: TrackingResult) -> CursorState:
        if self._dispatcher.is_enabled and self._escape_key_checker():
            self.emergency_stop()

        features = extract_hand_features(result.detection)

        with self._gesture_lock:
            state_update = self._gesture_state_manager.update(features, result.timestamp_ms)

        for event in state_update.events:
            if event.gesture == GestureType.PAUSE_TOGGLED:
                self.pause_changed.emit(state_update.state == GestureFSMState.PAUSED)

        os_active = self._dispatcher.is_enabled and self.is_running

        if state_update.cursor_should_move and features.index_tip is not None:
            mapped_pos: tuple[int, int] | None = self._mapper.map_point(features.index_tip)
        else:
            mapped_pos = None
            if not features.hand_detected and not state_update.dragging:
                self._mapper.reset_smoothing()

        dispatch_result = self._action_mapper.dispatch(state_update.events, mapped_pos)
        if (
            (not features.hand_detected and not state_update.dragging)
            or state_update.state == GestureFSMState.PAUSED
        ):
            self._dispatcher.release_all()

        with self._state_lock:
            self._latest_fingertip = features.index_tip
            emergency = self._emergency_stopped

            if dispatch_result.action_label is not None:
                self._last_action_label = dispatch_result.action_label
                self._last_action_until_ms = result.timestamp_ms + _ACTION_DISPLAY_TTL_MS
            elif self._last_action_until_ms and result.timestamp_ms > self._last_action_until_ms:
                self._last_action_label = "None"
                self._last_action_until_ms = 0

            action_label = self._last_action_label

        if emergency and not os_active:
            mode = ControlMode.EMERGENCY_STOPPED
        elif not os_active:
            mode = ControlMode.DISABLED
        elif state_update.state == GestureFSMState.PAUSED:
            mode = ControlMode.PAUSED
        elif state_update.state == GestureFSMState.SCROLLING:
            mode = ControlMode.SCROLLING
        elif state_update.dragging:
            mode = ControlMode.DRAGGING
        elif state_update.cursor_should_move:
            mode = ControlMode.CURSOR
        else:
            mode = ControlMode.STANDBY

        cursor_state = CursorState(
            mode=mode,
            index_extended=features.index_extended,
            fingertip=features.index_tip,
            mapped_position=mapped_pos,
            os_input_active=os_active,
            gesture_state=state_update.state.value,
            last_action=action_label,
            dragging=state_update.dragging,
            active_pinch=state_update.active_pinch.value,
            scrolling=state_update.scrolling,
            scroll_delta=state_update.scroll_delta,
            paused=state_update.state == GestureFSMState.PAUSED,
        )

        with self._state_lock:
            self._latest_cursor_state = cursor_state
        return cursor_state

    def _handle_camera_status(self, status: CameraStatus) -> None:
        self.status_changed.emit(status)

    def _handle_vision_result(self, frame_rgb: np.ndarray, result: TrackingResult) -> None:
        cursor_state = self._process_cursor_tracking(result)
        enriched = replace(result, cursor_state=cursor_state)
        should_notify = False
        with self._preview_lock:
            self._pending_preview = (frame_rgb, enriched)
            if not self._preview_notified:
                self._preview_notified = True
                should_notify = True
        if should_notify:
            self.preview_available.emit()

    def _handle_input_error(self, message: str) -> None:
        with self._gesture_lock:
            self._gesture_state_manager.reset(
                timestamp_ms=time.monotonic_ns() // 1_000_000,
                reason="input_error",
            )
        with self._state_lock:
            self._config.os_input_enabled = False
        self.os_control_changed.emit(False)
        self.error_occurred.emit(message)

    def _handle_worker_error(self, message: str) -> None:
        self.set_os_input_enabled(False)
        with self._lifecycle_lock:
            self._running = False
        # Stop on a helper thread if called from inside one of the worker threads
        # to avoid self-joining.
        threading.Thread(
            target=self._stop_workers,
            name="HandPilot-StopOnError",
            daemon=True,
        ).start()
        self.status_changed.emit(CameraStatus.ERROR)
        self.error_occurred.emit(message)
