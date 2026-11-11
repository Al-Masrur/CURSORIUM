"""Application controller coordinating camera, vision worker, and Qt UI signals."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QObject, Signal

from app.config import AppConfig
from app.events import CameraStatus, TrackingResult
from camera.camera_manager import CameraManager, CaptureFactory
from camera.frame_buffer import LatestFrameBuffer
from vision.hand_landmarker import HandLandmarkerService
from vision.vision_worker import VisionWorker

logger = logging.getLogger(__name__)


class AppController(QObject):
    """Owns the Phase 1 capture + tracking lifecycle and marshals updates to Qt."""

    preview_available = Signal()
    status_changed = Signal(object)
    error_occurred = Signal(str)

    def __init__(
        self,
        config: AppConfig | None = None,
        capture_factory: CaptureFactory | None = None,
        landmarker_factory: Callable[[], HandLandmarkerService] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config or AppConfig()
        self._frame_buffer = LatestFrameBuffer()

        self._lifecycle_lock = threading.Lock()
        self._preview_lock = threading.Lock()
        self._pending_preview: tuple[np.ndarray, TrackingResult] | None = None
        self._preview_notified = False
        self._running = False

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
            self._camera.start()
            self._vision.start()
            logger.info("AppController started Phase 1 pipeline")

    def stop(self) -> None:
        """Stop webcam capture and vision processing cleanly."""
        with self._lifecycle_lock:
            if not self._running and not self._camera.is_running and not self._vision.is_running:
                return
            self._running = False
        self._stop_workers()
        self._clear_pending_preview()

    def shutdown(self) -> None:
        """Cleanly terminate all background workers before application exit."""
        logger.info("Shutting down AppController")
        self.stop()

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

    def set_show_landmarks(self, enabled: bool) -> None:
        """Toggle landmark overlay visibility in configuration."""
        self._config.show_landmarks = enabled

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

    def _handle_camera_status(self, status: CameraStatus) -> None:
        self.status_changed.emit(status)

    def _handle_vision_result(self, frame_rgb: np.ndarray, result: TrackingResult) -> None:
        should_notify = False
        with self._preview_lock:
            self._pending_preview = (frame_rgb, result)
            if not self._preview_notified:
                self._preview_notified = True
                should_notify = True
        if should_notify:
            self.preview_available.emit()

    def _handle_worker_error(self, message: str) -> None:
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

