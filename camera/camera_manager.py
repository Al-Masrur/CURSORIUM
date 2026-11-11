"""Webcam initialization, capture loop, error detection, and cleanup."""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable
from typing import Protocol

import cv2
import numpy as np

from app.config import AppConfig
from app.events import CameraStatus, FramePacket
from camera.frame_buffer import LatestFrameBuffer

logger = logging.getLogger(__name__)


class VideoCaptureLike(Protocol):
    """Minimal subset of cv2.VideoCapture used by CameraManager."""

    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def set(self, prop_id: int, value: float) -> bool: ...
    def release(self) -> None: ...


CaptureFactory = Callable[[int], VideoCaptureLike]


def default_capture_factory(camera_index: int) -> VideoCaptureLike:
    """Open a VideoCapture device using DirectShow first on Windows."""
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(camera_index)


class CameraManager:
    """Manages webcam lifecycle on a dedicated background thread."""

    def __init__(
        self,
        config: AppConfig,
        frame_buffer: LatestFrameBuffer,
        capture_factory: CaptureFactory | None = None,
        on_status: Callable[[CameraStatus], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._config = config
        self._frame_buffer = frame_buffer
        self._capture_factory = capture_factory or default_capture_factory
        self._on_status = on_status
        self._on_error = on_error

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status: CameraStatus = CameraStatus.STOPPED
        self._sequence_id: int = 0
        self._fps: float = 0.0

    @property
    def status(self) -> CameraStatus:
        with self._lock:
            return self._status

    @property
    def current_fps(self) -> float:
        with self._lock:
            return self._fps

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start capturing webcam frames on a background thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._frame_buffer.reset()
            self._fps = 0.0
            self._set_status_locked(CameraStatus.STARTING)
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="HandPilot-Camera",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the capture loop to stop and wait cleanly for thread exit."""
        self._stop_event.set()
        self._frame_buffer.close()
        with self._lock:
            thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        with self._lock:
            if not (self._thread is not None and self._thread.is_alive()):
                self._thread = None
            if self._status != CameraStatus.ERROR:
                self._set_status_locked(CameraStatus.STOPPED)
            self._fps = 0.0

    def _set_status(self, status: CameraStatus) -> None:
        with self._lock:
            self._set_status_locked(status)

    def _set_status_locked(self, status: CameraStatus) -> None:
        if self._status == status:
            return
        self._status = status
        cb = self._on_status
        if cb is not None:
            cb(status)

    def _emit_error(self, message: str) -> None:
        logger.error("Camera error: %s", message)
        self._set_status(CameraStatus.ERROR)
        if self._on_error is not None:
            self._on_error(message)

    def _configure_capture(self, cap: VideoCaptureLike) -> None:
        # Request MJPG compressed USB transfer on Windows webcams first so
        # higher frame rates (e.g. 30 FPS) do not fall back to slow YUY2.
        try:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            cap.set(cv2.CAP_PROP_FOURCC, float(fourcc))
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._config.frame_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._config.frame_height))
        cap.set(cv2.CAP_PROP_FPS, float(self._config.target_fps))
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)
        except Exception:
            # Some backends ignore CAP_PROP_BUFFERSIZE; safe to continue.
            pass

    def _capture_loop(self) -> None:
        cap: VideoCaptureLike | None = None
        consecutive_failures = 0
        fps_window_start = time.monotonic()
        fps_frame_count = 0

        try:
            cap = self._capture_factory(self._config.camera_index)
            if cap is None or not cap.isOpened():
                self._emit_error(
                    f"Unable to open camera index {self._config.camera_index}. "
                    "Check that the webcam is connected and not in use by another app."
                )
                return

            self._configure_capture(cap)
            self._set_status(CameraStatus.RUNNING)
            logger.info(
                "Camera %d opened (%dx%d @ %d FPS target)",
                self._config.camera_index,
                self._config.frame_width,
                self._config.frame_height,
                self._config.target_fps,
            )

            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    if consecutive_failures >= self._config.max_consecutive_read_failures:
                        self._emit_error(
                            f"Camera {self._config.camera_index} disconnected or stopped delivering frames."
                        )
                        return
                    time.sleep(0.01)
                    continue

                consecutive_failures = 0
                self._sequence_id += 1
                timestamp_ms = time.monotonic_ns() // 1_000_000
                packet = FramePacket(
                    frame_bgr=frame,
                    timestamp_ms=timestamp_ms,
                    sequence_id=self._sequence_id,
                )
                self._frame_buffer.put(packet)

                fps_frame_count += 1
                now = time.monotonic()
                elapsed = now - fps_window_start
                if elapsed >= 0.5:
                    with self._lock:
                        self._fps = fps_frame_count / elapsed
                    fps_window_start = now
                    fps_frame_count = 0

        except Exception as exc:
            self._emit_error(f"Unexpected camera failure: {exc}")
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    logger.exception("Failed while releasing camera resource")
            self._frame_buffer.close()
            if self.status != CameraStatus.ERROR:
                self._set_status(CameraStatus.STOPPED)
            logger.info("Camera capture loop exited cleanly")

