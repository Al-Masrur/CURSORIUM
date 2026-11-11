"""Background worker that consumes the latest frame and runs hand landmarking."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import cv2
import numpy as np

from app.config import AppConfig
from app.events import TrackingResult
from camera.frame_buffer import LatestFrameBuffer
from vision.hand_landmarker import HandLandmarkerService

logger = logging.getLogger(__name__)


class VisionWorker:
    """Reads the latest buffered frame and publishes RGB preview + tracking results."""

    def __init__(
        self,
        config: AppConfig,
        frame_buffer: LatestFrameBuffer,
        landmarker_factory: Callable[[], HandLandmarkerService] | None = None,
        camera_fps_provider: Callable[[], float] | None = None,
        on_result: Callable[[np.ndarray, TrackingResult], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._config = config
        self._frame_buffer = frame_buffer
        self._landmarker_factory = landmarker_factory or (
            lambda: HandLandmarkerService(self._config)
        )
        self._camera_fps_provider = camera_fps_provider or (lambda: 0.0)
        self._on_result = on_result
        self._on_error = on_error

        self._lock = threading.Lock()
        self._mirror_preview = config.mirror_preview
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def mirror_preview(self) -> bool:
        with self._lock:
            return self._mirror_preview

    def set_mirror_preview(self, enabled: bool) -> None:
        with self._lock:
            self._mirror_preview = enabled

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the vision processing thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="HandPilot-Vision",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the vision loop cleanly and wait for thread exit."""
        self._stop_event.set()
        self._frame_buffer.close()
        with self._lock:
            thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        with self._lock:
            if not (self._thread is not None and self._thread.is_alive()):
                self._thread = None

    def _run_loop(self) -> None:
        landmarker: HandLandmarkerService | None = None
        last_sequence_id = 0
        fps_window_start = time.monotonic()
        fps_frame_count = 0
        tracking_fps = 0.0

        try:
            landmarker = self._landmarker_factory()
            landmarker.initialize()

            while not self._stop_event.is_set():
                packet = self._frame_buffer.get_latest(
                    last_sequence_id=last_sequence_id,
                    timeout=0.1,
                )
                if packet is None:
                    if self._frame_buffer.is_closed:
                        break
                    continue

                last_sequence_id = packet.sequence_id
                frame_bgr = packet.frame_bgr
                if self.mirror_preview:
                    frame_bgr = cv2.flip(frame_bgr, 1)

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                detection = landmarker.detect(frame_rgb, packet.timestamp_ms)

                fps_frame_count += 1
                now = time.monotonic()
                elapsed = now - fps_window_start
                if elapsed >= 0.5:
                    tracking_fps = fps_frame_count / elapsed
                    fps_window_start = now
                    fps_frame_count = 0

                result = TrackingResult(
                    timestamp_ms=packet.timestamp_ms,
                    hand_detected=detection is not None,
                    detection=detection,
                    camera_fps=self._camera_fps_provider(),
                    tracking_fps=tracking_fps,
                )
                if self._on_result is not None:
                    self._on_result(frame_rgb, result)

        except Exception as exc:
            logger.exception("Vision worker failed")
            if self._on_error is not None:
                self._on_error(f"Hand tracking error: {exc}")
        finally:
            if landmarker is not None:
                try:
                    landmarker.close()
                except Exception:
                    logger.exception("Failed while closing hand landmarker")
            logger.info("Vision worker exited cleanly")

