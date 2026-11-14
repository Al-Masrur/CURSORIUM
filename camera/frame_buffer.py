"""Thread-safe single-slot latest-frame buffer that drops stale frames."""

from __future__ import annotations

import threading

from app.events import FramePacket


class LatestFrameBuffer:
    """Stores only the newest FramePacket so processing never builds latency."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: FramePacket | None = None
        self._unconsumed: bool = False
        self._dropped_count: int = 0
        self._closed: bool = False

    def put(self, packet: FramePacket) -> None:
        """Publish a new frame, overwriting any unread older frame."""
        with self._condition:
            if self._closed:
                return
            if self._unconsumed:
                self._dropped_count += 1
            self._latest = packet
            self._unconsumed = True
            self._condition.notify_all()

    def get_latest(
        self,
        last_sequence_id: int = 0,
        timeout: float = 0.1,
    ) -> FramePacket | None:
        """Wait up to ``timeout`` seconds for a frame newer than ``last_sequence_id``."""
        with self._condition:
            if not self._has_newer_frame(last_sequence_id) and not self._closed:
                self._condition.wait(timeout=timeout)

            if self._closed or not self._has_newer_frame(last_sequence_id):
                return None

            self._unconsumed = False
            return self._latest

    def _has_newer_frame(self, last_sequence_id: int) -> bool:
        return self._latest is not None and self._latest.sequence_id > last_sequence_id

    def close(self) -> None:
        """Close the buffer and unblock any waiting consumers."""
        with self._condition:
            self._closed = True
            self._latest = None
            self._unconsumed = False
            self._condition.notify_all()

    def reset(self) -> None:
        """Clear stored frame state and re-open the buffer for a new capture session."""
        with self._condition:
            self._closed = False
            self._latest = None
            self._unconsumed = False
            self._dropped_count = 0
            self._condition.notify_all()

    @property
    def dropped_count(self) -> int:
        """Number of frames overwritten before being consumed."""
        with self._condition:
            return self._dropped_count

    @property
    def dropped_frames(self) -> int:
        """Alias for dropped_count."""
        return self.dropped_count

    @property
    def is_closed(self) -> bool:
        with self._condition:
            return self._closed

