"""Unit tests for the bounded single-slot LatestFrameBuffer."""

from __future__ import annotations

import threading
import time

import numpy as np

from app.events import FramePacket
from camera.frame_buffer import LatestFrameBuffer


def _make_packet(seq: int) -> FramePacket:
    return FramePacket(
        frame_bgr=np.full((4, 4, 3), seq, dtype=np.uint8),
        timestamp_ms=seq * 33,
        sequence_id=seq,
    )


def test_put_overwrites_unconsumed_frame_and_tracks_drops() -> None:
    buf = LatestFrameBuffer()
    buf.put(_make_packet(1))
    buf.put(_make_packet(2))
    buf.put(_make_packet(3))

    assert buf.dropped_count == 2

    latest = buf.get_latest(last_sequence_id=0, timeout=0.05)
    assert latest is not None
    assert latest.sequence_id == 3


def test_get_latest_does_not_return_already_seen_sequence() -> None:
    buf = LatestFrameBuffer()
    buf.put(_make_packet(1))

    first = buf.get_latest(last_sequence_id=0, timeout=0.05)
    assert first is not None and first.sequence_id == 1

    second = buf.get_latest(last_sequence_id=1, timeout=0.02)
    assert second is None


def test_close_unblocks_waiting_consumer_and_reset_reopens() -> None:
    buf = LatestFrameBuffer()
    result: list[FramePacket | None] = []

    def consumer() -> None:
        result.append(buf.get_latest(last_sequence_id=0, timeout=2.0))

    thread = threading.Thread(target=consumer, daemon=True)
    thread.start()
    time.sleep(0.03)
    buf.close()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert result == [None]
    assert buf.is_closed is True

    buf.reset()
    assert buf.is_closed is False
    assert buf.dropped_count == 0
    buf.put(_make_packet(10))
    packet = buf.get_latest(last_sequence_id=0, timeout=0.05)
    assert packet is not None and packet.sequence_id == 10

