"""Live webcam preview widget with crisp 21-landmark Qt overlay."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from app.events import HandDetection
from vision.hand_landmarker import HAND_CONNECTIONS

FINGERTIP_INDICES = frozenset({4, 8, 12, 16, 20})
WRIST_INDEX = 0


class CameraView(QWidget):
    """Renders the latest RGB video frame and optional 21 hand landmarks."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._detection: HandDetection | None = None
        self._show_landmarks: bool = True
        self._placeholder_text: str = (
            "Camera is stopped.\nClick 'Start Camera' to preview webcam & 21 hand landmarks."
        )

        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def update_frame(self, frame_rgb: np.ndarray, detection: HandDetection | None) -> None:
        """Update the displayed RGB frame and hand detection on the UI thread."""
        height, width, channels = frame_rgb.shape
        if channels != 3:
            return
        bytes_per_line = channels * width
        contiguous = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        self._image = QImage(
            contiguous.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        self._detection = detection
        self.update()

    def set_show_landmarks(self, enabled: bool) -> None:
        self._show_landmarks = enabled
        self.update()

    def clear(self, message: str | None = None) -> None:
        """Clear the preview image and show a status message."""
        self._image = None
        self._detection = None
        if message is not None:
            self._placeholder_text = message
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#14161a"))

        if self._image is None or self._image.isNull():
            painter.setPen(QColor("#a1a7b3"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self._placeholder_text,
            )
            return

        target_rect = self._fit_rect(self._image.width(), self._image.height())
        painter.drawImage(target_rect, self._image)

        if self._show_landmarks and self._detection is not None:
            self._draw_landmarks(painter, target_rect, self._detection)

    def _fit_rect(self, img_w: int, img_h: int) -> QRectF:
        widget_w = max(1, self.width())
        widget_h = max(1, self.height())
        scale = min(widget_w / img_w, widget_h / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale
        left = (widget_w - draw_w) / 2.0
        top = (widget_h - draw_h) / 2.0
        return QRectF(left, top, draw_w, draw_h)

    @staticmethod
    def _draw_landmarks(
        painter: QPainter,
        target_rect: QRectF,
        detection: HandDetection,
    ) -> None:
        points = [
            QPointF(
                target_rect.left() + pt.x * target_rect.width(),
                target_rect.top() + pt.y * target_rect.height(),
            )
            for pt in detection.landmarks
        ]

        line_pen = QPen(QColor("#38bdf8"), 2.0)
        painter.setPen(line_pen)
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                painter.drawLine(points[start_idx], points[end_idx])

        for idx, pt in enumerate(points):
            if idx in FINGERTIP_INDICES:
                color = QColor("#22c55e")
                radius = 5.0
            elif idx == WRIST_INDEX:
                color = QColor("#f59e0b")
                radius = 5.0
            else:
                color = QColor("#e2e8f0")
                radius = 3.5

            painter.setPen(QPen(QColor("#0f172a"), 1.2))
            painter.setBrush(color)
            painter.drawEllipse(pt, radius, radius)

