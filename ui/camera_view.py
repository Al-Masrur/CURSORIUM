"""Live webcam preview widget with crisp 21-landmark, active-region, and pinch overlay."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from app.events import HandDetection
from vision.coordinate_mapper import ActiveRegion
from vision.hand_landmarker import HAND_CONNECTIONS
from vision.landmark_features import INDEX_TIP, MIDDLE_TIP, THUMB_TIP

FINGERTIP_INDICES = frozenset({4, 8, 12, 16, 20})
WRIST_INDEX = 0


class CameraView(QWidget):
    """Renders the latest RGB video frame, active region box, and 21 hand landmarks."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._detection: HandDetection | None = None
        self._show_landmarks: bool = True
        self._active_region: ActiveRegion | None = ActiveRegion()
        self._index_extended: bool = False
        self._os_control_active: bool = False
        self._active_pinch: str = "None"
        self._dragging: bool = False
        self._scrolling: bool = False
        self._scroll_delta: int = 0
        self._paused: bool = False
        self._placeholder_text: str = (
            "Camera is stopped.\nClick 'Start Camera' to preview webcam & 21 hand landmarks."
        )

        self.setMinimumSize(640, 420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def update_frame(
        self,
        frame_rgb: np.ndarray,
        detection: HandDetection | None,
        *,
        active_region: ActiveRegion | None = None,
        index_extended: bool = False,
        os_control_active: bool = False,
        active_pinch: str = "None",
        dragging: bool = False,
        scrolling: bool = False,
        scroll_delta: int = 0,
        paused: bool = False,
    ) -> None:
        """Update the displayed RGB frame and tracking overlay on the UI thread."""
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
        if active_region is not None:
            self._active_region = active_region
        self._index_extended = index_extended
        self._os_control_active = os_control_active
        self._active_pinch = active_pinch
        self._dragging = dragging
        self._scrolling = scrolling
        self._scroll_delta = scroll_delta
        self._paused = paused
        self.update()

    def set_show_landmarks(self, enabled: bool) -> None:
        self._show_landmarks = enabled
        self.update()

    def set_active_region(self, region: ActiveRegion) -> None:
        self._active_region = region
        self.update()

    def set_os_control_active(self, active: bool) -> None:
        self._os_control_active = active
        self.update()

    def clear(self, message: str | None = None) -> None:
        """Clear the preview image and show a status message."""
        self._image = None
        self._detection = None
        self._index_extended = False
        self._active_pinch = "None"
        self._dragging = False
        self._scrolling = False
        self._scroll_delta = 0
        self._paused = False
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

        if self._active_region is not None:
            self._draw_active_region(painter, target_rect, self._active_region)

        if self._show_landmarks and self._detection is not None:
            self._draw_landmarks(
                painter,
                target_rect,
                self._detection,
                index_extended=self._index_extended,
                active_pinch=self._active_pinch,
                dragging=self._dragging,
                scrolling=self._scrolling,
            )

        if self._scrolling:
            self._draw_scrolling_indicator(painter, target_rect, self._scroll_delta)

        if self._paused:
            self._draw_paused_overlay(painter, target_rect)

    def _fit_rect(self, img_w: int, img_h: int) -> QRectF:
        widget_w = max(1, self.width())
        widget_h = max(1, self.height())
        scale = min(widget_w / img_w, widget_h / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale
        left = (widget_w - draw_w) / 2.0
        top = (widget_h - draw_h) / 2.0
        return QRectF(left, top, draw_w, draw_h)

    def _draw_active_region(
        self,
        painter: QPainter,
        target_rect: QRectF,
        region: ActiveRegion,
    ) -> None:
        box = QRectF(
            target_rect.left() + region.left * target_rect.width(),
            target_rect.top() + region.top * target_rect.height(),
            region.width * target_rect.width(),
            region.height * target_rect.height(),
        )
        color = QColor("#f59e0b") if self._os_control_active else QColor("#38bdf8")
        pen = QPen(color, 1.8, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(box)

        painter.setPen(color)
        painter.drawText(
            QPointF(box.left() + 6.0, max(target_rect.top() + 14.0, box.top() + 16.0)),
            "Active Cursor Region",
        )

    @staticmethod
    def _draw_landmarks(
        painter: QPainter,
        target_rect: QRectF,
        detection: HandDetection,
        *,
        index_extended: bool = False,
        active_pinch: str = "None",
        dragging: bool = False,
        scrolling: bool = False,
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

        # Highlight active pinch pair visually
        if active_pinch == "Thumb-Index" and len(points) > INDEX_TIP:
            pinch_color = QColor("#ef4444") if dragging else QColor("#f97316")
            painter.setPen(QPen(pinch_color, 3.5))
            painter.drawLine(points[THUMB_TIP], points[INDEX_TIP])
        elif active_pinch == "Thumb-Middle" and len(points) > MIDDLE_TIP:
            painter.setPen(QPen(QColor("#a855f7"), 3.5))
            painter.drawLine(points[THUMB_TIP], points[MIDDLE_TIP])

        for idx, pt in enumerate(points):
            if idx == INDEX_TIP and (index_extended or dragging):
                color = QColor("#ef4444") if dragging else QColor("#facc15")
                radius = 7.0
            elif idx in FINGERTIP_INDICES:
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

    @staticmethod
    def _draw_scrolling_indicator(
        painter: QPainter,
        target_rect: QRectF,
        scroll_delta: int,
    ) -> None:
        badge_w, badge_h = 160.0, 34.0
        badge_rect = QRectF(
            target_rect.left() + (target_rect.width() - badge_w) / 2.0,
            target_rect.top() + 16.0,
            badge_w,
            badge_h,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(14, 165, 233, 200))
        painter.drawRoundedRect(badge_rect, 8.0, 8.0)

        arrow = "▲" if scroll_delta > 0 else ("▼" if scroll_delta < 0 else "↕")
        text = f"{arrow} SCROLLING {scroll_delta:+d}" if scroll_delta != 0 else "↕ SCROLL MODE"
        painter.setPen(QColor("#ffffff"))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, text)

    @staticmethod
    def _draw_paused_overlay(
        painter: QPainter,
        target_rect: QRectF,
    ) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 140))
        painter.drawRect(target_rect)

        badge_w, badge_h = 240.0, 50.0
        badge_rect = QRectF(
            target_rect.left() + (target_rect.width() - badge_w) / 2.0,
            target_rect.top() + (target_rect.height() - badge_h) / 2.0,
            badge_w,
            badge_h,
        )
        painter.setBrush(QColor(30, 41, 59, 230))
        painter.setPen(QPen(QColor("#94a3b8"), 1.5))
        painter.drawRoundedRect(badge_rect, 10.0, 10.0)

        painter.setPen(QColor("#f8fafc"))
        painter.drawText(
            badge_rect,
            Qt.AlignmentFlag.AlignCenter,
            "⏸ TRACKING PAUSED\n(Hold open palm or click Resume)",
        )
