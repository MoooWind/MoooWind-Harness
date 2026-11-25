import json
import time
from pathlib import Path

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from brand_logo import ACCENT, INK, MUTED, draw_mark, render_wordmark


CONFIG_PATH = Path(__file__).resolve().parent / "data" / "startup_animation.json"


def load_startup_animation_config():
    defaults = {
        "enabled": True,
        "type": "brand_logo",
        "duration_seconds": 5,
        "style": "mirror",
    }
    if not CONFIG_PATH.exists():
        return defaults

    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError("启动动画配置必须是 JSON 对象。")

    duration = config.get("duration_seconds", defaults["duration_seconds"])
    try:
        duration = max(3, min(10, int(duration)))
    except (TypeError, ValueError) as error:
        raise ValueError("启动动画 duration_seconds 必须是 3 到 10 的整数。") from error

    style = config.get("style", defaults["style"])
    if style not in ("mirror", "stack", "offset"):
        raise ValueError("启动动画 style 仅支持 mirror、stack 或 offset。")

    return {
        "enabled": bool(config.get("enabled", defaults["enabled"])),
        "type": "brand_logo",
        "duration_seconds": duration,
        "style": style,
    }


_TIMELINE = {
    "panel": (0.0, 0.34),
    "corners": (0.12, 0.62),
    "mark": (0.32, 1.72),
    "nodes": (1.62, 1.96),
    "sweep": (1.72, 2.42),
    "divider": (1.55, 1.98),
    "wordmark": (1.72, 2.46),
    "subtitle": (2.34, 2.72),
    "rail": (2.50, 2.86),
}


def _ease_out(value):
    value = max(0.0, min(1.0, value))
    return 1.0 - (1.0 - value) ** 3


def _progress(elapsed, interval):
    start, end = interval
    return _ease_out((elapsed - start) / max(end - start, 1e-6))


class LogoSplash(QWidget):
    """Brand splash with a real model-loader progress and completion lifecycle."""

    ready = pyqtSignal()

    def __init__(
        self,
        duration_seconds=5,
        style="mirror",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(660, 430)

        self.duration_seconds = max(float(duration_seconds), _TIMELINE["rail"][1])
        self.style = style
        self.elapsed = 0.0
        self.status = "INITIALIZING LOCAL MODEL"
        self.percent = 0
        self.loader_done = False
        self.loader_success = False
        self.done_at = None
        self.outro_at = None
        self._finish_scheduled = False
        self._clock = time.monotonic()

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    def attach_loader(self, loader):
        loader.progress_signal.connect(self.set_progress)
        loader.done_signal.connect(self.set_complete)

    def showEvent(self, event):
        self._clock = time.monotonic()
        self._timer.start()
        super().showEvent(event)

    def set_progress(self, percent, status):
        self.percent = max(0, min(100, int(percent)))
        self.status = str(status).upper()
        self.update()

    def set_complete(self, success, message):
        self.loader_done = True
        self.loader_success = bool(success)
        self.percent = 100 if success else self.percent
        self.status = "MODEL READY" if success else "MODEL LOAD FAILED"
        self.done_message = str(message)
        self.done_at = time.monotonic()
        self.update()

    def _tick(self):
        now = time.monotonic()
        delta = max(0.0, min(0.1, now - self._clock))
        self._clock = now
        self.elapsed += delta

        hold_time = 0.42 if self.loader_success else 1.05
        if (
            self.loader_done
            and self.elapsed >= self.duration_seconds
            and self.done_at is not None
            and now - self.done_at >= hold_time
            and self.outro_at is None
        ):
            self.outro_at = now
        if self.outro_at is not None and now - self.outro_at >= 0.46:
            if not self._finish_scheduled:
                self._finish_scheduled = True
                QTimer.singleShot(0, self._finish)
        self.update()

    def _finish(self):
        self._timer.stop()
        self.hide()
        self.ready.emit()

    def skip_intro(self):
        self.elapsed = max(self.duration_seconds, _TIMELINE["rail"][1])
        self._clock = time.monotonic()
        self.update()

    def mousePressEvent(self, event):
        self.skip_intro()
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_Space, Qt.Key_Return):
            self.skip_intro()
            return
        super().keyPressEvent(event)

    def _draw_panel(self, painter, elapsed, base_opacity):
        panel_progress = _progress(elapsed, _TIMELINE["panel"])
        if panel_progress <= 0:
            return
        alpha = int(255 * panel_progress)
        bounds = QRectF(12, 12, self.width() - 24, self.height() - 24)
        painter.setOpacity(base_opacity * alpha / 255.0)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(INK.red(), INK.green(), INK.blue(), 248))
        painter.drawRoundedRect(bounds, 18, 18)

        border = QColor(232, 228, 220, int(38 * panel_progress))
        painter.setPen(QPen(border, 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(bounds, 18, 18)
        painter.setOpacity(base_opacity)

    def _draw_corners(self, painter, elapsed, base_opacity):
        amount = _progress(elapsed, _TIMELINE["corners"])
        if amount <= 0:
            return
        painter.save()
        painter.setOpacity(base_opacity * amount * 0.7)
        pen = QPen(QColor(107, 114, 128, 180), 1.2)
        painter.setPen(pen)
        margin, length = 28, 12
        right, bottom = self.width() - margin, self.height() - margin
        for x, y, sx, sy in (
            (margin, margin, 1, 1),
            (right, margin, -1, 1),
            (margin, bottom, 1, -1),
            (right, bottom, -1, -1),
        ):
            painter.drawLine(QPointF(x, y), QPointF(x + sx * length, y))
            painter.drawLine(QPointF(x, y), QPointF(x, y + sy * length))
        painter.restore()

    def paintEvent(self, event):
        elapsed = self.elapsed
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setClipRegion(event.region())

        fade = 1.0
        if self.outro_at is not None:
            fade = max(0.0, 1.0 - (time.monotonic() - self.outro_at) / 0.46)
        painter.setOpacity(fade)
        self._draw_panel(painter, elapsed, fade)
        self._draw_corners(painter, elapsed, fade)

        center_x = self.width() / 2
        mark_progress = _progress(elapsed, _TIMELINE["mark"])
        if mark_progress > 0:
            mark_size = 152
            mark_box = (
                center_x - mark_size / 2,
                40,
                mark_size,
                mark_size,
            )
            painter.save()
            painter.setOpacity(fade * _progress(elapsed, _TIMELINE["mark"]))
            draw_mark(
                painter,
                mark_box,
                style=self.style,
                progress=mark_progress,
                per_stroke_stagger=0.25,
                node_progress=_progress(elapsed, _TIMELINE["nodes"]),
            )
            painter.restore()

        sweep_progress = _progress(elapsed, _TIMELINE["sweep"])
        if 0 < sweep_progress < 1:
            y = 40 + 152 * sweep_progress
            gradient = QLinearGradient(center_x - 100, y, center_x + 100, y)
            gradient.setColorAt(0.0, QColor(217, 164, 65, 0))
            gradient.setColorAt(0.5, QColor(217, 164, 65, 90))
            gradient.setColorAt(1.0, QColor(217, 164, 65, 0))
            painter.setPen(Qt.NoPen)
            painter.setBrush(gradient)
            painter.drawRoundedRect(QRectF(center_x - 100, y - 1, 200, 2), 1, 1)

        wordmark_progress = _progress(elapsed, _TIMELINE["wordmark"])
        if wordmark_progress > 0:
            render_wordmark(
                painter,
                QRectF(120, 208, self.width() - 240, 32),
                opacity=fade * wordmark_progress,
                reveal=wordmark_progress,
            )

        subtitle_progress = _progress(elapsed, _TIMELINE["subtitle"])
        if subtitle_progress > 0:
            painter.save()
            painter.setOpacity(fade * subtitle_progress)
            font = QFont("Segoe UI")
            font.setPixelSize(10)
            font.setLetterSpacing(QFont.AbsoluteSpacing, 3.2)
            painter.setFont(font)
            painter.setPen(QPen(MUTED))
            painter.drawText(
                QRectF(40, 246, self.width() - 80, 20),
                Qt.AlignCenter,
                "LOCAL INTELLIGENCE  /  PERSONAL WORKSPACE",
            )
            painter.restore()

        divider_progress = _progress(elapsed, _TIMELINE["divider"])
        if divider_progress > 0:
            half_width = 136 * divider_progress
            painter.setPen(QPen(QColor(107, 114, 128, 110), 1))
            painter.drawLine(
                QPointF(center_x - half_width, 278),
                QPointF(center_x + half_width, 278),
            )

        rail_progress = _progress(elapsed, _TIMELINE["rail"])
        if rail_progress > 0:
            left, right, y = 154, self.width() - 154, 330
            rail_width = right - left
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 20))
            painter.drawRoundedRect(QRectF(left, y, rail_width, 2), 1, 1)
            painter.setBrush(ACCENT)
            painter.drawRoundedRect(
                QRectF(left, y, rail_width * self.percent / 100, 2),
                1,
                1,
            )

            painter.setPen(QPen(MUTED))
            font = QFont("Consolas")
            font.setPixelSize(10)
            painter.setFont(font)
            painter.drawText(
                QRectF(left, 343, rail_width * 0.8, 18),
                Qt.AlignLeft | Qt.AlignVCenter,
                self.status[:32],
            )
            painter.setPen(QPen(ACCENT))
            painter.drawText(
                QRectF(left + rail_width * 0.8, 343, rail_width * 0.2, 18),
                Qt.AlignRight | Qt.AlignVCenter,
                f"{self.percent:02d}%",
            )

        if self.loader_done and not self.loader_success:
            painter.setPen(QPen(QColor("#E58A7A")))
            font = QFont("Segoe UI")
            font.setPixelSize(9)
            painter.setFont(font)
            painter.drawText(
                QRectF(70, 371, self.width() - 140, 18),
                Qt.AlignCenter,
                getattr(self, "done_message", "")[:96],
            )

        painter.end()
