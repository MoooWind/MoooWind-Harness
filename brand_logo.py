from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)


INK = QColor("#0E1013")
LINE = QColor("#E8E4DC")
ACCENT = QColor("#D9A441")
MUTED = QColor("#6B7280")
STROKE_RATIO = 0.052
SHOW_NODES = True
DEFAULT_STYLE = "mirror"


def _m_stroke():
    return [(0.0, 1.0), (0.0, 0.0), (0.5, 0.66), (1.0, 0.0), (1.0, 1.0)]


def _w_stroke():
    return [(x, 1.0 - y) for x, y in _m_stroke()]


def get_polylines(style=DEFAULT_STYLE):
    if style == "mirror":
        return [_m_stroke(), _w_stroke()]
    if style == "stack":
        gap = 0.06
        top = [(x, y * (0.5 - gap / 2)) for x, y in _m_stroke()]
        bottom = [
            (x, 0.5 + gap / 2 + (1.0 - y) * (0.5 - gap / 2))
            for x, y in _m_stroke()
        ]
        return [top, bottom]
    if style == "offset":
        shift = 0.12
        left = [(x * (1.0 - shift), y) for x, y in _m_stroke()]
        right = [(x * (1.0 - shift) + shift, y) for x, y in _w_stroke()]
        return [left, right]
    raise ValueError(f"未知 Logo 样式：{style}")


def mark_strokes(style=DEFAULT_STYLE):
    m, w = _m_stroke(), _w_stroke()
    if style == "mirror":
        return [
            ([(0.0, 1.0), (0.0, 0.0)], False),
            ([m[1], m[2], m[3]], False),
            ([(1.0, 0.0), (1.0, 1.0)], False),
            ([w[1], w[2], w[3]], True),
        ]
    return [
        (line, index == 1)
        for index, line in enumerate(get_polylines(style))
    ]


def scale_points(points, box):
    x, y, width, height = box
    side = min(width, height)
    offset_x = x + (width - side) / 2
    offset_y = y + (height - side) / 2
    return [
        (offset_x + px * side, offset_y + py * side)
        for px, py in points
    ]


def polyline_partial(points, progress):
    if not points or progress <= 0:
        return []
    if progress >= 1:
        return list(points)

    segments = []
    total_length = 0.0
    for start, end in zip(points, points[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = (dx * dx + dy * dy) ** 0.5
        segments.append((start, end, length))
        total_length += length
    if total_length <= 0:
        return [points[0]]

    target_length = total_length * progress
    passed_length = 0.0
    result = [points[0]]
    for start, end, length in segments:
        if passed_length + length >= target_length:
            fraction = (target_length - passed_length) / max(length, 1e-12)
            result.append(
                (
                    start[0] + (end[0] - start[0]) * fraction,
                    start[1] + (end[1] - start[1]) * fraction,
                )
            )
            return result
        result.append(end)
        passed_length += length
    return result


def draw_mark(
    painter,
    box,
    style=DEFAULT_STYLE,
    progress=1.0,
    per_stroke_stagger=0.0,
    node_progress=1.0,
    color=None,
    accent=None,
    glow=True,
):
    color = color or LINE
    accent = accent or ACCENT
    side = min(box[2], box[3])
    if side < 48:
        line_ratio, glow = 0.085, False
    elif side < 128:
        line_ratio = 0.068
    else:
        line_ratio = STROKE_RATIO
    line_width = max(1.0, side * line_ratio)
    strokes = mark_strokes(style)

    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    for index, (points, highlighted) in enumerate(strokes):
        if per_stroke_stagger > 0 and len(strokes) > 1:
            span = 1.0 / (1.0 + per_stroke_stagger * (len(strokes) - 1))
            start = index / (len(strokes) - 1) * (1.0 - span)
            stroke_progress = (progress - start) / span
        else:
            stroke_progress = progress
        stroke_progress = max(0.0, min(1.0, stroke_progress))
        visible = polyline_partial(scale_points(points, box), stroke_progress)
        if len(visible) < 2:
            continue

        polygon = QPolygonF([QPointF(x, y) for x, y in visible])
        stroke_color = accent if highlighted else color
        if glow:
            glow_pen = QPen(QColor(stroke_color.red(), stroke_color.green(), stroke_color.blue(), 46))
            glow_pen.setWidthF(line_width * 3)
            glow_pen.setCapStyle(Qt.RoundCap)
            glow_pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(glow_pen)
            painter.drawPolyline(polygon)

        pen = QPen(stroke_color)
        pen.setWidthF(line_width)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawPolyline(polygon)

        if SHOW_NODES and stroke_progress >= 0.999 and side >= 128:
            painter.save()
            painter.setOpacity(node_progress)
            painter.setPen(Qt.NoPen)
            painter.setBrush(stroke_color)
            radius = line_width * 0.78
            for x, y in scale_points(points, box):
                painter.drawEllipse(QPointF(x, y), radius, radius)
            painter.restore()
    painter.restore()


def render_wordmark(
    painter,
    rect,
    text="MOOOWIND",
    opacity=1.0,
    color=None,
    reveal=1.0,
    tracking_ratio=0.42,
):
    color = color or LINE
    painter.save()
    font_size = max(8, rect.height() * 0.34)
    font = QFont("Consolas")
    font.setPixelSize(int(round(font_size)))
    font.setWeight(QFont.Normal)
    painter.setFont(font)
    painter.setPen(QPen(color))
    metrics = QFontMetricsF(font)
    tracking = font_size * tracking_ratio
    characters = list(text)
    advances = [metrics.horizontalAdvance(char) for char in characters]
    full_width = sum(advances) + tracking * max(0, len(characters) - 1)
    shown = int(len(characters) * max(0.0, min(1.0, reveal)))
    x = rect.center().x() - full_width / 2
    y = rect.center().y() + font_size * 0.36
    for index in range(shown):
        local_progress = max(
            0.0,
            min(1.0, (reveal * len(characters) - index) * 3),
        )
        painter.setOpacity(opacity * (0.25 + 0.75 * local_progress))
        painter.drawText(QPointF(x, y), characters[index])
        x += advances[index] + tracking
    painter.restore()


def make_mark_pixmap(size=96, style=DEFAULT_STYLE, background=None):
    pixmap = QPixmap(size, size)
    pixmap.fill(background if background is not None else Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    padding = size * 0.18
    draw_mark(
        painter,
        (padding, padding, size - 2 * padding, size - 2 * padding),
        style=style,
        color=LINE,
        accent=ACCENT,
        glow=size >= 128,
    )
    painter.end()
    return pixmap
