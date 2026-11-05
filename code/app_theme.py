import os


THEME_COLORS = {
    "special": "#E5484D",      # errors, pressed state, destructive actions
    "bg": "#0B1426",           # window background
    "title_bar": "#070E1C",
    "panel": "#15253F",        # inputs, tables, raised surfaces
    "card": "#101E36",         # cards on top of the window background
    "border": "#27446E",       # quiet borders
    "accent": "#1BB5A6",       # primary teal
    "accent_hi": "#4FE0D0",    # highlights, focus rings
    "accent_dim": "#136F69",   # selected rows, pressed tabs
    "success": "#1BB5A6",
    "warn": "#F5A524",
    "ok": "#36D399",
    "graph_bg": "#0B1426",
    "text": "#E8EFF7",
    "muted": "#93A9C2",
    "disabled": "#5C7189",
}

# One colour per EMG channel (distinct on a dark background, colour-blind friendly order).
CHANNEL_COLORS = ["#4FE0D0", "#F5A524", "#7AA2FF", "#FF7A90", "#A5E36B", "#C792EA", "#FFD166", "#5AC8FA",
                  "#FF9F6E", "#9AA8FF", "#6EE7B7"]


def apply_dark_title_bar(window):
    """Request a dark native title bar on Windows where supported."""
    try:
        import ctypes
        import sys
        from ctypes import wintypes
    except Exception:
        return False

    if sys.platform != "win32":
        return False

    try:
        hwnd = int(window.winId())
    except Exception:
        return False

    def _hex_to_colorref(hex_color):
        val = (hex_color or "").strip().lstrip("#")
        if len(val) != 6:
            return None
        try:
            r = int(val[0:2], 16)
            g = int(val[2:4], 16)
            b = int(val[4:6], 16)
        except ValueError:
            return None
        return (b << 16) | (g << 8) | r

    use_dark = ctypes.c_int(1)
    use_dark_size = ctypes.sizeof(use_dark)
    attrs = (20, 19)  # Win10 20H1+, then legacy fallback
    enabled_dark = False
    for attr in attrs:
        try:
            result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                wintypes.HWND(hwnd),
                wintypes.DWORD(attr),
                ctypes.byref(use_dark),
                wintypes.DWORD(use_dark_size),
            )
            if result == 0:
                enabled_dark = True
                break
        except Exception:
            continue

    caption_color = _hex_to_colorref(THEME_COLORS.get("title_bar", THEME_COLORS["bg"]))
    text_color = _hex_to_colorref(THEME_COLORS["text"])
    if caption_color is not None:
        try:
            caption_val = ctypes.c_int(caption_color)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                wintypes.HWND(hwnd),
                wintypes.DWORD(35),  # DWMWA_CAPTION_COLOR
                ctypes.byref(caption_val),
                wintypes.DWORD(ctypes.sizeof(caption_val)),
            )
        except Exception:
            pass
    if text_color is not None:
        try:
            text_val = ctypes.c_int(text_color)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                wintypes.HWND(hwnd),
                wintypes.DWORD(36),  # DWMWA_TEXT_COLOR
                ctypes.byref(text_val),
                wintypes.DWORD(ctypes.sizeof(text_val)),
            )
        except Exception:
            pass
    return enabled_dark


def default_font_family():
    """Native UI font for the running OS (Segoe UI on Windows, SF on macOS)."""
    if os.name == "nt":
        return "Bahnschrift"
    try:
        from PyQt5.QtGui import QFontDatabase

        return QFontDatabase.systemFont(QFontDatabase.GeneralFont).family() or "Sans Serif"
    except Exception:
        return "Sans Serif"


def recommended_font_px(default=16, lo=12, hi=18):
    """Base font size (logical px) scaled to the primary screen's usable height.

    A fixed 18 px font fills a 13" MacBook (about 775 logical px of usable height)
    and a Windows laptop at 150 % scaling, but is fine on a 1080p desktop. Scaling
    with the screen keeps the layout's proportions the same on every device.
    """
    try:
        from PyQt5.QtWidgets import QApplication

        screen = QApplication.primaryScreen()
        if screen is None:
            return default
        return int(max(lo, min(hi, round(screen.availableGeometry().height() / 58.0))))
    except Exception:
        return default


def ensure_ui_icons():
    """Draw the small glyphs the stylesheet needs (tick, arrows) once into a cache folder and return it.

    Qt style sheets can only show images from files, so they are rendered with QPainter at start-up. Returns
    ``None`` if drawing is not possible (the stylesheet then simply omits the images).
    """
    try:
        import tempfile
        from PyQt5.QtCore import QPointF, Qt
        from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF

        folder = os.path.join(tempfile.gettempdir(), "biowave_ui_icons_v2")
        os.makedirs(folder, exist_ok=True)

        def draw(name, painter_fn, size=40):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                return
            pm = QPixmap(size, size)
            pm.fill(Qt.transparent)
            painter = QPainter(pm)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter_fn(painter, size)
            painter.end()
            pm.save(path, "PNG")

        def tick(p, n):
            p.setPen(QPen(QColor("#06201D"), n * 0.13, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPolyline(QPolygonF([QPointF(n * .24, n * .52), QPointF(n * .43, n * .70), QPointF(n * .76, n * .30)]))

        def arrow(color, up):
            def fn(p, n):
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(color))
                y0, y1 = (n * .66, n * .32) if up else (n * .34, n * .68)
                p.drawPolygon(QPolygonF([QPointF(n * .22, y0), QPointF(n * .78, y0), QPointF(n * .50, y1)]))
            return fn

        draw("check.png", tick)
        draw("arrow_up.png", arrow(THEME_COLORS["muted"], True))
        draw("arrow_down.png", arrow(THEME_COLORS["muted"], False))
        return folder.replace("\\", "/")
    except Exception:
        return None


def control_height(font_size):
    """Outer height (px) used for buttons and inputs, derived from the base font so nothing is ever clipped."""
    return int(round(int(font_size) * 2.25))


def app_stylesheet(font_size=16):
    c = THEME_COLORS
    f = int(font_size)
    default_font = default_font_family()
    h = control_height(f)
    content_h = max(14, h - 10)                       # min-height is the content box: + padding + border = h
    icons = ensure_ui_icons()
    check_img = f"image: url({icons}/check.png);" if icons else ""
    up_img = f"image: url({icons}/arrow_up.png);" if icons else ""
    down_img = f"image: url({icons}/arrow_down.png);" if icons else ""
    return f"""
QWidget {{
    color: {c['text']};
    font-family: "{default_font}";
    font-size: {f}px;
}}
QMainWindow, QDialog {{
    background-color: {c['bg']};
}}
QLabel, QCheckBox, QRadioButton, QGroupBox {{
    background: transparent;
}}
QToolTip {{
    background-color: {c['title_bar']};
    color: {c['text']};
    border: 1px solid {c['accent']};
    border-radius: 6px;
    padding: 6px 8px;
}}
QPushButton {{
    background-color: {c['accent']};
    color: #06201D;
    border: 1px solid {c['accent']};
    border-radius: 8px;
    padding: 4px 14px;
    min-height: {content_h}px;
    font-weight: 700;
}}
QPushButton:hover:!disabled {{
    background-color: {c['accent_hi']};
    border-color: {c['accent_hi']};
}}
QPushButton:pressed:!disabled {{
    background-color: {c['accent_dim']};
    border-color: {c['accent_dim']};
    color: {c['text']};
}}
QPushButton:disabled {{
    background-color: {c['panel']};
    color: {c['disabled']};
    border-color: {c['border']};
}}
QPushButton:focus {{
    border: 1px solid {c['accent_hi']};
}}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {c['panel']};
    color: {c['text']};
    border: 1px solid {c['border']};
    border-radius: 8px;
    padding: 4px 8px;
    selection-background-color: {c['accent_dim']};
    selection-color: {c['text']};
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    min-height: {content_h}px;
}}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
    border-color: {c['accent']};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {c['accent_hi']};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {c['disabled']};
    background-color: {c['card']};
}}
QSpinBox, QDoubleSpinBox {{
    padding-right: 24px;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    border: none;
    background: transparent;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 22px;
    border: none;
    background: transparent;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    {up_img}
    width: 10px;
    height: 10px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    {down_img}
    width: 10px;
    height: 10px;
}}
QComboBox {{
    padding-right: 26px;
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 24px;
    border: none;
}}
QComboBox::down-arrow {{
    {down_img}
    width: 10px;
    height: 10px;
}}
QComboBox QAbstractItemView {{
    background-color: {c['panel']};
    color: {c['text']};
    selection-background-color: {c['accent_dim']};
    border: 1px solid {c['border']};
    outline: none;
}}
QCheckBox, QRadioButton {{
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 1px solid {c['border']};
    border-radius: 5px;
    background: {c['panel']};
}}
QCheckBox::indicator:hover {{
    border-color: {c['accent']};
}}
QCheckBox::indicator:checked {{
    background: {c['accent']};
    border-color: {c['accent']};
    {check_img}
}}
QCheckBox::indicator:disabled {{
    background: {c['card']};
    border-color: {c['border']};
}}
QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {c['border']};
    border-radius: 9px;
    background: {c['panel']};
}}
QRadioButton::indicator:checked {{
    background: {c['accent']};
    border: 4px solid {c['panel']};
    outline: 1px solid {c['accent']};
}}
QGroupBox {{
    border: 1px solid {c['border']};
    border-radius: 10px;
    margin-top: {f + 6}px;
    padding: 10px;
    font-weight: 700;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {c['accent_hi']};
}}
QTabWidget::pane {{
    border: 1px solid {c['border']};
    border-radius: 10px;
    background: {c['card']};
    top: -1px;
}}
QTabBar {{
    font-weight: 600;
}}
QTabBar::tab {{
    background: transparent;
    color: {c['muted']};
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 7px 18px;
    margin: 0 4px 4px 0;
}}
QTabBar::tab:hover:!selected {{
    color: {c['text']};
    background: {c['panel']};
}}
QTabBar::tab:selected {{
    background: {c['accent_dim']};
    color: #ffffff;
    border-color: {c['accent']};
}}
QHeaderView::section {{
    background-color: {c['panel']};
    color: {c['muted']};
    border: none;
    border-right: 1px solid {c['border']};
    border-bottom: 1px solid {c['border']};
    padding: 6px 8px;
    font-weight: 700;
}}
QTableWidget, QListWidget, QTreeWidget {{
    background-color: {c['card']};
    alternate-background-color: {c['panel']};
    color: {c['text']};
    gridline-color: {c['border']};
    border: 1px solid {c['border']};
    border-radius: 8px;
    outline: none;
}}
QTableWidget::item, QListWidget::item {{
    padding: 4px 6px;
}}
QTableWidget::item:selected, QListWidget::item:selected {{
    background-color: {c['accent_dim']};
    color: #ffffff;
}}
QListWidget::item:hover {{
    background-color: {c['panel']};
}}
QProgressBar {{
    border: 1px solid {c['border']};
    border-radius: 8px;
    text-align: center;
    background-color: {c['panel']};
    color: {c['text']};
    min-height: {max(16, f + 4)}px;
}}
QProgressBar::chunk {{
    border-radius: 7px;
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {c['accent']}, stop:1 {c['accent_hi']});
}}
QFrame#card {{
    background-color: {c['card']};
    border: 1px solid {c['border']};
    border-radius: 12px;
}}
QFrame#card QLabel, QFrame#card QCheckBox {{
    background: transparent;
    border: none;
}}
QFrame#header {{
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10294D, stop:1 {c['bg']});
    border: 1px solid {c['border']};
    border-radius: 12px;
}}
QFrame#header QLabel {{
    background: transparent;
    border: none;
}}
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QScrollArea > QWidget > QWidget {{
    background-color: transparent;
}}
QStatusBar {{
    background-color: {c['title_bar']};
    border-top: 1px solid {c['border']};
}}
QStatusBar::item {{
    border: none;
}}
QSplitter::handle {{
    background-color: {c['border']};
}}
QScrollBar:vertical {{
    border: none;
    background: transparent;
    width: 12px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {c['border']};
    min-height: 28px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical:hover {{
    background: {c['accent']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
    background: transparent;
    border: none;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}
QScrollBar:horizontal {{
    border: none;
    background: transparent;
    height: 12px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {c['border']};
    min-width: 28px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {c['accent']};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
    background: transparent;
    border: none;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent;
}}
"""


def configure_high_dpi():
    """Call once, before constructing QApplication.

    Without this, PyQt5 can report window/widget sizes in physical pixels on
    a Retina/HiDPI display (every MacBook since ~2012, plus most Windows
    laptops today) instead of logical points, which is what makes a window
    sized for a "normal" screen come out oversized and clip its own buttons.
    """
    try:
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication

        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        # Windows 125 % / 150 % scaling: use the exact factor instead of rounding it
        # to 100 % / 200 %, which is what makes windows look too big or too small there.
        if hasattr(QApplication, "setHighDpiScaleFactorRoundingPolicy"):
            QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except Exception:
        pass


def fit_window_to_screen(window, desired_width, desired_height, margin_ratio=0.92, max_upscale=1.15):
    """Resize+center ``window`` for the screen it opens on, keeping its aspect ratio.

    The requested size defines the window's aspect ratio. It is scaled by one common
    factor so it fits inside ``margin_ratio`` of the usable screen area (menu bar /
    dock / taskbar excluded) - shrinking on a 13" MacBook, and growing slightly (up
    to ``max_upscale``) on large Windows monitors - so the proportions are the same
    on every device instead of being cut off or stretched in one direction. The
    window stays freely resizable afterwards.
    """
    try:
        from PyQt5.QtWidgets import QApplication

        screen = window.screen() if hasattr(window, "screen") else None
        screen = screen or QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
    except Exception:
        avail = None

    width, height = int(desired_width), int(desired_height)
    if avail is not None and width > 0 and height > 0:
        scale = min(avail.width() * margin_ratio / width, avail.height() * margin_ratio / height)
        scale = min(scale, float(max_upscale))
        width, height = int(width * scale), int(height * scale)
    window.resize(width, height)
    if avail is not None:
        window.move(max(avail.x(), avail.center().x() - width // 2),
                    max(avail.y(), avail.center().y() - height // 2))


def clamp_minimum_size(window, min_width, min_height, margin_ratio=0.9):
    """setMinimumSize that never exceeds the usable screen (small laptops)."""
    try:
        from PyQt5.QtWidgets import QApplication

        screen = window.screen() if hasattr(window, "screen") else None
        screen = screen or QApplication.primaryScreen()
        avail = screen.availableGeometry()
        min_width = min(int(min_width), int(avail.width() * margin_ratio))
        min_height = min(int(min_height), int(avail.height() * margin_ratio))
    except Exception:
        pass
    window.setMinimumSize(int(min_width), int(min_height))


def install_screen_fit_filter(app, margin_ratio=0.94):
    """Keep every top-level window/dialog inside the usable screen when first shown.

    Many dialogs are laid out for a fixed size; on a small display they would open
    partly off-screen. This clamps size and position once per window, device-agnostic.
    """
    from PyQt5.QtCore import QEvent, QObject
    from PyQt5.QtWidgets import QWidget

    class _ScreenFitFilter(QObject):
        def eventFilter(self, obj, event):
            if event.type() == QEvent.Show and isinstance(obj, QWidget) and obj.isWindow():
                if not obj.property("_screen_fit_done"):
                    obj.setProperty("_screen_fit_done", True)
                    try:
                        screen = obj.screen() or app.primaryScreen()
                        avail = screen.availableGeometry()
                        w = min(obj.width(), int(avail.width() * margin_ratio))
                        h = min(obj.height(), int(avail.height() * margin_ratio))
                        if (w, h) != (obj.width(), obj.height()):
                            obj.resize(w, h)
                        x = min(max(obj.x(), avail.left()), avail.right() - w + 1)
                        y = min(max(obj.y(), avail.top()), avail.bottom() - h + 1)
                        obj.move(x, y)
                    except Exception:
                        pass
            return False

    flt = _ScreenFitFilter(app)
    app.installEventFilter(flt)
    return flt


def wrap_in_scroll_area(window, central_widget):
    """Put ``central_widget`` inside a borderless, resizable QScrollArea and
    make it ``window``'s central widget.

    A window built this way never permanently hides a control when shown
    smaller than its ideal size (a MacBook's logical desktop, a split
    screen, an external-monitor scale change) - content scrolls instead of
    clipping. Returns the QScrollArea in case a caller needs it.
    """
    from PyQt5.QtWidgets import QFrame, QScrollArea

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidget(central_widget)
    window.setCentralWidget(scroll)
    return scroll


def scrolling_layout(dialog, layout_cls=None, footer=False):
    """Give ``dialog`` a scrollable body and return the layout to fill it.

    Use in place of ``QVBoxLayout(self)``. On a small laptop screen (13" MacBook, 1366x768 or a 150 %-scaled
    Windows laptop) a dialog taller than the screen then scrolls instead of cutting off its buttons.
    """
    from PyQt5.QtWidgets import QFrame, QScrollArea, QVBoxLayout, QWidget
    from PyQt5.QtCore import Qt

    outer = QVBoxLayout(dialog)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    scroll = QScrollArea(dialog)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    outer.addWidget(scroll, 1)
    body = QWidget()
    scroll.setWidget(body)
    dialog._scroll_body = body
    dialog._scroll_area = scroll
    layout = (layout_cls or QVBoxLayout)(body)
    layout.setContentsMargins(16, 14, 16, 14)
    if footer:
        # A footer that never scrolls away: put the primary action buttons here (``dialog._footer_layout``).
        bar = QFrame(dialog)
        bar.setObjectName("card")
        bar.setStyleSheet("QFrame#card { border-left: none; border-right: none; border-bottom: none; border-radius: 0; }")
        dialog._footer_layout = QVBoxLayout(bar)
        dialog._footer_layout.setContentsMargins(16, 10, 16, 12)
        dialog._footer_layout.setSpacing(6)
        outer.addWidget(bar, 0)
    return layout


def apply_dark_theme(app, font_size=16):
    font = app.font()
    if os.name == "nt":
        from PyQt5.QtGui import QFontInfo

        font.setFamily("Bahnschrift")
        # Fallback for Windows systems where Bahnschrift is unavailable.
        if QFontInfo(font).family().lower() != "bahnschrift":
            font.setFamily("Segoe UI")
    else:
        font.setFamily(default_font_family())
    app.setFont(font)
    app.setStyleSheet(app_stylesheet(font_size=font_size))


def themed_button_style(kind="accent"):
    """Per-button style. kinds: accent (primary), success, muted (secondary), danger, ghost (outlined)."""
    c = THEME_COLORS
    if kind == "danger":
        bg, text, border, hover_bg = c["special"], "#ffffff", c["special"], "#F26B70"
    elif kind == "muted":
        bg, text, border, hover_bg = c["panel"], c["text"], c["border"], c["border"]
    elif kind == "ghost":
        bg, text, border, hover_bg = "transparent", c["accent_hi"], c["accent"], c["accent_dim"]
    else:                                   # accent / success
        bg, text, border, hover_bg = c["accent"], "#06201D", c["accent"], c["accent_hi"]
    hover_text = "#ffffff" if kind == "ghost" else text
    return (
        f"QPushButton {{ "
        f"background-color: {bg}; color: {text}; font-weight: bold; "
        f"border: 1px solid {border}; border-radius: 8px; padding: 4px 14px; "
        f"}} "
        f"QPushButton:hover:!disabled {{ "
        f"background-color: {hover_bg}; border-color: {c['accent_hi'] if kind != 'danger' else hover_bg}; color: {hover_text}; "
        f"}} "
        f"QPushButton:pressed:!disabled {{ "
        f"background-color: {c['accent_dim']}; border-color: {c['accent_dim']}; color: #ffffff; "
        f"}} "
        f"QPushButton:disabled {{ "
        f"background-color: {c['panel']}; color: {c['disabled']}; border-color: {c['border']}; "
        f"}}"
    )


def themed_label_style(kind="muted"):
    c = THEME_COLORS
    color = {"danger": c["special"], "warn": c["warn"], "success": c["ok"]}.get(kind, c["muted"])
    return f"color: {color}; font-weight: bold;"


def themed_status_color(color_hint=None):
    hint = (color_hint or "").strip().lower()
    success_hints = {"#2e7d32", "#00c853", "#4caf50", "#78a083", "#3b9797", "#00897b"}
    warning_hints = {"#ff9800", "#f57c00"}
    error_hints = {"#f44336", "#c62828", "#bf092f"}
    if hint in error_hints:
        return THEME_COLORS["special"]
    if hint in warning_hints:
        return THEME_COLORS["warn"]
    if hint in success_hints:
        return THEME_COLORS["ok"]
    return THEME_COLORS["muted"]
