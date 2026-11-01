import os


THEME_COLORS = {
    "special": "#BF092F",
    "bg": "#132440",
    "title_bar": "#0D1B33",
    "panel": "#16476A",
    "accent": "#3B9797",
    "success": "#3B9797",
    "graph_bg": "#132440",
    "text": "#E8EEF0",
    "muted": "#A9C2CF",
    "disabled": "#6F8A99",
}


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


def app_stylesheet(font_size=16):
    c = THEME_COLORS
    default_font = default_font_family()
    return f"""
QWidget {{
    background-color: {c['bg']};
    color: {c['text']};
    font-family: "{default_font}";
    font-size: {int(font_size)}px;
}}
QMainWindow, QDialog {{
    background-color: {c['bg']};
}}
QLabel {{
    color: {c['text']};
}}
QPushButton {{
    background-color: {c['accent']};
    color: {c['text']};
    border: 1px solid {c['accent']};
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: {c['panel']};
    color: {c['text']};
}}
QPushButton:pressed {{
    background-color: {c['special']};
    border-color: {c['special']};
}}
QPushButton:disabled {{
    background-color: {c['panel']};
    color: {c['disabled']};
    border-color: {c['accent']};
}}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {c['panel']};
    color: {c['text']};
    border: 1px solid {c['accent']};
    border-radius: 6px;
    padding: 4px 6px;
    selection-background-color: {c['accent']};
}}
QComboBox QAbstractItemView {{
    background-color: {c['panel']};
    color: {c['text']};
    selection-background-color: {c['accent']};
    border: 1px solid {c['accent']};
}}
QCheckBox {{
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
}}
QTabWidget::pane {{
    border: 1px solid {c['accent']};
    background: {c['panel']};
}}
QTabBar::tab {{
    background: {c['panel']};
    color: {c['muted']};
    border: 1px solid {c['accent']};
    padding: 6px 12px;
}}
QTabBar::tab:selected {{
    background: {c['accent']};
    color: {c['text']};
}}
QHeaderView::section {{
    background-color: {c['panel']};
    color: {c['text']};
    border: 1px solid {c['accent']};
    padding: 4px;
}}
QTableWidget {{
    background-color: {c['panel']};
    color: {c['text']};
    gridline-color: {c['accent']};
    border: 1px solid {c['accent']};
}}
QProgressBar {{
    border: 1px solid {c['accent']};
    border-radius: 5px;
    text-align: center;
    background-color: {c['panel']};
    color: {c['text']};
}}
QProgressBar::chunk {{
    background-color: {c['success']};
}}
QScrollArea {{
    border: 1px solid {c['accent']};
    background-color: {c['panel']};
}}
QScrollBar:vertical {{
    border: none;
    background: {c['bg']};
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {c['accent']};
    min-height: 24px;
    border-radius: 5px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
    background: transparent;
    border: none;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: {c['bg']};
}}
QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {{
    background: transparent;
    width: 0px;
    height: 0px;
}}
QScrollBar:horizontal {{
    border: none;
    background: {c['bg']};
    height: 10px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {c['accent']};
    min-width: 24px;
    border-radius: 5px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
    background: transparent;
    border: none;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: {c['bg']};
}}
QScrollBar::left-arrow:horizontal, QScrollBar::right-arrow:horizontal {{
    background: transparent;
    width: 0px;
    height: 0px;
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
    c = THEME_COLORS
    if kind == "danger":
        bg = c["special"]
        text = c["text"]
        border = c["special"]
        hover_bg = c["panel"]
        hover_border = c["panel"]
    elif kind == "success":
        bg = c["success"]
        text = c["text"]
        border = c["accent"]
        hover_bg = c["panel"]
        hover_border = c["accent"]
    elif kind == "muted":
        bg = c["panel"]
        text = c["text"]
        border = c["accent"]
        hover_bg = c["bg"]
        hover_border = c["accent"]
    else:
        bg = c["accent"]
        text = c["text"]
        border = c["accent"]
        hover_bg = c["panel"]
        hover_border = c["accent"]
    return (
        f"QPushButton {{ "
        f"background-color: {bg}; color: {text}; font-weight: bold; "
        f"border: 1px solid {border}; border-radius: 6px; padding: 5px 12px; "
        f"}} "
        f"QPushButton:hover:!disabled {{ "
        f"background-color: {hover_bg}; border-color: {hover_border}; "
        f"}} "
        f"QPushButton:pressed:!disabled {{ "
        f"background-color: {c['special']}; border-color: {c['special']}; color: {c['text']}; "
        f"}} "
        f"QPushButton:disabled {{ "
        f"background-color: {c['panel']}; color: {c['disabled']}; border-color: {c['panel']}; "
        f"}}"
    )


def themed_label_style(kind="muted"):
    c = THEME_COLORS
    if kind == "danger":
        color = c["special"]
    elif kind == "success":
        color = c["success"]
    else:
        color = c["muted"]
    return f"color: {color}; font-weight: bold;"


def themed_status_color(color_hint=None):
    hint = (color_hint or "").strip().lower()
    success_hints = {"#2e7d32", "#00c853", "#4caf50", "#78a083", "#3b9797"}
    warning_hints = {"#ff9800", "#f57c00"}
    error_hints = {"#f44336", "#c62828", "#bf092f"}
    if hint in error_hints:
        return THEME_COLORS["special"]
    if hint in warning_hints:
        return THEME_COLORS["special"]
    if hint in success_hints:
        return THEME_COLORS["success"]
    return THEME_COLORS["muted"]
