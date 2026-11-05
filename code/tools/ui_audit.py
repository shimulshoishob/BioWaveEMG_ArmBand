#!/usr/bin/env python3
"""Open every window of the app at laptop-sized screens and report anything that is clipped.

Profiles emulate the usable screen area (after menu bar / dock / taskbar) and the base font size the app
would pick there. For each window the audit flags widgets that are squeezed below their minimum size,
text wider than its widget, inputs shorter than their text, and controls pushed outside the window with
no scroll area to reach them. Optionally saves a screenshot per window.

    python code/tools/ui_audit.py [--profile mac13] [--shots DIR]
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

PROFILES = {                       # usable width, usable height, base font px
    "mac13": (1440, 805, 14),      # 13" MacBook, default scaling, menu bar + dock removed
    "mac_small": (1280, 690, 12),  # MacBook scaled to "more space" / small external display
    "win125": (1536, 816, 14),     # 1080p laptop at 125 % scaling
    "win150": (1280, 672, 12),     # 1080p laptop at 150 % scaling
    "win_hd": (1366, 728, 13),     # 1366x768 laptop at 100 %
}


def audit_window(win, name):
    from PyQt5.QtWidgets import (QAbstractButton, QAbstractSpinBox, QComboBox, QLabel, QLineEdit, QScrollArea,
                                 QWidget)
    issues = []
    top = win.window()

    def in_scroll(w):
        p = w.parentWidget()
        while p is not None and p is not top:
            if isinstance(p, QScrollArea):
                return True
            p = p.parentWidget()
        return False

    for w in win.findChildren(QWidget):
        if not w.isVisibleTo(win) or w.width() <= 1 or w.height() <= 1:
            continue
        cls = type(w).__name__
        if isinstance(w, QLineEdit) and isinstance(w.parentWidget(), (QAbstractSpinBox, QComboBox)):
            continue                                   # the inner editor of a spin box / combo: the parent is checked
        label = (w.text() if hasattr(w, "text") and callable(w.text) else "") or w.objectName() or ""
        label = str(label)[:40]
        msh = w.minimumSizeHint()
        # 1. squeezed below its own minimum
        if isinstance(w, (QAbstractButton, QAbstractSpinBox, QComboBox, QLineEdit)):
            if w.height() < msh.height() - 6:
                issues.append(f"{name}: {cls} '{label}' height {w.height()} < min {msh.height()}")
            if w.width() < msh.width() - 2:
                issues.append(f"{name}: {cls} '{label}' width {w.width()} < min {msh.width()}")
        # 2. text wider than the widget
        if isinstance(w, (QAbstractButton, QLabel)) and label and not (isinstance(w, QLabel) and w.wordWrap()):
            if isinstance(w, QLabel) and w.textFormat() == 1:      # rich text: skip (HTML)
                continue
            fm = w.fontMetrics()
            if "<" not in label and fm.horizontalAdvance(label) > w.width() + 1 and len(label) < 40:
                issues.append(f"{name}: {cls} '{label}' text {fm.horizontalAdvance(label)}px > width {w.width()}px")
        # 3. input shorter than its text
        if isinstance(w, (QAbstractSpinBox, QComboBox, QLineEdit)):
            if w.height() < w.fontMetrics().height() + 6:
                issues.append(f"{name}: {cls} '{label}' too short ({w.height()}px) for its text")
        # 4. outside the window without a scroll area
        g = w.mapTo(top, w.rect().topLeft())
        if (g.x() + w.width() > top.width() + 2 or g.y() + w.height() > top.height() + 2) and not in_scroll(w):
            if isinstance(w, (QAbstractButton, QAbstractSpinBox, QComboBox, QLineEdit)):
                issues.append(f"{name}: {cls} '{label}' outside the window")
    return issues


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="mac13", choices=sorted(PROFILES))
    ap.add_argument("--shots", default="")
    args = ap.parse_args(argv)
    avail_w, avail_h, font_px = PROFILES[args.profile]

    from PyQt5.QtCore import QRect
    from PyQt5.QtGui import QScreen
    from PyQt5.QtWidgets import QApplication, QMessageBox
    # make the offscreen platform behave like a laptop screen of this profile
    QScreen.availableGeometry = lambda self: QRect(0, 0, avail_w, avail_h)
    QScreen.geometry = lambda self: QRect(0, 0, avail_w, avail_h + 95)
    import main as app_main
    from app_theme import apply_dark_theme, install_screen_fit_filter
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    qapp = QApplication.instance() or QApplication([])
    apply_dark_theme(qapp, font_size=font_px)

    main_win = app_main.EMGVisualizer()
    builders = {
        "main": lambda: main_win,
        "data_collection": lambda: (main_win.open_data_collection_dialog(), main_win.data_collection_dialog)[1],
        "training": lambda: (main_win.open_rf_training_dialog(), main_win.training_dialog)[1],
        "packet_tracer": lambda: (main_win.open_packet_tracer_window(), main_win.packet_tracer_window)[1],
        "options_info": lambda: (main_win.open_options_info_dialog(), main_win.options_info_dialog)[1],
        "calibration": lambda: (main_win.open_calibration_dialog(), main_win.calibration_dialog)[1],
        "connection_mode": lambda: app_main.ConnectionModeDialog(main_win),
        "port_config": lambda: app_main.PortConfigDialog("", False, main_win),
        "usb_provision": lambda: app_main.USBProvisionDialog(main_win),
        "wireless_config": lambda: app_main.WirelessConfigDialog("", main_win),
        "study_procedure": lambda: app_main.StudyProcedureDialog(main_win),
        "class_labels": lambda: app_main.ClassLabelEditorDialog(["left", "right", "rest"], main_win),
    }
    # windows that need a live stream
    main_win.is_connected = True
    main_win.is_calibrated = True
    builders["analysis"] = lambda: (main_win.open_analysis_window(), main_win.analysis_window)[1]
    builders["realtime"] = lambda: (main_win.open_realtime_classification_window(), main_win.realtime_classification_window)[1]

    all_issues = []
    for name, build in builders.items():
        try:
            win = build()
            if win is None:
                print(f"[skip] {name}: could not open")
                continue
            win.show()
            qapp.processEvents()
            # what install_screen_fit_filter does on a real screen of this size
            w = min(win.width(), int(avail_w * 0.94))
            h = min(win.height(), int(avail_h * 0.94))
            win.resize(w, h)
            for _ in range(3):
                qapp.processEvents()
            issues = audit_window(win, name)
            if win.width() > avail_w or win.height() > avail_h:
                issues.insert(0, f"{name}: window {win.width()}x{win.height()} is larger than the usable screen "
                                 f"{avail_w}x{avail_h} (its minimum size is too big)")
            all_issues += issues
            print(f"[{'ok' if not issues else 'ISSUES'}] {name:16s} {win.width()}x{win.height()}  {len(issues)} issue(s)")
            for line in issues[:6]:
                print("      ", line)
            if args.shots:
                os.makedirs(args.shots, exist_ok=True)
                win.grab().save(os.path.join(args.shots, f"{args.profile}_{name}.png"))
            if name != "main":
                win.close()
        except Exception as exc:                       # keep auditing the rest
            print(f"[error] {name}: {type(exc).__name__}: {exc}")
    print(f"\n{args.profile}: {len(all_issues)} issue(s) in total")
    return 1 if all_issues else 0


if __name__ == "__main__":
    sys.exit(main())
