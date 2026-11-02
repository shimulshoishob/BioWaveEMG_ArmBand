import csv
import hashlib
import hmac
import json
import os
import random
import secrets
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from html import escape as html_escape
from typing import Optional
from urllib.parse import quote

import numpy as np
import serial
import serial.tools.list_ports

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")
import joblib
import pyqtgraph as pg
from PyQt5.QtCore import QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRegion,
)
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app_theme import (
    THEME_COLORS,
    apply_dark_theme,
    apply_dark_title_bar,
    clamp_minimum_size,
    configure_high_dpi,
    fit_window_to_screen,
    install_screen_fit_filter,
    recommended_font_px,
    wrap_in_scroll_area,
    themed_button_style,
    themed_label_style,
    themed_status_color,
)
from study_meta import (CONSENT_FORM_VERSION, default_hand_order, normalize_participant_id,
                        session_folder_name, valid_participant_id)
from session_recording import (ContinuousRecorder, EventLog, LABEL_CODES, MIN_VALID_SAMPLE_RATIO, SIGNAL_COLUMNS,
                               finalize_session)
from emg_v4_core import (StreamAssembler, UsbFrameDecoder, WirelessStats, parse_bwim_packet,
                         segments_to_payload)

try:
    import pywt  # Optional for wavelet-energy features
    HAS_PYWT = True
except Exception:
    pywt = None
    HAS_PYWT = False

try:
    from rf_features import extract_window_features as rf_extract_window_features
    from rf_features import build_windows_from_sequence as rf_build_windows_from_sequence
    HAS_RF_FEATURES = True
except Exception:
    rf_extract_window_features = None
    rf_build_windows_from_sequence = None
    HAS_RF_FEATURES = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    HAS_SKLEARN = True
except Exception:
    RandomForestClassifier = None
    accuracy_score = None
    classification_report = None
    confusion_matrix = None
    train_test_split = None
    GroupShuffleSplit = None
    HAS_SKLEARN = False

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB_PARALLEL = True
except Exception:
    Parallel = None
    delayed = None
    HAS_JOBLIB_PARALLEL = False

# --- PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR).lower() == "code":
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    PROJECT_ROOT = SCRIPT_DIR
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
TRAINED_MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_model")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)
APP_ICON_PATH = os.path.join(SCRIPT_DIR, "images", "app_icon.png")

# --- GESTURE REFERENCE IMAGES (shown during data collection) ---
# Searched in order; put the PNGs in images/gestures/ (preferred) or images/.
GESTURE_IMAGE_DIRS = [
    os.path.join(SCRIPT_DIR, "images", "gestures"),
    os.path.join(SCRIPT_DIR, "images"),
    os.path.join(PROJECT_ROOT, "images", "gestures"),
    os.path.join(PROJECT_ROOT, "images"),
]
# Box (logical px) the reference image is fitted into, aspect ratio kept.
# Sized for landscape photos (~2:1); square images simply use less width.
GESTURE_IMAGE_MAX_W = 300
GESTURE_IMAGE_MAX_H = 160
GESTURE_IMAGE_RADIUS = 10
# Keys are normalized labels: lowercase, letters/digits only
# (so "Double_Closed_Fist", "double closed fist" and "DoubleClosedFist" all match).
GESTURE_IMAGE_FILES = {
    "right": "right.png",
    "left": "left.png",
    "up": "up.png",
    "down": "down.png",
    "fistclose": "fist_close.png",
    "closefist": "fist_close.png",
    "fist": "fist_close.png",
    "victorysign": "v_sign.png",
    "victory": "v_sign.png",
    "vsign": "v_sign.png",
    "doubleclosedfist": "double_fist_close.png",
    "doublefistclose": "double_fist_close.png",
    "doublefist": "double_fist_close.png",
    "pinch": "pinch.png",
    "pinchthumbindex": "pinch.png",
    "pinchusingthumbandindexfinger": "pinch.png",
    "indexthumbpinch": "pinch.png",
    "thumbindexpinch": "pinch.png",
    "thumbsup": "thumbs_up.png",
    "thumbup": "thumbs_up.png",
    "rest": "rest.png",
}
_GESTURE_PIXMAP_CACHE = {}


def _normalize_gesture_key(label):
    return "".join(ch for ch in str(label or "").lower() if ch.isalnum())


def resolve_gesture_image_path(label):
    key = _normalize_gesture_key(label)
    if not key:
        return ""
    candidates = []
    mapped = GESTURE_IMAGE_FILES.get(key)
    if mapped:
        candidates.append(mapped)
    # Fallback: a file named after the label itself (e.g. a custom class "wave" -> wave.png).
    raw = str(label).strip().lower().replace(" ", "_")
    for ext in (".png", ".jpg", ".jpeg"):
        candidates.extend([f"{raw}{ext}", f"{key}{ext}"])
    for folder in GESTURE_IMAGE_DIRS:
        for name in candidates:
            p = os.path.join(folder, name)
            if os.path.isfile(p):
                return p
    return ""


def get_gesture_pixmap(label, max_w=GESTURE_IMAGE_MAX_W, max_h=GESTURE_IMAGE_MAX_H, dpr=1.0):
    """Return a scaled, rounded QPixmap for a gesture label, or None if no image is found.

    Results (including misses) are cached, so after the first lookup a label
    costs one dict hit - no disk access or rescaling while a trial is running.
    """
    dpr = max(1.0, float(dpr or 1.0))
    label_key = (_normalize_gesture_key(label), int(max_w), int(max_h), round(dpr, 2))
    if label_key in _GESTURE_PIXMAP_CACHE:
        return _GESTURE_PIXMAP_CACHE[label_key]
    pix = None
    p = resolve_gesture_image_path(label)
    if p:
        src_pix = QPixmap(p)
        if not src_pix.isNull():
            # Scale in device pixels so the image stays sharp on high-DPI screens.
            scaled = src_pix.scaled(
                int(round(max_w * dpr)), int(round(max_h * dpr)),
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            pix = QPixmap(scaled.size())
            pix.fill(Qt.transparent)
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.Antialiasing, True)
            clip = QPainterPath()
            r = GESTURE_IMAGE_RADIUS * dpr
            clip.addRoundedRect(QRectF(0, 0, scaled.width(), scaled.height()), r, r)
            painter.setClipPath(clip)
            painter.drawPixmap(0, 0, scaled)
            painter.end()
            pix.setDevicePixelRatio(dpr)
    _GESTURE_PIXMAP_CACHE[label_key] = pix
    return pix


class StableLabel(QLabel):
    """QLabel that ignores setText/setStyleSheet calls which would not change anything.

    Re-applying an identical stylesheet still makes Qt re-polish and re-layout the
    widget, which adds up for labels refreshed many times per second.
    """

    def setText(self, text):
        if text != self.text():
            super().setText(text)

    def setStyleSheet(self, style):
        if style != self.styleSheet():
            super().setStyleSheet(style)


# --- PARTICIPANT INFO (data collection; saved to the metadata file) ---
BLOOD_GROUP_OPTIONS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]


def _value_or_na(value, fmt, cast=float):
    """Format a participant measurement, or 'N/A' when it was left empty (<= 0)."""
    try:
        v = cast(value)
    except (TypeError, ValueError):
        return "N/A"
    return fmt.format(v) if v > 0 else "N/A"

APP_ICON = None


def get_app_icon():
    global APP_ICON
    if APP_ICON is not None:
        return APP_ICON
    if os.path.isfile(APP_ICON_PATH):
        icon = QIcon(APP_ICON_PATH)
        APP_ICON = icon if not icon.isNull() else QIcon()
    else:
        APP_ICON = QIcon()
    return APP_ICON


def apply_app_icon(window):
    icon = get_app_icon()
    if icon is not None and not icon.isNull():
        window.setWindowIcon(icon)


def center_window(window, parent_widget=None):
    if window is None:
        return

    anchor = parent_widget
    if anchor is not None and isinstance(anchor, QWidget):
        while anchor.parentWidget() is not None:
            anchor = anchor.parentWidget()

    frame = window.frameGeometry()
    if frame.width() <= 0 or frame.height() <= 0:
        return

    app = QApplication.instance()
    if anchor is not None and isinstance(anchor, QWidget):
        anchor_geom = anchor.frameGeometry() if anchor.isVisible() else anchor.geometry()
        center_pt = anchor_geom.center()
        screen = None
        if app is not None:
            if anchor.windowHandle() is not None:
                screen = anchor.windowHandle().screen()
            if screen is None:
                screen = app.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else anchor_geom
    else:
        screen = app.primaryScreen() if app is not None else None
        avail = screen.availableGeometry() if screen is not None else window.geometry()
        center_pt = avail.center()

    frame.moveCenter(center_pt)
    target_x = frame.left()
    target_y = frame.top()
    max_x = avail.left() + max(0, avail.width() - frame.width())
    max_y = avail.top() + max(0, avail.height() - frame.height())
    target_x = min(max(target_x, avail.left()), max_x)
    target_y = min(max(target_y, avail.top()), max_y)
    window.move(target_x, target_y)

# --- DEFAULTS ---
APP_NAME = "BioWave - EMG"
DEFAULT_BAUD_RATE = 921600   # keep equal to USB_SERIAL_BAUD and the firmware's USB_SERIAL_BAUD
USB_KEEPALIVE_INTERVAL_S = 2.0
WINDOW_SIZE = 1000               # Number of samples shown on each channel
SAMPLE_RATE = 500                # Hz (matching ESP32 sketch)
PLOT_REFRESH_MS = 33             # ~30 FPS UI refresh (EMG does not need 50 FPS)
PLOT_RANGE_REFRESH_MS = 250      # Axis autoscale is expensive; refresh it at 4 FPS.
LIVE_ANALYSIS_REFRESH_MS = 100   # Cheap live metrics cadence (~10 Hz).
LIVE_ANALYSIS_HEAVY_REFRESH_MS = 350  # Expensive matrices (corr/lag/coherence) cadence (~3 Hz).

# --- RENDER PERFORMANCE ---
# Antialiasing is the single biggest pyqtgraph cost; keep it off for live streaming.
# OpenGL offloads rasterization to the GPU but is driver-flaky on some Windows setups.
ENABLE_ANTIALIAS = False
ENABLE_OPENGL = False
SERIAL_BATCH_SIZE = 25           # 50 ms @ 500 Hz
DEFAULT_ANALYSIS_MS = 200        # Analysis window for MAV/RMS
DEFAULT_THRESHOLD = 45.0         # EMG RMS threshold before calibration auto-tunes it
DEFAULT_HZ_THRESHOLD = 30.0      # EMG dominant-frequency gate before calibration auto-tunes it
FFT_MIN_HZ = 20.0                # Typical lower EMG band edge
FFT_MAX_HZ = 220.0               # Keep below Nyquist (250Hz at 500Hz sample rate)
BAND_DEFS = [(20.0, 60.0), (60.0, 120.0), (120.0, 220.0)]
MAINS_FREQS = [50.0, 60.0]
MAINS_BAND_HZ = 2.0
MAX_LAG_SAMPLES = int(0.10 * SAMPLE_RATE)
# Dataset label names (see session_recording.LABEL_MAP for the old names and numeric codes).
DEFAULT_TASK_LABELS = "left,right,fist_close,up,down,double_fist_close,victory_sign,thumbs_up,pinch_thumb_index,rest"
# --- STUDY PROTOCOL (defaults follow the approved study procedure) ---
# One task cycle = 4 s prepare + 3 s perform/hold + 3 s rest = 10 s. The rest period is
# recorded as its own class. Each day: 9 repetitions per hand (left + right) of every class.
DEFAULT_TASK_PREP_S = 4.0
DEFAULT_TASK_HOLD_S = 3.0
DEFAULT_TASK_REST_S = 3.0
DEFAULT_TASK_SETTLE_S = 0.0   # Optional un-recorded gap after each hold (0 keeps the cycle at 10 s).
DEFAULT_TASK_REPEATS = 9      # Repetitions per class PER HAND in one session.
DEFAULT_SESSION_DAY = 1
DEFAULT_HAND_ORDER = "left_right"
HAND_ORDER_OPTIONS = [
    ("left_right", "Left hand, then Right hand"),
    ("right_left", "Right hand, then Left hand"),
    ("left", "Left hand only"),
    ("right", "Right hand only"),
]
DEFAULT_MAX_CYCLES_PER_BLOCK = 30   # <= 30 cycles (~5 min) without a break
DEFAULT_BLOCK_BREAK_S = 210         # 3-4 min break between blocks
DEFAULT_HAND_SWITCH_BREAK_S = 300   # ~5 min when the armband moves to the other arm
STUDY_ENV_KEYS = ("temp_start_c", "humidity_start_pct")
DEFAULT_RECORD_CSV = "realtime_collected_emg.csv"
DEFAULT_RF_MODEL_ARTIFACT = "rf_realtime_model.joblib"
MIN_RECORD_SAMPLE_RATIO = MIN_VALID_SAMPLE_RATIO   # 0.98 of expected frames must be valid
USB_SERIAL_BAUD = 921600
SERIAL_BOOT_WAIT_S = 3.5
SERIAL_RESPONSE_TIMEOUT_S = 15.0
WIFI_STREAM_PORT = 5000
WIFI_CONTROL_PORT = 5001
PLACEHOLDER_DEVICE_ACCESS_KEY = "CHANGE_THIS_TO_A_LONG_RANDOM_KEY"
MIN_DEVICE_ACCESS_KEY_LEN = 16
DEVICE_SECRET_PATH = os.path.join(SCRIPT_DIR, "device_secret.json")   # gitignored; see device_secret.example.json
CONTROL_RETRY_BACKOFF_S = (0.2, 0.5, 1.0)   # waits before retry 1..3 of a control command
KEEPALIVE_RESTART_AFTER = 2                 # consecutive failed pings before START is re-issued
DISCOVERY_ADDRESS = "255.255.255.255"
DISCOVERY_TIMEOUT = 1.2
CONTROL_TIMEOUT = 2.0
KEEPALIVE_INTERVAL_MS = 2000
GESTURE_BRIDGE_HOST = "127.0.0.1"   # Predictions are mirrored here for the separate gesture_game_app.py
GESTURE_BRIDGE_PORT = 5005
UDP_RECV_BUFFER_BYTES = 4 * 1024 * 1024   # Kernel buffer: rides out UI stalls without dropping packets
STREAM_EMIT_INTERVAL_S = 0.020            # Receiver hands the UI one merged batch per ~20 ms
PACKET_TRACER_TICK_MS = 250
WIRELESS_EMG_CHANNELS = 8
WIRELESS_IMU_CHANNELS = 3
WIRELESS_TOTAL_CHANNELS = WIRELESS_EMG_CHANNELS + WIRELESS_IMU_CHANNELS
WIFI_PACKET_HEADER_FORMAT = "<4sBBHI"
WIFI_PACKET_HEADER_SIZE = struct.calcsize(WIFI_PACKET_HEADER_FORMAT)
WIRELESS_FRAME_FORMAT = "<IIII8HfffB3x"
WIRELESS_FRAME_SIZE = struct.calcsize(WIRELESS_FRAME_FORMAT)
WIRELESS_FRAMES_PER_PACKET = 5
WIRELESS_PACKET_SIZE = WIFI_PACKET_HEADER_SIZE + (WIRELESS_FRAME_SIZE * WIRELESS_FRAMES_PER_PACKET)

# --- REALTIME PREDICTION ---
RF_MAX_PREDICTION_HZ = 12         # Cap inference cadence; gestures don't change faster.
RF_MIN_SUBMIT_INTERVAL_S = 1.0 / RF_MAX_PREDICTION_HZ
RF_LABEL_SMOOTH_WINDOW = 5        # Majority-vote window for the displayed label.
RF_INFLIGHT_TIMEOUT_S = 1.0       # Self-heal if a submitted window never reports back.
# Training defaults tuned for real-time inference: far fewer/shallower trees than before
# (was 400 trees / unbounded depth) for ~4x faster prediction at negligible accuracy cost.
RF_DEFAULT_N_ESTIMATORS = 150
RF_DEFAULT_MAX_DEPTH = 16
RF_DEFAULT_MIN_SAMPLES_LEAF = 2
CAL_REST_MS = 3000               # Rest capture duration
CAL_FLEX_MS = 3000               # Flex capture duration
CAL_DURATION_MIN_S = 3           # Per-phase calibration duration lower bound
CAL_DURATION_MAX_S = 10          # Per-phase calibration duration upper bound
CAL_TICK_MS = 100                # Calibration UI timer tick
BASE_ADAPT_ALPHA = 0.001         # Slow baseline drift compensation in Python
BASE_ADAPT_GUARD = 80.0          # Update baseline only when near rest

# Cache Hanning windows / FFT bin frequencies by (length, rate); these never change per
# window size, so recomputing them for every analysis frame is wasted work.
_FFT_CONST_CACHE = {}


def fft_window_and_freqs(n_samples, sample_rate):
    key = (int(n_samples), float(sample_rate))
    cached = _FFT_CONST_CACHE.get(key)
    if cached is None:
        window = np.hanning(n_samples).astype(np.float32)
        freqs = np.fft.rfftfreq(int(n_samples), d=1.0 / float(sample_rate))
        cached = (window, freqs)
        _FFT_CONST_CACHE[key] = cached
    return cached

Y_MIN = -1200
Y_MAX = 1200
IMU_Y_MIN = -90
IMU_Y_MAX = 90
Y_AXIS_FIXED_WIDTH = 25           # Keep only a slim gutter when Y tick values are hidden
Y_AXIS_TICK_TEXT_WIDTH = 0       # No horizontal space needed for hidden Y tick values
LIVE_Y_PADDING_RATIO = 0.10
LIVE_Y_MIN_HALF_RANGE = 25.0
IMU_LIVE_Y_MIN_HALF_RANGE = 5.0
PLOT_AXIS_TICK_FONT_PT = 9
MONO_FONT_CSS = "Menlo, Consolas, 'Courier New', monospace"   # Menlo on macOS, Consolas on Windows


# Packed layout of one wireless frame - identical to struct format WIRELESS_FRAME_FORMAT.
_WIRELESS_FRAME_DTYPE = np.dtype([
    ("frame_id", "<u4"), ("frame_ts", "<u4"), ("imu_id", "<u4"), ("imu_ts", "<u4"),
    ("emg", "<u2", (WIRELESS_EMG_CHANNELS,)), ("imu", "<f4", (WIRELESS_IMU_CHANNELS,)),
    ("fresh", "u1"), ("pad", "V3"),
])
assert _WIRELESS_FRAME_DTYPE.itemsize == WIRELESS_FRAME_SIZE, "wireless frame layout mismatch"


class PacketTracer:
    """Transport health counters: the receive thread writes, the UI thread reads.

    Wireless: counts UDP datagrams and detects missing / out-of-order packets from the
    firmware's 32-bit packet sequence number. Wired: the text stream has no sequence
    number, so every malformed/truncated line is counted as one lost sample, and the
    achieved sample rate is compared against the nominal rate.
    All writers hold one tiny lock for a few integer updates per packet.
    """

    HISTORY = 120   # points kept for the tracer plots (HISTORY * tick = 30 s)

    def __init__(self, expected_sample_rate=SAMPLE_RATE):
        self._lock = threading.Lock()
        self.expected_sample_rate = float(expected_sample_rate)
        self.reset()

    def reset(self, kind="wireless", frames_per_packet=WIRELESS_FRAMES_PER_PACKET):
        with self._lock:
            self.kind = str(kind)
            self.frames_per_packet = max(1, int(frames_per_packet))
            self.stats = WirelessStats()
            self.asm = {}
            self.invalid_lines = 0
            self.samples = 0
            self.bytes = 0
            self.started = time.monotonic()
            self.last_arrival = 0.0
            self.max_gap_s = 0.0
            self.pkt_rate = 0.0
            self.sample_rate = 0.0
            self._points = deque(maxlen=5)
            self.rate_history = deque(maxlen=self.HISTORY)
            self.sample_rate_history = deque(maxlen=self.HISTORY)
            self.missing_history = deque(maxlen=self.HISTORY)

    # -- writers (receive thread) ------------------------------------------------
    def observe_packet(self, n_samples, n_bytes, arrival):
        """Record one received datagram/USB frame (arrival order; loss accounting comes from the assembler)."""
        with self._lock:
            self.stats.packets_received += 1
            if self.stats.last_arrival is not None:
                self.stats.arrival_intervals.append(arrival - self.stats.last_arrival)
            self.stats.last_arrival = arrival
            self.samples += int(n_samples)
            self.bytes += int(n_bytes)
            if self.last_arrival:
                self.max_gap_s = max(self.max_gap_s, arrival - self.last_arrival)
            self.last_arrival = arrival

    def sync_assembler(self, asm_stats):
        """Copy the sequencer's loss / late / duplicate counters (called by the receive thread)."""
        with self._lock:
            self.stats.packets_missing = int(asm_stats["packets_missing"])
            self.stats.packets_out_of_order = int(asm_stats["packets_reordered"] + asm_stats["packets_late"]
                                                  + asm_stats["packets_duplicate"])
            self.asm = dict(asm_stats)

    def observe_invalid_packet(self):
        with self._lock:
            self.stats.invalid_packets += 1

    def observe_serial(self, n_valid, n_bad, n_bytes, arrival):
        with self._lock:
            self.stats.packets_received += int(n_valid)
            self.invalid_lines += int(n_bad)
            self.samples += int(n_valid)
            self.bytes += int(n_bytes)
            if self.last_arrival:
                self.max_gap_s = max(self.max_gap_s, arrival - self.last_arrival)
            self.last_arrival = arrival

    # -- reader (UI thread) --------------------------------------------------------
    def tick(self):
        """Sample the counters once per UI tick to derive rates and history."""
        now = time.monotonic()
        with self._lock:
            self._points.append((now, self.stats.packets_received, self.samples))
            if len(self._points) >= 2:
                t0, p0, s0 = self._points[0]
                t1, p1, s1 = self._points[-1]
                dt = max(1e-6, t1 - t0)
                self.pkt_rate = (p1 - p0) / dt
                self.sample_rate = (s1 - s0) / dt
            self.rate_history.append(self.pkt_rate)
            self.sample_rate_history.append(self.sample_rate)
            self.missing_history.append(self._missing_locked())

    def _missing_locked(self):
        if self.kind == "wireless":
            return int(self.stats.packets_missing)
        return int(self.invalid_lines)

    def snapshot(self):
        with self._lock:
            received = int(self.stats.packets_received)
            missing = self._missing_locked()
            total = received + missing
            elapsed = max(1e-6, time.monotonic() - self.started)
            return {
                "kind": self.kind,
                "received": received,
                "missing": missing,
                "missing_samples": missing * (self.frames_per_packet if self.kind == "wireless" else 1),
                "out_of_order": int(self.stats.packets_out_of_order),
                "late": int(self.asm.get("packets_late", 0)),
                "duplicate": int(self.asm.get("packets_duplicate", 0)),
                "frames_missing": int(self.asm.get("frames_missing", 0)),
                "device_counters": dict(self.asm.get("device_counters", {})),
                "invalid": int(self.stats.invalid_packets) + int(self.invalid_lines),
                "loss_pct": (100.0 * missing / total) if total else 0.0,
                "pkt_rate": float(self.pkt_rate),
                "sample_rate": float(self.sample_rate),
                "avg_sample_rate": self.samples / elapsed,
                "expected_pkt_rate": (self.expected_sample_rate / self.frames_per_packet) if self.kind == "wireless" else self.expected_sample_rate,
                "expected_sample_rate": self.expected_sample_rate,
                "jitter_ms": float(self.stats.jitter_ms),
                "max_gap_ms": self.max_gap_s * 1000.0,
                "samples": int(self.samples),
                "bytes": int(self.bytes),
                "uptime_s": elapsed,
                "rate_history": list(self.rate_history),
                "missing_history": list(self.missing_history),
            }


class SerialWorker(QThread):
    """Wired receiver.

    A real COM port carries the same BWIM v2 packets as Wi-Fi, COBS-framed with a CRC-16, so the
    wired stream has packet sequence numbers, 64-bit device timestamps and all 11 channels. The
    app sends USBSTART after opening the port (repeated until data flows), USBPING every 2 s as a
    keepalive and USBSTOP on shutdown. ``socket://`` URLs (the simulator) keep the legacy CSV text.
    """
    batch_received = pyqtSignal(object)  # dict: batch (n, channels), packet_numbers, source
    error_occurred = pyqtSignal(str)

    FLUSH_INTERVAL_S = 0.040

    def __init__(self, port_name, baud_rate, num_channels, batch_size=SERIAL_BATCH_SIZE, tracer=None,
                 binary=None):
        super().__init__()
        self.port_name = port_name
        self.baud_rate = baud_rate
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.tracer = tracer
        self.is_socket_url = str(port_name).strip().lower().startswith("socket://")
        self.binary = (not self.is_socket_url) if binary is None else bool(binary)
        self._running = True
        self._serial = None
        self._packet_sequence_counter = 0
        self._assembler = StreamAssembler()

    def _emit_batch(self, batch):
        arr = np.asarray(batch, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] <= 0:
            return
        packet_no = int(self._packet_sequence_counter)
        self._packet_sequence_counter += 1
        packet_numbers = np.full(arr.shape[0], packet_no, dtype=np.int64)
        self.batch_received.emit({"batch": arr, "packet_numbers": packet_numbers, "source": "serial"})

    def _close_serial(self, send_stop=False):
        try:
            if self._serial and self._serial.is_open:
                if send_stop and self.binary:
                    try:
                        self._serial.write(b"USBSTOP\n")
                        self._serial.flush()
                    except Exception:
                        pass
                self._serial.close()
        except Exception:
            pass
        self._serial = None

    def _open_serial(self):
        # Supports both COM ports (e.g. COM3) and URL handlers (e.g. socket://127.0.0.1:7000).
        ser = serial.serial_for_url(self.port_name, self.baud_rate, timeout=0.02, write_timeout=1.0, do_not_open=True)
        if self.binary:
            try:
                ser.dtr = False   # do not reset the ESP32 when the port opens
                ser.rts = False
            except Exception:
                pass
        ser.open()
        try:
            ser.set_buffer_size(rx_size=262144)   # Windows only; headroom against UI stalls
        except Exception:
            pass
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
        return ser

    # -- binary (BWIM v2 over COBS) ------------------------------------------------------
    def _run_binary(self):
        decoder = UsbFrameDecoder()
        asm = self._assembler = StreamAssembler()
        pending, last_flush = [], time.monotonic()
        last_command, last_packet = 0.0, 0.0
        while self._running:
            try:
                if self._serial is None or not self._serial.is_open:
                    self._serial = self._open_serial()
                    decoder = UsbFrameDecoder()
                    last_command = last_packet = 0.0
                now = time.monotonic()
                if now - last_command >= USB_KEEPALIVE_INTERVAL_S:
                    # USBSTART (re)starts the stream until packets flow; USBPING keeps it alive after.
                    streaming = last_packet and (now - last_packet) < USB_KEEPALIVE_INTERVAL_S
                    self._serial.write(b"USBPING\n" if streaming else b"USBSTART\n")
                    last_command = now
                chunk = self._serial.read(max(1, self._serial.in_waiting))
                now = time.monotonic()
                if chunk:
                    packets, bad = decoder.feed(chunk)
                    if self.tracer is not None:
                        for _ in range(bad):
                            self.tracer.observe_invalid_packet()
                    for pkt in packets:
                        last_packet = now
                        if self.tracer is not None:
                            self.tracer.observe_packet(len(pkt.frames), 0, now)
                        pending.extend(asm.push(pkt, now))
                pending.extend(asm.poll(now))
                if pending and (now - last_flush) >= STREAM_EMIT_INTERVAL_S:
                    self.batch_received.emit(segments_to_payload(pending, now, "wired_binary"))
                    if self.tracer is not None:
                        self.tracer.sync_assembler(asm.stats())
                    pending, last_flush = [], now
            except Exception as e:
                if not self._running:
                    break
                self._close_serial()
                self.error_occurred.emit(str(e))
                break
        pending.extend(asm.flush())
        if pending:
            self.batch_received.emit(segments_to_payload(pending, time.monotonic(), "wired_binary"))
            if self.tracer is not None:
                self.tracer.sync_assembler(asm.stats())

    # -- legacy CSV text -------------------------------------------------------------------
    def _run_csv(self):
        nch = int(self.num_channels)
        tail = b""
        batch = []
        last_flush = time.monotonic()
        while self._running:
            try:
                if self._serial is None or not self._serial.is_open:
                    self._serial = self._open_serial()
                    tail = b""

                chunk = self._serial.read(max(1, self._serial.in_waiting))
                now = time.monotonic()
                if chunk:
                    lines = (tail + chunk).split(b"\n")
                    tail = lines.pop()  # Keep incomplete tail for next read.
                    good = bad = 0
                    for raw_line in lines:
                        parts = raw_line.replace(b",", b" ").split()
                        if not parts:
                            continue
                        if len(parts) < nch:
                            bad += 1
                            continue
                        try:
                            batch.append([float(x) for x in parts[:nch]])
                            good += 1
                        except ValueError:
                            bad += 1
                    if self.tracer is not None and (good or bad):
                        self.tracer.observe_serial(good, bad, len(chunk), now)

                if batch and (len(batch) >= self.batch_size or (now - last_flush) >= self.FLUSH_INTERVAL_S):
                    self._emit_batch(batch)
                    batch = []
                    last_flush = now
            except Exception as e:
                if not self._running:
                    break
                self._close_serial()
                tail = b""
                if self.is_socket_url:
                    # Auto-recover TCP socket disconnects without hard-failing the app.
                    time.sleep(0.3)
                    continue
                self.error_occurred.emit(str(e))
                break
        if batch:
            self._emit_batch(batch)

    def run(self):
        try:
            if self.binary:
                self._run_binary()
            else:
                self._run_csv()
        finally:
            self._close_serial(send_stop=True)

    def stop(self):
        self._running = False
        self.wait()


def _parse_rate_field(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


@dataclass
class DeviceInfo:
    ip: str
    device_id: str
    device_name: str
    wifi_mode: str
    reported_ip: str
    imu_ready: bool
    streaming: bool
    firmware: str
    sample_rate_hz: Optional[float] = None

    @property
    def summary(self):
        return f"{self.device_name} @ {self.ip} ({self.wifi_mode})"


@dataclass
class SerialDeviceInfo:
    port_name: str
    device_id: str
    device_name: str
    imu_ready: bool
    wifi_saved: bool
    firmware: str
    sample_rate_hz: Optional[float] = None


def access_key_problem(key):
    """Why ``key`` must not be used to talk to a device, or "" when it is acceptable."""
    key = str(key or "").strip()
    if not key:
        return "No device access key is set. Put it in code/device_secret.json (see device_secret.example.json)."
    if key == PLACEHOLDER_DEVICE_ACCESS_KEY:
        return "The device access key is still the placeholder. Set a real key in secrets.h (firmware) and code/device_secret.json."
    if len(key) < MIN_DEVICE_ACCESS_KEY_LEN:
        return f"The device access key is too short (minimum {MIN_DEVICE_ACCESS_KEY_LEN} characters)."
    return ""


def load_device_access_key(path=DEVICE_SECRET_PATH):
    """Key from the gitignored device_secret.json, or "" when it is missing or unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return str(json.load(fh).get("device_access_key", "")).strip()
    except (OSError, ValueError, AttributeError):
        return ""


def sign_message(secret, *parts):
    message = "|".join(str(part) for part in parts)
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def get_local_ip_for_target(target_ip):
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect((target_ip, 1))
        return probe.getsockname()[0]
    finally:
        probe.close()


def run_command_text(command):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
        )
    except Exception:
        return ""
    return result.stdout or ""


def get_connected_ssid():
    if sys.platform != "win32":
        return ""

    output = run_command_text(["netsh", "wlan", "show", "interfaces"])
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("SSID") and "BSSID" not in stripped and ":" in stripped:
            return stripped.split(":", 1)[1].strip()
    return ""


class ControlProtocol:
    @staticmethod
    def parse_device_info(message, source_ip):
        parts = message.strip().split("|")
        # 8 fields from firmware < 2.0; 2.0+ appends the sampling rate (Hz).
        if len(parts) not in (8, 9) or parts[0] != "HELLO":
            raise ValueError("Unexpected device response.")

        return DeviceInfo(
            ip=source_ip,
            device_id=parts[1],
            device_name=parts[2],
            wifi_mode=parts[3],
            reported_ip=parts[4],
            imu_ready=parts[5] == "1",
            streaming=parts[6] == "1",
            firmware=parts[7],
            sample_rate_hz=_parse_rate_field(parts[8]) if len(parts) > 8 else None,
        )

    @staticmethod
    def send_and_receive(message, target_ip, expect_multiple=False, timeout=CONTROL_TIMEOUT, broadcast=False):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.2 if expect_multiple else timeout)
        if broadcast:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        responses = []
        deadline = time.monotonic() + timeout
        try:
            sock.sendto(message.encode("utf-8"), (target_ip, WIFI_CONTROL_PORT))
            if expect_multiple:
                while time.monotonic() < deadline:
                    try:
                        data, addr = sock.recvfrom(2048)
                        responses.append((data.decode("utf-8", errors="replace"), addr[0]))
                    except socket.timeout:
                        continue
                return responses

            data, addr = sock.recvfrom(2048)
            return data.decode("utf-8", errors="replace"), addr[0]
        finally:
            sock.close()

    @staticmethod
    def discover():
        devices = {}
        responses = ControlProtocol.send_and_receive(
            "DISCOVER",
            DISCOVERY_ADDRESS,
            expect_multiple=True,
            timeout=DISCOVERY_TIMEOUT,
            broadcast=True,
        )
        for response, source_ip in responses:
            try:
                device = ControlProtocol.parse_device_info(response, source_ip)
                devices[device.ip] = device
            except ValueError:
                continue
        return list(devices.values())

    @staticmethod
    def get_challenge(target_ip, timeout=None):
        response, _ = ControlProtocol.send_and_receive("CHALLENGE", target_ip,
                                                       timeout=CONTROL_TIMEOUT if timeout is None else timeout)
        parts = response.strip().split("|")
        if len(parts) != 2 or parts[0] != "CHALLENGE":
            raise RuntimeError("Device did not return a valid challenge.")
        return parts[1]

    @staticmethod
    def _authenticated_once(target_ip, secret, command, payload, timeout):
        # The firmware ties a challenge to the sender's (ip, port), so the CHALLENGE request and the
        # signed command must leave from the SAME socket.
        timeout = CONTROL_TIMEOUT if timeout is None else timeout
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        try:
            destination = (target_ip, WIFI_CONTROL_PORT)
            sock.sendto(b"CHALLENGE", destination)
            parts = sock.recvfrom(2048)[0].decode("utf-8", errors="replace").strip().split("|")
            if len(parts) != 2 or parts[0] != "CHALLENGE":
                raise RuntimeError("Device did not return a valid challenge.")
            challenge = parts[1]
            auth = sign_message(secret, command, challenge, *payload)
            sock.sendto("|".join([command, challenge, *payload, auth]).encode("utf-8"), destination)
            parts = sock.recvfrom(2048)[0].decode("utf-8", errors="replace").strip().split("|")
        finally:
            sock.close()

        if parts[0] == "ERR":
            detail = parts[1] if len(parts) > 1 else "UNKNOWN"
            raise RuntimeError(f"Device rejected command: {detail}")
        if parts[0] != "ACK":
            raise RuntimeError("Unexpected device acknowledgement.")
        return parts[1:]

    @staticmethod
    def authenticated_command(target_ip, secret, command, *payload, timeout=None, backoff=CONTROL_RETRY_BACKOFF_S):
        """Run one signed command, retrying lost datagrams with a fresh challenge each attempt.

        UDP loses packets and a device can answer a challenge for another client, so up to
        ``len(backoff)`` retries are made (0.2 / 0.5 / 1.0 s by default). Blocking: call it from a
        worker thread, never from the GUI thread when it can be avoided.
        """
        problem = access_key_problem(secret)
        if problem:
            raise RuntimeError(problem)

        last_error = None
        for attempt in range(len(backoff) + 1):
            if attempt:
                time.sleep(backoff[attempt - 1])
            try:
                return ControlProtocol._authenticated_once(target_ip, secret, command, payload, timeout)
            except (OSError, RuntimeError) as exc:      # socket.timeout is an OSError
                last_error = exc
        raise RuntimeError(f"{command} failed after {len(backoff) + 1} attempts: {last_error}")

    @staticmethod
    def start_stream(target_ip, secret, client_ip, client_port):
        return ControlProtocol.authenticated_command(
            target_ip,
            secret,
            "START",
            client_ip,
            str(client_port),
        )

    @staticmethod
    def stop_stream(target_ip, secret):
        return ControlProtocol.authenticated_command(target_ip, secret, "STOP")

    @staticmethod
    def ping(target_ip, secret):
        return ControlProtocol.authenticated_command(target_ip, secret, "PING")


class WiFiSerialProvisionProtocol:
    @staticmethod
    def available_ports():
        return list(serial.tools.list_ports.comports())

    @staticmethod
    def _exchange_line(port_name, command, expected_prefixes, timeout=SERIAL_RESPONSE_TIMEOUT_S):
        try:
            with serial.Serial(port_name, USB_SERIAL_BAUD, timeout=0.3, write_timeout=1) as ser:
                ser.setDTR(False)
                ser.setRTS(False)
                time.sleep(0.15)
                time.sleep(SERIAL_BOOT_WAIT_S)
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                ser.write((command + "\n").encode("utf-8"))
                ser.flush()

                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    raw_line = ser.readline()
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    if any(line.startswith(prefix) for prefix in expected_prefixes):
                        return line
        except serial.SerialException as exc:
            raise RuntimeError(f"Serial communication failed on {port_name}: {exc}") from exc

        raise RuntimeError("The ESP32 did not return a serial response in time.")

    @staticmethod
    def query_info(port_name):
        response = WiFiSerialProvisionProtocol._exchange_line(port_name, "INFO", expected_prefixes=("INFO|", "ERR|"))
        parts = response.split("|")
        if len(parts) >= 2 and parts[0] == "ERR":
            raise RuntimeError(f"ESP32 returned an error: {parts[1]}")
        if len(parts) not in (6, 7) or parts[0] != "INFO":
            raise RuntimeError(f"Unexpected serial response: {response}")

        return SerialDeviceInfo(
            port_name=port_name,
            device_id=parts[1],
            device_name=parts[2],
            imu_ready=parts[3] == "1",
            wifi_saved=parts[4] == "1",
            firmware=parts[5],
            sample_rate_hz=_parse_rate_field(parts[6]) if len(parts) > 6 else None,
        )

    @staticmethod
    def provision(port_name, ssid, password):
        encoded_ssid = quote(ssid, safe="")
        encoded_password = quote(password, safe="")
        response = WiFiSerialProvisionProtocol._exchange_line(
            port_name,
            f"PROVISION|{encoded_ssid}|{encoded_password}",
            expected_prefixes=("ACK|", "ERR|"),
            timeout=8.0,
        )
        parts = response.split("|")
        if len(parts) >= 2 and parts[0] == "ACK" and parts[1] == "PROVISIONED":
            return
        if len(parts) >= 2 and parts[0] == "ERR":
            raise RuntimeError(f"ESP32 rejected provisioning: {parts[1]}")
        raise RuntimeError(f"Unexpected serial response: {response}")


class WirelessStreamWorker(QThread):
    """UDP receiver. Its only jobs are to drain the socket quickly and sequence packets.

    * A 4 MB kernel receive buffer rides out UI stalls (a UDP datagram that arrives while
      the buffer is full is lost for good, so the buffer is the first line of defence).
    * Runs at time-critical thread priority and does no per-sample Python work: a packet
      is decoded with one numpy view instead of a struct.unpack per frame.
    * A small reorder buffer (5 packets / 50 ms) releases packets in sequence order; lost
      packets become explicit NaN frames with the correct frame IDs and Valid=0, and late or
      duplicate packets are counted but never inserted.
    * Segments are merged for ~20 ms before being handed to the UI, so the main thread
      does ~50 updates/s instead of one per datagram.
    """
    batch_received = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, port=WIFI_STREAM_PORT, tracer=None):
        super().__init__()
        self.port = int(port)
        self._running = True
        self._sock = None
        self._fallback_packet_sequence = 0
        self.tracer = tracer if tracer is not None else PacketTracer()
        self.stats = self.tracer.stats
        self.assembler = StreamAssembler()

    @staticmethod
    def _frames_to_batch(frames):
        batch = np.empty((frames.shape[0], WIRELESS_TOTAL_CHANNELS), dtype=np.float32)
        batch[:, :WIRELESS_EMG_CHANNELS] = frames["emg"]
        batch[:, WIRELESS_EMG_CHANNELS:] = frames["imu"]
        return batch

    def _legacy_payload(self, frames, source):
        batch = self._frames_to_batch(frames)
        packet_no = int(self._fallback_packet_sequence)
        self._fallback_packet_sequence += 1
        return {"batch": batch, "packet_numbers": np.full(batch.shape[0], packet_no, dtype=np.int64),
                "source": source}

    def _handle_datagram(self, data, arrival):
        """Returns (segments, legacy_payload); at most one of them is non-empty."""
        size = len(data)
        pkt = parse_bwim_packet(data)
        if pkt is not None:
            self.tracer.observe_packet(len(pkt.frames), size, arrival)
            return self.assembler.push(pkt, arrival), None

        if size == WIRELESS_FRAME_SIZE * WIRELESS_FRAMES_PER_PACKET:
            frames = np.frombuffer(data, dtype=_WIRELESS_FRAME_DTYPE, count=WIRELESS_FRAMES_PER_PACKET)
            return [], self._legacy_payload(frames, "wireless_legacy")
        if size == WIRELESS_FRAME_SIZE:
            frames = np.frombuffer(data, dtype=_WIRELESS_FRAME_DTYPE, count=1)
            return [], self._legacy_payload(frames, "wireless_single")

        self.tracer.observe_invalid_packet()
        return [], None

    def _emit_segments(self, segments, now):
        if segments:
            self.batch_received.emit(segments_to_payload(segments, now, "wireless"))
        self.tracer.sync_assembler(self.assembler.stats())

    def run(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, UDP_RECV_BUFFER_BYTES)
        except OSError:
            pass
        self._sock.bind(("0.0.0.0", self.port))
        self._sock.settimeout(0.02)

        pending = []
        last_emit = time.monotonic()
        try:
            while self._running:
                try:
                    data, _addr = self._sock.recvfrom(2048)
                except socket.timeout:
                    data = None
                except OSError:
                    break

                now = time.monotonic()
                if data is not None:
                    segments, legacy = self._handle_datagram(data, now)
                    pending.extend(segments)
                    if legacy is not None:
                        self._emit_segments(pending, now); pending = []
                        self.batch_received.emit(legacy)
                pending.extend(self.assembler.poll(now))

                if pending and (data is None or (now - last_emit) >= STREAM_EMIT_INTERVAL_S):
                    self._emit_segments(pending, now)
                    pending = []
                    last_emit = now
            pending.extend(self.assembler.flush())
            self._emit_segments(pending, time.monotonic())
        except Exception as exc:
            if self._running:
                self.error_occurred.emit(f"Wireless stream error: {exc}")
        finally:
            try:
                if self._sock is not None:
                    self._sock.close()
            except Exception:
                pass
            self._sock = None

    def stop(self):
        self._running = False
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        self.wait(2000)


class ConnectionModeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        apply_app_icon(self)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowTitle("Connection Medium")
        self.setModal(True)
        self.resize(420, 230)
        apply_dark_title_bar(self)
        self.selected_medium = ""

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        lbl_title = QLabel("Choose how this BioWave session will connect.")
        lbl_title.setWordWrap(True)
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(lbl_title)

        lbl_note = QLabel(
            "Wired keeps the original COM/socket workflow. Wireless opens Wi-Fi provisioning and device control."
        )
        lbl_note.setWordWrap(True)
        lbl_note.setStyleSheet(themed_label_style("muted"))
        layout.addWidget(lbl_note)
        layout.addStretch()

        btn_row = QHBoxLayout()
        self.btn_wired = QPushButton("Wired")
        self.btn_wired.setMinimumHeight(48)
        self.btn_wired.setStyleSheet(themed_button_style("accent"))
        self.btn_wired.clicked.connect(lambda: self._select_medium("wired"))
        btn_row.addWidget(self.btn_wired)

        self.btn_wireless = QPushButton("Wireless")
        self.btn_wireless.setMinimumHeight(48)
        self.btn_wireless.setStyleSheet(themed_button_style("success"))
        self.btn_wireless.clicked.connect(lambda: self._select_medium("wireless"))
        btn_row.addWidget(self.btn_wireless)
        layout.addLayout(btn_row)

    def _select_medium(self, medium):
        self.selected_medium = str(medium)
        self.accept()


class USBProvisionDialog(QDialog):
    def __init__(self, parent=None, initial_ssid=""):
        super().__init__(parent)
        apply_app_icon(self)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowTitle("Wireless USB Provisioning")
        self.setModal(True)
        self.resize(620, 420)
        apply_dark_title_bar(self)
        self.selected_port = ""
        self.device_info = None

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        intro = QLabel(
            "Connect the ESP32-S3 with USB, probe the port, then send Wi-Fi credentials for wireless mode."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.current_wifi_label = QLabel(f"Current PC Wi-Fi: {get_connected_ssid() or 'Not detected'}")
        self.current_wifi_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.current_wifi_label)

        row_port = QHBoxLayout()
        row_port.addWidget(QLabel("Serial Port:"))
        self.port_selector = QComboBox()
        self.port_selector.setEditable(True)
        row_port.addWidget(self.port_selector, stretch=1)
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setStyleSheet(themed_button_style("accent"))
        self.btn_refresh.clicked.connect(self.refresh_ports)
        row_port.addWidget(self.btn_refresh)
        layout.addLayout(row_port)

        self.port_hint_label = QLabel("Serial device: not probed yet")
        self.port_hint_label.setWordWrap(True)
        self.port_hint_label.setStyleSheet(themed_label_style("muted"))
        layout.addWidget(self.port_hint_label)

        form = QFormLayout()
        self.ssid_input = QLineEdit(initial_ssid or get_connected_ssid())
        self.pass_input = QLineEdit()
        self.pass_input.setEchoMode(QLineEdit.Password)
        form.addRow("Home Wi-Fi SSID:", self.ssid_input)
        form.addRow("Home Wi-Fi Password:", self.pass_input)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.btn_probe = QPushButton("Probe Selected Port")
        self.btn_probe.setStyleSheet(themed_button_style("accent"))
        self.btn_probe.clicked.connect(self.probe_selected_port)
        btn_row.addWidget(self.btn_probe)

        self.btn_send = QPushButton("Send Wi-Fi to ESP")
        self.btn_send.setStyleSheet(themed_button_style("success"))
        self.btn_send.clicked.connect(self.send_credentials)
        btn_row.addWidget(self.btn_send)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(themed_button_style("muted"))
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

        self.status_label = QLabel("Select the ESP32 serial port, probe it, then send the Wi-Fi credentials.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(themed_label_style("muted"))
        layout.addWidget(self.status_label)

        self.refresh_ports()

    def selected_port_name(self):
        data = self.port_selector.currentData()
        if data:
            return str(data).strip()
        return self.port_selector.currentText().strip()

    def refresh_ports(self):
        selected = self.selected_port_name()
        self.port_selector.clear()
        ports = WiFiSerialProvisionProtocol.available_ports()
        if not ports:
            self.status_label.setText("No serial ports found. Plug in the ESP32-S3 and refresh again.")
            return

        for port in ports:
            self.port_selector.addItem(f"{port.device} - {port.description}", port.device)

        if selected:
            idx = self.port_selector.findData(selected)
            if idx >= 0:
                self.port_selector.setCurrentIndex(idx)
            else:
                self.port_selector.setCurrentText(selected)

    def probe_selected_port(self):
        port_name = self.selected_port_name()
        if not port_name:
            QMessageBox.warning(self, "No Serial Port Selected", "Select the ESP32 serial port first.")
            return

        try:
            self.device_info = WiFiSerialProvisionProtocol.query_info(port_name)
        except Exception as exc:
            self.device_info = None
            QMessageBox.warning(self, "Probe Failed", str(exc))
            self.port_hint_label.setText(f"Selected serial port: {port_name}")
            return

        self.selected_port = port_name
        self.port_hint_label.setText(
            f"Detected {self.device_info.device_name} on {port_name} | IMU ready={self.device_info.imu_ready} | "
            f"Wi-Fi saved={self.device_info.wifi_saved} | FW={self.device_info.firmware}"
        )
        self.status_label.setText("ESP32 responded over USB. You can now send the Wi-Fi credentials.")

    def send_credentials(self):
        port_name = self.selected_port_name()
        if not port_name:
            QMessageBox.warning(self, "No Serial Port Selected", "Select the ESP32 serial port first.")
            return

        ssid = self.ssid_input.text().strip()
        password = self.pass_input.text()
        if not ssid:
            QMessageBox.warning(self, "Missing SSID", "Home Wi-Fi SSID cannot be empty.")
            return

        try:
            WiFiSerialProvisionProtocol.provision(port_name, ssid, password)
        except Exception as exc:
            QMessageBox.critical(self, "USB Provisioning Failed", str(exc))
            return

        QMessageBox.information(
            self,
            "Provisioned",
            "The ESP32 accepted the Wi-Fi credentials over USB and will restart.\n\n"
            "After it joins Wi-Fi, discover it from the Wireless configuration window.",
        )
        self.accept()


class WirelessConfigDialog(QDialog):
    connect_requested = pyqtSignal(object, str)
    discover_requested = pyqtSignal()
    provision_requested = pyqtSignal()

    def __init__(self, access_key="", parent=None):
        super().__init__(parent)
        apply_app_icon(self)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowTitle("Wireless Configuration")
        self.setModal(True)
        self.resize(700, 360)
        apply_dark_title_bar(self)
        self.devices = []
        self.connected_ok = False

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        intro = QLabel(
            "Provision the ESP32 over USB if needed, discover it on the current Wi-Fi, then connect the wireless stream."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.lbl_hint = QLabel(
            "Wireless stream uses the BioWave Wi-Fi control plane and the 11-channel EMG + IMU UDP data stream."
        )
        self.lbl_hint.setWordWrap(True)
        self.lbl_hint.setStyleSheet(themed_label_style("muted"))
        layout.addWidget(self.lbl_hint)

        row_device = QHBoxLayout()
        row_device.addWidget(QLabel("Device:"))
        self.device_selector = QComboBox()
        row_device.addWidget(self.device_selector, stretch=1)
        self.btn_discover = QPushButton("Discover Device")
        self.btn_discover.setStyleSheet(themed_button_style("accent"))
        self.btn_discover.clicked.connect(self.discover_requested.emit)
        row_device.addWidget(self.btn_discover)
        layout.addLayout(row_device)

        self.device_info_label = QLabel("No wireless device selected.")
        self.device_info_label.setWordWrap(True)
        self.device_info_label.setStyleSheet(themed_label_style("muted"))
        layout.addWidget(self.device_info_label)

        row_key = QHBoxLayout()
        row_key.addWidget(QLabel("Access Key:"))
        self.access_key_input = QLineEdit(access_key)
        self.access_key_input.setEchoMode(QLineEdit.Password)
        self.access_key_input.setPlaceholderText("Device access key")
        row_key.addWidget(self.access_key_input, stretch=1)
        layout.addLayout(row_key)

        btn_row = QHBoxLayout()
        self.btn_provision = QPushButton("Provision via USB")
        self.btn_provision.setStyleSheet(themed_button_style("accent"))
        self.btn_provision.clicked.connect(self.provision_requested.emit)
        btn_row.addWidget(self.btn_provision)

        self.btn_connect = QPushButton("Connect Wireless")
        self.btn_connect.setStyleSheet(themed_button_style("success"))
        self.btn_connect.clicked.connect(self.on_connect_clicked)
        btn_row.addWidget(self.btn_connect)

        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setStyleSheet(themed_button_style("accent"))
        self.btn_apply.clicked.connect(self.accept)
        self.btn_apply.setEnabled(False)
        btn_row.addWidget(self.btn_apply)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(themed_button_style("muted"))
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

        self.status_label = QLabel("Discover a device and connect the wireless stream before applying.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(themed_label_style("muted"))
        layout.addWidget(self.status_label)

        self.device_selector.currentIndexChanged.connect(self._refresh_device_info)

    def set_devices(self, devices, selected_ip=""):
        self.devices = list(devices or [])
        self.device_selector.clear()
        for device in self.devices:
            self.device_selector.addItem(device.summary, device)

        if selected_ip:
            for index, device in enumerate(self.devices):
                if device.ip == selected_ip:
                    self.device_selector.setCurrentIndex(index)
                    break
        self._refresh_device_info()

    def selected_device(self):
        data = self.device_selector.currentData()
        return data if isinstance(data, DeviceInfo) else None

    def _refresh_device_info(self):
        device = self.selected_device()
        if device is None:
            self.device_info_label.setText("No wireless device selected.")
            return
        self.device_info_label.setText(
            f"{device.device_name} | IP={device.ip} | Mode={device.wifi_mode} | FW={device.firmware} | "
            f"IMU ready={device.imu_ready}"
        )

    def on_connect_clicked(self):
        device = self.selected_device()
        if device is None:
            QMessageBox.warning(self, "No Device", "Discover and select a BioWave wireless device first.")
            return
        access_key = self.access_key_input.text().strip()
        if not access_key:
            QMessageBox.warning(self, "Missing Access Key", "Enter the device access key before connecting.")
            return
        self.connect_requested.emit(device, access_key)

    def set_connection_state(self, is_connected, message=""):
        self.connected_ok = bool(is_connected)
        self.btn_apply.setEnabled(self.connected_ok)
        if self.connected_ok:
            self.status_label.setText(message or "Wireless stream connected. Click Apply to unlock the workflow.")
            self.status_label.setStyleSheet(themed_label_style("success"))
        else:
            self.status_label.setText(message or "Discover a device and connect the wireless stream before applying.")
            self.status_label.setStyleSheet(themed_label_style("muted"))


class RFRealtimeInferenceWorker(QThread):
    prediction_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, sample_rate=SAMPLE_RATE):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self._running = True
        self._lock = threading.Lock()
        self._pending_event = threading.Event()
        self._model = None
        self._class_names = []
        self._window_batch = None

    def set_model(self, model, class_names, sample_rate=None):
        with self._lock:
            self._model = model
            self._class_names = [str(x) for x in list(class_names or [])]
            if sample_rate is not None:
                self.sample_rate = int(max(1, sample_rate))
            self._window_batch = None
        self._pending_event.clear()

    def clear_model(self):
        with self._lock:
            self._model = None
            self._class_names = []
            self._window_batch = None
        self._pending_event.clear()

    def submit_window(self, window_batch):
        if window_batch is None:
            return
        arr = np.asarray(window_batch, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
            return
        with self._lock:
            self._window_batch = arr
        self._pending_event.set()

    def stop(self):
        self._running = False
        self._pending_event.set()
        self.wait()

    def run(self):
        while self._running:
            self._pending_event.wait(0.2)
            if not self._running:
                break
            if not self._pending_event.is_set():
                continue
            self._pending_event.clear()

            with self._lock:
                model = self._model
                class_names = list(self._class_names)
                window_batch = self._window_batch
                self._window_batch = None

            if model is None or window_batch is None:
                continue
            if not HAS_RF_FEATURES or rf_extract_window_features is None:
                continue

            try:
                t0 = time.perf_counter()
                feats = rf_extract_window_features(window_batch, sample_rate=self.sample_rate).reshape(1, -1)

                confidences = np.zeros(len(class_names), dtype=np.float32)
                conf = 0.0
                if hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(feats)[0]
                        model_classes = list(getattr(model, "classes_", []))
                        if len(model_classes) == len(proba) and len(class_names) > 0:
                            for i, cls_id in enumerate(model_classes):
                                idx = -1
                                try:
                                    idx = int(cls_id)
                                except Exception:
                                    cls_text = str(cls_id)
                                    if cls_text in class_names:
                                        idx = class_names.index(cls_text)
                                if 0 <= idx < len(confidences):
                                    confidences[idx] = float(proba[i])
                            best_i = int(np.argmax(proba))
                            best_cls = model_classes[best_i]
                            pred_idx = -1
                            try:
                                pred_idx = int(best_cls)
                            except Exception:
                                cls_text = str(best_cls)
                                if cls_text in class_names:
                                    pred_idx = class_names.index(cls_text)
                            if 0 <= pred_idx < len(class_names):
                                pred_label = class_names[pred_idx]
                            else:
                                pred_label = str(best_cls)
                        elif len(proba) == len(confidences):
                            confidences = np.asarray(proba, dtype=np.float32)
                            pred_idx = int(np.argmax(confidences))
                            pred_label = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else "N/A"
                        else:
                            pred_idx = -1
                            pred_label = "N/A"
                        conf = float(np.max(confidences)) if len(confidences) > 0 else float(np.max(proba))
                    except Exception:
                        pred_idx = -1
                        pred_label = "N/A"
                        conf = 0.0
                else:
                    pred_raw = model.predict(feats)[0]
                    pred_idx = -1
                    if isinstance(pred_raw, (np.integer, int)) and 0 <= int(pred_raw) < len(class_names):
                        pred_idx = int(pred_raw)
                        pred_label = class_names[pred_idx]
                    else:
                        pred_label = str(pred_raw)
                        if pred_label in class_names:
                            pred_idx = class_names.index(pred_label)

                if np.max(confidences) <= 0 and 0 <= pred_idx < len(confidences):
                    confidences[pred_idx] = 1.0
                    if conf <= 0.0:
                        conf = 1.0

                t1 = time.perf_counter()
                self.prediction_ready.emit(
                    {
                        "pred_label": pred_label,
                        "pred_conf": float(conf),
                        "class_confidences": confidences,
                        "latency_ms": float(max(0.0, (t1 - t0) * 1000.0)),
                        "completed_ts": float(t1),
                    }
                )
            except Exception as e:
                self.error_occurred.emit(str(e))


class SessionFinalizeWorker(QThread):
    """Drains the recorder, then writes signals_/events_/quality_ files and the metadata text.

    Runs off the GUI thread: a checkpoint (``close_recorder=False``) happens at every break, the
    final pass (``close_recorder=True``) when the protocol ends or is cancelled.
    """
    save_completed = pyqtSignal(object)

    def __init__(self, recorder, events, out_dir, session, fs, transport, keepalive_restarts, host_origin_s,
                 metadata_path, metadata_builder, close_recorder, parent=None):
        super().__init__(parent)
        self._args = (recorder, events, out_dir, session, fs, transport, keepalive_restarts, host_origin_s)
        self._metadata_path, self._metadata_builder, self._close = metadata_path, metadata_builder, bool(close_recorder)

    def run(self):
        recorder, events, out_dir, session, fs, transport, restarts, origin = self._args
        try:
            rows = recorder.close() if self._close else recorder.checkpoint()
            if recorder.error:
                raise RuntimeError(f"recorder writer failed: {recorder.error}")
            stats = {"valid_rows": recorder.valid_rows, "clip_counts": recorder.clip_counts.copy(),
                     "device_counters": dict(recorder.device_counters), "late_rows": recorder.late_rows}
            result = finalize_session(recorder.raw_path, rows, events, recorder.arrival_index(), out_dir, session,
                                      fs=fs, transport=transport, recorder_stats=stats,
                                      keepalive_restarts=restarts, host_origin_s=origin)
            with open(self._metadata_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(self._metadata_builder(rows, result["quality"])) + "\n")
            self.save_completed.emit({"ok": True, "data_csv_path": result["signals_path"],
                                      "metadata_path": self._metadata_path, "quality": result["quality"],
                                      "events_path": result["events_path"], "quality_path": result["quality_path"]})
        except Exception as e:
            self.save_completed.emit({"ok": False, "error": str(e)})


class RFTrainingWorker(QThread):
    status_updated = pyqtSignal(str, str)
    training_completed = pyqtSignal(object)
    training_failed = pyqtSignal(str)

    def __init__(self, visualizer, config, parent=None):
        super().__init__(parent)
        self._visualizer = visualizer
        self._config = dict(config or {})

    def _emit_status(self, text, color="#999999"):
        self.status_updated.emit(str(text), str(color))

    def run(self):
        try:
            cfg = dict(self._config)
            cfg["_status_callback"] = self._emit_status
            cfg["_defer_auto_load"] = True
            result = self._visualizer.train_rf_with_config(cfg)
            self.training_completed.emit(result)
        except Exception as e:
            self.training_failed.emit(f"{e}\n\n{traceback.format_exc()}")


class CalibrationDialog(QDialog):
    start_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    channel_count_applied = pyqtSignal(int)
    port_config_requested = pyqtSignal()

    def __init__(
        self,
        channel_count=4,
        rest_sec=CAL_REST_MS // 1000,
        flex_sec=CAL_FLEX_MS // 1000,
        is_connected=False,
        is_port_applied=False,
        parent=None,
    ):
        super().__init__(parent)
        apply_app_icon(self)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowTitle("Channel Calibration")
        self.setModal(True)
        self.resize(680, 560)
        apply_dark_title_bar(self)
        self.is_connected = bool(is_connected)
        self.is_port_applied = bool(is_port_applied)
        self.is_running = False
        self.is_calibrated = False

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        cal_spin_style = (
            f"QSpinBox {{ "
            f"background-color: {THEME_COLORS['panel']}; "
            f"color: {THEME_COLORS['text']}; "
            f"border: 1px solid {THEME_COLORS['accent']}; "
            "border-radius: 6px; font-weight: 700; padding: 4px 6px; min-width: 72px; }"
        )

        channel_row = QHBoxLayout()
        lbl_channel = QLabel("Channel Count:")
        lbl_channel.setStyleSheet("font-weight: bold;")
        channel_row.addWidget(lbl_channel)
        self.spin_channels = QSpinBox()
        self.spin_channels.setRange(2, 11)
        self.spin_channels.setValue(int(max(2, min(11, channel_count))))
        self.spin_channels.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
        self.spin_channels.setStyleSheet(cal_spin_style)
        self.spin_channels.setMinimumHeight(34)
        self.spin_channels.valueChanged.connect(lambda _v: self._update_current_setup_label())
        channel_row.addWidget(self.spin_channels)

        self.btn_apply_channels = QPushButton("Apply")
        self.btn_apply_channels.setStyleSheet(themed_button_style("accent"))
        self.btn_apply_channels.clicked.connect(self._on_apply_channels)
        channel_row.addWidget(self.btn_apply_channels)

        channel_row.addStretch()
        layout.addLayout(channel_row)
        layout.addSpacing(14)

        self.lbl_current_setup = QLabel("")
        self.lbl_current_setup.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.lbl_current_setup)

        self.lbl_status_msg = QLabel("")
        self.lbl_status_msg.setWordWrap(True)
        self.lbl_status_msg.setStyleSheet(themed_label_style("muted"))
        layout.addWidget(self.lbl_status_msg)
        layout.addSpacing(14)
        layout.addSpacing(8)
        layout.addSpacing(8)

        self.lbl_cal_title = QLabel("Calibrate Channels")
        self.lbl_cal_title.setStyleSheet("font-weight: bold; font-size: 19px;")
        layout.addWidget(self.lbl_cal_title)
        layout.addSpacing(8)

        self.lbl_steps = QLabel(
            "Calibration Steps:<br>"
            "1. <b>REST</b>: Keep your arm relaxed and steady, and try to remain calm.<br>"
            "2. <b>FLEX</b>: Get stretch and strong flex in the target muscle."
        )
        self.lbl_steps.setWordWrap(True)
        layout.addWidget(self.lbl_steps)
        layout.addSpacing(10)

        timing_row = QHBoxLayout()
        timing_row.addWidget(QLabel("REST (s):"))
        self.spin_rest_sec = QSpinBox()
        self.spin_rest_sec.setRange(CAL_DURATION_MIN_S, CAL_DURATION_MAX_S)
        self.spin_rest_sec.setValue(int(max(CAL_DURATION_MIN_S, min(CAL_DURATION_MAX_S, rest_sec))))
        self.spin_rest_sec.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
        self.spin_rest_sec.setStyleSheet(cal_spin_style)
        self.spin_rest_sec.setMinimumHeight(34)
        timing_row.addWidget(self.spin_rest_sec)
        timing_row.addWidget(QLabel("FLEX (s):"))
        self.spin_flex_sec = QSpinBox()
        self.spin_flex_sec.setRange(CAL_DURATION_MIN_S, CAL_DURATION_MAX_S)
        self.spin_flex_sec.setValue(int(max(CAL_DURATION_MIN_S, min(CAL_DURATION_MAX_S, flex_sec))))
        self.spin_flex_sec.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
        self.spin_flex_sec.setStyleSheet(cal_spin_style)
        self.spin_flex_sec.setMinimumHeight(34)
        timing_row.addWidget(self.spin_flex_sec)
        timing_row.addWidget(QLabel("(3-10 sec)"))
        timing_row.addStretch()
        layout.addLayout(timing_row)
        layout.addSpacing(10)

        self.lbl_press = QLabel('Press "<b>Start Calibration</b>" and follow instructions.')
        self.lbl_press.setWordWrap(True)
        layout.addWidget(self.lbl_press)
        layout.addSpacing(10)
        layout.addSpacing(8)

        self.phase_box = QWidget()
        self.phase_box.setStyleSheet(
            f"background-color: {THEME_COLORS['accent']}; border-radius: 8px; padding: 8px 12px;"
        )
        phase_box_layout = QVBoxLayout(self.phase_box)
        phase_box_layout.setContentsMargins(8, 6, 8, 6)
        phase_box_layout.setSpacing(6)

        self.lbl_phase = QLabel("Phase: Waiting")
        self.lbl_phase.setAlignment(Qt.AlignCenter)
        self.lbl_phase.setStyleSheet("font-size: 24px; font-weight: 800; color: #FFFFFF;")
        phase_box_layout.addWidget(self.lbl_phase)

        self.lbl_phase_status = QLabel("Phase status will appear here.")
        self.lbl_phase_status.setAlignment(Qt.AlignCenter)
        self.lbl_phase_status.setWordWrap(True)
        self.lbl_phase_status.setStyleSheet("color: #FFFFFF; font-weight: normal;")
        phase_box_layout.addWidget(self.lbl_phase_status)

        layout.addWidget(self.phase_box)
        layout.addSpacing(8)

        self.lbl_time = QLabel("Remaining: --.- sec")
        self.lbl_time.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.lbl_time)
        layout.addSpacing(8)

        self.lbl_status_bar = QLabel("Status Bar")
        self.lbl_status_bar.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.lbl_status_bar)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start Calibration")
        self.btn_start.setStyleSheet(themed_button_style("accent"))
        self.btn_start.clicked.connect(self.start_requested.emit)
        btn_row.addWidget(self.btn_start)

        self.btn_recalibrate = QPushButton("Recalibrate")
        self.btn_recalibrate.setStyleSheet(themed_button_style("accent"))
        self.btn_recalibrate.clicked.connect(self.start_requested.emit)
        self.btn_recalibrate.setEnabled(False)
        btn_row.addWidget(self.btn_recalibrate)

        self.btn_done = QPushButton("Done")
        self.btn_done.clicked.connect(self._on_done)
        btn_row.addWidget(self.btn_done)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.lock_overlay = QWidget(self)
        self.lock_overlay.setStyleSheet(
            f"background-color: rgba(10, 24, 44, 225); border: 1px solid {THEME_COLORS['accent']}; border-radius: 8px;"
        )
        overlay_layout = QVBoxLayout(self.lock_overlay)
        overlay_layout.setContentsMargins(20, 20, 20, 20)
        overlay_layout.setSpacing(12)
        overlay_layout.addStretch()

        self.lbl_overlay = QLabel("")
        self.lbl_overlay.setAlignment(Qt.AlignCenter)
        self.lbl_overlay.setWordWrap(True)
        self.lbl_overlay.setStyleSheet("font-weight: bold;")
        overlay_layout.addWidget(self.lbl_overlay)

        self.btn_open_port = QPushButton("Open Port Configuration")
        self.btn_open_port.setStyleSheet(themed_button_style("accent"))
        self.btn_open_port.clicked.connect(self.port_config_requested.emit)
        overlay_layout.addWidget(self.btn_open_port, alignment=Qt.AlignCenter)

        overlay_layout.addStretch()
        self.lock_overlay.hide()

        self._update_current_setup_label()
        self.set_port_state(self.is_connected, self.is_port_applied)
        self._position_overlay()

    def _on_done(self):
        self.reject()

    def _on_apply_channels(self):
        self.channel_count_applied.emit(int(self.spin_channels.value()))

    def set_channel_count(self, count):
        self.spin_channels.setValue(int(max(2, min(11, count))))
        self._update_current_setup_label()

    def _update_current_setup_label(self):
        self.lbl_current_setup.setText(f"Current channel setup: {int(self.spin_channels.value())}")

    def set_status_message(self, text, kind="muted"):
        self.lbl_status_msg.setText(str(text))
        self.lbl_status_msg.setStyleSheet(themed_label_style(kind))

    def set_port_state(self, is_connected, is_port_applied):
        self.is_connected = bool(is_connected)
        self.is_port_applied = bool(is_port_applied)
        unlocked = self.is_connected and self.is_port_applied

        self._update_current_setup_label()
        if unlocked:
            self.set_status_message("Port configured and connected. Channel setup and calibration are enabled.", "success")
        elif self.is_connected:
            self.set_status_message(
                "Port connected but not applied. Click Apply in Port Configuration to unlock.",
                "muted",
            )
        else:
            self.set_status_message(
                "Port is not connected. Configure and connect from Port Configuration first.",
                "muted",
            )

        can_edit_channels = unlocked and (not self.is_running)
        self.spin_channels.setEnabled(can_edit_channels)
        self.btn_apply_channels.setEnabled(can_edit_channels)
        self.spin_rest_sec.setEnabled(can_edit_channels)
        self.spin_flex_sec.setEnabled(can_edit_channels)
        can_calibrate = unlocked and (not self.is_running)
        self.btn_start.setEnabled(can_calibrate and (not self.is_calibrated))
        self.btn_recalibrate.setEnabled(can_calibrate and self.is_calibrated)

        self.lock_overlay.setVisible(not unlocked)
        if not unlocked:
            if self.is_connected and not self.is_port_applied:
                self.lbl_overlay.setText(
                    "Go to Port Configuration and click Apply after connecting.\n"
                    "Channel Calibration stays locked until that step is completed."
                )
            else:
                self.lbl_overlay.setText(
                    "Configure port and connect from Port Configuration first.\n"
                    "Then click Apply to unlock Channel Calibration."
                )
            self.lock_overlay.raise_()
        self._position_overlay()

    def _position_overlay(self):
        margin = 12
        self.lock_overlay.setGeometry(
            margin,
            margin,
            max(0, self.width() - (2 * margin)),
            max(0, self.height() - (2 * margin)),
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_overlay()

    def set_phase(self, name, instruction, remaining_ms, total_ms):
        remaining_s = max(0.0, remaining_ms / 1000.0)
        self.lbl_phase.setText(f"Phase: {name}")
        self.lbl_phase_status.setText(str(instruction))
        self.lbl_time.setText(f"Remaining: {remaining_s:.1f} s")

        elapsed = max(0, total_ms - max(0, remaining_ms))
        pct = int(100.0 * elapsed / max(1, total_ms))
        self.progress.setValue(max(0, min(100, pct)))

    def set_running_state(self):
        self.is_running = True
        self.spin_channels.setEnabled(False)
        self.btn_apply_channels.setEnabled(False)
        self.spin_rest_sec.setEnabled(False)
        self.spin_flex_sec.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_recalibrate.setEnabled(False)
        self.btn_start.setText("Calibrating...")

    def set_finished(self, summary_text):
        self.is_running = False
        self.is_calibrated = True
        self.lbl_phase.setText("Phase: Complete")
        self.lbl_phase_status.setText("Calibration complete.")
        self.set_status_message(summary_text, "success")
        self.lbl_time.setText("Remaining: 0.0 s")
        self.progress.setValue(100)
        self.btn_start.setEnabled(False)
        self.btn_start.setText("Completed")
        self.btn_recalibrate.setEnabled(self.is_connected and self.is_port_applied)

    def set_calibrated_state(self, is_calibrated):
        self.is_calibrated = bool(is_calibrated)
        can_calibrate = self.is_connected and self.is_port_applied and (not self.is_running)
        self.btn_start.setEnabled(can_calibrate and (not self.is_calibrated))
        self.btn_recalibrate.setEnabled(can_calibrate and self.is_calibrated)

    def calibration_durations_ms(self):
        rest_s = int(max(CAL_DURATION_MIN_S, min(CAL_DURATION_MAX_S, self.spin_rest_sec.value())))
        flex_s = int(max(CAL_DURATION_MIN_S, min(CAL_DURATION_MAX_S, self.spin_flex_sec.value())))
        return rest_s * 1000, flex_s * 1000

    def set_calibration_seconds(self, rest_s, flex_s):
        self.spin_rest_sec.setValue(int(max(CAL_DURATION_MIN_S, min(CAL_DURATION_MAX_S, rest_s))))
        self.spin_flex_sec.setValue(int(max(CAL_DURATION_MIN_S, min(CAL_DURATION_MAX_S, flex_s))))


class AnimatedStatePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._state_colors = {
            "rest": "#3A6EA5",      # blue - relax (kept distinct from the green Perform)
            "prepare": "#C62828",   # red - get ready, don't move yet
            "perform": "#2E7D32",   # green - perform the gesture now
            "break": "#6D4C9F",     # purple - block / hand-change break
            "activity": THEME_COLORS["accent"],  # idle / ready / complete
        }
        self._state = ""
        self.set_state("activity")

    def set_state(self, state_key):
        key = str(state_key or "activity").strip().lower()
        if key not in self._state_colors:
            key = "activity"
        if key == self._state:
            return
        self._state = key
        fill = self._state_colors[key]
        self.setStyleSheet(
            f"background-color: {fill}; border-radius: 8px; padding: 8px 12px;"
        )


def format_duration(ms, with_tenths=True):
    """'12.3 s' below a minute (when with_tenths), otherwise 'm:ss'."""
    ms = max(0, int(ms))
    if with_tenths and ms < 60000:
        return f"{ms / 1000.0:.1f} s"
    total_s = int(round(ms / 1000.0))
    return f"{total_s // 60}:{total_s % 60:02d}"


HAND_SEQUENCES = {
    "left_right": ["Left", "Right"],
    "right_left": ["Right", "Left"],
    "left": ["Left"],
    "right": ["Right"],
}


def hands_for_order(hand_order):
    return list(HAND_SEQUENCES.get(str(hand_order), HAND_SEQUENCES[DEFAULT_HAND_ORDER]))


def plan_summary(settings):
    """Numbers for a session plan, shared by the dialog preview and the recorder."""
    labels = [x for x in (settings.get("task_labels") or []) if str(x).strip()]
    hands = hands_for_order(settings.get("hand_order", DEFAULT_HAND_ORDER))
    repeats = int(settings.get("repeats", DEFAULT_TASK_REPEATS))
    prep = float(settings.get("prep_s", DEFAULT_TASK_PREP_S))
    hold = float(settings.get("hold_s", DEFAULT_TASK_HOLD_S))
    rest = float(settings.get("rest_s", DEFAULT_TASK_REST_S))
    settle = float(settings.get("settle_s", DEFAULT_TASK_SETTLE_S))
    max_block = max(1, int(settings.get("max_cycles_per_block", DEFAULT_MAX_CYCLES_PER_BLOCK)))
    block_break = float(settings.get("block_break_s", DEFAULT_BLOCK_BREAK_S))
    switch_break = float(settings.get("hand_switch_s", DEFAULT_HAND_SWITCH_BREAK_S))
    cycle_s = prep + hold + settle + rest
    cycles_per_hand = len(labels) * repeats
    blocks_per_hand = -(-cycles_per_hand // max_block) if cycles_per_hand else 0
    cycles = cycles_per_hand * len(hands)
    block_breaks = max(0, blocks_per_hand - 1) * len(hands)
    switch_breaks = max(0, len(hands) - 1)
    active_s = cycles * cycle_s
    total_s = active_s + block_breaks * block_break + switch_breaks * switch_break
    return {
        "hands": hands, "cycles": cycles, "cycles_per_hand": cycles_per_hand,
        "blocks_per_hand": blocks_per_hand, "cycle_s": cycle_s,
        "active_s": active_s, "total_s": total_s,
        "recorded_s": cycles * (hold + (rest if settings.get("record_rest", True) else 0.0)),
    }


class TaskProtocolWidget(QWidget):
    start_clicked = pyqtSignal()
    # label, trial_id, phase_name, record_enabled, hand
    phase_started = pyqtSignal(str, int, str, bool, str, int, int, int)   # label, trial, phase, record, hand, block, cycle, planned ms
    protocol_finished = pyqtSignal()
    protocol_canceled = pyqtSignal()
    paused_changed = pyqtSignal(bool)
    rewind_requested = pyqtSignal()   # resume restarts the interrupted cycle; drop its partial data

    def __init__(self, parent=None):
        super().__init__(parent)
        self.paused = False
        self._pause_remaining_ms = 0
        self._pause_saved = None
        self.labels = []
        self.repeats = 1
        self.prep_ms = int(DEFAULT_TASK_PREP_S * 1000)
        self.hold_ms = int(DEFAULT_TASK_HOLD_S * 1000)
        self.rest_ms = int(DEFAULT_TASK_REST_S * 1000)
        self.settle_ms = int(DEFAULT_TASK_SETTLE_S * 1000)
        self.record_rest = True
        self.randomize_order = True
        self.seed = 0
        self.hand_order = DEFAULT_HAND_ORDER
        self.session_day = DEFAULT_SESSION_DAY
        self.max_cycles_per_block = DEFAULT_MAX_CYCLES_PER_BLOCK
        self.block_break_ms = int(DEFAULT_BLOCK_BREAK_S * 1000)
        self.hand_switch_ms = int(DEFAULT_HAND_SWITCH_BREAK_S * 1000)

        self.steps = []
        self.step_idx = -1
        self.remaining_ms = 0
        self.current_step_total_ms = 1
        self.step_deadline = 0.0
        self.session_total_ms = 1
        self.session_done_ms = 0
        self.is_running = False
        self.is_ready = False

        # Wall-clock driven (not "subtract 100 ms per tick") so a busy UI cannot stretch a
        # 3 s hold into 3.4 s and silently shorten the recording.
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.on_tick)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.lbl_title = QLabel("Session Activity")
        self.lbl_title.setStyleSheet("font-weight: bold; font-size: 17px;")
        layout.addWidget(self.lbl_title)

        self.phase_box = AnimatedStatePanel()
        self.phase_box.setMinimumHeight(GESTURE_IMAGE_MAX_H + 36)
        self.phase_box.set_state("activity")
        phase_box_layout = QHBoxLayout(self.phase_box)
        phase_box_layout.setContentsMargins(18, 18, 18, 18)
        phase_box_layout.setSpacing(20)

        # Gesture reference image. The slot has a fixed size so the text never
        # shifts between gestures whose photos have slightly different shapes;
        # the image is centred inside it. Hidden when there is no image.
        self.lbl_gesture_image = QLabel()
        self.lbl_gesture_image.setFixedSize(GESTURE_IMAGE_MAX_W, GESTURE_IMAGE_MAX_H)
        self.lbl_gesture_image.setAlignment(Qt.AlignCenter)
        self.lbl_gesture_image.setStyleSheet("background: transparent; border: none;")
        self.lbl_gesture_image.hide()
        self._gesture_image_label = None
        phase_box_layout.addWidget(self.lbl_gesture_image, 0, Qt.AlignVCenter)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(8)
        text_col.addStretch(1)
        phase_box_layout.addLayout(text_col, 1)

        self.lbl_phase = QLabel("Waiting")
        self.lbl_phase.setAlignment(Qt.AlignCenter)
        self.lbl_phase.setWordWrap(True)
        self.lbl_phase.setStyleSheet("font-size: 30px; font-weight: 800; color: #FFFFFF; background: transparent;")
        text_col.addWidget(self.lbl_phase)

        self.lbl_instruction = QLabel("Press Proceed to prepare, then Start Recording.")
        self.lbl_instruction.setAlignment(Qt.AlignCenter)
        self.lbl_instruction.setWordWrap(True)
        self.lbl_instruction.setStyleSheet(
            "color: #FFFFFF; font-size: 15px; font-weight: normal; background: transparent;"
        )
        text_col.addWidget(self.lbl_instruction)
        text_col.addStretch(1)
        layout.addWidget(self.phase_box)

        self.lbl_meta_info = StableLabel("Day - | Hand - | Block -/- | Cycle -/- | Label: -")
        self.lbl_meta_info.setWordWrap(True)
        self.lbl_meta_info.setStyleSheet(themed_label_style("muted"))
        layout.addWidget(self.lbl_meta_info)

        self.lbl_count = StableLabel("Phase: --.- s | Session: --:-- left")
        self.lbl_count.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.lbl_count)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(10)
        layout.addWidget(self.progress)

        self.progress_session = QProgressBar()
        self.progress_session.setRange(0, 1000)
        self.progress_session.setValue(0)
        self.progress_session.setFormat("Session %p%")
        self.progress_session.setTextVisible(False)
        self.progress_session.setFixedHeight(6)
        layout.addWidget(self.progress_session)

        # Live CSV counter: rows captured so far versus what the protocol should produce.
        self.lbl_csv_counter = StableLabel("CSV rows recorded: 0")
        self.lbl_csv_counter.setStyleSheet(f"font-weight: bold; color: {THEME_COLORS['accent']};")
        layout.addWidget(self.lbl_csv_counter)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start Recording")
        self.btn_start.setStyleSheet(themed_button_style("accent"))
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_clicked.emit)
        btn_row.addWidget(self.btn_start)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setStyleSheet(themed_button_style("danger"))
        self.btn_pause.setEnabled(False)
        self.btn_pause.setToolTip("Emergency / interruption: stops the timer and recording. "
                                  "Resume restarts the current cycle from Prepare.")
        self.btn_pause.clicked.connect(self.toggle_pause)
        btn_row.addWidget(self.btn_pause)

        self.btn_skip_break = QPushButton("Skip Break")
        self.btn_skip_break.setStyleSheet(themed_button_style("muted"))
        self.btn_skip_break.setEnabled(False)
        self.btn_skip_break.setToolTip("End the current break early (participant is ready).")
        self.btn_skip_break.clicked.connect(self.skip_break)
        btn_row.addWidget(self.btn_skip_break)

        self.btn_cancel = QPushButton("Cancel Session")
        self.btn_cancel.setStyleSheet(themed_button_style("muted"))
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(lambda: self.cancel_protocol(emit_signal=True))
        btn_row.addWidget(self.btn_cancel)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    # -- configuration -----------------------------------------------------------
    def configure(self, labels, repeats, prep_s, hold_s, rest_s, record_rest=True,
                  settle_s=DEFAULT_TASK_SETTLE_S, randomize_order=True, hand_order=DEFAULT_HAND_ORDER,
                  session_day=DEFAULT_SESSION_DAY, max_cycles_per_block=DEFAULT_MAX_CYCLES_PER_BLOCK,
                  block_break_s=DEFAULT_BLOCK_BREAK_S, hand_switch_s=DEFAULT_HAND_SWITCH_BREAK_S, seed=0):
        self.seed = int(seed)
        self.labels = [str(x).strip() for x in (labels or []) if str(x).strip()]
        self.repeats = max(1, int(repeats))
        self.prep_ms = int(max(0.2, prep_s) * 1000)
        self.hold_ms = int(max(0.2, hold_s) * 1000)
        self.rest_ms = int(max(0.2, rest_s) * 1000)
        self.settle_ms = int(max(0.0, settle_s) * 1000)
        self.record_rest = bool(record_rest)
        self.randomize_order = bool(randomize_order)
        self.hand_order = str(hand_order)
        self.session_day = int(session_day)
        self.max_cycles_per_block = max(1, int(max_cycles_per_block))
        self.block_break_ms = int(max(0.0, block_break_s) * 1000)
        self.hand_switch_ms = int(max(0.0, hand_switch_s) * 1000)
        self.steps = self.build_steps()
        self.session_total_ms = max(1, sum(int(st["duration_ms"]) for st in self.steps))
        self.session_done_ms = 0
        self._preload_gesture_images()
        self.step_idx = -1
        self.remaining_ms = 0
        self.current_step_total_ms = 1
        self.is_ready = True
        self.progress.setValue(0)
        self.progress_session.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.btn_skip_break.setEnabled(False)
        self.lbl_phase.setText("Ready")
        self.lbl_instruction.setText("Session configured. Press Start Recording to begin.")
        self._set_gesture_image(None)
        self.phase_box.set_state("activity")
        self.lbl_meta_info.setText(
            f"Day {self.session_day} | {' + '.join(hands_for_order(self.hand_order))} hand | "
            f"{self.cycles_total} cycles | about {format_duration(self.session_total_ms, False)} incl. breaks"
        )
        self.lbl_count.setText(f"Phase: --.- s | Session: {format_duration(self.session_total_ms, False)} left")

    @property
    def cycles_total(self):
        return sum(1 for st in self.steps if st["phase"] == "Prepare")

    def set_start_enabled(self, enabled):
        self.btn_start.setEnabled(bool(enabled) and self.is_ready and (not self.is_running))

    def set_cancel_enabled(self, enabled):
        if not self.is_running:
            self.btn_cancel.setEnabled(bool(enabled) and self.is_ready)

    def set_csv_counter(self, text, kind="accent"):
        color = {"accent": THEME_COLORS["accent"], "warn": "#f57c00", "danger": "#f44336"}.get(kind, THEME_COLORS["accent"])
        self.lbl_csv_counter.setText(text)
        self.lbl_csv_counter.setStyleSheet(f"font-weight: bold; color: {color};")

    def build_steps(self):
        """Flatten the study design into timed steps.

        Per hand: `repeats` repetitions of every class (order shuffled per repetition so
        fatigue is spread across classes), split into blocks of at most
        `max_cycles_per_block` cycles with a break in between; a longer break separates
        the hands (the armband moves to the other forearm). Trial ids keep counting up
        across hands so (label, trial) stays unique in the CSV.
        """
        steps = []
        rng = random.Random(self.seed)   # per-session seed, saved in the metadata: the order is reproducible
        hands = hands_for_order(self.hand_order)
        cycles_total = len(self.labels) * self.repeats * len(hands)
        cycle_no = 0
        trial_offset = 0
        for hand_idx, hand in enumerate(hands):
            cycles = []
            for rep in range(1, self.repeats + 1):
                trial_labels = list(self.labels)
                if self.randomize_order:
                    rng.shuffle(trial_labels)
                cycles.extend((trial_offset + rep, label) for label in trial_labels)
            blocks = [cycles[i:i + self.max_cycles_per_block] for i in range(0, len(cycles), self.max_cycles_per_block)]
            for block_idx, block in enumerate(blocks):
                for trial, label in block:
                    cycle_no += 1
                    common = {
                        "trial_id": trial, "hand": hand, "block": block_idx + 1, "blocks": len(blocks),
                        "cycle_no": cycle_no, "cycles_total": cycles_total,
                    }
                    steps.append({**common, "phase": "Prepare", "label": label, "duration_ms": self.prep_ms,
                                  "record": False, "instruction": "Prepare and keep still until countdown ends."})
                    steps.append({**common, "phase": "Perform", "label": label, "duration_ms": self.hold_ms,
                                  "record": True, "instruction": "Perform and hold the target activity."})
                    if self.settle_ms > 0:
                        # Not recorded: EMG doesn't drop to baseline the instant a
                        # contraction ends, so without this gap the start of every
                        # "Rest" window is really relaxation decay, not true rest.
                        steps.append({**common, "phase": "Settle", "label": "Rest", "duration_ms": self.settle_ms,
                                      "record": False,
                                      "instruction": "Relax. (Not recorded - settling before Rest capture.)"})
                    steps.append({**common, "phase": "Rest", "label": "Rest", "duration_ms": self.rest_ms,
                                  "record": self.record_rest, "instruction": "Rest arm fully. Relax muscles."})
                is_last_block = block_idx == len(blocks) - 1
                is_last_hand = hand_idx == len(hands) - 1
                common = {"trial_id": cycles[-1][0] if is_last_block else block[-1][0], "hand": hand,
                          "block": block_idx + 1, "blocks": len(blocks), "cycle_no": cycle_no,
                          "cycles_total": cycles_total, "label": "Rest", "record": False}
                if not is_last_block and self.block_break_ms > 0:
                    steps.append({**common, "phase": "Break", "duration_ms": self.block_break_ms,
                                  "instruction": f"Block {block_idx + 1}/{len(blocks)} done. Relax and shake out your "
                                                 "arm. The next block starts when the countdown ends."})
                elif is_last_block and not is_last_hand and self.hand_switch_ms > 0:
                    next_hand = hands[hand_idx + 1]
                    steps.append({**common, "phase": "Switch Hand", "duration_ms": self.hand_switch_ms,
                                  "instruction": f"{hand} hand finished. Move the armband to the {next_hand.lower()} "
                                                 "forearm (re-run Channel Calibration if needed)."})
            trial_offset += self.repeats
        return steps

    def start_protocol(self):
        if self.is_running:
            return False
        if (not self.is_ready) or len(self.steps) == 0:
            return False
        self.is_running = True
        self.paused = False
        self.btn_pause.setText("Pause")
        self.btn_pause.setEnabled(True)
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.step_idx = -1
        self.session_done_ms = 0
        self.next_step()
        return True

    @staticmethod
    def _state_for_phase(phase_name):
        p = str(phase_name or "").strip().lower()
        if p in ("rest", "settle"):
            return "rest"
        if p == "prepare":
            return "prepare"
        if p == "perform":
            return "perform"
        if p in ("break", "switch hand"):
            return "break"
        return "activity"

    def next_step(self, advance_done=True):
        if advance_done:
            if 0 <= self.step_idx < len(self.steps):
                self.session_done_ms += int(self.steps[self.step_idx]["duration_ms"])
            self.step_idx += 1
        # else: re-enter the current step (resume after a pause)
        if self.step_idx >= len(self.steps):
            self.timer.stop()
            self.is_running = False
            self.is_ready = False
            self.btn_cancel.setEnabled(False)
            self.btn_skip_break.setEnabled(False)
            self.btn_pause.setEnabled(False)
            self.progress.setValue(100)
            self.progress_session.setValue(1000)
            self.lbl_phase.setText("Complete")
            self.lbl_instruction.setText("Protocol complete. Saving the recording...")
            self._set_gesture_image(None)
            self.phase_box.set_state("activity")
            self.lbl_meta_info.setText("Session complete.")
            self.lbl_count.setText("Phase: 0.0 s | Session: 0:00 left")
            self.protocol_finished.emit()
            return

        st = self.steps[self.step_idx]
        self.remaining_ms = int(st["duration_ms"])
        self.current_step_total_ms = max(1, int(st["duration_ms"]))
        self.step_deadline = time.monotonic() + self.remaining_ms / 1000.0
        phase_name = str(st["phase"]).strip().lower()
        pretty_label = str(st["label"]).replace("_", " ")
        is_break = phase_name in ("break", "switch hand")
        if phase_name == "perform":
            phase_display = f"Perform: {pretty_label}"
        elif phase_name == "prepare":
            phase_display = f"Prepare for {pretty_label}"
        elif phase_name == "rest":
            phase_display = "Rest"
        elif phase_name == "settle":
            phase_display = "Settling..."
        elif phase_name == "break":
            phase_display = "Break"
        elif phase_name == "switch hand":
            phase_display = "Switch Hand"
        else:
            phase_display = str(st["phase"])
        self.lbl_phase.setText(phase_display)
        self.lbl_instruction.setText(st["instruction"])
        self.phase_box.set_state(self._state_for_phase(st["phase"]))
        self._set_gesture_image(None if is_break else st["label"])
        self.btn_skip_break.setEnabled(is_break)
        self.lbl_meta_info.setText(
            f"Day {self.session_day} | {st['hand']} hand | Block {st['block']}/{st['blocks']} | "
            f"Cycle {st['cycle_no']}/{st['cycles_total']} | Trial {st['trial_id']}"
            + ("" if is_break else f" | Label: {st['label']}")
        )
        self._update_countdown_labels(self.remaining_ms)
        self.progress.setValue(0)

        self._emit_phase_started(st, int(st["duration_ms"]))
        self.timer.start(50 if not is_break else 200)

    def _emit_phase_started(self, st, planned_ms):
        self.phase_started.emit(
            str(st["label"]),
            int(st["trial_id"]),
            str(st["phase"]),
            bool(st["record"]),
            str(st["hand"]),
            int(st["block"]),
            int(st["cycle_no"]),
            int(planned_ms),
        )

    def toggle_pause(self):
        if self.paused:
            self.resume_protocol()
        else:
            self.pause_protocol()

    def pause_protocol(self):
        if not self.is_running or self.paused:
            return
        self.paused = True
        self._pause_remaining_ms = max(0, int(round((self.step_deadline - time.monotonic()) * 1000.0)))
        self.timer.stop()
        self._pause_saved = (self.lbl_phase.text(), self.lbl_instruction.text(), self.phase_box._state)
        self.lbl_phase.setText("Paused")
        self.lbl_instruction.setText("Session paused - recording is stopped. Press Resume when ready.")
        self.phase_box.set_state("break")
        self.btn_pause.setText("Resume")
        self.btn_skip_break.setEnabled(False)
        self.paused_changed.emit(True)

    def resume_protocol(self):
        if not self.paused:
            return
        self.paused = False
        self.btn_pause.setText("Pause")
        st = self.steps[self.step_idx]
        if str(st["phase"]).strip().lower() in ("break", "switch hand"):
            self.step_deadline = time.monotonic() + self._pause_remaining_ms / 1000.0
            if self._pause_saved:
                self.lbl_phase.setText(self._pause_saved[0])
                self.lbl_instruction.setText(self._pause_saved[1])
                self.phase_box.set_state(self._pause_saved[2])
            self.btn_skip_break.setEnabled(True)
            self.paused_changed.emit(False)
            self._emit_phase_started(st, self._pause_remaining_ms)   # the break label resumes at this sample
            self.timer.start(200)
            return
        # Mid-cycle: restart the cycle from its Prepare step so every recorded trial is complete.
        j = self.step_idx
        while j > 0 and self.steps[j]["phase"] != "Prepare":
            j -= 1
        self.session_done_ms = sum(int(x["duration_ms"]) for x in self.steps[:j])
        self.step_idx = j
        self.paused_changed.emit(False)
        self.rewind_requested.emit()
        self.next_step(advance_done=False)

    def skip_break(self):
        if not self.is_running or not (0 <= self.step_idx < len(self.steps)):
            return
        if str(self.steps[self.step_idx]["phase"]).strip().lower() in ("break", "switch hand"):
            self.timer.stop()
            self.next_step()

    def _preload_gesture_images(self):
        dpr = self.devicePixelRatioF()
        for label in set(self.labels) | {"Rest"}:
            get_gesture_pixmap(label, dpr=dpr)

    def _set_gesture_image(self, label):
        if label == self._gesture_image_label:
            return  # same gesture as the previous step (e.g. Settle -> Rest): nothing to redraw
        self._gesture_image_label = label
        pix = get_gesture_pixmap(label, dpr=self.devicePixelRatioF()) if label else None
        if pix is None:
            self.lbl_gesture_image.clear()
            self.lbl_gesture_image.hide()
            return
        self.lbl_gesture_image.setPixmap(pix)
        self.lbl_gesture_image.show()

    def _update_countdown_labels(self, remaining_ms):
        elapsed_in_step = self.current_step_total_ms - remaining_ms
        session_left = max(0, self.session_total_ms - self.session_done_ms - elapsed_in_step)
        self.lbl_count.setText(
            f"Phase: {format_duration(remaining_ms)} left | Session: {format_duration(session_left, False)} left"
        )
        self.progress_session.setValue(int(1000 * min(1.0, (self.session_done_ms + elapsed_in_step) / self.session_total_ms)))

    def on_tick(self):
        self.remaining_ms = max(0, int(round((self.step_deadline - time.monotonic()) * 1000.0)))
        self._update_countdown_labels(self.remaining_ms)
        elapsed = self.current_step_total_ms - self.remaining_ms
        self.progress.setValue(int(np.clip((elapsed / self.current_step_total_ms) * 100.0, 0.0, 100.0)))

        if self.remaining_ms <= 0:
            self.timer.stop()
            self.next_step()

    def cancel_protocol(self, emit_signal=False):
        was_running = self.is_running or self.timer.isActive() or self.paused
        was_prepared = self.is_ready
        self.timer.stop()
        self.paused = False
        self.btn_pause.setText("Pause")
        self.btn_pause.setEnabled(False)
        self.is_running = False
        self.is_ready = False
        self.btn_cancel.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_skip_break.setEnabled(False)
        self.progress.setValue(0)
        self.progress_session.setValue(0)
        self.lbl_phase.setText("Canceled" if was_running else "Waiting")
        self._set_gesture_image(None)
        self.lbl_instruction.setText(
            "Protocol canceled. Press Proceed to prepare again." if was_running else "Press Proceed to prepare."
        )
        self.phase_box.set_state("activity")
        self.lbl_meta_info.setText("Day - | Hand - | Block -/- | Cycle -/- | Label: -")
        self.lbl_count.setText("Phase: --.- s | Session: --:-- left")
        if emit_signal and (was_running or was_prepared):
            self.protocol_canceled.emit()



class PortConfigDialog(QDialog):
    connect_requested = pyqtSignal(str)
    refresh_requested = pyqtSignal()

    def __init__(self, selected_port="", is_connected=False, parent=None):
        super().__init__(parent)
        apply_app_icon(self)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowTitle("Port Configuration - Wired")
        self.setModal(True)
        self.resize(640, 320)
        apply_dark_title_bar(self)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        lbl_title = QLabel("Instructions:")
        lbl_title.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl_title)
        layout.addSpacing(8)

        self.lbl_hint = QLabel(
            "Refresh for port selection\n"
            "Live data: COM Port\n"
            "Demo data: Socket"
        )
        self.lbl_hint.setStyleSheet(themed_label_style("muted"))
        self.lbl_hint.setWordWrap(True)
        layout.addWidget(self.lbl_hint)
        layout.addSpacing(12)

        row = QHBoxLayout()
        row.addWidget(QLabel("Port Selection:"))
        self.port_selector = QComboBox()
        self.port_selector.setEditable(True)
        row.addWidget(self.port_selector, stretch=1)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setStyleSheet(themed_button_style("accent"))
        self.btn_refresh.clicked.connect(self.refresh_requested.emit)
        row.addWidget(self.btn_refresh)
        layout.addLayout(row)
        layout.addSpacing(12)

        btn_row = QHBoxLayout()
        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setStyleSheet(themed_button_style("success"))
        self.btn_connect.clicked.connect(self.on_connect_clicked)
        btn_row.addWidget(self.btn_connect)

        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setStyleSheet(themed_button_style("accent"))
        self.btn_apply.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_apply)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        layout.addStretch()

        self.set_ports([], selected_port)
        self.update_connection_state(is_connected)

    def set_ports(self, items, selected_port=""):
        self.port_selector.clear()
        for label, value in items:
            self.port_selector.addItem(label, value)

        selected = (selected_port or "").strip()
        if selected:
            idx = self.port_selector.findData(selected)
            if idx >= 0:
                self.port_selector.setCurrentIndex(idx)
            else:
                self.port_selector.setCurrentText(selected)

    def selected_port(self):
        data = self.port_selector.currentData()
        if data:
            return str(data).strip()
        return self.port_selector.currentText().strip()

    def update_connection_state(self, is_connected):
        self.btn_connect.setEnabled(not is_connected)
        if is_connected:
            self.btn_connect.setStyleSheet(themed_button_style("muted"))
        else:
            self.btn_connect.setStyleSheet(themed_button_style("success"))

    def on_connect_clicked(self):
        port = self.selected_port()
        if not port:
            QMessageBox.warning(self, "No Port", "Please select a valid serial port.")
            return
        self.connect_requested.emit(port)


class TrainResultsDialog(QDialog):
    def __init__(self, summary_text, report_text, cm, class_names, parent=None):
        super().__init__(parent)
        apply_app_icon(self)
        self.setWindowTitle("RF Training Results")
        self.resize(980, 720)

        layout = QVBoxLayout(self)
        lbl = QLabel(summary_text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl)

        report = QTextEdit()
        report.setReadOnly(True)
        report.setStyleSheet("QTextEdit { font-family: Consolas; font-size: 15px; }")
        report.setPlainText(report_text)
        layout.addWidget(report)

        cm_title = QLabel("Confusion Matrix")
        cm_title.setStyleSheet("font-weight: bold;")
        layout.addWidget(cm_title)

        table = QTableWidget(cm.shape[0], cm.shape[1])
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setHorizontalHeaderLabels(class_names)
        table.setVerticalHeaderLabels(class_names)
        table.setStyleSheet("QTableWidget { font-family: Consolas; font-size: 15px; }")

        max_val = max(1, int(np.max(cm)))
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                v = int(cm[r, c])
                it = QTableWidgetItem(str(v))
                it.setTextAlignment(Qt.AlignCenter)
                shade = int(255 - (180 * (v / max_val)))
                it.setBackground(QColor(shade, 255, shade))
                table.setItem(r, c, it)
        layout.addWidget(table)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)


class RFTrainingDialog(QDialog):
    def __init__(self, visualizer, parent=None):
        super().__init__(parent)
        apply_app_icon(self)
        self.visualizer = visualizer
        self.current_model_path = ""
        self.current_run_dir = ""
        self._is_destroying = False
        self.training_worker = None
        self.dataset_channel_counts = {}
        self.dataset_channel_errors = {}
        self._dataset_signature = tuple()

        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowTitle("RF Training")
        self.setModal(False)
        self.resize(1080, 760)
        apply_dark_title_bar(self)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        dataset_tab = QWidget()
        dataset_layout = QVBoxLayout(dataset_tab)
        dataset_layout.setSpacing(8)
        lbl_data = QLabel("Select one or more dataset CSV files")
        lbl_data.setStyleSheet("font-weight: bold;")
        dataset_layout.addWidget(lbl_data)

        self.dataset_list = QListWidget()
        self.dataset_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.dataset_list.setMinimumHeight(260)
        dataset_layout.addWidget(self.dataset_list, 1)
        self.dataset_model = self.dataset_list.model()
        self.dataset_model.rowsInserted.connect(self._on_dataset_model_changed)
        self.dataset_model.rowsRemoved.connect(self._on_dataset_model_changed)
        self.dataset_model.modelReset.connect(self._on_dataset_model_changed)
        self.lbl_dataset_channel_info = QLabel("Detected channels per file: n/a")
        self.lbl_dataset_channel_info.setWordWrap(True)
        self.lbl_dataset_channel_info.setStyleSheet(themed_label_style("muted"))
        dataset_layout.addWidget(self.lbl_dataset_channel_info)

        row_data_actions = QHBoxLayout()
        self.btn_add_files = QPushButton("Add CSV Files")
        self.btn_add_files.setStyleSheet(themed_button_style("accent"))
        self.btn_add_files.clicked.connect(self.on_add_dataset_files)
        row_data_actions.addWidget(self.btn_add_files)

        self.btn_add_folder = QPushButton("Add Folder")
        self.btn_add_folder.setStyleSheet(themed_button_style("accent"))
        self.btn_add_folder.clicked.connect(self.on_add_dataset_folder)
        row_data_actions.addWidget(self.btn_add_folder)

        self.btn_remove_selected = QPushButton("Remove Selected")
        self.btn_remove_selected.setStyleSheet(themed_button_style("muted"))
        self.btn_remove_selected.clicked.connect(self.on_remove_selected_datasets)
        row_data_actions.addWidget(self.btn_remove_selected)

        self.btn_clear_datasets = QPushButton("Clear")
        self.btn_clear_datasets.setStyleSheet(themed_button_style("muted"))
        self.btn_clear_datasets.clicked.connect(self.on_clear_datasets)
        row_data_actions.addWidget(self.btn_clear_datasets)
        row_data_actions.addStretch()
        dataset_layout.addLayout(row_data_actions)
        self.tabs.addTab(dataset_tab, "1) Datasets")

        self.train_model_tab = QWidget()
        setup_layout = QVBoxLayout(self.train_model_tab)
        setup_layout.setSpacing(8)
        lbl_train = QLabel("Configure model, split parameters, and output folder")
        lbl_train.setStyleSheet("font-weight: bold;")
        setup_layout.addWidget(lbl_train)

        form = QFormLayout()
        form.setSpacing(8)

        self.spin_window_ms = QSpinBox()
        self.spin_window_ms.setRange(20, 4000)
        self.spin_window_ms.setValue(int(max(20, self.visualizer.analysis_ms_spin.value())))
        form.addRow("Window (ms):", self.spin_window_ms)

        self.spin_stride_ms = QSpinBox()
        self.spin_stride_ms.setRange(5, 2000)
        self.spin_stride_ms.setValue(50)
        form.addRow("Stride (ms):", self.spin_stride_ms)

        self.spin_estimators = QSpinBox()
        self.spin_estimators.setRange(50, 5000)
        self.spin_estimators.setValue(RF_DEFAULT_N_ESTIMATORS)
        self.spin_estimators.setToolTip("Fewer trees = faster live prediction. 100-150 is usually enough.")
        form.addRow("RF Trees:", self.spin_estimators)

        self.spin_max_depth = QSpinBox()
        self.spin_max_depth.setRange(0, 200)
        self.spin_max_depth.setValue(RF_DEFAULT_MAX_DEPTH)
        self.spin_max_depth.setToolTip("0 means no max depth. A bound like 16 keeps trees fast and reduces overfitting.")
        form.addRow("Max Depth:", self.spin_max_depth)

        self.spin_test_size = QDoubleSpinBox()
        self.spin_test_size.setRange(0.05, 0.5)
        self.spin_test_size.setDecimals(2)
        self.spin_test_size.setSingleStep(0.05)
        self.spin_test_size.setValue(0.20)
        form.addRow("Test Split:", self.spin_test_size)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 99999)
        self.spin_seed.setValue(42)
        form.addRow("Random Seed:", self.spin_seed)

        self.check_balanced = QCheckBox("Use balanced class weights")
        self.check_balanced.setChecked(True)
        form.addRow("", self.check_balanced)

        self.check_auto_load_model = QCheckBox("Load trained model after save")
        self.check_auto_load_model.setChecked(True)
        form.addRow("", self.check_auto_load_model)
        setup_layout.addLayout(form)
        row_output_dir = QHBoxLayout()
        self.output_dir_edit = QLineEdit("trained_model")
        row_output_dir.addWidget(self.output_dir_edit)
        self.btn_browse_output_dir = QPushButton("Browse")
        self.btn_browse_output_dir.setStyleSheet(themed_button_style("accent"))
        self.btn_browse_output_dir.clicked.connect(self.on_browse_output_dir)
        row_output_dir.addWidget(self.btn_browse_output_dir)
        form.addRow("Output Folder:", self._wrap_layout_widget(row_output_dir))

        self.run_name_edit = QLineEdit("rf_training")
        form.addRow("Run Name:", self.run_name_edit)
        setup_layout.addStretch()
        self.tabs.addTab(self.train_model_tab, "2) Train Model")

        self.results_tab = QWidget()
        results_layout = QVBoxLayout(self.results_tab)
        results_layout.setSpacing(8)

        lbl_summary = QLabel("Run Summary")
        lbl_summary.setStyleSheet("font-weight: bold;")
        results_layout.addWidget(lbl_summary)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMinimumHeight(120)
        self.summary_text.setStyleSheet(
            "QTextEdit { font-family: Consolas, 'Courier New', monospace; font-size: 14px; }"
        )
        results_layout.addWidget(self.summary_text)

        lbl_report = QLabel("Classification Report")
        lbl_report.setStyleSheet("font-weight: bold;")
        results_layout.addWidget(lbl_report)
        self.report_text = QPlainTextEdit()
        self.report_text.setObjectName("rfReportText")
        self.report_text.setReadOnly(True)
        self.report_text.setMinimumHeight(190)
        fixed_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        fixed_font.setPointSize(12)
        self.report_text.setFont(fixed_font)
        self.report_text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.report_text.setStyleSheet(
            f"QPlainTextEdit#rfReportText {{ "
            f"font-family: Consolas, 'Courier New', monospace; "
            f"font-size: 14px; "
            f"background-color: {THEME_COLORS['panel']}; "
            f"color: {THEME_COLORS['text']}; "
            f"border: 1px solid {THEME_COLORS['accent']}; "
            f"border-radius: 6px; "
            f"padding: 4px 6px; "
            f"}}"
        )
        results_layout.addWidget(self.report_text)

        lbl_cm = QLabel("Confusion Matrix")
        lbl_cm.setStyleSheet("font-weight: bold;")
        results_layout.addWidget(lbl_cm)

        self.cm_table = QTableWidget(0, 0)
        self.cm_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cm_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.cm_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cm_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cm_table.setMinimumHeight(220)
        self.cm_table.setStyleSheet(
            f"QTableWidget {{ "
            f"font-family: Consolas, 'Courier New', monospace; font-size: 14px; "
            f"background-color: {THEME_COLORS['panel']}; color: {THEME_COLORS['text']}; "
            f"gridline-color: {THEME_COLORS['accent']}; border: 1px solid {THEME_COLORS['accent']}; "
            f"}} "
            f"QHeaderView::section {{ "
            f"background-color: {THEME_COLORS['panel']}; color: {THEME_COLORS['text']}; "
            f"border: 1px solid {THEME_COLORS['accent']}; padding: 4px; "
            f"}} "
            f"QTableCornerButton::section {{ "
            f"background-color: {THEME_COLORS['panel']}; border: 1px solid {THEME_COLORS['accent']}; "
            f"}}"
        )
        results_layout.addWidget(self.cm_table, 1)
        self.tabs.addTab(self.results_tab, "3) Results")

        row_actions = QHBoxLayout()
        self.btn_train = QPushButton("Train And Save")
        self.btn_train.setStyleSheet(themed_button_style("success"))
        self.btn_train.clicked.connect(self.on_train_clicked)
        row_actions.addWidget(self.btn_train)

        self.btn_load_previous = QPushButton("Load Previous Run")
        self.btn_load_previous.setStyleSheet(themed_button_style("accent"))
        self.btn_load_previous.clicked.connect(self.on_load_previous_run)
        row_actions.addWidget(self.btn_load_previous)

        row_actions.addStretch()
        self.btn_close = QPushButton("Close")
        self.btn_close.setStyleSheet(themed_button_style("muted"))
        self.btn_close.clicked.connect(self.close)
        row_actions.addWidget(self.btn_close)
        layout.addLayout(row_actions)

        self.lbl_status = QLabel("Status: Ready")
        self.lbl_status.setStyleSheet(themed_label_style("muted"))
        layout.addWidget(self.lbl_status)

        self._is_training = False
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.update_action_button_states()

    @staticmethod
    def _wrap_layout_widget(layout):
        w = QWidget()
        w.setLayout(layout)
        return w

    @staticmethod
    def _hex_to_rgb(hex_color, fallback=(22, 71, 106)):
        value = str(hex_color or "").strip().lstrip("#")
        if len(value) != 6:
            return fallback
        try:
            return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
        except ValueError:
            return fallback

    @staticmethod
    def _blend_rgb(c1, c2, t):
        alpha = max(0.0, min(1.0, float(t)))
        return (
            int(round(c1[0] + (c2[0] - c1[0]) * alpha)),
            int(round(c1[1] + (c2[1] - c1[1]) * alpha)),
            int(round(c1[2] + (c2[2] - c1[2]) * alpha)),
        )

    def set_status(self, text, color="#999999"):
        self.lbl_status.setText(f"Status: {text}")
        self.lbl_status.setStyleSheet(
            f"color: {themed_status_color(color)}; font-weight: bold;"
        )

    def on_tab_changed(self, _idx):
        self.update_action_button_states()

    def _on_dataset_model_changed(self, *_args):
        self.refresh_dataset_channel_info()
        self.update_action_button_states()

    @staticmethod
    def _short_basename(path):
        try:
            return os.path.basename(str(path or "").strip()) or str(path or "").strip()
        except Exception:
            return str(path or "")

    def refresh_dataset_channel_info(self, force=False):
        paths = self.dataset_paths()
        signature = tuple(os.path.normcase(os.path.abspath(p)) for p in paths)
        if (not force) and signature == self._dataset_signature:
            return
        self._dataset_signature = signature
        self.dataset_channel_counts = {}
        self.dataset_channel_errors = {}
        for path in paths:
            try:
                ch_count = int(self.visualizer.detect_training_csv_channel_count(path))
                if ch_count <= 0:
                    raise ValueError("No channel columns found.")
                self.dataset_channel_counts[path] = ch_count
            except Exception as e:
                self.dataset_channel_errors[path] = str(e)
        self._render_dataset_channel_info()

    def _render_dataset_channel_info(self):
        if self.dataset_list.count() == 0:
            self.lbl_dataset_channel_info.setText("Detected channels per file: n/a")
            self.lbl_dataset_channel_info.setStyleSheet(themed_label_style("muted"))
            return

        if len(self.dataset_channel_errors) > 0:
            first_path = next(iter(self.dataset_channel_errors.keys()))
            msg = self.dataset_channel_errors[first_path]
            self.lbl_dataset_channel_info.setText(
                "Detected channels per file: failed. "
                f"{self._short_basename(first_path)} -> {msg}. "
                "Training is disabled until dataset headers are valid."
            )
            self.lbl_dataset_channel_info.setStyleSheet(themed_label_style("danger"))
            return

        items = list(self.dataset_channel_counts.items())
        preview = ", ".join(
            [f"{self._short_basename(path)}:{count}" for path, count in items[:6]]
        )
        if len(items) > 6:
            preview = f"{preview}, +{len(items) - 6} more"
        uniq = sorted(set(int(v) for v in self.dataset_channel_counts.values()))
        if len(uniq) == 1:
            self.lbl_dataset_channel_info.setText(
                f"Detected channels per file: {preview} | Using: {uniq[0]}"
            )
            self.lbl_dataset_channel_info.setStyleSheet(themed_label_style("success"))
        else:
            self.lbl_dataset_channel_info.setText(
                f"Detected channels per file: {preview} | Mismatch detected ({uniq}). "
                "All selected files must have the same channel count. Training is disabled."
            )
            self.lbl_dataset_channel_info.setStyleSheet(themed_label_style("danger"))

    def update_action_button_states(self):
        if self._is_destroying:
            return
        try:
            self.refresh_dataset_channel_info()
            if self._is_training:
                self.btn_train.setEnabled(False)
                self.btn_load_previous.setEnabled(False)
                return

            in_train_model_tab = self.tabs.currentWidget() is self.train_model_tab
            has_dataset = self.dataset_list.count() > 0
            has_channel_errors = len(self.dataset_channel_errors) > 0
            unique_counts = sorted(set(int(v) for v in self.dataset_channel_counts.values()))
            is_channel_compatible = (not has_channel_errors) and (len(unique_counts) <= 1) and has_dataset

            # Train button is only available in Train Model tab.
            self.btn_train.setEnabled(in_train_model_tab and has_dataset and is_channel_compatible)
            # Load actions are locked in Train Model tab.
            self.btn_load_previous.setEnabled(not in_train_model_tab)
        except RuntimeError:
            # Qt may emit model signals while dialog widgets are being destroyed.
            return

    def closeEvent(self, event):
        if self._is_training:
            QMessageBox.information(self, "Training Running", "Wait for RF training to finish before closing this window.")
            event.ignore()
            return
        self._is_destroying = True
        try:
            if getattr(self, "dataset_model", None) is not None:
                self.dataset_model.rowsInserted.disconnect(self._on_dataset_model_changed)
                self.dataset_model.rowsRemoved.disconnect(self._on_dataset_model_changed)
                self.dataset_model.modelReset.disconnect(self._on_dataset_model_changed)
        except Exception:
            pass
        super().closeEvent(event)

    def dataset_paths(self):
        return [self.dataset_list.item(i).text() for i in range(self.dataset_list.count())]

    def add_dataset_paths(self, paths, clear=False):
        if clear:
            self.dataset_list.clear()
        existing = {os.path.normcase(os.path.abspath(p)) for p in self.dataset_paths()}
        for path in list(paths or []):
            p = os.path.abspath(str(path).strip())
            if not p or not os.path.isfile(p):
                continue
            key = os.path.normcase(p)
            if key in existing:
                continue
            self.dataset_list.addItem(p)
            existing.add(key)
        self.refresh_dataset_channel_info(force=True)
        self.update_action_button_states()

    def apply_default_values(self, payload):
        payload = dict(payload or {})
        self.add_dataset_paths(payload.get("dataset_paths", []), clear=True)

        self.spin_window_ms.setValue(int(max(20, payload.get("window_ms", self.spin_window_ms.value()))))
        self.spin_stride_ms.setValue(int(max(5, payload.get("stride_ms", self.spin_stride_ms.value()))))
        self.spin_estimators.setValue(int(max(50, payload.get("n_estimators", self.spin_estimators.value()))))
        self.spin_max_depth.setValue(int(max(0, payload.get("max_depth", self.spin_max_depth.value()))))
        self.spin_seed.setValue(int(max(0, payload.get("random_seed", self.spin_seed.value()))))
        self.spin_test_size.setValue(float(min(0.5, max(0.05, payload.get("test_size", self.spin_test_size.value())))))
        self.check_balanced.setChecked(bool(payload.get("class_weight_balanced", self.check_balanced.isChecked())))
        self.check_auto_load_model.setChecked(bool(payload.get("auto_load_model", self.check_auto_load_model.isChecked())))

        output_dir = str(payload.get("output_dir", self.output_dir_edit.text())).strip()
        if output_dir:
            resolved_output_dir = self.visualizer._from_project_relative_path(output_dir)
            self.output_dir_edit.setText(self.visualizer._to_project_relative_path(resolved_output_dir))
        else:
            self.output_dir_edit.setText("trained_model")

        run_name = str(payload.get("run_name", self.run_name_edit.text())).strip()
        if run_name:
            self.run_name_edit.setText(run_name)

    def on_add_dataset_files(self):
        start_dir = self.visualizer._from_project_relative_path(
            self.output_dir_edit.text().strip() or "trained_model"
        )
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Dataset CSV Files",
            start_dir,
            "CSV Files (*.csv);;All Files (*.*)",
        )
        self.add_dataset_paths(paths)

    def on_add_dataset_folder(self):
        start_dir = self.visualizer._from_project_relative_path(
            self.output_dir_edit.text().strip() or "trained_model"
        )
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder", start_dir)
        if not folder:
            return
        csv_paths = []
        for root, _dirs, files in os.walk(folder):
            for name in files:
                if name.lower().endswith(".csv"):
                    csv_paths.append(os.path.join(root, name))
        csv_paths.sort()
        if len(csv_paths) == 0:
            QMessageBox.information(self, "No CSV", "No CSV files found in selected folder.")
            return
        self.add_dataset_paths(csv_paths)

    def on_remove_selected_datasets(self):
        selected = self.dataset_list.selectedItems()
        if not selected:
            return
        rows = sorted((self.dataset_list.row(it) for it in selected), reverse=True)
        for row in rows:
            self.dataset_list.takeItem(row)
        self.update_action_button_states()

    def on_clear_datasets(self):
        self.dataset_list.clear()
        self.update_action_button_states()

    def on_browse_output_dir(self):
        start_dir = self.visualizer._from_project_relative_path(
            self.output_dir_edit.text().strip() or "trained_model"
        )
        folder = QFileDialog.getExistingDirectory(self, "Select Training Output Base Folder", start_dir)
        if folder:
            self.output_dir_edit.setText(self.visualizer._to_project_relative_path(folder))

    def build_training_config(self):
        csv_paths = self.dataset_paths()
        if len(csv_paths) == 0:
            raise ValueError("Select at least one dataset CSV file.")

        output_dir = self.output_dir_edit.text().strip() or "trained_model"
        run_name = self.visualizer._sanitize_filename_token(self.run_name_edit.text().strip() or "rf_training")

        return {
            "dataset_paths": csv_paths,
            "output_dir": output_dir,
            "run_name": run_name,
            "window_ms": int(self.spin_window_ms.value()),
            "stride_ms": int(self.spin_stride_ms.value()),
            "n_estimators": int(self.spin_estimators.value()),
            "max_depth": int(self.spin_max_depth.value()),
            "random_seed": int(self.spin_seed.value()),
            "test_size": float(self.spin_test_size.value()),
            "class_weight_balanced": bool(self.check_balanced.isChecked()),
            "auto_load_model": bool(self.check_auto_load_model.isChecked()),
        }

    def on_train_clicked(self):
        try:
            if getattr(self.visualizer, "task_session_active", False):
                QMessageBox.warning(self, "Training Setup", "Finish or cancel the active data collection session before training.")
                return
            if len(getattr(self.visualizer, "record_save_workers", [])) > 0:
                QMessageBox.warning(self, "Training Setup", "A recorded CSV is still saving. Wait for the save to finish before training.")
                return
            config = self.build_training_config()
        except Exception as e:
            QMessageBox.warning(self, "Training Setup", str(e))
            return

        self._is_training = True
        self.update_action_button_states()
        self.set_status("Training started...", "#6a1b9a")
        self.training_worker = RFTrainingWorker(self.visualizer, config, self)
        self.training_worker.status_updated.connect(self.set_status)
        self.training_worker.training_completed.connect(self.on_training_completed)
        self.training_worker.training_failed.connect(self.on_training_failed)
        self.training_worker.finished.connect(self.on_training_worker_finished)
        self.training_worker.start(QThread.LowPriority)

    def on_training_completed(self, result):
        self.apply_training_result(result)
        self.tabs.setCurrentWidget(self.results_tab)
        result = dict(result or {})
        should_auto_load = bool(dict(result.get("config", {})).get("auto_load_model", False))
        model_path = str(result.get("model_path", "")).strip()
        if should_auto_load and model_path:
            try:
                artifact = result.get("artifact", None)
                if isinstance(artifact, dict):
                    self.visualizer.apply_rf_model_artifact(artifact, model_path)
                else:
                    self.visualizer.load_rf_model(model_path)
                self.set_status("Training complete and model loaded.", "#2e7d32")
            except Exception as e:
                self.set_status("Training complete; model load failed.", "#f57c00")
                QMessageBox.warning(self, "Model Load", str(e))
        else:
            self.set_status("Training complete.", "#2e7d32")

    def on_training_failed(self, detail):
        QMessageBox.critical(self, "RF Training Error", str(detail))
        self.set_status("Training failed.", "#c62828")

    def on_training_worker_finished(self):
        if self.training_worker is not None:
            self.training_worker.deleteLater()
            self.training_worker = None
        self._is_training = False
        self.update_action_button_states()

    def on_load_previous_run(self):
        start_dir = self.visualizer._from_project_relative_path(
            self.output_dir_edit.text().strip() or "trained_model"
        )
        folder = QFileDialog.getExistingDirectory(self, "Select Saved Training Run Folder", start_dir)
        if not folder:
            return
        try:
            loaded = self.visualizer.load_saved_training_run(folder)
            setup = dict(loaded.get("config", {}))
            if "dataset_paths" not in setup:
                setup["dataset_paths"] = loaded.get("dataset_paths", [])
            if "output_dir" not in setup:
                setup["output_dir"] = os.path.dirname(os.path.abspath(folder))
            if "run_name" not in setup:
                setup["run_name"] = os.path.basename(os.path.abspath(folder))
            self.apply_default_values(setup)
            result = loaded.get("result", {})
            if result:
                self.apply_training_result(result)
                self.tabs.setCurrentWidget(self.results_tab)
            self.set_status("Loaded previous run.", "#2e7d32")
            self.update_action_button_states()
        except Exception as e:
            QMessageBox.warning(self, "Load Run", str(e))
            self.set_status("Failed to load run.", "#c62828")

    def apply_training_result(self, result):
        result = dict(result or {})
        self.current_model_path = str(result.get("model_path", "")).strip()
        self.current_run_dir = str(result.get("run_dir", "")).strip()
        self.update_action_button_states()

        summary_text = str(result.get("summary_text", "")).strip()
        if summary_text:
            self.summary_text.setPlainText(summary_text)
        else:
            self.summary_text.clear()

        report_text = str(result.get("report_text", ""))
        if report_text.strip():
            self.report_text.setPlainText(report_text.rstrip("\n"))
        else:
            self.report_text.clear()

        class_names = [str(x) for x in list(result.get("classes", []))]
        cm_list = result.get("cm", [])
        cm = np.asarray(cm_list, dtype=np.int32) if len(cm_list) > 0 else np.zeros((0, 0), dtype=np.int32)
        if cm.ndim != 2:
            cm = np.zeros((0, 0), dtype=np.int32)
        self.populate_confusion_matrix(cm, class_names)

    def populate_confusion_matrix(self, cm, class_names):
        if cm.size == 0:
            self.cm_table.clear()
            self.cm_table.setRowCount(0)
            self.cm_table.setColumnCount(0)
            return

        rows = int(cm.shape[0])
        cols = int(cm.shape[1])
        labels = list(class_names)
        if len(labels) != rows:
            labels = [f"C{i + 1}" for i in range(rows)]

        self.cm_table.clear()
        self.cm_table.setRowCount(rows)
        self.cm_table.setColumnCount(cols)
        self.cm_table.setHorizontalHeaderLabels(labels)
        self.cm_table.setVerticalHeaderLabels(labels)

        max_val = max(1, int(np.max(cm)))
        base_rgb = self._hex_to_rgb(THEME_COLORS["panel"])
        high_rgb = self._hex_to_rgb(THEME_COLORS["success"])
        text_rgb = self._hex_to_rgb(THEME_COLORS["text"], fallback=(232, 238, 240))
        muted_rgb = self._hex_to_rgb(THEME_COLORS["muted"], fallback=(169, 194, 207))
        for r in range(rows):
            for c in range(cols):
                val = int(cm[r, c])
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                ratio = (val / max_val) if max_val > 0 else 0.0
                rgb = self._blend_rgb(base_rgb, high_rgb, ratio)
                item.setBackground(QColor(rgb[0], rgb[1], rgb[2]))
                fg = muted_rgb if val <= 0 else text_rgb
                item.setForeground(QColor(fg[0], fg[1], fg[2]))
                self.cm_table.setItem(r, c, item)


class ClassLabelEditorDialog(QDialog):
    def __init__(self, labels, parent=None):
        super().__init__(parent)
        apply_app_icon(self)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowTitle("Edit Class Labels")
        self.setModal(True)
        self.resize(620, 420)
        apply_dark_title_bar(self)

        self.saved_labels = []
        self.rows = []

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self.rows_widget = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_widget)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(6)
        layout.addWidget(self.rows_widget)

        src_labels = list(labels or [])
        if len(src_labels) == 0:
            src_labels = ["" for _ in range(6)]
        elif len(src_labels) < 6:
            src_labels.extend([""] * (6 - len(src_labels)))
        for t in src_labels:
            self._add_row(str(t or ""))

        row_actions = QHBoxLayout()
        self.btn_add = QPushButton("Add Class Label")
        self.btn_add.setStyleSheet(themed_button_style("accent"))
        self.btn_add.clicked.connect(lambda: self._add_row(""))
        row_actions.addWidget(self.btn_add)

        self.btn_save = QPushButton("Save Class Label")
        self.btn_save.setStyleSheet(themed_button_style("accent"))
        self.btn_save.clicked.connect(self._save)
        row_actions.addWidget(self.btn_save)
        row_actions.addStretch()
        layout.addLayout(row_actions)

    def _refresh_titles(self):
        for idx, (row_widget, _edit, _btn) in enumerate(self.rows):
            row_layout = row_widget.layout()
            lbl = row_layout.itemAt(0).widget()
            if isinstance(lbl, QLabel):
                lbl.setText(f"Class {idx + 1}:")

    def _add_row(self, text):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        lbl = QLabel("Class:")
        row_layout.addWidget(lbl)
        edit = QLineEdit(str(text or ""))
        row_layout.addWidget(edit, stretch=1)
        btn_delete = QPushButton("Delete")
        btn_delete.setStyleSheet(themed_button_style("muted"))
        btn_delete.clicked.connect(lambda _checked=False, w=row_widget: self._delete_row(w))
        row_layout.addWidget(btn_delete)
        self.rows_layout.addWidget(row_widget)
        self.rows.append((row_widget, edit, btn_delete))
        self._refresh_titles()

    def _delete_row(self, row_widget):
        if len(self.rows) <= 1:
            return
        idx = -1
        for i, (w, _e, _b) in enumerate(self.rows):
            if w is row_widget:
                idx = i
                break
        if idx < 0:
            return
        w, _e, _b = self.rows.pop(idx)
        self.rows_layout.removeWidget(w)
        w.deleteLater()
        self._refresh_titles()

    def _save(self):
        labels = []
        for _w, edit, _b in self.rows:
            t = edit.text().strip()
            if t:
                labels.append(t)
        if len(labels) == 0:
            QMessageBox.warning(self, "Class Labels", "Provide at least one class label.")
            return
        self.saved_labels = labels
        self.accept()


STUDY_PROCEDURE_HTML = """
<h2>Duration and Procedures of the Study</h2>
<p>If you agree to participate, the researcher will first explain the study procedure and answer any
questions you may have. After you give consent, an <b>8-channel OYMotion Gravity surface EMG armband</b>
will be placed on your forearm. The device uses <b>dry electrodes</b> and is <b>non-invasive</b>.</p>
<p>You will be asked to perform repeated trials of the gesture classes. Each <b>task cycle lasts 10 seconds</b>:</p>
<ul>
<li><b>Preparation:</b> 4 seconds</li>
<li><b>Task / activity:</b> 3 seconds - perform and hold the assigned gesture</li>
<li><b>Rest:</b> 3 seconds - relax your hand and forearm</li>
</ul>
<p>The rest period is also recorded as a class for machine-learning purposes.</p>
<h3>Sessions</h3>
<ul>
<li>The study runs over <b>two sessions on two separate days</b>, normally 1-7 days apart and at about the
same time of day.</li>
<li>Each day you complete <b>18 repetitions of each gesture class</b>: 9 with the left hand and 9 with the
right hand.</li>
<li>Across both days: <b>36 repetitions x 10 classes = about 360 task cycles</b>.</li>
<li>Estimated active task time: 36 x 10 x 10 s = 3600 s, i.e. about <b>60 minutes</b> in total, or about
<b>30 minutes of active recording on each day</b>.</li>
</ul>
<h3>Breaks and fatigue</h3>
<ul>
<li>No more than <b>30 task cycles (about 5 minutes)</b> without a break.</li>
<li>After each block: a <b>3-4 minute break</b>.</li>
<li>When the armband moves to your other arm: a further break of about <b>5 minutes</b>.</li>
<li>You may ask for an additional break at any time.</li>
<li>Room temperature and humidity are noted at the start and end of each session.</li>
</ul>
<p>Including breaks, hand changeover, consent, device fitting and initial instructions, each session is
expected to take about <b>60 to 70 minutes</b>.</p>
"""


class StudyProcedureDialog(QDialog):
    """Read-only summary of the study duration and procedure (replaces the old T&C popup)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        apply_app_icon(self)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowTitle("Study Procedure")
        self.resize(720, 640)
        apply_dark_title_bar(self)
        layout = QVBoxLayout(self)
        view = QTextBrowser()
        view.setOpenExternalLinks(False)
        view.setHtml(STUDY_PROCEDURE_HTML)
        layout.addWidget(view, 1)
        btn_close = QPushButton("Close")
        btn_close.setStyleSheet(themed_button_style("accent"))
        btn_close.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(btn_close)
        layout.addLayout(row)


def _section_title(text, hint=""):
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    title = QLabel(text)
    title.setStyleSheet(f"font-weight: bold; font-size: 16px; color: {THEME_COLORS['text']};")
    row.addWidget(title)
    if hint:
        lbl = QLabel(hint)
        lbl.setStyleSheet(themed_label_style("muted"))
        row.addWidget(lbl)
    row.addStretch()
    return row


class DataCollectionDialog(QDialog):
    start_requested = pyqtSignal(dict)
    settings_changed = pyqtSignal(dict)
    protocol_phase_started = pyqtSignal(str, int, str, bool, str, int, int, int)
    protocol_finished = pyqtSignal()
    protocol_canceled = pyqtSignal()
    protocol_paused = pyqtSignal(bool)
    protocol_rewind = pyqtSignal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        apply_app_icon(self)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowTitle("Data Collection")
        self.setModal(False)
        # Content scrolls inside the dialog, so any screen size can reach every control.
        target_w, target_h = 1020, 860
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            target_w = min(target_w, max(640, int(avail.width() * 0.92)))
            target_h = min(target_h, max(520, int(avail.height() * 0.92)))
        self.resize(target_w, target_h)
        apply_dark_title_bar(self)

        settings = dict(settings or {})
        labels = settings.get("task_labels", [])
        if not isinstance(labels, list):
            labels = [x.strip() for x in str(labels).split(",") if x.strip()]
        self.task_labels = [str(x).strip() for x in labels if str(x).strip()]
        self.labels_saved = bool(settings.get("labels_saved", len(self.task_labels) > 0))
        self.protocol_running = False
        self.session_prepared = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignTop)

        self.lbl_plan = QLabel("")
        self.lbl_plan.setWordWrap(True)
        self.lbl_plan.setStyleSheet(f"color: {THEME_COLORS['accent']}; font-weight: bold;")
        layout.addWidget(self.lbl_plan)

        self.consent_section = QWidget(self)
        consent_layout = QVBoxLayout(self.consent_section)
        consent_layout.setContentsMargins(0, 0, 0, 0)
        consent_layout.setSpacing(8)

        row_contributor = QHBoxLayout()
        row_contributor.addWidget(QLabel("Participant ID:"))
        self.input_contributor = QLineEdit(str(settings.get("participant_id", "")))
        self.input_contributor.setPlaceholderText("S01  (pseudonymous ID - never a name)")
        self.input_contributor.setMaxLength(4)
        self.input_contributor.setToolTip("Pseudonymous participant ID: the letter S followed by 2-3 digits (S01, S02, ... S101).\n"
                                          "Names are never stored in the dataset.")
        row_contributor.addWidget(self.input_contributor, stretch=1)
        self.btn_terms = QPushButton("Study Procedure")
        self.btn_terms.setStyleSheet(themed_button_style("accent"))
        self.btn_terms.setToolTip("Read the duration and procedures of the study.")
        self.btn_terms.clicked.connect(self._show_terms)
        row_contributor.addWidget(self.btn_terms)
        self.check_agree_yes = QCheckBox("Agree")
        self.check_agree_no = QCheckBox("Disagree")
        is_agreed = bool(settings.get("agreed", False))
        self.check_agree_yes.setChecked(is_agreed)
        self.check_agree_no.setChecked(not is_agreed)
        self.check_agree_yes.setToolTip("I have read the study procedure and agree to take part.")
        row_contributor.addWidget(self.check_agree_yes)
        row_contributor.addWidget(self.check_agree_no)
        consent_layout.addLayout(row_contributor)

        # Additional information (participant info + room conditions) is tucked behind a button.
        self.btn_additional = QPushButton("Additional Information  \u25b8")
        self.btn_additional.setCheckable(True)
        self.btn_additional.setStyleSheet(themed_button_style("muted"))
        self.btn_additional.setToolTip("Participant info and room conditions (all optional).")
        consent_layout.addWidget(self.btn_additional, 0, Qt.AlignLeft)
        self.extra_panel = QWidget()
        extra_layout = QVBoxLayout(self.extra_panel)
        extra_layout.setContentsMargins(0, 0, 0, 0)
        extra_layout.setSpacing(8)
        self.extra_panel.setVisible(False)
        self.btn_additional.toggled.connect(self._toggle_additional)

        # Participant info (saved to the metadata file with each recording)
        extra_layout.addLayout(_section_title(
            "Participant Info", "(all optional - fields left empty are saved as N/A)"))

        def _make_spin(minimum, maximum, step, decimals, suffix, value):
            sp = QDoubleSpinBox()
            sp.setRange(minimum, maximum)
            sp.setDecimals(decimals)
            sp.setSingleStep(step)
            sp.setSuffix(suffix)
            sp.setSpecialValueText("N/A")  # shown while the value is at the minimum (= not provided)
            try:
                sp.setValue(float(value))
            except (TypeError, ValueError):
                sp.setValue(minimum)
            return sp

        participant_grid = QGridLayout()
        participant_grid.setHorizontalSpacing(10)
        participant_grid.setVerticalSpacing(6)
        self.combo_blood_group = QComboBox()
        self.combo_blood_group.addItem("N/A")
        self.combo_blood_group.addItems(BLOOD_GROUP_OPTIONS)
        saved_bg = str(settings.get("blood_group", "")).strip().upper()
        if saved_bg in BLOOD_GROUP_OPTIONS:
            self.combo_blood_group.setCurrentText(saved_bg)
        self.spin_height = _make_spin(0.0, 250.0, 0.5, 1, " cm", settings.get("height_cm", 0.0))
        self.spin_weight = _make_spin(0.0, 300.0, 0.5, 1, " kg", settings.get("weight_kg", 0.0))
        self.spin_age = QSpinBox()
        self.spin_age.setRange(0, 120)
        self.spin_age.setSuffix(" yrs")
        self.spin_age.setSpecialValueText("N/A")
        self.spin_age.setValue(int(settings.get("age_years", 0) or 0))
        # No practical upper cap (Qt spin boxes need a maximum, so it is set very high).
        circumference = settings.get("forearm_circumference_cm", settings.get("forearm_radius_cm", 0.0))
        self.spin_forearm_circumference = _make_spin(0.0, 9999.99, 0.1, 2, " cm", circumference)
        self.spin_forearm_circumference.setToolTip(
            "Forearm circumference at the armband position, in cm.\n"
            "Wrap a tape measure once around the forearm where the armband sits."
        )
        for col, (text, widget) in enumerate([
            ("Blood Group:", self.combo_blood_group), ("Height:", self.spin_height),
            ("Weight:", self.spin_weight), ("Age:", self.spin_age),
            ("Forearm Circumference:", self.spin_forearm_circumference),
        ]):
            row, base = divmod(col, 2)
            participant_grid.addWidget(QLabel(text), row, base * 2)
            participant_grid.addWidget(widget, row, base * 2 + 1)
        participant_grid.setColumnStretch(1, 1)
        participant_grid.setColumnStretch(3, 1)
        extra_layout.addLayout(participant_grid)

        extra_layout.addLayout(_section_title(
            "Room Conditions", "(note at the start of the session; optional)"))
        env_grid = QGridLayout()
        env_grid.setHorizontalSpacing(10)
        env_grid.setVerticalSpacing(6)
        self.spin_temp_start = _make_spin(0.0, 60.0, 0.5, 1, " °C", settings.get("temp_start_c", 0.0))
        self.spin_humidity_start = _make_spin(0.0, 100.0, 1.0, 0, " %", settings.get("humidity_start_pct", 0.0))
        for col, (text, widget) in enumerate([
            ("Temperature:", self.spin_temp_start), ("Humidity:", self.spin_humidity_start),
        ]):
            row, base = divmod(col, 2)
            env_grid.addWidget(QLabel(text), row, base * 2)
            env_grid.addWidget(widget, row, base * 2 + 1)
        env_grid.setColumnStretch(1, 1)
        env_grid.setColumnStretch(3, 1)
        extra_layout.addLayout(env_grid)

        consent_layout.addWidget(self.extra_panel)
        layout.addWidget(self.consent_section)

        self.session_section = QWidget()
        session_layout = QVBoxLayout(self.session_section)
        session_layout.setContentsMargins(0, 0, 0, 0)
        session_layout.setSpacing(8)
        session_layout.setAlignment(Qt.AlignTop)

        self.session_config_section = QWidget(self.session_section)
        config_layout = QVBoxLayout(self.session_config_section)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(8)
        config_layout.setAlignment(Qt.AlignTop)

        row_setup = QHBoxLayout()
        lbl_setup = QLabel("Class Label Setup")
        lbl_setup.setStyleSheet("font-weight: bold; font-size: 16px;")
        row_setup.addWidget(lbl_setup)
        row_setup.addStretch()
        self.btn_edit_labels = QPushButton("Edit")
        self.btn_edit_labels.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self.btn_edit_labels.setStyleSheet(themed_button_style("accent"))
        self.btn_edit_labels.clicked.connect(self._open_class_label_editor)
        row_setup.addWidget(self.btn_edit_labels)
        config_layout.addLayout(row_setup)

        row_labels = QHBoxLayout()
        row_labels.addWidget(QLabel("Class Labels:"))
        self.input_class_labels = QLineEdit("")
        self.input_class_labels.setReadOnly(True)
        row_labels.addWidget(self.input_class_labels, stretch=1)
        config_layout.addLayout(row_labels)

        self.lbl_labels_state = QLabel("")
        self.lbl_labels_state.setStyleSheet(themed_label_style("muted"))
        self.lbl_labels_state.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        config_layout.addWidget(self.lbl_labels_state)

        config_layout.addLayout(_section_title("Session Plan"))
        plan_grid = QGridLayout()
        plan_grid.setHorizontalSpacing(10)
        plan_grid.setVerticalSpacing(6)

        self.spin_session_day = QSpinBox()
        self.spin_session_day.setRange(1, 2)
        self.spin_session_day.setPrefix("Day ")
        self.spin_session_day.setValue(int(settings.get("session_day", DEFAULT_SESSION_DAY)))
        self.combo_hand_order = QComboBox()
        for key, text in HAND_ORDER_OPTIONS:
            self.combo_hand_order.addItem(text, key)
        idx = self.combo_hand_order.findData(str(settings.get("hand_order", DEFAULT_HAND_ORDER)))
        self.combo_hand_order.setCurrentIndex(max(0, idx))
        self.spin_task_repeats = QSpinBox()
        self.spin_task_repeats.setRange(1, 100)
        self.spin_task_repeats.setValue(int(settings.get("repeats", DEFAULT_TASK_REPEATS)))
        self.spin_task_repeats.setToolTip("Repetitions of every class per hand (study: 9 per hand per day).")

        def _seconds_spin(minimum, maximum, step, value, decimals=1, suffix=" s"):
            sp = QDoubleSpinBox()
            sp.setRange(minimum, maximum)
            sp.setSingleStep(step)
            sp.setDecimals(decimals)
            sp.setSuffix(suffix)
            sp.setValue(float(value))
            return sp

        self.spin_task_prep = _seconds_spin(0.2, 30.0, 0.5, settings.get("prep_s", DEFAULT_TASK_PREP_S))
        self.spin_task_hold = _seconds_spin(0.2, 30.0, 0.5, settings.get("hold_s", DEFAULT_TASK_HOLD_S))
        self.spin_task_rest = _seconds_spin(0.2, 30.0, 0.5, settings.get("rest_s", DEFAULT_TASK_REST_S))
        # Fixed by the study protocol, so not shown: no settle gap, 30 cycles per block, 3.5 min block break.
        self._fixed_plan = {
            "settle_s": float(settings.get("settle_s", DEFAULT_TASK_SETTLE_S)),
            "max_cycles_per_block": int(settings.get("max_cycles_per_block", DEFAULT_MAX_CYCLES_PER_BLOCK)),
            "block_break_s": float(settings.get("block_break_s", DEFAULT_BLOCK_BREAK_S)),
        }
        self.spin_hand_switch = _seconds_spin(
            0, 1800, 10, settings.get("hand_switch_s", DEFAULT_HAND_SWITCH_BREAK_S), 0)
        self.spin_hand_switch.setToolTip("Break while the armband is moved to the other arm (study: about 5 minutes).")
        self.check_record_rest = QCheckBox("Record Rest")
        self.check_record_rest.setChecked(bool(settings.get("record_rest", True)))
        self.session_seed = int(settings.get("seed", 0) or 0) or secrets.randbits(32)
        self._hand_order_manual = str(settings.get("hand_order_source", "auto")) == "manual"
        self.check_practice = QCheckBox("Practice / demo mode")
        self.check_practice.setChecked(bool(settings.get("practice_mode", False)))
        self.check_practice.setToolTip("Practice runs are saved under a 'practice_' folder and are NOT part of the dataset.\n"
                                       "Only in this mode can randomization be switched off.")
        self.check_randomize_order = QCheckBox("Randomize Gesture Order")
        self.check_randomize_order.setChecked(bool(settings.get("randomize_order", True)))
        self.check_randomize_order.setToolTip(
            "Shuffle gesture order independently each repeat, so fatigue doesn't always "
            "degrade whichever gesture is fixed last in the cycle."
        )

        plan_fields = [
            ("Session:", self.spin_session_day), ("Hands:", self.combo_hand_order),
            ("Repeats per hand:", self.spin_task_repeats),
            ("Prepare:", self.spin_task_prep), ("Perform:", self.spin_task_hold), ("Rest:", self.spin_task_rest),
            ("Hand-switch break:", self.spin_hand_switch),
        ]
        for i, (text, widget) in enumerate(plan_fields):
            row, base = divmod(i, 2)
            plan_grid.addWidget(QLabel(text), row, base * 2)
            plan_grid.addWidget(widget, row, base * 2 + 1)
        for col in (1, 3):
            plan_grid.setColumnStretch(col, 1)
        config_layout.addLayout(plan_grid)

        row_checks = QHBoxLayout()
        row_checks.addWidget(self.check_record_rest)
        row_checks.addWidget(self.check_randomize_order)
        row_checks.addWidget(self.check_practice)
        row_checks.addStretch()
        config_layout.addLayout(row_checks)

        row_csv = QHBoxLayout()
        row_csv.addWidget(QLabel("CSV Save Folder:"))
        self.input_record_dir = QLineEdit(str(settings.get("csv_dir", DATASET_DIR)))
        row_csv.addWidget(self.input_record_dir, stretch=1)
        self.btn_browse_record_dir = QPushButton("Browse")
        self.btn_browse_record_dir.setStyleSheet(themed_button_style("accent"))
        self.btn_browse_record_dir.clicked.connect(self._browse_record_dir)
        row_csv.addWidget(self.btn_browse_record_dir)
        config_layout.addLayout(row_csv)

        row_actions = QHBoxLayout()
        self.btn_start_session = QPushButton("Proceed")
        self.btn_start_session.setStyleSheet(themed_button_style("accent"))
        self.btn_start_session.clicked.connect(self._emit_start_requested)
        row_actions.addWidget(self.btn_start_session)
        row_actions.addStretch()
        config_layout.addLayout(row_actions)

        config_layout.addSpacing(8)
        self.protocol_divider = QFrame()
        self.protocol_divider.setFrameShape(QFrame.HLine)
        self.protocol_divider.setFrameShadow(QFrame.Plain)
        self.protocol_divider.setStyleSheet(
            f"border: none; background-color: {THEME_COLORS['accent']}; min-height: 1px; max-height: 1px;"
        )
        config_layout.addWidget(self.protocol_divider)
        config_layout.addSpacing(8)

        session_layout.addWidget(self.session_config_section)

        self.task_protocol = TaskProtocolWidget(self.session_section)
        self.task_protocol.start_clicked.connect(self._on_protocol_start_clicked)
        self.task_protocol.phase_started.connect(self._on_protocol_phase_started)
        self.task_protocol.protocol_finished.connect(self._on_protocol_finished)
        self.task_protocol.protocol_canceled.connect(self._on_protocol_canceled)
        self.task_protocol.paused_changed.connect(self.protocol_paused.emit)
        self.task_protocol.rewind_requested.connect(self.protocol_rewind.emit)
        self.consent_opacity = QGraphicsOpacityEffect(self.consent_section)
        self.consent_section.setGraphicsEffect(self.consent_opacity)
        self.session_config_opacity = QGraphicsOpacityEffect(self.session_config_section)
        self.session_config_section.setGraphicsEffect(self.session_config_opacity)
        self._set_upper_section_locked(False)
        self.protocol_opacity = QGraphicsOpacityEffect(self.task_protocol)
        self.task_protocol.setGraphicsEffect(self.protocol_opacity)
        self._set_protocol_dimmed(True)
        session_layout.addWidget(self.task_protocol)
        layout.addWidget(self.session_section)

        for sig in (
            self.input_contributor.textChanged,
            self.combo_blood_group.currentIndexChanged,
            self.spin_height.valueChanged, self.spin_weight.valueChanged, self.spin_age.valueChanged,
            self.spin_forearm_circumference.valueChanged,
            self.spin_temp_start.valueChanged, self.spin_humidity_start.valueChanged,
            self.spin_session_day.valueChanged, self.combo_hand_order.currentIndexChanged,
            self.spin_task_repeats.valueChanged, self.spin_task_prep.valueChanged,
            self.spin_task_hold.valueChanged, self.spin_task_rest.valueChanged,
            self.spin_hand_switch.valueChanged,
            self.check_record_rest.stateChanged, self.check_randomize_order.stateChanged,
            self.check_practice.stateChanged,
            self.input_record_dir.textChanged,
        ):
            sig.connect(self._emit_settings_changed)
        self.check_practice.stateChanged.connect(self._apply_practice_rules)
        self.combo_hand_order.activated.connect(self._on_hand_order_chosen)
        self.input_contributor.textChanged.connect(self._on_participant_id_changed)
        self._apply_practice_rules()
        self._on_participant_id_changed()
        self.check_agree_yes.stateChanged.connect(self._on_agree_yes_changed)
        self.check_agree_no.stateChanged.connect(self._on_agree_no_changed)

        self._refresh_class_labels_display()
        self._update_access_state()
        self._update_plan_label()

    # -- helpers ---------------------------------------------------------------
    def _update_plan_label(self):
        plan = plan_summary(self.settings_payload())
        hands = " + ".join(plan["hands"])
        self.lbl_plan.setText(
            f"This session: {hands} hand | {plan['cycles']} cycles x {plan['cycle_s']:.0f} s = "
            f"{format_duration(plan['active_s'] * 1000, False)} active | {plan['blocks_per_hand']} block(s) per hand | "
            f"about {format_duration(plan['total_s'] * 1000, False)} including breaks"
        )

    def _refresh_class_labels_display(self):
        text = ", ".join(self.task_labels)
        self.input_class_labels.setText(text)
        if self.labels_saved and len(self.task_labels) > 0:
            self.lbl_labels_state.setText(f"Saved {len(self.task_labels)} class labels.")
            self.lbl_labels_state.setStyleSheet(themed_label_style("success"))
        else:
            self.lbl_labels_state.setText("No class labels saved yet. Click Edit to configure and save.")
            self.lbl_labels_state.setStyleSheet(themed_label_style("muted"))

    def _open_class_label_editor(self):
        dlg = ClassLabelEditorDialog(self.task_labels, self)
        center_window(dlg, self)
        if dlg.exec_() == QDialog.Accepted:
            self.task_labels = list(dlg.saved_labels)
            self.labels_saved = True
            self._refresh_class_labels_display()
            self._update_access_state()
            self._emit_settings_changed()

    def settings_payload(self):
        return {
            "participant_id": normalize_participant_id(self.input_contributor.text()),
            "blood_group": self.combo_blood_group.currentText() if self.combo_blood_group.currentIndex() > 0 else "",
            "height_cm": float(self.spin_height.value()),
            "weight_kg": float(self.spin_weight.value()),
            "age_years": int(self.spin_age.value()),
            "forearm_circumference_cm": float(self.spin_forearm_circumference.value()),
            "temp_start_c": float(self.spin_temp_start.value()),
            "humidity_start_pct": float(self.spin_humidity_start.value()),
            "agreed": bool(self.check_agree_yes.isChecked() and not self.check_agree_no.isChecked()),
            "seed": int(self.session_seed),
            "practice_mode": bool(self.check_practice.isChecked()),
            "hand_order_source": "manual" if self._hand_order_manual else "auto",
            "hand_order_default": (default_hand_order(self.input_contributor.text())
                                   if valid_participant_id(self.input_contributor.text()) else ""),
            "task_labels": list(self.task_labels),
            "labels_saved": bool(self.labels_saved),
            "session_day": int(self.spin_session_day.value()),
            "hand_order": str(self.combo_hand_order.currentData() or DEFAULT_HAND_ORDER),
            "repeats": int(self.spin_task_repeats.value()),
            "prep_s": float(self.spin_task_prep.value()),
            "hold_s": float(self.spin_task_hold.value()),
            "rest_s": float(self.spin_task_rest.value()),
            "record_rest": bool(self.check_record_rest.isChecked()),
            **self._fixed_plan,
            "hand_switch_s": float(self.spin_hand_switch.value()),
            "randomize_order": bool(self.check_randomize_order.isChecked()),
            "csv_dir": self.input_record_dir.text().strip(),
        }

    def set_csv_dir(self, path):
        self.input_record_dir.setText(str(path or ""))

    def set_csv_counter(self, text, kind="accent"):
        self.task_protocol.set_csv_counter(text, kind)

    def _toggle_additional(self, shown):
        self.extra_panel.setVisible(bool(shown))
        self.btn_additional.setText("Additional Information  " + ("\u25be" if shown else "\u25b8"))

    def _show_terms(self):
        dlg = StudyProcedureDialog(self)
        center_window(dlg, self)
        dlg.exec_()

    def _browse_record_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select CSV Save Folder",
            self.input_record_dir.text().strip() or DATASET_DIR,
        )
        if path:
            self.input_record_dir.setText(path)

    def _on_agree_yes_changed(self, _state):
        if self.check_agree_yes.isChecked():
            self.check_agree_no.blockSignals(True)
            self.check_agree_no.setChecked(False)
            self.check_agree_no.blockSignals(False)
        elif not self.check_agree_no.isChecked():
            self.check_agree_no.blockSignals(True)
            self.check_agree_no.setChecked(True)
            self.check_agree_no.blockSignals(False)
        self._update_access_state()
        self._emit_settings_changed()

    def _on_agree_no_changed(self, _state):
        if self.check_agree_no.isChecked():
            self.check_agree_yes.blockSignals(True)
            self.check_agree_yes.setChecked(False)
            self.check_agree_yes.blockSignals(False)
        elif not self.check_agree_yes.isChecked():
            self.check_agree_yes.blockSignals(True)
            self.check_agree_yes.setChecked(True)
            self.check_agree_yes.blockSignals(False)
        self._update_access_state()
        self._emit_settings_changed()

    def _update_access_state(self):
        unlocked = self.check_agree_yes.isChecked() and (not self.check_agree_no.isChecked())
        lock_upper = self.protocol_running or self.session_prepared
        if lock_upper:
            self._set_upper_section_locked(True)
        else:
            self._set_upper_section_locked(False)
            self.session_config_section.setEnabled(unlocked)
        self.session_section.setEnabled(True)
        can_proceed = unlocked and self.labels_saved and len(self.task_labels) > 0 and (not self.protocol_running) and (not self.session_prepared)
        self.btn_start_session.setEnabled(can_proceed)
        self.task_protocol.set_start_enabled(self.session_prepared and (not self.protocol_running))
        self.task_protocol.set_cancel_enabled(self.session_prepared and (not self.protocol_running))

    def _apply_practice_rules(self, *_args):
        """Study mode: randomization is mandatory. Only practice/demo mode may switch it off."""
        practice = self.check_practice.isChecked()
        if not practice:
            self.check_randomize_order.setChecked(True)
        self.check_randomize_order.setEnabled(practice)
        self.check_randomize_order.setToolTip(
            "Shuffle gesture order independently each repeat, so fatigue doesn't always degrade whichever "
            "gesture is fixed last in the cycle." + ("" if practice else
                                                      "\nMandatory in study mode (tick Practice / demo mode to change it)."))

    def _on_hand_order_chosen(self, *_args):
        self._hand_order_manual = True          # the operator overrode the counterbalanced default

    def _on_participant_id_changed(self, *_args):
        """Odd participant numbers start with the left hand, even with the right, unless overridden."""
        pid = self.input_contributor.text()
        if self._hand_order_manual or not valid_participant_id(pid):
            return
        if str(self.combo_hand_order.currentData()) not in ("left_right", "right_left"):
            return
        idx = self.combo_hand_order.findData(default_hand_order(pid))
        if idx >= 0 and idx != self.combo_hand_order.currentIndex():
            self.combo_hand_order.setCurrentIndex(idx)

    def _emit_settings_changed(self, *_args):
        self._update_plan_label()
        self.settings_changed.emit(self.settings_payload())

    def _protocol_kwargs(self, settings):
        return dict(
            labels=settings["task_labels"],
            repeats=settings["repeats"],
            prep_s=settings["prep_s"],
            hold_s=settings["hold_s"],
            rest_s=settings["rest_s"],
            record_rest=settings["record_rest"],
            settle_s=settings["settle_s"],
            randomize_order=settings["randomize_order"],
            hand_order=settings["hand_order"],
            session_day=settings["session_day"],
            max_cycles_per_block=settings["max_cycles_per_block"],
            block_break_s=settings["block_break_s"],
            hand_switch_s=settings["hand_switch_s"],
            seed=settings["seed"],
        )

    def _emit_start_requested(self):
        if self.protocol_running:
            QMessageBox.information(self, "Session Running", "A task session is already running.")
            return
        if self.session_prepared:
            QMessageBox.information(self, "Ready", "Session prepared. Click Start Recording below.")
            return
        if not self.check_agree_yes.isChecked() or self.check_agree_no.isChecked():
            QMessageBox.warning(self, "Agreement Required", "You must agree to contribute before proceeding.")
            return
        if not valid_participant_id(self.input_contributor.text()):
            QMessageBox.warning(self, "Participant ID", "The participant ID must be the letter S followed by 2-3 digits "
                                "(for example S01). Do not enter a name.")
            return
        # Participant info and room conditions are optional; anything left empty is
        # written to the metadata file as N/A.
        if not self.labels_saved or len(self.task_labels) == 0:
            QMessageBox.warning(self, "Task Labels", "Click Edit and save class labels before proceeding.")
            return
        self.session_seed = secrets.randbits(32)     # fresh seed per session; saved with the recording
        self._emit_settings_changed()                # let the main window store the seed it will start with
        self.task_protocol.configure(**self._protocol_kwargs(self.settings_payload()))
        self.session_prepared = True
        self._set_protocol_dimmed(False)
        self._update_access_state()

    def start_task_protocol(self, **kwargs):
        if self.protocol_running or (not self.session_prepared):
            return False
        self.task_protocol.configure(**kwargs)
        started = self.task_protocol.start_protocol()
        if not started:
            return False
        self.protocol_running = True
        self._set_upper_section_locked(True)
        self._set_protocol_dimmed(False)
        self._update_access_state()
        return True

    def _on_protocol_start_clicked(self):
        if self.protocol_running:
            return
        if not self.session_prepared:
            QMessageBox.information(self, "Proceed First", "Click Proceed first, then Start Recording.")
            return
        self.start_requested.emit(self.settings_payload())

    def cancel_task_protocol(self, emit_signal=False):
        self.task_protocol.cancel_protocol(emit_signal=emit_signal)

    def _set_upper_section_locked(self, locked):
        opacity = 0.42 if locked else 1.0
        if hasattr(self, "consent_opacity") and self.consent_opacity is not None:
            self.consent_opacity.setOpacity(opacity)
        if hasattr(self, "session_config_opacity") and self.session_config_opacity is not None:
            self.session_config_opacity.setOpacity(opacity)
        self.consent_section.setEnabled(not locked)
        if locked:
            self.session_config_section.setEnabled(False)

    def _set_protocol_dimmed(self, dimmed):
        if hasattr(self, "protocol_opacity") and self.protocol_opacity is not None:
            self.protocol_opacity.setOpacity(0.42 if dimmed else 1.0)

    def _on_protocol_phase_started(self, label, trial_id, phase_name, record_enabled, hand, block, cycle_no, planned_ms):
        self.protocol_phase_started.emit(label, trial_id, phase_name, record_enabled, hand, block, cycle_no, planned_ms)

    def _on_protocol_finished(self):
        self.protocol_running = False
        self.session_prepared = False
        self._set_upper_section_locked(False)
        self._set_protocol_dimmed(True)
        self.btn_start_session.setText("Proceed")
        self._update_access_state()
        self.protocol_finished.emit()

    def _on_protocol_canceled(self):
        self.protocol_running = False
        self.session_prepared = False
        self._set_upper_section_locked(False)
        self._set_protocol_dimmed(True)
        self.btn_start_session.setText("Proceed")
        self._update_access_state()
        self.protocol_canceled.emit()

    def closeEvent(self, event):
        if self.protocol_running:
            self.cancel_task_protocol(emit_signal=True)
        super().closeEvent(event)



class AnalysisWindow(QMainWindow):
    closed = pyqtSignal()

    def __init__(self, num_channels):
        super().__init__()
        apply_app_icon(self)
        self.num_channels = num_channels
        self.setWindowTitle("Analysis Window")
        fit_window_to_screen(self, 1220, 900)
        apply_dark_title_bar(self)

        central = QWidget()
        wrap_in_scroll_area(self, central)
        self.setStyleSheet(
            f"QMainWindow {{ background: {THEME_COLORS['bg']}; }} "
            f"QWidget {{ background: {THEME_COLORS['bg']}; color: {THEME_COLORS['text']}; }} "
            f"QLabel {{ color: {THEME_COLORS['text']}; font-size: 18px; }} "
            f"QPushButton#sectionNav {{ background: {THEME_COLORS['panel']}; color: {THEME_COLORS['muted']}; "
            f"padding: 10px 16px; border: 1px solid {THEME_COLORS['accent']}; font-size: 17px; font-weight: 600; "
            f"min-height: 26px; min-width: 120px; border-radius: 6px; }} "
            f"QPushButton#sectionNav:checked {{ background: {THEME_COLORS['accent']}; color: {THEME_COLORS['text']}; }} "
            f"QPushButton#sectionNav:hover:!checked {{ background: {THEME_COLORS['bg']}; color: {THEME_COLORS['text']}; }} "
            f"QHeaderView::section {{ background-color: {THEME_COLORS['panel']}; color: {THEME_COLORS['text']}; "
            f"border: 1px solid {THEME_COLORS['accent']}; padding: 6px; font-size: 16px; font-weight: 600; }}"
        )
        layout = QVBoxLayout(central)

        title = QLabel("Analysis Window")
        title.setWordWrap(True)
        title.setStyleSheet("font-size: 22px; font-weight: bold; padding-left: 9px;")
        layout.addWidget(title)

        self.page_overview, self.tbl_overview, self.lbl_desc_overview = self.make_table_tab(
            ["Metric", "Value"],
            "Overview summarizes overall system state and high-level intent outputs.",
        )
        self.page_time, self.tbl_time, self.lbl_desc_time = self.make_table_tab(
            ["CH", "MAV", "RMS", "iEMG", "VAR", "WL", "ZC", "SSC", "WAMP"],
            "Time-domain features per channel. MAV/RMS/iEMG track amplitude-energy; VAR/WL track variability/complexity; "
            "ZC/SSC/WAMP are event-count features.",
        )
        self.page_freq, self.tbl_freq, self.lbl_desc_freq = self.make_table_tab(
            ["CH", "MeanF", "MedianF", "PeakF", "Entropy", "P20-60", "P60-120", "P120-220", "MainsScore"],
            "Frequency-domain features per channel. Band columns are percentage power in each band; MainsScore estimates "
            "50/60 Hz contamination.",
        )
        self.freq_plot_title = QLabel("Realtime Band Power (%)")
        self.freq_plot_title.setStyleSheet("font-weight: bold;")
        self.page_freq.layout().addWidget(self.freq_plot_title)
        self.freq_band_plot = pg.PlotWidget()
        self.freq_band_plot.setBackground(THEME_COLORS["panel"])
        self.freq_band_plot.setMinimumHeight(230)
        self.freq_band_plot.setMouseEnabled(x=False, y=False)
        self.freq_band_plot.setMenuEnabled(False)
        self.freq_band_plot.showGrid(x=True, y=True, alpha=0.18)
        self.freq_band_plot.setYRange(0, 100)
        self.freq_band_plot.setLabel("left", "Power (%)")
        self.freq_band_plot.setLabel("bottom", "Channel")
        self.freq_band_plot.getAxis("left").setPen(pg.mkPen(THEME_COLORS["muted"]))
        self.freq_band_plot.getAxis("bottom").setPen(pg.mkPen(THEME_COLORS["muted"]))
        self.freq_band_plot.getAxis("left").setTextPen(pg.mkPen(THEME_COLORS["text"]))
        self.freq_band_plot.getAxis("bottom").setTextPen(pg.mkPen(THEME_COLORS["text"]))
        self.page_freq.layout().addWidget(self.freq_band_plot)
        self.lbl_desc_freq_plot = self.make_description_label(
            "Grouped bars show per-channel band-power percentage for 20-60 Hz, 60-120 Hz, and 120-220 Hz."
        )
        self.page_freq.layout().addWidget(self.lbl_desc_freq_plot)
        self.freq_band_colors = ["#2EC4B6", "#3A86FF", "#FFB703"]
        legend_row = QHBoxLayout()
        legend_row.setSpacing(18)
        band_labels = ["20-60 Hz", "60-120 Hz", "120-220 Hz"]
        for idx, band_text in enumerate(band_labels):
            swatch = QLabel("   ")
            swatch.setFixedSize(18, 18)
            swatch.setStyleSheet(
                f"background: {self.freq_band_colors[idx]}; border: 1px solid {THEME_COLORS['accent']}; border-radius: 3px;"
            )
            txt = QLabel(band_text)
            txt.setStyleSheet(f"color: {THEME_COLORS['text']}; font-size: 15px;")
            legend_row.addWidget(swatch)
            legend_row.addWidget(txt)
        legend_row.addStretch()
        self.page_freq.layout().addLayout(legend_row)
        self.freq_band_offsets = np.array([-0.26, 0.0, 0.26], dtype=np.float32)
        self.freq_band_width = 0.22
        self.freq_band_items = []
        self._freq_band_plot_nch = None
        self._refresh_band_plot_ticks()
        self.page_tfr, self.tbl_tfr, self.lbl_desc_tfr = self.make_table_tab(
            ["CH", "STFT Mean", "STFT Std", "dP20-60", "dP60-120", "dP120-220", "A3", "D3", "D2", "D1"],
            "Time-frequency features per channel. dP values are end-minus-start band-power deltas. A3/D3/D2/D1 are wavelet "
            "energy percentages (if pywt is installed).",
        )
        self.page_events, self.tbl_events, self.lbl_desc_events = self.make_table_tab(
            ["CH", "Onset", "Offset", "Reps", "BurstMean(ms)", "BurstNow(ms)", "QualityScore"],
            "Event detection/segmentation metrics per channel: onset/offset counts, repetition count, burst timing, and "
            "contraction-quality score.",
        )
        self.page_ml, self.tbl_ml, self.lbl_desc_ml = self.make_table_tab(
            ["Metric", "Value"],
            "ML/Regression section reports RF model status, predicted class confidence, gesture label, force estimate, "
            "movement phase, and anomaly label.",
        )
        self.page_quality, self.tbl_quality, self.lbl_desc_quality = self.make_table_tab(
            ["CH", "SNR(dB)", "BaselineDrift", "ClipRatio", "MainsNoise", "ContactQuality"],
            "Signal quality indicators per channel. Higher SNR and lower drift/clip/mains generally indicate better "
            "electrode contact and cleaner signals.",
        )

        self.coord_page = QWidget()
        coord_layout = QVBoxLayout(self.coord_page)
        self.tbl_coord = self.make_table(["Metric", "Value"])
        coord_layout.addWidget(self.tbl_coord)
        self.lbl_desc_coord = self.make_description_label(
            "Coordination summary includes channel ratios, symmetry/co-contraction indices, lag values, and coherence values."
        )
        coord_layout.addWidget(self.lbl_desc_coord)

        corr_title = QLabel("Correlation Matrix (Synergy)")
        corr_title.setStyleSheet("font-weight: bold;")
        coord_layout.addWidget(corr_title)

        self.corr_table = QTableWidget(self.num_channels, self.num_channels)
        self.corr_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.corr_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.corr_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.corr_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.corr_table.horizontalHeader().setStyleSheet(
            f"QHeaderView {{ background-color: {THEME_COLORS['panel']}; }}"
            f"QHeaderView::section {{ background-color: {THEME_COLORS['panel']}; color: {THEME_COLORS['text']}; "
            f"border: 1px solid {THEME_COLORS['accent']}; padding: 6px; font-size: 16px; font-weight: 600; }}"
        )
        self.corr_table.setStyleSheet(
            "QTableWidget { font-size: 17px; "
            f"background: {THEME_COLORS['panel']}; color: {THEME_COLORS['text']}; border: 1px solid {THEME_COLORS['accent']}; }} "
            f"QTableCornerButton::section {{ background-color: {THEME_COLORS['panel']}; border: 1px solid {THEME_COLORS['accent']}; }} "
            f"QScrollBar:vertical {{ border: none; background: {THEME_COLORS['bg']}; width: 12px; margin: 0; }} "
            f"QScrollBar::handle:vertical {{ background: {THEME_COLORS['accent']}; min-height: 28px; border-radius: 6px; }} "
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; border: none; background: transparent; }} "
            f"QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: {THEME_COLORS['bg']}; }} "
            f"QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {{ background: transparent; width: 0px; height: 0px; }} "
            f"QScrollBar:horizontal {{ border: none; background: {THEME_COLORS['bg']}; height: 12px; margin: 0; }} "
            f"QScrollBar::handle:horizontal {{ background: {THEME_COLORS['accent']}; min-width: 28px; border-radius: 6px; }} "
            f"QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; border: none; background: transparent; }} "
            f"QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: {THEME_COLORS['bg']}; }} "
            f"QScrollBar::left-arrow:horizontal, QScrollBar::right-arrow:horizontal {{ background: transparent; width: 0px; height: 0px; }}"
        )
        self.corr_table.setVerticalHeaderLabels([f"CH{i + 1}" for i in range(self.num_channels)])
        self.corr_table.setHorizontalHeaderLabels([f"CH{i + 1}" for i in range(self.num_channels)])
        coord_layout.addWidget(self.corr_table)
        self.lbl_desc_corr = self.make_description_label(
            "Correlation matrix shows linear similarity between channel pairs (-1 to +1)."
        )
        coord_layout.addWidget(self.lbl_desc_corr)

        self.section_stack = QStackedWidget()
        self.section_stack.addWidget(self.page_overview)
        self.section_stack.addWidget(self.page_time)
        self.section_stack.addWidget(self.page_freq)
        self.section_stack.addWidget(self.page_tfr)
        self.section_stack.addWidget(self.coord_page)
        self.section_stack.addWidget(self.page_events)
        self.section_stack.addWidget(self.page_ml)
        self.section_stack.addWidget(self.page_quality)

        section_names = [
            "Overview",
            "Time",
            "Frequency",
            "Time-Frequency",
            "Coordination",
            "Events",
            "ML/Regression",
            "Quality",
        ]
        self.section_buttons = []
        nav_container = QWidget()
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(9, 0, 9, 0)
        nav_layout.setSpacing(8)

        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(8)
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(8)
        split_idx = (len(section_names) + 1) // 2

        for idx, name in enumerate(section_names):
            btn = QPushButton(name)
            btn.setObjectName("sectionNav")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _checked, i=idx: self.switch_section(i))
            self.section_buttons.append(btn)
            if idx < split_idx:
                row1.addWidget(btn, 1)
            else:
                row2.addWidget(btn, 1)

        nav_layout.addLayout(row1)
        nav_layout.addLayout(row2)
        layout.addWidget(nav_container)
        layout.addWidget(self.section_stack)
        self.switch_section(0)

        self.set_initial_table()

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

    def switch_section(self, index):
        if index < 0 or index >= self.section_stack.count():
            return
        self.section_stack.setCurrentIndex(index)
        for i, btn in enumerate(self.section_buttons):
            btn.setChecked(i == index)

    @staticmethod
    def make_description_label(text):
        lbl = QLabel(str(text))
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {THEME_COLORS['muted']}; font-size: 16px;")
        return lbl

    def _refresh_band_plot_ticks(self):
        ticks = [(i + 1, f"CH{i + 1}") for i in range(self.num_channels)]
        self.freq_band_plot.getAxis("bottom").setTicks([ticks])

    def _update_band_power_plot(self, band_power_pct):
        if band_power_pct is None:
            return
        arr = np.asarray(band_power_pct, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != self.num_channels:
            return

        n_bands = min(arr.shape[1], len(self.freq_band_offsets))
        # Rebuild bars only when the layout changes; otherwise update heights in place so
        # we don't churn a new BarGraphItem set on every analysis frame.
        if len(self.freq_band_items) != n_bands or self._freq_band_plot_nch != self.num_channels:
            for item in self.freq_band_items:
                try:
                    self.freq_band_plot.removeItem(item)
                except Exception:
                    pass
            self.freq_band_items.clear()
            x_base = np.arange(1, self.num_channels + 1, dtype=np.float32)
            for bi in range(n_bands):
                bar = pg.BarGraphItem(
                    x=x_base + self.freq_band_offsets[bi],
                    height=np.clip(arr[:, bi], 0.0, 100.0),
                    width=self.freq_band_width,
                    brush=pg.mkBrush(self.freq_band_colors[bi]),
                    pen=pg.mkPen(self.freq_band_colors[bi]),
                )
                self.freq_band_plot.addItem(bar)
                self.freq_band_items.append(bar)
            self._freq_band_plot_nch = self.num_channels
        else:
            for bi in range(n_bands):
                self.freq_band_items[bi].setOpts(height=np.clip(arr[:, bi], 0.0, 100.0))

    def make_table_tab(self, columns, description_text):
        page = QWidget()
        v = QVBoxLayout(page)
        table = self.make_table(columns)
        v.addWidget(table)
        desc = self.make_description_label(description_text)
        v.addWidget(desc)
        return page, table, desc

    @staticmethod
    def make_table(columns):
        table = QTableWidget(0, len(columns))
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(False)
        table.horizontalHeader().setStyleSheet(
            f"QHeaderView {{ background-color: {THEME_COLORS['panel']}; }}"
            f"QHeaderView::section {{ background-color: {THEME_COLORS['panel']}; color: {THEME_COLORS['text']}; "
            f"border: 1px solid {THEME_COLORS['accent']}; padding: 6px; font-size: 16px; font-weight: 600; }}"
        )
        for c in range(len(columns)):
            mode = QHeaderView.ResizeToContents
            table.horizontalHeader().setSectionResizeMode(c, mode)
        table.setHorizontalHeaderLabels(columns)
        table.setStyleSheet(
            "QTableWidget { font-size: 17px; "
            f"background: {THEME_COLORS['panel']}; color: {THEME_COLORS['text']}; border: 1px solid {THEME_COLORS['accent']}; }} "
            f"QTableCornerButton::section {{ background-color: {THEME_COLORS['panel']}; border: 1px solid {THEME_COLORS['accent']}; }} "
            f"QScrollBar:vertical {{ border: none; background: {THEME_COLORS['bg']}; width: 12px; margin: 0; }} "
            f"QScrollBar::handle:vertical {{ background: {THEME_COLORS['accent']}; min-height: 28px; border-radius: 6px; }} "
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; border: none; background: transparent; }} "
            f"QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: {THEME_COLORS['bg']}; }} "
            f"QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {{ background: transparent; width: 0px; height: 0px; }} "
            f"QScrollBar:horizontal {{ border: none; background: {THEME_COLORS['bg']}; height: 12px; margin: 0; }} "
            f"QScrollBar::handle:horizontal {{ background: {THEME_COLORS['accent']}; min-width: 28px; border-radius: 6px; }} "
            f"QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; border: none; background: transparent; }} "
            f"QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: {THEME_COLORS['bg']}; }} "
            f"QScrollBar::left-arrow:horizontal, QScrollBar::right-arrow:horizontal {{ background: transparent; width: 0px; height: 0px; }}"
        )
        return table

    @staticmethod
    def set_rows(table, rows, center_all=False):
        # Reuse existing cell items and only touch text that changed, instead of
        # allocating a fresh QTableWidgetItem for every cell on every (~10 Hz) refresh.
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                text = str(val)
                item = table.item(r, c)
                if item is None:
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(r, c, item)
                elif item.text() != text:
                    item.setText(text)

    def set_initial_table(self):
        self.set_rows(self.tbl_overview, [("Status", "Waiting")])
        self.set_rows(self.tbl_time, [("-", "-", "-", "-", "-", "-", "-", "-", "-")], center_all=True)
        self.set_rows(self.tbl_freq, [("-", "-", "-", "-", "-", "-", "-", "-", "-")], center_all=True)
        self.set_rows(self.tbl_tfr, [("-", "-", "-", "-", "-", "-", "-", "-", "-", "-")], center_all=True)
        self.set_rows(self.tbl_coord, [("Status", "Waiting")])
        self.set_rows(self.tbl_events, [("-", "-", "-", "-", "-", "-", "-")], center_all=True)
        self.set_rows(self.tbl_ml, [("Status", "Waiting")])
        self.set_rows(self.tbl_quality, [("-", "-", "-", "-", "-", "-")], center_all=True)

        for r in range(self.num_channels):
            for c in range(self.num_channels):
                item = QTableWidgetItem("0.00")
                item.setTextAlignment(Qt.AlignCenter)
                item.setBackground(pg.mkColor(THEME_COLORS["panel"]))
                item.setForeground(pg.mkColor(THEME_COLORS["text"]))
                self.corr_table.setItem(r, c, item)
        self._update_band_power_plot(np.zeros((self.num_channels, len(BAND_DEFS)), dtype=np.float32))

    def color_for_corr(self, value):
        v = float(np.clip(value, -1.0, 1.0))
        if v >= 0.0:
            r = int(22 + 37 * v)   # #16476A -> #3B9797
            g = int(71 + 80 * v)
            b = int(106 + 45 * v)
        else:
            a = abs(v)
            r = int(22 + 169 * a)  # #16476A -> #BF092F
            g = int(71 - 62 * a)
            b = int(106 - 59 * a)
        return pg.mkColor((r, g, b))

    def update_analysis_view(self, m):
        n_ch = self.num_channels

        states = ["ACTIVE" if bool(x) else "REST" for x in m["active_flags"]]
        overview_rows = [
            ("Gesture", m["gesture_label"]),
            ("Force Estimate", f"{m['force_level_pct']:.1f}%"),
            ("Movement Phase", m["movement_phase"]),
            ("Anomaly", m["anomaly_label"]),
            ("Mean RMS", f"{np.mean(m['rms']):.2f}"),
            ("Mean Median Freq", f"{np.mean(m['median_hz']):.2f} Hz"),
            ("Per-channel state", " | ".join([f"CH{i+1}:{states[i]}" for i in range(n_ch)])),
        ]
        self.set_rows(self.tbl_overview, overview_rows)

        time_rows = []
        for i in range(n_ch):
            time_rows.append(
                (
                    f"CH{i+1}",
                    f"{m['mav'][i]:.1f}",
                    f"{m['rms'][i]:.1f}",
                    f"{m['iemg'][i]:.1f}",
                    f"{m['var'][i]:.1f}",
                    f"{m['wl'][i]:.1f}",
                    f"{int(m['zc'][i])}",
                    f"{int(m['ssc'][i])}",
                    f"{int(m['wamp'][i])}",
                )
            )
        self.set_rows(self.tbl_time, time_rows, center_all=True)

        freq_rows = []
        for i in range(n_ch):
            b = m["band_power_pct"][i]
            freq_rows.append(
                (
                    f"CH{i+1}",
                    f"{m['mean_hz'][i]:.1f}",
                    f"{m['median_hz'][i]:.1f}",
                    f"{m['peak_hz'][i]:.1f}",
                    f"{m['spec_entropy'][i]:.3f}",
                    f"{b[0]:.1f}%",
                    f"{b[1]:.1f}%",
                    f"{b[2]:.1f}%",
                    f"{m['mains_noise_score'][i]:.1f}%",
                )
            )
        self.set_rows(self.tbl_freq, freq_rows, center_all=True)
        self.lbl_desc_freq.setText(
            "Frequency-domain features per channel. Band columns are percentage power in each band; MainsScore estimates "
            f"50/60 Hz contamination. Fatigue trend slope: {m['fatigue_slope_hz_per_s']:+.3f} Hz/s."
        )
        self._update_band_power_plot(m["band_power_pct"])

        tfr_rows = []
        for i in range(n_ch):
            d = m["short_time_band_delta"][i]
            if m["wavelet_available"]:
                w = m["wavelet_energy_pct"][i]
                wavelet_vals = (f"{w[0]:.1f}%", f"{w[1]:.1f}%", f"{w[2]:.1f}%", f"{w[3]:.1f}%")
            else:
                wavelet_vals = ("N/A", "N/A", "N/A", "N/A")
            tfr_rows.append(
                (
                    f"CH{i+1}",
                    f"{m['stft_dom_mean_hz'][i]:.1f}",
                    f"{m['stft_dom_std_hz'][i]:.1f}",
                    f"{d[0]:+.1f}%",
                    f"{d[1]:+.1f}%",
                    f"{d[2]:+.1f}%",
                    wavelet_vals[0],
                    wavelet_vals[1],
                    wavelet_vals[2],
                    wavelet_vals[3],
                )
            )
        self.set_rows(self.tbl_tfr, tfr_rows, center_all=True)

        coord_rows = [("Channel Ratios", ", ".join([f"CH{i+1}:{m['channel_ratio'][i]:.2f}" for i in range(n_ch)]))]
        if m["symmetry_index"]:
            for k, v in m["symmetry_index"].items():
                coord_rows.append((f"Symmetry {k}", f"{v:.3f}"))
        if m["co_contraction_index"]:
            for k, v in m["co_contraction_index"].items():
                coord_rows.append((f"Co-contraction {k}", f"{v:.3f}"))
        lag = m["lag_ms_matrix"]
        coh = m["coherence_matrix"]
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                coord_rows.append((f"Lag CH{i+1}-CH{j+1}", f"{lag[i, j]:.1f} ms"))
                coord_rows.append((f"Coherence CH{i+1}-CH{j+1}", f"{coh[i, j]:.3f}"))
        self.set_rows(self.tbl_coord, coord_rows)

        event_rows = []
        for i in range(n_ch):
            event_rows.append(
                (
                    f"CH{i+1}",
                    f"{int(m['onset_count'][i])}",
                    f"{int(m['offset_count'][i])}",
                    f"{int(m['repetition_count'][i])}",
                    f"{m['burst_mean_ms'][i]:.1f}",
                    f"{m['burst_current_ms'][i]:.1f}",
                    f"{m['contraction_quality_score'][i]:.1f}",
                )
            )
        self.set_rows(self.tbl_events, event_rows, center_all=True)

        ml_rows = [
            ("RF Model Loaded", str(m["rf_model_loaded"])),
            ("RF Model Path", m["rf_model_path"]),
            ("RF Prediction", f"{m['rf_pred_label']} ({m['rf_pred_conf_pct']:.1f}%)"),
            ("Gesture/Intention", m["gesture_label"]),
            ("Force Level", f"{m['force_level_pct']:.1f}%"),
            ("Movement Phase", m["movement_phase"]),
            ("Anomaly", m["anomaly_label"]),
        ]
        self.set_rows(self.tbl_ml, ml_rows)

        quality_rows = []
        for i in range(n_ch):
            quality_rows.append(
                (
                    f"CH{i+1}",
                    f"{m['snr_db'][i]:.2f}",
                    f"{m['baseline_drift'][i]:.2f}",
                    f"{m['clip_ratio_pct'][i]:.2f}%",
                    f"{m['mains_noise_score'][i]:.2f}%",
                    m["contact_quality"][i],
                )
            )
        self.set_rows(self.tbl_quality, quality_rows, center_all=True)

        corr_matrix = m["corr_matrix"]
        rows, cols = corr_matrix.shape
        for r in range(min(rows, self.num_channels)):
            for c in range(min(cols, self.num_channels)):
                value = float(corr_matrix[r, c])
                item = self.corr_table.item(r, c)
                if item is None:
                    item = QTableWidgetItem()
                    item.setTextAlignment(Qt.AlignCenter)
                    self.corr_table.setItem(r, c, item)
                item.setText(f"{value:+.2f}")
                item.setBackground(self.color_for_corr(value))
                item.setForeground(pg.mkColor("#E8EEF0"))


class RealtimeClassificationWindow(QMainWindow):
    closed = pyqtSignal()

    def __init__(self, visualizer):
        super().__init__()
        apply_app_icon(self)
        self.visualizer = visualizer
        self.setWindowTitle("Real-time Classification")
        fit_window_to_screen(self, 920, 680)
        apply_dark_title_bar(self)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)

        self.lbl_model_status = QLabel("Model: Not loaded")
        self.lbl_model_status.setStyleSheet(themed_label_style("muted"))
        layout.addWidget(self.lbl_model_status)

        self.meta_box = QPlainTextEdit()
        self.meta_box.setReadOnly(True)
        self.meta_box.setMaximumHeight(130)
        self.meta_box.setStyleSheet(
            "QPlainTextEdit { font-family: Consolas, 'Courier New', monospace; font-size: 13px; }"
        )
        layout.addWidget(self.meta_box)

        self.lbl_pred = QLabel("Classification: N/A")
        self.lbl_pred.setStyleSheet("font-size: 22px; font-weight: bold; color: #E8EEF0;")
        layout.addWidget(self.lbl_pred)

        self.lbl_conf = QLabel("Confidence: 0.0%")
        self.lbl_conf.setStyleSheet("font-size: 18px; font-weight: 600; color: #A9C2CF;")
        layout.addWidget(self.lbl_conf)

        self.lbl_latency = QLabel("Classification Latency: N/A")
        self.lbl_latency.setStyleSheet("font-size: 17px; font-weight: 600; color: #A9C2CF;")
        layout.addWidget(self.lbl_latency)

        self.lbl_rate = QLabel("Classification Rate: 0.0/sec")
        self.lbl_rate.setStyleSheet("font-size: 17px; font-weight: 600; color: #A9C2CF;")
        layout.addWidget(self.lbl_rate)

        self.bar_plot = pg.PlotWidget()
        self.bar_plot.setBackground(THEME_COLORS["bg"])
        self.bar_plot.showGrid(x=True, y=True, alpha=0.2)
        self.bar_plot.setMouseEnabled(x=False, y=False)
        self.bar_plot.setMenuEnabled(False)
        self.bar_plot.hideButtons()
        self.bar_plot.setYRange(0.0, 100.0, padding=0.02)
        self.bar_plot.setLabel("left", "Confidence (%)")
        self.bar_plot.setLabel("bottom", "Class")
        self.bar_plot.getAxis("left").setPen(pg.mkPen(THEME_COLORS["text"]))
        self.bar_plot.getAxis("bottom").setPen(pg.mkPen(THEME_COLORS["text"]))
        self.bar_plot.getAxis("left").setTextPen(pg.mkPen(THEME_COLORS["text"]))
        self.bar_plot.getAxis("bottom").setTextPen(pg.mkPen(THEME_COLORS["text"]))
        self.bar_plot.getViewBox().setMouseEnabled(x=False, y=False)
        self.bar_plot.getViewBox().setMenuEnabled(False)
        layout.addWidget(self.bar_plot, 1)

        self._bar_item = None
        self._last_classes = []
        self._view_text_cache = {}
        self._model_status_style_loaded = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_from_visualizer)
        self.timer.start(200)
        self.refresh_from_visualizer()

    def _set_text_cached(self, widget, key, text):
        if self._view_text_cache.get(key) != text:
            widget.setText(text)
            self._view_text_cache[key] = text

    def refresh_from_visualizer(self):
        if self.visualizer is None:
            return
        payload = self.visualizer.get_realtime_classification_payload()
        self.update_view(payload)

    def update_view(self, payload):
        p = dict(payload or {})
        model_loaded = bool(p.get("model_loaded", False))
        status_text = str(p.get("model_status_text", "Model: Not loaded"))
        self._set_text_cached(self.lbl_model_status, "model_status", status_text)
        if self._model_status_style_loaded != model_loaded:
            self.lbl_model_status.setStyleSheet(
                themed_label_style("success" if model_loaded else "muted")
            )
            self._model_status_style_loaded = model_loaded

        meta_text = str(p.get("meta_text", "")).strip()
        if self._view_text_cache.get("meta") != meta_text:
            self.meta_box.setPlainText(meta_text)
            self._view_text_cache["meta"] = meta_text

        pred_label = str(p.get("pred_label", "N/A"))
        conf_pct = float(p.get("pred_conf_pct", 0.0))
        latency_ms = float(p.get("classification_latency_ms", 0.0))
        rate_hz = float(p.get("classification_rate_hz", 0.0))
        self._set_text_cached(self.lbl_pred, "pred", f"Classification: {pred_label}")
        self._set_text_cached(self.lbl_conf, "conf", f"Confidence: {conf_pct:.1f}%")
        if model_loaded and pred_label != "N/A" and latency_ms > 0.0:
            self._set_text_cached(self.lbl_latency, "latency", f"Classification Latency: {latency_ms:.2f} ms")
        else:
            self._set_text_cached(self.lbl_latency, "latency", "Classification Latency: N/A")
        if model_loaded and pred_label != "N/A":
            self._set_text_cached(self.lbl_rate, "rate", f"Classification Rate: {max(0.0, rate_hz):.1f}/sec")
        else:
            self._set_text_cached(self.lbl_rate, "rate", "Classification Rate: 0.0/sec")

        classes = [str(x) for x in list(p.get("classes", []))]
        scores = np.asarray(list(p.get("class_confidences_pct", [])), dtype=np.float32)
        if len(classes) == 0:
            classes = ["N/A"]
            scores = np.asarray([0.0], dtype=np.float32)
        if scores.size != len(classes):
            scores = np.zeros(len(classes), dtype=np.float32)

        x = np.arange(1, len(classes) + 1, dtype=np.float32)
        heights = np.clip(scores, 0.0, 100.0)
        # Rebuild the bar item only when the class set changes; otherwise update heights
        # in place so we don't churn a new BarGraphItem every refresh.
        if self._bar_item is None or classes != self._last_classes:
            if self._bar_item is not None:
                try:
                    self.bar_plot.removeItem(self._bar_item)
                except Exception:
                    pass
            self._bar_item = pg.BarGraphItem(
                x=x,
                height=heights,
                width=0.62,
                brush=pg.mkBrush(THEME_COLORS["accent"]),
                pen=pg.mkPen(THEME_COLORS["accent"]),
            )
            self.bar_plot.addItem(self._bar_item)
            ticks = [(i + 1, classes[i]) for i in range(len(classes))]
            self.bar_plot.getAxis("bottom").setTicks([ticks])
            self.bar_plot.setXRange(0.4, len(classes) + 0.6, padding=0.0)
            self._last_classes = list(classes)
        else:
            self._bar_item.setOpts(height=heights)

    def closeEvent(self, event):
        self.timer.stop()
        self.closed.emit()
        super().closeEvent(event)


class RoundedClipWidget(QWidget):
    def __init__(self, radius=10, parent=None):
        super().__init__(parent)
        self.radius = int(max(1, radius))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.width() <= 2 or self.height() <= 2:
            return
        r = self.rect().adjusted(0, 0, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(float(r.x()), float(r.y()), float(r.width()), float(r.height()), self.radius, self.radius)
        poly = path.toFillPolygon().toPolygon()
        self.setMask(QRegion(poly))


class LiveAnalysisWorker(QThread):
    """Runs heavy spectral/time-frequency/correlation analysis off the UI thread."""
    analysis_ready = pyqtSignal(dict)
    analysis_finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = threading.Lock()
        self._pending = threading.Event()
        self._running = True
        self._segment = None
        self._params = {}
        # Cache of the expensive pairwise matrices so cheap frames can carry them forward.
        self._heavy_cache = None

    def submit(self, segment, params):
        with self._lock:
            self._segment = segment
            self._params = params
        self._pending.set()

    def stop(self):
        self._running = False
        self._pending.set()
        self.wait()

    def run(self):
        while self._running:
            self._pending.wait(0.5)
            self._pending.clear()
            if not self._running:
                break
            with self._lock:
                segment = self._segment
                params = dict(self._params)
            if segment is None:
                continue
            try:
                include_heavy = bool(params.get("include_heavy", True))
                centered = segment - np.mean(segment, axis=1, keepdims=True)
                time_metrics = EMGVisualizer.compute_time_domain_features(centered)
                spectral = EMGVisualizer.compute_spectral_features(centered)
                # Time-frequency (STFT/wavelet) and correlation/lag/coherence are the most
                # expensive blocks; refresh them at a lower cadence and reuse the last result
                # on the cheap frames in between. Recompute if the cache is stale or its
                # channel count no longer matches the stream (e.g. after a reconnect).
                n_ch_now = centered.shape[0]
                cache_valid = (
                    self._heavy_cache is not None
                    and np.asarray(self._heavy_cache[1]).shape[0] == n_ch_now
                )
                if include_heavy or not cache_valid:
                    timefreq = EMGVisualizer.compute_time_frequency_features(centered)
                    corr_matrix = EMGVisualizer.compute_corr_matrix(centered)
                    lag_ms_matrix = EMGVisualizer.compute_lag_ms_matrix(centered)
                    coherence_matrix = EMGVisualizer.compute_coherence_matrix(centered)
                    self._heavy_cache = (timefreq, corr_matrix, lag_ms_matrix, coherence_matrix)
                else:
                    timefreq, corr_matrix, lag_ms_matrix, coherence_matrix = self._heavy_cache
                coord_summary = EMGVisualizer.compute_coordination_indices(centered, time_metrics["rms"])
                self.analysis_ready.emit({
                    "time_metrics": time_metrics,
                    "spectral": spectral,
                    "timefreq": timefreq,
                    "corr_matrix": corr_matrix,
                    "lag_ms_matrix": lag_ms_matrix,
                    "coherence_matrix": coherence_matrix,
                    "coord_summary": coord_summary,
                    "params": params,
                })
            except Exception:
                pass
            finally:
                self.analysis_finished.emit()


class GesturePublisher:
    """Mirrors each smoothed prediction to localhost UDP for the separate gesture_game_app.py.

    One ~60-byte datagram at <= 12 Hz; send errors are ignored so it can never slow the
    EMG pipeline down or raise into the UI.
    """

    def __init__(self, host=GESTURE_BRIDGE_HOST, port=GESTURE_BRIDGE_PORT):
        self._addr = (host, int(port))
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setblocking(False)
        except OSError:
            self._sock = None

    def publish(self, label, conf_pct, model=""):
        if self._sock is None:
            return
        try:
            msg = json.dumps({"label": str(label), "conf_pct": round(float(conf_pct), 1), "model": str(model)})
            self._sock.sendto(msg.encode("utf-8"), self._addr)
        except (OSError, ValueError):
            pass

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


def packet_health_kind(snapshot):
    """'ok' / 'warn' / 'bad' for a PacketTracer snapshot (drives label colours)."""
    if snapshot["received"] == 0:
        return "idle"
    if snapshot["loss_pct"] >= 1.0:
        return "bad"
    if snapshot["missing"] > 0 or snapshot["invalid"] > 0:
        return "warn"
    return "ok"


PACKET_KIND_COLORS = {"ok": "#43A047", "warn": "#f57c00", "bad": "#f44336", "idle": THEME_COLORS["muted"]}


class PacketTracerWindow(QMainWindow):
    """Live transport monitor: packets/s, missing packets, loss %, jitter and a rolling plot."""

    closed = pyqtSignal()

    def __init__(self, tracer, parent=None):
        super().__init__(parent)
        apply_app_icon(self)
        self.tracer = tracer
        self.setWindowTitle("Packet Tracer")
        fit_window_to_screen(self, 760, 560)
        clamp_minimum_size(self, 520, 420)
        apply_dark_title_bar(self)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)

        self.lbl_health = StableLabel("No packets yet")
        self.lbl_health.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(self.lbl_health)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        self._value_labels = {}
        fields = [
            ("pkt_rate", "Packets / s"), ("sample_rate", "Samples / s"),
            ("received", "Packets received"), ("missing", "Packets missing"),
            ("loss", "Packet loss"), ("missing_samples", "Samples lost"),
            ("out_of_order", "Out of order"), ("invalid", "Invalid / malformed"),
            ("jitter", "Jitter"), ("max_gap", "Longest gap"),
            ("throughput", "Throughput"), ("uptime", "Uptime"),
        ]
        for i, (key, title) in enumerate(fields):
            card = QFrame()
            card.setObjectName("pktCard")
            card.setStyleSheet(
                f"QFrame#pktCard {{ background-color: {THEME_COLORS['panel']}; "
                f"border: 1px solid {THEME_COLORS['accent']}; border-radius: 8px; }} "
                "QFrame#pktCard QLabel { background: transparent; border: none; }"
            )
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 6, 10, 6)
            card_layout.setSpacing(0)
            cap = QLabel(title)
            cap.setStyleSheet(themed_label_style("muted"))
            val = StableLabel("--")
            val.setStyleSheet("font-size: 20px; font-weight: bold;")
            card_layout.addWidget(cap)
            card_layout.addWidget(val)
            self._value_labels[key] = val
            grid.addWidget(card, i // 3, i % 3)
        layout.addLayout(grid)

        self.lbl_note = QLabel("")
        self.lbl_note.setWordWrap(True)
        self.lbl_note.setStyleSheet(themed_label_style("muted"))
        layout.addWidget(self.lbl_note)

        self.rate_plot = pg.PlotWidget()
        self.rate_plot.setBackground("k")
        self.rate_plot.setMinimumHeight(140)
        self.rate_plot.showGrid(x=False, y=True, alpha=0.3)
        self.rate_plot.setLabel("left", "packets / s")
        self.rate_plot.setLabel("bottom", "last 30 s")
        self.rate_plot.setMouseEnabled(x=False, y=False)
        self.rate_plot.hideButtons()
        self.rate_plot.setMenuEnabled(False)
        self.rate_curve = self.rate_plot.plot(pen=pg.mkPen("#3B9797", width=2))
        self.missing_plot = pg.ViewBox()
        self.rate_plot.showAxis("right")
        self.rate_plot.getAxis("right").setLabel("missing")
        self.rate_plot.scene().addItem(self.missing_plot)
        self.rate_plot.getAxis("right").linkToView(self.missing_plot)
        self.missing_plot.setXLink(self.rate_plot)
        self.missing_curve = pg.PlotCurveItem(pen=pg.mkPen("#f44336", width=2))
        self.missing_plot.addItem(self.missing_curve)
        self.rate_plot.getViewBox().sigResized.connect(self._sync_views)
        layout.addWidget(self.rate_plot, 1)

        btn_row = QHBoxLayout()
        self.btn_reset = QPushButton("Reset Counters")
        self.btn_reset.setStyleSheet(themed_button_style("muted"))
        self.btn_reset.setToolTip("Zero the counters (the stream keeps running).")
        self.btn_reset.clicked.connect(self._reset_counters)
        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.refresh()

    def _sync_views(self):
        self.missing_plot.setGeometry(self.rate_plot.getViewBox().sceneBoundingRect())
        self.missing_plot.linkedViewChanged(self.rate_plot.getViewBox(), self.missing_plot.XAxis)

    def _reset_counters(self):
        self.tracer.reset(self.tracer.kind, self.tracer.frames_per_packet)

    def refresh(self):
        snap = self.tracer.snapshot()
        kind = packet_health_kind(snap)
        color = PACKET_KIND_COLORS[kind]
        wireless = snap["kind"] == "wireless"
        headline = {
            "idle": "Waiting for packets...",
            "ok": "Link healthy - no packets missing",
            "warn": "Link degraded - packets missing",
            "bad": "Link poor - significant packet loss",
        }[kind]
        self.lbl_health.setText(headline)
        self.lbl_health.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {color};")

        vals = self._value_labels
        vals["pkt_rate"].setText(f"{snap['pkt_rate']:.1f}  / {snap['expected_pkt_rate']:.0f}")
        vals["sample_rate"].setText(f"{snap['sample_rate']:.0f}  / {snap['expected_sample_rate']:.0f}")
        vals["received"].setText(f"{snap['received']:,}")
        vals["missing"].setText(f"{snap['missing']:,}" if wireless else f"{snap['missing']:,} (bad lines)")
        vals["loss"].setText(f"{snap['loss_pct']:.3f} %")
        vals["missing_samples"].setText(f"{snap['missing_samples']:,}")
        vals["out_of_order"].setText(f"{snap['out_of_order']:,}" if wireless else "n/a")
        vals["invalid"].setText(f"{snap['invalid']:,}")
        vals["jitter"].setText(f"{snap['jitter_ms']:.2f} ms" if wireless else "n/a")
        vals["max_gap"].setText(f"{snap['max_gap_ms']:.0f} ms")
        vals["throughput"].setText(f"{snap['bytes'] / max(1e-6, snap['uptime_s']) / 1024.0:.1f} KB/s")
        vals["uptime"].setText(format_duration(snap["uptime_s"] * 1000, False))
        for key in ("missing", "loss", "missing_samples"):
            vals[key].setStyleSheet(
                f"font-size: 20px; font-weight: bold; color: "
                f"{PACKET_KIND_COLORS['ok'] if snap['missing'] == 0 else PACKET_KIND_COLORS['bad']};"
            )
        self.lbl_note.setText(
            "Wireless: every UDP packet carries a sequence number, so missing and out-of-order packets are exact "
            f"(each packet holds {snap['missing_samples'] // max(1, snap['missing']) if snap['missing'] else WIRELESS_FRAMES_PER_PACKET} samples)."
            if wireless else
            "Wired: the text stream has no sequence numbers, so malformed/truncated lines are counted as lost "
            "samples and the achieved sample rate is compared with the nominal rate."
        )
        rates = snap["rate_history"]
        if rates:
            xs = np.arange(-len(rates) + 1, 1) * (PACKET_TRACER_TICK_MS / 1000.0)
            self.rate_curve.setData(xs, np.asarray(rates, dtype=np.float32), skipFiniteCheck=True)
            self.missing_curve.setData(xs, np.asarray(snap["missing_history"], dtype=np.float32), skipFiniteCheck=True)
            self.rate_plot.setXRange(float(xs[0]), 0.0, padding=0)
            self.rate_plot.setYRange(0.0, max(snap["expected_pkt_rate"] * 1.2, max(rates) * 1.05), padding=0)
            self.missing_plot.setYRange(0.0, max(1.0, float(max(snap["missing_history"])) * 1.2), padding=0)
        self._sync_views()

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


class OptionsInfoDialog(QDialog):
    """One window listing every option of the app: current value and what it does."""

    def __init__(self, sections, parent=None):
        super().__init__(parent)
        apply_app_icon(self)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowTitle("Options Info")
        self.resize(880, 720)
        apply_dark_title_bar(self)
        layout = QVBoxLayout(self)
        intro = QLabel("Every option in BioWave EMG, grouped by where you find it. "
                       "'Current value' shows your live setting.")
        intro.setWordWrap(True)
        intro.setStyleSheet(themed_label_style("muted"))
        layout.addWidget(intro)
        view = QTextBrowser()
        view.setOpenExternalLinks(False)
        view.setHtml(self._build_html(sections))
        layout.addWidget(view, 1)
        row = QHBoxLayout()
        row.addStretch()
        btn = QPushButton("Close")
        btn.setStyleSheet(themed_button_style("accent"))
        btn.clicked.connect(self.accept)
        row.addWidget(btn)
        layout.addLayout(row)

    @staticmethod
    def _build_html(sections):
        c = THEME_COLORS
        parts = [
            f"<style>h2 {{ color: {c['accent']}; margin-top: 18px; }} "
            f"table {{ border-collapse: collapse; width: 100%; }} "
            f"th {{ text-align: left; color: {c['muted']}; padding: 4px 8px; }} "
            f"td {{ padding: 5px 8px; vertical-align: top; border-top: 1px solid {c['panel']}; }} "
            f"td.v {{ color: {c['accent']}; font-weight: bold; white-space: nowrap; }}</style>"
        ]
        for title, rows in sections:
            parts.append(f"<h2>{html_escape(title)}</h2><table><tr><th>Option</th><th>Current value</th>"
                         "<th>What it does</th></tr>")
            for name, value, desc in rows:
                parts.append(
                    f"<tr><td><b>{html_escape(name)}</b></td><td class='v'>{html_escape(value)}</td>"
                    f"<td>{html_escape(desc)}</td></tr>"
                )
            parts.append("</table>")
        return "".join(parts)


class EMGVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        apply_app_icon(self)

        self.setWindowTitle(APP_NAME)
        # 3:2 window scaled to the screen it opens on (Mac and Windows differ in usable
        # height, DPI and aspect), never larger than the desktop and never clipped.
        fit_window_to_screen(self, 1380, 920)
        clamp_minimum_size(self, 900, 600)
        apply_dark_title_bar(self)

        # State variables
        self.serial_worker = None
        self.is_connected = False
        self.is_calibrated = False
        self.calibration_active = False
        self.data_buffer = None
        self.raw_data_buffer = None
        self._buffer_lock = threading.RLock()
        self.baseline_offsets = np.zeros(4, dtype=np.float32)

        self.curves = []
        self.plots = []
        self.plot_rows = []
        self.threshold_lines_pos = []
        self.threshold_lines_neg = []
        self.zero_lines = []
        self.metrics_labels = []

        self.latest_mav = None
        self.latest_rms = None
        self.latest_dom_hz = None
        self.latest_mean_hz = None
        self.latest_median_hz = None
        self.latest_spec_entropy = None
        self.latest_band_power_pct = None
        self.latest_stft_dom_mean_hz = None
        self.latest_stft_dom_std_hz = None
        self.latest_short_time_band_delta = None
        self.latest_wavelet_energy_pct = None
        self.wavelet_available = HAS_PYWT
        self.latest_corr_matrix = None
        self.latest_lag_ms_matrix = None
        self.latest_coherence_matrix = None
        self.latest_channel_ratio = None
        self.latest_symmetry_index = {}
        self.latest_co_contraction_index = {}
        self.latest_mains_noise_score = None
        self.latest_active = None
        self.latest_hz_active = None
        self.onset_state = None
        self.onset_count = None
        self.offset_count = None
        self.current_burst_ms = None
        self.burst_history_ms = None
        self.repetition_count = None
        self.latest_quality_score = None
        self.rest_rms_ref = None
        self.flex_rms_ref = None
        self.baseline_reference = None
        self.last_clip_ratio = None
        self.fatigue_track = []
        self.force_level_pct = 0.0
        self.prev_force_level_pct = 0.0
        self.movement_phase = "REST"
        self.gesture_label = "REST"
        self.anomaly_label = "NORMAL"
        self.contact_quality = []
        self.analysis_window = None
        self.rf_model = None
        self.rf_class_names = []
        self.rf_window_samples = 100
        self.rf_stride_samples = 0
        self.rf_model_created_at_text = "N/A"
        self.rf_model_sample_rate = int(SAMPLE_RATE)
        self.rf_model_input_channels = 4
        self.rf_model_path = ""
        self.rf_last_pred_label = "N/A"
        self.rf_last_pred_conf = 0.0
        self.rf_last_class_confidences = np.zeros(0, dtype=np.float32)
        self.rf_last_latency_ms = 0.0
        self.rf_prediction_rate_hz = 0.0
        self.rf_last_prediction_ts = 0.0
        self.rf_samples_since_submit = 0
        self.rf_valid_sample_count = 0
        self.rf_last_status_reason = "Model not loaded"
        # Cadence cap + in-flight guard so inference never falls behind the stream.
        self.rf_last_submit_ts = 0.0
        self.rf_inference_in_flight = False
        self.last_transport_metadata = {}
        # Short label history for majority-vote smoothing of the displayed prediction.
        self.rf_label_history = deque(maxlen=RF_LABEL_SMOOTH_WINDOW)
        self.rf_worker = RFRealtimeInferenceWorker(sample_rate=SAMPLE_RATE)
        self.rf_worker.prediction_ready.connect(self.on_rf_prediction_ready)
        self.rf_worker.error_occurred.connect(self.on_rf_worker_error)
        self.rf_worker.start(QThread.LowPriority)   # Never compete with packet reception
        self._analysis_worker = LiveAnalysisWorker()
        self._analysis_worker.analysis_ready.connect(self._on_analysis_ready)
        self._analysis_worker.analysis_finished.connect(self._on_analysis_finished)
        self._analysis_worker.start(QThread.LowPriority)
        self.live_analysis_enabled = False
        self.analysis_idle_labels_active = False
        self._analysis_pending = False
        self._last_analysis_submit_ts = 0.0
        self._last_heavy_analysis_submit_ts = 0.0
        self.realtime_classification_window = None

        self.data_collection_dialog = None
        self.training_dialog = None
        self.participant_id = ""
        self.participant_blood_group = ""
        self.participant_height_cm = 0.0
        self.participant_weight_kg = 0.0
        self.participant_age_years = 0
        self.participant_forearm_circumference_cm = 0.0
        # Session-plan and room-condition settings coming from the Data Collection dialog.
        self.study_meta = {
            "session_day": DEFAULT_SESSION_DAY,
            "hand_order": DEFAULT_HAND_ORDER,
            "max_cycles_per_block": DEFAULT_MAX_CYCLES_PER_BLOCK,
            "block_break_s": float(DEFAULT_BLOCK_BREAK_S),
            "hand_switch_s": float(DEFAULT_HAND_SWITCH_BREAK_S),
            **{key: 0.0 for key in STUDY_ENV_KEYS},
            "seed": 0, "practice_mode": False, "hand_order_source": "auto", "hand_order_default": "",
        }
        self.record_missing_packets = 0
        self.record_gap_events = 0
        self._session_csv_path = ""
        # Continuous session recording (see session_recording.py)
        self.session_recorder = None
        self.session_events = None
        self._session_name = ""
        self._session_dir = ""
        self._session_total_ms = 0
        self._session_host_origin = 0.0
        self._session_start_utc = ""
        self._cycle_event_start = 0
        self._last_session_rows = 0
        self._last_session_quality = None
        self.contribution_agreed = False
        self.task_labels = [x.strip() for x in DEFAULT_TASK_LABELS.split(",") if x.strip()]
        self.task_labels_locked = False
        self.task_labels_text = DEFAULT_TASK_LABELS
        self.task_repeats = int(DEFAULT_TASK_REPEATS)
        self.task_prep_s = float(DEFAULT_TASK_PREP_S)
        self.task_hold_s = float(DEFAULT_TASK_HOLD_S)
        self.task_rest_s = float(DEFAULT_TASK_REST_S)
        self.task_record_rest = True
        self.task_settle_s = float(DEFAULT_TASK_SETTLE_S)
        self.task_randomize_order = True
        self.task_session_active = False
        self.record_save_dir = DATASET_DIR
        self.last_recorded_csv_path = os.path.join(DATASET_DIR, DEFAULT_RECORD_CSV)
        self._record_graph_stream_prev_checked = True
        self._record_live_analysis_prev_enabled = False
        self.record_save_workers = []
        self.connection_medium = "wired"
        self.wired_binary = False
        self.wired_channel_count = 4
        self.current_port_name = "socket://127.0.0.1:7000"
        self.current_device = None
        self.discovered_wireless_devices = []
        self.wireless_access_key = load_device_access_key()
        self.keepalive_failures = 0
        self.keepalive_restarts = []      # every automatic START re-issue, for the quality report
        self._stream_client_ip = ""
        self.port_config_dialog = None
        self.wireless_config_dialog = None
        self.port_config_applied = False
        self.ui_fps = 0.0
        self.data_fps = 0.0
        self._ui_fps_frame_count = 0
        self._data_fps_sample_count = 0
        self._ui_fps_last_ts = time.perf_counter()
        self._data_fps_last_ts = time.perf_counter()
        self._last_plot_range_refresh_ts = 0.0
        self._plot_ranges_dirty = True
        self._centered_once = False

        # Caches so per-frame UI updates skip redundant (expensive) style re-parses.
        self._metrics_color_cache = {}
        self._threshold_line_color_cache = {}
        self._threshold_line_value_cache = None

        # Transport health (packet rate / loss) shared with the receive thread.
        self.packet_tracer = PacketTracer()
        self.packet_tracer_window = None
        self.options_info_dialog = None
        self._plot_dirty = False
        self._keepalive_thread = None
        self.gesture_bridge = GesturePublisher()

        self.num_channels = 4
        self.x_axis = (np.arange(WINDOW_SIZE, dtype=np.float32) - (WINDOW_SIZE - 1)) / SAMPLE_RATE

        # Calibration workflow state
        self.calibration_dialog = None
        self.calibration_timer = QTimer(self)
        self.calibration_timer.timeout.connect(self.on_calibration_tick)
        self.cal_rest_seconds = int(max(CAL_DURATION_MIN_S, min(CAL_DURATION_MAX_S, CAL_REST_MS // 1000)))
        self.cal_flex_seconds = int(max(CAL_DURATION_MIN_S, min(CAL_DURATION_MAX_S, CAL_FLEX_MS // 1000)))
        self.calibration_phases = [
            {
                "key": "rest",
                "name": "REST",
                "duration_ms": self.cal_rest_seconds * 1000,
                "instruction": "Keep your arm fully relaxed. Do not move.",
            },
            {
                "key": "flex",
                "name": "FLEX",
                "duration_ms": self.cal_flex_seconds * 1000,
                "instruction": "Flex the target muscle steadily until this phase ends.",
            },
        ]
        self.current_cal_phase_idx = -1
        self.current_phase_key = ""
        self.current_phase_remaining_ms = 0
        self.current_phase_total_ms = 0
        self.rest_capture = []
        self.flex_capture = []

        # Main UI Layout. Wrapped in a resizable QScrollArea so a smaller-than-
        # ideal display (e.g. a 13" MacBook Air's 1280x800 logical desktop)
        # scrolls instead of clipping controls.
        central_widget = QWidget()
        wrap_in_scroll_area(self, central_widget)
        self.main_layout = QVBoxLayout(central_widget)
        self.main_layout.setSpacing(4)

        # --- TOP CONTROLS BAR ---
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(12)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_port_config = QPushButton("Port Configuration")
        self.btn_port_config.setStyleSheet(themed_button_style("accent"))
        self.btn_port_config.clicked.connect(self.open_port_configuration_dialog)
        controls_layout.addWidget(self.btn_port_config)
        controls_layout.addStretch(1)

        self.btn_disconnect = QPushButton("Disconnect")
        self.btn_disconnect.setEnabled(False)
        self.btn_disconnect.setStyleSheet(self.disconnect_button_stylesheet())
        self.btn_disconnect.clicked.connect(self.disconnect_serial)
        controls_layout.addWidget(self.btn_disconnect)
        controls_layout.addStretch(1)

        self.btn_calibrate = QPushButton("Channel Calibration")
        self.btn_calibrate.setStyleSheet(themed_button_style("accent"))
        self.btn_calibrate.clicked.connect(self.open_calibration_dialog)
        controls_layout.addWidget(self.btn_calibrate)
        controls_layout.addStretch(1)

        self.btn_analysis = QPushButton("Live Analysis")
        self.btn_analysis.setEnabled(False)
        self.btn_analysis.setStyleSheet(themed_button_style("accent"))
        self.btn_analysis.clicked.connect(self.open_analysis_window)
        controls_layout.addWidget(self.btn_analysis)

        self.btn_packet_tracer = QPushButton("Packet Tracer")
        self.btn_packet_tracer.setStyleSheet(themed_button_style("accent"))
        self.btn_packet_tracer.setToolTip("Live packet rate, missing packets and loss.")
        self.btn_packet_tracer.clicked.connect(self.open_packet_tracer_window)
        controls_layout.addWidget(self.btn_packet_tracer)

        self.btn_options_info = QPushButton("Options Info")
        self.btn_options_info.setStyleSheet(themed_button_style("accent"))
        self.btn_options_info.setToolTip("Every option in the app with its current value and what it does.")
        self.btn_options_info.clicked.connect(self.open_options_info_dialog)
        controls_layout.addWidget(self.btn_options_info)
        self.main_layout.addLayout(controls_layout)
        self.main_layout.addSpacing(4)

        # Second row: analysis parameters and display switches.
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(12)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        # Selected channel count is configured via the Channel Calibration dialog.
        self.channel_count = 4

        selector_label_style = (
            f"background-color: {THEME_COLORS['panel']}; "
            f"color: {THEME_COLORS['text']}; "
            f"border: 1px solid {THEME_COLORS['accent']}; "
            "border-top-left-radius: 6px; border-bottom-left-radius: 6px; "
            "border-top-right-radius: 0px; border-bottom-right-radius: 0px; "
            "font-weight: 700; padding: 0 10px;"
        )
        selector_spin_style = (
            f"QSpinBox, QDoubleSpinBox {{ "
            f"background-color: {THEME_COLORS['panel']}; "
            f"color: {THEME_COLORS['text']}; "
            f"border: 1px solid {THEME_COLORS['accent']}; "
            "border-left: 0px; "
            "border-top-left-radius: 0px; border-bottom-left-radius: 0px; "
            "border-top-right-radius: 6px; border-bottom-right-radius: 6px; "
            "font-weight: 700; padding: 4px 6px; }"
        )

        self.lbl_analysis_ms = QLabel("Analysis (ms)")
        self.lbl_analysis_ms.setStyleSheet(selector_label_style)
        self.lbl_analysis_ms.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        analysis_pair_layout = QHBoxLayout()
        analysis_pair_layout.setContentsMargins(0, 0, 0, 0)
        analysis_pair_layout.setSpacing(0)
        analysis_pair_layout.addWidget(self.lbl_analysis_ms)
        self.analysis_ms_spin = QSpinBox()
        self.analysis_ms_spin.setRange(50, 2000)
        self.analysis_ms_spin.setValue(DEFAULT_ANALYSIS_MS)
        self.analysis_ms_spin.setStyleSheet(selector_spin_style)
        self.analysis_ms_spin.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
        analysis_pair_layout.addWidget(self.analysis_ms_spin)
        controls_layout.addLayout(analysis_pair_layout)
        controls_layout.addStretch(1)

        self.lbl_rms_threshold = QLabel("RMS Threshold")
        self.lbl_rms_threshold.setStyleSheet(selector_label_style)
        self.lbl_rms_threshold.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        rms_pair_layout = QHBoxLayout()
        rms_pair_layout.setContentsMargins(0, 0, 0, 0)
        rms_pair_layout.setSpacing(0)
        rms_pair_layout.addWidget(self.lbl_rms_threshold)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 4095.0)
        self.threshold_spin.setDecimals(1)
        self.threshold_spin.setSingleStep(1.0)
        self.threshold_spin.setValue(DEFAULT_THRESHOLD)
        self.threshold_spin.setStyleSheet(selector_spin_style)
        self.threshold_spin.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
        self.threshold_spin.valueChanged.connect(self.update_threshold_overlays)
        rms_pair_layout.addWidget(self.threshold_spin)
        controls_layout.addLayout(rms_pair_layout)
        controls_layout.addStretch(1)

        self.lbl_hz_threshold = QLabel("Hz Threshold")
        self.lbl_hz_threshold.setStyleSheet(selector_label_style)
        self.lbl_hz_threshold.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        hz_pair_layout = QHBoxLayout()
        hz_pair_layout.setContentsMargins(0, 0, 0, 0)
        hz_pair_layout.setSpacing(0)
        hz_pair_layout.addWidget(self.lbl_hz_threshold)
        self.hz_threshold_spin = QDoubleSpinBox()
        self.hz_threshold_spin.setRange(0.0, SAMPLE_RATE / 2.0)
        self.hz_threshold_spin.setDecimals(1)
        self.hz_threshold_spin.setSingleStep(1.0)
        self.hz_threshold_spin.setValue(DEFAULT_HZ_THRESHOLD)
        self.hz_threshold_spin.setStyleSheet(selector_spin_style)
        self.hz_threshold_spin.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
        hz_pair_layout.addWidget(self.hz_threshold_spin)
        controls_layout.addLayout(hz_pair_layout)
        controls_layout.addStretch(1)

        selector_height = max(
            self.analysis_ms_spin.sizeHint().height(),
            self.threshold_spin.sizeHint().height(),
            self.hz_threshold_spin.sizeHint().height(),
        )
        self.lbl_analysis_ms.setFixedHeight(selector_height)
        self.lbl_rms_threshold.setFixedHeight(selector_height)
        self.lbl_hz_threshold.setFixedHeight(selector_height)
        selector_width = max(
            self.analysis_ms_spin.sizeHint().width(),
            self.threshold_spin.sizeHint().width(),
            self.hz_threshold_spin.sizeHint().width(),
        )
        self.analysis_ms_spin.setFixedWidth(selector_width)
        self.threshold_spin.setFixedWidth(selector_width)
        self.hz_threshold_spin.setFixedWidth(selector_width)

        self.check_use_hz_gate = QCheckBox("Use Hz Gate")
        self.check_use_hz_gate.setChecked(True)
        controls_layout.addWidget(self.check_use_hz_gate)
        controls_layout.addStretch(1)

        self.check_autoscale = QCheckBox("Auto Scale Y")
        self.check_autoscale.setChecked(True)
        self.check_autoscale.toggled.connect(self.toggle_autoscale)
        controls_layout.addWidget(self.check_autoscale)
        controls_layout.addStretch(1)

        self.check_graph_stream = QCheckBox("Graph Stream")
        self.check_graph_stream.setChecked(True)
        controls_layout.addWidget(self.check_graph_stream)
        controls_layout.addStretch(1)

        self.main_layout.addLayout(controls_layout)
        self.main_layout.addSpacing(6)

        info_actions_layout = QHBoxLayout()
        info_actions_layout.setSpacing(12)
        info_actions_layout.setAlignment(Qt.AlignTop)

        info_text_layout = QVBoxLayout()
        info_text_layout.setContentsMargins(0, 0, 0, 0)
        info_text_layout.setSpacing(2)
        self.lbl_status = StableLabel("Status: Disconnected")
        self.lbl_status.setStyleSheet(themed_label_style("muted"))
        info_text_layout.addWidget(self.lbl_status)

        self.lbl_rf = StableLabel("RF Model: Not loaded")
        self.lbl_rf.setStyleSheet(themed_label_style("muted"))
        info_text_layout.addWidget(self.lbl_rf)
        info_actions_layout.addLayout(info_text_layout, 0)

        info_actions_layout.addStretch()
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 2, 0, 0)
        buttons_layout.setSpacing(12)

        self.btn_data_collection = QPushButton("Data Collection")
        self.btn_data_collection.setStyleSheet(themed_button_style("accent"))
        self.btn_data_collection.clicked.connect(self.open_data_collection_dialog)
        buttons_layout.addWidget(self.btn_data_collection)

        self.btn_train_model = QPushButton("Train Model")
        self.btn_train_model.setStyleSheet(themed_button_style("accent"))
        self.btn_train_model.clicked.connect(self.train_rf_from_app)
        buttons_layout.addWidget(self.btn_train_model)

        self.btn_load_trained_model = QPushButton("Load Trained Model")
        self.btn_load_trained_model.setStyleSheet(themed_button_style("accent"))
        self.btn_load_trained_model.clicked.connect(self.load_rf_model_dialog)
        buttons_layout.addWidget(self.btn_load_trained_model)

        self.btn_realtime_classification = QPushButton("Real-time Classification")
        self.btn_realtime_classification.setStyleSheet(themed_button_style("accent"))
        self.btn_realtime_classification.clicked.connect(self.open_realtime_classification_window)
        buttons_layout.addWidget(self.btn_realtime_classification)
        info_actions_layout.addLayout(buttons_layout, 0)
        self.main_layout.addLayout(info_actions_layout)
        self.main_layout.addSpacing(8)

        # --- GRAPH + SIDE STATUS AREA ---
        self.graph_side_layout = QHBoxLayout()
        self.graph_side_layout.setSpacing(10)

        self.plot_container = RoundedClipWidget(radius=10)
        self.plot_container.setObjectName("mainPlotContainer")
        self.plot_container.setStyleSheet(
            f"QWidget#mainPlotContainer {{ "
            f"background-color: #000000; border: 1px solid {THEME_COLORS['accent']}; border-radius: 10px; }}"
        )
        plot_container_layout = QVBoxLayout(self.plot_container)
        plot_container_layout.setContentsMargins(1, 1, 1, 1)
        plot_container_layout.setSpacing(0)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setObjectName("mainPlotWidget")
        self.plot_widget.setBackground("k")
        self.plot_widget.setStyleSheet(
            "QGraphicsView#mainPlotWidget { background-color: #000000; border: 0px; border-radius: 9px; }"
        )
        self.plot_widget.viewport().setStyleSheet(
            "background-color: #000000; border: 0px; border-radius: 9px;"
        )
        plot_container_layout.addWidget(self.plot_widget)
        self.graph_side_layout.addWidget(self.plot_container, 4)  # ~80% width

        self.status_panel = QWidget()
        self.status_panel.setObjectName("channelStatusPanel")
        self.status_panel.setStyleSheet(
            f"QWidget#channelStatusPanel {{ "
            f"background-color: {THEME_COLORS['panel']}; border: 1px solid {THEME_COLORS['accent']}; border-radius: 10px; }} "
            f"QWidget#channelStatusPanel QLabel {{ background: transparent; border: none; }}"
        )
        self.status_panel.setMinimumWidth(260)
        self.status_panel.setMaximumWidth(420)
        self.status_panel_layout = QVBoxLayout(self.status_panel)
        self.status_panel_layout.setContentsMargins(10, 10, 10, 10)
        self.status_panel_layout.setSpacing(8)

        self.lbl_side_title = QLabel("Channel Status")
        self.lbl_side_title.setStyleSheet(
            f"font-weight: bold; color: {THEME_COLORS['text']}; background: transparent; border: none;"
        )
        self.status_panel_layout.addWidget(self.lbl_side_title)

        self.metrics_scroll = QScrollArea()
        self.metrics_scroll.setObjectName("metricsScroll")
        self.metrics_scroll.setStyleSheet(
            f"QScrollArea#metricsScroll {{ "
            f"border: 1px solid {THEME_COLORS['accent']}; border-radius: 8px; background-color: {THEME_COLORS['panel']}; }}"
        )
        self.metrics_scroll.setWidgetResizable(True)
        self.metrics_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.metrics_scroll.viewport().setStyleSheet(
            f"background-color: {THEME_COLORS['panel']}; border-radius: 8px;"
        )
        self.metrics_scroll_content = QWidget()
        self.metrics_scroll_content.setObjectName("metricsScrollContent")
        self.metrics_scroll_content.setStyleSheet(
            f"QWidget#metricsScrollContent {{ background-color: {THEME_COLORS['panel']}; border: none; }}"
        )
        self.metrics_layout = QVBoxLayout(self.metrics_scroll_content)
        self.metrics_layout.setContentsMargins(8, 8, 8, 8)
        self.metrics_layout.setSpacing(6)
        self.metrics_scroll.setWidget(self.metrics_scroll_content)
        self.status_panel_layout.addWidget(self.metrics_scroll, 1)

        self.fps_panel = QWidget()
        self.fps_panel.setStyleSheet(
            f"QWidget {{ background: {THEME_COLORS['panel']}; border: none; }}"
        )
        fps_layout = QVBoxLayout(self.fps_panel)
        fps_layout.setContentsMargins(4, 2, 4, 2)
        fps_layout.setSpacing(2)

        self.lbl_perf_title = QLabel("Performance")
        self.lbl_perf_title.setStyleSheet(f"font-weight: bold; color: {THEME_COLORS['text']};")
        fps_layout.addWidget(self.lbl_perf_title)

        mono_style = f"color: {THEME_COLORS['muted']}; font-family: {MONO_FONT_CSS};"
        self.lbl_ui_fps = StableLabel("UI FPS: --")
        self.lbl_ui_fps.setStyleSheet(mono_style)
        fps_layout.addWidget(self.lbl_ui_fps)

        self.lbl_data_fps = StableLabel("Data FPS: --")
        self.lbl_data_fps.setStyleSheet(mono_style)
        fps_layout.addWidget(self.lbl_data_fps)

        self.lbl_link_title = QLabel("Link")
        self.lbl_link_title.setStyleSheet(f"font-weight: bold; color: {THEME_COLORS['text']};")
        fps_layout.addWidget(self.lbl_link_title)
        self.lbl_pkt_rate = StableLabel("Packets: -- /s")
        self.lbl_pkt_rate.setStyleSheet(mono_style)
        fps_layout.addWidget(self.lbl_pkt_rate)
        self.lbl_pkt_loss = StableLabel("Missing: -- (--%)")
        self.lbl_pkt_loss.setStyleSheet(mono_style)
        fps_layout.addWidget(self.lbl_pkt_loss)
        self.lbl_csv_rows = StableLabel("CSV rows: 0")
        self.lbl_csv_rows.setStyleSheet(mono_style)
        fps_layout.addWidget(self.lbl_csv_rows)

        self.status_panel_layout.addWidget(self.fps_panel, 0)

        self.graph_side_layout.addWidget(self.status_panel, 1)  # ~20% width
        self.main_layout.addLayout(self.graph_side_layout, 1)

        # --- PLOT REFRESH TIMER ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.keepalive_timer = QTimer(self)
        self.keepalive_timer.timeout.connect(self.send_wireless_keepalive)
        self.keepalive_timer.start(KEEPALIVE_INTERVAL_MS)
        self.packet_timer = QTimer(self)
        self.packet_timer.timeout.connect(self.on_packet_tick)
        self.packet_timer.start(PACKET_TRACER_TICK_MS)

        self.reset_fps_counters()
        self.refresh_ports()
        self.setup_plots(self.num_channels)
        self.setup_metrics_labels(self.num_channels)
        self.set_waiting_for_calibration_labels()

        default_rf = os.path.join(TRAINED_MODEL_DIR, DEFAULT_RF_MODEL_ARTIFACT)
        if os.path.isfile(default_rf):
            self.load_rf_model(default_rf)

    def set_status(self, text, color="#999"):
        self.lbl_status.setText(f"Status: {text}")
        self.lbl_status.setStyleSheet(
            f"color: {themed_status_color(color)}; font-weight: bold;"
        )

    @staticmethod
    def disconnect_button_stylesheet():
        return (
            f"QPushButton {{ "
            f"background-color: {THEME_COLORS['accent']}; "
            f"color: {THEME_COLORS['text']}; "
            f"font-weight: bold; "
            f"border: 1px solid {THEME_COLORS['accent']}; "
            f"border-radius: 6px; padding: 5px 12px; "
            f"}} "
            f"QPushButton:hover:!disabled {{ "
            f"background-color: {THEME_COLORS['special']}; "
            f"border-color: {THEME_COLORS['special']}; "
            f"}} "
            f"QPushButton:disabled {{ "
            f"background-color: {THEME_COLORS['panel']}; "
            f"color: {THEME_COLORS['disabled']}; "
            f"border-color: {THEME_COLORS['panel']}; "
            f"}}"
        )

    @staticmethod
    def connected_port_button_stylesheet():
        return (
            "QPushButton { background-color: #78A083; color: #132440; font-weight: bold; "
            "border: 1px solid #78A083; border-radius: 6px; padding: 5px 12px; } "
            "QPushButton:hover:!disabled { background-color: #8DB596; border-color: #8DB596; } "
            "QPushButton:pressed:!disabled { background-color: #5F8A6A; border-color: #5F8A6A; } "
            f"QPushButton:disabled {{ background-color: {THEME_COLORS['panel']}; "
            f"color: {THEME_COLORS['disabled']}; border-color: {THEME_COLORS['panel']}; }}"
        )

    def layout_has_imu(self):
        """Wi-Fi and binary USB both deliver the 8 EMG + Roll/Pitch/Yaw layout."""
        return self.connection_medium == "wireless" or self.wired_binary

    def active_emg_channel_count(self):
        if self.layout_has_imu():
            return WIRELESS_EMG_CHANNELS
        return int(self.num_channels)

    def channel_display_name(self, index):
        idx = int(index)
        if self.layout_has_imu() and self.num_channels >= WIRELESS_TOTAL_CHANNELS:
            if idx == 8:
                return "ROLL"
            if idx == 9:
                return "PITCH"
            if idx == 10:
                return "YAW"
        return f"CH {idx + 1}"

    def is_wireless_imu_channel(self, index):
        return self.layout_has_imu() and int(index) >= WIRELESS_EMG_CHANNELS

    def reset_fps_counters(self):
        self.ui_fps = 0.0
        self.data_fps = 0.0
        self._ui_fps_frame_count = 0
        self._data_fps_sample_count = 0
        now = time.perf_counter()
        self._ui_fps_last_ts = now
        self._data_fps_last_ts = now
        self._refresh_fps_labels(active=False)

    def send_wireless_keepalive(self):
        if not self.is_connected or self.connection_medium != "wireless":
            return
        if self.current_device is None or not self.wireless_access_key:
            return

        # The ping blocks for up to CONTROL_TIMEOUT when the device is slow; doing that on the
        # GUI thread freezes the UI (and the consumer of the stream) - run it in the background.
        if self.keepalive_failures >= 3:
            self.set_status("Wireless keepalive lost", "#f57c00")
        if self._keepalive_thread is not None and self._keepalive_thread.is_alive():
            return
        ip, key = self.current_device.ip, self.wireless_access_key

        client_ip = self._stream_client_ip

        def _ping():
            try:
                ControlProtocol.ping(ip, key)
                self.keepalive_failures = 0
            except Exception as exc:
                self.keepalive_failures += 1
                if self.keepalive_failures >= KEEPALIVE_RESTART_AFTER and client_ip:
                    # The device probably lost its stream target (reboot, Wi-Fi drop, timeout):
                    # re-issue START from this background thread and keep a record of it.
                    event = {"host_monotonic_s": time.monotonic(), "reason": str(exc), "ok": False}
                    try:
                        ControlProtocol.start_stream(ip, key, client_ip, WIFI_STREAM_PORT)
                        event["ok"] = True
                        self.keepalive_failures = 0
                    except Exception as restart_exc:
                        event["error"] = str(restart_exc)
                    self.keepalive_restarts.append(event)

        self._keepalive_thread = threading.Thread(target=_ping, name="biowave-keepalive", daemon=True)
        self._keepalive_thread.start()

    def _refresh_fps_labels(self, active=True):
        if not active:
            self.lbl_ui_fps.setText("UI FPS: --")
            self.lbl_data_fps.setText("Data FPS: --")
            return
        self.lbl_ui_fps.setText(f"UI FPS: {self.ui_fps:.1f}")
        self.lbl_data_fps.setText(f"Data FPS: {self.data_fps:.1f} Hz")

    def _tick_ui_fps(self):
        self._ui_fps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._ui_fps_last_ts
        if elapsed < 0.5:
            return
        instant = self._ui_fps_frame_count / max(1e-6, elapsed)
        self.ui_fps = (0.65 * self.ui_fps) + (0.35 * instant) if self.ui_fps > 0 else instant
        self._ui_fps_frame_count = 0
        self._ui_fps_last_ts = now
        self._refresh_fps_labels(active=True)

    def _tick_data_fps(self, sample_count):
        self._data_fps_sample_count += int(max(0, sample_count))
        now = time.perf_counter()
        elapsed = now - self._data_fps_last_ts
        if elapsed < 0.5:
            return
        instant = self._data_fps_sample_count / max(1e-6, elapsed)
        self.data_fps = (0.65 * self.data_fps) + (0.35 * instant) if self.data_fps > 0 else instant
        self._data_fps_sample_count = 0
        self._data_fps_last_ts = now
        self._refresh_fps_labels(active=True)

    def list_available_ports(self):
        sim_url = "socket://127.0.0.1:7000"
        items = [(f"{sim_url} - Local Simulator (TCP)", sim_url)]
        ports = serial.tools.list_ports.comports()
        for port in ports:
            items.append((f"{port.device} - {port.description}", port.device))
        return items

    def refresh_ports(self, preferred_port=None):
        if preferred_port is not None:
            self.current_port_name = str(preferred_port).strip() or self.current_port_name
        items = self.list_available_ports()
        if self.port_config_dialog and self.port_config_dialog.isVisible():
            self.port_config_dialog.set_ports(items, self.current_port_name)
        return items

    def sync_port_dialog_state(self):
        if self.port_config_dialog and self.port_config_dialog.isVisible():
            self.port_config_dialog.update_connection_state(self.is_connected)
            self.port_config_dialog.set_ports(
                self.list_available_ports(),
                self.current_port_name,
            )
        if self.wireless_config_dialog and self.wireless_config_dialog.isVisible():
            selected_ip = self.current_device.ip if self.current_device is not None else ""
            self.wireless_config_dialog.set_devices(self.discovered_wireless_devices, selected_ip=selected_ip)
            if self.is_connected and self.connection_medium == "wireless" and self.current_device is not None:
                self.wireless_config_dialog.set_connection_state(
                    True,
                    f"Connected to {self.current_device.summary}. Click Apply to continue.",
                )
            else:
                self.wireless_config_dialog.set_connection_state(False)

    def open_port_configuration_dialog(self):
        mode_dialog = ConnectionModeDialog(self)
        center_window(mode_dialog, self)
        if mode_dialog.exec_() != QDialog.Accepted:
            return

        if mode_dialog.selected_medium == "wireless":
            self.open_wireless_configuration_dialog()
        else:
            self.open_wired_port_configuration_dialog()

    def open_wired_port_configuration_dialog(self):
        if self.port_config_dialog and self.port_config_dialog.isVisible():
            self.port_config_dialog.raise_()
            self.port_config_dialog.activateWindow()
            return

        selected_port = self.current_port_name if not str(self.current_port_name).startswith("wifi://") else ""
        dlg = PortConfigDialog(selected_port, self.is_connected, self)
        dlg.connect_requested.connect(self.connect_serial)
        dlg.refresh_requested.connect(lambda: self.refresh_ports(dlg.selected_port()))
        self.port_config_dialog = dlg
        self.refresh_ports(selected_port)
        center_window(dlg, self)
        result = dlg.exec_()
        chosen_port = dlg.selected_port().strip()
        if chosen_port:
            self.current_port_name = chosen_port
        self.port_config_dialog = None
        if result == QDialog.Accepted and self.is_connected:
            self.port_config_applied = True
        elif not self.is_connected:
            self.port_config_applied = False
        self.sync_calibration_dialog_state()

    def open_usb_wifi_provision_dialog(self):
        dialog = USBProvisionDialog(self)
        center_window(dialog, self)
        if dialog.exec_() == QDialog.Accepted and self.wireless_config_dialog and self.wireless_config_dialog.isVisible():
            self.discover_wireless_devices(self.wireless_config_dialog)

    def discover_wireless_devices(self, dialog=None):
        try:
            devices = ControlProtocol.discover()
        except Exception as exc:
            target_dialog = dialog or self.wireless_config_dialog
            if target_dialog is not None:
                target_dialog.set_connection_state(False, f"Wireless discovery failed: {exc}")
            QMessageBox.warning(self, "Wireless Discovery Failed", str(exc))
            return

        self.discovered_wireless_devices = devices
        target_dialog = dialog or self.wireless_config_dialog
        if target_dialog is not None:
            selected_ip = self.current_device.ip if self.current_device is not None else ""
            target_dialog.set_devices(devices, selected_ip=selected_ip)
            if devices:
                target_dialog.set_connection_state(False, f"Discovered {len(devices)} BioWave wireless device(s).")
            else:
                target_dialog.set_connection_state(False, "No BioWave wireless devices replied on the current Wi-Fi.")

    def connect_wireless(self, device, access_key, dialog=None):
        if device is None:
            raise RuntimeError("No BioWave wireless device is selected.")

        if self.is_connected:
            self.disconnect_serial()

        self.live_analysis_enabled = False
        self.connection_medium = "wireless"
        self.wired_binary = False
        self.current_device = device
        self.wireless_access_key = str(access_key or "").strip()
        problem = access_key_problem(self.wireless_access_key)
        if problem:
            raise RuntimeError(problem)
        self.keepalive_failures = 0
        self.keepalive_restarts = []
        self.channel_count = WIRELESS_TOTAL_CHANNELS
        self.reset_runtime_state_for_channels(self.channel_count)

        if self.analysis_window is not None:
            self.analysis_window.close()
            self.analysis_window = None

        try:
            self.packet_tracer.reset("wireless", WIRELESS_FRAMES_PER_PACKET)
            self.serial_worker = WirelessStreamWorker(WIFI_STREAM_PORT, tracer=self.packet_tracer)
            self.serial_worker.batch_received.connect(self.on_serial_batch)
            self.serial_worker.error_occurred.connect(self.on_serial_error)
            self.serial_worker.start(QThread.TimeCriticalPriority)

            client_ip = get_local_ip_for_target(device.ip)
            self._stream_client_ip = client_ip
            ControlProtocol.start_stream(device.ip, self.wireless_access_key, client_ip, WIFI_STREAM_PORT)

            self.current_port_name = f"wifi://{device.ip}"
            self.is_connected = True
            self.port_config_applied = False
            self.btn_port_config.setStyleSheet(self.connected_port_button_stylesheet())
            self.btn_disconnect.setEnabled(True)
            self.btn_calibrate.setEnabled(True)
            self.btn_analysis.setEnabled(False)
            self.timer.stop()
            self.reset_fps_counters()
            self.set_status("Wireless connected - calibration required", "#ff9800")
            if self.rf_model is not None:
                model_name = os.path.basename(self.rf_model_path) if self.rf_model_path else "Loaded"
                self.rf_last_status_reason = "Waiting calibrated stream..."
                self.lbl_rf.setText(f"RF Model: {model_name} | Waiting calibrated stream...")
                self.lbl_rf.setStyleSheet(themed_label_style("success"))
            else:
                self.lbl_rf.setText("RF Model: Not loaded")
                self.lbl_rf.setStyleSheet(themed_label_style("muted"))

            if dialog is not None:
                dialog.set_connection_state(True, f"Connected to {device.summary}. Click Apply to continue.")
            self.sync_port_dialog_state()
            self.sync_calibration_dialog_state()
        except Exception:
            if self.serial_worker:
                self.serial_worker.stop()
                self.serial_worker = None
            self.is_connected = False
            self.current_device = None
            self.port_config_applied = False
            if dialog is not None:
                dialog.set_connection_state(False, "Wireless connection failed.")
            raise

    def on_wireless_connect_requested(self, device, access_key):
        try:
            self.connect_wireless(device, access_key, self.wireless_config_dialog)
        except Exception as exc:
            QMessageBox.critical(self, "Wireless Connection Error", str(exc))

    def open_wireless_configuration_dialog(self):
        if self.wireless_config_dialog and self.wireless_config_dialog.isVisible():
            self.wireless_config_dialog.raise_()
            self.wireless_config_dialog.activateWindow()
            return

        dlg = WirelessConfigDialog(access_key=self.wireless_access_key, parent=self)
        dlg.discover_requested.connect(lambda: self.discover_wireless_devices(dlg))
        dlg.provision_requested.connect(self.open_usb_wifi_provision_dialog)
        dlg.connect_requested.connect(self.on_wireless_connect_requested)
        self.wireless_config_dialog = dlg
        dlg.set_devices(self.discovered_wireless_devices, selected_ip=self.current_device.ip if self.current_device else "")
        if self.is_connected and self.connection_medium == "wireless" and self.current_device is not None:
            dlg.set_connection_state(True, f"Connected to {self.current_device.summary}. Click Apply to continue.")
        else:
            self.discover_wireless_devices(dlg)

        center_window(dlg, self)
        result = dlg.exec_()
        self.wireless_access_key = dlg.access_key_input.text().strip()
        self.wireless_config_dialog = None
        if result == QDialog.Accepted and self.is_connected and self.connection_medium == "wireless":
            self.port_config_applied = True
        elif not self.is_connected:
            self.port_config_applied = False
        self.sync_calibration_dialog_state()

    def reset_runtime_state_for_channels(self, num_channels):
        self.num_channels = int(max(1, num_channels))
        with self._buffer_lock:
            self.raw_data_buffer = np.zeros((self.num_channels, WINDOW_SIZE), dtype=np.float32)
            self.data_buffer = np.zeros((self.num_channels, WINDOW_SIZE), dtype=np.float32)
        self.baseline_offsets = np.zeros(self.num_channels, dtype=np.float32)
        self.latest_mav = np.zeros(self.num_channels, dtype=np.float32)
        self.latest_rms = np.zeros(self.num_channels, dtype=np.float32)
        self.latest_dom_hz = np.zeros(self.num_channels, dtype=np.float32)
        self.latest_mean_hz = np.zeros(self.num_channels, dtype=np.float32)
        self.latest_median_hz = np.zeros(self.num_channels, dtype=np.float32)
        self.latest_spec_entropy = np.zeros(self.num_channels, dtype=np.float32)
        self.latest_band_power_pct = np.zeros((self.num_channels, len(BAND_DEFS)), dtype=np.float32)
        self.latest_stft_dom_mean_hz = np.zeros(self.num_channels, dtype=np.float32)
        self.latest_stft_dom_std_hz = np.zeros(self.num_channels, dtype=np.float32)
        self.latest_short_time_band_delta = np.zeros((self.num_channels, len(BAND_DEFS)), dtype=np.float32)
        self.latest_wavelet_energy_pct = np.zeros((self.num_channels, 4), dtype=np.float32)
        self.latest_corr_matrix = np.eye(self.num_channels, dtype=np.float32)
        self.latest_lag_ms_matrix = np.zeros((self.num_channels, self.num_channels), dtype=np.float32)
        self.latest_coherence_matrix = np.eye(self.num_channels, dtype=np.float32)
        self.latest_channel_ratio = np.ones(self.num_channels, dtype=np.float32)
        self.latest_symmetry_index = {}
        self.latest_co_contraction_index = {}
        self.latest_mains_noise_score = np.zeros(self.num_channels, dtype=np.float32)
        self.latest_active = np.zeros(self.num_channels, dtype=bool)
        self.latest_hz_active = np.zeros(self.num_channels, dtype=bool)
        self.onset_state = np.zeros(self.num_channels, dtype=bool)
        self.onset_count = np.zeros(self.num_channels, dtype=np.int32)
        self.offset_count = np.zeros(self.num_channels, dtype=np.int32)
        self.current_burst_ms = np.zeros(self.num_channels, dtype=np.float32)
        self.burst_history_ms = [[] for _ in range(self.num_channels)]
        self.repetition_count = np.zeros(self.num_channels, dtype=np.int32)
        self.latest_quality_score = np.zeros(self.num_channels, dtype=np.float32)
        self.rest_rms_ref = np.ones(self.num_channels, dtype=np.float32) * 5.0
        self.flex_rms_ref = np.ones(self.num_channels, dtype=np.float32) * 50.0
        self.baseline_reference = np.zeros(self.num_channels, dtype=np.float32)
        self.last_clip_ratio = np.zeros(self.num_channels, dtype=np.float32)
        self.fatigue_track = []
        self.force_level_pct = 0.0
        self.prev_force_level_pct = 0.0
        self.movement_phase = "REST"
        self.gesture_label = "REST"
        self.anomaly_label = "NORMAL"
        self.contact_quality = ["UNKNOWN"] * self.num_channels
        self.rest_capture = []
        self.flex_capture = []
        self.is_calibrated = False
        self.calibration_active = False
        self.rf_valid_sample_count = 0
        self.rf_samples_since_submit = 0
        self._analysis_pending = False
        self._last_analysis_submit_ts = 0.0
        self._plot_ranges_dirty = True
        self._last_plot_range_refresh_ts = 0.0

        self.setup_plots(self.num_channels)
        self.setup_metrics_labels(self.num_channels)
        self.set_waiting_for_calibration_labels()

    def set_calibration_phase_durations(self, rest_s, flex_s):
        self.cal_rest_seconds = int(max(CAL_DURATION_MIN_S, min(CAL_DURATION_MAX_S, rest_s)))
        self.cal_flex_seconds = int(max(CAL_DURATION_MIN_S, min(CAL_DURATION_MAX_S, flex_s)))
        for phase in self.calibration_phases:
            if phase["key"] == "rest":
                phase["duration_ms"] = int(self.cal_rest_seconds * 1000)
            elif phase["key"] == "flex":
                phase["duration_ms"] = int(self.cal_flex_seconds * 1000)

    def connect_serial(self, port_name=None):
        if self.is_connected:
            return
        self.live_analysis_enabled = False
        self.connection_medium = "wired"
        self.current_device = None

        if port_name is not None:
            self.current_port_name = str(port_name).strip()

        port_name = (self.current_port_name or "").strip()
        if not port_name:
            QMessageBox.warning(self, "No Port", "Please select a valid serial port.")
            return

        # A real COM port streams the fixed 11-channel BWIM layout; socket:// simulators stream CSV
        # with a user-chosen 2-11 channels.
        self.wired_binary = not port_name.lower().startswith("socket://")
        if self.wired_binary:
            self.channel_count = WIRELESS_TOTAL_CHANNELS
        self.wired_channel_count = int(max(2, min(11, self.channel_count)))
        self.channel_count = self.wired_channel_count
        self.reset_runtime_state_for_channels(self.channel_count)
        if self.analysis_window is not None:
            self.analysis_window.close()
            self.analysis_window = None

        try:
            if self.wired_binary:
                self.packet_tracer.reset("wireless", WIRELESS_FRAMES_PER_PACKET)
            else:
                self.packet_tracer.reset("serial", 1)
            self.serial_worker = SerialWorker(port_name, DEFAULT_BAUD_RATE, self.num_channels,
                                              tracer=self.packet_tracer)
            self.serial_worker.batch_received.connect(self.on_serial_batch)
            self.serial_worker.error_occurred.connect(self.on_serial_error)
            self.serial_worker.start(QThread.HighestPriority)

            self.current_port_name = port_name
            self.is_connected = True
            self.port_config_applied = False
            self.btn_port_config.setStyleSheet(self.connected_port_button_stylesheet())
            self.btn_disconnect.setEnabled(True)
            self.btn_calibrate.setEnabled(True)
            self.btn_analysis.setEnabled(False)

            self.timer.stop()  # Hard requirement: no graph/analysis before calibration.
            self.reset_fps_counters()
            self.set_status("Connected - calibration required", "#ff9800")
            if self.rf_model is not None:
                model_name = os.path.basename(self.rf_model_path) if self.rf_model_path else "Loaded"
                self.rf_last_status_reason = "Waiting calibrated stream..."
                self.lbl_rf.setText(f"RF Model: {model_name} | Waiting calibrated stream...")
                self.lbl_rf.setStyleSheet(themed_label_style("success"))
            else:
                self.lbl_rf.setText("RF Model: Not loaded")
                self.lbl_rf.setStyleSheet(themed_label_style("muted"))
            self.sync_port_dialog_state()
            self.sync_calibration_dialog_state()

        except Exception as e:
            QMessageBox.critical(self, "Connection Error", str(e))
            self.disconnect_serial()

    def disconnect_serial(self):
        self.timer.stop()
        self.stop_calibration_if_running()
        self.reset_fps_counters()
        self.live_analysis_enabled = False
        self._analysis_pending = False
        self._last_analysis_submit_ts = 0.0
        self._plot_ranges_dirty = True
        self._last_plot_range_refresh_ts = 0.0
        self.rf_samples_since_submit = 0
        self.rf_last_latency_ms = 0.0
        self.rf_prediction_rate_hz = 0.0
        self.rf_last_prediction_ts = 0.0
        self.rf_valid_sample_count = 0

        if self.connection_medium == "wireless" and self.current_device is not None and self.wireless_access_key:
            try:
                ControlProtocol.stop_stream(self.current_device.ip, self.wireless_access_key)
            except Exception:
                pass

        if self.serial_worker:
            try:
                self.serial_worker.batch_received.disconnect()
                self.serial_worker.error_occurred.disconnect()
            except Exception:
                pass
            self.serial_worker.stop()
            self.serial_worker = None

        if self.calibration_dialog and self.calibration_dialog.isVisible():
            self.calibration_dialog.close()
        self.calibration_dialog = None
        if self.analysis_window and self.analysis_window.isVisible():
            self.analysis_window.close()
        self.analysis_window = None
        if self.data_collection_dialog and self.data_collection_dialog.isVisible():
            self.data_collection_dialog.cancel_task_protocol(emit_signal=True)
        self._set_task_collection_mode(False)
        self.live_analysis_enabled = False

        self.is_connected = False
        self.is_calibrated = False
        self.port_config_applied = False
        self.current_device = None
        self.keepalive_failures = 0
        self.btn_port_config.setStyleSheet(themed_button_style("accent"))
        self.btn_disconnect.setEnabled(False)
        self.btn_calibrate.setEnabled(True)
        self.btn_analysis.setEnabled(False)
        self.set_status("Disconnected", "#999")
        self.reset_link_labels()
        self.set_waiting_for_calibration_labels()
        if self.rf_model is not None:
            model_name = os.path.basename(self.rf_model_path) if self.rf_model_path else "Loaded"
            self.rf_last_status_reason = "Disconnected"
            self.lbl_rf.setText(f"RF Model: {model_name} | Disconnected")
            self.lbl_rf.setStyleSheet(themed_label_style("success"))
        else:
            self.lbl_rf.setText("RF Model: Not loaded")
            self.lbl_rf.setStyleSheet(themed_label_style("muted"))
        self.sync_port_dialog_state()
        self.sync_calibration_dialog_state()

    def setup_metrics_labels(self, num_channels):
        while self.metrics_layout.count():
            item = self.metrics_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.metrics_labels = []
        for i in range(num_channels):
            label = QLabel(f"{self.channel_display_name(i)} | Waiting calibration")
            label.setWordWrap(True)
            label.setStyleSheet("color: #B7C6CC; font-family: Consolas; font-size: 15px;")
            self.metrics_layout.addWidget(label)
            self.metrics_labels.append(label)
        self.metrics_layout.addStretch()
        self.analysis_idle_labels_active = False

    def set_waiting_for_calibration_labels(self):
        for i in range(len(self.metrics_labels)):
            self.metrics_labels[i].setText(f"{self.channel_display_name(i)} | Waiting calibration")
            self.metrics_labels[i].setStyleSheet("color: #B7C6CC; font-family: Consolas; font-size: 15px;")
        self.analysis_idle_labels_active = False

    def set_analysis_idle_labels(self):
        if self.analysis_idle_labels_active:
            return
        for i in range(len(self.metrics_labels)):
            self.metrics_labels[i].setText(f"{self.channel_display_name(i)} | Stream ready | Live Analysis OFF")
            self.metrics_labels[i].setStyleSheet("color: #B7C6CC; font-family: Consolas; font-size: 15px;")
        self.analysis_idle_labels_active = True

    def setup_plots(self, num_channels):
        self.plot_widget.clear()
        self.curves = []
        self.plots = []
        self.plot_rows = []
        self.threshold_lines_pos = []
        self.threshold_lines_neg = []
        self.zero_lines = []
        self._metrics_color_cache = {}
        self._threshold_line_color_cache = {}
        self._threshold_line_value_cache = None

        colors = [
            "#3B9797",
            "#BF092F",
            "#5CC8FF",
            "#FFB703",
            "#A8E6CF",
            "#C77DFF",
            "#FF6B6B",
            "#4DD0E1",
            "#FFD166",
        ]
        threshold = self.threshold_spin.value()
        x_min = float(self.x_axis[0])
        x_max = float(self.x_axis[-1])
        x_span = x_max - x_min

        def configure_plot(plot_item, left_label, show_bottom, is_imu=False):
            plot_item.showGrid(x=False, y=True, alpha=0.3)
            plot_item.setLabel("left", left_label)
            left_axis = plot_item.getAxis("left")
            left_axis.setStyle(autoExpandTextSpace=False, tickTextWidth=Y_AXIS_TICK_TEXT_WIDTH, showValues=False)
            left_axis.setWidth(Y_AXIS_FIXED_WIDTH)
            try:
                tick_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
                tick_font.setPointSize(PLOT_AXIS_TICK_FONT_PT)
                left_axis.setTickFont(tick_font)
            except Exception:
                pass

            if not show_bottom:
                plot_item.hideAxis("bottom")
            else:
                bottom_axis = plot_item.getAxis("bottom")
                bottom_axis.setStyle(showValues=False)
                plot_item.setLabel("bottom", "Time (s)")

            plot_item.setMouseEnabled(x=False, y=False)
            plot_item.hideButtons()
            plot_item.setMenuEnabled(False)
            # Draw far fewer segments: peak-preserving downsampling keeps EMG spikes
            # visible while cutting vertices, and clip-to-view skips off-screen points.
            try:
                plot_item.setDownsampling(mode="peak", auto=True)
                plot_item.setClipToView(True)
            except Exception:
                pass
            plot_item.disableAutoRange(axis="x")
            plot_item.setLimits(xMin=x_min, xMax=x_max, minXRange=x_span, maxXRange=x_span)
            plot_item.setXRange(x_min, x_max, padding=0)
            if self.check_autoscale.isChecked():
                plot_item.enableAutoRange(axis="y")
            else:
                plot_item.disableAutoRange(axis="y")
                if is_imu:
                    plot_item.setYRange(IMU_Y_MIN, IMU_Y_MAX, padding=0)
                else:
                    plot_item.setYRange(Y_MIN, Y_MAX, padding=0)

        def add_emg_plot(row_index, channel_index, show_bottom):
            p = self.plot_widget.addPlot(row=row_index, col=0)
            configure_plot(p, self.channel_display_name(channel_index), show_bottom, is_imu=False)
            try:
                p.setMinimumHeight(52)
            except Exception:
                pass
            curve = p.plot(pen=pg.mkPen(colors[channel_index % len(colors)], width=2))
            threshold_line_pos = pg.InfiniteLine(
                pos=threshold,
                angle=0,
                movable=False,
                pen=pg.mkPen(THEME_COLORS["muted"], width=1, style=Qt.DashLine),
            )
            threshold_line_neg = pg.InfiniteLine(
                pos=-threshold,
                angle=0,
                movable=False,
                pen=pg.mkPen(THEME_COLORS["muted"], width=1, style=Qt.DashLine),
            )
            zero_line = pg.InfiniteLine(
                pos=0.0,
                angle=0,
                movable=False,
                pen=pg.mkPen(THEME_COLORS["accent"], width=1),
            )
            p.addItem(threshold_line_pos)
            p.addItem(threshold_line_neg)
            p.addItem(zero_line)

            self.curves.append(curve)
            self.plots.append(p)
            self.plot_rows.append({"kind": "single", "plot": p, "channel": channel_index, "curve": curve, "is_imu": False})
            self.threshold_lines_pos.append(threshold_line_pos)
            self.threshold_lines_neg.append(threshold_line_neg)
            self.zero_lines.append(zero_line)

        if self.layout_has_imu() and num_channels >= WIRELESS_TOTAL_CHANNELS:
            for i in range(WIRELESS_EMG_CHANNELS):
                add_emg_plot(i, i, show_bottom=False)

            imu_row = WIRELESS_EMG_CHANNELS
            p = self.plot_widget.addPlot(row=imu_row, col=0, rowspan=3)
            configure_plot(p, "IMU", show_bottom=True, is_imu=True)
            try:
                p.setMinimumHeight(150)
            except Exception:
                pass
            p.addLegend(offset=(8, 8))
            imu_curves = [
                p.plot(pen=pg.mkPen("#d32f2f", width=2), name="Roll"),
                p.plot(pen=pg.mkPen("#2e7d32", width=2), name="Pitch"),
                p.plot(pen=pg.mkPen("#1565c0", width=2), name="Yaw"),
            ]
            zero_line = pg.InfiniteLine(
                pos=0.0,
                angle=0,
                movable=False,
                pen=pg.mkPen(THEME_COLORS["accent"], width=1),
            )
            p.addItem(zero_line)
            self.plots.append(p)
            self.plot_rows.append(
                {
                    "kind": "imu_combined",
                    "plot": p,
                    "channels": [WIRELESS_EMG_CHANNELS, WIRELESS_EMG_CHANNELS + 1, WIRELESS_EMG_CHANNELS + 2],
                    "curves": imu_curves,
                    "is_imu": True,
                }
            )
            self.threshold_lines_pos.extend([None, None, None])
            self.threshold_lines_neg.extend([None, None, None])
            self.zero_lines.extend([zero_line, zero_line, zero_line])
            self._plot_ranges_dirty = True
            self._last_plot_range_refresh_ts = 0.0
            QTimer.singleShot(0, self.apply_plot_row_layout)
            return

        for i in range(num_channels):
            add_emg_plot(i, i, show_bottom=(i == num_channels - 1))
        self._plot_ranges_dirty = True
        self._last_plot_range_refresh_ts = 0.0
        QTimer.singleShot(0, self.apply_plot_row_layout)

    def apply_plot_row_layout(self):
        if not self.plot_rows:
            return

        try:
            grid = self.plot_widget.ci.layout
        except Exception:
            return

        if self.layout_has_imu() and any(row["kind"] == "imu_combined" for row in self.plot_rows):
            logical_row_count = WIRELESS_TOTAL_CHANNELS
            viewport_height = int(self.plot_widget.viewport().height()) - 8
            if viewport_height <= 0:
                return
            slot_height = max(42, int(viewport_height / logical_row_count))
            imu_height = slot_height * 3

            for row_index in range(logical_row_count):
                try:
                    grid.setRowStretchFactor(row_index, 1)
                    grid.setRowMinimumHeight(row_index, max(26, int(slot_height * 0.6)))
                    grid.setRowPreferredHeight(row_index, slot_height)
                except Exception:
                    pass

            for row in self.plot_rows:
                try:
                    if row["kind"] == "imu_combined":
                        row["plot"].setMinimumHeight(max(120, int(imu_height * 0.82)))
                    else:
                        row["plot"].setMinimumHeight(max(40, int(slot_height * 0.75)))
                except Exception:
                    pass
            return

        row_count = max(1, len(self.plot_rows))
        viewport_height = int(self.plot_widget.viewport().height()) - 8
        if viewport_height <= 0:
            return
        row_height = max(54, int(viewport_height / row_count))

        for row_index in range(row_count):
            try:
                grid.setRowStretchFactor(row_index, 1)
                grid.setRowMinimumHeight(row_index, max(26, int(row_height * 0.65)))
                grid.setRowPreferredHeight(row_index, row_height)
            except Exception:
                pass

        for row in self.plot_rows:
            try:
                row["plot"].setMinimumHeight(max(40, int(row_height * 0.8)))
            except Exception:
                pass

    @staticmethod
    def _format_axis_tick(value, magnitude_only=False):
        value = float(value)
        if abs(value) < 1e-6:
            return "0"
        tick_value = abs(value) if magnitude_only else value
        if abs(tick_value - round(tick_value)) < 0.05:
            return str(int(round(tick_value)))
        if abs(tick_value) >= 100.0:
            return f"{tick_value:.0f}"
        if abs(tick_value) >= 10.0:
            return f"{tick_value:.1f}"
        return f"{tick_value:.2f}"

    def _set_plot_axis_ticks(self, plot_item, tick_values, magnitude_only=False):
        ticks = []
        seen = set()
        for value in tick_values:
            if value is None or not np.isfinite(value):
                continue
            numeric = float(value)
            key = round(numeric, 6)
            if key in seen:
                continue
            seen.add(key)
            ticks.append((numeric, self._format_axis_tick(numeric, magnitude_only=magnitude_only)))

        if not ticks:
            return

        ticks.sort(key=lambda item: item[0])
        plot_item.getAxis("left").setTicks([ticks])

    def _set_fixed_plot_y_range(self, plot_item, is_imu=False):
        y_min = IMU_Y_MIN if is_imu else Y_MIN
        y_max = IMU_Y_MAX if is_imu else Y_MAX
        plot_item.disableAutoRange(axis="y")
        plot_item.setYRange(y_min, y_max, padding=0)
        self._set_plot_axis_ticks(plot_item, [y_min, 0.0, y_max], magnitude_only=not is_imu)

    @staticmethod
    def _center_emg_series_for_display(values):
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return arr.copy()

        centered = arr.copy()
        finite = centered[np.isfinite(centered)]
        if finite.size == 0:
            return centered

        centered -= float(np.median(finite))
        return centered

    def _emg_display_values(self, channel_index):
        if self.data_buffer is None:
            return None
        return self._center_emg_series_for_display(self.data_buffer[int(channel_index)])

    def _apply_live_emg_y_range(self, plot_item, values):
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            self._set_fixed_plot_y_range(plot_item, is_imu=False)
            return

        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            self._set_fixed_plot_y_range(plot_item, is_imu=False)
            return

        data_min = float(np.min(finite))
        data_max = float(np.max(finite))
        amplitude = max(abs(data_min), abs(data_max), LIVE_Y_MIN_HALF_RANGE)
        pad = max(1.5, amplitude * LIVE_Y_PADDING_RATIO)
        plot_half = amplitude + pad

        plot_item.disableAutoRange(axis="y")
        plot_item.setYRange(-plot_half, plot_half, padding=0)
        plot_item.getAxis("left").setTicks(
            [[
                (-plot_half, self._format_axis_tick(data_min, magnitude_only=True)),
                (0.0, "0"),
                (plot_half, self._format_axis_tick(data_max, magnitude_only=True)),
            ]]
        )

    def refresh_live_plot_ranges(self, plot_values_by_channel=None):
        if not self.plot_rows:
            return

        plot_values_by_channel = plot_values_by_channel or {}
        for row in self.plot_rows:
            if row["kind"] == "single":
                if not self.check_autoscale.isChecked() or self.data_buffer is None:
                    self._set_fixed_plot_y_range(row["plot"], is_imu=False)
                    continue

                ch = int(row["channel"])
                values = plot_values_by_channel.get(ch)
                if values is None:
                    values = self._emg_display_values(ch)
                if values is not None:
                    self._apply_live_emg_y_range(row["plot"], values)
                continue

            if row.get("is_imu"):
                if self.check_autoscale.isChecked():
                    row["plot"].enableAutoRange(axis="y")
                else:
                    self._set_fixed_plot_y_range(row["plot"], is_imu=True)

    def toggle_autoscale(self):
        if not self.plot_rows:
            return
        self._plot_ranges_dirty = True
        self.refresh_live_plot_ranges()
        self._last_plot_range_refresh_ts = time.perf_counter()

    def update_threshold_overlays(self, _value=None):
        threshold = self.threshold_spin.value()
        for line in self.threshold_lines_pos:
            if line is not None:
                line.setValue(threshold)
        for line in self.threshold_lines_neg:
            if line is not None:
                line.setValue(-threshold)

    def apply_channel_count_from_dialog(self, count):
        if self.layout_has_imu():
            QMessageBox.information(
                self,
                "Fixed Channel Layout",
                "Wi-Fi and USB (COM port) streaming use a fixed 11-channel layout: 8 EMG channels plus Roll, Pitch, and Yaw.",
            )
            self.channel_count = WIRELESS_TOTAL_CHANNELS
            self.sync_calibration_dialog_state()
            return

        new_count = int(max(2, min(11, count)))
        if not (self.is_connected and self.port_config_applied):
            QMessageBox.information(
                self,
                "Locked",
                "Complete Port Configuration first: connect the port and click Apply.",
            )
            return

        if new_count == int(self.channel_count):
            self.set_status(f"Channel count is already {new_count}. Ready for calibration.", "#607d8b")
            return

        self.channel_count = new_count
        self.wired_channel_count = new_count
        port_name = (self.current_port_name or "").strip()
        if not port_name:
            QMessageBox.warning(self, "No Port", "No active port found. Configure port first.")
            return

        self.stop_calibration_if_running()
        self.timer.stop()
        if self.serial_worker:
            try:
                self.serial_worker.batch_received.disconnect()
                self.serial_worker.error_occurred.disconnect()
            except Exception:
                pass
            self.serial_worker.stop()
            self.serial_worker = None

        self.reset_runtime_state_for_channels(self.channel_count)

        try:
            if self.wired_binary:
                self.packet_tracer.reset("wireless", WIRELESS_FRAMES_PER_PACKET)
            else:
                self.packet_tracer.reset("serial", 1)
            self.serial_worker = SerialWorker(port_name, DEFAULT_BAUD_RATE, self.num_channels,
                                              tracer=self.packet_tracer)
            self.serial_worker.batch_received.connect(self.on_serial_batch)
            self.serial_worker.error_occurred.connect(self.on_serial_error)
            self.serial_worker.start(QThread.HighestPriority)

            self.is_connected = True
            self.port_config_applied = True
            self.btn_port_config.setStyleSheet(self.connected_port_button_stylesheet())
            self.btn_disconnect.setEnabled(True)
            self.btn_analysis.setEnabled(False)
            self.set_status(
                f"Channel count applied: {self.channel_count}. Run calibration now.",
                "#ff9800",
            )
            if self.rf_model is not None:
                model_name = os.path.basename(self.rf_model_path) if self.rf_model_path else "Loaded"
                self.lbl_rf.setText(f"RF Model: {model_name} | Waiting calibrated stream...")
                self.lbl_rf.setStyleSheet(themed_label_style("success"))
            else:
                self.lbl_rf.setText("RF Model: Not loaded")
                self.lbl_rf.setStyleSheet(themed_label_style("muted"))
            self.sync_port_dialog_state()
            self.sync_calibration_dialog_state()
        except Exception as e:
            QMessageBox.critical(self, "Channel Reconfigure Error", str(e))
            self.disconnect_serial()

    def sync_calibration_dialog_state(self):
        if self.calibration_dialog and self.calibration_dialog.isVisible():
            self.calibration_dialog.set_channel_count(self.channel_count)
            self.calibration_dialog.set_calibration_seconds(self.cal_rest_seconds, self.cal_flex_seconds)
            self.calibration_dialog.set_port_state(self.is_connected, self.port_config_applied)
            self.calibration_dialog.set_calibrated_state(self.is_calibrated)

    def open_calibration_dialog(self):
        if self.calibration_dialog and self.calibration_dialog.isVisible():
            self.sync_calibration_dialog_state()
            self.calibration_dialog.raise_()
            self.calibration_dialog.activateWindow()
            return

        self.calibration_dialog = CalibrationDialog(
            channel_count=self.channel_count,
            rest_sec=self.cal_rest_seconds,
            flex_sec=self.cal_flex_seconds,
            is_connected=self.is_connected,
            is_port_applied=self.port_config_applied,
            parent=self,
        )
        self.calibration_dialog.start_requested.connect(self.start_calibration_sequence)
        self.calibration_dialog.cancel_requested.connect(self.cancel_calibration_sequence)
        self.calibration_dialog.channel_count_applied.connect(self.apply_channel_count_from_dialog)
        self.calibration_dialog.port_config_requested.connect(self.open_port_configuration_dialog)
        self.calibration_dialog.finished.connect(self.on_calibration_dialog_closed)
        self.calibration_dialog.set_calibrated_state(self.is_calibrated)
        center_window(self.calibration_dialog, self)
        self.calibration_dialog.show()

    def open_analysis_window(self):
        if not self.is_connected or not self.is_calibrated:
            QMessageBox.information(
                self,
                "Calibration Required",
                "Complete calibration first. Analysis window requires calibrated data.",
            )
            return

        if self.analysis_window is not None and self.analysis_window.num_channels != self.num_channels:
            self.analysis_window.close()
            self.analysis_window = None

        if self.analysis_window is None:
            self.analysis_window = AnalysisWindow(self.num_channels)
            self.analysis_window.closed.connect(self.on_analysis_window_closed)

        self.live_analysis_enabled = True
        center_window(self.analysis_window, self)
        self.analysis_window.show()
        self.analysis_window.raise_()
        self.analysis_window.activateWindow()

    def on_analysis_window_closed(self):
        self.live_analysis_enabled = False
        if self.is_connected and self.is_calibrated:
            self.set_analysis_idle_labels()

    def open_realtime_classification_window(self):
        if self.realtime_classification_window and self.realtime_classification_window.isVisible():
            self.realtime_classification_window.raise_()
            self.realtime_classification_window.activateWindow()
            return

        self.realtime_classification_window = RealtimeClassificationWindow(self)
        self.realtime_classification_window.closed.connect(self.on_realtime_classification_window_closed)
        center_window(self.realtime_classification_window, self)
        self.realtime_classification_window.show()
        self.realtime_classification_window.raise_()
        self.realtime_classification_window.activateWindow()

    def on_realtime_classification_window_closed(self):
        self.realtime_classification_window = None

    def expected_rf_feature_count(num_channels):
        n_ch = int(max(1, num_channels))
        per_channel_features = 15
        rms_ratio_features = n_ch
        corr_features = (n_ch * (n_ch - 1)) // 2
        return int((per_channel_features * n_ch) + rms_ratio_features + corr_features)

    @staticmethod
    def infer_rf_input_channels_from_feature_count(feature_count):
        try:
            target = int(feature_count)
        except Exception:
            return 0
        for n_ch in range(1, 65):
            if EMGVisualizer.expected_rf_feature_count(n_ch) == target:
                return n_ch
        return 0

    def set_rf_status_reason(self, reason, style_kind="success"):
        reason_text = str(reason or "").strip()
        self.rf_last_status_reason = reason_text
        if self.rf_model is None:
            self.lbl_rf.setText("RF Model: Not loaded")
            self.lbl_rf.setStyleSheet(themed_label_style("muted"))
            return
        model_name = os.path.basename(self.rf_model_path) if self.rf_model_path else "Loaded"
        if reason_text:
            self.lbl_rf.setText(f"RF Model: {model_name} | {reason_text}")
        else:
            self.lbl_rf.setText(
                f"RF Model: {model_name} | Pred: {self.rf_last_pred_label} ({self.rf_last_pred_conf * 100.0:.1f}%)"
            )
        self.lbl_rf.setStyleSheet(themed_label_style(style_kind))

    def request_realtime_prediction(self):
        if self.rf_model is None or self.rf_worker is None:
            self.rf_last_status_reason = "Model not loaded"
            return
        if self.task_session_active:
            self.set_rf_status_reason("Paused during data collection", "muted")
            return
        if not self.is_connected or not self.is_calibrated or self.data_buffer is None:
            self.set_rf_status_reason("Waiting calibrated stream...", "success")
            return
        model_ch = int(max(1, self.rf_model_input_channels))
        if self.data_buffer.shape[0] < model_ch:
            self.set_rf_status_reason(
                f"Incompatible channels (needs {model_ch}, stream has {self.data_buffer.shape[0]})",
                "muted",
            )
            return
        n_avail = int(min(self.rf_valid_sample_count, self.data_buffer.shape[1]))
        if n_avail < self.rf_window_samples:
            self.set_rf_status_reason(
                f"Buffering stream ({n_avail}/{self.rf_window_samples} samples)",
                "success",
            )
            return
        with self._buffer_lock:
            win = np.ascontiguousarray(self.data_buffer[:model_ch, -self.rf_window_samples:].T, dtype=np.float32)
        self.rf_last_status_reason = ""
        self.rf_inference_in_flight = True
        self.rf_last_submit_ts = time.perf_counter()
        self.rf_worker.submit_window(win)

    def schedule_realtime_prediction(self, new_samples):
        if self.task_session_active:
            self.rf_samples_since_submit = 0
            return
        self.rf_samples_since_submit += int(max(0, new_samples))
        stride = int(max(1, self.rf_stride_samples))
        if self.rf_samples_since_submit < stride:
            return
        self.rf_samples_since_submit = self.rf_samples_since_submit % stride

        # Don't stack work on the inference thread: skip while a prediction is in flight,
        # and never submit faster than the cadence cap regardless of a tiny stride.
        now = time.perf_counter()
        if self.rf_inference_in_flight:
            if (now - self.rf_last_submit_ts) < RF_INFLIGHT_TIMEOUT_S:
                return
            self.rf_inference_in_flight = False  # self-heal a dropped window
        if (now - self.rf_last_submit_ts) < RF_MIN_SUBMIT_INTERVAL_S:
            return
        self.request_realtime_prediction()

    def on_rf_prediction_ready(self, payload):
        self.rf_inference_in_flight = False
        data = dict(payload or {})
        pred_label = str(data.get("pred_label", "N/A")).strip() or "N/A"
        pred_conf = float(data.get("pred_conf", 0.0))
        raw_conf = data.get("class_confidences", [])
        conf_arr = np.asarray(raw_conf, dtype=np.float32)
        if conf_arr.size != len(self.rf_class_names):
            conf_arr = np.zeros(len(self.rf_class_names), dtype=np.float32)

        # Majority-vote smoothing keeps the displayed label stable at low cadence.
        self.rf_label_history.append(pred_label)
        smoothed_label = max(set(self.rf_label_history), key=self.rf_label_history.count)

        self.rf_last_pred_label = smoothed_label
        self.rf_last_pred_conf = float(np.clip(pred_conf, 0.0, 1.0))
        self.rf_last_class_confidences = conf_arr
        self.rf_last_latency_ms = float(max(0.0, data.get("latency_ms", 0.0)))
        self.rf_last_status_reason = ""

        now_ts = float(data.get("completed_ts", time.perf_counter()))
        if self.rf_last_prediction_ts > 0.0:
            dt = max(1e-6, now_ts - self.rf_last_prediction_ts)
            inst_rate = 1.0 / dt
            if self.rf_prediction_rate_hz > 0.0:
                self.rf_prediction_rate_hz = (0.70 * self.rf_prediction_rate_hz) + (0.30 * inst_rate)
            else:
                self.rf_prediction_rate_hz = inst_rate
        else:
            self.rf_prediction_rate_hz = 0.0
        self.rf_last_prediction_ts = now_ts
        self.gesture_bridge.publish(
            self.rf_last_pred_label, self.rf_last_pred_conf * 100.0,
            os.path.basename(self.rf_model_path) if self.rf_model_path else "",
        )

        if self.rf_model is not None:
            model_name = os.path.basename(self.rf_model_path) if self.rf_model_path else "Loaded"
            self.lbl_rf.setText(
                f"RF Model: {model_name} | Pred: {self.rf_last_pred_label} ({self.rf_last_pred_conf * 100.0:.1f}%)"
            )
            self.lbl_rf.setStyleSheet(themed_label_style("success"))

    def on_rf_worker_error(self, message):
        # Keep UI responsive on worker errors; next valid batch can recover state.
        self.rf_inference_in_flight = False
        self.rf_label_history.clear()
        self.rf_last_pred_label = "N/A"
        self.rf_last_pred_conf = 0.0
        self.rf_last_class_confidences = np.zeros(len(self.rf_class_names), dtype=np.float32)
        self.rf_last_latency_ms = 0.0
        self.rf_prediction_rate_hz = 0.0
        self.rf_last_prediction_ts = 0.0
        self.rf_samples_since_submit = 0
        detail = str(message or "Unknown RF worker error").strip()
        if len(detail) > 180:
            detail = detail[:177] + "..."
        self.set_rf_status_reason(f"Prediction error: {detail}", "muted")

    def get_realtime_classification_payload(self):
        model_loaded = self.rf_model is not None
        model_name = os.path.basename(self.rf_model_path) if self.rf_model_path else "Not loaded"
        model_path_text = self._to_project_relative_path(self.rf_model_path) if self.rf_model_path else "N/A"

        if model_loaded:
            if self.rf_last_status_reason:
                model_status_text = f"Model: {model_name} | {self.rf_last_status_reason}"
            else:
                model_status_text = (
                    f"Model: {model_name} | Pred: {self.rf_last_pred_label} ({self.rf_last_pred_conf * 100.0:.1f}%)"
                )
        else:
            model_status_text = "Model: Not loaded"

        classes = [str(x) for x in list(self.rf_class_names)]
        scores = np.asarray(self.rf_last_class_confidences, dtype=np.float32)
        if scores.size != len(classes):
            scores = np.zeros(len(classes), dtype=np.float32)

        meta_lines = [
            f"Model path: {model_path_text}",
            f"Created at: {self.rf_model_created_at_text}",
            f"Sample rate: {int(self.rf_model_sample_rate)} Hz",
            f"Window samples: {int(self.rf_window_samples)}",
            f"Stride samples: {int(self.rf_stride_samples)}",
            f"Input channels: {int(self.rf_model_input_channels)}",
            f"Buffered samples: {int(self.rf_valid_sample_count)}/{int(self.rf_window_samples)}",
            f"Classes: {len(classes)}",
        ]
        return {
            "model_loaded": model_loaded,
            "model_status_text": model_status_text,
            "meta_text": "\n".join(meta_lines),
            "pred_label": self.rf_last_pred_label,
            "pred_conf_pct": self.rf_last_pred_conf * 100.0,
            "classification_latency_ms": float(self.rf_last_latency_ms),
            "classification_rate_hz": float(self.rf_prediction_rate_hz),
            "classes": classes,
            "class_confidences_pct": (scores * 100.0).tolist(),
        }

    def update_realtime_classification_state(self):
        if self.rf_model is None:
            if self.rf_worker is not None:
                self.rf_worker.clear_model()
            self.rf_samples_since_submit = 0
            self.rf_last_pred_label = "N/A"
            self.rf_last_pred_conf = 0.0
            self.rf_last_class_confidences = np.zeros(len(self.rf_class_names), dtype=np.float32)
            self.rf_last_latency_ms = 0.0
            self.rf_prediction_rate_hz = 0.0
            self.rf_last_prediction_ts = 0.0
            self.rf_valid_sample_count = 0
            self.rf_last_status_reason = "Model not loaded"
            self.lbl_rf.setText("RF Model: Not loaded")
            self.lbl_rf.setStyleSheet(themed_label_style("muted"))
            return

        model_name = os.path.basename(self.rf_model_path) if self.rf_model_path else "Loaded"
        if self.data_buffer is not None and self.data_buffer.shape[0] < int(max(1, self.rf_model_input_channels)):
            self.lbl_rf.setText(
                f"RF Model: {model_name} | Incompatible channels "
                f"(needs {self.rf_model_input_channels}, stream has {self.data_buffer.shape[0]})"
            )
            self.lbl_rf.setStyleSheet(themed_label_style("muted"))
            self.rf_last_status_reason = (
                f"Incompatible channels (needs {self.rf_model_input_channels}, stream has {self.data_buffer.shape[0]})"
            )
            return
        if self.rf_last_status_reason:
            self.lbl_rf.setText(f"RF Model: {model_name} | {self.rf_last_status_reason}")
            self.lbl_rf.setStyleSheet(themed_label_style("success"))
            return
        self.lbl_rf.setText(
            f"RF Model: {model_name} | Pred: {self.rf_last_pred_label} ({self.rf_last_pred_conf * 100.0:.1f}%)"
        )
        self.lbl_rf.setStyleSheet(themed_label_style("success"))

    def load_rf_model_dialog(self):
        if not HAS_RF_FEATURES:
            QMessageBox.warning(
                self,
                "Missing Feature Module",
                "rf_features.py is required for RF inference and was not found.",
            )
            return

        start_dir = os.path.dirname(self.rf_model_path) if self.rf_model_path else TRAINED_MODEL_DIR
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select RF Model Artifact",
            start_dir,
            "Joblib Files (*.joblib);;All Files (*.*)",
        )
        if not path:
            return

        self.load_rf_model(path)

    def load_rf_model(self, path):
        try:
            artifact = joblib.load(path)
            self.apply_rf_model_artifact(artifact, path)
        except Exception as e:
            QMessageBox.critical(self, "RF Model Load Error", str(e))

    def apply_rf_model_artifact(self, artifact, path=""):
        try:
            if not isinstance(artifact, dict) or "model" not in artifact:
                raise ValueError("Artifact must be a dict with key 'model'.")

            model = artifact["model"]
            class_names = list(artifact.get("class_names", artifact.get("classes", [])))
            if not class_names and hasattr(model, "classes_"):
                class_names = [str(x) for x in list(getattr(model, "classes_", []))]

            artifact_input_channels = artifact.get("input_channels", None)
            model_feature_count = int(getattr(model, "n_features_in_", 0) or 0)
            if model_feature_count <= 0:
                try:
                    model_feature_count = int(len(artifact.get("feature_names", []) or []))
                except Exception:
                    model_feature_count = 0
            if artifact_input_channels is None and model_feature_count > 0:
                inferred_channels = self.infer_rf_input_channels_from_feature_count(model_feature_count)
                input_channels = inferred_channels if inferred_channels > 0 else 4
            else:
                try:
                    input_channels = int(artifact_input_channels if artifact_input_channels is not None else 4)
                except Exception:
                    input_channels = 4
                input_channels = int(max(1, input_channels))

            expected_features = self.expected_rf_feature_count(input_channels)
            if model_feature_count > 0 and model_feature_count != expected_features:
                inferred_channels = self.infer_rf_input_channels_from_feature_count(model_feature_count)
                if artifact_input_channels is None and inferred_channels > 0:
                    input_channels = inferred_channels
                    expected_features = self.expected_rf_feature_count(input_channels)
                else:
                    raise ValueError(
                        "Model feature count does not match artifact channel metadata. "
                        f"Model expects {model_feature_count} feature(s), but {input_channels} channel(s) "
                        f"produce {expected_features}. Load a matching model or retrain from the current dataset."
                    )

            self.rf_model = model
            self.rf_class_names = [str(x) for x in class_names]
            self.rf_window_samples = int(max(8, artifact.get("window_samples", 100)))
            self.rf_stride_samples = int(max(1, artifact.get("stride_samples", self.rf_window_samples)))
            self.rf_model_created_at_text = str(artifact.get("created_at_text", "N/A"))
            self.rf_model_sample_rate = int(artifact.get("sample_rate", SAMPLE_RATE))
            self.rf_model_input_channels = int(max(1, input_channels))
            self.rf_model_path = path
            self.rf_last_pred_label = "N/A"
            self.rf_last_pred_conf = 0.0
            self.rf_last_class_confidences = np.zeros(len(self.rf_class_names), dtype=np.float32)
            self.rf_last_latency_ms = 0.0
            self.rf_prediction_rate_hz = 0.0
            self.rf_last_prediction_ts = 0.0
            self.rf_samples_since_submit = 0
            self.rf_inference_in_flight = False
            self.rf_last_submit_ts = 0.0
            self.rf_label_history.clear()
            self.rf_last_status_reason = (
                f"Loaded | Ch: {self.rf_model_input_channels} | Window: {self.rf_window_samples} samples"
            )
            # Warm up the forest so the first real prediction isn't slowed by lazy init.
            self._warm_up_rf_model(model, input_channels)
            if self.rf_worker is not None:
                self.rf_worker.set_model(self.rf_model, self.rf_class_names, sample_rate=self.rf_model_sample_rate)
            self.request_realtime_prediction()

            if self.rf_last_status_reason:
                self.set_rf_status_reason(self.rf_last_status_reason, "success")
            else:
                self.set_rf_status_reason("Prediction queued...", "success")
        except Exception:
            raise

    def _warm_up_rf_model(self, model, input_channels):
        """Run one throwaway prediction so first live inference isn't slowed by lazy init."""
        if model is None or not HAS_RF_FEATURES or rf_extract_window_features is None:
            return
        try:
            n_ch = int(max(1, input_channels))
            win = np.zeros((int(max(8, self.rf_window_samples)), n_ch), dtype=np.float32)
            feats = rf_extract_window_features(win, sample_rate=self.rf_model_sample_rate).reshape(1, -1)
            if hasattr(model, "predict_proba"):
                model.predict_proba(feats)
            else:
                model.predict(feats)
        except Exception:
            pass

    def data_collection_settings_payload(self):
        return {
            "participant_id": self.participant_id,
            "blood_group": self.participant_blood_group,
            "height_cm": float(self.participant_height_cm),
            "weight_kg": float(self.participant_weight_kg),
            "age_years": int(self.participant_age_years),
            "forearm_circumference_cm": float(self.participant_forearm_circumference_cm),
            **{key: float(self.study_meta.get(key, 0.0)) for key in STUDY_ENV_KEYS},
            "session_day": int(self.study_meta["session_day"]),
            "hand_order": str(self.study_meta["hand_order"]),
            "max_cycles_per_block": int(self.study_meta["max_cycles_per_block"]),
            "block_break_s": float(self.study_meta["block_break_s"]),
            "hand_switch_s": float(self.study_meta["hand_switch_s"]),
            "seed": int(self.study_meta["seed"]),
            "practice_mode": bool(self.study_meta["practice_mode"]),
            "hand_order_source": str(self.study_meta["hand_order_source"]),
            "hand_order_default": str(self.study_meta["hand_order_default"]),
            "agreed": bool(self.contribution_agreed),
            "task_labels": list(self.task_labels),
            "labels_saved": bool(self.task_labels_locked),
            "repeats": int(self.task_repeats),
            "prep_s": float(self.task_prep_s),
            "hold_s": float(self.task_hold_s),
            "rest_s": float(self.task_rest_s),
            "record_rest": bool(self.task_record_rest),
            "settle_s": float(self.task_settle_s),
            "randomize_order": bool(self.task_randomize_order),
            "csv_dir": self.record_save_dir,
        }

    def update_data_collection_settings(self, settings):
        settings = dict(settings or {})
        self.participant_id = normalize_participant_id(settings.get("participant_id", self.participant_id))
        self.participant_blood_group = str(settings.get("blood_group", self.participant_blood_group)).strip()
        self.participant_height_cm = float(settings.get("height_cm", self.participant_height_cm) or 0.0)
        self.participant_weight_kg = float(settings.get("weight_kg", self.participant_weight_kg) or 0.0)
        self.participant_age_years = int(settings.get("age_years", self.participant_age_years) or 0)
        self.participant_forearm_circumference_cm = float(
            settings.get("forearm_circumference_cm", self.participant_forearm_circumference_cm) or 0.0
        )
        for key in STUDY_ENV_KEYS:
            self.study_meta[key] = float(settings.get(key, self.study_meta[key]) or 0.0)
        self.study_meta["session_day"] = int(settings.get("session_day", self.study_meta["session_day"]))
        self.study_meta["hand_order"] = str(settings.get("hand_order", self.study_meta["hand_order"]))
        self.study_meta["max_cycles_per_block"] = int(max(1, settings.get(
            "max_cycles_per_block", self.study_meta["max_cycles_per_block"])))
        self.study_meta["block_break_s"] = float(max(0.0, settings.get("block_break_s", self.study_meta["block_break_s"])))
        self.study_meta["hand_switch_s"] = float(max(0.0, settings.get("hand_switch_s", self.study_meta["hand_switch_s"])))
        for key, cast in (("seed", int), ("practice_mode", bool), ("hand_order_source", str), ("hand_order_default", str)):
            self.study_meta[key] = cast(settings.get(key, self.study_meta[key]))
        self.contribution_agreed = bool(settings.get("agreed", self.contribution_agreed))
        labels_in = settings.get("task_labels", self.task_labels)
        if isinstance(labels_in, list):
            labels = [str(x).strip() for x in labels_in if str(x).strip()]
        else:
            labels = [x.strip() for x in str(labels_in).split(",") if x.strip()]
        if len(labels) > 0:
            self.task_labels = labels
            self.task_labels_text = ",".join(labels)
        self.task_labels_locked = bool(settings.get("labels_saved", self.task_labels_locked))
        self.task_repeats = int(max(1, settings.get("repeats", self.task_repeats)))
        self.task_prep_s = float(max(0.2, settings.get("prep_s", self.task_prep_s)))
        self.task_hold_s = float(max(0.2, settings.get("hold_s", self.task_hold_s)))
        self.task_rest_s = float(max(0.2, settings.get("rest_s", self.task_rest_s)))
        self.task_record_rest = bool(settings.get("record_rest", self.task_record_rest))
        self.task_settle_s = float(max(0.0, settings.get("settle_s", self.task_settle_s)))
        self.task_randomize_order = bool(settings.get("randomize_order", self.task_randomize_order))
        csv_dir = str(settings.get("csv_dir", self.record_save_dir)).strip()
        if csv_dir:
            self.record_save_dir = csv_dir

    def open_data_collection_dialog(self):
        if self.data_collection_dialog and self.data_collection_dialog.isVisible():
            self.data_collection_dialog.raise_()
            self.data_collection_dialog.activateWindow()
            return

        self.data_collection_dialog = DataCollectionDialog(self.data_collection_settings_payload(), self)
        self.data_collection_dialog.start_requested.connect(self.on_data_collection_start_requested)
        self.data_collection_dialog.settings_changed.connect(self.update_data_collection_settings)
        self.data_collection_dialog.protocol_phase_started.connect(self.on_task_phase_started)
        self.data_collection_dialog.protocol_finished.connect(self.on_task_protocol_finished)
        self.data_collection_dialog.protocol_canceled.connect(self.on_task_protocol_canceled)
        self.data_collection_dialog.protocol_paused.connect(self.on_task_protocol_paused)
        self.data_collection_dialog.protocol_rewind.connect(self.on_task_protocol_rewind)
        self.data_collection_dialog.finished.connect(lambda _result: setattr(self, "data_collection_dialog", None))
        center_window(self.data_collection_dialog, self)
        self.data_collection_dialog.show()
        self.data_collection_dialog.raise_()
        self.data_collection_dialog.activateWindow()

    def on_data_collection_start_requested(self, settings):
        self.update_data_collection_settings(settings)
        self.start_task_timer_recording()

    def parse_task_labels(self, raw_text=None):
        if raw_text is None:
            labels = [x.strip() for x in self.task_labels if str(x).strip()]
            return labels
        raw = str(raw_text).strip()
        labels = [x.strip() for x in raw.split(",") if x.strip()]
        if len(labels) == 0:
            return []
        return labels

    @staticmethod
    def _sanitize_filename_token(text):
        safe = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(text or ""))
        safe = safe.strip("_")
        return safe or "anonymous"

    def build_auto_record_csv_path(self):
        save_dir = (self.record_save_dir or DATASET_DIR).strip() or DATASET_DIR
        bundle_name = session_folder_name(self.participant_id, self.study_meta["session_day"],
                                          self.study_meta["hand_order"], practice=self.study_meta["practice_mode"])
        return os.path.join(save_dir, bundle_name, f"signals_{bundle_name}.csv")

    @staticmethod
    def _resolve_record_output_paths(path):
        target_path = str(path or "").strip()
        if not target_path:
            raise ValueError("Save path is empty.")

        base_dir = os.path.dirname(target_path) or "."
        stem = os.path.splitext(os.path.basename(target_path))[0].strip()
        if stem.startswith("signals_"):
            bundle_name = stem[len("signals_"):].strip()
        elif stem.startswith("emg_data_"):
            bundle_name = stem[len("emg_data_"):].strip()
        elif stem.startswith("metadata_"):
            bundle_name = stem[len("metadata_"):].strip()
        else:
            bundle_name = stem
        bundle_name = EMGVisualizer._sanitize_filename_token(bundle_name)

        if os.path.basename(os.path.normpath(base_dir)) == bundle_name:
            bundle_dir = base_dir
            root_dir = os.path.dirname(base_dir) or base_dir
        else:
            bundle_dir = os.path.join(base_dir, bundle_name)
            root_dir = base_dir

        data_csv_path = os.path.join(bundle_dir, f"signals_{bundle_name}.csv")
        metadata_path = os.path.join(bundle_dir, f"metadata_{bundle_name}.txt")
        return data_csv_path, metadata_path, root_dir

    @staticmethod
    def _channel_columns_from_fieldnames(fieldnames):
        cols = []
        for raw_name in list(fieldnames or []):
            name = str(raw_name or "").strip()
            low = name.lower()
            if low.startswith("ch") and low[2:].isdigit():
                cols.append((int(low[2:]), name))
        cols.sort(key=lambda x: x[0])
        return [name for _idx, name in cols]

    @staticmethod
    def _record_channel_columns():
        return [f"Ch{i}" for i in range(1, WIRELESS_TOTAL_CHANNELS + 1)]

    @staticmethod
    def _channel_columns_from_rows(rows):
        seen = set()
        cols = []
        for row in list(rows or []):
            if not isinstance(row, dict):
                continue
            for key in row.keys():
                name = str(key or "").strip()
                low = name.lower()
                if low.startswith("ch") and low[2:].isdigit():
                    ch_idx = int(low[2:])
                    if ch_idx not in seen:
                        seen.add(ch_idx)
                        cols.append((ch_idx, f"Ch{ch_idx}"))
        cols.sort(key=lambda x: x[0])
        return [name for _idx, name in cols]

    @classmethod
    def _active_channel_columns_from_csv(cls, path):
        channel_cols = []
        active = set()
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV has no header.")
            channel_cols = cls._channel_columns_from_fieldnames(reader.fieldnames)
            for row in reader:
                for ch_name in channel_cols:
                    raw_val = row.get(ch_name, "")
                    if str(raw_val).strip() != "":
                        active.add(ch_name)
        if not channel_cols:
            return []
        if not active:
            return list(channel_cols)
        return [name for name in channel_cols if name in active]

    @classmethod
    def detect_training_csv_channel_count(cls, path):
        csv_path = cls._from_project_relative_path(path)
        if not csv_path or not os.path.isfile(csv_path):
            raise ValueError(f"Dataset CSV not found: {path}")
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV has no header.")
            field_map = {str(name).strip().lower(): str(name).strip() for name in list(reader.fieldnames)}
            if "label" not in field_map:
                raise ValueError("CSV missing required column: Label")
            channel_cols = cls._active_channel_columns_from_csv(csv_path)
            if len(channel_cols) == 0:
                raise ValueError("CSV must contain channel columns (Ch1..ChN)")
            return int(len(channel_cols))

    def recorded_sample_count(self):
        """Frames written so far by the continuous recorder (valid and gap frames)."""
        rec = self.session_recorder
        return int(rec.rows) if rec is not None else int(self._last_session_rows)

    def build_recorded_metadata_lines(self, data_csv_path, root_dir, sample_count, quality=None):
        labels = [str(x).strip() for x in self.task_labels if str(x).strip()]
        root_abs = os.path.abspath(root_dir or ".")
        data_abs = os.path.abspath(data_csv_path)
        try:
            data_rel = os.path.relpath(data_abs, root_abs).replace("\\", "/")
        except ValueError:
            data_rel = os.path.basename(data_csv_path)
        valid_count = int((quality or {}).get("rows_valid", sample_count))
        sample_eval = self.evaluate_recording_sample_coverage(valid_count)
        snap = self.packet_tracer.snapshot()
        return [
            f"created_at={time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"participant_id={self.participant_id}",
            f"blood_group={self.participant_blood_group or 'N/A'}",
            f"height_cm={_value_or_na(self.participant_height_cm, '{:.1f}')}",
            f"weight_kg={_value_or_na(self.participant_weight_kg, '{:.1f}')}",
            f"age_years={_value_or_na(self.participant_age_years, '{:d}', cast=int)}",
            f"forearm_circumference_cm={_value_or_na(self.participant_forearm_circumference_cm, '{:.2f}')}",
            f"session_day={int(self.study_meta['session_day'])}",
            f"hand_order={'+'.join(hands_for_order(self.study_meta['hand_order'])).lower()}",
            f"max_cycles_per_block={int(self.study_meta['max_cycles_per_block'])}",
            f"block_break_s={float(self.study_meta['block_break_s']):.0f}",
            f"hand_switch_break_s={float(self.study_meta['hand_switch_s']):.0f}",
            f"temp_start_c={_value_or_na(self.study_meta['temp_start_c'], '{:.1f}')}",
            f"humidity_start_pct={_value_or_na(self.study_meta['humidity_start_pct'], '{:.0f}')}",
            f"consent_form_version={CONSENT_FORM_VERSION if self.contribution_agreed else 'NOT_CONSENTED'}",
            f"labels={','.join(labels)}",
            "label_codes=" + ",".join(f"{name}:{code}" for name, code in sorted(LABEL_CODES.items(), key=lambda kv: kv[1])),
            f"labels_saved={bool(self.task_labels_locked)}",
            f"repeats={int(self.task_repeats)}",
            f"prep_s={float(self.task_prep_s)}",
            f"hold_s={float(self.task_hold_s)}",
            f"rest_s={float(self.task_rest_s)}",
            f"record_rest={bool(self.task_record_rest)}",
            f"settle_s={float(self.task_settle_s)}",
            f"randomize_order={bool(self.task_randomize_order or not self.study_meta['practice_mode'])}",
            f"random_seed={int(self.study_meta['seed'])}",
            f"practice_mode={bool(self.study_meta['practice_mode'])}",
            f"hand_order_source={self.study_meta['hand_order_source']}",
            f"hand_order_counterbalanced_default={self.study_meta['hand_order_default'] or 'N/A'}",
            f"channel_count={int(self.num_channels)}",
            f"connection_medium={self.connection_medium}",
            f"sample_rate_hz={int(SAMPLE_RATE)}",
            f"num_samples={int(sample_count)}",
            f"expected_num_samples={int(self.expected_recorded_samples())}",
            f"sample_coverage_pct={sample_eval['sample_ratio_pct']:.2f}",
            f"transport={snap['kind']}",
            f"packets_received={snap['received']}",
            f"packets_missing={snap['missing']}",
            f"packets_missing_during_recording={int(self.record_missing_packets)}",
            f"packet_gap_events_during_recording={int(self.record_gap_events)}",
            f"packets_out_of_order={snap['out_of_order']}",
            f"packets_invalid={snap['invalid']}",
            f"packet_loss_pct={snap['loss_pct']:.3f}",
            f"max_packet_gap_ms={snap['max_gap_ms']:.1f}",
            f"csv_signal_mode=raw",
            f"csv_columns={','.join(SIGNAL_COLUMNS)}",
            f"num_valid_samples={valid_count}",
            f"events_file=events_{os.path.basename(os.path.dirname(data_csv_path))}.csv",
            f"quality_file=quality_{os.path.basename(os.path.dirname(data_csv_path))}.json",
            f"session_start_utc={self._session_start_utc}",
            f"data_file={os.path.basename(data_csv_path)}",
            f"data_path={data_rel}",
        ]

    def save_recorded_metadata_txt(self, path, data_csv_path, root_dir, sample_count=None):
        lines = self.build_recorded_metadata_lines(
            data_csv_path,
            root_dir,
            self.recorded_sample_count() if sample_count is None else int(sample_count),
        )
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _set_task_collection_mode(self, active):
        active = bool(active)
        if active:
            if not self.task_session_active:
                self._record_graph_stream_prev_checked = bool(self.check_graph_stream.isChecked())
                self._record_live_analysis_prev_enabled = bool(self.live_analysis_enabled)
            self.task_session_active = True
            self.check_graph_stream.setChecked(False)
            self.live_analysis_enabled = False
            self.btn_analysis.setEnabled(False)
            self.rf_samples_since_submit = 0
            if self.rf_model is not None:
                self.set_rf_status_reason("Paused during data collection", "muted")
            if self.analysis_window and self.analysis_window.isVisible():
                self.set_analysis_idle_labels()
            return

        was_active = self.task_session_active
        self.task_session_active = False
        self.rf_samples_since_submit = 0
        if was_active:
            self.check_graph_stream.setChecked(bool(self._record_graph_stream_prev_checked))
            self.live_analysis_enabled = bool(self._record_live_analysis_prev_enabled)
        self.btn_analysis.setEnabled(self.is_connected and self.is_calibrated)
        if self.rf_model is not None:
            self.request_realtime_prediction()
            if not self.rf_last_status_reason:
                self.set_rf_status_reason("Prediction queued...", "success")

    def _on_record_save_completed(self, worker, context, result):
        if worker in self.record_save_workers:
            self.record_save_workers.remove(worker)
        try:
            worker.wait(50)
        except Exception:
            pass
        worker.deleteLater()

        payload = dict(result or {})
        if context.get("checkpoint"):
            # Mid-session safety copy: stay silent unless it failed.
            if not payload.get("ok", False):
                self.set_status(f"Break checkpoint could not be saved: {payload.get('error', 'unknown error')}", "#f57c00")
            return
        if not payload.get("ok", False):
            self.set_status("Task finished but the recording could not be finalised.", "#f44336")
            QMessageBox.warning(self, "Recording Save", f"The recording could not be finalised:\n{payload.get('error', 'Unknown error')}\n\n"
                                "The raw frames were still written to disk (raw_*.bin in the session folder).")
            return

        data_csv_path = str(payload.get("data_csv_path", "")).strip()
        self.last_recorded_csv_path = data_csv_path or self.last_recorded_csv_path
        self.record_save_dir = str(context.get("root_dir", self.record_save_dir) or self.record_save_dir)
        if self.data_collection_dialog and self.data_collection_dialog.isVisible():
            self.data_collection_dialog.set_csv_dir(self.record_save_dir)
        if context.get("show_saved_dialog"):
            QMessageBox.information(self, "Saved", f"Recording saved:\\n{data_csv_path}")

        quality = payload.get("quality") or {}
        self._last_session_quality = quality
        ratio = float(quality.get("valid_sample_ratio", 0.0))
        summary = (f"Data collection session complete | Frames: {quality.get('rows_valid', 0):,} valid of "
                   f"{quality.get('rows_total', 0):,} ({100 * ratio:.2f}% valid) | "
                   f"Packets missing: {quality.get('transport', {}).get('packets_missing', 0):,}")
        if not quality.get("meets_valid_sample_threshold", False):
            self.set_status(summary, "#f57c00")
            QMessageBox.warning(
                self, "Low Valid-Sample Ratio",
                (f"Only {100 * ratio:.2f}% of the recorded frames are valid; the minimum for a usable session is "
                 f"{100 * MIN_RECORD_SAMPLE_RATIO:.0f}%.\\n\\nMissing frames are kept as explicit Valid=0 rows. "
                 f"Gaps: {len(quality.get('gaps', []))}. See quality_*.json next to the signal file.\\n\\n"
                 "Consider re-recording this session."))
        else:
            self.set_status(summary, "#2e7d32")

    def _transport_summary(self):
        snap = self.packet_tracer.snapshot()
        return {"kind": snap["kind"], "packets_received": snap["received"], "packets_missing": snap["missing"],
                "packets_late": snap.get("late", 0), "packets_duplicate": snap.get("duplicate", 0),
                "packets_out_of_order": snap["out_of_order"], "packets_invalid": snap["invalid"],
                "frames_missing": snap.get("frames_missing", 0), "loss_pct_packets": snap["loss_pct"],
                "max_packet_gap_ms": snap["max_gap_ms"], "jitter_ms": snap["jitter_ms"]}

    def queue_session_finalize(self, close_recorder, checkpoint=False, show_saved_dialog=False):
        """Hand the recorder to a background worker that writes the labelled files."""
        rec = self.session_recorder
        if rec is None or self.session_events is None:
            raise ValueError("No recording session is active.")
        data_csv_path, metadata_path, root_dir = self._resolve_record_output_paths(self._session_csv_path)
        events = [dict(e) for e in self.session_events.events]
        builder = lambda rows, quality, d=data_csv_path, r=root_dir: self.build_recorded_metadata_lines(d, r, rows, quality)
        worker = SessionFinalizeWorker(
            rec, events, self._session_dir, self._session_name, SAMPLE_RATE, self._transport_summary(),
            list(self.keepalive_restarts), self._session_host_origin, metadata_path, builder, close_recorder, self)
        context = {"root_dir": root_dir, "checkpoint": bool(checkpoint), "show_saved_dialog": bool(show_saved_dialog)}
        worker.save_completed.connect(lambda result, w=worker, ctx=context: self._on_record_save_completed(w, ctx, result))
        self.record_save_workers.append(worker)
        if close_recorder:
            self._last_session_rows = int(rec.rows)
            self.session_recorder = None       # nothing more is appended; the worker drains and closes it
        worker.start(QThread.LowPriority)
        return data_csv_path, metadata_path

    def start_task_timer_recording(self):
        if not self.data_collection_dialog or not self.data_collection_dialog.isVisible():
            QMessageBox.warning(self, "Data Collection", "Open Data Collection to start a task session.")
            return
        if not self.is_connected or not self.is_calibrated:
            QMessageBox.warning(
                self,
                "Calibration Required",
                "Connect and complete calibration before timed recording.",
            )
            return
        if not self.contribution_agreed:
            QMessageBox.warning(self, "Agreement Required", "Open Data Collection and agree to contribute first.")
            return
        if not valid_participant_id(self.participant_id):
            QMessageBox.warning(self, "Participant ID", "Provide a pseudonymous participant ID (S01, S02, ...) in Data Collection before starting.")
            return
        if not self.task_labels_locked:
            QMessageBox.warning(self, "Task Labels", "Save task labels in Data Collection before starting.")
            return

        labels = self.parse_task_labels()
        if len(labels) == 0:
            QMessageBox.warning(self, "Task Labels", "Provide at least one gesture label.")
            return

        self._session_csv_path = self.build_auto_record_csv_path()
        data_csv_path, _metadata_path, _root = self._resolve_record_output_paths(self._session_csv_path)
        self._session_dir = os.path.dirname(data_csv_path)
        self._session_name = os.path.basename(self._session_dir)
        try:
            self.session_recorder = ContinuousRecorder(
                os.path.join(self._session_dir, f"raw_{self._session_name}.bin"), SAMPLE_RATE)
        except OSError as exc:
            QMessageBox.warning(self, "Recording", f"Cannot create the session files:\\n{exc}")
            return
        self.session_events = EventLog()
        self._last_session_rows = 0
        self._last_session_quality = None
        self._cycle_event_start = 0
        self.record_missing_packets = 0
        self.record_gap_events = 0
        self.keepalive_restarts = []
        self._session_host_origin = time.monotonic()                 # all host times are monotonic
        self._session_start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())   # metadata only

        started = self.data_collection_dialog.start_task_protocol(
            labels=labels,
            repeats=self.task_repeats,
            prep_s=self.task_prep_s,
            hold_s=self.task_hold_s,
            rest_s=self.task_rest_s,
            record_rest=self.task_record_rest,
            settle_s=self.task_settle_s,
            randomize_order=True if not self.study_meta["practice_mode"] else self.task_randomize_order,
            seed=self.study_meta["seed"],
            hand_order=self.study_meta["hand_order"],
            session_day=self.study_meta["session_day"],
            max_cycles_per_block=self.study_meta["max_cycles_per_block"],
            block_break_s=self.study_meta["block_break_s"],
            hand_switch_s=self.study_meta["hand_switch_s"],
        )
        if not started:
            rec, self.session_recorder, self.session_events = self.session_recorder, None, None
            rec.close()
            QMessageBox.warning(self, "Task Timer", "Unable to start task session.")
            return
        self._session_total_ms = int(self.data_collection_dialog.task_protocol.session_total_ms)
        self._set_task_collection_mode(True)
        self.set_status("Task Timer: session started | recording every frame continuously", "#00897b")

    def on_task_phase_started(self, label, trial_id, phase_name, record_enabled, hand="", block=0, cycle_no=0, planned_ms=0):
        if self.session_events is not None:
            idx = self.session_events.add(phase_name, label, hand, trial_id, block, cycle_no, time.monotonic(), planned_ms)
            if str(phase_name).strip().lower() == "prepare":
                self._cycle_event_start = idx
        if str(phase_name).strip().lower() in ("break", "switch hand") and self.recorded_sample_count() > 0:
            # Protect the data collected so far: a 60-minute session should not hinge on one final save.
            try:
                self.queue_session_finalize(close_recorder=False, checkpoint=True)
            except Exception:
                pass
        self.set_status(
            f"Task Timer: {hand} hand | Trial {trial_id} | {phase_name} | Label={label} | RECORDING",
            "#00897b",
        )

    def on_task_protocol_finished(self):
        self._set_task_collection_mode(False)
        if self.session_events is not None:
            self.session_events.add("end", "end", "", 0, 0, 0, time.monotonic(), 0)
        self.refresh_csv_counter()
        for worker in list(self.record_save_workers):
            worker.wait()   # let an in-flight break checkpoint finish before the final write
        try:
            self.queue_session_finalize(close_recorder=True)
            self.set_status("Task complete - writing signals, events and quality report in background...", "#00897b")
        except Exception as e:
            QMessageBox.warning(self, "Recording Save", f"Task finished but saving failed:\\n{e}")

    def on_task_protocol_paused(self, paused):
        if paused:
            if self.session_events is not None:
                last = self.session_events.events[-1] if len(self.session_events) else None
                self.session_events.add("paused", "paused", last["hand"] if last else "", last["trial_id"] if last else 0,
                                        last["block"] if last else 0, last["cycle_no"] if last else 0,
                                        time.monotonic(), 0)
            self.set_status("Task Timer: PAUSED - frames keep being recorded, labelled 'paused'", "#f57c00")

    def on_task_protocol_rewind(self):
        """Resume after a pause: the interrupted cycle is kept but relabelled 'aborted'; it restarts from Prepare."""
        if self.session_events is not None:
            self.session_events.mark_aborted_since(self._cycle_event_start)

    def on_task_protocol_canceled(self):
        self._set_task_collection_mode(False)
        self.set_status("Task timer canceled.", "#f57c00")
        rec = self.session_recorder
        rows = self.recorded_sample_count()
        if rec is None:
            return
        if self.session_events is not None:
            self.session_events.add("end", "end", "", 0, 0, 0, time.monotonic(), 0)
        if rows > 0:
            answer = QMessageBox.question(
                self, "Save Partial Recording",
                f"The session was cancelled after {rows:,} frames were recorded.\\nWrite them as a partial recording?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if answer == QMessageBox.Yes:
                try:
                    self.queue_session_finalize(close_recorder=True, show_saved_dialog=True)
                    return
                except Exception as e:
                    QMessageBox.warning(self, "Recording Save", f"Partial save failed:\\n{e}")
        # Declined (or failed): the raw frames stay on disk, nothing is deleted.
        self._last_session_rows = int(rec.rows)
        self.session_recorder = None
        rec.close()

    def expected_recorded_samples(self):
        """Frames the whole protocol should produce: every phase is recorded now, breaks included."""
        return int(round(self._session_total_ms / 1000.0 * float(SAMPLE_RATE)))

    def evaluate_recording_sample_coverage(self, actual_samples):
        expected = int(max(0, self.expected_recorded_samples()))
        actual = int(max(0, actual_samples))
        ratio = (float(actual) / float(expected)) if expected > 0 else 0.0
        return {
            "expected_samples": expected,
            "actual_samples": actual,
            "sample_ratio": ratio,
            "sample_ratio_pct": ratio * 100.0,
            "effective_hz": ratio * float(SAMPLE_RATE),
            "below_threshold": bool(expected > 0 and ratio < float(MIN_RECORD_SAMPLE_RATIO)),
        }

    @classmethod
    def load_segments_from_record_csv(cls, path):
        segments = []
        csv_path = cls._from_project_relative_path(path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV has no header.")
            field_map = {str(name).strip().lower(): str(name).strip() for name in list(reader.fieldnames)}
            label_col = field_map.get("label", "")
            trial_col = field_map.get("trial_id", "")
            phase_col = field_map.get("phase", "")
            valid_col = field_map.get("valid", "")
            channel_cols = cls._active_channel_columns_from_csv(csv_path)
            if not label_col:
                raise ValueError("CSV missing required column: Label")
            if len(channel_cols) == 0:
                raise ValueError("CSV must contain at least one channel column (Ch1..ChN).")

            current_key = None
            current_label = None
            buf = []
            for row in reader:
                label = str(row.get(label_col, "")).strip()
                if not label:
                    continue
                phase = str(row.get(phase_col, "")).strip() if phase_col else ""
                invalid = bool(valid_col) and str(row.get(valid_col, "1")).strip() == "0"
                # Continuous recordings also contain prepare/settle/break/pause/aborted frames and
                # NaN gap frames: only task (perform) and rest frames are training data, and a gap
                # ends the current segment instead of being bridged.
                if invalid or (phase and phase.lower() not in ("perform", "rest")):
                    if len(buf) > 0 and current_label:
                        segments.append((current_label, np.asarray(buf, dtype=np.float32)))
                    buf, current_key, current_label = [], None, None
                    continue
                trial = str(row.get(trial_col, "1")).strip() if trial_col else "1"
                key = (label, trial, phase)
                try:
                    sample = []
                    for ch_name in channel_cols:
                        raw_val = str(row.get(ch_name, "")).strip()
                        sample.append(float(raw_val) if raw_val != "" else 0.0)
                except Exception:
                    continue

                if current_key is None:
                    current_key = key
                    current_label = label
                if key != current_key:
                    if len(buf) > 0 and current_label:
                        segments.append((current_label, np.asarray(buf, dtype=np.float32)))
                    buf = []
                    current_key = key
                    current_label = label

                buf.append(sample)

            if len(buf) > 0 and current_label:
                segments.append((current_label, np.asarray(buf, dtype=np.float32)))
        return segments, int(len(channel_cols))

    def train_rf_from_app(self):
        self.open_rf_training_dialog()

    def open_rf_training_dialog(self):
        if self.training_dialog and self.training_dialog.isVisible():
            self.training_dialog.raise_()
            self.training_dialog.activateWindow()
            return

        setup = self.default_rf_training_setup()
        self.training_dialog = RFTrainingDialog(self, self)
        self.training_dialog.apply_default_values(setup)
        self.training_dialog.finished.connect(lambda _result: setattr(self, "training_dialog", None))
        center_window(self.training_dialog, self)
        self.training_dialog.show()
        self.training_dialog.raise_()
        self.training_dialog.activateWindow()

    def default_rf_training_setup(self):
        return {
            "dataset_paths": [],
            "output_dir": "trained_model",
            "run_name": "rf_training",
            "window_ms": int(max(20, self.analysis_ms_spin.value())),
            "stride_ms": 50,
            "n_estimators": RF_DEFAULT_N_ESTIMATORS,
            "max_depth": RF_DEFAULT_MAX_DEPTH,
            "random_seed": 42,
            "test_size": 0.20,
            "class_weight_balanced": True,
            "auto_load_model": True,
        }

    @staticmethod
    def _unique_run_dir(output_dir, run_name):
        output_root = os.path.abspath(str(output_dir or TRAINED_MODEL_DIR).strip() or TRAINED_MODEL_DIR)
        safe_name = EMGVisualizer._sanitize_filename_token(run_name or "rf_training")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base_folder = f"{safe_name}_{stamp}"
        run_dir = os.path.join(output_root, base_folder)
        suffix = 2
        while os.path.exists(run_dir):
            run_dir = os.path.join(output_root, f"{base_folder}_{suffix:02d}")
            suffix += 1
        return run_dir

    @staticmethod
    def _project_root_dir():
        return PROJECT_ROOT

    @classmethod
    def _to_project_relative_path(cls, path):
        p = os.path.abspath(str(path or "").strip())
        if not p:
            return ""
        root = cls._project_root_dir()
        try:
            rel = os.path.relpath(p, root)
        except ValueError:
            return p
        if rel.startswith(".."):
            return p
        return rel.replace("\\", "/")

    @classmethod
    def _from_project_relative_path(cls, path):
        p = str(path or "").strip()
        if not p:
            return ""
        if os.path.isabs(p):
            return os.path.abspath(p)
        return os.path.abspath(os.path.join(cls._project_root_dir(), p))

    @staticmethod
    def _normalize_training_csv_paths(paths):
        normalized = []
        seen = set()
        for path in list(paths or []):
            p = EMGVisualizer._from_project_relative_path(path)
            if not p:
                continue
            if not os.path.isfile(p):
                raise ValueError(f"Dataset CSV not found:\n{p}")
            key = os.path.normcase(p)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(p)
        return normalized

    def train_rf_with_config(self, config):
        if not HAS_RF_FEATURES:
            raise ValueError("rf_features.py is required for training.")
        if not HAS_SKLEARN:
            raise ValueError("scikit-learn is required. Install with: py -3 -m pip install scikit-learn")

        cfg = dict(config or {})
        csv_paths = self._normalize_training_csv_paths(cfg.get("dataset_paths", []))
        if len(csv_paths) == 0:
            raise ValueError("No dataset CSV files selected.")
        status_callback = cfg.get("_status_callback", None)
        default_window_ms = DEFAULT_ANALYSIS_MS if callable(status_callback) else self.analysis_ms_spin.value()

        output_dir = str(cfg.get("output_dir", TRAINED_MODEL_DIR)).strip() or TRAINED_MODEL_DIR
        output_dir_abs = self._from_project_relative_path(output_dir)
        run_name = self._sanitize_filename_token(cfg.get("run_name", "rf_training"))
        window_ms = int(max(20, cfg.get("window_ms", default_window_ms)))
        stride_ms = int(max(5, cfg.get("stride_ms", 50)))
        n_estimators = int(max(50, cfg.get("n_estimators", RF_DEFAULT_N_ESTIMATORS)))
        max_depth_value = int(max(0, cfg.get("max_depth", RF_DEFAULT_MAX_DEPTH)))
        test_size = float(min(0.5, max(0.05, cfg.get("test_size", 0.2))))
        random_seed = int(max(0, cfg.get("random_seed", 42)))
        class_weight_balanced = bool(cfg.get("class_weight_balanced", True))
        auto_load_model = bool(cfg.get("auto_load_model", True))
        defer_auto_load = bool(cfg.get("_defer_auto_load", False))

        def emit_training_status(text, color="#6a1b9a"):
            if callable(status_callback):
                status_callback(text, color)
            else:
                self.set_status(text, color)
                QApplication.processEvents()

        emit_training_status("RF training: loading dataset CSV files...")
        segments = []
        segment_count_by_file = {}
        channel_count_by_file = {}
        for csv_path in csv_paths:
            file_segments, file_channel_count = self.load_segments_from_record_csv(csv_path)
            segment_count_by_file[csv_path] = int(len(file_segments))
            channel_count_by_file[csv_path] = int(file_channel_count)
            segments.extend(file_segments)

        if len(segments) == 0:
            raise ValueError("No segments found across selected datasets.")

        valid_channel_counts = [c for c in channel_count_by_file.values() if int(c) > 0]
        if len(valid_channel_counts) == 0:
            raise ValueError("No channel columns were detected in selected datasets.")
        unique_channel_counts = sorted(set(int(c) for c in valid_channel_counts))
        if len(unique_channel_counts) != 1:
            details = ", ".join(
                [
                    f"{os.path.basename(path)}:{int(count)}"
                    for path, count in list(channel_count_by_file.items())[:8]
                ]
            )
            if len(channel_count_by_file) > 8:
                details = f"{details}, +{len(channel_count_by_file) - 8} more"
            raise ValueError(
                "Channel count mismatch across selected datasets. "
                "All files must have the same channel count.\n"
                f"Detected: {details}"
            )
        target_channels = int(unique_channel_counts[0])
        if target_channels <= 0:
            raise ValueError("Invalid dataset channel configuration.")

        win_samples = max(8, int((window_ms / 1000.0) * SAMPLE_RATE))
        stride_samples = max(1, int((stride_ms / 1000.0) * SAMPLE_RATE))

        emit_training_status("RF training: extracting features...")
        windows_all = []
        y = []
        groups = []  # one id per source segment, so overlapping windows never split across sets
        skipped_channel_segments = 0
        mismatched_segments = 0
        seg_id = 0
        for label, seq in segments:
            arr = np.asarray(seq, dtype=np.float32)
            if arr.ndim != 2:
                skipped_channel_segments += 1
                continue
            if arr.shape[1] != target_channels:
                mismatched_segments += 1
                continue
            windows = rf_build_windows_from_sequence(arr, win_samples, stride_samples)
            for w in windows:
                windows_all.append(w)
                y.append(label)
                groups.append(seg_id)
            seg_id += 1

        if mismatched_segments > 0:
            raise ValueError(
                "Channel count mismatch detected inside loaded segments. "
                f"Expected {target_channels} channel(s), found {mismatched_segments} mismatched segment(s)."
            )

        if len(windows_all) < 20:
            raise ValueError(
                f"Too few windows for training ({len(windows_all)}). "
                f"Add more data or reduce window size. "
                f"Using {target_channels} channel(s)."
            )

        # Feature extraction is embarrassingly parallel across windows. Use threads: the
        # extractor is numpy/FFT-bound (releases the GIL), and threads avoid the
        # process-spawn hazards of running from inside this worker QThread on Windows.
        feats_list = None
        if HAS_JOBLIB_PARALLEL and len(windows_all) >= 256:
            feat_jobs = int(max(1, (os.cpu_count() or 2) - 1))
            try:
                feats_list = Parallel(n_jobs=feat_jobs, prefer="threads")(
                    delayed(rf_extract_window_features)(w, SAMPLE_RATE) for w in windows_all
                )
            except Exception:
                feats_list = None  # fall back to sequential below
        if feats_list is None:
            feats_list = [rf_extract_window_features(w, sample_rate=SAMPLE_RATE) for w in windows_all]

        X = np.asarray(feats_list, dtype=np.float32)
        y = np.asarray(y)
        groups = np.asarray(groups, dtype=np.int64)
        classes = sorted(list(np.unique(y)))
        if len(classes) < 2:
            raise ValueError("Need at least 2 classes for classification.")

        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = np.asarray([class_to_idx[v] for v in y], dtype=np.int32)
        class_counts = np.bincount(y_idx, minlength=len(classes))
        if np.min(class_counts) < 2:
            raise ValueError("Each class needs at least 2 windows for stratified split.")

        emit_training_status("RF training: splitting train/test...")
        X_train, X_test, y_train, y_test = self._split_train_test(
            X, y_idx, groups, classes, test_size, random_seed
        )

        emit_training_status("RF training: fitting model...")
        rf_jobs = int(max(1, (os.cpu_count() or 2) - 1))
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None if max_depth_value <= 0 else max_depth_value,
            min_samples_leaf=RF_DEFAULT_MIN_SAMPLES_LEAF,
            random_state=random_seed,
            class_weight="balanced" if class_weight_balanced else None,
            n_jobs=rf_jobs,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        report_dict = classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(classes)),
            target_names=classes,
            digits=4,
            zero_division=0,
            output_dict=True,
        )
        report = self.build_aligned_classification_report(report_dict, classes)
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(classes)))

        run_dir = self._unique_run_dir(output_dir_abs, run_name)
        os.makedirs(run_dir, exist_ok=True)
        model_path = os.path.join(run_dir, DEFAULT_RF_MODEL_ARTIFACT)
        run_dir_rel = self._to_project_relative_path(run_dir)
        model_path_rel = self._to_project_relative_path(model_path)

        created_at_text = time.strftime("%Y-%m-%d %H:%M:%S")
        csv_paths_rel = [self._to_project_relative_path(p) for p in csv_paths]
        setup_payload = {
            "created_at_text": created_at_text,
            "dataset_paths": csv_paths_rel,
            "output_dir": self._to_project_relative_path(output_dir_abs),
            "run_name": run_name,
            "window_ms": window_ms,
            "stride_ms": stride_ms,
            "window_samples": win_samples,
            "stride_samples": stride_samples,
            "n_estimators": n_estimators,
            "max_depth": max_depth_value,
            "test_size": test_size,
            "random_seed": random_seed,
            "class_weight_balanced": class_weight_balanced,
            "auto_load_model": auto_load_model,
            "sample_rate_hz": int(SAMPLE_RATE),
            "input_channels": int(target_channels),
        }
        results_payload = {
            "created_at_text": created_at_text,
            "accuracy": acc,
            "train_windows": int(len(y_train)),
            "test_windows": int(len(y_test)),
            "num_features": int(X.shape[1]),
            "classes": classes,
            "class_window_counts": {classes[i]: int(class_counts[i]) for i in range(len(classes))},
            "segment_count_by_file": {
                self._to_project_relative_path(path): count for path, count in segment_count_by_file.items()
            },
            "channel_count_by_file": {
                self._to_project_relative_path(path): count for path, count in channel_count_by_file.items()
            },
            "input_channels": int(target_channels),
            "skipped_channel_segments": int(skipped_channel_segments),
            "confusion_matrix": cm.tolist(),
            "classification_report_text": report,
            "classification_report_dict": report_dict,
            "model_file": os.path.basename(model_path),
            "run_dir": run_dir_rel,
            "model_path": model_path_rel,
        }

        artifact = {
            "model": model,
            "class_names": classes,
            "sample_rate": SAMPLE_RATE,
            "window_samples": win_samples,
            "stride_samples": stride_samples,
            "created_at_text": created_at_text,
            "input_channels": int(target_channels),
            "run_dir": run_dir_rel,
            "setup": setup_payload,
            "metrics": {
                "accuracy": acc,
                "train_windows": int(len(y_train)),
                "test_windows": int(len(y_test)),
            },
        }
        joblib.dump(artifact, model_path)

        setup_path = os.path.join(run_dir, "training_setup.json")
        results_path = os.path.join(run_dir, "training_results.json")
        report_path = os.path.join(run_dir, "classification_report.txt")
        summary_path = os.path.join(run_dir, "training_summary.txt")
        cm_csv_path = os.path.join(run_dir, "confusion_matrix.csv")

        with open(setup_path, "w", encoding="utf-8") as f:
            json.dump(setup_payload, f, indent=2)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_payload, f, indent=2)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report.rstrip("\n") + "\n")
        with open(cm_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["actual/predicted"] + classes)
            for idx, row in enumerate(cm.tolist()):
                writer.writerow([classes[idx]] + [int(v) for v in row])

        summary = (
            f"Run Folder: {run_dir_rel}\n"
            f"Model File: {model_path_rel}\n"
            f"Datasets: {len(csv_paths)} file(s)\n"
            f"Accuracy: {acc:.4f}\n"
            f"Train windows: {len(y_train)} | Test windows: {len(y_test)}\n"
            f"Window: {window_ms} ms ({win_samples} samples) | Stride: {stride_ms} ms ({stride_samples} samples)\n"
            f"Input channels: {target_channels}\n"
            f"Classes: {', '.join(classes)}"
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary + "\n")

        if auto_load_model and not defer_auto_load:
            self.load_rf_model(model_path)
            self.set_status("RF training complete and model loaded.", "#2e7d32")
        elif auto_load_model and defer_auto_load:
            emit_training_status("RF training complete; loading model...", "#2e7d32")
        else:
            emit_training_status("RF training complete.", "#2e7d32")

        return {
            "run_dir": run_dir,
            "model_path": model_path,
            "artifact": artifact if defer_auto_load else None,
            "summary_text": summary,
            "report_text": report,
            "cm": cm.tolist(),
            "classes": classes,
            "config": setup_payload,
            "metrics": results_payload,
        }

    def load_saved_training_run(self, run_dir):
        run_dir_abs = os.path.abspath(str(run_dir or "").strip())
        if not run_dir_abs or not os.path.isdir(run_dir_abs):
            raise ValueError(f"Run folder not found:\n{run_dir}")

        setup_path = os.path.join(run_dir_abs, "training_setup.json")
        results_path = os.path.join(run_dir_abs, "training_results.json")
        report_path = os.path.join(run_dir_abs, "classification_report.txt")
        summary_path = os.path.join(run_dir_abs, "training_summary.txt")

        setup_payload = {}
        if os.path.isfile(setup_path):
            with open(setup_path, "r", encoding="utf-8") as f:
                setup_payload = dict(json.load(f) or {})

        results_payload = {}
        if os.path.isfile(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                results_payload = dict(json.load(f) or {})

        report_text = ""
        if os.path.isfile(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report_text = f.read().rstrip("\n")
        elif "classification_report_text" in results_payload:
            report_text = str(results_payload.get("classification_report_text", "")).rstrip("\n")
        elif "classification_report" in results_payload:
            report_text = str(results_payload.get("classification_report", "")).rstrip("\n")
        elif "classification_report_dict" in results_payload:
            report_text = self.build_aligned_classification_report(
                dict(results_payload.get("classification_report_dict", {}) or {}),
                [str(x) for x in list(results_payload.get("classes", []))],
            )

        summary_text = ""
        if os.path.isfile(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_text = f.read().strip()

        classes = [str(x) for x in list(results_payload.get("classes", []))]
        cm = results_payload.get("confusion_matrix", [])
        cm_arr = np.asarray(cm, dtype=np.float64) if len(cm) > 0 else np.zeros((0, 0), dtype=np.float64)
        if cm_arr.ndim == 2 and cm_arr.size > 0 and len(classes) == int(cm_arr.shape[0]):
            report_text = self.build_aligned_report_from_confusion_matrix(cm_arr, classes)
        if not summary_text:
            model_file = str(results_payload.get("model_file", DEFAULT_RF_MODEL_ARTIFACT)).strip() or DEFAULT_RF_MODEL_ARTIFACT
            model_path_guess = os.path.join(run_dir_abs, model_file)
            accuracy = results_payload.get("accuracy", None)
            train_windows = results_payload.get("train_windows", None)
            test_windows = results_payload.get("test_windows", None)
            summary_lines = [
                f"Run Folder: {self._to_project_relative_path(run_dir_abs)}",
                f"Model File: {self._to_project_relative_path(model_path_guess)}",
            ]
            if accuracy is not None:
                summary_lines.append(f"Accuracy: {float(accuracy):.4f}")
            if train_windows is not None and test_windows is not None:
                summary_lines.append(f"Train windows: {int(train_windows)} | Test windows: {int(test_windows)}")
            if len(classes) > 0:
                summary_lines.append(f"Classes: {', '.join(classes)}")
            summary_text = "\n".join(summary_lines)

        model_file = str(results_payload.get("model_file", DEFAULT_RF_MODEL_ARTIFACT)).strip() or DEFAULT_RF_MODEL_ARTIFACT
        model_path = os.path.join(run_dir_abs, model_file)
        if not os.path.isfile(model_path):
            fallback_path = os.path.join(run_dir_abs, DEFAULT_RF_MODEL_ARTIFACT)
            model_path = fallback_path if os.path.isfile(fallback_path) else ""

        if "dataset_paths" in setup_payload:
            setup_payload["dataset_paths"] = [
                self._from_project_relative_path(p) for p in list(setup_payload.get("dataset_paths", []))
            ]
        if "output_dir" not in setup_payload:
            setup_payload["output_dir"] = os.path.abspath(os.path.dirname(run_dir_abs))
        else:
            setup_payload["output_dir"] = self._from_project_relative_path(setup_payload.get("output_dir"))
        if "run_name" not in setup_payload:
            setup_payload["run_name"] = os.path.basename(run_dir_abs)

        result_payload = {
            "run_dir": run_dir_abs,
            "model_path": model_path,
            "summary_text": summary_text,
            "report_text": report_text,
            "cm": cm,
            "classes": classes,
            "metrics": results_payload,
        }
        return {
            "run_dir": run_dir_abs,
            "config": setup_payload,
            "result": result_payload,
        }

    @staticmethod
    def _report_num(val):
        try:
            return float(val)
        except Exception:
            return 0.0

    @staticmethod
    def _report_int(val):
        try:
            return int(round(float(val)))
        except Exception:
            return 0

    def _split_train_test(self, X, y_idx, groups, classes, test_size, random_seed):
        """Group-aware split: overlapping windows from one recording never straddle the
        train/test boundary (removes leakage that inflates accuracy). Falls back to a
        stratified split when grouping can't keep every class on both sides."""
        n_groups = int(np.unique(groups).size)
        if GroupShuffleSplit is not None and n_groups >= 2:
            try:
                gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
                train_idx, test_idx = next(gss.split(X, y_idx, groups=groups))
                y_train = y_idx[train_idx]
                y_test = y_idx[test_idx]
                # Only accept the grouped split if both sides see every class.
                n_classes = len(classes)
                if (len(train_idx) > 0 and len(test_idx) > 0
                        and np.unique(y_train).size == n_classes
                        and np.unique(y_test).size == n_classes):
                    return X[train_idx], X[test_idx], y_train, y_test
            except Exception:
                pass
        # Fallback: stratified split (may leak across overlapping windows, but keeps
        # training usable on small single-recording datasets).
        return train_test_split(
            X, y_idx, test_size=test_size, random_state=random_seed, stratify=y_idx,
        )

    @classmethod
    def build_aligned_classification_report(cls, report_dict, classes):
        data = dict(report_dict or {})
        labels = [str(x) for x in list(classes or [])]
        footer_labels = ["accuracy", "macro avg", "weighted avg"]
        label_width = max([len(x) for x in (labels + footer_labels + ["label"])], default=12)

        col_w = 10
        lines = []
        lines.append(
            f"{'':>{label_width}} "
            f"{'precision':>{col_w}} "
            f"{'recall':>{col_w}} "
            f"{'f1-score':>{col_w}} "
            f"{'support':>{col_w}}"
        )
        lines.append("")

        total_support = 0
        for label in labels:
            row = dict(data.get(label, {}) or {})
            precision = cls._report_num(row.get("precision", 0.0))
            recall = cls._report_num(row.get("recall", 0.0))
            f1 = cls._report_num(row.get("f1-score", 0.0))
            support = cls._report_int(row.get("support", 0))
            total_support += support
            lines.append(
                f"{label:>{label_width}} "
                f"{precision:>{col_w}.4f} "
                f"{recall:>{col_w}.4f} "
                f"{f1:>{col_w}.4f} "
                f"{support:>{col_w}d}"
            )

        lines.append("")
        acc_raw = data.get("accuracy", 0.0)
        if isinstance(acc_raw, dict):
            acc = cls._report_num(acc_raw.get("f1-score", 0.0))
            acc_support = cls._report_int(acc_raw.get("support", total_support))
        else:
            acc = cls._report_num(acc_raw)
            acc_support = total_support
        lines.append(
            f"{'accuracy':>{label_width}} "
            f"{'':>{col_w}} "
            f"{'':>{col_w}} "
            f"{acc:>{col_w}.4f} "
            f"{acc_support:>{col_w}d}"
        )

        for footer_key in ("macro avg", "weighted avg"):
            row = dict(data.get(footer_key, {}) or {})
            precision = cls._report_num(row.get("precision", 0.0))
            recall = cls._report_num(row.get("recall", 0.0))
            f1 = cls._report_num(row.get("f1-score", 0.0))
            support = cls._report_int(row.get("support", total_support))
            lines.append(
                f"{footer_key:>{label_width}} "
                f"{precision:>{col_w}.4f} "
                f"{recall:>{col_w}.4f} "
                f"{f1:>{col_w}.4f} "
                f"{support:>{col_w}d}"
            )

        return "\n".join(lines).rstrip()

    @classmethod
    def build_aligned_report_from_confusion_matrix(cls, cm, classes):
        arr = np.asarray(cm, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[0] != arr.shape[1]:
            return ""

        labels = [str(x) for x in list(classes or [])]
        if len(labels) != int(arr.shape[0]):
            labels = [f"C{i + 1}" for i in range(int(arr.shape[0]))]

        row_sum = np.sum(arr, axis=1)
        col_sum = np.sum(arr, axis=0)
        tp = np.diag(arr)

        precision = np.divide(tp, col_sum, out=np.zeros_like(tp), where=col_sum > 0)
        recall = np.divide(tp, row_sum, out=np.zeros_like(tp), where=row_sum > 0)
        f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)

        total = float(np.sum(row_sum))
        accuracy = float(np.sum(tp) / total) if total > 0 else 0.0

        macro_precision = float(np.mean(precision)) if len(precision) > 0 else 0.0
        macro_recall = float(np.mean(recall)) if len(recall) > 0 else 0.0
        macro_f1 = float(np.mean(f1)) if len(f1) > 0 else 0.0

        if total > 0:
            weights = row_sum / total
            weighted_precision = float(np.sum(precision * weights))
            weighted_recall = float(np.sum(recall * weights))
            weighted_f1 = float(np.sum(f1 * weights))
        else:
            weighted_precision = 0.0
            weighted_recall = 0.0
            weighted_f1 = 0.0

        report_dict = {}
        for i, label in enumerate(labels):
            report_dict[label] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1-score": float(f1[i]),
                "support": int(round(float(row_sum[i]))),
            }
        report_dict["accuracy"] = accuracy
        report_dict["macro avg"] = {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1-score": macro_f1,
            "support": int(round(total)),
        }
        report_dict["weighted avg"] = {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1-score": weighted_f1,
            "support": int(round(total)),
        }
        return cls.build_aligned_classification_report(report_dict, labels)

    def on_calibration_dialog_closed(self, _result=None):
        # If user closes while calibration is active, cancel the sequence.
        if self.calibration_active:
            self.stop_calibration_if_running()
            self.set_status("Calibration canceled - calibration required", "#ff9800")
            self.btn_analysis.setEnabled(False)
        self.calibration_dialog = None
        self.btn_calibrate.setEnabled(True)

    def start_calibration_sequence(self):
        if not self.is_connected:
            return
        if self.calibration_active:
            return
        if self.calibration_dialog:
            rest_ms, flex_ms = self.calibration_dialog.calibration_durations_ms()
            self.set_calibration_phase_durations(rest_ms // 1000, flex_ms // 1000)

        self.calibration_active = True
        self.is_calibrated = False
        self.rest_capture = []
        self.flex_capture = []
        self.current_cal_phase_idx = -1
        self.data_buffer[:, :] = 0
        if self.raw_data_buffer is not None:
            self.raw_data_buffer[:, :] = 0
        self.rf_valid_sample_count = 0
        self.rf_samples_since_submit = 0

        self.timer.stop()
        self.btn_calibrate.setEnabled(False)
        self.btn_analysis.setEnabled(False)
        self.set_status("Calibrating... follow dialog instructions", "#ff9800")
        self.set_waiting_for_calibration_labels()

        if self.calibration_dialog:
            self.calibration_dialog.set_running_state()

        self.begin_next_calibration_phase()

    def begin_next_calibration_phase(self):
        self.current_cal_phase_idx += 1
        if self.current_cal_phase_idx >= len(self.calibration_phases):
            self.finish_calibration_sequence()
            return

        phase = self.calibration_phases[self.current_cal_phase_idx]
        self.current_phase_key = phase["key"]
        self.current_phase_total_ms = phase["duration_ms"]
        self.current_phase_remaining_ms = phase["duration_ms"]

        if self.calibration_dialog:
            self.calibration_dialog.set_phase(
                phase["name"],
                phase["instruction"],
                self.current_phase_remaining_ms,
                self.current_phase_total_ms,
            )

        self.calibration_timer.start(CAL_TICK_MS)

    def on_calibration_tick(self):
        if not self.calibration_active:
            self.calibration_timer.stop()
            return

        self.current_phase_remaining_ms -= CAL_TICK_MS
        phase = self.calibration_phases[self.current_cal_phase_idx]

        if self.calibration_dialog:
            self.calibration_dialog.set_phase(
                phase["name"],
                phase["instruction"],
                self.current_phase_remaining_ms,
                self.current_phase_total_ms,
            )

        if self.current_phase_remaining_ms <= 0:
            self.calibration_timer.stop()
            self.begin_next_calibration_phase()

    def finish_calibration_sequence(self):
        self.calibration_timer.stop()
        self.calibration_active = False

        if len(self.rest_capture) == 0:
            self.set_status("Calibration failed: no REST samples", "#f44336")
            QMessageBox.warning(self, "Calibration Failed", "No REST samples captured.")
            self.btn_calibrate.setEnabled(True)
            return

        rest = np.vstack(self.rest_capture).astype(np.float32)      # (samples, channels)
        emg_count = int(min(self.active_emg_channel_count(), rest.shape[1]))
        self.baseline_offsets = np.zeros(self.num_channels, dtype=np.float32)
        if emg_count > 0:
            self.baseline_offsets[:emg_count] = np.median(rest[:, :emg_count], axis=0).astype(np.float32)
        self.baseline_reference = self.baseline_offsets.copy()

        rest_centered = rest.copy()
        if emg_count > 0:
            rest_centered[:, :emg_count] = rest_centered[:, :emg_count] - self.baseline_offsets[np.newaxis, :emg_count]
        rest_rms = np.ones(self.num_channels, dtype=np.float32)
        if emg_count > 0:
            rest_rms[:emg_count] = np.sqrt(np.mean(np.square(rest_centered[:, :emg_count]), axis=0)).astype(np.float32)
        self.rest_rms_ref = np.maximum(rest_rms.astype(np.float32), 1.0)
        auto_rms_source = rest_rms[:emg_count] if emg_count > 0 else rest_rms
        auto_rms = float(np.clip(np.mean(auto_rms_source) * 3.0, 20.0, 800.0))
        self.threshold_spin.setValue(round(auto_rms, 1))

        if len(self.flex_capture) > 0:
            flex = np.vstack(self.flex_capture).astype(np.float32)  # (samples, channels)
            flex_centered = flex.copy()
            if emg_count > 0:
                flex_centered[:, :emg_count] = flex_centered[:, :emg_count] - self.baseline_offsets[np.newaxis, :emg_count]
            flex_rms = np.ones(self.num_channels, dtype=np.float32)
            if emg_count > 0:
                flex_rms[:emg_count] = np.sqrt(np.mean(np.square(flex_centered[:, :emg_count]), axis=0)).astype(np.float32)
            self.flex_rms_ref = np.maximum(flex_rms.astype(np.float32), self.rest_rms_ref + 10.0)
            dom_hz = self.compute_dominant_frequency(flex_centered[:, :emg_count].T if emg_count > 0 else flex_centered.T)
            auto_hz = float(np.clip(np.median(dom_hz) * 0.7, 20.0, FFT_MAX_HZ))
            self.hz_threshold_spin.setValue(round(auto_hz, 1))
        else:
            self.flex_rms_ref = np.maximum(self.rest_rms_ref * 3.0, self.rest_rms_ref + 10.0)

        self.data_buffer[:, :] = 0
        if self.raw_data_buffer is not None:
            self.raw_data_buffer[:, :] = 0
        self.rf_valid_sample_count = 0
        self.rf_samples_since_submit = 0
        self.is_calibrated = True
        self.onset_state[:] = False
        self.onset_count[:] = 0
        self.offset_count[:] = 0
        self.current_burst_ms[:] = 0
        self.repetition_count[:] = 0
        self.burst_history_ms = [[] for _ in range(self.num_channels)]
        self.fatigue_track = []
        self.force_level_pct = 0.0
        self.prev_force_level_pct = 0.0
        self.gesture_label = "REST"
        self.movement_phase = "REST"
        self.anomaly_label = "NORMAL"
        self.btn_calibrate.setEnabled(True)
        self.btn_analysis.setEnabled(True)
        self.live_analysis_enabled = False
        self.set_analysis_idle_labels()
        self.reset_fps_counters()
        self._analysis_pending = False
        self._last_analysis_submit_ts = 0.0
        self._plot_ranges_dirty = True
        self._last_plot_range_refresh_ts = 0.0
        self.timer.start(PLOT_REFRESH_MS)
        self.set_status("Calibrated - live graph/analysis running", "#00c853")
        if self.rf_model is not None:
            self.set_rf_status_reason(f"Buffering stream (0/{self.rf_window_samples} samples)", "success")

        summary = (
            f"Calibration complete.\n"
            f"Baseline (ADC): {np.array2string(self.baseline_offsets, precision=1)}\n"
            f"REST/FLEX duration: {self.cal_rest_seconds}s / {self.cal_flex_seconds}s\n"
            f"RMS threshold: {self.threshold_spin.value():.1f}\n"
            f"Hz threshold: {self.hz_threshold_spin.value():.1f}"
        )
        if self.calibration_dialog:
            self.calibration_dialog.set_finished(summary)
            self.calibration_dialog.accept()

    def stop_calibration_if_running(self):
        if self.calibration_active:
            self.calibration_active = False
            self.calibration_timer.stop()
            self.current_phase_key = ""

    def cancel_calibration_sequence(self):
        self.stop_calibration_if_running()
        if not self.is_calibrated and self.is_connected:
            self.set_status("Calibration canceled - calibration required", "#ff9800")
            self.set_waiting_for_calibration_labels()
            self.btn_analysis.setEnabled(False)
        self.btn_calibrate.setEnabled(True)

    def apply_python_baseline(self, raw_batch_T):
        # raw_batch_T shape: (channels, samples), baseline offsets shape: (channels,)
        adjusted = np.ascontiguousarray(raw_batch_T, dtype=np.float32)
        emg_count = int(min(self.active_emg_channel_count(), adjusted.shape[0]))
        if emg_count <= 0:
            return adjusted

        centered = adjusted[:emg_count, :] - self.baseline_offsets[:emg_count, np.newaxis]

        # Slow adaptive baseline correction near rest for EMG channels only.
        for ch in range(emg_count):
            near_rest = np.abs(centered[ch, :]) < BASE_ADAPT_GUARD
            if np.any(near_rest):
                mean_err = float(np.mean(centered[ch, near_rest]))
                self.baseline_offsets[ch] += BASE_ADAPT_ALPHA * mean_err

        adjusted[:emg_count, :] = adjusted[:emg_count, :] - self.baseline_offsets[:emg_count, np.newaxis]
        return adjusted

    @staticmethod
    def _normalize_stream_payload(payload):
        if isinstance(payload, dict):
            batch = np.asarray(payload.get("batch", []), dtype=np.float32)
            packet_numbers = np.asarray(payload.get("packet_numbers", []), dtype=np.int64).reshape(-1)
            metadata = {key: payload.get(key) for key in (
                "source", "frame_ids", "emg_timestamps", "imu_ids", "imu_timestamps",
                "host_received_monotonic", "packet_gap", "gap_frames", "valid", "late_flags",
                "imu_fresh", "host_received_s", "device_counters", "restart")}
            return batch, packet_numbers, metadata

        batch = np.asarray(payload, dtype=np.float32)
        return batch, np.zeros((0,), dtype=np.int64), {}

    def on_serial_batch(self, payload):
        if self.data_buffer is None or self.raw_data_buffer is None:
            return
        batch, packet_numbers, transport_metadata = self._normalize_stream_payload(payload)
        if batch is None or batch.size == 0:
            return
        if batch.ndim != 2 or batch.shape[1] != self.num_channels:
            return
        if self.session_recorder is not None and isinstance(payload, dict):
            # Every frame (gap frames included) goes to disk via the writer thread; labels come later.
            self.session_recorder.append(payload)
        self.last_transport_metadata = transport_metadata
        # Packet gaps are not silently treated as continuous EMG. Require a
        # fresh full model window before classification resumes.
        packet_gap = int(transport_metadata.get("packet_gap") or 0)
        if int(transport_metadata.get("gap_frames") or 0) > 0:
            packet_gap = max(packet_gap, 1)
        valid = transport_metadata.get("valid")
        if valid is not None and not np.all(valid):
            # NaN gap rows exist for the recording; the live filters/plots must only see real samples.
            keep = np.asarray(valid, dtype=bool)
            batch = batch[keep]
            if packet_numbers.shape[0] == keep.shape[0]:
                packet_numbers = packet_numbers[keep]
            if batch.shape[0] == 0:
                if packet_gap > 0:
                    self.rf_valid_sample_count = 0
                    self.rf_samples_since_submit = 0
                return
        if packet_gap > 0:
            self.rf_valid_sample_count = 0
            self.rf_samples_since_submit = 0
            if self.session_recorder is not None:
                self.record_missing_packets += packet_gap
                self.record_gap_events += 1

        self._tick_data_fps(batch.shape[0])

        clip_now = np.zeros(self.num_channels, dtype=np.float32)
        emg_count = int(min(self.active_emg_channel_count(), batch.shape[1]))
        if emg_count > 0:
            clip_now[:emg_count] = np.mean((batch[:, :emg_count] <= 5.0) | (batch[:, :emg_count] >= 4090.0), axis=0) * 100.0
        self.last_clip_ratio = 0.90 * self.last_clip_ratio + 0.10 * clip_now.astype(np.float32)

        if self.calibration_active:
            if self.current_phase_key == "rest":
                self.rest_capture.append(batch.copy())
            elif self.current_phase_key == "flex":
                self.flex_capture.append(batch.copy())
            return

        if not self.is_calibrated:
            return  # Hard requirement: no graph/analysis before calibration.

        # Single transpose: (samples, channels) → (channels, samples) used for both buffers.
        raw_new_data = batch.T
        raw_num_new = raw_new_data.shape[1]
        new_data = self.apply_python_baseline(raw_new_data)  # (channels, samples) in/out
        num_new = new_data.shape[1]

        with self._buffer_lock:
            if raw_num_new >= WINDOW_SIZE:
                self.raw_data_buffer[:, :] = raw_new_data[:, -WINDOW_SIZE:]
            else:
                self.raw_data_buffer[:, :-raw_num_new] = self.raw_data_buffer[:, raw_num_new:]
                self.raw_data_buffer[:, -raw_num_new:] = raw_new_data

            # Overflow-safe update: if backlog is larger than window, keep newest WINDOW_SIZE.
            if num_new >= WINDOW_SIZE:
                self.data_buffer[:, :] = new_data[:, -WINDOW_SIZE:]
            else:
                self.data_buffer[:, :-num_new] = self.data_buffer[:, num_new:]
                self.data_buffer[:, -num_new:] = new_data

        self._plot_dirty = True
        self.rf_valid_sample_count = int(min(WINDOW_SIZE, self.rf_valid_sample_count + raw_num_new))
        self.schedule_realtime_prediction(num_new)

    def _on_analysis_finished(self):
        self._analysis_pending = False

    def _on_analysis_ready(self, result):
        """Slot called on the main thread when LiveAnalysisWorker finishes a frame."""
        if not (self.live_analysis_enabled and not self.task_session_active
                and self.analysis_window and self.analysis_window.isVisible()):
            return
        if self.data_buffer is None or not self.is_calibrated:
            return
        self.analysis_idle_labels_active = False

        time_metrics = result["time_metrics"]
        spectral = result["spectral"]
        timefreq = result["timefreq"]
        params = result["params"]

        self.latest_mav = time_metrics["mav"]
        self.latest_rms = time_metrics["rms"]
        self.latest_dom_hz = spectral["peak_hz"]
        self.latest_mean_hz = spectral["mean_hz"]
        self.latest_median_hz = spectral["median_hz"]
        self.latest_spec_entropy = spectral["spec_entropy"]
        self.latest_band_power_pct = spectral["band_power_pct"]
        self.latest_mains_noise_score = spectral["mains_noise_score"]
        self.latest_stft_dom_mean_hz = timefreq["stft_dom_mean_hz"]
        self.latest_stft_dom_std_hz = timefreq["stft_dom_std_hz"]
        self.latest_short_time_band_delta = timefreq["short_time_band_delta"]
        self.latest_wavelet_energy_pct = timefreq["wavelet_energy_pct"]
        self.wavelet_available = timefreq["wavelet_available"]
        self.latest_corr_matrix = result["corr_matrix"]
        self.latest_lag_ms_matrix = result["lag_ms_matrix"]
        self.latest_coherence_matrix = result["coherence_matrix"]
        coord = result["coord_summary"]
        self.latest_channel_ratio = coord["channel_ratio"]
        self.latest_symmetry_index = coord["symmetry_index"]
        self.latest_co_contraction_index = coord["co_contraction_index"]

        amp_threshold = float(params.get("amp_threshold", self.threshold_spin.value()))
        hz_threshold = float(params.get("hz_threshold", self.hz_threshold_spin.value()))
        use_hz_gate = bool(params.get("use_hz_gate", self.check_use_hz_gate.isChecked()))

        amp_active = self.latest_rms >= amp_threshold
        self.latest_hz_active = self.latest_dom_hz >= hz_threshold
        if use_hz_gate:
            self.latest_active = np.logical_and(amp_active, self.latest_hz_active)
        else:
            self.latest_active = amp_active

        self.update_onset_offset(amp_active, amp_threshold)
        burst_mean_ms = np.array(
            [np.mean(h) if len(h) > 0 else 0.0 for h in self.burst_history_ms], dtype=np.float32
        )
        self.latest_quality_score = self.compute_contraction_quality_score(
            self.latest_rms, self.last_clip_ratio, self.latest_mains_noise_score, amp_threshold,
        )
        self.contact_quality = self.compute_contact_quality(
            self.latest_rms, self.rest_rms_ref, self.last_clip_ratio, self.latest_mains_noise_score,
        )
        snr_db = self.compute_snr_db(self.latest_rms, self.rest_rms_ref)
        baseline_drift = self.baseline_offsets - self.baseline_reference
        fatigue_slope = self.update_fatigue_trend(float(np.mean(self.latest_median_hz)))
        self.force_level_pct = self.estimate_force_level_pct(self.latest_rms)
        self.movement_phase = self.estimate_movement_phase(self.force_level_pct)
        self.gesture_label = self.estimate_gesture_label(self.latest_rms, self.latest_active)
        if self.rf_last_pred_label != "N/A":
            self.gesture_label = self.rf_last_pred_label

        if self.rf_model is not None:
            model_name = os.path.basename(self.rf_model_path) if self.rf_model_path else "Loaded"
            if self.rf_last_status_reason:
                self.lbl_rf.setText(f"RF Model: {model_name} | {self.rf_last_status_reason}")
            else:
                self.lbl_rf.setText(
                    f"RF Model: {model_name} | Pred: {self.rf_last_pred_label} ({self.rf_last_pred_conf * 100.0:.1f}%)"
                )
            self.lbl_rf.setStyleSheet(themed_label_style("success"))
        else:
            self.lbl_rf.setText("RF Model: Not loaded")
            self.lbl_rf.setStyleSheet(themed_label_style("muted"))

        self.anomaly_label = self.detect_anomaly_label(
            self.last_clip_ratio, self.latest_mains_noise_score, baseline_drift, self.contact_quality,
        )

        with self._buffer_lock:
            imu_last = {i: float(self.data_buffer[i, -1]) for i in range(self.num_channels)
                        if self.is_wireless_imu_channel(i)}

        for i in range(self.num_channels):
            active = bool(self.latest_active[i])
            state = "Active" if active else "Rest"
            color = THEME_COLORS["success"] if active else THEME_COLORS["muted"]
            hz_flag = "ON" if self.latest_hz_active[i] else "OFF"
            if self.is_wireless_imu_channel(i):
                self.metrics_labels[i].setText(
                    f"{self.channel_display_name(i)} | Angle\n"
                    f"Value: {imu_last.get(i, 0.0):.2f} deg"
                )
            else:
                self.metrics_labels[i].setText(
                    f"{self.channel_display_name(i)} | {state} | Hz: {hz_flag}\n"
                    f"MAV: {self.latest_mav[i]:.2f} - RMS: {self.latest_rms[i]:.2f} - FREQ: {self.latest_dom_hz[i]:.2f}Hz"
                )
            if self._metrics_color_cache.get(i) != color:
                self.metrics_labels[i].setStyleSheet(
                    f"color: {color}; font-family: Consolas; font-size: 15px; font-weight: bold;"
                )
                self._metrics_color_cache[i] = color
            if self.threshold_lines_pos[i] is not None and self.threshold_lines_neg[i] is not None:
                line_color = THEME_COLORS["success"] if active else THEME_COLORS["accent"]
                if self._threshold_line_value_cache != amp_threshold:
                    self.threshold_lines_pos[i].setValue(amp_threshold)
                    self.threshold_lines_neg[i].setValue(-amp_threshold)
                if self._threshold_line_color_cache.get(i) != line_color:
                    self.threshold_lines_pos[i].setPen(pg.mkPen(line_color, width=1, style=Qt.DashLine))
                    self.threshold_lines_neg[i].setPen(pg.mkPen(line_color, width=1, style=Qt.DashLine))
                    self._threshold_line_color_cache[i] = line_color
        self._threshold_line_value_cache = amp_threshold

        if self.analysis_window and self.analysis_window.isVisible():
            self.analysis_window.update_analysis_view(
                {
                    "mav": self.latest_mav,
                    "rms": self.latest_rms,
                    "iemg": time_metrics["iemg"],
                    "var": time_metrics["var"],
                    "wl": time_metrics["wl"],
                    "zc": time_metrics["zc"],
                    "ssc": time_metrics["ssc"],
                    "wamp": time_metrics["wamp"],
                    "mean_hz": self.latest_mean_hz,
                    "median_hz": self.latest_median_hz,
                    "peak_hz": self.latest_dom_hz,
                    "spec_entropy": self.latest_spec_entropy,
                    "band_power_pct": self.latest_band_power_pct,
                    "mains_noise_score": self.latest_mains_noise_score,
                    "stft_dom_mean_hz": self.latest_stft_dom_mean_hz,
                    "stft_dom_std_hz": self.latest_stft_dom_std_hz,
                    "short_time_band_delta": self.latest_short_time_band_delta,
                    "wavelet_energy_pct": self.latest_wavelet_energy_pct,
                    "wavelet_available": self.wavelet_available,
                    "corr_matrix": self.latest_corr_matrix,
                    "lag_ms_matrix": self.latest_lag_ms_matrix,
                    "coherence_matrix": self.latest_coherence_matrix,
                    "channel_ratio": self.latest_channel_ratio,
                    "symmetry_index": self.latest_symmetry_index,
                    "co_contraction_index": self.latest_co_contraction_index,
                    "active_flags": self.latest_active,
                    "onset_count": self.onset_count,
                    "offset_count": self.offset_count,
                    "repetition_count": self.repetition_count,
                    "burst_mean_ms": burst_mean_ms,
                    "burst_current_ms": self.current_burst_ms,
                    "contraction_quality_score": self.latest_quality_score,
                    "force_level_pct": self.force_level_pct,
                    "movement_phase": self.movement_phase,
                    "gesture_label": self.gesture_label,
                    "anomaly_label": self.anomaly_label,
                    "rf_model_loaded": self.rf_model is not None,
                    "rf_model_path": self.rf_model_path if self.rf_model_path else "N/A",
                    "rf_pred_label": self.rf_last_pred_label,
                    "rf_pred_conf_pct": self.rf_last_pred_conf * 100.0,
                    "snr_db": snr_db,
                    "baseline_drift": baseline_drift,
                    "clip_ratio_pct": self.last_clip_ratio,
                    "contact_quality": self.contact_quality,
                    "fatigue_slope_hz_per_s": fatigue_slope,
                }
            )

    def update_onset_offset(self, amp_active, amp_threshold):
        if self.onset_state is None:
            return

        off_threshold = amp_threshold * 0.60
        dt_ms = float(PLOT_REFRESH_MS)
        for ch in range(self.num_channels):
            prev = bool(self.onset_state[ch])
            curr_rms = float(self.latest_rms[ch])

            if not prev and curr_rms >= amp_threshold:
                self.onset_state[ch] = True
                self.onset_count[ch] += 1
                self.repetition_count[ch] += 1
                self.current_burst_ms[ch] = 0.0
            elif prev and curr_rms <= off_threshold:
                self.onset_state[ch] = False
                self.offset_count[ch] += 1
                if self.current_burst_ms[ch] > 0:
                    self.burst_history_ms[ch].append(float(self.current_burst_ms[ch]))
                    if len(self.burst_history_ms[ch]) > 30:
                        self.burst_history_ms[ch] = self.burst_history_ms[ch][-30:]
                self.current_burst_ms[ch] = 0.0
            else:
                self.onset_state[ch] = prev

            if self.onset_state[ch]:
                self.current_burst_ms[ch] += dt_ms

    @staticmethod
    def compute_time_domain_features(segment):
        # segment shape: (channels, samples), centered signal expected
        n_ch, n_samples = segment.shape
        abs_seg = np.abs(segment)
        diff_seg = np.diff(segment, axis=1) if n_samples > 1 else np.zeros((n_ch, 0), dtype=np.float32)

        mav = np.mean(abs_seg, axis=1)
        rms = np.sqrt(np.mean(np.square(segment), axis=1))
        iemg = np.sum(abs_seg, axis=1)
        var = np.var(segment, axis=1)
        wl = np.sum(np.abs(diff_seg), axis=1) if diff_seg.size else np.zeros(n_ch, dtype=np.float32)

        zc_thresh = 10.0
        ssc_thresh = 8.0
        wamp_thresh = 12.0

        zc = np.zeros(n_ch, dtype=np.int32)
        ssc = np.zeros(n_ch, dtype=np.int32)
        wamp = np.zeros(n_ch, dtype=np.int32)

        for ch in range(n_ch):
            x = segment[ch]
            if n_samples > 1:
                zc[ch] = int(np.sum(((x[:-1] * x[1:]) < 0) & (np.abs(x[:-1] - x[1:]) >= zc_thresh)))
                wamp[ch] = int(np.sum(np.abs(x[1:] - x[:-1]) >= wamp_thresh))
            if n_samples > 2:
                s1 = x[1:-1] - x[:-2]
                s2 = x[1:-1] - x[2:]
                ssc[ch] = int(np.sum(((s1 * s2) > 0) & ((np.abs(s1) + np.abs(s2)) >= ssc_thresh)))

        return {
            "mav": mav.astype(np.float32),
            "rms": rms.astype(np.float32),
            "iemg": iemg.astype(np.float32),
            "var": var.astype(np.float32),
            "wl": wl.astype(np.float32),
            "zc": zc,
            "ssc": ssc,
            "wamp": wamp,
        }

    @staticmethod
    def compute_spectral_features(segment):
        # segment shape: (channels, samples)
        n_ch, n_samples = segment.shape
        zeros = np.zeros(n_ch, dtype=np.float32)
        band_zeros = np.zeros((n_ch, len(BAND_DEFS)), dtype=np.float32)

        if n_samples < 8:
            return {
                "peak_hz": zeros,
                "mean_hz": zeros,
                "median_hz": zeros,
                "spec_entropy": zeros,
                "band_power_pct": band_zeros,
                "mains_noise_score": zeros,
            }

        window, freqs = fft_window_and_freqs(n_samples, SAMPLE_RATE)
        windowed = segment * window[np.newaxis, :]
        spectrum = np.abs(np.fft.rfft(windowed, axis=1)) ** 2

        valid = (freqs >= FFT_MIN_HZ) & (freqs <= FFT_MAX_HZ)
        valid_idx = np.where(valid)[0]
        if valid_idx.size == 0:
            return {
                "peak_hz": zeros,
                "mean_hz": zeros,
                "median_hz": zeros,
                "spec_entropy": zeros,
                "band_power_pct": band_zeros,
                "mains_noise_score": zeros,
            }

        spec_valid = spectrum[:, valid_idx]
        freq_valid = freqs[valid_idx]
        total_power = np.sum(spec_valid, axis=1) + 1e-9

        peak_idx = np.argmax(spec_valid, axis=1)
        peak_hz = freq_valid[peak_idx].astype(np.float32)

        mean_hz = (np.sum(spec_valid * freq_valid[np.newaxis, :], axis=1) / total_power).astype(np.float32)
        cumsum_power = np.cumsum(spec_valid, axis=1)
        med_idx = np.argmax(cumsum_power >= (total_power[:, np.newaxis] * 0.5), axis=1)
        median_hz = freq_valid[med_idx].astype(np.float32)

        pnorm = spec_valid / total_power[:, np.newaxis]
        spec_entropy = (-np.sum(pnorm * np.log2(pnorm + 1e-12), axis=1) / np.log2(pnorm.shape[1] + 1e-9)).astype(
            np.float32
        )

        band_power_pct = np.zeros((n_ch, len(BAND_DEFS)), dtype=np.float32)
        for bi, (f_lo, f_hi) in enumerate(BAND_DEFS):
            bmask = (freq_valid >= f_lo) & (freq_valid < f_hi)
            if np.any(bmask):
                bp = np.sum(spec_valid[:, bmask], axis=1)
                band_power_pct[:, bi] = (bp / total_power) * 100.0

        mains_noise_score = np.zeros(n_ch, dtype=np.float32)
        for mains_f in MAINS_FREQS:
            mmask = (freq_valid >= (mains_f - MAINS_BAND_HZ)) & (freq_valid <= (mains_f + MAINS_BAND_HZ))
            if np.any(mmask):
                mains_noise_score += (np.sum(spec_valid[:, mmask], axis=1) / total_power * 100.0).astype(np.float32)

        return {
            "peak_hz": peak_hz,
            "mean_hz": mean_hz,
            "median_hz": median_hz,
            "spec_entropy": spec_entropy,
            "band_power_pct": band_power_pct,
            "mains_noise_score": mains_noise_score,
        }

    @staticmethod
    def compute_time_frequency_features(segment):
        n_ch, n_samples = segment.shape
        stft_mean = np.zeros(n_ch, dtype=np.float32)
        stft_std = np.zeros(n_ch, dtype=np.float32)
        band_delta = np.zeros((n_ch, len(BAND_DEFS)), dtype=np.float32)
        wavelet_energy = np.zeros((n_ch, 4), dtype=np.float32)

        # STFT-like dominant-frequency summary.
        stft_win = min(64, n_samples)
        if stft_win >= 16:
            hop = max(4, stft_win // 2)
            dom_frames = [[] for _ in range(n_ch)]
            for start in range(0, n_samples - stft_win + 1, hop):
                frame = segment[:, start:start + stft_win]
                spec = EMGVisualizer.compute_spectral_features(frame)
                for ch in range(n_ch):
                    dom_frames[ch].append(float(spec["peak_hz"][ch]))
            for ch in range(n_ch):
                if len(dom_frames[ch]) > 0:
                    stft_mean[ch] = float(np.mean(dom_frames[ch]))
                    stft_std[ch] = float(np.std(dom_frames[ch]))

        # Short-time band-power changes: final window minus first window.
        chunk = max(16, n_samples // 4)
        first = segment[:, :chunk]
        last = segment[:, -chunk:]
        bp_first = EMGVisualizer.compute_spectral_features(first)["band_power_pct"]
        bp_last = EMGVisualizer.compute_spectral_features(last)["band_power_pct"]
        band_delta = (bp_last - bp_first).astype(np.float32)

        wavelet_available = False
        if HAS_PYWT and pywt is not None:
            try:
                wavelet_available = True
                for ch in range(n_ch):
                    coeffs = pywt.wavedec(segment[ch], "db4", level=3)
                    # [cA3, cD3, cD2, cD1]
                    energies = np.array([np.sum(np.square(c)) for c in coeffs], dtype=np.float64)
                    e_sum = np.sum(energies) + 1e-12
                    wavelet_energy[ch] = (energies / e_sum * 100.0).astype(np.float32)
            except Exception:
                wavelet_available = False
                wavelet_energy[:] = 0.0

        return {
            "stft_dom_mean_hz": stft_mean,
            "stft_dom_std_hz": stft_std,
            "short_time_band_delta": band_delta,
            "wavelet_energy_pct": wavelet_energy,
            "wavelet_available": wavelet_available,
        }

    @staticmethod
    def compute_corr_matrix(segment):
        if segment.shape[0] < 2 or segment.shape[1] < 8:
            return np.eye(segment.shape[0], dtype=np.float32)
        std = np.std(segment, axis=1)
        valid = np.isfinite(std) & (std > 1e-8)
        if not np.any(valid):
            return np.eye(segment.shape[0], dtype=np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(segment)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if corr.ndim != 2:
            return np.eye(segment.shape[0], dtype=np.float32)
        if not np.all(valid):
            corr[~valid, :] = 0.0
            corr[:, ~valid] = 0.0
        np.fill_diagonal(corr, 1.0)
        return np.clip(corr, -1.0, 1.0)

    @staticmethod
    def compute_lag_ms_matrix(segment):
        n_ch, n_samples = segment.shape
        lag_matrix = np.zeros((n_ch, n_ch), dtype=np.float32)
        if n_samples < 8:
            return lag_matrix

        max_lag = min(MAX_LAG_SAMPLES, n_samples - 1)
        for i in range(n_ch):
            for j in range(n_ch):
                if i == j:
                    continue
                x = segment[i] - np.mean(segment[i])
                y = segment[j] - np.mean(segment[j])
                denom = (np.std(x) * np.std(y) * len(x)) + 1e-9
                corr_full = np.correlate(x, y, mode="full") / denom
                lags = np.arange(-len(x) + 1, len(x))
                mask = (lags >= -max_lag) & (lags <= max_lag)
                if np.any(mask):
                    sel_corr = corr_full[mask]
                    sel_lags = lags[mask]
                    lag_samples = sel_lags[int(np.argmax(np.abs(sel_corr)))]
                    lag_matrix[i, j] = float(lag_samples) * (1000.0 / SAMPLE_RATE)
        return lag_matrix

    @staticmethod
    def compute_coherence_matrix(segment):
        n_ch, n_samples = segment.shape
        coh = np.eye(n_ch, dtype=np.float32)
        if n_samples < 8:
            return coh

        window, freqs = fft_window_and_freqs(n_samples, SAMPLE_RATE)
        xw = segment * window[np.newaxis, :]
        X = np.fft.rfft(xw, axis=1)
        band = (freqs >= FFT_MIN_HZ) & (freqs <= FFT_MAX_HZ)
        if not np.any(band):
            return coh

        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                Xi = X[i, band]
                Xj = X[j, band]
                pxx = np.sum(np.abs(Xi) ** 2) + 1e-9
                pyy = np.sum(np.abs(Xj) ** 2) + 1e-9
                pxy = np.sum(Xi * np.conj(Xj))
                val = float((np.abs(pxy) ** 2) / (pxx * pyy))
                val = float(np.clip(val, 0.0, 1.0))
                coh[i, j] = val
                coh[j, i] = val
        return coh

    @staticmethod
    def compute_coordination_indices(segment, rms_vals):
        n_ch = segment.shape[0]
        mean_rms = float(np.mean(rms_vals) + 1e-9)
        channel_ratio = (rms_vals / mean_rms).astype(np.float32)

        pairs = []
        if n_ch >= 2:
            pairs.append((0, 1))
        if n_ch >= 4:
            pairs.append((2, 3))
        if n_ch == 3:
            pairs.append((1, 2))

        symmetry = {}
        cocon = {}
        abs_seg = np.abs(segment)
        for a, b in pairs:
            key = f"CH{a+1}-CH{b+1}"
            ra = float(rms_vals[a])
            rb = float(rms_vals[b])
            symmetry[key] = float(abs(ra - rb) / (ra + rb + 1e-9))
            overlap = np.sum(np.minimum(abs_seg[a], abs_seg[b]))
            total = np.sum(abs_seg[a]) + np.sum(abs_seg[b]) + 1e-9
            cocon[key] = float((2.0 * overlap) / total)

        return {
            "channel_ratio": channel_ratio,
            "symmetry_index": symmetry,
            "co_contraction_index": cocon,
        }

    @staticmethod
    def compute_snr_db(rms_vals, rest_rms_ref):
        return (20.0 * np.log10((rms_vals + 1e-6) / (rest_rms_ref + 1e-6))).astype(np.float32)

    @staticmethod
    def compute_contact_quality(rms_vals, rest_rms_ref, clip_ratio, mains_noise_score):
        snr_db = 20.0 * np.log10((rms_vals + 1e-6) / (rest_rms_ref + 1e-6))
        quality = []
        for i in range(len(rms_vals)):
            if clip_ratio[i] > 1.0 or mains_noise_score[i] > 50.0 or snr_db[i] < 6.0:
                quality.append("POOR")
            elif clip_ratio[i] > 0.3 or mains_noise_score[i] > 30.0 or snr_db[i] < 12.0:
                quality.append("FAIR")
            else:
                quality.append("GOOD")
        return quality

    @staticmethod
    def compute_contraction_quality_score(rms_vals, clip_ratio, mains_noise_score, amp_threshold):
        amp_component = np.clip(rms_vals / (amp_threshold * 1.5 + 1e-9), 0.0, 1.0)
        clip_component = np.clip(1.0 - (clip_ratio / 3.0), 0.0, 1.0)
        mains_component = np.clip(1.0 - (mains_noise_score / 80.0), 0.0, 1.0)
        score = (0.55 * amp_component + 0.20 * clip_component + 0.25 * mains_component) * 100.0
        return score.astype(np.float32)

    def update_fatigue_trend(self, median_freq_mean):
        t_sec = time.time()
        self.fatigue_track.append((t_sec, median_freq_mean))
        if len(self.fatigue_track) > 240:
            self.fatigue_track = self.fatigue_track[-240:]
        if len(self.fatigue_track) < 6:
            return 0.0
        arr = np.array(self.fatigue_track, dtype=np.float64)
        t = arr[:, 0] - arr[0, 0]
        y = arr[:, 1]
        denom = np.sum((t - np.mean(t)) ** 2) + 1e-9
        slope = np.sum((t - np.mean(t)) * (y - np.mean(y))) / denom
        return float(slope)

    def estimate_force_level_pct(self, rms_vals):
        rms_mean = float(np.mean(rms_vals))
        ref = float(np.mean(np.maximum(self.flex_rms_ref, self.rest_rms_ref + 1.0)))
        force_pct = np.clip((rms_mean / (ref + 1e-9)) * 100.0, 0.0, 150.0)
        return float(force_pct)

    def estimate_movement_phase(self, force_level_pct):
        delta = force_level_pct - self.prev_force_level_pct
        self.prev_force_level_pct = force_level_pct
        if force_level_pct < 8.0:
            return "REST"
        if delta > 1.5:
            return "RAMP_UP"
        if delta < -1.5:
            return "RAMP_DOWN"
        return "HOLD"

    def estimate_gesture_label(self, rms_vals, active_flags):
        if float(np.mean(rms_vals)) < self.threshold_spin.value() * 0.8:
            return "REST"
        top_idx = np.argsort(rms_vals)[::-1]
        if len(top_idx) < 2:
            return f"CH{int(top_idx[0]) + 1}_DOMINANT"
        ratio = float(rms_vals[top_idx[0]] / (rms_vals[top_idx[1]] + 1e-6))
        if ratio < 1.2 and np.sum(active_flags) >= 2:
            return "CO_CONTRACTION"
        return f"CH{int(top_idx[0]) + 1}_DOMINANT"

    @staticmethod
    def detect_anomaly_label(clip_ratio, mains_noise_score, baseline_drift, contact_quality):
        if np.max(clip_ratio) > 1.0:
            return "ANOMALY: CLIPPING"
        if np.max(mains_noise_score) > 45.0:
            return "ANOMALY: MAINS_NOISE_HIGH"
        if np.max(np.abs(baseline_drift)) > 180.0:
            return "ANOMALY: BASELINE_DRIFT_HIGH"
        if any(q == "POOR" for q in contact_quality):
            return "ANOMALY: ELECTRODE_CONTACT"
        return "NORMAL"

    @staticmethod
    def compute_dominant_frequency(segment):
        return EMGVisualizer.compute_spectral_features(segment)["peak_hz"]

    def update_plot(self):
        if not self.is_connected or not self.is_calibrated or self.data_buffer is None:
            return

        if self.check_graph_stream.isChecked() and (self._plot_dirty or self._plot_ranges_dirty):
            self._plot_dirty = False
            with self._buffer_lock:
                plot_snapshots = []
                plot_values_by_channel = {}
                for row in self.plot_rows:
                    if row["kind"] == "single":
                        ch = int(row["channel"])
                        vals = self._emg_display_values(ch)
                        plot_snapshots.append(("single", row, vals))
                        plot_values_by_channel[ch] = vals
                    elif row["kind"] == "imu_combined":
                        imu_data = {ch: self.data_buffer[ch].copy() for ch in row["channels"]}
                        plot_snapshots.append(("imu", row, imu_data))
            for kind, row, data in plot_snapshots:
                if kind == "single":
                    if data is not None:
                        row["curve"].setData(self.x_axis, data, skipFiniteCheck=True)
                elif kind == "imu":
                    for curve, ch in zip(row["curves"], row["channels"]):
                        curve.setData(self.x_axis, data[ch], skipFiniteCheck=True)
            now = time.perf_counter()
            range_elapsed_ms = (now - self._last_plot_range_refresh_ts) * 1000.0
            if self._plot_ranges_dirty or range_elapsed_ms >= PLOT_RANGE_REFRESH_MS:
                self.refresh_live_plot_ranges(plot_values_by_channel)
                self._plot_ranges_dirty = False
                self._last_plot_range_refresh_ts = now

        if (not self.task_session_active) and self.live_analysis_enabled and self.analysis_window and self.analysis_window.isVisible():
            now = time.perf_counter()
            analysis_elapsed_ms = (now - self._last_analysis_submit_ts) * 1000.0
            if not self._analysis_pending and analysis_elapsed_ms >= LIVE_ANALYSIS_REFRESH_MS:
                analysis_ms = self.analysis_ms_spin.value()
                win = max(1, min(WINDOW_SIZE, int((analysis_ms / 1000.0) * SAMPLE_RATE)))
                with self._buffer_lock:
                    segment = self.data_buffer[:, -win:].copy()
                heavy_elapsed_ms = (now - self._last_heavy_analysis_submit_ts) * 1000.0
                include_heavy = heavy_elapsed_ms >= LIVE_ANALYSIS_HEAVY_REFRESH_MS
                if include_heavy:
                    self._last_heavy_analysis_submit_ts = now
                self._analysis_pending = True
                self._last_analysis_submit_ts = now
                self._analysis_worker.submit(segment, {
                    "amp_threshold": self.threshold_spin.value(),
                    "hz_threshold": self.hz_threshold_spin.value(),
                    "use_hz_gate": self.check_use_hz_gate.isChecked(),
                    "include_heavy": include_heavy,
                })
        else:
            if self.is_connected and self.is_calibrated:
                self.set_analysis_idle_labels()
            if self.realtime_classification_window and self.realtime_classification_window.isVisible():
                self.update_realtime_classification_state()
        self._tick_ui_fps()

    # ------------------------------------------------------------------ link / counters
    def reset_link_labels(self):
        self.lbl_pkt_rate.setText("Packets: -- /s")
        self.lbl_pkt_loss.setText("Missing: -- (--%)")
        for lbl in (self.lbl_pkt_rate, self.lbl_pkt_loss):
            lbl.setStyleSheet(f"color: {THEME_COLORS['muted']}; font-family: {MONO_FONT_CSS};")

    def on_packet_tick(self):
        """4 Hz: update packet statistics, the tracer window and the live CSV counter."""
        if self.is_connected:
            self.packet_tracer.tick()
            snap = self.packet_tracer.snapshot()
            color = PACKET_KIND_COLORS[packet_health_kind(snap)]
            self.lbl_pkt_rate.setText(f"Packets: {snap['pkt_rate']:.0f}/{snap['expected_pkt_rate']:.0f} /s")
            self.lbl_pkt_loss.setText(f"Missing: {snap['missing']:,} ({snap['loss_pct']:.2f}%)")
            style = f"color: {color}; font-family: {MONO_FONT_CSS};"
            self.lbl_pkt_rate.setStyleSheet(style)
            self.lbl_pkt_loss.setStyleSheet(style)
            if self.packet_tracer_window is not None and self.packet_tracer_window.isVisible():
                self.packet_tracer_window.refresh()
        self.refresh_csv_counter()

    def refresh_csv_counter(self):
        rows = int(self.recorded_sample_count())
        expected = int(self.expected_recorded_samples())
        self.lbl_csv_rows.setText(f"CSV rows: {rows:,}")
        if self.data_collection_dialog is None:
            return
        pct = (100.0 * rows / expected) if expected > 0 else 0.0
        text = f"CSV rows recorded: {rows:,} / {expected:,} expected ({pct:.1f}%)"
        kind = "accent"
        rec = self.session_recorder
        if rec is not None and rec.rows >= 2000 and rec.valid_rows / rec.rows < MIN_RECORD_SAMPLE_RATIO:
            text += f" | VALID {100.0 * rec.valid_rows / rec.rows:.1f}% (< {100 * MIN_RECORD_SAMPLE_RATIO:.0f}%)"
            kind = "warn"
        if self.record_missing_packets > 0:
            text += f" | lost packets: {self.record_missing_packets:,}"
            kind = "warn"
        self.data_collection_dialog.set_csv_counter(text, kind)

    def open_packet_tracer_window(self):
        if self.packet_tracer_window is not None and self.packet_tracer_window.isVisible():
            self.packet_tracer_window.raise_()
            self.packet_tracer_window.activateWindow()
            return
        self.packet_tracer_window = PacketTracerWindow(self.packet_tracer, self)
        self.packet_tracer_window.closed.connect(lambda: setattr(self, "packet_tracer_window", None))
        center_window(self.packet_tracer_window, self)
        self.packet_tracer_window.show()

    def build_options_info_sections(self):
        on_off = lambda flag: "On" if flag else "Off"
        hands = " + ".join(hands_for_order(self.study_meta["hand_order"]))
        link = "Disconnected"
        if self.is_connected:
            link = f"{self.connection_medium.title()} | {self.current_port_name}"
        model = os.path.basename(self.rf_model_path) if self.rf_model_path else "Not loaded"
        return [
            ("Connection & calibration", [
                ("Port Configuration", link, "Choose wired (USB serial / simulator socket) or wireless (Wi-Fi) transport, "
                 "scan for devices, provision Wi-Fi over USB and connect."),
                ("Disconnect", "Enabled" if self.is_connected else "Disabled", "Stops the stream and releases the port."),
                ("Channel Calibration", "Done" if self.is_calibrated else "Required",
                 f"Captures REST then FLEX ({self.cal_rest_seconds} s / {self.cal_flex_seconds} s) to set baselines "
                 "and thresholds. Graph, analysis, recording and classification stay locked until it is done."),
                ("Channels", str(self.num_channels), "Wired CSV simulator: 2-11 channels (set in the calibration dialog); COM port and Wi-Fi: fixed 11. "
                 "Wireless: 8 EMG + 3 IMU channels."),
            ]),
            ("Toolbar", [
                ("Analysis (ms)", f"{self.analysis_ms_spin.value()} ms", "Window length used for MAV / RMS and the live-analysis metrics."),
                ("RMS Threshold", f"{self.threshold_spin.value():.1f}", "Amplitude level above which a channel counts as active (dashed lines on the graph)."),
                ("Hz Threshold", f"{self.hz_threshold_spin.value():.1f} Hz", "Minimum dominant frequency for a channel to count as a real contraction."),
                ("Use Hz Gate", on_off(self.check_use_hz_gate.isChecked()), "Require both RMS and frequency thresholds before a channel is 'active'."),
                ("Auto Scale Y", on_off(self.check_autoscale.isChecked()), "Fit each channel's vertical range to its live signal; off uses a fixed range."),
                ("Graph Stream", on_off(self.check_graph_stream.isChecked()), "Draw the live traces. It pauses itself during a recording session for stable capture."),
                ("Live Analysis", "Available" if self.is_calibrated else "Locked", "Opens the metrics window: time/frequency features, correlation, coherence, fatigue and quality."),
            ]),
            ("Monitoring", [
                ("Packet Tracer", "Open" if self.packet_tracer_window else "Closed", "Real-time packets per second, missing and out-of-order packets, loss %, jitter and a 30 s plot."),
                ("Link panel (right side)", "Live", "Compact packets/s, missing packets and CSV row counter beside the graph."),
                ("UI FPS / Data FPS", f"{self.ui_fps:.0f} / {self.data_fps:.0f}", "Screen refresh rate and the sample rate actually arriving."),
                ("Options Info", "This window", "Lists every option with its live value."),
            ]),
            ("Data Collection", [
                ("Participant ID", self.participant_id or "-", "Pseudonymous ID (S01...); used in the output folder name. Names are never stored."),
                ("Agreement", "Yes" if self.contribution_agreed else "No", "Consent to take part; required before a session can start."),
                ("Participant info", f"BG {self.participant_blood_group or 'N/A'}, {_value_or_na(self.participant_height_cm, '{:.1f}')} cm, "
                 f"{_value_or_na(self.participant_weight_kg, '{:.1f}')} kg, age {_value_or_na(self.participant_age_years, '{:d}', cast=int)}",
                 "Optional; saved to the metadata file (N/A when empty)."),
                ("Forearm circumference", f"{_value_or_na(self.participant_forearm_circumference_cm, '{:.2f}')} cm", "Measured around the forearm where the armband sits."),
                ("Session day", str(self.study_meta["session_day"]), "Study session 1 or 2 (two days, 1-7 days apart)."),
                ("Hands", hands, "Order of the left/right hand blocks; the armband is moved between them."),
                ("Class labels", ", ".join(self.task_labels), "Gesture classes recorded; the rest period is its own class."),
                ("Repeats per hand", str(self.task_repeats), "Repetitions of every class per hand (study: 9)."),
                ("Prepare / Perform / Rest", f"{self.task_prep_s:g} / {self.task_hold_s:g} / {self.task_rest_s:g} s", "One task cycle (study: 4 + 3 + 3 = 10 s)."),
                ("Settle", f"{self.task_settle_s:g} s", "Optional un-recorded gap after each hold."),
                ("Max cycles per block", str(self.study_meta["max_cycles_per_block"]), "Longest run without a break (study: 30 cycles, ~5 min)."),
                ("Block break / hand-switch break", f"{self.study_meta['block_break_s']:.0f} / {self.study_meta['hand_switch_s']:.0f} s", "Rest between blocks (3-4 min) and when the armband changes arm (~5 min). 'Skip Break' ends one early."),
                ("Record Rest / Randomize order", f"{on_off(self.task_record_rest)} / {on_off(self.task_randomize_order)}", "Record the rest period as a class; shuffle gesture order each repetition."),
                ("CSV save folder", self.record_save_dir, "Where the session CSV and metadata are written; a checkpoint is saved at every break."),
                ("Live CSV counter", f"{self.recorded_sample_count():,} rows", "Rows captured so far versus the number the protocol should produce."),
            ]),
            ("Models", [
                ("Train Model", "-", "Train a Random Forest from recorded CSV files and review accuracy and the confusion matrix."),
                ("Load Trained Model", model, "Load a saved model for real-time prediction."),
                ("Real-time Classification", "Open" if self.realtime_classification_window else "Closed", "Shows the live predicted gesture, confidence and per-class scores."),
                ("Gesture game", "Separate app", f"Run gesture_game_app.py; it listens to this app on UDP {GESTURE_BRIDGE_PORT} (keyboard works without it)."),
            ]),
        ]

    def open_options_info_dialog(self):
        if self.options_info_dialog is not None and self.options_info_dialog.isVisible():
            self.options_info_dialog.raise_()
            self.options_info_dialog.activateWindow()
            return
        self.options_info_dialog = OptionsInfoDialog(self.build_options_info_sections(), self)
        center_window(self.options_info_dialog, self)
        self.options_info_dialog.show()

    def on_serial_error(self, message):
        QMessageBox.critical(self, "Stream Error", message)
        self.disconnect_serial()

    def closeEvent(self, event):
        for worker in list(self.record_save_workers):
            try:
                worker.wait()
            except Exception:
                pass
        self.record_save_workers = []
        for attr in ("packet_tracer_window", "options_info_dialog"):
            win = getattr(self, attr, None)
            if win is not None:
                try:
                    win.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        self.packet_timer.stop()
        self.gesture_bridge.close()
        if self.realtime_classification_window is not None:
            try:
                self.realtime_classification_window.close()
            except Exception:
                pass
            self.realtime_classification_window = None
        if self.rf_worker is not None:
            try:
                self.rf_worker.stop()
            except Exception:
                pass
            self.rf_worker = None
        if self._analysis_worker is not None:
            try:
                self._analysis_worker.stop()
            except Exception:
                pass
            self._analysis_worker = None
        self.disconnect_serial()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self.apply_plot_row_layout)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._centered_once:
            center_window(self, None)
            self._centered_once = True


if __name__ == "__main__":
    if hasattr(pg, "setConfigOptions"):
        pg.setConfigOptions(antialias=ENABLE_ANTIALIAS, useOpenGL=ENABLE_OPENGL)

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("BioWave.EMG.App")
        except Exception:
            pass

    configure_high_dpi()
    app = QApplication(sys.argv)
    app_icon = get_app_icon()
    if app_icon is not None and not app_icon.isNull():
        app.setWindowIcon(app_icon)
    # Font scales with the screen (13" MacBook vs 1080p/4K Windows) so layouts keep their proportions.
    apply_dark_theme(app, font_size=recommended_font_px())
    install_screen_fit_filter(app)
    window = EMGVisualizer()
    window.show()
    sys.exit(app.exec_())
