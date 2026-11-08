"""
BioWave - Standalone Adaptive Mouse Controller
================================================
A fully self-sufficient companion to the BioWave EMG data-collection app
(main.py). It does NOT import or require main.py - it has its own wired
(USB/serial) and wireless (Wi-Fi) connectivity, its own REST/FLEX calibration
workflow, and its own real-time RF inference pipeline built on top of the
shared, unit-tested core modules (rf_features.py for feature extraction,
emg_v4_core.py for calibration/signal-quality/decision logic, and
realtime_pipeline.py for the sample ring buffer). Point it at a pretrained
`.joblib` model (trained in main.py) and it goes straight from "connect" to
"controlling the mouse" - no graphing, data collection, or training UI.

This is the single mouse-controller entry point; it replaces the former
Mouse_wireless_v2.py and Mouse_wireless_standalone_v3.py forks (see git
history / code/LAB_SUITE_CHANGES.md for that evolution).

Requires: PyQt5, numpy, joblib, pyserial, and pyautogui for actually moving
the cursor.
"""

import sys
import os
import time
import threading
import json
import socket
import struct
import subprocess
import hashlib
import hmac
import logging
from dataclasses import dataclass
from collections import deque
from urllib.parse import quote

import numpy as np
import joblib
import serial
import serial.tools.list_ports

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QMessageBox, QFileDialog, QFrame,
    QGroupBox, QSpinBox, QDoubleSpinBox, QScrollArea, QTabWidget, QDialog,
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont

# Cross-platform theme (fonts/colors adapt automatically on Windows/macOS/Linux;
# apply_dark_title_bar() is a no-op on non-Windows platforms).
from app_theme import app_stylesheet, apply_dark_title_bar, configure_high_dpi, THEME_COLORS

# biowave_lab_suite merges performance_logger.py, performance_analysis.py,
# publication_figures.py, experiment_manager.py, data_tools_ui.py,
# controller_adapters.py, async_csv.py, and iso_9241_9_task.py into one tool.
from biowave_lab_suite import PerformanceLogger, PerformanceTarget, AnalysisSuiteWindow
from emg_v4_core import (
    FEATURE_EXTRACTOR_VERSION, LEGACY_PREPROCESSING_VERSION, PREPROCESSING_VERSION,
    CalibrationProfile, GestureDecisionEngine, PreprocessingConfig, RealTimePreprocessor,
    SampleBatch, SignalQuality, WirelessStats, assess_signal_quality, compute_calibration,
    expected_feature_count, validate_model_artifact,
)
from realtime_pipeline import SampleRingBuffer, StageProfiler
# The single canonical feature extractor, shared with the trainer in main.py.
# Controller and trainer therefore cannot drift feature ordering.
from rf_features import extract_window_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOG = logging.getLogger("biowave.mouse")

HAS_RF_FEATURES = True

# Mouse control library.
try:
    import pyautogui
    pyautogui.FAILSAFE = True  # Slam the mouse to a screen corner to abort.
    HAS_PYAUTOGUI = True
except Exception:
    pyautogui = None
    HAS_PYAUTOGUI = False


# ============================== CONSTANTS ================================

DEFAULT_BAUD_RATE = 921600         # Wired EMG stream baud rate.
USB_SERIAL_BAUD = 115200           # Baud rate used only for USB provisioning handshakes.
SERIAL_BOOT_WAIT_S = 3.5
SERIAL_RESPONSE_TIMEOUT_S = 15.0
SAMPLE_RATE = 500                  # Hz, matches the ESP32 firmware.
WINDOW_SIZE = 1000                 # Rolling sample buffer length (channels x samples).
DEFAULT_WIRED_CHANNELS = 4

PLACEHOLDER_DEVICE_ACCESS_KEY = "CHANGE_THIS_TO_A_LONG_RANDOM_KEY"
MIN_DEVICE_ACCESS_KEY_LEN = 16
CONTROL_RETRY_BACKOFF_S = (0.2, 0.5, 1.0)   # waits before retry 1..3 of a control command
DEVICE_SECRET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "device_secret.json")
GESTURE_BRIDGE_HOST = "127.0.0.1"   # Predictions are mirrored here for gesture_game_app.py
GESTURE_BRIDGE_PORT = 5005
WIFI_STREAM_PORT = 5000
WIFI_CONTROL_PORT = 5001
DISCOVERY_ADDRESS = "255.255.255.255"
DISCOVERY_TIMEOUT = 1.2
CONTROL_TIMEOUT = 2.0
KEEPALIVE_INTERVAL_MS = 2000       # Matches main app: periodic PING keeps the ESP32 stream alive.
KEEPALIVE_MAX_FAILURES = 3         # Consecutive failed pings before we flag the link as lost.

WIRELESS_EMG_CHANNELS = 8
WIRELESS_IMU_CHANNELS = 3
WIRELESS_TOTAL_CHANNELS = WIRELESS_EMG_CHANNELS + WIRELESS_IMU_CHANNELS
WIFI_PACKET_HEADER_FORMAT = "<4sBBHI"
WIFI_PACKET_HEADER_SIZE = struct.calcsize(WIFI_PACKET_HEADER_FORMAT)
WIRELESS_FRAME_FORMAT = "<IIII8HfffB3x"
WIRELESS_FRAME_SIZE = struct.calcsize(WIRELESS_FRAME_FORMAT)
WIRELESS_FRAMES_PER_PACKET = 5
WIRELESS_PACKET_SIZE = WIFI_PACKET_HEADER_SIZE + (WIRELESS_FRAME_SIZE * WIRELESS_FRAMES_PER_PACKET)

CAL_TICK_MS = 100
CAL_REST_MS = 3000
CAL_FLEX_MS = 3000
CAL_DURATION_MIN_S = 3
CAL_DURATION_MAX_S = 10
BASE_ADAPT_ALPHA = 0.001           # Slow baseline drift compensation.
BASE_ADAPT_GUARD = 80.0            # Only adapt baseline while signal is near rest.
RF_LABEL_SMOOTH_WINDOW = 5         # Majority-vote smoothing window for displayed label.
GESTURE_MIN_CONFIDENCE_DEFAULT = 65.0
MAX_INFERENCE_PACKET_GAP = 1        # A larger UDP loss invalidates a window; never classify it.

MOUSE_ACTIONS = [
    "Ignore",
    "Move Up", "Move Down", "Move Left", "Move Right",
    "Left Click", "Right Click", "Double Click",
]


# ============================ HELPER FUNCTIONS ============================

def load_device_access_key(path=DEVICE_SECRET_PATH):
    """Key from the gitignored device_secret.json, or "" when missing/unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return str(json.load(fh).get("device_access_key", "")).strip()
    except (OSError, ValueError, AttributeError):
        return ""


def access_key_problem(key):
    """Why ``key`` cannot be used with a device, or "" when acceptable."""
    key = str(key or "").strip()
    if not key:
        return "No device access key is set. Put it in code/device_secret.json or type it into the Access Key box."
    if key == PLACEHOLDER_DEVICE_ACCESS_KEY:
        return "The device access key is still the placeholder. Use the same key as in the firmware's secrets.h."
    if len(key) < MIN_DEVICE_ACCESS_KEY_LEN:
        return f"The device access key is too short (minimum {MIN_DEVICE_ACCESS_KEY_LEN} characters)."
    return ""


def local_broadcast_addresses():
    """Subnet-broadcast address of every local IPv4 adapter plus the limited broadcast.

    On Windows a 255.255.255.255 datagram leaves through ONE adapter only (often a VPN or
    virtual switch), so the ESP32 on the real Wi-Fi never sees DISCOVER. Sending to each
    adapter's /24 broadcast fixes that.
    """
    addrs = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip.startswith("127.") or ip.startswith("169.254."):
                continue
            addrs.append(".".join(ip.split(".")[:3] + ["255"]))
    except OSError:
        pass
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("192.0.2.1", 1))
            addrs.append(".".join(probe.getsockname()[0].split(".")[:3] + ["255"]))
        finally:
            probe.close()
    except OSError:
        pass
    addrs.append(DISCOVERY_ADDRESS)
    return list(dict.fromkeys(addrs))


def _parse_rate_field(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


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


class GesturePublisher:
    """Mirrors each prediction to localhost UDP for gesture_game_app.py (never raises)."""

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


# ================================ DATACLASSES ==============================

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
    sample_rate_hz: object = None

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
    sample_rate_hz: object = None


# ============================ WIRELESS PROTOCOL =============================

class ControlProtocol:
    """UDP control-plane protocol for discovering and driving the wireless
    BioWave EMG device (mirrors the ESP32 firmware's HELLO/CHALLENGE/START/STOP)."""

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
    def discover(extra_hosts=()):
        """Broadcast DISCOVER on every local subnet and unicast to any ``extra_hosts``."""
        devices = {}
        targets = list(extra_hosts) + local_broadcast_addresses()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(0.2)
        try:
            for _attempt in range(2):          # UDP is lossy; send twice
                for target in targets:
                    try:
                        sock.sendto(b"DISCOVER", (target, WIFI_CONTROL_PORT))
                    except OSError:
                        continue
                deadline = time.monotonic() + DISCOVERY_TIMEOUT
                while time.monotonic() < deadline:
                    try:
                        data, addr = sock.recvfrom(2048)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    try:
                        device = ControlProtocol.parse_device_info(
                            data.decode("utf-8", errors="replace"), addr[0])
                        devices[device.ip] = device
                    except ValueError:
                        continue
        finally:
            sock.close()
        return list(devices.values())

    @staticmethod
    def _authenticated_once(target_ip, secret, command, payload, timeout):
        # The firmware ties a challenge to the sender's (ip, port), so CHALLENGE and the
        # signed command must leave from the SAME socket.
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
            raise RuntimeError(f"Device rejected command: {parts[1] if len(parts) > 1 else 'UNKNOWN'}")
        if parts[0] != "ACK":
            raise RuntimeError("Unexpected device acknowledgement.")
        return parts[1:]

    @staticmethod
    def authenticated_command(target_ip, secret, command, *payload, timeout=CONTROL_TIMEOUT,
                              backoff=CONTROL_RETRY_BACKOFF_S):
        """Signed command with retries (fresh challenge each attempt) for lost UDP datagrams."""
        problem = access_key_problem(secret)
        if problem:
            raise RuntimeError(problem)
        last_error = None
        for attempt in range(len(backoff) + 1):
            if attempt:
                time.sleep(backoff[attempt - 1])
            try:
                return ControlProtocol._authenticated_once(target_ip, secret, command, payload, timeout)
            except (OSError, RuntimeError) as exc:
                last_error = exc
                if isinstance(exc, RuntimeError) and "rejected" in str(exc):
                    break                      # wrong key etc.: retrying won't help
        raise RuntimeError(f"{command} failed: {last_error}")

    @staticmethod
    def start_stream(target_ip, secret, client_ip, client_port):
        return ControlProtocol.authenticated_command(target_ip, secret, "START", client_ip, str(client_port))

    @staticmethod
    def stop_stream(target_ip, secret):
        return ControlProtocol.authenticated_command(target_ip, secret, "STOP")

    @staticmethod
    def ping(target_ip, secret):
        return ControlProtocol.authenticated_command(target_ip, secret, "PING")


class WiFiSerialProvisionProtocol:
    """One-time USB handshake used to hand Wi-Fi credentials to a fresh ESP32."""

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
            port_name=port_name, device_id=parts[1], device_name=parts[2],
            imu_ready=parts[3] == "1", wifi_saved=parts[4] == "1", firmware=parts[5],
            sample_rate_hz=_parse_rate_field(parts[6]) if len(parts) > 6 else None,
        )

    @staticmethod
    def provision(port_name, ssid, password):
        encoded_ssid = quote(ssid, safe="")
        encoded_password = quote(password, safe="")
        response = WiFiSerialProvisionProtocol._exchange_line(
            port_name, f"PROVISION|{encoded_ssid}|{encoded_password}",
            expected_prefixes=("ACK|", "ERR|"), timeout=8.0,
        )
        parts = response.split("|")
        if len(parts) >= 2 and parts[0] == "ACK" and parts[1] == "PROVISIONED":
            return
        if len(parts) >= 2 and parts[0] == "ERR":
            raise RuntimeError(f"ESP32 rejected provisioning: {parts[1]}")
        raise RuntimeError(f"Unexpected serial response: {response}")


# ============================== BACKGROUND THREADS ==========================

class SerialWorker(QThread):
    """Reads a live EMG stream over USB/COM serial (or a socket:// simulator)."""
    batch_received = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, port_name, baud_rate, num_channels, batch_size=25):
        super().__init__()
        self.port_name = port_name
        self.baud_rate = baud_rate
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.is_socket_url = str(port_name).strip().lower().startswith("socket://")
        self._running = True
        self._serial = None

    def _close_serial(self):
        try:
            if self._serial and self._serial.is_open:
                self._serial.close()
        except Exception:
            pass
        self._serial = None

    def run(self):
        partial_line = ""
        batch = []
        try:
            while self._running:
                try:
                    if self._serial is None or not self._serial.is_open:
                        self._serial = serial.serial_for_url(self.port_name, self.baud_rate, timeout=0.02)
                        try:
                            self._serial.reset_input_buffer()
                        except Exception:
                            pass
                        partial_line = ""

                    waiting = self._serial.in_waiting
                    chunk = self._serial.read(waiting if waiting else 1)
                    if not chunk:
                        time.sleep(0.001)
                        continue

                    partial_line += chunk.decode("utf-8", errors="ignore")
                    lines = partial_line.split("\n")
                    partial_line = lines.pop()

                    for raw_line in lines:
                        line = raw_line.strip()
                        if not line:
                            continue
                        parts = line.replace(",", " ").split()
                        if len(parts) < self.num_channels:
                            continue
                        try:
                            vals = [float(parts[i]) for i in range(self.num_channels)]
                        except ValueError:
                            continue
                        batch.append(vals)
                        if len(batch) >= self.batch_size:
                            self.batch_received.emit(np.asarray(batch, dtype=np.float32))
                            batch = []
                except Exception as e:
                    if not self._running:
                        break
                    self._close_serial()
                    partial_line = ""
                    if self.is_socket_url:
                        time.sleep(0.3)
                        continue
                    self.error_occurred.emit(str(e))
                    break
        finally:
            self._close_serial()

    def stop(self):
        self._running = False
        self.wait()


class WirelessStreamWorker(QThread):
    """Receives the UDP EMG+IMU packet stream from a wireless BioWave device."""
    batch_received = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, port=WIFI_STREAM_PORT):
        super().__init__()
        self.port = int(port)
        self._running = True
        self._sock = None
        self._fallback_packet_sequence = 0
        self.stats = WirelessStats()

    def _decode_frames(self, payload, count):
        rows, frame_ids, frame_ts, imu_ids, imu_ts = [], [], [], [], []
        for offset in range(0, count * WIRELESS_FRAME_SIZE, WIRELESS_FRAME_SIZE):
            frame = payload[offset: offset + WIRELESS_FRAME_SIZE]
            frame_id, emg_ts, imu_id, imu_ts_value, *frame_fields = struct.unpack(WIRELESS_FRAME_FORMAT, frame)
            row = [float(v) for v in frame_fields[:WIRELESS_EMG_CHANNELS]]
            row.extend([float(frame_fields[8]), float(frame_fields[9]), float(frame_fields[10])])
            rows.append(row)
            frame_ids.append(frame_id); frame_ts.append(emg_ts)
            imu_ids.append(imu_id); imu_ts.append(imu_ts_value)
        return rows, frame_ids, frame_ts, imu_ids, imu_ts

    def _parse_datagram(self, data):
        if len(data) == WIRELESS_PACKET_SIZE:
            magic, version, frame_count, frame_size, packet_sequence = struct.unpack(
                WIFI_PACKET_HEADER_FORMAT, data[:WIFI_PACKET_HEADER_SIZE]
            )
            if magic != b"BWIM" or version != 1 or frame_size != WIRELESS_FRAME_SIZE or frame_count != WIRELESS_FRAMES_PER_PACKET:
                self.stats.invalid_packets += 1
                return None
            payload = data[WIFI_PACKET_HEADER_SIZE:]
            rows, frame_ids, frame_ts, imu_ids, imu_ts = self._decode_frames(payload, WIRELESS_FRAMES_PER_PACKET)
            batch = np.asarray(rows, dtype=np.float32)
            packet_numbers = np.full(batch.shape[0], int(packet_sequence), dtype=np.int64)
            arrival = time.monotonic()
            return SampleBatch(batch, packet_numbers, np.asarray(frame_ids), np.asarray(frame_ts),
                               np.asarray(imu_ids), np.asarray(imu_ts), arrival,
                               gap_before=self.stats.observe(packet_sequence, arrival))

        if len(data) == (WIRELESS_FRAME_SIZE * WIRELESS_FRAMES_PER_PACKET):
            rows, frame_ids, frame_ts, imu_ids, imu_ts = self._decode_frames(data, WIRELESS_FRAMES_PER_PACKET)
            batch = np.asarray(rows, dtype=np.float32)
            packet_no = int(self._fallback_packet_sequence)
            self._fallback_packet_sequence += 1
            packet_numbers = np.full(batch.shape[0], packet_no, dtype=np.int64)
            return SampleBatch(batch, packet_numbers, np.asarray(frame_ids), np.asarray(frame_ts),
                               np.asarray(imu_ids), np.asarray(imu_ts), time.monotonic())

        if len(data) == WIRELESS_FRAME_SIZE:
            rows, frame_ids, frame_ts, imu_ids, imu_ts = self._decode_frames(data, 1)
            batch = np.asarray(rows, dtype=np.float32)
            packet_no = int(self._fallback_packet_sequence)
            self._fallback_packet_sequence += 1
            packet_numbers = np.full(batch.shape[0], packet_no, dtype=np.int64)
            return SampleBatch(batch, packet_numbers, np.asarray(frame_ids), np.asarray(frame_ts),
                               np.asarray(imu_ids), np.asarray(imu_ts), time.monotonic())

        return None

    def run(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", self.port))
        self._sock.settimeout(1.0)
        try:
            while self._running:
                try:
                    data, _addr = self._sock.recvfrom(2048)
                except socket.timeout:
                    continue
                except OSError:
                    break
                payload = self._parse_datagram(data)
                if payload is None:
                    continue
                batch = np.asarray(payload.samples, dtype=np.float32)
                if batch.size > 0:
                    self.batch_received.emit(payload)
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
        self.wait()


class InferenceWorker(QThread):
    """Extracts features and runs the pretrained Random Forest prediction."""
    # label, confidence (None if unavailable), top-1/top-2 margin, inference ms
    prediction_ready = pyqtSignal(str, object, object, float)
    inference_error = pyqtSignal(str)

    def __init__(self, sample_rate):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.model = None
        self.class_names = []
        self._window = None
        self._running = True
        self._lock = threading.Lock()
        self._event = threading.Event()

    def load_model(self, model, class_names, sample_rate=None):
        with self._lock:
            self.model = model
            self.class_names = [str(x) for x in list(class_names or [])]
            if sample_rate is not None:
                self.sample_rate = int(max(1, sample_rate))
            self._window = None
        self._event.clear()

    def clear_model(self):
        with self._lock:
            self.model = None
            self.class_names = []
            self._window = None
        self._event.clear()

    def submit_window(self, window):
        with self._lock:
            self._window = window
        self._event.set()

    def run(self):
        while self._running:
            self._event.wait(0.1)
            if not self._running:
                break
            if not self._event.is_set():
                continue
            self._event.clear()

            with self._lock:
                win = self._window
                model = self.model
                classes = list(self.class_names)
                self._window = None

            if win is None or model is None or not HAS_RF_FEATURES:
                continue

            try:
                started = time.monotonic()
                feats = extract_window_features(win, sample_rate=self.sample_rate).reshape(1, -1)
                pred_label = "N/A"
                conf = None
                margin = None

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(feats)[0]
                    confidences = np.zeros(len(classes), dtype=np.float32)
                    model_classes = list(getattr(model, "classes_", []))
                    if len(model_classes) == len(proba) and len(classes) > 0:
                        for i, cls_id in enumerate(model_classes):
                            idx = -1
                            try:
                                idx = int(cls_id)
                            except Exception:
                                cls_text = str(cls_id)
                                if cls_text in classes:
                                    idx = classes.index(cls_text)
                            if 0 <= idx < len(confidences):
                                confidences[idx] = float(proba[i])
                        pred_idx = int(np.argmax(confidences)) if np.max(confidences) > 0 else int(np.argmax(proba))
                        pred_label = classes[pred_idx] if 0 <= pred_idx < len(classes) else str(model_classes[int(np.argmax(proba))])
                        conf = float(np.max(confidences)) if np.max(confidences) > 0 else float(np.max(proba))
                        sorted_p = np.sort(proba)
                        margin = float(sorted_p[-1] - sorted_p[-2]) if len(sorted_p) > 1 else float(sorted_p[-1])
                    else:
                        pred_idx = int(np.argmax(proba))
                        pred_label = classes[pred_idx] if 0 <= pred_idx < len(classes) else str(pred_idx)
                        conf = float(proba[pred_idx])
                        sorted_p = np.sort(proba)
                        margin = float(sorted_p[-1] - sorted_p[-2]) if len(sorted_p) > 1 else float(sorted_p[-1])
                else:
                    pred_raw = model.predict(feats)[0]
                    if isinstance(pred_raw, (int, np.integer)) and 0 <= int(pred_raw) < len(classes):
                        pred_label = classes[int(pred_raw)]
                    else:
                        pred_label = str(pred_raw)
                    # A classifier without probabilities does not provide a confidence.
                    # Safety gating consequently holds it in UNKNOWN rather than inventing 100%.
                    conf = None

                self.prediction_ready.emit(pred_label, conf, margin, (time.monotonic() - started) * 1000.0)
            except Exception as e:
                LOG.exception("Inference error")
                self.inference_error.emit(str(e))

    def stop(self):
        self._running = False
        self._event.set()
        self.wait()


# ================================ DIALOGS ==================================

class ProvisionDialog(QDialog):
    """One-off USB step: hand Wi-Fi credentials to a fresh ESP32 device."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Provision Wireless Device (USB)")
        self.resize(480, 300)
        self.setModal(True)

        layout = QVBoxLayout(self)
        intro = QLabel("Connect the ESP32 over USB, pick its port, then send your Wi-Fi credentials.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("USB Port:"))
        self.combo_port = QComboBox()
        port_row.addWidget(self.combo_port, 1)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh_ports)
        port_row.addWidget(btn_refresh)
        layout.addLayout(port_row)

        btn_info = QPushButton("Query Device Info")
        btn_info.clicked.connect(self.query_info)
        layout.addWidget(btn_info)

        self.lbl_info = QLabel("No device queried yet.")
        self.lbl_info.setWordWrap(True)
        layout.addWidget(self.lbl_info)

        form = QFormLayout()
        self.txt_ssid = QLineEdit()
        form.addRow("Wi-Fi SSID:", self.txt_ssid)
        self.txt_password = QLineEdit()
        self.txt_password.setEchoMode(QLineEdit.Password)
        form.addRow("Wi-Fi Password:", self.txt_password)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.btn_send = QPushButton("Send Credentials")
        self.btn_send.clicked.connect(self.send_credentials)
        btn_row.addWidget(self.btn_send)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        self.refresh_ports()

    def refresh_ports(self):
        self.combo_port.clear()
        for p in WiFiSerialProvisionProtocol.available_ports():
            self.combo_port.addItem(f"{p.device} - {p.description}", p.device)

    def _selected_port(self):
        data = self.combo_port.currentData()
        if data:
            return data
        text = self.combo_port.currentText()
        return text.split()[0] if text else ""

    def query_info(self):
        port = self._selected_port()
        if not port:
            QMessageBox.warning(self, "No Port", "Select a USB port first.")
            return
        try:
            info = WiFiSerialProvisionProtocol.query_info(port)
            self.lbl_info.setText(
                f"{info.device_name} | FW={info.firmware} | IMU ready={info.imu_ready} | Wi-Fi saved={info.wifi_saved}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Query Failed", str(e))

    def send_credentials(self):
        port = self._selected_port()
        ssid = self.txt_ssid.text().strip()
        password = self.txt_password.text()
        if not port or not ssid:
            QMessageBox.warning(self, "Missing Info", "Select a port and enter an SSID.")
            return
        try:
            WiFiSerialProvisionProtocol.provision(port, ssid, password)
            QMessageBox.information(self, "Provisioned", "Wi-Fi credentials sent. The device will reboot onto your network.")
        except Exception as e:
            QMessageBox.critical(self, "Provisioning Failed", str(e))


class CalibrationDialog(QDialog):
    """Guides the user through a REST -> FLEX capture used to zero the baseline."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.resize(420, 220)
        self.setModal(True)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)

        layout = QVBoxLayout(self)
        self.lbl_phase = QLabel("REST")
        self.lbl_phase.setAlignment(Qt.AlignCenter)
        f = QFont()
        f.setPointSize(22)
        f.setBold(True)
        self.lbl_phase.setFont(f)
        layout.addWidget(self.lbl_phase)

        self.lbl_instruction = QLabel("")
        self.lbl_instruction.setWordWrap(True)
        self.lbl_instruction.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_instruction)

        self.lbl_countdown = QLabel("")
        self.lbl_countdown.setAlignment(Qt.AlignCenter)
        cf = QFont()
        cf.setPointSize(16)
        self.lbl_countdown.setFont(cf)
        layout.addWidget(self.lbl_countdown)

        self.btn_cancel = QPushButton("Cancel")
        layout.addWidget(self.btn_cancel)

    def set_phase(self, name, instruction, remaining_ms, total_ms):
        self.lbl_phase.setText(name)
        self.lbl_instruction.setText(instruction)
        self.lbl_countdown.setText(f"{max(0, remaining_ms) / 1000.0:0.1f}s remaining")

    def set_finished(self, summary):
        self.lbl_phase.setText("Done")
        self.lbl_instruction.setText(summary)
        self.lbl_countdown.setText("")
        self.btn_cancel.setText("Close")


# ============================== MAIN APPLICATION ============================

class MouseControllerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BioWave - Standalone Mouse Controller")
        # A resizable window (not a fixed size) plus the QScrollArea wrapper
        # added in init_ui() means every control stays reachable no matter
        # what display ratio/scaling a MacBook (or any screen) uses - nothing
        # gets clipped, it just scrolls.
        self.resize(1040, 660)
        self.setMinimumSize(760, 520)
        self.apply_dark_theme()
        self._lab_suite_window = None

        # --- connection state ---
        self.serial_worker = None
        self.connection_medium = ""   # "wired" | "wireless"
        self.is_connected = False
        self.current_device = None
        self.discovered_devices = []
        self.wireless_access_key = load_device_access_key()
        self.num_channels = DEFAULT_WIRED_CHANNELS
        self.emg_channel_count = DEFAULT_WIRED_CHANNELS

        # --- model state ---
        self.model_loaded = False
        self.rf_model_path = ""
        self.rf_class_names = []
        self.rf_window_samples = 100
        self.rf_stride_samples = 25
        self.rf_model_input_channels = DEFAULT_WIRED_CHANNELS
        self.rf_model_sample_rate = SAMPLE_RATE
        self.rf_preprocessing_version = LEGACY_PREPROCESSING_VERSION
        self.model_compatible = False
        self.model_compatibility_message = "No model loaded."
        self.model_expected_feature_count = None

        # --- calibration state ---
        self.is_calibrated = False
        self.calibration_active = False
        self.calibration_dialog = None
        self.calibration_timer = QTimer(self)
        self.calibration_timer.timeout.connect(self.on_calibration_tick)
        self.keepalive_timer = QTimer(self)
        self.keepalive_timer.timeout.connect(self.send_wireless_keepalive)
        self.keepalive_timer.start(KEEPALIVE_INTERVAL_MS)
        self.keepalive_failures = 0
        self.cal_rest_seconds = CAL_REST_MS // 1000
        self.cal_flex_seconds = CAL_FLEX_MS // 1000
        self.calibration_phases = []
        self.current_cal_phase_idx = -1
        self.current_phase_key = ""
        self.current_phase_remaining_ms = 0
        self.current_phase_total_ms = 0
        self.rest_capture = []
        self.flex_capture = []
        self.baseline_offsets = np.zeros(1, dtype=np.float32)
        self.calibration_profile = None
        self.preprocessor = None

        # --- streaming buffer state ---
        self.data_buffer = None            # (channels, WINDOW_SIZE), baseline-centered
        self.sample_ring = None            # canonical ring: (samples, channels)
        self.pipeline_profiler = StageProfiler()
        self._buffer_lock = threading.RLock()
        self.rf_valid_sample_count = 0
        self.rf_samples_since_submit = 0
        self.rf_last_pred_label = "N/A"
        self.rf_last_pred_conf = 0.0
        self.rf_label_history = deque(maxlen=RF_LABEL_SMOOTH_WINDOW)
        self.last_signal_quality = None
        self.last_batch_received_monotonic = 0.0
        self.stream_invalid_until = 0.0
        self.decision_engine = GestureDecisionEngine()
        self.gesture_bridge = GesturePublisher()

        # --- mouse control state ---
        self.class_action_map = {}
        self.mapping_combos = []
        self.mouse_control_active = False
        self.last_click_time = 0.0
        self.performance_logger = PerformanceLogger()

        self.init_ui()

        self.inference_worker = InferenceWorker(SAMPLE_RATE)
        self.inference_worker.prediction_ready.connect(self.on_prediction_ready)
        self.inference_worker.inference_error.connect(self.on_inference_error)
        self.inference_worker.start()

    # ---------------------------------------------------------------- theme
    def apply_dark_theme(self):
        # app_stylesheet() picks a native-looking font per OS (Segoe UI on
        # Windows, the system Sans Serif family - San Francisco - on macOS,
        # etc.) instead of hardcoding a Windows-only font family.
        self.setStyleSheet(app_stylesheet(13))
        apply_dark_title_bar(self)  # no-op on macOS/Linux, dark titlebar on Windows

    # ------------------------------------------------------------------ UI
    def init_ui(self):
        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setSpacing(14)

        # Wrapping the whole layout in a resizable QScrollArea means that
        # shrinking the window (any MacBook display ratio, split-screen,
        # external-monitor scaling, etc.) never hides a control behind the
        # window edge - the content scrolls instead of clipping.
        outer_scroll = QScrollArea()
        outer_scroll.setWidgetResizable(True)
        outer_scroll.setFrameShape(QFrame.NoFrame)
        outer_scroll.setWidget(central)
        self.setCentralWidget(outer_scroll)

        left_col = QVBoxLayout()
        left_col.setSpacing(10)
        right_col = QVBoxLayout()
        right_col.setSpacing(10)

        # --- LEFT COLUMN: setup steps (connect -> load model -> calibrate) ---

        # Connection
        grp_conn = QGroupBox("Connect Device")
        conn_layout = QVBoxLayout(grp_conn)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_wired_tab_widget(), "Wired (USB / Serial)")
        self.tabs.addTab(self.build_wireless_tab_widget(), "Wireless (Wi-Fi)")
        conn_layout.addWidget(self.tabs)
        self.lbl_conn_status = QLabel("Status: Disconnected")
        self.lbl_conn_status.setAlignment(Qt.AlignCenter)
        conn_layout.addWidget(self.lbl_conn_status)
        left_col.addWidget(grp_conn)

        # Model
        grp_model = QGroupBox("Load Pretrained Model")
        model_layout = QHBoxLayout(grp_model)
        model_layout.addWidget(QLabel("RF Model:"))
        self.txt_model_path = QLineEdit()
        self.txt_model_path.setReadOnly(True)
        model_layout.addWidget(self.txt_model_path, 1)
        btn_browse = QPushButton("Browse .joblib")
        btn_browse.clicked.connect(self.browse_model)
        model_layout.addWidget(btn_browse)
        left_col.addWidget(grp_model)

        # Calibration
        grp_cal = QGroupBox("Calibrate")
        cal_layout = QVBoxLayout(grp_cal)
        cal_form = QFormLayout()
        self.spin_rest_sec = QSpinBox()
        self.spin_rest_sec.setRange(CAL_DURATION_MIN_S, CAL_DURATION_MAX_S)
        self.spin_rest_sec.setValue(self.cal_rest_seconds)
        self.spin_rest_sec.setSuffix(" sec")
        cal_form.addRow("Rest duration:", self.spin_rest_sec)
        self.spin_flex_sec = QSpinBox()
        self.spin_flex_sec.setRange(CAL_DURATION_MIN_S, CAL_DURATION_MAX_S)
        self.spin_flex_sec.setValue(self.cal_flex_seconds)
        self.spin_flex_sec.setSuffix(" sec")
        cal_form.addRow("Flex duration:", self.spin_flex_sec)
        cal_layout.addLayout(cal_form)
        self.btn_calibrate = QPushButton("Calibrate")
        self.btn_calibrate.setEnabled(False)
        self.btn_calibrate.clicked.connect(self.start_calibration_sequence)
        cal_layout.addWidget(self.btn_calibrate)
        self.lbl_cal_status = QLabel("Connect and load a model to calibrate.")
        self.lbl_cal_status.setWordWrap(True)
        self.lbl_cal_status.setStyleSheet("color: #A9C2CF;")
        cal_layout.addWidget(self.lbl_cal_status)
        left_col.addWidget(grp_cal)
        left_col.addStretch(1)

        # --- RIGHT COLUMN: mapping, settings, and live control ---

        # Mapping
        grp_map = QGroupBox("Gesture to Mouse-Action Mapping")
        map_layout = QVBoxLayout(grp_map)
        self.scroll_map = QScrollArea()
        self.scroll_map.setWidgetResizable(True)
        self.map_content = QWidget()
        self.map_form = QFormLayout(self.map_content)
        self.scroll_map.setWidget(self.map_content)
        map_layout.addWidget(self.scroll_map)
        self.lbl_map_hint = QLabel("Load a model to view your classes.")
        self.lbl_map_hint.setStyleSheet("color: #A9C2CF;")
        self.map_form.addRow(self.lbl_map_hint)
        right_col.addWidget(grp_map, 1)

        # Settings
        grp_settings = QGroupBox("Control Settings")
        set_layout = QFormLayout(grp_settings)
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(10.0, 99.9)
        self.spin_conf.setValue(GESTURE_MIN_CONFIDENCE_DEFAULT)
        self.spin_conf.setSuffix("%")
        set_layout.addRow("Minimum Confidence:", self.spin_conf)

        self.spin_margin = QDoubleSpinBox()
        self.spin_margin.setRange(0.0, 50.0)
        self.spin_margin.setValue(self.decision_engine.min_margin * 100.0)
        self.spin_margin.setSuffix(" pts")
        self.spin_margin.setToolTip(
            "Required gap between the top prediction and the runner-up (often Rest) "
            "before a gesture is allowed to act. Raise this if Rest keeps triggering moves/clicks."
        )
        self.spin_margin.valueChanged.connect(lambda v: setattr(self.decision_engine, "min_margin", v / 100.0))
        set_layout.addRow("Confidence Margin:", self.spin_margin)

        self.spin_debounce = QSpinBox()
        self.spin_debounce.setRange(1, 10)
        self.spin_debounce.setValue(self.decision_engine.consecutive_required)
        self.spin_debounce.setSuffix(" frames")
        self.spin_debounce.setToolTip("Consecutive matching windows required before a gesture is allowed to act.")
        self.spin_debounce.valueChanged.connect(lambda v: setattr(self.decision_engine, "consecutive_required", v))
        set_layout.addRow("Debounce Count:", self.spin_debounce)

        self.spin_click_refractory = QDoubleSpinBox()
        self.spin_click_refractory.setRange(0.1, 5.0)
        self.spin_click_refractory.setValue(self.decision_engine.refractory_s)
        self.spin_click_refractory.setSuffix(" sec")
        self.spin_click_refractory.setToolTip("Minimum time between clicks, regardless of gesture activity.")
        self.spin_click_refractory.valueChanged.connect(lambda v: setattr(self.decision_engine, "refractory_s", v))
        set_layout.addRow("Click Refractory:", self.spin_click_refractory)

        self.spin_speed = QSpinBox()
        self.spin_speed.setRange(1, 150)
        self.spin_speed.setValue(30)
        self.spin_speed.setSuffix(" px")
        set_layout.addRow("Mouse Speed (per tick):", self.spin_speed)
        self.spin_cooldown = QDoubleSpinBox()
        self.spin_cooldown.setRange(0.1, 5.0)
        self.spin_cooldown.setValue(1.0)
        self.spin_cooldown.setSuffix(" sec")
        set_layout.addRow("Click Cooldown:", self.spin_cooldown)
        right_col.addWidget(grp_settings)

        # Live status + control
        grp_live = QGroupBox("Live Control")
        live_layout = QVBoxLayout(grp_live)

        self.lbl_status = QLabel("Status: Idle")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #A9C2CF;")
        live_layout.addWidget(self.lbl_status)

        self.lbl_prediction = QLabel("REST")
        self.lbl_prediction.setAlignment(Qt.AlignCenter)
        self.lbl_prediction.setFont(QFont("Arial", 26, QFont.Bold))
        self.lbl_prediction.setStyleSheet("color: #6F8A99;")
        live_layout.addWidget(self.lbl_prediction)

        self.lbl_conf = QLabel("Conf: 0.0%")
        self.lbl_conf.setAlignment(Qt.AlignCenter)
        live_layout.addWidget(self.lbl_conf)

        self.lbl_diagnostics = QLabel("Signal: WAITING | Calibration: NOT VALID | Packet loss: n/a")
        self.lbl_diagnostics.setWordWrap(True)
        self.lbl_diagnostics.setAlignment(Qt.AlignCenter)
        self.lbl_diagnostics.setStyleSheet("color: #A9C2CF;")
        live_layout.addWidget(self.lbl_diagnostics)

        self.btn_mouse_toggle = QPushButton("ENABLE MOUSE CONTROL")
        self.btn_mouse_toggle.setStyleSheet("background-color: #2e7d32; font-size: 16px; padding: 12px;")
        self.btn_mouse_toggle.setCheckable(True)
        self.btn_mouse_toggle.setEnabled(False)
        self.btn_mouse_toggle.toggled.connect(self.toggle_mouse_control)
        live_layout.addWidget(self.btn_mouse_toggle)

        lbl_safety = QLabel("Safety Feature: move the physical mouse to a screen corner to abort!")
        lbl_safety.setWordWrap(True)
        lbl_safety.setAlignment(Qt.AlignCenter)
        lbl_safety.setStyleSheet("color: #BF092F; font-size: 11px;")
        live_layout.addWidget(lbl_safety)

        right_col.addWidget(grp_live)

        # Experiments & Analysis launcher - opens the merged Lab Suite
        # (Experiment Manager, ISO 9241-9 task, Performance Analysis,
        # Publication Figures, Session Comparison) as its own tabbed window.
        grp_lab = QGroupBox("Experiments && Analysis")
        lab_layout = QVBoxLayout(grp_lab)
        lbl_lab = QLabel("Run matched controller experiments, analyze logs, and export figures/tables.")
        lbl_lab.setWordWrap(True)
        lbl_lab.setStyleSheet(f"color: {THEME_COLORS['muted']};")
        lab_layout.addWidget(lbl_lab)
        self.btn_open_lab_suite = QPushButton("Open Lab Suite")
        self.btn_open_lab_suite.clicked.connect(self.open_lab_suite)
        lab_layout.addWidget(self.btn_open_lab_suite)
        right_col.addWidget(grp_lab)

        outer.addLayout(left_col, 1)
        outer.addLayout(right_col, 1)

        if not HAS_PYAUTOGUI:
            QMessageBox.warning(self, "Missing Library", "pyautogui not found. Run: pip install pyautogui")
        if not HAS_RF_FEATURES:
            QMessageBox.warning(self, "Missing File", "rf_features.py must be in the same folder as this script.")

    # -------------------------------------------------------------- lab suite
    def open_lab_suite(self):
        """Open the merged Experiment Manager / Analysis / Figures window.

        The suite is created lazily and kept alive on ``self`` so it isn't
        garbage-collected the moment this method returns, and so calling it
        again just re-raises the existing window instead of duplicating it.
        """
        if self._lab_suite_window is None:
            self._lab_suite_window = AnalysisSuiteWindow(
                emg_controller=self,
                performance_logger_instance=self.performance_logger,
            )
        self._lab_suite_window.show()
        self._lab_suite_window.raise_()
        self._lab_suite_window.activateWindow()

    def _build_wired_tab_widget(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("Port:"))
        self.combo_ports = QComboBox()
        port_row.addWidget(self.combo_ports, 1)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh_wired_ports)
        port_row.addWidget(btn_refresh)
        layout.addLayout(port_row)

        chan_row = QHBoxLayout()
        chan_row.addWidget(QLabel("EMG Channels:"))
        self.spin_wired_channels = QSpinBox()
        self.spin_wired_channels.setRange(2, 9)
        self.spin_wired_channels.setValue(DEFAULT_WIRED_CHANNELS)
        chan_row.addWidget(self.spin_wired_channels)
        chan_row.addStretch()
        layout.addLayout(chan_row)

        self.btn_wired_connect = QPushButton("Connect")
        self.btn_wired_connect.clicked.connect(self.toggle_wired_connection)
        layout.addWidget(self.btn_wired_connect)

        self.refresh_wired_ports()
        return w

    def build_wireless_tab_widget(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("Access Key:"))
        self.txt_access_key = QLineEdit(self.wireless_access_key or "")
        self.txt_access_key.setEchoMode(QLineEdit.Password)
        key_row.addWidget(self.txt_access_key, 1)
        layout.addLayout(key_row)

        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Device:"))
        self.combo_devices = QComboBox()
        device_row.addWidget(self.combo_devices, 1)
        btn_discover = QPushButton("Discover")
        btn_discover.clicked.connect(self.discover_wireless_devices)
        device_row.addWidget(btn_discover)
        layout.addLayout(device_row)

        ip_row = QHBoxLayout()
        ip_row.addWidget(QLabel("Device IP:"))
        self.txt_manual_ip = QLineEdit()
        self.txt_manual_ip.setPlaceholderText("optional, e.g. 192.168.0.42 (if auto-discovery finds nothing)")
        ip_row.addWidget(self.txt_manual_ip, 1)
        layout.addLayout(ip_row)

        self.lbl_device_info = QLabel("No wireless device discovered yet.")
        self.lbl_device_info.setWordWrap(True)
        self.lbl_device_info.setStyleSheet("color: #A9C2CF;")
        layout.addWidget(self.lbl_device_info)
        self.combo_devices.currentIndexChanged.connect(self._refresh_device_info)

        btn_provision = QPushButton("Provision New Device (USB)")
        btn_provision.clicked.connect(self.open_provision_dialog)
        layout.addWidget(btn_provision)

        self.btn_wireless_connect = QPushButton("Connect Wireless")
        self.btn_wireless_connect.clicked.connect(self.toggle_wireless_connection)
        layout.addWidget(self.btn_wireless_connect)

        return w

    # --------------------------------------------------------------- wired
    def refresh_wired_ports(self):
        self.combo_ports.clear()
        self.combo_ports.addItem("socket://127.0.0.1:7000 (Simulator)", "socket://127.0.0.1:7000")
        for p in serial.tools.list_ports.comports():
            self.combo_ports.addItem(f"{p.device} - {p.description}", p.device)

    def toggle_wired_connection(self):
        if self.is_connected:
            self.disconnect_stream()
        else:
            self.connect_wired()

    def connect_wired(self):
        port = self.combo_ports.currentData()
        if not port:
            text = self.combo_ports.currentText()
            port = text.split()[0] if text else ""
        if not port:
            QMessageBox.warning(self, "No Port", "Please select a valid serial port.")
            return

        self.num_channels = int(self.spin_wired_channels.value())
        self.emg_channel_count = self.num_channels
        self._reset_stream_state()

        self.serial_worker = SerialWorker(port, DEFAULT_BAUD_RATE, self.num_channels, batch_size=25)
        self.serial_worker.batch_received.connect(self.on_stream_batch)
        self.serial_worker.error_occurred.connect(self.on_stream_error)
        self.serial_worker.start()

        self.connection_medium = "wired"
        self.is_connected = True
        self.btn_wired_connect.setText("Disconnect")
        self.btn_wireless_connect.setEnabled(False)
        self.lbl_conn_status.setText(f"Status: Connected (wired, {self.num_channels} ch)")
        self.lbl_conn_status.setStyleSheet("color: #3B9797;")
        self.check_ready_state()

    # ------------------------------------------------------------ wireless
    def discover_wireless_devices(self):
        try:
            manual_ip = self.txt_manual_ip.text().strip()
            self.btn_wireless_connect.setEnabled(False)
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self.discovered_devices = ControlProtocol.discover([manual_ip] if manual_ip else [])
            finally:
                QApplication.restoreOverrideCursor()
                self.btn_wireless_connect.setEnabled(not self.is_connected or self.connection_medium == "wireless")
        except Exception as exc:
            QMessageBox.warning(self, "Discovery Failed", str(exc))
            return
        self.combo_devices.clear()
        for device in self.discovered_devices:
            self.combo_devices.addItem(device.summary, device)
        if not self.discovered_devices:
            self.lbl_device_info.setText(
                "No BioWave device replied. Check: PC and ESP32 on the same Wi-Fi (not guest/isolated), "
                "device provisioned over USB, UDP 5001 allowed in the firewall, or type the device IP above.")
        self._refresh_device_info()

    def _refresh_device_info(self):
        device = self.combo_devices.currentData()
        if not isinstance(device, DeviceInfo):
            return
        self.lbl_device_info.setText(
            f"{device.device_name} | IP={device.ip} | Mode={device.wifi_mode} | FW={device.firmware} | IMU ready={device.imu_ready}"
        )

    def open_provision_dialog(self):
        dlg = ProvisionDialog(self)
        dlg.exec_()

    def toggle_wireless_connection(self):
        if self.is_connected:
            self.disconnect_stream()
        else:
            self.connect_wireless()

    def connect_wireless(self):
        device = self.combo_devices.currentData()
        if not isinstance(device, DeviceInfo):
            QMessageBox.warning(self, "No Device", "Discover and select a wireless device first.")
            return
        access_key = self.txt_access_key.text().strip()
        problem = access_key_problem(access_key)
        if problem:
            QMessageBox.warning(self, "Access Key", problem)
            return
        self.wireless_access_key = access_key

        self.num_channels = WIRELESS_TOTAL_CHANNELS
        self.emg_channel_count = WIRELESS_EMG_CHANNELS
        self._reset_stream_state()

        try:
            self.serial_worker = WirelessStreamWorker(WIFI_STREAM_PORT)
            self.serial_worker.batch_received.connect(self.on_stream_batch)
            self.serial_worker.error_occurred.connect(self.on_stream_error)
            self.serial_worker.start()

            client_ip = get_local_ip_for_target(device.ip)
            ControlProtocol.start_stream(device.ip, self.wireless_access_key, client_ip, WIFI_STREAM_PORT)

            self.current_device = device
            self.connection_medium = "wireless"
            self.is_connected = True
            self.keepalive_failures = 0
            self.btn_wireless_connect.setText("Disconnect")
            self.btn_wired_connect.setEnabled(False)
            self.lbl_conn_status.setText(f"Status: Connected (wireless, {device.summary})")
            self.lbl_conn_status.setStyleSheet("color: #3B9797;")
            self.check_ready_state()
        except Exception as exc:
            if self.serial_worker:
                self.serial_worker.stop()
                self.serial_worker = None
            QMessageBox.critical(self, "Wireless Connection Error", str(exc))

    # ------------------------------------------------------------ keepalive
    def send_wireless_keepalive(self):
        """Ping the wireless device periodically so its firmware doesn't
        time out the stream. Mirrors the main app's watchdog behaviour."""
        if not self.is_connected or self.connection_medium != "wireless":
            return
        if self.current_device is None or not self.wireless_access_key:
            return
        try:
            ControlProtocol.ping(self.current_device.ip, self.wireless_access_key)
            self.keepalive_failures = 0
        except Exception:
            self.keepalive_failures += 1
            if self.keepalive_failures >= KEEPALIVE_MAX_FAILURES:
                self.lbl_conn_status.setText("Status: Wireless keepalive lost")
                self.lbl_conn_status.setStyleSheet("color: #f57c00;")
                self.disable_mouse_control("Wireless connection lost")

    def on_stream_error(self, message):
        LOG.error("Stream error: %s", message)
        self.disable_mouse_control("Connection error")
        QMessageBox.warning(self, "Stream Error", message)

    # ------------------------------------------------------------- shared
    def disconnect_stream(self):
        self.stop_calibration_if_running()
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

        self.is_connected = False
        self.is_calibrated = False
        self.preprocessor = None
        self.calibration_profile = None
        self.disable_mouse_control("Disconnected")
        self.current_device = None
        self.connection_medium = ""
        self.keepalive_failures = 0

        self.btn_wired_connect.setText("Connect")
        self.btn_wired_connect.setEnabled(True)
        self.btn_wireless_connect.setText("Connect Wireless")
        self.btn_wireless_connect.setEnabled(True)
        self.lbl_conn_status.setText("Status: Disconnected")
        self.lbl_conn_status.setStyleSheet("color: #A9C2CF;")
        self.lbl_cal_status.setText("Connect and load a model to calibrate.")
        self.btn_mouse_toggle.setChecked(False)
        self.check_ready_state()

    def _reset_stream_state(self):
        with self._buffer_lock:
            self.data_buffer = np.zeros((self.num_channels, WINDOW_SIZE), dtype=np.float32)
            self.sample_ring = SampleRingBuffer(self.num_channels, WINDOW_SIZE)
        self.baseline_offsets = np.zeros(self.num_channels, dtype=np.float32)
        self.rf_valid_sample_count = 0
        self.rf_samples_since_submit = 0
        self.is_calibrated = False
        self.calibration_active = False
        self.rest_capture = []
        self.flex_capture = []

    # ------------------------------------------------------------- model
    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select RF Model", "", "Joblib Files (*.joblib)")
        if not path:
            return
        try:
            artifact = joblib.load(path)
            model = artifact["model"]
            class_names = list(artifact.get("class_names", artifact.get("classes", [])))
            if not class_names and hasattr(model, "classes_"):
                class_names = [str(x) for x in list(getattr(model, "classes_", []))]

            self.rf_window_samples = int(max(8, artifact.get("window_samples", 100)))
            self.rf_stride_samples = int(max(1, artifact.get("stride_samples", self.rf_window_samples)))
            self.rf_model_input_channels = int(max(1, artifact.get("input_channels", self.emg_channel_count or DEFAULT_WIRED_CHANNELS)))
            self.rf_model_sample_rate = int(artifact.get("sample_rate", SAMPLE_RATE))
            self.rf_class_names = [str(x) for x in class_names]
            self.rf_model_path = path
            # A trained model's "input_channels" is the total column count it was
            # trained on (EMG + IMU for wireless sessions - e.g. 8 EMG + 3 IMU = 11),
            # not the EMG-only count. Compare against num_channels (everything the
            # current connection actually supplies), matching how the trainer and the
            # real-time window slicing both work - not emg_channel_count, which would
            # make every wirelessly-trained model permanently "incompatible".
            compatibility = validate_model_artifact(artifact, SAMPLE_RATE, self.num_channels)
            self.rf_preprocessing_version = compatibility.preprocessing_version
            self.model_compatible = compatibility.compatible
            self.model_compatibility_message = "\n".join(compatibility.errors + compatibility.warnings) or "Model compatibility verified."
            self.model_expected_feature_count = getattr(model, "n_features_in_", None)
            if not compatibility.compatible:
                raise ValueError("Model is incompatible:\n" + "\n".join(compatibility.errors))

            self.inference_worker.load_model(model, self.rf_class_names, sample_rate=self.rf_model_sample_rate)
            self.txt_model_path.setText(path)
            self.build_mapping_ui(self.rf_class_names)

            self.model_loaded = True
            if compatibility.warnings:
                QMessageBox.warning(self, "Model Compatibility Warning", "\n".join(compatibility.warnings))
            self.check_ready_state()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load model:\n{e}")

    def build_mapping_ui(self, class_names):
        while self.map_form.count():
            item = self.map_form.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.mapping_combos = []
        self.class_action_map = {}

        for cls in class_names:
            combo = QComboBox()
            combo.addItems(MOUSE_ACTIONS)

            clower = cls.lower()
            if "up" in clower:
                combo.setCurrentText("Move Up")
            elif "down" in clower:
                combo.setCurrentText("Move Down")
            elif "left" in clower and "click" not in clower:
                combo.setCurrentText("Move Left")
            elif "right" in clower and "click" not in clower:
                combo.setCurrentText("Move Right")
            elif "double" in clower:
                combo.setCurrentText("Double Click")
            elif "right" in clower and "click" in clower:
                combo.setCurrentText("Right Click")
            elif "click" in clower or "fist" in clower:
                combo.setCurrentText("Left Click")
            else:
                combo.setCurrentText("Ignore")

            combo.currentTextChanged.connect(lambda text, c=cls: self.update_map(c, text))
            self.update_map(cls, combo.currentText())

            self.map_form.addRow(f"Gesture: {cls}", combo)
            self.mapping_combos.append(combo)

    def update_map(self, cls, action):
        self.class_action_map[cls] = action

    def check_ready_state(self):
        # Recheck when a connection changes the real acquisition channel count.
        # num_channels (not emg_channel_count) matches what the model was trained
        # on - see the comment in browse_model().
        if self.model_loaded:
            expected = self.model_expected_feature_count
            if expected is not None and expected != expected_feature_count(self.num_channels):
                self.model_compatible = False
                self.model_compatibility_message = "Feature count does not match the connected device's channel count."
            elif expected is not None:
                self.model_compatible = True
                self.model_compatibility_message = "Model feature count matches the connected device's channels."
        ready = self.model_loaded and self.is_connected and self.model_compatible
        self.btn_calibrate.setEnabled(ready and not self.calibration_active)
        control_ready = ready and self.is_calibrated
        self.btn_mouse_toggle.setEnabled(control_ready and HAS_PYAUTOGUI)
        if not control_ready and self.btn_mouse_toggle.isChecked():
            self.btn_mouse_toggle.setChecked(False)
        if ready and not self.is_calibrated:
            self.lbl_cal_status.setText("Ready to calibrate. Click Calibrate and follow the prompts.")
        elif not ready:
            self.lbl_cal_status.setText(self.model_compatibility_message if self.model_loaded and not self.model_compatible else "Connect and load a model to calibrate.")

    # ---------------------------------------------------------- calibration
    def start_calibration_sequence(self):
        if not self.is_connected or self.calibration_active:
            return
        self.cal_rest_seconds = int(self.spin_rest_sec.value())
        self.cal_flex_seconds = int(self.spin_flex_sec.value())
        self.calibration_phases = [
            {"key": "rest", "name": "REST", "duration_ms": self.cal_rest_seconds * 1000,
             "instruction": "Keep your arm fully relaxed. Do not move."},
            {"key": "flex", "name": "FLEX", "duration_ms": self.cal_flex_seconds * 1000,
             "instruction": "Flex the target muscle steadily until this phase ends."},
        ]
        self.calibration_active = True
        self.is_calibrated = False
        self.rest_capture = []
        self.flex_capture = []
        self.current_cal_phase_idx = -1
        self.rf_valid_sample_count = 0
        self.rf_samples_since_submit = 0

        self.btn_calibrate.setEnabled(False)
        self.lbl_cal_status.setText("Calibrating - follow the on-screen prompts.")

        self.calibration_dialog = CalibrationDialog(self)
        self.calibration_dialog.btn_cancel.clicked.connect(self.cancel_calibration_sequence)
        self.calibration_dialog.show()

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
            self.calibration_dialog.set_phase(phase["name"], phase["instruction"],
                                               self.current_phase_remaining_ms, self.current_phase_total_ms)
        self.calibration_timer.start(CAL_TICK_MS)

    def on_calibration_tick(self):
        if not self.calibration_active:
            self.calibration_timer.stop()
            return
        self.current_phase_remaining_ms -= CAL_TICK_MS
        phase = self.calibration_phases[self.current_cal_phase_idx]
        if self.calibration_dialog:
            self.calibration_dialog.set_phase(phase["name"], phase["instruction"],
                                               self.current_phase_remaining_ms, self.current_phase_total_ms)
        if self.current_phase_remaining_ms <= 0:
            self.calibration_timer.stop()
            self.begin_next_calibration_phase()

    def finish_calibration_sequence(self):
        self.calibration_timer.stop()
        self.calibration_active = False

        if len(self.rest_capture) == 0 or len(self.flex_capture) == 0:
            self.lbl_cal_status.setText("Calibration failed: no REST samples captured.")
            QMessageBox.warning(self, "Calibration Failed", "No REST samples were captured.")
            self.check_ready_state()
            if self.calibration_dialog:
                self.calibration_dialog.close()
                self.calibration_dialog = None
            return

        rest = np.vstack(self.rest_capture).astype(np.float32)  # (samples, channels)
        flex = np.vstack(self.flex_capture).astype(np.float32)
        emg_count = int(min(self.emg_channel_count, rest.shape[1]))
        try:
            self.calibration_profile = compute_calibration(rest[:, :emg_count], flex[:, :emg_count])
        except ValueError as exc:
            self.lbl_cal_status.setText(f"Calibration failed: {exc}")
            self.check_ready_state()
            return
        if not self.calibration_profile.valid:
            report = ", ".join(f"CH{i + 1} {q}" for i, q in enumerate(self.calibration_profile.quality))
            self.lbl_cal_status.setText("Calibration rejected: " + report)
            QMessageBox.warning(self, "Calibration Rejected", "Unsafe channel quality:\n" + "\n".join(self.calibration_profile.reasons))
            self.check_ready_state()
            return
        self.baseline_offsets = np.zeros(self.num_channels, dtype=np.float32)
        if emg_count > 0:
            self.baseline_offsets[:emg_count] = np.median(rest[:, :emg_count], axis=0).astype(np.float32)

        with self._buffer_lock:
            if self.sample_ring is not None:
                self.sample_ring.reset()
        self.rf_valid_sample_count = 0
        self.rf_samples_since_submit = 0
        # Rebuilt fresh so its rolling vote history can't span the calibration
        # boundary, but every user-tuned setting (confidence/margin/debounce/
        # click refractory) carries over instead of silently resetting to defaults.
        self.decision_engine = GestureDecisionEngine(
            min_confidence=self.spin_conf.value() / 100.0,
            min_margin=self.spin_margin.value() / 100.0,
            consecutive_required=self.spin_debounce.value(),
            refractory_s=self.spin_click_refractory.value(),
        )
        # Existing V3 models were trained on baseline-centered ADC values.
        # Activating V4 filtering/normalization without matching model metadata
        # would alter feature order/scale and invalidate those models.
        self.preprocessor = None
        if self.rf_preprocessing_version == PREPROCESSING_VERSION:
            cfg = PreprocessingConfig(sample_rate=SAMPLE_RATE)
            self.preprocessor = RealTimePreprocessor(emg_count, cfg, self.calibration_profile)
        self.is_calibrated = True

        summary = (
            f"Calibration complete.\n"
            f"Baseline (ADC): {np.array2string(self.baseline_offsets, precision=1)}\n"
            f"Channel quality: {', '.join(f'CH{i + 1} {q}' for i, q in enumerate(self.calibration_profile.quality))}\n"
            f"REST/FLEX duration: {self.cal_rest_seconds}s / {self.cal_flex_seconds}s"
        )
        self.lbl_cal_status.setText("Calibrated - live inference running.")
        if self.calibration_dialog:
            self.calibration_dialog.set_finished(summary)
            QTimer.singleShot(900, self._close_calibration_dialog)
        self.check_ready_state()

    def _close_calibration_dialog(self):
        if self.calibration_dialog:
            self.calibration_dialog.close()
            self.calibration_dialog = None

    def stop_calibration_if_running(self):
        if self.calibration_active:
            self.calibration_active = False
            self.calibration_timer.stop()
            self.current_phase_key = ""
        if self.calibration_dialog:
            self.calibration_dialog.close()
            self.calibration_dialog = None
        self.check_ready_state()

    def cancel_calibration_sequence(self):
        self.stop_calibration_if_running()
        self.lbl_cal_status.setText("Calibration canceled.")

    # ----------------------------------------------------------- streaming
    def apply_baseline(self, raw_batch_T):
        # raw_batch_T shape: (channels, samples)
        adjusted = np.ascontiguousarray(raw_batch_T, dtype=np.float32)
        emg_count = int(min(self.emg_channel_count, adjusted.shape[0]))
        if emg_count <= 0:
            return adjusted

        centered = adjusted[:emg_count, :] - self.baseline_offsets[:emg_count, np.newaxis]
        for ch in range(emg_count):
            near_rest = np.abs(centered[ch, :]) < BASE_ADAPT_GUARD
            if np.any(near_rest):
                mean_err = float(np.mean(centered[ch, near_rest]))
                self.baseline_offsets[ch] += BASE_ADAPT_ALPHA * mean_err

        adjusted[:emg_count, :] = adjusted[:emg_count, :] - self.baseline_offsets[:emg_count, np.newaxis]
        return adjusted

    def on_stream_batch(self, batch):
        metadata = batch if isinstance(batch, SampleBatch) else None
        batch = np.asarray(metadata.samples if metadata is not None else batch, dtype=np.float32)
        if batch.ndim != 2 or batch.shape[1] != self.num_channels or self.sample_ring is None:
            return

        self.last_batch_received_monotonic = metadata.host_received_monotonic if metadata else time.monotonic()
        if metadata and metadata.gap_before:
            # No synthetic EMG is inserted.  Resetting the valid count forces a
            # fully observed window after any gap; large gaps additionally block
            # control immediately until a clean window has accumulated.
            self.rf_valid_sample_count = 0
            self.rf_samples_since_submit = 0
            if metadata.gap_before > MAX_INFERENCE_PACKET_GAP:
                self.stream_invalid_until = time.monotonic() + (self.rf_window_samples / SAMPLE_RATE)
            LOG.warning("UDP gap: %d packet(s); inference window invalidated", metadata.gap_before)

        if self.calibration_active:
            if self.current_phase_key == "rest":
                self.rest_capture.append(batch.copy())
            elif self.current_phase_key == "flex":
                self.flex_capture.append(batch.copy())
            return

        if not self.is_calibrated:
            return

        raw_T = batch.T  # (channels, samples)
        if self.preprocessor is not None:
            processed_emg = self.preprocessor.process(batch[:, :self.emg_channel_count]).T
            centered = raw_T.copy()
            centered[:self.emg_channel_count] = processed_emg
        else:
            centered = self.apply_baseline(raw_T)
        num_new = centered.shape[1]

        ring_started = time.perf_counter_ns()
        with self._buffer_lock:
            # One write per sample batch; no O(WINDOW_SIZE) shift/copy.
            self.sample_ring.append(centered.T, discontinuity=bool(metadata and metadata.gap_before))
        self.pipeline_profiler.record_ns("buffer_append", ring_started)
        self.rf_valid_sample_count = self.sample_ring.sample_count
        self.rf_samples_since_submit += num_new

        model_ch = int(max(1, self.rf_model_input_channels))
        if (self.rf_samples_since_submit >= self.rf_stride_samples
                and self.sample_ring.has_window(self.rf_window_samples)
                and self.num_channels >= model_ch):
            self.rf_samples_since_submit = 0
            with self._buffer_lock:
                # A single final chronological copy is required by sklearn/FFT.
                win = self.sample_ring.latest(self.rf_window_samples)[:, :model_ch]
            quality_profile = self.calibration_profile
            quality = assess_signal_quality(win, quality_profile) if quality_profile is not None else SignalQuality("SIGNAL_POOR", [], "uncalibrated")
            self.last_signal_quality = quality
            if quality.state != "GOOD" or time.monotonic() < self.stream_invalid_until:
                self.disable_mouse_control("Signal quality or packet continuity failed")
                self.update_diagnostics(quality)
                return
            self.update_diagnostics(quality)
            self.inference_worker.submit_window(win)

    # --------------------------------------------------------- prediction
    def update_diagnostics(self, quality=None, inference_ms=None):
        quality = quality or self.last_signal_quality
        q = quality.state if quality else "WAITING"
        channel_text = ""
        if quality and quality.channel_states:
            channel_text = " | " + ", ".join(f"CH{i + 1} {s}" for i, s in enumerate(quality.channel_states))
        packet = "n/a"
        if isinstance(self.serial_worker, WirelessStreamWorker):
            packet = f"{self.serial_worker.stats.packet_loss_percent:.2f}% (jitter {self.serial_worker.stats.jitter_ms:.1f} ms)"
        latency = f" | inference {inference_ms:.1f} ms" if inference_ms is not None else ""
        calibration = "VALID" if self.is_calibrated else "NOT VALID"
        self.lbl_diagnostics.setText(f"Signal: {q} | Calibration: {calibration} | Packet loss: {packet}{latency}{channel_text}")

    def on_inference_error(self, message):
        LOG.error("Model inference failed: %s", message)
        self.disable_mouse_control("Model inference error")
        self.lbl_prediction.setText("UNKNOWN")
        self.lbl_status.setText("Status: Model error; mouse disabled")

    def on_prediction_ready(self, label, conf, margin, inference_ms):
        conf_pct = (conf * 100.0) if conf is not None else None

        self.lbl_prediction.setText(label.upper())
        self.lbl_conf.setText(f"Conf: {conf_pct:.1f}%" if conf_pct is not None else "Conf: unavailable")
        if conf_pct is not None:
            self.gesture_bridge.publish(label, conf_pct, os.path.basename(self.rf_model_path) if self.rf_model_path else "")
        quality_good = self.last_signal_quality is not None and self.last_signal_quality.state == "GOOD"
        self.decision_engine.min_confidence = self.spin_conf.value() / 100.0
        decision = self.decision_engine.update(label, conf, margin, quality_good)
        self.update_diagnostics(inference_ms=inference_ms)

        if not decision.action_allowed or self.class_action_map.get(label) == "Ignore":
            self.lbl_prediction.setStyleSheet("color: #6F8A99;")
            return

        self.lbl_prediction.setStyleSheet("color: #3B9797;")

        if self.mouse_control_active:
            action = self.class_action_map.get(label, "Ignore")
            if "Click" in action and not self.decision_engine.trigger_click_once():
                return
            self.execute_mouse_action(action)

    # --------------------------------------------------------------- mouse
    def toggle_mouse_control(self, checked):
        self.mouse_control_active = checked and HAS_PYAUTOGUI and self.is_connected and self.is_calibrated and self.model_compatible
        if checked:
            self.btn_mouse_toggle.setText("STOP MOUSE CONTROL")
            self.btn_mouse_toggle.setStyleSheet("background-color: #BF092F; font-size: 16px; padding: 12px;")
        else:
            self.btn_mouse_toggle.setText("ENABLE MOUSE CONTROL")
            self.btn_mouse_toggle.setStyleSheet("background-color: #2e7d32; font-size: 16px; padding: 12px;")

    def disable_mouse_control(self, reason):
        """Single fail-safe path used for model, link, packet, and quality failures."""
        if self.mouse_control_active:
            LOG.warning("Mouse control disabled: %s", reason)
        self.mouse_control_active = False
        if self.btn_mouse_toggle.isChecked():
            self.btn_mouse_toggle.setChecked(False)
        self.lbl_status.setText(f"Status: {reason}")

    def set_performance_target(self, target_id, center_x, center_y, radius):
        """Open a performance-logging trial when an experiment displays a target."""
        self.performance_logger.set_target(target_id, center_x, center_y, radius)

    def execute_mouse_action(self, action):
        if action == "Ignore":
            return
        speed = self.spin_speed.value()
        now = time.time()
        try:
            if action == "Move Up":
                pyautogui.move(0, -speed)
            elif action == "Move Down":
                pyautogui.move(0, speed)
            elif action == "Move Left":
                pyautogui.move(-speed, 0)
            elif action == "Move Right":
                pyautogui.move(speed, 0)
            if action in ("Move Up", "Move Down", "Move Left", "Move Right"):
                cursor = pyautogui.position()
                self.performance_logger.log_movement(cursor.x, cursor.y, action)
            elif "Click" in action:
                if now - self.last_click_time > self.spin_cooldown.value():
                    if action == "Left Click":
                        pyautogui.click(button="left")
                    elif action == "Right Click":
                        pyautogui.click(button="right")
                    elif action == "Double Click":
                        pyautogui.doubleClick()
                    self.last_click_time = now
                    cursor = pyautogui.position()
                    self.performance_logger.log_click(cursor.x, cursor.y, action)
        except pyautogui.FailSafeException:
            self.btn_mouse_toggle.setChecked(False)
            QMessageBox.critical(self, "Safety Triggered", "Mouse hit the screen corner. Control disabled for safety.")

    # ------------------------------------------------------------- window
    def closeEvent(self, event):
        self.keepalive_timer.stop()
        self.disconnect_stream()
        if self.inference_worker:
            self.inference_worker.stop()
        self.performance_logger.stop()
        self.gesture_bridge.close()
        if self._lab_suite_window is not None:
            self._lab_suite_window.close()
        event.accept()


if __name__ == "__main__":
    configure_high_dpi()
    app = QApplication(sys.argv)
    window = MouseControllerApp()
    window.show()
    sys.exit(app.exec_())
