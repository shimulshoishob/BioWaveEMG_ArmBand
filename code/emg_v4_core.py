"""Safety-critical, GUI-independent building blocks for the BioWave V4 controller.

This module deliberately has no Qt, serial, UDP socket, or mouse dependency so it
can be replayed and unit tested.  The V3 feature extractor remains the default for
old models; V4 preprocessing must be explicitly declared by model metadata.
"""
from __future__ import annotations

import logging
import math
import struct
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

PREPROCESSING_VERSION = "v2"
LEGACY_PREPROCESSING_VERSION = "legacy-baseline-v1"
FEATURE_EXTRACTOR_VERSION = "rf_features.v1"
LOG = logging.getLogger("biowave.v4")


@dataclass
class SampleBatch:
    """Samples plus transport metadata. Values are never silently discarded."""
    samples: np.ndarray
    packet_sequences: Optional[np.ndarray] = None
    frame_ids: Optional[np.ndarray] = None
    emg_timestamps: Optional[np.ndarray] = None
    imu_ids: Optional[np.ndarray] = None
    imu_timestamps: Optional[np.ndarray] = None
    host_received_monotonic: float = field(default_factory=time.monotonic)
    gap_before: int = 0
    invalid: bool = False


@dataclass
class WirelessStats:
    packets_received: int = 0
    packets_missing: int = 0
    packets_out_of_order: int = 0
    invalid_packets: int = 0
    previous_sequence: Optional[int] = None
    last_arrival: Optional[float] = None
    arrival_intervals: deque = field(default_factory=lambda: deque(maxlen=200))

    def observe(self, sequence: int, arrival: Optional[float] = None) -> int:
        """Record one UDP packet and return the number missing before it.

        UDP sequence numbers are unsigned 32-bit; wrap-around is handled.  A
        duplicate/out-of-order packet is retained in diagnostics but never used
        to make a gap look like valid continuous data.
        """
        arrival = time.monotonic() if arrival is None else arrival
        self.packets_received += 1
        gap = 0
        if self.previous_sequence is not None:
            delta = (int(sequence) - self.previous_sequence) & 0xFFFFFFFF
            if delta == 0 or delta > 0x7FFFFFFF:
                self.packets_out_of_order += 1
            elif delta > 1:
                gap = delta - 1
                self.packets_missing += gap
        if self.last_arrival is not None:
            self.arrival_intervals.append(arrival - self.last_arrival)
        self.previous_sequence = int(sequence)
        self.last_arrival = arrival
        return gap

    @property
    def packet_loss_percent(self) -> float:
        total = self.packets_received + self.packets_missing
        return 100.0 * self.packets_missing / total if total else 0.0

    @property
    def jitter_ms(self) -> float:
        return float(np.std(self.arrival_intervals) * 1000.0) if len(self.arrival_intervals) > 1 else 0.0


@dataclass
class CalibrationProfile:
    rest_baseline: np.ndarray
    rest_std: np.ndarray
    rest_rms: np.ndarray
    flex_rms: np.ndarray
    flex_peak: np.ndarray
    activation_scale: np.ndarray
    quality: list[str]
    valid: bool
    reasons: list[str]
    normalization: str = "activation_rms"


def compute_calibration(rest: np.ndarray, flex: np.ndarray, *, min_samples: int = 250,
                        dead_std: float = 1e-3, rest_noise_limit: float = 80.0,
                        min_activation_ratio: float = 1.25,
                        saturation_limit: Optional[float] = 65534.0) -> CalibrationProfile:
    """Quantify calibration, rejecting unsafe captures rather than guessing."""
    rest = np.asarray(rest, dtype=np.float32)
    flex = np.asarray(flex, dtype=np.float32)
    if rest.ndim != 2 or flex.ndim != 2 or rest.shape[1] != flex.shape[1]:
        raise ValueError("REST and FLEX must be 2-D with identical channel count")
    channels = rest.shape[1]
    if rest.shape[0] < min_samples or flex.shape[0] < min_samples:
        zeros = np.zeros(channels, dtype=np.float32)
        return CalibrationProfile(zeros, zeros, zeros, zeros, zeros, np.ones(channels),
                                  ["INSUFFICIENT"] * channels, False,
                                  ["Insufficient REST or FLEX samples"])
    baseline = np.median(rest, axis=0)
    rest_centered = rest - baseline
    flex_centered = flex - baseline
    rest_std = np.std(rest_centered, axis=0)
    rest_rms = np.sqrt(np.mean(rest_centered ** 2, axis=0))
    flex_rms = np.sqrt(np.mean(flex_centered ** 2, axis=0))
    flex_peak = np.max(np.abs(flex_centered), axis=0)
    # RMS activation is robust against one transient; epsilon prevents divide-by-zero.
    scale = np.maximum(flex_rms, 1e-6)
    quality, reasons = [], []
    finite = np.isfinite(rest).all(axis=0) & np.isfinite(flex).all(axis=0)
    for ch in range(channels):
        if not finite[ch]:
            quality.append("INVALID"); reasons.append(f"CH{ch + 1}: NaN/Inf")
        elif saturation_limit is not None and ((np.abs(rest[:, ch]) >= saturation_limit).any() or (np.abs(flex[:, ch]) >= saturation_limit).any()):
            quality.append("SATURATED"); reasons.append(f"CH{ch + 1}: saturated")
        elif rest_std[ch] <= dead_std and flex_rms[ch] <= dead_std:
            quality.append("DEAD"); reasons.append(f"CH{ch + 1}: no measurable signal")
        elif rest_std[ch] > rest_noise_limit:
            quality.append("NOISY"); reasons.append(f"CH{ch + 1}: unstable REST")
        elif flex_rms[ch] < max(rest_rms[ch] * min_activation_ratio, dead_std * 2):
            quality.append("WEAK"); reasons.append(f"CH{ch + 1}: insufficient FLEX activation")
        else:
            quality.append("GOOD")
    valid = all(q == "GOOD" for q in quality)
    return CalibrationProfile(baseline.astype(np.float32), rest_std.astype(np.float32), rest_rms.astype(np.float32),
                              flex_rms.astype(np.float32), flex_peak.astype(np.float32), scale.astype(np.float32),
                              quality, valid, reasons)


@dataclass
class PreprocessingConfig:
    version: str = PREPROCESSING_VERSION
    sample_rate: float = 500.0
    highpass_hz: float = 20.0
    lowpass_hz: float = 220.0
    notch_hz: float = 50.0
    notch_q: float = 30.0
    normalization: str = "activation_rms"  # "none" retains centered ADC scale.


class RealTimePreprocessor:
    """Causal, stateful band-pass + notch processor.

    Cascaded one-pole high/low pass filters and a normalized RBJ notch are
    stable for the configured 500 Hz stream.  States persist across batches,
    avoiding the non-causal look-ahead and edge artifacts of ``filtfilt``.
    """
    def __init__(self, channels: int, config: PreprocessingConfig, profile: CalibrationProfile):
        self.channels, self.config, self.profile = int(channels), config, profile
        if not 0 < config.highpass_hz < config.lowpass_hz < config.sample_rate / 2:
            raise ValueError("invalid causal band-pass configuration")
        self._hp_x = np.zeros(channels, np.float64); self._hp_y = np.zeros(channels, np.float64)
        self._lp_y = np.zeros(channels, np.float64)
        self._z1 = np.zeros(channels, np.float64); self._z2 = np.zeros(channels, np.float64)
        dt = 1.0 / config.sample_rate
        self._hp_a = 1.0 / (1.0 + 1.0 / (2.0 * math.pi * config.highpass_hz * dt))
        self._lp_a = dt / ((1.0 / (2.0 * math.pi * config.lowpass_hz)) + dt)
        w0 = 2.0 * math.pi * config.notch_hz / config.sample_rate
        alpha = math.sin(w0) / (2.0 * config.notch_q)
        b0, b1, b2, a0, a1, a2 = 1, -2 * math.cos(w0), 1, 1 + alpha, -2 * math.cos(w0), 1 - alpha
        self._b0, self._b1, self._b2 = b0 / a0, b1 / a0, b2 / a0
        self._a1, self._a2 = a1 / a0, a2 / a0

    def process(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.channels or not np.isfinite(x).all():
            raise ValueError("invalid sample batch for preprocessing")
        out = np.empty_like(x)
        baseline = self.profile.rest_baseline
        scale = self.profile.activation_scale
        for i, row in enumerate(x):
            centered = row.astype(np.float64) - baseline
            hp = self._hp_a * (self._hp_y + centered - self._hp_x)
            self._hp_x, self._hp_y = centered, hp
            lp = self._lp_y + self._lp_a * (hp - self._lp_y)
            self._lp_y = lp
            y = self._b0 * lp + self._z1
            self._z1 = self._b1 * lp - self._a1 * y + self._z2
            self._z2 = self._b2 * lp - self._a2 * y
            out[i] = y / scale if self.config.normalization != "none" else y
        return out


@dataclass
class SignalQuality:
    state: str
    channel_states: list[str]
    reason: str = ""


def assess_signal_quality(window: np.ndarray, profile: CalibrationProfile, *, saturation_limit: float = 65534.0,
                          max_noise_multiplier: float = 8.0, min_active_std: float = 1e-4) -> SignalQuality:
    x = np.asarray(window, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(profile.quality) or not np.isfinite(x).all():
        return SignalQuality("SIGNAL_POOR", ["INVALID"] * len(profile.quality), "missing or non-finite samples")
    states = []
    for ch in range(x.shape[1]):
        std, peak = float(np.std(x[:, ch])), float(np.max(np.abs(x[:, ch])))
        if peak >= saturation_limit: states.append("SATURATED")
        elif std <= min_active_std: states.append("DEAD")
        elif profile.rest_std[ch] > 0 and std > max_noise_multiplier * max(profile.flex_rms[ch], profile.rest_std[ch]): states.append("NOISY")
        else: states.append(profile.quality[ch] if profile.quality[ch] != "GOOD" else "GOOD")
    bad = [s for s in states if s != "GOOD"]
    return SignalQuality("GOOD" if not bad else "SIGNAL_POOR", states, ", ".join(bad))


@dataclass
class ModelCompatibility:
    compatible: bool
    warnings: list[str]
    errors: list[str]
    preprocessing_version: str


def expected_feature_count(channels: int) -> int:
    return 15 * channels + channels + channels * (channels - 1) // 2


def validate_model_artifact(artifact: Any, acquisition_rate: Optional[float] = None,
                            acquisition_channels: Optional[int] = None) -> ModelCompatibility:
    warnings, errors = [], []
    if not isinstance(artifact, dict) or "model" not in artifact:
        return ModelCompatibility(False, warnings, ["Joblib artifact must contain a model"], "unknown")
    model = artifact["model"]
    if not hasattr(model, "predict"):
        errors.append("Model has no predict()")
    classes = artifact.get("class_names", artifact.get("classes", getattr(model, "classes_", [])))
    if not list(classes): errors.append("Model classes are missing")
    version = artifact.get("preprocessing_version", LEGACY_PREPROCESSING_VERSION)
    if "preprocessing_version" not in artifact: warnings.append("Model preprocessing metadata is missing; using legacy baseline compatibility mode.")
    if "input_channels" not in artifact: warnings.append("Model input channel metadata is missing; feature count will be used when available.")
    rate = artifact.get("sample_rate")
    if rate is None: warnings.append("Model sample-rate metadata is missing; compatibility cannot be fully verified.")
    elif acquisition_rate is not None and not math.isclose(float(rate), float(acquisition_rate), rel_tol=0, abs_tol=0.01):
        errors.append(f"Model sample rate ({rate} Hz) differs from acquisition ({acquisition_rate} Hz); resampling is not enabled.")
    channels = artifact.get("input_channels")
    if channels is not None and acquisition_channels is not None and int(channels) != int(acquisition_channels):
        # "input_channels" is the total column count the model was trained on
        # (EMG plus any IMU columns logged alongside them), not an EMG-only count -
        # callers must pass the acquisition's matching total, not its EMG-only count.
        errors.append(f"Model expects {channels} input channel(s) (as trained), device supplies {acquisition_channels}.")
    count = getattr(model, "n_features_in_", None)
    if count is not None and channels is not None and int(count) != expected_feature_count(int(channels)):
        errors.append(f"Model has {count} features; expected {expected_feature_count(int(channels))} for declared channel order.")
    return ModelCompatibility(not errors, warnings, errors, version)


@dataclass
class Decision:
    state: str
    action_allowed: bool
    click_triggered: bool = False


class GestureDecisionEngine:
    """Temporal gate: actions require sustained, confident, quality-approved labels."""
    def __init__(self, *, vote_windows: int = 5, consecutive_required: int = 3,
                 min_confidence: float = .65, min_margin: float = .10, refractory_s: float = .8):
        self.history = deque(maxlen=vote_windows); self.consecutive_required = consecutive_required
        self.min_confidence, self.min_margin, self.refractory_s = min_confidence, min_margin, refractory_s
        self.active: Optional[str] = None; self.candidate: Optional[str] = None; self.candidate_count = 0
        self.last_click = -float("inf")

    def update(self, label: Optional[str], confidence: Optional[float], margin: Optional[float], quality_good: bool,
               now: Optional[float] = None) -> Decision:
        now = time.monotonic() if now is None else now
        valid = quality_good and label not in (None, "", "REST", "UNKNOWN") and confidence is not None and confidence >= self.min_confidence and (margin is None or margin >= self.min_margin)
        if not valid:
            self.history.clear(); self.candidate = None; self.candidate_count = 0; self.active = None
            return Decision("REST" if label == "REST" and quality_good else "UNKNOWN", False)
        self.history.append(label)
        voted = Counter(self.history).most_common(1)[0][0]
        if voted != self.candidate:
            self.candidate, self.candidate_count = voted, 1
        else: self.candidate_count += 1
        if self.candidate_count < self.consecutive_required:
            return Decision("GESTURE_CANDIDATE", False)
        if self.active != voted:
            self.active = voted
            return Decision(f"{voted}_ACTIVE", True)
        return Decision(f"{voted}_HELD", True)

    def trigger_click_once(self, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        if now - self.last_click < self.refractory_s: return False
        self.last_click = now
        return True


class SessionRecorder:
    """Records raw batches and metadata for deterministic offline replay.

    The recorder intentionally stores raw values, not filtered values, so new
    preprocessing algorithms can be compared against the same acquisition.
    ``.npz`` is used to avoid adding a database or file-format dependency.
    """
    def __init__(self) -> None:
        self._batches: list[SampleBatch] = []

    def append(self, batch: SampleBatch) -> None:
        self._batches.append(batch)

    def save(self, path: str) -> None:
        if not self._batches:
            raise ValueError("no samples recorded")
        samples = np.vstack([b.samples for b in self._batches])
        def concat(name: str, dtype: Any) -> np.ndarray:
            values = [getattr(b, name) for b in self._batches]
            return np.concatenate([v if v is not None else np.full(len(b.samples), -1, dtype=dtype)
                                   for b, v in zip(self._batches, values)])
        np.savez_compressed(path, samples=samples,
                            packet_sequences=concat("packet_sequences", np.int64),
                            frame_ids=concat("frame_ids", np.int64), emg_timestamps=concat("emg_timestamps", np.int64),
                            imu_ids=concat("imu_ids", np.int64), imu_timestamps=concat("imu_timestamps", np.int64))

    @staticmethod
    def replay(path: str, batch_size: int = 25):
        """Yield SampleBatch objects in their original sample order."""
        with np.load(path) as data:
            samples = data["samples"]
            for start in range(0, len(samples), batch_size):
                end = start + batch_size
                yield SampleBatch(samples[start:end], data["packet_sequences"][start:end],
                                  data["frame_ids"][start:end], data["emg_timestamps"][start:end],
                                  data["imu_ids"][start:end], data["imu_timestamps"][start:end])


# ---------------------------------------------------------------------------------------------
# BWIM wire format (v1 and v2) and the USB framing (COBS + CRC-16)
# ---------------------------------------------------------------------------------------------
BWIM_MAGIC = b"BWIM"
BWIM_V1_HEADER = struct.Struct("<4sBBHI")                 # magic, version, frames, frame size, sequence
BWIM_V2_HEADER = struct.Struct("<4sBBHIIIIHH")            # ... + queue drops, overruns, send failures, reconnects, flags
EMG_CHANNELS = 8
IMU_CHANNELS = 3

FRAME_DTYPE_V1 = np.dtype([
    ("frame_id", "<u4"), ("frame_ts", "<u4"), ("imu_id", "<u4"), ("imu_ts", "<u4"),
    ("emg", "<u2", (EMG_CHANNELS,)), ("imu", "<f4", (IMU_CHANNELS,)), ("fresh", "u1"), ("pad", "V3"),
])
FRAME_DTYPE_V2 = np.dtype([
    ("frame_id", "<u4"), ("frame_ts", "<u8"), ("imu_id", "<u4"), ("imu_ts", "<u8"),
    ("emg", "<u2", (EMG_CHANNELS,)), ("imu", "<f4", (IMU_CHANNELS,)), ("fresh", "u1"), ("flags", "u1"),
    ("reserved", "<u2"),
])
assert FRAME_DTYPE_V1.itemsize == 48 and FRAME_DTYPE_V2.itemsize == 56
FRAME_FLAG_LATE = 0x01
DEVICE_COUNTER_KEYS = ("queue_drops", "acq_overruns", "send_failures", "wifi_reconnects")


@dataclass
class BwimPacket:
    """One decoded datagram/USB frame. ``frames`` always uses FRAME_DTYPE_V2 (v1 is widened)."""
    version: int
    sequence: int
    frames: np.ndarray
    device_counters: dict = field(default_factory=dict)   # empty for v1


def parse_bwim_packet(data: bytes) -> Optional[BwimPacket]:
    """Decode a BWIM v1 or v2 packet; ``None`` when it is not a well-formed packet."""
    size = len(data)
    if size < BWIM_V1_HEADER.size or data[:4] != BWIM_MAGIC:
        return None
    version = data[4]
    if version == 1:
        header_size = BWIM_V1_HEADER.size
        _, _, count, frame_size, sequence = BWIM_V1_HEADER.unpack_from(data, 0)
        src_dtype, counters = FRAME_DTYPE_V1, {}
    elif version == 2:
        header_size = BWIM_V2_HEADER.size
        if size < header_size:
            return None
        _, _, count, frame_size, sequence, drops, overruns, failures, reconnects, _flags = BWIM_V2_HEADER.unpack_from(data, 0)
        src_dtype = FRAME_DTYPE_V2
        counters = {"queue_drops": drops, "acq_overruns": overruns,
                    "send_failures": failures, "wifi_reconnects": reconnects}
    else:
        return None
    if count < 1 or frame_size != src_dtype.itemsize or count * frame_size != size - header_size:
        return None
    raw = np.frombuffer(data, dtype=src_dtype, count=count, offset=header_size)
    if version == 2:
        frames = raw
    else:
        frames = np.zeros(count, dtype=FRAME_DTYPE_V2)
        for name in ("frame_id", "frame_ts", "imu_id", "imu_ts", "emg", "imu", "fresh"):
            frames[name] = raw[name]
    return BwimPacket(version, int(sequence), frames, counters)


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF), matching the firmware."""
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def cobs_encode(data: bytes) -> bytes:
    out = bytearray([0])
    code_index, code = 0, 1
    for byte in data:
        if byte == 0:
            out[code_index] = code
            code_index, code = len(out), 1
            out.append(0)
        else:
            out.append(byte)
            code += 1
            if code == 0xFF:
                out[code_index] = code
                code_index, code = len(out), 1
                out.append(0)
    out[code_index] = code
    return bytes(out)


def cobs_decode(data: bytes) -> Optional[bytes]:
    """Inverse of cobs_encode; ``None`` for a malformed block."""
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        code = data[i]
        if code == 0:
            return None
        end = i + code
        if end > n:
            return None
        out += data[i + 1:end]
        i = end
        if code < 0xFF and i < n:
            out.append(0)
    return bytes(out)


def encode_usb_frame(packet_bytes: bytes) -> bytes:
    """What the firmware writes for one packet: COBS(packet + CRC16 LE) + 0x00."""
    return cobs_encode(packet_bytes + struct.pack("<H", crc16_ccitt(packet_bytes))) + b"\x00"


class UsbFrameDecoder:
    """Incremental splitter/decoder for the binary USB stream.

    ``feed`` returns ``(packets, bad)``. ``bad`` counts frames that failed COBS/CRC/format checks;
    plain text lines (boot messages, replies to INFO) are recognised and ignored, not counted.
    """
    MAX_FRAME_BYTES = 2048

    def __init__(self) -> None:
        self._tail = bytearray()
        self._overflow = False

    def feed(self, chunk: bytes):
        packets, bad = [], 0
        parts = bytes(chunk).split(b"\x00")
        parts[0] = bytes(self._tail) + parts[0]
        self._tail = bytearray(parts.pop())
        if len(self._tail) > self.MAX_FRAME_BYTES:
            self._tail.clear()
            bad += 1
        for frame in parts:
            if not frame:
                continue
            decoded = cobs_decode(frame) if len(frame) <= self.MAX_FRAME_BYTES else None
            if decoded is not None and len(decoded) > 2:
                body, crc = decoded[:-2], struct.unpack("<H", decoded[-2:])[0]
                if crc == crc16_ccitt(body):
                    parsed = parse_bwim_packet(body)
                    if parsed is not None:
                        packets.append(parsed)
                        continue
            if self._looks_like_text(frame):
                continue
            bad += 1
        return packets, bad

    @staticmethod
    def _looks_like_text(frame: bytes) -> bool:
        stripped = frame.strip()
        return bool(stripped) and b"|" in stripped and all(32 <= b < 127 or b in (9, 10, 13) for b in stripped)


class TimestampUnwrapper:
    """Turns wrapping uint32 microsecond counters (BWIM v1) into a monotonic int64."""
    def __init__(self, bits: int = 32) -> None:
        self._span = 1 << bits
        self._last_raw: Optional[int] = None
        self._last_out = 0

    def unwrap(self, values) -> np.ndarray:
        raw = np.asarray(values, dtype=np.int64).reshape(-1)
        if raw.size == 0:
            return raw
        if self._last_raw is None:
            self._last_raw, self._last_out = int(raw[0]), int(raw[0])
        deltas = np.diff(np.concatenate(([self._last_raw], raw)))
        deltas = (deltas + self._span // 2) % self._span - self._span // 2   # shortest signed step
        out = self._last_out + np.cumsum(deltas)
        self._last_raw, self._last_out = int(raw[-1]), int(out[-1])
        return out
