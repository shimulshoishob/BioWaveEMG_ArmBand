"""Continuous recording with after-the-fact labelling (no Qt, no GUI).

During a session every received frame, including NaN gap frames, is appended to a raw binary file
by a writer thread, so nothing is ever discarded and memory stays bounded. The protocol only logs
*events* (phase changes) with a host monotonic time. After the session (and at every break as a
checkpoint) ``finalize_session`` resolves each event to a **device frame ID** and streams the raw
file through that event table, so a label switches at an exact sample.

Outputs, all written next to each other:
  signals_<session>.csv   one row per device frame, raw ADC counts, labelled by frame ID
  events_<session>.csv    the event table with the resolved frame IDs
  quality_<session>.json  packets, gaps, device counters, clipping, expected vs actual frames
"""
from __future__ import annotations

import csv
import json
import os
import queue
import threading
import time
from typing import Optional

import numpy as np

SAMPLE_RATE_HZ = 500
EMG_CHANNELS = 8
MIN_VALID_SAMPLE_RATIO = 0.98            # below this a session is flagged as poor quality
CLIP_LOW, CLIP_HIGH = 5, 4090            # ADC counts at or beyond these are counted as clipped

# Old (pre-publication) label name -> (dataset name, numeric code). Prepare/Settle/Rest are phases that
# also carry a label code so a classifier can learn them as classes if wanted.
LABEL_MAP = {
    "Rest": ("rest", 0),
    "Left": ("left", 1),
    "Right": ("right", 2),
    "fist_close": ("fist_close", 3),
    "Up": ("up", 4),
    "Down": ("down", 5),
    "Double_Closed_Fist": ("double_fist_close", 6),
    "Victory_Sign": ("victory_sign", 7),
    "Thumbs_Up": ("thumbs_up", 8),
    "Pinch_using_thumb_and_index_finger": ("pinch_thumb_index", 9),
    "Settle": ("settle", 98),
    "Prepare": ("prepare", 99),
}
# Labels the recorder itself assigns (never produced by the protocol's gesture list).
PSEUDO_LABELS = {"paused": 97, "aborted": 96}
# Phases whose label is fixed, whatever gesture the protocol step names.
PHASE_LABELS = {"prepare": "Prepare", "settle": "Settle", "rest": "Rest", "break": "Rest", "switch_hand": "Rest",
                "paused": "paused"}
LABEL_CODES = {name: code for name, code in LABEL_MAP.values()} | PSEUDO_LABELS

SIGNAL_COLUMNS = (["Device_Frame_ID", "Device_Timestamp_us", "Packet_Sequence", "Valid", "Late_Flag"]
                  + [f"Ch{i}" for i in range(1, EMG_CHANNELS + 1)]
                  + ["IMU_roll", "IMU_pitch", "IMU_yaw", "IMU_Sample_ID", "IMU_Fresh", "Host_Received_s",
                     "Label", "Label_Code", "Phase", "Hand", "Trial_ID"])
EVENT_COLUMNS = ["event_index", "phase", "label", "label_code", "hand", "trial_id", "block", "cycle_no",
                 "host_monotonic_s", "device_frame_id_at_event", "device_frame_id_uncorrected",
                 "latency_estimate_s", "planned_ms", "aborted"]

RAW_DTYPE = np.dtype([
    ("frame_id", "<i8"), ("ts_us", "<i8"), ("seq", "<i8"), ("valid", "u1"), ("late", "u1"), ("fresh", "u1"),
    ("emg", "<u2", (EMG_CHANNELS,)), ("imu", "<f4", (3,)), ("imu_id", "<i8"), ("host_s", "<f8"),
])

def _line_format(emg_channels: int) -> str:
    """Row format; EMG columns beyond the channels in use are written empty (like older recordings)."""
    n = max(1, min(EMG_CHANNELS, int(emg_channels)))
    emg = ",".join(["%g"] * n + [""] * (EMG_CHANNELS - n))
    return "%d,%d,%d,%d,%d," + emg + ",%.3f,%.3f,%.3f,%d,%d,%.6f,%s,%d,%s,%s,%d\n"


def canonical_label(name: str) -> tuple:
    """(dataset label, code) for a protocol label; unknown names get code -1 rather than a guess."""
    key = str(name or "").strip()
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    low = key.lower()
    if low in LABEL_CODES:
        return low, LABEL_CODES[low]
    return key, -1


def phase_slug(phase: str) -> str:
    return str(phase or "").strip().lower().replace(" ", "_")


# ---------------------------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------------------------
class EventLog:
    """Ordered phase-change events. Frame IDs are resolved later, from the arrival index."""

    def __init__(self) -> None:
        self.events: list = []

    def add(self, phase, label, hand, trial_id, block, cycle_no, host_monotonic_s, planned_ms=0) -> int:
        slug = phase_slug(phase)
        # Non-task phases carry their own label, not the gesture that is about to be (or was just) performed.
        label = PHASE_LABELS.get(slug, label)
        name, code = canonical_label(label)
        self.events.append({
            "event_index": len(self.events), "phase": slug, "label": name, "label_code": code,
            "hand": str(hand or ""), "trial_id": int(trial_id), "block": int(block), "cycle_no": int(cycle_no),
            "host_monotonic_s": float(host_monotonic_s), "planned_ms": int(planned_ms), "aborted": 0,
        })
        return len(self.events) - 1

    def mark_aborted_since(self, event_index: int) -> None:
        """Flag an interrupted cycle's events (from its Prepare event on) as aborted."""
        for ev in self.events[max(0, int(event_index)):]:
            if ev["phase"] not in ("paused",):
                ev["aborted"] = 1

    def __len__(self) -> int:
        return len(self.events)


def estimate_latency_s(arrival_host_s: np.ndarray, arrival_ts_us: np.ndarray) -> float:
    """Typical transport delay: how much later than its best case a packet arrives.

    The device and host clocks are not synchronised, so one-way delay is not measurable. The offset
    ``host_arrival - device_time`` is constant plus the delay; its median minus its minimum is the
    queueing/batching delay the typical packet saw on top of the best case.
    """
    host = np.asarray(arrival_host_s, dtype=np.float64)
    dev = np.asarray(arrival_ts_us, dtype=np.float64) * 1e-6
    ok = np.isfinite(host) & (dev >= 0)
    if ok.sum() < 3:
        return 0.0
    offset = host[ok] - dev[ok]
    return float(max(0.0, np.median(offset) - np.min(offset)))


def resolve_events(events: list, arrival_host_s, arrival_frame_id, latency_s: float, fs: float) -> list:
    """Attach device_frame_id_at_event (corrected) and device_frame_id_uncorrected to every event.

    uncorrected = the newest frame already received at the event's host time;
    corrected   = uncorrected + latency_s * fs  (frames still in flight when the cue was shown).
    Returned frame IDs are made non-decreasing so the labelling is a clean step function.
    """
    host = np.asarray(arrival_host_s, dtype=np.float64)
    frames = np.asarray(arrival_frame_id, dtype=np.int64)
    out = []
    if host.size == 0:
        return [dict(ev, device_frame_id_uncorrected=-1, device_frame_id_at_event=-1,
                     latency_estimate_s=float(latency_s)) for ev in events]
    shift = int(round(latency_s * fs))
    last = int(frames[0])
    for ev in events:
        i = int(np.searchsorted(host, ev["host_monotonic_s"], side="right")) - 1
        raw = int(frames[0] if i < 0 else frames[i])
        corrected = max(last, raw + (shift if i >= 0 else 0))
        last = corrected
        out.append(dict(ev, device_frame_id_uncorrected=raw, device_frame_id_at_event=corrected,
                        latency_estimate_s=float(latency_s)))
    return out


def assign_events(frame_ids: np.ndarray, event_frame_ids: np.ndarray) -> np.ndarray:
    """Index of the event each frame belongs to: the last event whose frame ID is <= the frame's.

    Frames before the first event use event 0. Pure and vectorised; the heart of exact label edges.
    """
    idx = np.searchsorted(np.asarray(event_frame_ids, dtype=np.int64),
                          np.asarray(frame_ids, dtype=np.int64), side="right") - 1
    return np.maximum(idx, 0)


# ---------------------------------------------------------------------------------------------
# Continuous recorder
# ---------------------------------------------------------------------------------------------
def payload_to_raw(payload: dict, synth_state: Optional[dict] = None, fs: float = SAMPLE_RATE_HZ) -> np.ndarray:
    """Convert a receiver payload into RAW_DTYPE rows (one per frame, gap rows included)."""
    batch = np.asarray(payload["batch"])
    n = batch.shape[0]
    raw = np.zeros(n, dtype=RAW_DTYPE)
    valid = payload.get("valid")
    raw["valid"] = np.ones(n, dtype=np.uint8) if valid is None else np.asarray(valid, dtype=np.uint8)
    n_emg = min(EMG_CHANNELS, batch.shape[1])      # simulators / wired CSV may have fewer than 8 channels
    emg = batch[:, :n_emg]
    raw["emg"][:, :n_emg] = np.where(np.isfinite(emg), emg, 0).clip(0, 65535).astype(np.uint16)
    if batch.shape[1] >= EMG_CHANNELS + 3:
        raw["imu"] = batch[:, EMG_CHANNELS:EMG_CHANNELS + 3]
    else:
        raw["imu"] = np.nan
    ids = payload.get("frame_ids")
    if ids is None:                                   # CSV simulator: no device framing, synthesise a counter
        state = synth_state if synth_state is not None else {"next": 0}
        raw["frame_id"] = state["next"] + np.arange(n)
        raw["ts_us"] = (raw["frame_id"] * (1e6 / fs)).astype(np.int64)
        state["next"] += n
        raw["seq"] = -1
        raw["imu_id"] = -1
        raw["host_s"] = float(payload.get("host_received_monotonic", time.monotonic()))
        return raw
    raw["frame_id"] = np.asarray(ids, dtype=np.int64)
    raw["ts_us"] = np.asarray(payload.get("emg_timestamps", np.full(n, -1)), dtype=np.int64)
    raw["seq"] = np.asarray(payload.get("packet_numbers", np.full(n, -1)), dtype=np.int64)
    raw["imu_id"] = np.asarray(payload.get("imu_ids", np.full(n, -1)), dtype=np.int64)
    for key, field in (("late_flags", "late"), ("imu_fresh", "fresh")):
        if payload.get(key) is not None:
            raw[field] = np.asarray(payload[key], dtype=np.uint8)
    host = payload.get("host_received_s")
    raw["host_s"] = (np.asarray(host, dtype=np.float64) if host is not None
                     else float(payload.get("host_received_monotonic", time.monotonic())))
    return raw


class ContinuousRecorder:
    """Appends every frame to ``raw_path`` from a writer thread; ``append`` never touches the disk."""

    def __init__(self, raw_path: str, fs: float = SAMPLE_RATE_HZ, emg_channels: int = EMG_CHANNELS) -> None:
        self.raw_path, self.fs = raw_path, float(fs)
        self.emg_channels = max(1, min(EMG_CHANNELS, int(emg_channels)))
        os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
        self._fh = open(raw_path, "wb", buffering=1 << 20)
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._synth = {"next": 0}
        self.rows = 0
        self.valid_rows = 0
        self.late_rows = 0
        self.clip_counts = np.zeros(EMG_CHANNELS, dtype=np.int64)
        self.device_counters: dict = {}
        self._arrival: list = []                     # (host_s, frame_id, ts_us) per packet
        self.error: Optional[str] = None
        self._thread = threading.Thread(target=self._run, name="biowave-recorder", daemon=True)
        self._thread.start()

    # -- producer side (any thread, non-blocking) -----------------------------------------
    def append(self, payload: dict) -> None:
        self._queue.put(payload)

    def checkpoint(self, timeout: float = 30.0) -> int:
        """Block until everything queued so far is on disk (fsync'd); returns the row count. Not for the GUI thread."""
        done = threading.Event()
        self._queue.put(done)
        done.wait(timeout)
        return self.rows

    def close(self, timeout: float = 60.0) -> int:
        self._queue.put(None)
        self._thread.join(timeout)
        return self.rows

    def arrival_index(self):
        with self._lock:
            if not self._arrival:
                return np.zeros(0), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
            arr = np.concatenate(self._arrival)
        return arr["host_s"], arr["frame_id"], arr["ts_us"]

    # -- writer thread --------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    self._fh.flush(); os.fsync(self._fh.fileno()); self._fh.close()
                    return
                if isinstance(item, threading.Event):
                    self._fh.flush(); os.fsync(self._fh.fileno()); item.set()
                    continue
                self._write(item)
            except Exception as exc:                  # never kill the writer silently
                self.error = f"{type(exc).__name__}: {exc}"
                if isinstance(item, threading.Event):
                    item.set()
                if item is None:
                    return

    def _write(self, payload: dict) -> None:
        raw = payload_to_raw(payload, self._synth, self.fs)
        if raw.size == 0:
            return
        raw.tofile(self._fh)
        ok = raw["valid"] == 1
        with self._lock:
            self.rows += raw.size
            self.valid_rows += int(ok.sum())
            self.late_rows += int(raw["late"][ok].sum())
            emg = raw["emg"][ok][:, :self.emg_channels]
            if emg.size:
                self.clip_counts[:self.emg_channels] += ((emg <= CLIP_LOW) | (emg >= CLIP_HIGH)).sum(axis=0)
            if payload.get("device_counters"):
                self.device_counters = dict(payload["device_counters"])
            if ok.any():
                # One arrival entry per burst of frames that reached the host together (same packet / same batch):
                # the NEWEST frame of the burst is the latest one received at that time. Streams without packet
                # numbers (the CSV simulator) are grouped by arrival time alone.
                v = raw[ok]
                if len(v) > 1:
                    change = (v["seq"][1:] != v["seq"][:-1]) | (v["host_s"][1:] != v["host_s"][:-1])
                    last = np.concatenate((np.flatnonzero(change), [len(v) - 1]))
                else:
                    last = np.array([0])
                self._arrival.append(v[last][["host_s", "frame_id", "ts_us"]].copy())


# ---------------------------------------------------------------------------------------------
# Post-processing: raw file + events -> labelled signal CSV, events CSV, quality JSON
# ---------------------------------------------------------------------------------------------
def finalize_session(raw_path: str, rows: int, events: list, arrival: tuple, out_dir: str, session: str,
                     fs: float = SAMPLE_RATE_HZ, transport: Optional[dict] = None,
                     recorder_stats: Optional[dict] = None, keepalive_restarts: Optional[list] = None,
                     host_origin_s: Optional[float] = None, extra_quality: Optional[dict] = None,
                     chunk_rows: int = 50_000, emg_channels: int = EMG_CHANNELS) -> dict:
    """Write signals_/events_/quality_ files for the first ``rows`` raw frames. Safe to call repeatedly."""
    os.makedirs(out_dir, exist_ok=True)
    host_s, arr_frames, arr_ts = arrival
    latency_s = estimate_latency_s(host_s, arr_ts)
    resolved = resolve_events(events, host_s, arr_frames, latency_s, fs)
    ev_frames = np.array([e["device_frame_id_at_event"] for e in resolved], dtype=np.int64)

    # Event table
    events_path = os.path.join(out_dir, f"events_{session}.csv")
    with open(events_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=EVENT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(resolved)

    # Per-event labelling attributes (aborted cycles are relabelled so they cannot be mistaken for trials)
    names, codes, phases, hands, trials = [], [], [], [], []
    for e in resolved:
        if e["aborted"]:
            names.append("aborted"); codes.append(PSEUDO_LABELS["aborted"]); phases.append("aborted")
        else:
            names.append(e["label"]); codes.append(e["label_code"]); phases.append(e["phase"])
        hands.append(e["hand"]); trials.append(e["trial_id"])

    mm = np.memmap(raw_path, dtype=RAW_DTYPE, mode="r", shape=(int(rows),)) if rows > 0 else np.zeros(0, RAW_DTYPE)
    first_valid = np.flatnonzero(mm["valid"] == 1)[:1] if rows > 0 else []
    ts0 = int(mm["ts_us"][first_valid[0]]) if len(first_valid) else 0
    origin = float(host_origin_s) if host_origin_s is not None else (float(host_s[0]) if len(host_s) else 0.0)

    line_format = _line_format(emg_channels)
    n_emg = max(1, min(EMG_CHANNELS, int(emg_channels)))
    phase_actual: dict = {}
    phase_valid: dict = {}
    gaps: list = []
    open_gap: Optional[dict] = None
    signals_path = os.path.join(out_dir, f"signals_{session}.csv")
    with open(signals_path, "w", newline="", encoding="utf-8") as fh:
        fh.write(",".join(SIGNAL_COLUMNS) + "\n")
        for start in range(0, int(rows), chunk_rows):
            c = mm[start:start + chunk_rows]
            ev = assign_events(c["frame_id"], ev_frames) if len(ev_frames) else np.zeros(len(c), dtype=np.int64)
            valid = c["valid"] == 1
            emg = np.where(valid[:, None], c["emg"].astype(np.float32), np.float32("nan"))
            ts = np.where(valid, c["ts_us"] - ts0, -1)
            host_rel = np.where(np.isfinite(c["host_s"]), c["host_s"] - origin, np.nan)
            phase_col = [phases[k] if len(phases) else "" for k in ev.tolist()]
            lines = []
            cols = zip(c["frame_id"].tolist(), ts.tolist(), c["seq"].tolist(), valid.astype(int).tolist(),
                       c["late"].astype(int).tolist(), *[emg[:, j].tolist() for j in range(n_emg)],
                       c["imu"][:, 0].tolist(), c["imu"][:, 1].tolist(), c["imu"][:, 2].tolist(),
                       c["imu_id"].tolist(), c["fresh"].astype(int).tolist(), host_rel.tolist(), ev.tolist())
            for row in cols:
                k = row[-1]
                lines.append(line_format % (*row[:-1], names[k] if names else "", codes[k] if codes else -1,
                                            phases[k] if phases else "", hands[k] if hands else "",
                                            trials[k] if trials else 0))
            fh.write("".join(lines))
            # quality bookkeeping per phase and gap runs
            for k in np.unique(ev):
                m = ev == k
                p = phases[k] if phases else ""
                phase_actual[p] = phase_actual.get(p, 0) + int(m.sum())
                phase_valid[p] = phase_valid.get(p, 0) + int((valid & m).sum())
            gaps, open_gap = _track_gaps(gaps, open_gap, c["frame_id"], ~valid, ev, names, phases)
    if open_gap is not None:
        gaps.append(open_gap)

    # expected frames per phase from planned durations (paused / aborted time is not planned data)
    expected: dict = {}
    for e in resolved:
        if not e["aborted"] and e["phase"] != "paused":
            expected[e["phase"]] = expected.get(e["phase"], 0) + e["planned_ms"] / 1000.0 * fs
    stats = dict(recorder_stats or {})
    total = int(rows)
    valid_total = int(stats.get("valid_rows", sum(phase_valid.values())))
    clip = stats.get("clip_counts")
    quality = {
        "session": session,
        "rows_total": total, "rows_valid": valid_total,
        "valid_sample_ratio": (valid_total / total) if total else 0.0,
        "loss_pct_valid_samples": (100.0 * (1.0 - valid_total / total)) if total else 0.0,
        "min_valid_sample_ratio": MIN_VALID_SAMPLE_RATIO,
        "meets_valid_sample_threshold": bool(total and valid_total / total >= MIN_VALID_SAMPLE_RATIO),
        "sample_rate_hz": fs,
        "transport": transport or {},
        "gaps": gaps,
        "device_counters": stats.get("device_counters", {}),
        "late_frames": int(stats.get("late_rows", 0)),
        "keepalive_restarts": list(keepalive_restarts or []),
        "emg_channels_recorded": n_emg,
        "clipping_pct_per_channel": ([round(100.0 * float(x) / max(1, valid_total), 4) for x in clip][:n_emg]
                                     if clip is not None else []),
        "phases": {p: {"expected_frames": int(round(expected.get(p, 0))), "actual_frames": phase_actual.get(p, 0),
                       "valid_frames": phase_valid.get(p, 0)} for p in sorted(set(expected) | set(phase_actual))},
        "latency_estimate_s": latency_s,
        "events": len(resolved),
    }
    quality.update(extra_quality or {})
    quality_path = os.path.join(out_dir, f"quality_{session}.json")
    with open(quality_path, "w", encoding="utf-8") as fh:
        json.dump(quality, fh, indent=2, default=float)
    del mm
    return {"signals_path": signals_path, "events_path": events_path, "quality_path": quality_path,
            "quality": quality, "rows": total}


def _track_gaps(gaps: list, open_gap: Optional[dict], fid: np.ndarray, bad: np.ndarray, ev: np.ndarray,
                names: list, phases: list):
    """Extend / close runs of invalid frames across chunk boundaries (one entry per contiguous gap)."""
    n = len(bad)
    if n == 0:
        return gaps, open_gap
    if open_gap is not None and not bad[0]:           # the previous chunk's trailing gap ended at the boundary
        gaps.append(open_gap); open_gap = None
    edges = np.flatnonzero(np.diff(np.concatenate(([0], bad.astype(np.int8), [0]))))
    for s, e in zip(edges[0::2].tolist(), edges[1::2].tolist()):    # [s, e) runs of bad frames
        if open_gap is not None and s == 0:
            open_gap["length"] += e
        else:
            k = int(ev[s])
            open_gap = {"start_frame": int(fid[s]), "length": e - s,
                        "label": names[k] if names else "", "phase": phases[k] if phases else ""}
        if e < n:
            gaps.append(open_gap); open_gap = None
    return gaps, open_gap
