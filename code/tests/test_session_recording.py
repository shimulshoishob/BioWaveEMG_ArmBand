import csv
import json
import os
import tempfile
import time
import unittest

import numpy as np

from session_recording import (ContinuousRecorder, EventLog, SIGNAL_COLUMNS, assign_events, canonical_label,
                               estimate_latency_s, finalize_session, resolve_events)

FS = 500


def payload(first, n, t0, valid=None, seq0=0, period_us=2000):
    ids = np.arange(first, first + n)
    batch = np.tile(np.arange(11, dtype=np.float32) + 100, (n, 1))
    v = np.ones(n, np.uint8) if valid is None else np.asarray(valid, np.uint8)
    batch[v == 0] = np.nan
    return {"batch": batch, "frame_ids": ids, "emg_timestamps": np.where(v == 1, 5_000_000 + ids * period_us, -1),
            "packet_numbers": seq0 + ids // 5, "imu_ids": ids // 2, "valid": v,
            "late_flags": np.zeros(n, np.uint8), "imu_fresh": np.ones(n, np.uint8),
            "host_received_s": t0 + (ids - first) / FS, "host_received_monotonic": t0 + n / FS,
            "device_counters": {"queue_drops": 0, "acq_overruns": 2, "send_failures": 1, "wifi_reconnects": 0}}


class MappingTests(unittest.TestCase):
    def test_assign_events_switches_at_exact_frame(self):
        idx = assign_events(np.arange(0, 30), np.array([5, 12, 20]))
        self.assertEqual([int(idx[f]) for f in (0, 4, 5, 11, 12, 19, 20, 29)], [0, 0, 0, 0, 1, 1, 2, 2])

    def test_resolve_events_applies_latency_and_is_monotonic(self):
        host = np.arange(0, 10, 0.01); frames = (host * FS).astype(np.int64)
        ev = [{"host_monotonic_s": t} for t in (2.0, 2.0, 5.0, 1.0)]       # last one out of order
        out = resolve_events(ev, host, frames, latency_s=0.02, fs=FS)
        self.assertEqual(out[0]["device_frame_id_uncorrected"], 1000)
        self.assertEqual(out[0]["device_frame_id_at_event"], 1010)          # + 0.02 s * 500 Hz
        got = [e["device_frame_id_at_event"] for e in out]
        self.assertEqual(got, sorted(got))

    def test_estimate_latency(self):
        host = np.arange(100) * 0.01 + 5.0
        ts = (np.arange(100) * 10_000).astype(np.int64)
        host[50:] += 0.03                                                    # half the packets were delayed 30 ms
        self.assertAlmostEqual(estimate_latency_s(host, ts), 0.015, delta=0.0155)
        self.assertEqual(estimate_latency_s(host[:2], ts[:2]), 0.0)

    def test_non_task_phases_carry_their_own_label(self):
        log = EventLog()
        for phase in ("Prepare", "Perform", "Settle", "Rest", "Break", "Switch Hand", "Paused"):
            log.add(phase, "Left", "Left", 1, 1, 1, 0.0, 0)
        got = [(e["phase"], e["label"], e["label_code"]) for e in log.events]
        self.assertEqual(got, [("prepare", "prepare", 99), ("perform", "left", 1), ("settle", "settle", 98),
                               ("rest", "rest", 0), ("break", "rest", 0), ("switch_hand", "rest", 0),
                               ("paused", "paused", 97)])

    def test_canonical_label(self):
        self.assertEqual(canonical_label("Double_Closed_Fist"), ("double_fist_close", 6))
        self.assertEqual(canonical_label("Rest"), ("rest", 0))
        self.assertEqual(canonical_label("fist_close"), ("fist_close", 3))
        self.assertEqual(canonical_label("mystery")[1], -1)


class SessionFilesTests(unittest.TestCase):
    def test_end_to_end_labels_gaps_and_quality(self):
        with tempfile.TemporaryDirectory() as d:
            rec = ContinuousRecorder(os.path.join(d, "raw.bin"), FS)
            t0 = 1000.0
            log = EventLog()
            log.add("Prepare", "Prepare", "Left", 1, 1, 1, t0 + 0.0, 4000)          # frames   0..
            log.add("Perform", "Left", "Left", 1, 1, 1, t0 + 1.0, 3000)             # frame  500
            log.add("Rest", "Rest", "Left", 1, 1, 1, t0 + 2.0, 3000)                # frame 1000
            valid = np.ones(1500, np.uint8); valid[700:760] = 0                       # a 60-frame gap inside Perform
            for s in range(0, 1500, 250):
                rec.append(payload(s, 250, t0 + s / FS, valid[s:s + 250]))
            rec.checkpoint()
            out = finalize_session(rec.raw_path, rec.rows, log.events, rec.arrival_index(), d, "S01_test", FS,
                                   transport={"packets_missing": 12},
                                   recorder_stats={"valid_rows": rec.valid_rows, "clip_counts": rec.clip_counts,
                                                   "device_counters": rec.device_counters, "late_rows": rec.late_rows},
                                   host_origin_s=t0)
            rec.close()
            with open(out["signals_path"], newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(list(rows[0].keys()), SIGNAL_COLUMNS)
            self.assertEqual(len(rows), 1500)
            self.assertEqual({r["Phase"] for r in rows}, {"prepare", "perform", "rest"})
            by_frame = {int(r["Device_Frame_ID"]): r for r in rows}
            # 'perform' starts exactly at frame 500 (no latency: arrival pattern is uniform)
            first_perform = min(f for f, r in by_frame.items() if r["Phase"] == "perform")
            self.assertEqual(first_perform, 500)
            self.assertEqual(by_frame[499]["Phase"], "prepare"); self.assertEqual(by_frame[500]["Label"], "left")
            self.assertEqual(by_frame[500]["Label_Code"], "1")
            self.assertEqual(by_frame[1000]["Label"], "rest"); self.assertEqual(by_frame[1000]["Label_Code"], "0")
            self.assertEqual(by_frame[720]["Valid"], "0"); self.assertEqual(by_frame[720]["Ch1"], "nan")
            self.assertEqual(by_frame[720]["Device_Timestamp_us"], "-1")      # never invented
            self.assertEqual(by_frame[0]["Device_Timestamp_us"], "0")          # relative to session start
            self.assertEqual(by_frame[1]["Ch3"], "102")                        # raw counts untouched
            self.assertEqual(by_frame[300]["Hand"], "Left")
            with open(out["events_path"], newline="") as fh:
                ev = list(csv.DictReader(fh))
            self.assertEqual([e["phase"] for e in ev], ["prepare", "perform", "rest"])
            self.assertEqual(ev[1]["device_frame_id_at_event"], "500")
            q = json.load(open(out["quality_path"]))
            self.assertEqual(q["gaps"], [{"start_frame": 700, "length": 60, "label": "left", "phase": "perform"}])
            self.assertEqual(q["rows_valid"], 1440)
            self.assertAlmostEqual(q["loss_pct_valid_samples"], 4.0)
            self.assertFalse(q["meets_valid_sample_threshold"])                 # 96 % < 98 %
            self.assertEqual(q["phases"]["perform"]["actual_frames"], 500)
            self.assertEqual(q["phases"]["perform"]["valid_frames"], 440)
            self.assertEqual(q["phases"]["perform"]["expected_frames"], 1500)
            self.assertEqual(q["device_counters"]["acq_overruns"], 2)
            self.assertEqual(q["transport"]["packets_missing"], 12)

    def test_gap_spanning_chunk_boundary_is_one_gap(self):
        with tempfile.TemporaryDirectory() as d:
            rec = ContinuousRecorder(os.path.join(d, "raw.bin"), FS)
            valid = np.ones(400, np.uint8); valid[90:130] = 0
            rec.append(payload(0, 400, 10.0, valid)); rec.checkpoint()
            log = EventLog(); log.add("Perform", "Left", "Left", 1, 1, 1, 10.0, 1000)
            out = finalize_session(rec.raw_path, rec.rows, log.events, rec.arrival_index(), d, "x", FS, chunk_rows=100,
                                   host_origin_s=10.0)
            self.assertEqual([(g["start_frame"], g["length"]) for g in out["quality"]["gaps"]], [(90, 40)])
            rec.close()

    def test_nothing_is_dropped_and_aborted_cycle_is_relabelled(self):
        with tempfile.TemporaryDirectory() as d:
            rec = ContinuousRecorder(os.path.join(d, "raw.bin"), FS)
            log = EventLog()
            first = log.add("Prepare", "Prepare", "Left", 1, 1, 1, 0.0, 4000)
            log.add("Perform", "Left", "Left", 1, 1, 1, 1.0, 3000)
            log.add("Paused", "paused", "Left", 1, 1, 1, 1.5, 0)
            log.mark_aborted_since(first)
            log.add("Prepare", "Prepare", "Left", 1, 1, 1, 2.0, 4000)
            rec.append(payload(0, 1500, 0.0)); rec.checkpoint()
            out = finalize_session(rec.raw_path, rec.rows, log.events, rec.arrival_index(), d, "y", FS, host_origin_s=0.0)
            rows = list(csv.DictReader(open(out["signals_path"])))
            self.assertEqual(len(rows), 1500)
            self.assertEqual(rows[100]["Phase"], "aborted"); self.assertEqual(rows[100]["Label_Code"], "96")
            self.assertEqual(rows[800]["Phase"], "paused")
            self.assertEqual(rows[1200]["Phase"], "prepare")
            rec.close()

    def test_recording_does_not_block_and_survives_many_rows(self):
        with tempfile.TemporaryDirectory() as d:
            rec = ContinuousRecorder(os.path.join(d, "raw.bin"), FS)
            t = time.perf_counter()
            for s in range(0, 100_000, 100):
                rec.append(payload(s, 100, s / FS))
            self.assertLess(time.perf_counter() - t, 3.0)                       # producer never waits for the disk
            rec.checkpoint(); self.assertEqual(rec.rows, 100_000)
            self.assertEqual(os.path.getsize(rec.raw_path), 100_000 * rec_size())
            rec.close()


def rec_size():
    from session_recording import RAW_DTYPE
    return RAW_DTYPE.itemsize


if __name__ == "__main__":
    unittest.main()
