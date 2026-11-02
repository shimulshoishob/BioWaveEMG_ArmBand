#!/usr/bin/env python3
"""Repair pilot recordings: restore packet order, make real packet loss explicit, never edit in place.

Reads every ``emg_data_*.csv`` under ``dataset/`` and writes a repaired copy under ``dataset_repaired/``
(the source tree is only ever read). For each session:

  * rows are sorted by ``Packet_Number`` inside each trial (stable: rows of one packet keep their order);
  * every jump in packet numbers is classified:
        - inside a cycle (gesture -> Rest): a real loss (the recording was continuous), unless it matches the
          configured settle time (an unrecorded settle phase);
        - between cycles: the session's regular jump (median of its cycle boundaries, ~200 / ~400 packets) is
          the unrecorded prepare period; a jump much smaller than that is a real loss; anything else
          (breaks, hand switches) is "unrecorded_other";
  * real loss becomes NaN rows with ``Valid=0`` (5 rows per missing packet, label copied from the row
    before the gap). Device timestamps are never invented: ``Timestamp_ms`` is left empty on those rows;
  * a per-session gap table and quality JSON are written and the output is marked ``pilot``.

Participant names in the source folder names are not carried over: outputs are named PILOT_001, PILOT_002,
... and the mapping is written to ``code/participant_map.json`` (gitignored, never exported).

Usage:  python code/tools/repair_sessions.py [--dataset DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Optional

FRAMES_PER_PACKET = 5
SAMPLE_RATE_HZ = 500
PACKET_RATE_HZ = SAMPLE_RATE_HZ / FRAMES_PER_PACKET           # 100 packets / s
DEFAULT_PREPARE_PACKETS = (200, 400)                          # 2 s and 4 s prepare at 100 packets / s
SEQ_SPAN = 1 << 32

REAL_LOSS, PREPARE, SETTLE, OTHER = "real_loss", "prepare", "settle", "unrecorded_other"


def _signed_delta(a: int, b: int) -> int:
    return ((b - a + SEQ_SPAN // 2) % SEQ_SPAN) - SEQ_SPAN // 2


def is_boundary(trial_changed: bool, label_before: str, label_after: str = "") -> bool:
    """A jump between two cycles (Prepare is not recorded, so it shows up as missing packets).

    In the pilot files a trial id is a *repetition* shared by all gestures, so a trial change is not the only
    boundary: after the Rest phase the next gesture starts a new cycle. Inside a cycle (gesture -> Rest) the
    recording is continuous.
    """
    before, after = label_before.strip().lower(), label_after.strip().lower()
    new_gesture = bool(after) and after != "rest" and after != before     # e.g. recordings without Rest rows
    return trial_changed or before == "rest" or new_gesture


def classify_jump(missing: int, boundary: bool, label_before: str, label_after: str,
                  prepare_ref: int, settle_packets: int, tolerance: float = 0.15) -> str:
    """Name the cause of ``missing`` absent packets between two recorded rows.

    ``prepare_ref`` is the session's regular between-cycle jump (the unrecorded prepare period, ~200 or ~400
    packets). Inside a cycle every missing packet is real loss (except an unrecorded settle phase). Between cycles
    a jump well below the regular one is real loss, one close to it is the prepare period, anything else is a
    longer unrecorded interval (block break, hand switch).
    """
    if not boundary:
        if (settle_packets and abs(missing - settle_packets) <= 1 and label_after.lower() == "rest"
                and label_before.lower() != "rest"):
            return SETTLE
        return REAL_LOSS
    if missing < 0.5 * prepare_ref:
        return REAL_LOSS
    if abs(missing - prepare_ref) <= max(10, tolerance * prepare_ref):
        return PREPARE
    return OTHER


def regular_prepare_jump(sizes, hint) -> int:
    """The session's usual between-cycle jump: the median of clear cycle boundaries, else the configured hint."""
    sizes = sorted(sizes)
    if len(sizes) >= 3:
        return int(sizes[len(sizes) // 2])
    return int(hint[0] if hint else DEFAULT_PREPARE_PACKETS[0])


def repair_rows(header: list, rows: list, prepare_packets=DEFAULT_PREPARE_PACKETS, settle_packets: int = 0):
    """Sort, classify and fill. Returns (header_out, rows_out, gaps, stats). Pure: no file access."""
    idx = {name.strip().lower(): i for i, name in enumerate(header)}
    for need in ("packet_number", "trial_id", "label"):
        if need not in idx:
            raise ValueError(f"CSV is missing the {need!r} column")
    c_pkt, c_trial, c_label = idx["packet_number"], idx["trial_id"], idx["label"]
    c_hand, c_ts = idx.get("hand"), idx.get("timestamp_ms")
    ch_cols = [i for name, i in idx.items() if name.startswith("ch") and name[2:].isdigit()]

    def pkt(row) -> int:
        return int(float(row[c_pkt]))

    if any(pkt(r) < 0 for r in rows[:50]):
        raise ValueError("Packet_Number is not available in this file (negative placeholders)")

    # group by trial (and hand) in order of first appearance, then sort each group by packet number
    groups: dict = {}
    for original_pos, row in enumerate(rows):
        key = (row[c_hand] if c_hand is not None else "", row[c_trial])
        groups.setdefault(key, []).append((pkt(row), original_pos, row))
    flat, keys, moved, duplicate_rows = [], [], 0, 0
    for key, items in groups.items():
        items_sorted = sorted(items, key=lambda t: (t[0], t[1]))
        moved += sum(1 for a, b in zip(items, items_sorted) if a[1] != b[1])
        per_packet: dict = {}
        for number, _pos, _row in items_sorted:
            per_packet[number] = per_packet.get(number, 0) + 1
        duplicate_rows += sum(n - FRAMES_PER_PACKET for n in per_packet.values() if n > FRAMES_PER_PACKET)
        flat.extend(t[2] for t in items_sorted)
        keys.extend([key] * len(items_sorted))

    # pass 1: every packet-number jump
    jumps = []                                    # (row position, missing packets, boundary?)
    for pos in range(1, len(flat)):
        delta = _signed_delta(pkt(flat[pos - 1]), pkt(flat[pos]))
        if delta > 1:
            jumps.append((pos, delta - 1, is_boundary(keys[pos] != keys[pos - 1], flat[pos - 1][c_label],
                                                      flat[pos][c_label])))
    clear = [m for pos, m, b in jumps
             if b and flat[pos][c_label].strip().lower() != "rest" and flat[pos][c_label] != flat[pos - 1][c_label]]
    ref = regular_prepare_jump(clear, prepare_packets)

    # pass 2: classify and fill
    out, gaps = [], []
    jump_at = {pos: (m, b) for pos, m, b in jumps}
    for pos, row in enumerate(flat):
        if pos in jump_at:
            missing, boundary = jump_at[pos]
            previous = flat[pos - 1]
            kind = classify_jump(missing, boundary, previous[c_label], row[c_label], ref, settle_packets)
            gap = {"gap_index": len(gaps), "class": kind, "trial_before": previous[c_trial], "trial_after": row[c_trial],
                   "label_before": previous[c_label], "label_after": row[c_label],
                   "last_packet_before": pkt(previous), "first_packet_after": pkt(row),
                   "missing_packets": missing, "missing_frames": missing * FRAMES_PER_PACKET, "rows_inserted": 0}
            if kind == REAL_LOSS:
                for k in range(missing):
                    nan_row = list(previous)
                    for c in ch_cols:
                        nan_row[c] = "nan"
                    nan_row[c_pkt] = str(pkt(previous) + 1 + k)
                    if c_ts is not None:
                        nan_row[c_ts] = ""                  # no invented timestamps
                    out.extend([nan_row + ["0"] for _ in range(FRAMES_PER_PACKET)])
                gap["rows_inserted"] = missing * FRAMES_PER_PACKET
            gaps.append(gap)
        out.append(row + ["1"])
    stats = {"rows_in": len(rows), "rows_out": len(out), "rows_moved_by_sort": moved, "duplicate_rows": duplicate_rows,
             "rows_inserted": sum(g["rows_inserted"] for g in gaps),
             "regular_prepare_jump_packets": ref,
             "real_loss_gaps": sum(1 for g in gaps if g["class"] == REAL_LOSS),
             "real_loss_packets": sum(g["missing_packets"] for g in gaps if g["class"] == REAL_LOSS),
             "prepare_jumps": sum(1 for g in gaps if g["class"] == PREPARE),
             "settle_jumps": sum(1 for g in gaps if g["class"] == SETTLE),
             "unrecorded_other_jumps": sum(1 for g in gaps if g["class"] == OTHER)}
    stats["loss_pct_of_output_rows"] = (100.0 * stats["rows_inserted"] / stats["rows_out"]) if stats["rows_out"] else 0.0
    return list(header) + ["Valid"], out, gaps, stats


def read_metadata(path: str) -> dict:
    meta = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if "=" in line:
                    k, v = line.rstrip("\n").split("=", 1)
                    meta[k.strip()] = v.strip()
    except OSError:
        pass
    return meta


def find_sessions(dataset_dir: str) -> list:
    found = []
    for root, _dirs, files in os.walk(dataset_dir):
        for name in sorted(files):
            if name.startswith("emg_data_") and name.endswith(".csv"):
                found.append(os.path.join(root, name))
    return sorted(found)


def repair_session(csv_path: str, out_dir: str, alias: str) -> dict:
    meta_path = os.path.join(os.path.dirname(csv_path), "metadata_" + os.path.basename(csv_path)[len("emg_data_"):].replace(".csv", ".txt"))
    meta = read_metadata(meta_path)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [r for r in reader if r]
    prep_s = float(meta["prep_s"]) if meta.get("prep_s") else None
    prepare = (int(round(prep_s * PACKET_RATE_HZ)),) if prep_s else DEFAULT_PREPARE_PACKETS   # only a fallback hint
    settle = int(round(float(meta.get("settle_s", 0) or 0) * PACKET_RATE_HZ))
    header_out, rows_out, gaps, stats = repair_rows(header, rows, prepare, settle)

    session_dir = os.path.join(out_dir, alias)
    os.makedirs(session_dir, exist_ok=True)
    with open(os.path.join(session_dir, f"pilot_signals_{alias}.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header_out)
        writer.writerows(rows_out)
    gap_fields = ["gap_index", "class", "trial_before", "trial_after", "label_before", "label_after",
                  "last_packet_before", "first_packet_after", "missing_packets", "missing_frames", "rows_inserted"]
    with open(os.path.join(session_dir, f"gaps_{alias}.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=gap_fields)
        writer.writeheader()
        writer.writerows(gaps)
    quality = {"status": "pilot", "session": alias, "source": "dataset/ (read only; names not carried over)",
               "frames_per_packet": FRAMES_PER_PACKET, "prepare_packets_from_metadata": list(prepare),
               "settle_packets_assumed": settle,
               "device_timestamps": "not available in pilot data; none were invented", **stats,
               "gap_classes": {k: sum(1 for g in gaps if g["class"] == k) for k in (REAL_LOSS, PREPARE, SETTLE, OTHER)}}
    with open(os.path.join(session_dir, f"quality_{alias}.json"), "w", encoding="utf-8") as fh:
        json.dump(quality, fh, indent=2)
    kept = [f"{k}={v}" for k, v in meta.items() if k not in ("contributor", "data_file", "data_path", "created_at")]
    with open(os.path.join(session_dir, f"metadata_{alias}.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(kept + [f"data_status=pilot", f"session_alias={alias}",
                                   f"data_file=pilot_signals_{alias}.csv", "repaired_by=code/tools/repair_sessions.py"]) + "\n")
    return {"alias": alias, **stats}


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=os.path.join(root, "dataset"))
    parser.add_argument("--out", default=os.path.join(root, "dataset_repaired"))
    parser.add_argument("--map-file", default=os.path.join(root, "code", "participant_map.json"))
    args = parser.parse_args(argv)
    if os.path.abspath(args.out) == os.path.abspath(args.dataset) or os.path.abspath(args.out).startswith(
            os.path.abspath(args.dataset) + os.sep):
        print("refusing to write inside the source dataset directory", file=sys.stderr)
        return 2

    sessions = find_sessions(args.dataset)
    mapping, results = {}, []
    for n, path in enumerate(sessions, 1):
        alias = f"PILOT_{n:03d}"
        mapping[alias] = os.path.relpath(path, args.dataset)
        try:
            results.append(repair_session(path, args.out, alias))
        except Exception as exc:                       # keep going: report the session that could not be repaired
            results.append({"alias": alias, "error": str(exc)})
    if mapping:
        with open(args.map_file, "w", encoding="utf-8") as fh:
            json.dump(mapping, fh, indent=2)

    cols = ("alias", "rows_in", "rows_out", "rows_moved_by_sort", "real_loss_gaps", "real_loss_packets",
            "prepare_jumps", "unrecorded_other_jumps", "loss_pct_of_output_rows")
    print("  ".join(f"{c[:12]:>12}" for c in cols))
    for r in results:
        if "error" in r:
            print(f"{r['alias']:>12}  ERROR: {r['error']}")
            continue
        print("  ".join(f"{(r[c] if c == 'alias' else (f'{r[c]:.3f}' if isinstance(r[c], float) else r[c])):>12}" for c in cols))
    print(f"\n{len(results)} session(s) -> {args.out} (status: pilot). Source folder untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
