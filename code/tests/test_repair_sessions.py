import csv
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import repair_sessions as rs

HEADER = ["Timestamp_ms", "Packet_Number", "Trial_ID"] + [f"Ch{i}" for i in range(1, 12)] + ["Label", "Hand"]


def packet_rows(pkt, trial, label, hand="Left"):
    return [[f"{pkt * 10 + k}", str(pkt), str(trial)] + [str(2000 + k)] * 11 + [label, hand] for k in range(5)]


def build(spec):
    rows = []
    for pkt, trial, label in spec:
        rows += packet_rows(pkt, trial, label)
    return rows


class RepairTests(unittest.TestCase):
    def test_sorting_within_trial_and_untouched_clean_data(self):
        spec = [(p, 1, "Left") for p in (3, 1, 2)]
        h, out, gaps, st = rs.repair_rows(HEADER, build(spec))
        self.assertEqual([r[1] for r in out[::5]], ["1", "2", "3"])
        self.assertEqual((st["rows_inserted"], st["real_loss_gaps"], gaps), (0, 0, []))
        self.assertEqual(h[-1], "Valid"); self.assertTrue(all(r[-1] == "1" for r in out))
        self.assertGreater(st["rows_moved_by_sort"], 0)

    def test_real_loss_inside_trial_becomes_nan_rows(self):
        spec = [(10, 1, "Left"), (11, 1, "Left"), (14, 1, "Left"), (15, 1, "Rest")]   # packets 12, 13 lost
        h, out, gaps, st = rs.repair_rows(HEADER, build(spec))
        self.assertEqual((st["real_loss_gaps"], st["real_loss_packets"], st["rows_inserted"]), (1, 2, 10))
        inserted = [r for r in out if r[-1] == "0"]
        self.assertEqual(len(inserted), 10)
        self.assertEqual([r[1] for r in inserted[::5]], ["12", "13"])
        self.assertTrue(all(r[0] == "" for r in inserted))                     # timestamps not invented
        self.assertTrue(all(r[3] == "nan" and r[13] == "nan" for r in inserted))
        self.assertEqual(inserted[0][14], "Left")                              # label of the row before the gap
        self.assertEqual(gaps[0]["class"], "real_loss")

    def test_regular_prepare_jumps_are_not_repaired(self):
        spec = [(100, 1, "Rest"), (301, 2, "Left"), (302, 2, "Left")]          # 200 packets missing = prepare (2 s)
        _, out, gaps, st = rs.repair_rows(HEADER, build(spec), prepare_packets=(200,))
        self.assertEqual((st["rows_inserted"], st["prepare_jumps"]), (0, 1))
        self.assertEqual(gaps[0]["class"], "prepare")

    def test_small_jump_between_trials_is_real_loss_and_long_jump_is_other(self):
        spec = [(100, 1, "Rest"), (150, 2, "Left"), (150 + 1 + 900, 3, "Left")]
        _, out, gaps, st = rs.repair_rows(HEADER, build(spec), prepare_packets=(200,))
        self.assertEqual([g["class"] for g in gaps], ["real_loss", "unrecorded_other"])
        self.assertEqual(st["rows_inserted"], 49 * 5)

    def test_regular_jump_is_learned_from_the_session_and_trial_ids_are_not_continuity(self):
        # Pilot layout: trial id = repetition shared by every gesture; each gesture cycle is Left -> Rest.
        spec, p = [], 0
        for gesture in ("Left", "Right", "Up", "Down"):
            for _ in range(2):
                spec.append((p, 1, gesture)); p += 1
            spec.append((p, 1, "Rest")); p += 1
            p += 330                                       # this session's unrecorded prepare: 330 packets
        spec.append((p - 330 + 20, 1, "Left"))             # 20 missing inside the last Rest -> next gesture: real loss
        _, out, gaps, st = rs.repair_rows(HEADER, build(spec), prepare_packets=(200,))
        self.assertEqual(st["regular_prepare_jump_packets"], 330)
        self.assertEqual([g["class"] for g in gaps], ["prepare"] * 3 + ["real_loss"])

    def test_settle_gap_is_not_loss(self):
        spec = [(10, 1, "Left"), (11, 1, "Left"), (21, 1, "Rest")]             # 9 packets missing = 0.09 s? use 9 as settle
        _, _, gaps, st = rs.repair_rows(HEADER, build(spec), settle_packets=9)
        self.assertEqual((gaps[0]["class"], st["rows_inserted"]), ("settle", 0))

    def test_packet_number_wrap(self):
        spec = [(2**32 - 1, 1, "Left"), (0, 1, "Left")]
        _, out, gaps, st = rs.repair_rows(HEADER, build(spec))
        # sorted by number, the wrapped packet sorts first: nothing is lost, nothing is invented
        self.assertEqual(st["rows_inserted"] if gaps == [] else -1, 0)

    def test_end_to_end_never_writes_into_the_source_and_drops_names(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "dataset", "Some_Person_20260101_000000")
            os.makedirs(src)
            with open(os.path.join(src, "emg_data_Some_Person_20260101_000000.csv"), "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(HEADER); w.writerows(build([(1, 1, "Left"), (2, 1, "Left"), (5, 1, "Left")]))
            with open(os.path.join(src, "metadata_Some_Person_20260101_000000.txt"), "w") as fh:
                fh.write("contributor=Some_Person\nprep_s=2.0\nsettle_s=0.0\nsample_rate_hz=500\n")
            before = sorted(os.listdir(src))
            rc = rs.main(["--dataset", os.path.join(d, "dataset"), "--out", os.path.join(d, "repaired"),
                          "--map-file", os.path.join(d, "map.json")])
            self.assertEqual(rc, 0)
            self.assertEqual(sorted(os.listdir(src)), before)
            out = os.path.join(d, "repaired", "PILOT_001")
            self.assertEqual(sorted(os.listdir(out)), ["gaps_PILOT_001.csv", "metadata_PILOT_001.txt",
                                                       "pilot_signals_PILOT_001.csv", "quality_PILOT_001.json"])
            q = json.load(open(os.path.join(out, "quality_PILOT_001.json")))
            self.assertEqual((q["status"], q["real_loss_packets"], q["rows_inserted"]), ("pilot", 2, 10))
            for name in os.listdir(out):
                self.assertNotIn("Some_Person", open(os.path.join(out, name)).read())
            self.assertIn("Some_Person", open(os.path.join(d, "map.json")).read())   # mapping lives outside the output
            self.assertEqual(rs.main(["--dataset", os.path.join(d, "dataset"), "--out", os.path.join(d, "dataset", "x")]), 2)


if __name__ == "__main__":
    unittest.main()
