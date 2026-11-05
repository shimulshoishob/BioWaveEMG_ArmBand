import csv
import json
import os
import pickle
import tempfile
import unittest

import numpy as np

import ml_models as ml

FS = 500


def synthetic_segments(n_per_class=6, seconds=3, channels=4, seed=0):
    """Classes differ in which channel is active and how strongly, plus a DC offset (like raw ADC counts)."""
    rng = np.random.default_rng(seed)
    segs = []
    for cls, name in enumerate(["rest", "left", "right"]):
        for _ in range(n_per_class):
            t = np.arange(seconds * FS) / FS
            x = 2048 + rng.normal(0, 3, (len(t), channels))
            if cls:
                x[:, cls] += 60 * np.sin(2 * np.pi * (80 + 30 * cls) * t) * rng.uniform(0.8, 1.2)
            segs.append((name, x.astype(np.float32)))
    return segs


def write_session(folder, session, rows):
    """A continuous-format signals CSV: prepare (ignored), perform, gap, rest, aborted (ignored)."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"signals_{session}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Device_Frame_ID", "Valid", "Ch1", "Ch2", "Label", "Phase", "Hand", "Trial_ID"])
        for r in rows:
            w.writerow(r)
    return path


class LoaderTests(unittest.TestCase):
    def test_continuous_format_filters_phases_and_gaps(self):
        with tempfile.TemporaryDirectory() as d:
            rows = []
            for i in range(50):  rows.append([i, 1, 100 + i, 200, "prepare", "prepare", "Left", 1])
            for i in range(100): rows.append([50 + i, 1, 1000 + i, 2000, "left", "perform", "Left", 1])
            for i in range(10):  rows.append([150 + i, 0, "nan", "nan", "left", "perform", "Left", 1])     # gap
            for i in range(80):  rows.append([160 + i, 1, 1500 + i, 2500, "left", "perform", "Left", 1])
            for i in range(60):  rows.append([240 + i, 1, 10 + i, 20, "rest", "rest", "Left", 1])
            for i in range(30):  rows.append([300 + i, 1, 5, 5, "aborted", "aborted", "Left", 1])
            path = write_session(d, "S01_x", rows)
            segs, ch = ml.load_segments(path)
            self.assertEqual(ch, 2)
            self.assertEqual([(l, len(a)) for l, a in segs], [("left", 100), ("left", 80), ("rest", 60)])
            self.assertEqual(float(segs[0][1][0, 0]), 1000.0)
            self.assertEqual(ml.detect_channel_count(path), 2)

    def test_legacy_format_without_phase_or_valid(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "emg_data_old.csv")
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["Timestamp_ms", "Packet_Number", "Trial_ID", "Ch1", "Ch2", "Ch3", "Ch4", "Label"])
                for i in range(40): w.writerow([i, i // 5, 1, 1, 2, "", "", "Left"])
                for i in range(40): w.writerow([i, 8 + i // 5, 1, 3, 4, "", "", "Rest"])
            segs, ch = ml.load_segments(path)
            self.assertEqual(ch, 2)                                  # blank padded channels are not used
            self.assertEqual([(l, a.shape) for l, a in segs], [("Left", (40, 2)), ("Rest", (40, 2))])

    def test_describe_and_find_recordings(self):
        with tempfile.TemporaryDirectory() as d:
            folder = os.path.join(d, "S02_day01_left_20261105T000000Z")
            path = write_session(folder, "S02_day01_left_20261105T000000Z", [[0, 1, 1, 1, "left", "perform", "Left", 1]])
            json.dump({"participant_id": "S02", "session_info": {"hand_order": "left", "session_day": 1},
                       "protocol": {"labels": ["left", "right"]}, "result": {"rows": 123, "valid_sample_ratio": 0.99}},
                      open(os.path.join(folder, "metadata_S02_day01_left_20261105T000000Z.json"), "w"))
            open(os.path.join(folder, "events_S02_day01_left_20261105T000000Z.csv"), "w").write("a,b\n")
            self.assertEqual(ml.find_recordings(d), [path])          # events_*.csv is not a recording
            info = ml.describe_session(path)
            self.assertEqual((info["participant"], info["hand"], info["rows"], info["format"]), ("S02", "left", 123, "continuous"))
            self.assertAlmostEqual(info["valid_pct"], 99.0)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = ml.prepare_dataset(synthetic_segments(), window_samples=100, stride_samples=50, sample_rate=FS,
                                    trim_samples=25, want_features=True, want_windows=True)
        cls.train, cls.test = ml.split_train_test(cls.ds.y, cls.ds.groups, 3, 0.25, 1)

    def test_split_is_group_aware(self):
        self.assertEqual(set(self.ds.groups[self.train]) & set(self.ds.groups[self.test]), set())
        self.assertEqual(np.unique(self.ds.y[self.test]).size, 3)

    def test_trim_removes_segment_edges(self):
        a = ml.prepare_dataset(synthetic_segments(2), window_samples=100, stride_samples=50, sample_rate=FS)
        b = ml.prepare_dataset(synthetic_segments(2), window_samples=100, stride_samples=50, sample_rate=FS, trim_samples=100)
        self.assertLess(len(b.y), len(a.y))

    def test_every_available_model_learns_the_synthetic_task(self):
        for key, spec in ml.MODEL_SPECS.items():
            ok, why = spec.availability()
            if not ok:
                continue
            with self.subTest(model=key):
                params = {"epochs": 8} if key == "cnn1d" else {}
                res = ml.fit_and_evaluate(spec, params, self.ds, self.train, self.test, seed=3, balanced=True, sample_rate=FS)
                self.assertGreater(res.accuracy, 0.85, key)
                proba = res.estimator.predict_proba(
                    self.ds.windows[self.test[:5]] if spec.input_kind == "window" else self.ds.features[self.test[:5]])
                self.assertEqual(proba.shape, (5, 3))
                np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)
                self.assertEqual(list(res.estimator.classes_ if not hasattr(res.estimator, "steps") else res.estimator.classes_), [0, 1, 2])
                self.assertEqual(res.confusion.shape, (3, 3))
                self.assertGreater(res.latency_ms, 0)
                clone = pickle.loads(pickle.dumps(res.estimator))               # saved artifacts must reload
                data = self.ds.windows[self.test[:5]] if spec.input_kind == "window" else self.ds.features[self.test[:5]]
                np.testing.assert_allclose(clone.predict_proba(data), proba, atol=1e-4)

    def test_registry_has_the_requested_models_and_folders(self):
        for key in ("random_forest", "svm_rbf", "cnn1d"):
            self.assertIn(key, ml.MODEL_SPECS)
        folders = [s.folder for s in ml.MODEL_SPECS.values()]
        self.assertEqual(len(folders), len(set(folders)))
        self.assertTrue(all(f == f.lower() and " " not in f for f in folders))

    def test_cnn_cancel_and_unavailable_message(self):
        spec = ml.MODEL_SPECS["cnn1d"]
        if spec.availability()[0]:
            import threading
            cancel = threading.Event(); cancel.set()
            est = ml.build_estimator(spec, {"epochs": 5}, n_classes=3, n_channels=4, window_samples=100, seed=1, balanced=True)
            est.fit(self.ds.windows[:50], self.ds.y[:50], cancel=cancel)    # stops before the first epoch
            self.assertEqual(est.history, [])
        else:
            self.assertIn("pip install torch", spec.availability()[1])


if __name__ == "__main__":
    unittest.main()
