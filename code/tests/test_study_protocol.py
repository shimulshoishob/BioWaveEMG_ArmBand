import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

import main as app

QAPP = QApplication.instance() or QApplication([])


def order(widget):
    return [(s["hand"], s["trial_id"], s["label"]) for s in widget.steps if s["phase"] == "Perform"]


class ProtocolTests(unittest.TestCase):
    def widget(self, seed, randomize=False, hand_order="left_right"):
        w = app.TaskProtocolWidget()
        w.configure(app.DEFAULT_TASK_LABELS.split(","), 3, 4, 3, 3, randomize_order=randomize, hand_order=hand_order, seed=seed)
        return w

    def test_default_order_is_sequential_in_every_repetition(self):
        labels = app.DEFAULT_TASK_LABELS.split(",")
        steps = order(self.widget(123))
        self.assertEqual(steps, order(self.widget(124)))
        n = len(labels)
        for i in range(0, 3 * n, n):
            self.assertEqual([x[2] for x in steps[i:i + n]], labels)

    def test_shuffle_only_when_explicitly_requested(self):
        self.assertNotEqual(order(self.widget(123, randomize=True)), order(self.widget(124, randomize=True)))

    def test_randomization_actually_shuffles_and_can_be_off_for_practice(self):
        fixed = order(self.widget(1, randomize=False))
        labels = app.DEFAULT_TASK_LABELS.split(",")
        self.assertEqual([x[2] for x in fixed[:10]], labels)
        self.assertNotEqual([x[2] for x in order(self.widget(1, randomize=True))[:10]], labels)

    def test_no_breaks_and_no_removed_options(self):
        w = self.widget(5, hand_order="left")
        phases = {st["phase"] for st in w.steps}
        self.assertEqual(phases, {"Prepare", "Perform", "Rest"})          # no Break / Switch Hand steps
        self.assertEqual({st["block"] for st in w.steps}, {1})            # one continuous block
        self.assertFalse(hasattr(w, "btn_skip_break"))
        self.assertEqual(w.session_total_ms, sum(st["duration_ms"] for st in w.steps))
        d = app.DataCollectionDialog({"participant_id": "S01"})
        for gone in ("spin_hand_switch", "check_practice", "check_randomize_order", "input_photo"):
            self.assertFalse(hasattr(d, gone), gone)
        payload = d.settings_payload()
        self.assertFalse(payload["randomize_order"])
        self.assertEqual((payload["block_break_s"], payload["hand_switch_s"]), (0.0, 0.0))

    def test_hand_must_be_chosen_every_session_and_only_left_or_right_exist(self):
        d = app.DataCollectionDialog({"hand_order": "right"})                  # even a stored hand is NOT pre-selected
        self.assertEqual(d.settings_payload()["hand_order"], "")
        self.assertEqual([d.combo_hand_order.itemData(i) for i in range(d.combo_hand_order.count())], ["", "left", "right"])
        self.assertIn("Choose the hand", d.lbl_plan.text())
        d.combo_hand_order.setCurrentIndex(d.combo_hand_order.findData("left"))
        self.assertEqual(d.settings_payload()["hand_order"], "left")
        self.assertIn("LEFT HAND", d.lbl_plan.text())

    def test_proceed_is_refused_until_a_hand_is_chosen(self):
        from PyQt5.QtWidgets import QMessageBox
        shown = []
        old = QMessageBox.warning
        QMessageBox.warning = staticmethod(lambda *a, **k: shown.append(a[1]))
        try:
            d = app.DataCollectionDialog({"participant_id": "S01", "agreed": True, "task_labels": ["left", "rest"],
                                          "labels_saved": True, "temp_start_c": 22.0, "humidity_start_pct": 45.0})
            d._emit_start_requested()
            self.assertEqual(shown, ["Select the Hand"])
            self.assertFalse(d.session_prepared)
        finally:
            QMessageBox.warning = old

    def test_time_button_reveals_the_fixed_timing(self):
        d = app.DataCollectionDialog({"task_labels": app.DEFAULT_TASK_LABELS.split(","), "labels_saved": True})
        self.assertEqual(d.btn_time.text().split()[0], "Time")
        self.assertTrue(d.time_panel.isHidden())
        d.btn_time.setChecked(True)
        self.assertFalse(d.time_panel.isHidden())
        text = d.time_panel.text()
        for expected in ("4 s", "3 s", "10 s", "90"):
            self.assertIn(expected, text)

    def test_repeats_editable_and_electrode_fields_fixed(self):
        d = app.DataCollectionDialog({})
        d.spin_task_repeats.setValue(2)
        self.assertEqual(d.settings_payload()["repeats"], 2)
        self.assertTrue(d.input_skin_prep.isReadOnly() and d.input_electrode_type.isReadOnly())
        d.combo_ch1_muscle.setCurrentIndex(2)
        el = d.settings_payload()["electrode"]
        self.assertEqual((el["skin_preparation"], el["electrode_type"]), ("Alcohol wipe", "Dry electrodes"))
        self.assertEqual(el["channel1_muscle"], app.CHANNEL1_MUSCLES[1])
        self.assertEqual(d.combo_ch1_muscle.count(), 1 + len(app.CHANNEL1_MUSCLES))

    def test_existing_session_for_the_same_hand_is_detected(self):
        import tempfile
        from study_meta import find_existing_sessions
        with tempfile.TemporaryDirectory() as t:
            os.makedirs(os.path.join(t, "S01_day01_left_20261103T045043Z"))
            os.makedirs(os.path.join(t, "practice_S01_day01_left_20261103T045043Z"))
            self.assertEqual(len(find_existing_sessions(t, "S01", 1, "left")), 1)
            self.assertEqual(find_existing_sessions(t, "S01", 1, "right"), [])
            self.assertEqual(find_existing_sessions(t, "S01", 2, "left"), [])


if __name__ == "__main__":
    unittest.main()


class MainsTests(unittest.TestCase):
    def test_mains_is_50hz_with_harmonics_inside_the_analysis_band(self):
        self.assertEqual(app.MAINS_FREQS, [50.0, 100.0, 150.0, 200.0])
        self.assertTrue(all(f < app.SAMPLE_RATE / 2 for f in app.MAINS_FREQS))
