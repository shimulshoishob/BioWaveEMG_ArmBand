import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

import main as app

QAPP = QApplication.instance() or QApplication([])


def order(widget):
    return [(s["hand"], s["trial_id"], s["label"]) for s in widget.steps if s["phase"] == "Perform"]


class ProtocolTests(unittest.TestCase):
    def widget(self, seed, randomize=True, hand_order="left_right"):
        w = app.TaskProtocolWidget()
        w.configure(app.DEFAULT_TASK_LABELS.split(","), 3, 4, 3, 3, randomize_order=randomize, hand_order=hand_order, seed=seed)
        return w

    def test_same_seed_same_order_different_seed_different_order(self):
        self.assertEqual(order(self.widget(123)), order(self.widget(123)))
        self.assertNotEqual(order(self.widget(123)), order(self.widget(124)))

    def test_randomization_actually_shuffles_and_can_be_off_for_practice(self):
        fixed = order(self.widget(1, randomize=False))
        labels = app.DEFAULT_TASK_LABELS.split(",")
        self.assertEqual([x[2] for x in fixed[:10]], labels)
        self.assertNotEqual([x[2] for x in order(self.widget(1))[:10]], labels)

    def test_study_mode_locks_randomization_and_practice_unlocks_it(self):
        d = app.DataCollectionDialog({"participant_id": "S01"})
        self.assertTrue(d.check_randomize_order.isChecked() and not d.check_randomize_order.isEnabledTo(d.session_config_section))
        d.check_randomize_order.setChecked(False)                       # cannot be turned off by code paths either
        d._apply_practice_rules()
        self.assertTrue(d.check_randomize_order.isChecked())
        d.check_practice.setChecked(True)
        self.assertTrue(d.check_randomize_order.isEnabledTo(d.session_config_section))
        d.check_randomize_order.setChecked(False)
        self.assertFalse(d.settings_payload()["randomize_order"])
        self.assertTrue(d.settings_payload()["practice_mode"])

    def test_hand_order_alternates_with_participant_and_manual_override_is_recorded(self):
        d = app.DataCollectionDialog({})
        d.input_contributor.setText("S01"); self.assertEqual(d.combo_hand_order.currentData(), "left_right")
        d.input_contributor.setText("S02"); self.assertEqual(d.combo_hand_order.currentData(), "right_left")
        self.assertEqual(d.settings_payload()["hand_order_source"], "auto")
        self.assertEqual(d.settings_payload()["hand_order_default"], "right_left")
        d.combo_hand_order.activated.emit(0)                           # operator picks an order by hand
        d.combo_hand_order.setCurrentIndex(0)
        d.input_contributor.setText("S04"); self.assertEqual(d.combo_hand_order.currentData(), "left_right")
        self.assertEqual(d.settings_payload()["hand_order_source"], "manual")


if __name__ == "__main__":
    unittest.main()


class MainsTests(unittest.TestCase):
    def test_mains_is_50hz_with_harmonics_inside_the_analysis_band(self):
        self.assertEqual(app.MAINS_FREQS, [50.0, 100.0, 150.0, 200.0])
        self.assertTrue(all(f < app.SAMPLE_RATE / 2 for f in app.MAINS_FREQS))
