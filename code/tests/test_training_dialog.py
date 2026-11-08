"""The Model Training dialog end to end on synthetic recordings (offscreen)."""
import json
import os
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QMessageBox

import main as app
from test_ml_models import synthetic_segments, write_session

QAPP = QApplication.instance() or QApplication([])


def write_recording(root, session, seed):
    """A continuous-format signals CSV (8 EMG channels) whose classes are separable."""
    import csv
    import numpy as np
    folder = os.path.join(root, session)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"signals_{session}.csv")
    frame = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Device_Frame_ID", "Valid"] + [f"Ch{i}" for i in range(1, 9)] + ["Label", "Phase", "Hand", "Trial_ID"])
        for label, seg in synthetic_segments(n_per_class=4, channels=8, seed=seed):
            phase = "rest" if label == "rest" else "perform"
            for row in seg:
                w.writerow([frame, 1] + [int(v) for v in row] + [label, phase, "Left", 1])
                frame += 1
    json.dump({"participant_id": "S0%d" % seed, "session_info": {"hand_order": "left", "session_day": 1},
               "protocol": {"labels": ["left", "right"]}, "result": {"rows": frame, "valid_sample_ratio": 1.0}},
              open(os.path.join(folder, f"metadata_{session}.json"), "w"))
    return path


class TrainingDialogTests(unittest.TestCase):
    def test_scan_select_train_two_models_and_files_land_in_model_folders(self):
        QMessageBox.warning = staticmethod(lambda *a, **k: self.fail(f"unexpected warning: {a[1:3]}"))
        QMessageBox.critical = staticmethod(lambda *a, **k: self.fail(f"unexpected error: {a[1:3]}"))
        with tempfile.TemporaryDirectory() as d:
            data = os.path.join(d, "data")
            write_recording(data, "S01_day01_left_20261105T000000Z", 1)
            write_recording(data, "S02_day01_left_20261105T000100Z", 2)
            win = app.EMGVisualizer()
            win.record_save_dir = data
            win.open_rf_training_dialog()
            dlg = win.training_dialog
            dlg.on_scan_dataset()
            self.assertEqual(len(dlg.dataset_paths()), 2)                        # new recordings are ticked
            self.assertIn("2 recording(s) selected", dlg.lbl_data_summary.text())
            dlg.output_dir_edit.setText(os.path.join(d, "models"))
            dlg._check_models(("random_forest", "lda"))
            dlg._param_widgets["random_forest"]["n_estimators"].setValue(40)
            dlg.spin_window_ms.setValue(200)
            dlg.combo_split.setCurrentIndex(dlg.combo_split.findData("recordings"))
            dlg.check_auto_load_model.setChecked(True)
            self.assertTrue(dlg.btn_train.isEnabled())
            dlg.on_train_clicked()
            deadline = time.time() + 120
            while dlg._is_training and time.time() < deadline:
                QAPP.processEvents(); time.sleep(0.02)
            QAPP.processEvents()
            self.assertFalse(dlg._is_training)
            self.assertEqual(sorted(os.listdir(os.path.join(d, "models"))), ["lda", "random_forest"])
            for folder in ("lda", "random_forest"):
                run = os.path.join(d, "models", folder, os.listdir(os.path.join(d, "models", folder))[0])
                self.assertTrue({"training_results.json", "training_setup.json", "confusion_matrix.csv",
                                 f"{folder}_model.joblib"} <= set(os.listdir(run)))
            setup = json.load(open(os.path.join(run, "training_setup.json")))
            self.assertEqual((setup["input_channels"], setup["split_mode"]), (8, "recordings"))
            self.assertEqual(dlg.compare_table.rowCount(), 2)
            self.assertGreater(float(dlg._runs[0]["metrics"]["accuracy"]), 0.8)
            self.assertIn(win.rf_model_type_name, ("Random Forest", "Linear Discriminant Analysis"))
            self.assertTrue(win.rf_model is not None)
            dlg.close(); win.close()

    def test_mismatched_channel_counts_disable_training(self):
        with tempfile.TemporaryDirectory() as d:
            new = write_recording(os.path.join(d, "a"), "S01_x", 1)
            legacy = os.path.join(d, "b", "emg_data_old.csv")
            os.makedirs(os.path.dirname(legacy))
            with open(legacy, "w") as fh:
                fh.write("Timestamp_ms,Packet_Number,Trial_ID," + ",".join(f"Ch{i}" for i in range(1, 12)) + ",Label\n")
                for i in range(30):
                    fh.write(f"{i},{i // 5},1," + ",".join(["1"] * 11) + ",Left\n")
            win = app.EMGVisualizer()
            win.open_rf_training_dialog()
            dlg = win.training_dialog
            dlg.add_dataset_paths([new, legacy])
            self.assertFalse(dlg.btn_train.isEnabled())
            self.assertIn("different channel counts", dlg.lbl_data_summary.text())
            dlg.close(); win.close()


class WorkflowBarTests(unittest.TestCase):
    def test_hint_follows_the_state(self):
        win = app.EMGVisualizer()
        win.refresh_workflow_hint()
        self.assertIn("Port Setup", win.lbl_workflow_hint.toolTip())
        win.is_connected = True; win.refresh_workflow_hint()
        self.assertIn("Calibrate", win.lbl_workflow_hint.toolTip())
        win.is_calibrated = True; win.refresh_workflow_hint()
        self.assertIn("Data Collection", win.lbl_workflow_hint.toolTip())
        win.rf_model = object(); win.refresh_workflow_hint()
        self.assertIn("Classify", win.lbl_workflow_hint.toolTip())
        win.rf_model = None; win.close()


class OptionsContentTests(unittest.TestCase):
    def test_every_toolbar_button_and_side_panel_control_is_described(self):
        win = app.EMGVisualizer()
        sections = win.build_options_info_sections()
        names = " | ".join([title for title, _ in sections] + [row[0] for _, rows in sections for row in rows]).lower()
        for btn in (win.btn_port_config, win.btn_disconnect, win.btn_calibrate, win.btn_analysis, win.btn_packet_tracer,
                    win.btn_options_info, win.btn_data_collection, win.btn_train_model, win.btn_load_trained_model,
                    win.btn_realtime_classification):
            self.assertIn(btn.text().lower(), names, f"Options window does not mention the '{btn.text()}' button")
        for text in ("window", "rms limit", "hz limit", "use hz gate", "auto-scale y axis", "draw live graph",
                     "room conditions", "electrode placement", "session length", "pause / resume"):
            self.assertIn(text, names)
        for stale in ("max cycles per block", "settle", "contributor", "toolbar"):
            self.assertNotIn(stale, [row[0].lower() for _, rows in sections for row in rows] + [t.lower() for t, _ in sections])
        win.close()


if __name__ == "__main__":
    unittest.main()
