"""The EMG simulator (TCP server, CSV lines) must connect to the app and record a session end to end."""
import csv
import json
import os
import socket
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QMessageBox

import emg_simulator_app as sim
import ml_models
import main as app

QAPP = QApplication.instance() or QApplication([])


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


def pump(seconds, until=lambda: False):
    end = time.time() + seconds
    while time.time() < end and not until():
        QAPP.processEvents(); time.sleep(0.01)


class SimulatorTests(unittest.TestCase):
    def start_sim(self, channels):
        port = free_port()
        worker = sim.SimulatorWorker("tcp_server", "", 921600, channels, 500, 3.0, "127.0.0.1", port)
        worker.start(); time.sleep(0.4)
        self.addCleanup(worker.stop)
        return port

    def test_connects_and_streams_for_4_and_8_channels(self):
        QMessageBox.critical = staticmethod(lambda *a, **k: self.fail(f"connection error: {a[1:3]}"))
        for channels in (4, 8):
            with self.subTest(channels=channels):
                port = self.start_sim(channels)
                win = app.EMGVisualizer()
                win.channel_count = channels
                win.connect_serial(f"socket://127.0.0.1:{port}")
                self.assertTrue(win.is_connected)
                self.assertFalse(win.wired_binary)                      # the simulator speaks CSV, not BWIM
                pump(2.0)
                snap = win.packet_tracer.snapshot()
                self.assertGreater(snap["samples"], 600)                # ~500 samples/s arrive
                win.disconnect_serial(); win.close()

    def test_records_a_session_from_a_four_channel_simulator(self):
        QMessageBox.warning = staticmethod(lambda *a, **k: self.fail(f"warning: {a[1:3]}"))
        QMessageBox.critical = staticmethod(lambda *a, **k: self.fail(f"error: {a[1:3]}"))
        QMessageBox.information = staticmethod(lambda *a, **k: None)
        with tempfile.TemporaryDirectory() as d:
            port = self.start_sim(4)
            win = app.EMGVisualizer()
            win.channel_count = 4
            win.connect_serial(f"socket://127.0.0.1:{port}")
            win.is_calibrated = True
            win.record_save_dir = d
            win.participant_id = "S01"; win.contribution_agreed = True
            win.study_meta.update(temp_start_c=22.0, humidity_start_pct=45.0, hand_order="left")
            win.open_data_collection_dialog()
            dlg = win.data_collection_dialog
            dlg._fixed_plan.update(prep_s=0.3, hold_s=0.4, rest_s=0.3)   # shortened for the test only
            dlg.spin_task_repeats.setValue(1)
            dlg.check_agree_yes.setChecked(True)
            dlg.input_contributor.setText("S01")
            dlg.spin_temp_start.setValue(22.0); dlg.spin_humidity_start.setValue(45.0)
            dlg.task_labels = ["left", "rest"]; dlg.labels_saved = True
            dlg.combo_hand_order.setCurrentIndex(dlg.combo_hand_order.findData("left"))
            QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
            dlg._update_access_state()
            dlg._emit_start_requested()
            win.start_task_timer_recording()
            self.assertIsNotNone(win.session_recorder)
            pump(30, until=lambda: not win.task_session_active)
            pump(2.5, until=lambda: not win.record_save_workers and win.session_recorder is None)
            self.assertFalse(win.task_session_active)
            folder = [f for f in os.listdir(d) if f.startswith("S01_")][0]
            files = os.listdir(os.path.join(d, folder))
            for prefix in ("signals_", "events_", "quality_", "metadata_"):
                self.assertTrue(any(f.startswith(prefix) for f in files), (prefix, files))
            signals = os.path.join(d, folder, [f for f in files if f.startswith("signals_")][0])
            rows = list(csv.DictReader(open(signals)))
            self.assertGreater(len(rows), 500)
            self.assertEqual(rows[10]["Ch5"], "")                         # a 4-channel stream leaves Ch5-Ch8 empty
            self.assertNotEqual(rows[10]["Ch4"], "")
            quality = json.load(open(os.path.join(d, folder, [f for f in files if f.startswith("quality_")][0])))
            self.assertEqual(quality["emg_channels_recorded"], 4)
            segments, channels = ml_models.load_segments(signals)         # and the recording is directly trainable
            self.assertEqual(channels, 4)
            self.assertTrue({"left", "rest"} <= {label for label, _ in segments}, [(l, len(a)) for l, a in segments] or __import__("collections").Counter((r["Phase"], r["Label"], r["Valid"]) for r in rows))
            win.disconnect_serial(); win.close()


if __name__ == "__main__":
    unittest.main()
