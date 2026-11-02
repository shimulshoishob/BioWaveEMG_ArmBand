import os
import socket
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtCore import QCoreApplication, Qt

import main as app
from test_emg_v4_core import make_v2_packet


class WirelessWorkerTests(unittest.TestCase):
    def test_loss_reorder_and_duplicate_over_udp(self):
        QCoreApplication.instance() or QCoreApplication([])
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]; probe.close()
        tracer = app.PacketTracer(); tracer.reset("wireless", 5)
        worker = app.WirelessStreamWorker(port, tracer=tracer)
        got = []
        worker.batch_received.connect(got.append, Qt.DirectConnection)
        worker.start(); time.sleep(0.2)

        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for seq in [0, 1, 2, 4, 3, 5, 5, 8, 9, 10]:        # 3/4 reordered, 5 duplicated, 6-7 lost
            tx.sendto(make_v2_packet(sequence=seq, first_frame=5 * seq), ("127.0.0.1", port))
            time.sleep(0.003)
        time.sleep(0.4)
        worker.stop()

        ids = np.concatenate([p["frame_ids"] for p in got]); valid = np.concatenate([p["valid"] for p in got])
        batch = np.concatenate([p["batch"] for p in got])
        self.assertTrue(np.array_equal(ids, np.arange(55)))
        self.assertTrue(np.array_equal(np.flatnonzero(valid == 0), np.arange(30, 40)))
        self.assertTrue(np.isnan(batch[valid == 0]).all())
        snap = tracer.snapshot()
        self.assertEqual((snap["missing"], snap["duplicate"], snap["frames_missing"]), (2, 1, 10))
        self.assertEqual(snap["device_counters"]["queue_drops"], 1)


if __name__ == "__main__":
    unittest.main()
