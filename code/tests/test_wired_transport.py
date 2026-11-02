"""SerialWorker against a fake firmware (TCP socket:// URL) that speaks the USB protocol."""
import os
import socket
import threading
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtCore import QCoreApplication, Qt

import main as app
from emg_v4_core import encode_usb_frame
from test_emg_v4_core import make_v2_packet


class FakeFirmware(threading.Thread):
    """Waits for USBSTART, then streams COBS frames; records every command it receives."""
    def __init__(self, n_packets=30, drop=()):
        super().__init__(daemon=True)
        self.server = socket.socket(); self.server.bind(("127.0.0.1", 0)); self.server.listen(1)
        self.port = self.server.getsockname()[1]
        self.commands, self.n_packets, self.drop = [], n_packets, set(drop)

    def run(self):
        conn, _ = self.server.accept(); conn.settimeout(0.05)
        buf, streaming, seq = b"", False, 0
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            try:
                data = conn.recv(256)
                if not data:
                    break
                buf += data
            except socket.timeout:
                pass
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self.commands.append(line.decode())
                if line == b"USBSTART":
                    streaming = True
                if line == b"USBSTOP":
                    return
            if streaming and seq < self.n_packets:
                if seq not in self.drop:
                    conn.sendall(encode_usb_frame(make_v2_packet(sequence=seq, first_frame=5 * seq)))
                seq += 1
                time.sleep(0.002)


class WiredTransportTests(unittest.TestCase):
    def test_binary_session_sends_commands_and_delivers_all_channels(self):
        QCoreApplication.instance() or QCoreApplication([])
        fw = FakeFirmware(); fw.start()
        got = []
        tracer = app.PacketTracer()
        tracer.reset("wireless", 5)
        worker = app.SerialWorker(f"socket://127.0.0.1:{fw.port}", app.DEFAULT_BAUD_RATE, 11, tracer=tracer, binary=True)
        worker.batch_received.connect(got.append, Qt.DirectConnection)
        worker.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and sum(len(p["batch"]) for p in got) < 150:
            time.sleep(0.02)
        worker.stop(); fw.join(3)
        self.assertEqual(sum(len(p["batch"]) for p in got), 150)
        self.assertEqual(got[0]["batch"].shape[1], 11)              # EMG + IMU, not limited to 2-9
        self.assertEqual(fw.commands[0], "USBSTART")
        self.assertEqual(fw.commands[-1], "USBSTOP")
        ids = np.concatenate([p["frame_ids"] for p in got])
        self.assertTrue(np.array_equal(ids, np.arange(150)))
        self.assertGreater(int(np.concatenate([p["emg_timestamps"] for p in got]).max()), 2 ** 32)


if __name__ == "__main__":
    unittest.main()
