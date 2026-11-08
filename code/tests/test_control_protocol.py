import hashlib
import hmac
import json
import os
import socket
import tempfile
import threading
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import main as app

KEY = "0123456789abcdef0123456789abcdef"


class FakeDevice(threading.Thread):
    """UDP device that ignores the first ``drop_challenges`` CHALLENGE requests."""
    def __init__(self, drop_challenges=0):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0)); self.sock.settimeout(0.05)
        self.port = self.sock.getsockname()[1]
        self.drop, self.issued, self.running = drop_challenges, [], True

    def run(self):
        pending = {}
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
            except socket.timeout:
                continue
            text = data.decode()
            if text == "CHALLENGE":
                if self.drop > 0:
                    self.drop -= 1
                    continue
                ch = os.urandom(8).hex(); pending[addr] = ch; self.issued.append(ch)
                self.sock.sendto(f"CHALLENGE|{ch}".encode(), addr)
                continue
            parts = text.split("|")
            ch = pending.pop(addr, None)
            expect = hmac.new(KEY.encode(), "|".join([parts[0], ch or "", *parts[2:-1]]).encode(), hashlib.sha256).hexdigest()
            ok = ch == parts[1] and hmac.compare_digest(expect, parts[-1])
            self.sock.sendto(b"ACK|PONG" if ok else b"ERR|AUTH_FAILED", addr)


class ControlProtocolTests(unittest.TestCase):
    def setUp(self):
        self._port = app.WIFI_CONTROL_PORT

    def tearDown(self):
        app.WIFI_CONTROL_PORT = self._port

    def run_with_device(self, drop):
        dev = FakeDevice(drop); dev.start(); app.WIFI_CONTROL_PORT = dev.port
        try:
            result = app.ControlProtocol.authenticated_command("127.0.0.1", KEY, "PING", timeout=0.15, backoff=(0.01, 0.01, 0.01))
        finally:
            dev.running = False
        return dev, result

    def test_retries_with_fresh_challenge_after_lost_datagrams(self):
        dev, result = self.run_with_device(drop=2)
        self.assertEqual(result, ["PONG"])
        self.assertEqual(len(dev.issued), 1)           # two requests lost, the third got through

    def test_gives_up_after_three_retries(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.run_with_device(drop=99)
        self.assertIn("4 attempts", str(ctx.exception))

    def test_placeholder_and_short_keys_are_refused_without_network(self):
        for key in ("", app.PLACEHOLDER_DEVICE_ACCESS_KEY, "short"):
            with self.assertRaises(RuntimeError):
                app.ControlProtocol.authenticated_command("127.0.0.1", key, "PING", timeout=0.05, backoff=())
            self.assertTrue(app.access_key_problem(key))
        self.assertEqual(app.access_key_problem(KEY), "")

    def test_load_device_access_key(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "device_secret.json")
            self.assertEqual(app.load_device_access_key(path), "")
            with open(path, "w") as fh:
                json.dump({"device_access_key": KEY}, fh)
            self.assertEqual(app.load_device_access_key(path), KEY)


if __name__ == "__main__":
    unittest.main()


class DiscoveryTests(unittest.TestCase):
    """Discovery must find a device whichever adapter it is reachable through (Mac and Windows)."""

    def setUp(self):
        self._port, self._timeout = app.WIFI_CONTROL_PORT, app.DISCOVERY_TIMEOUT
        app.DISCOVERY_TIMEOUT = 1.0

    def tearDown(self):
        app.WIFI_CONTROL_PORT, app.DISCOVERY_TIMEOUT = self._port, self._timeout

    def test_finds_a_device_that_answers_the_probe(self):
        dev = FakeDevice(); dev.running = True
        original_run = dev.run

        def hello_run():
            while dev.running:
                try:
                    data, addr = dev.sock.recvfrom(1024)
                except socket.timeout:
                    continue
                if data == b"DISCOVER":
                    dev.sock.sendto(b"HELLO|AABBCC|BioWave-EMG-IMU|STA|127.0.0.1|1|0|emg-imu-2.0.0|500", addr)
        dev.run = hello_run
        dev.start(); app.WIFI_CONTROL_PORT = dev.port
        try:
            found = app.ControlProtocol.discover(extra_targets=("127.0.0.1",))     # unicast probe: works without a LAN
        finally:
            dev.running = False
        self.assertEqual([(d.device_id, d.firmware, d.sample_rate_hz) for d in found], [("AABBCC", "emg-imu-2.0.0", 500.0)])

    def test_no_reply_returns_an_empty_list_within_the_timeout(self):
        app.WIFI_CONTROL_PORT = free_udp_port()
        t = time.monotonic()
        self.assertEqual(app.ControlProtocol.discover(), [])
        self.assertLess(time.monotonic() - t, 2.5)

    def test_local_addresses_exclude_loopback_and_link_local(self):
        for ip in app.ControlProtocol.local_ipv4_addresses():
            self.assertFalse(ip.startswith(("127.", "169.254.")))


def free_udp_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port
