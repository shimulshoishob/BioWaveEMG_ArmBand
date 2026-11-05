"""Every window must fit and be fully usable on laptop screens (Mac and Windows)."""
import os
import subprocess
import sys
import unittest

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class UiAuditTests(unittest.TestCase):
    def run_profile(self, profile):
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        proc = subprocess.run([sys.executable, os.path.join(CODE_DIR, "tools", "ui_audit.py"), "--profile", profile],
                              cwd=CODE_DIR, env=env, capture_output=True, text=True, timeout=240)
        report = "\n".join(l for l in proc.stdout.splitlines() if "ISSUES" in l or l.startswith("      ") or "[error]" in l)
        self.assertEqual(proc.returncode, 0, f"UI audit found problems on {profile}:\n{report}\n{proc.stderr[-400:]}")

    def test_13_inch_macbook(self):
        self.run_profile("mac13")

    def test_windows_laptop_125_percent(self):
        self.run_profile("win125")

    def test_small_laptops(self):
        self.run_profile("win150")
        self.run_profile("win_hd")


if __name__ == "__main__":
    unittest.main()
