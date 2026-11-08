import unittest

import numpy as np

import channel_quality as cq

FS, N = 500, 1500
T = np.arange(N) / FS


class ChannelQualityTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def good(self):
        return 2048 + self.rng.normal(0, 6, N), 2048 + self.rng.normal(0, 120, N)

    def rate(self, rest, flex):
        return cq.assess_channel(rest, flex, FS, "Ch1")

    def test_good_channel(self):
        r = self.rate(*self.good())
        self.assertEqual(r["grade"], cq.GOOD)
        self.assertGreater(r["snr_db"], 20)

    def test_no_reaction_to_flex_is_poor_contact(self):
        rest, _ = self.good()
        r = self.rate(rest, 2048 + self.rng.normal(0, 7, N))
        self.assertEqual(r["grade"], cq.POOR)
        self.assertEqual(r["issues"][0][0], "weak_flex")

    def test_mains_hum_is_flagged(self):
        hum = 60 * np.sin(2 * np.pi * 50 * T)
        r = self.rate(2048 + self.rng.normal(0, 3, N) + hum, 2048 + self.rng.normal(0, 90, N) + hum)
        self.assertGreater(r["hum_pct"], 80)
        self.assertIn("hum", [c for c, _ in r["issues"]])
        self.assertNotEqual(r["grade"], cq.GOOD)

    def test_60hz_hum_is_flagged_too(self):
        hum = 60 * np.sin(2 * np.pi * 60 * T)
        r = self.rate(2048 + self.rng.normal(0, 3, N) + hum, 2048 + self.rng.normal(0, 90, N))
        self.assertGreater(r["hum_pct"], 80)

    def test_flat_and_railed_channels_have_no_signal(self):
        flat = self.rate(np.full(N, 2048.0), np.full(N, 2048.0))
        rail = self.rate(np.full(N, 4095.0) + self.rng.normal(0, 0.1, N), np.full(N, 4095.0))
        self.assertEqual((flat["grade"], rail["grade"]), (cq.NO_SIGNAL, cq.NO_SIGNAL))
        self.assertEqual(rail["issues"][0][0], "rail")

    def test_report_blocks_readiness_only_for_bad_channels(self):
        chans = [self.good() for _ in range(3)]
        rest = np.stack([c[0] for c in chans], 1); flex = np.stack([c[1] for c in chans], 1)
        ok = cq.assess_channels(rest, flex, FS)
        self.assertTrue(ok["ready"]); self.assertEqual(ok["counts"]["GOOD"], 3)
        rest[:, 1] = 2048.0; flex[:, 1] = 2048.0
        bad = cq.assess_channels(rest, flex, FS, ["A", "B", "C"])
        self.assertFalse(bad["ready"]); self.assertIn("B", bad["message"])
        meta = cq.summary_for_metadata(bad)
        self.assertEqual(meta["channels"][1]["grade"], cq.NO_SIGNAL)

    def test_missing_flex_capture_still_rates_the_rest_signal(self):
        r = cq.assess_channels(self.good()[0][:, None], None, FS)
        self.assertEqual(len(r["channels"]), 1)


if __name__ == "__main__":
    unittest.main()
