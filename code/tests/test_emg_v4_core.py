import unittest
import numpy as np

from emg_v4_core import (CalibrationProfile, GestureDecisionEngine, WirelessStats,
                         assess_signal_quality, compute_calibration, expected_feature_count,
                         validate_model_artifact)
from realtime_pipeline import SampleRingBuffer
from rf_features import extract_window_features_legacy, extract_window_features_optimized


class FakeModel:
    classes_ = np.array([0, 1])
    n_features_in_ = 70
    def predict(self, x): return np.zeros(len(x))


class V4CoreTests(unittest.TestCase):
    def test_packet_gap_and_wrap(self):
        stats = WirelessStats()
        self.assertEqual(stats.observe(100, 1.0), 0)
        self.assertEqual(stats.observe(101, 1.01), 0)
        self.assertEqual(stats.observe(103, 1.02), 1)
        self.assertEqual(stats.packets_missing, 1)

    def test_calibration_and_bad_channel(self):
        rng = np.random.default_rng(4)
        rest = rng.normal(2000, 3, (300, 2)); flex = rest.copy(); flex[:, 0] += rng.normal(0, 50, 300)
        profile = compute_calibration(rest, flex)
        self.assertEqual(profile.quality[0], "GOOD")
        self.assertIn(profile.quality[1], {"DEAD", "WEAK"})

    def test_signal_quality_rejects_nonfinite(self):
        profile = CalibrationProfile(np.zeros(2), np.ones(2), np.ones(2), np.ones(2)*5, np.ones(2)*5, np.ones(2)*5, ["GOOD"]*2, True, [])
        self.assertEqual(assess_signal_quality(np.array([[np.nan, 1]], dtype=np.float32), profile).state, "SIGNAL_POOR")

    def test_feature_count_and_compatibility(self):
        self.assertEqual(expected_feature_count(4), 70)
        artifact = {"model": FakeModel(), "class_names": ["REST", "UP"], "sample_rate": 500, "input_channels": 4}
        self.assertTrue(validate_model_artifact(artifact, 500, 4).compatible)
        self.assertFalse(validate_model_artifact(artifact, 250, 4).compatible)

    def test_temporal_and_release_gating(self):
        d = GestureDecisionEngine(vote_windows=3, consecutive_required=2, min_margin=.1)
        self.assertFalse(d.update("UP", .9, .2, True, 1).action_allowed)
        self.assertTrue(d.update("UP", .9, .2, True, 2).action_allowed)
        self.assertFalse(d.update("REST", .9, .2, True, 3).action_allowed)
        self.assertTrue(d.update("UP", .9, .2, True, 4).action_allowed is False)

    def test_ring_wrap_and_discontinuity_recovery(self):
        ring = SampleRingBuffer(2, 5)
        ring.append(np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float32))
        ring.append(np.array([[4, 40], [5, 50], [6, 60]], dtype=np.float32))
        np.testing.assert_array_equal(ring.latest(5), np.array([[2, 20], [3, 30], [4, 40], [5, 50], [6, 60]], dtype=np.float32))
        ring.append(np.array([[7, 70]], dtype=np.float32), discontinuity=True)
        self.assertFalse(ring.has_window(5))
        ring.append(np.array([[8, 80], [9, 90], [10, 100], [11, 110]], dtype=np.float32))
        self.assertTrue(ring.has_window(5))

    def test_optimized_feature_equivalence(self):
        rng = np.random.default_rng(123)
        window = rng.normal(size=(100, 4)).astype(np.float32)
        np.testing.assert_allclose(extract_window_features_optimized(window, 500),
                                   extract_window_features_legacy(window, 500), rtol=2e-5, atol=2e-4)


if __name__ == "__main__":
    unittest.main()


import struct as _struct

from emg_v4_core import (BWIM_V1_HEADER, BWIM_V2_HEADER, FRAME_DTYPE_V1, FRAME_DTYPE_V2, TimestampUnwrapper,
                         UsbFrameDecoder, cobs_decode, cobs_encode, crc16_ccitt, encode_usb_frame,
                         parse_bwim_packet)


def make_v2_packet(sequence=7, first_frame=100, count=5, ts0=10_000_000_000, counters=(1, 2, 3, 4)):
    frames = np.zeros(count, dtype=FRAME_DTYPE_V2)
    frames["frame_id"] = np.arange(first_frame, first_frame + count)
    frames["frame_ts"] = ts0 + 2000 * np.arange(count, dtype=np.uint64)
    frames["emg"] = np.arange(count * 8).reshape(count, 8) % 4096
    frames["imu"] = 1.5
    frames["flags"][1] = 1
    header = BWIM_V2_HEADER.pack(b"BWIM", 2, count, FRAME_DTYPE_V2.itemsize, sequence, *counters, 0)
    return header + frames.tobytes()


def make_v1_packet(sequence=3, first_frame=0, count=5):
    frames = np.zeros(count, dtype=FRAME_DTYPE_V1)
    frames["frame_id"] = np.arange(first_frame, first_frame + count)
    frames["frame_ts"] = 4_000_000_000 + 2000 * np.arange(count)
    frames["emg"] = 2048
    return BWIM_V1_HEADER.pack(b"BWIM", 1, count, FRAME_DTYPE_V1.itemsize, sequence) + frames.tobytes()


class BwimCodecTests(unittest.TestCase):
    def test_parse_v2_fields_and_counters(self):
        pkt = parse_bwim_packet(make_v2_packet())
        self.assertEqual((pkt.version, pkt.sequence, len(pkt.frames)), (2, 7, 5))
        self.assertEqual(pkt.device_counters, {"queue_drops": 1, "acq_overruns": 2, "send_failures": 3, "wifi_reconnects": 4})
        self.assertEqual(int(pkt.frames["frame_ts"][0]), 10_000_000_000)   # beyond 2**32: no truncation
        self.assertEqual(int(pkt.frames["flags"][1]), 1)

    def test_parse_v1_still_works(self):
        pkt = parse_bwim_packet(make_v1_packet())
        self.assertEqual((pkt.version, pkt.sequence, pkt.device_counters), (1, 3, {}))
        self.assertEqual(int(pkt.frames["emg"][0, 0]), 2048)
        self.assertEqual(int(pkt.frames["frame_ts"][4]), 4_000_008_000)

    def test_rejects_malformed(self):
        good = make_v2_packet()
        for bad in (good[:-1], good + b"x", b"XXXX" + good[4:], good[:4] + bytes([9]) + good[5:], b"", good[:11]):
            self.assertIsNone(parse_bwim_packet(bad))

    def test_cobs_roundtrip_including_zeros_and_long_runs(self):
        for data in (b"", b"\x00", b"\x00\x00", bytes(range(256)) * 3, b"\x01" * 600, b"a\x00b\x00\x00c"):
            enc = cobs_encode(data)
            self.assertNotIn(0, enc)
            self.assertEqual(cobs_decode(enc), data)

    def test_cobs_decode_rejects_garbage(self):
        self.assertIsNone(cobs_decode(b"\x05ab"))
        self.assertIsNone(cobs_decode(b"\x00"))

    def test_crc_known_vector(self):
        self.assertEqual(crc16_ccitt(b"123456789"), 0x29B1)   # CRC-16/CCITT-FALSE check value

    def test_usb_decoder_split_chunks_corruption_and_text(self):
        frames = [encode_usb_frame(make_v2_packet(sequence=i, first_frame=5 * i)) for i in range(4)]
        corrupt = bytearray(frames[1]); corrupt[20] ^= 0xFF
        stream = frames[0] + bytes(corrupt) + b"INFO|abc|dev|1|0|fw|500\r\n\x00" + frames[2] + frames[3]
        dec = UsbFrameDecoder()
        packets, bad = [], 0
        for i in range(0, len(stream), 37):            # arbitrary chunking
            p, b = dec.feed(stream[i:i + 37]); packets += p; bad += b
        self.assertEqual([p.sequence for p in packets], [0, 2, 3])
        self.assertEqual(bad, 1)                       # the corrupt frame; text line is ignored

    def test_timestamp_unwrapper_across_32bit_wrap(self):
        un = TimestampUnwrapper()
        a = un.unwrap([2**32 - 4000, 2**32 - 2000])
        b = un.unwrap([0, 2000, 4000])
        out = np.concatenate([a, b])
        self.assertTrue(np.all(np.diff(out) == 2000))
