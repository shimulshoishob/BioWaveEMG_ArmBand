import unittest

import numpy as np

from emg_v4_core import (BWIM_V1_HEADER, FRAME_DTYPE_V1, GapSegment, PacketSequencer, StreamAssembler,
                         parse_bwim_packet, segments_to_payload)
from test_emg_v4_core import make_v1_packet, make_v2_packet

FPP = 5


def pkt(seq, first_frame=None, ts0=1_000_000):
    first = FPP * seq if first_frame is None else first_frame
    return parse_bwim_packet(make_v2_packet(sequence=seq & 0xFFFFFFFF, first_frame=first & 0xFFFFFFFF, ts0=ts0 + first * 2000))


def run(order, assembler=None, step=0.002, **kw):
    asm = assembler or StreamAssembler(**kw)
    segs, t = [], 0.0
    for seq in order:
        segs += asm.push(pkt(seq), t); t += step
    segs += asm.flush()
    return asm, segs


def frame_ids(segs):
    p = segments_to_payload(segs, 0.0, "t")
    return p["frame_ids"], p["valid"], p


class SequencingTests(unittest.TestCase):
    def test_no_loss(self):
        asm, segs = run(range(20))
        ids, valid, _ = frame_ids(segs)
        self.assertTrue(np.array_equal(ids, np.arange(100))); self.assertTrue(valid.all())
        self.assertEqual((asm.missing_packets, asm.sequencer.late, asm.sequencer.duplicates), (0, 0, 0))

    def test_single_loss_is_nan_with_correct_ids_and_sequences(self):
        asm, segs = run([0, 1, 2, 4, 5, 6])
        ids, valid, p = frame_ids(segs)
        self.assertTrue(np.array_equal(ids, np.arange(35)))
        self.assertTrue(np.array_equal(np.flatnonzero(valid == 0), np.arange(15, 20)))
        self.assertTrue(np.isnan(p["batch"][15:20]).all()); self.assertFalse(np.isnan(p["batch"][valid == 1]).any())
        self.assertTrue((p["packet_numbers"][15:20] == 3).all())
        self.assertTrue((p["emg_timestamps"][15:20] == -1).all())      # timestamps are never invented
        self.assertEqual((asm.missing_packets, asm.missing_frames, p["packet_gap"], p["gap_frames"]), (1, 5, 1, 5))

    def test_burst_loss(self):
        asm, segs = run([0, 1, 5, 6, 7])
        ids, valid, _ = frame_ids(segs)
        self.assertTrue(np.array_equal(ids, np.arange(40)))
        self.assertEqual(int((valid == 0).sum()), 15)
        self.assertEqual((asm.missing_packets, len(asm.gaps), asm.gaps[0].first_frame_id), (3, 1, 10))

    def test_reordering_within_window_is_released_in_order_without_loss(self):
        asm, segs = run([0, 2, 1, 3, 5, 4, 6])
        ids, valid, _ = frame_ids(segs)
        self.assertTrue(np.array_equal(ids, np.arange(35))); self.assertTrue(valid.all())
        self.assertEqual((asm.missing_packets, asm.sequencer.reordered), (0, 2))

    def test_late_packet_after_gap_declared_is_counted_not_inserted(self):
        asm = StreamAssembler(max_pending=2)
        segs = []
        for t, s in enumerate([0, 1, 3, 4, 5, 2, 6]):                   # 2 arrives after 3,4,5 forced a gap
            segs += asm.push(pkt(s), t * 0.002)
        segs += asm.flush()
        ids, valid, _ = frame_ids(segs)
        self.assertTrue(np.array_equal(ids, np.arange(35)))
        self.assertEqual(int((valid == 0).sum()), 5)
        self.assertEqual((asm.sequencer.late, asm.missing_packets), (1, 1))

    def test_timeout_declares_loss_without_more_packets(self):
        asm = StreamAssembler(max_pending=5, max_wait_s=0.05)
        segs = []
        for s, t in [(0, 0.0), (1, 0.002), (3, 0.004)]:
            segs += asm.push(pkt(s), t)
        self.assertEqual(len(segs), 2)                                   # packet 3 is held back
        segs += asm.poll(0.03); self.assertEqual(len(segs), 2)
        segs += asm.poll(0.06); self.assertEqual(asm.missing_packets, 1)
        self.assertEqual(len(frame_ids(segs)[0]), 20)

    def test_duplicates_are_dropped(self):
        asm, segs = run([0, 1, 1, 2, 2, 0, 3])
        ids, valid, _ = frame_ids(segs)
        self.assertTrue(np.array_equal(ids, np.arange(20))); self.assertTrue(valid.all())
        self.assertEqual(asm.sequencer.duplicates, 3)

    def test_sequence_wrap_uint32(self):
        base = 0xFFFFFFFD
        asm = StreamAssembler()
        segs = []
        for i, s in enumerate([base, base + 1, base + 2, base + 3, base + 5, base + 6]):   # loses 0 (after wrap)
            segs += asm.push(pkt(s, first_frame=5 * i if i < 4 else 5 * (i + 1)), i * 0.002)
        segs += asm.flush()
        self.assertEqual(asm.missing_packets, 1)
        ids, valid, p = frame_ids(segs)
        self.assertEqual(int((valid == 0).sum()), 5)
        self.assertEqual(int(p["packet_numbers"][20]), (base + 4) & 0xFFFFFFFF)   # the lost packet is identified across the wrap

    def test_frame_id_jump_without_sequence_gap_is_still_a_gap(self):
        # Wi-Fi outage: the device did not packetise, so sequence is continuous but frame IDs jump.
        asm = StreamAssembler()
        segs = asm.push(pkt(0), 0.0) + asm.push(pkt(1, first_frame=5000), 0.002) + asm.flush()
        ids, valid, p = frame_ids(segs)
        self.assertEqual(asm.missing_packets, 0)
        self.assertEqual(asm.missing_frames, 4995)
        self.assertEqual(int((valid == 0).sum()), 4995)
        self.assertEqual(ids[5], 5); self.assertEqual(ids[-1], 5004)

    def test_v1_timestamp_wrap_is_unwrapped_to_monotonic_int64(self):
        asm = StreamAssembler(); segs = []
        for i in range(3):
            frames = np.zeros(5, dtype=FRAME_DTYPE_V1)
            frames["frame_id"] = np.arange(5 * i, 5 * i + 5)
            frames["frame_ts"] = (2**32 - 10_000 + 2000 * np.arange(5 * i, 5 * i + 5)) & 0xFFFFFFFF
            raw = BWIM_V1_HEADER.pack(b"BWIM", 1, 5, 48, i) + frames.tobytes()
            segs += asm.push(parse_bwim_packet(raw), i * 0.01)
        ts = segments_to_payload(segs, 0.0, "t")["emg_timestamps"]
        self.assertTrue(np.all(np.diff(ts) == 2000)); self.assertGreater(int(ts[-1]), 2**32)

    def test_device_restart_does_not_swallow_the_new_stream(self):
        asm = StreamAssembler(); segs = []
        for i in range(2000):
            segs += asm.push(pkt(5000 + i), i * 0.002)
        for i in range(10):                                              # device rebooted: sequence and frames restart
            segs += asm.push(pkt(i), 5.0 + i * 0.002)
        segs += asm.flush()
        self.assertEqual(asm.sequencer.restarts, 1)
        self.assertEqual(frame_ids(segs)[2]["restart"], True)
        self.assertEqual(sum(1 for s in segs if not isinstance(s, (tuple, GapSegment))), 2010)


if __name__ == "__main__":
    unittest.main()
