import time
import unittest

from study_meta import (CONSENT_FORM_VERSION, default_hand_order, normalize_participant_id, participant_number,
                        session_folder_name, valid_participant_id)


class StudyMetaTests(unittest.TestCase):
    def test_participant_id_rule(self):
        for ok in ("S01", "S99", "S100", "s07", " S12 "):
            self.assertTrue(valid_participant_id(ok), ok)
        for bad in ("", "S1", "S1000", "Alice", "P01", "S0a", "S01_x", "Ibna Jiam", "01"):
            self.assertFalse(valid_participant_id(bad), bad)
        self.assertEqual(normalize_participant_id(" s07 "), "S07")

    def test_hand_order_alternates_by_participant_number(self):
        self.assertEqual([default_hand_order(p) for p in ("S01", "S02", "S03", "S10", "S101")],
                         ["left_right", "right_left", "left_right", "right_left", "left_right"])
        self.assertEqual(participant_number("S007"), 7)
        with self.assertRaises(ValueError):
            default_hand_order("bob")

    def test_session_folder_name_has_no_free_text(self):
        t = time.struct_time((2026, 11, 3, 1, 26, 8, 0, 0, 0))
        self.assertEqual(session_folder_name("s03", 2, "right_left", t), "S03_day02_right-left_20261103T012608Z")
        self.assertEqual(session_folder_name("S03", 1, "left", t, practice=True),
                         "practice_S03_day01_left_20261103T012608Z")
        with self.assertRaises(ValueError):
            session_folder_name("Fiza", 1, "left_right", t)

    def test_consent_version_is_a_string(self):
        self.assertIsInstance(CONSENT_FORM_VERSION, str)
        self.assertTrue(CONSENT_FORM_VERSION)


if __name__ == "__main__":
    unittest.main()


from study_meta import CHANNEL_MAP, build_metadata_json, parse_info_line


class MetadataJsonTests(unittest.TestCase):
    def test_parse_info_line(self):
        self.assertEqual(parse_info_line("INFO|ABC|BioWave|1|0|emg-imu-2.0.0|500"),
                         {"device_id": "ABC", "device_name": "BioWave", "imu_ready": True,
                          "firmware_version": "emg-imu-2.0.0", "sample_rate_hz": 500.0})
        self.assertNotIn("sample_rate_hz", parse_info_line("INFO|ABC|BioWave|0|0|emg-imu-1.6.0"))
        self.assertEqual(parse_info_line("HELLO|x"), {})

    def test_channel_map_matches_signal_columns(self):
        from session_recording import SIGNAL_COLUMNS
        self.assertEqual([c["column"] for c in CHANNEL_MAP], [c for c in SIGNAL_COLUMNS if c.startswith(("Ch", "IMU_roll", "IMU_pitch", "IMU_yaw"))])
        self.assertEqual([c["type"] for c in CHANNEL_MAP].count("emg"), 8)

    def test_missing_values_become_null_not_zero(self):
        meta = build_metadata_json(session="S01_x", label_map=[{"name": "rest", "code": 0}], phase_labels={"prepare": 99},
                                   device={"firmware_version": "", "sampling_rate_hz": 500},
                                   electrode_placement={"landmark": None, "distance_cm": 0})
        self.assertIsNone(meta["device"]["firmware_version"]); self.assertEqual(meta["device"]["sampling_rate_hz"], 500)
        self.assertEqual(meta["electrode_placement"]["distance_cm"], 0)
        self.assertIsNone(meta["electrode_placement"]["landmark"])
        self.assertEqual(meta["label_map"]["labels"][0]["code"], 0)
