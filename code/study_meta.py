"""Pseudonymous IDs, session naming and counterbalancing rules for the study (no Qt)."""
from __future__ import annotations

import re
import time

PARTICIPANT_ID_RE = re.compile(r"^S\d{2,3}$")
CONSENT_FORM_VERSION = "BWEMG-consent-v1.0"      # bump whenever the consent / procedure text changes


def normalize_participant_id(text: str) -> str:
    return str(text or "").strip().upper()


def valid_participant_id(text: str) -> bool:
    return bool(PARTICIPANT_ID_RE.match(normalize_participant_id(text)))


def participant_number(pid: str) -> int:
    if not valid_participant_id(pid):
        raise ValueError(f"Participant ID must look like S01 or S101, got {pid!r}")
    return int(normalize_participant_id(pid)[1:])


def default_hand_order(pid: str) -> str:
    """Counterbalancing: odd participants start with the left hand, even participants with the right."""
    return "left_right" if participant_number(pid) % 2 == 1 else "right_left"


def session_folder_name(pid: str, day: int, hand_order: str, utc: time.struct_time | None = None,
                        practice: bool = False) -> str:
    """``<ID>_day<NN>_<hand-order>_<UTC timestamp>`` (``practice_`` prefix keeps demo runs apart)."""
    pid = normalize_participant_id(pid)
    if not valid_participant_id(pid):
        raise ValueError(f"Participant ID must look like S01 or S101, got {pid!r}")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", utc or time.gmtime())
    name = f"{pid}_day{int(day):02d}_{str(hand_order).replace('_', '-')}_{stamp}"
    return f"practice_{name}" if practice else name


def find_existing_sessions(save_dir: str, pid: str, day: int, hand: str) -> list:
    """Folders in ``save_dir`` that already hold a (non-practice) session of this participant, day and hand.

    Recording the same participant / day / hand twice by accident is the usual way a dataset gets a wrong
    hand label, so the app asks before it allows it.
    """
    import os
    pid = normalize_participant_id(pid)
    prefix = f"{pid}_day{int(day):02d}_{str(hand).replace('_', '-')}_"
    try:
        names = os.listdir(save_dir)
    except OSError:
        return []
    return sorted(n for n in names if n.startswith(prefix) and os.path.isdir(os.path.join(save_dir, n)))


CHANNEL_MAP = ([{"channel": i, "column": f"Ch{i}", "name": f"EMG{i}", "type": "emg",
                 "unit": "ADC counts (12-bit, raw, 0-4095)"} for i in range(1, 9)]
               + [{"channel": 9, "column": "IMU_roll", "name": "roll", "type": "imu_orientation", "unit": "degrees"},
                  {"channel": 10, "column": "IMU_pitch", "name": "pitch", "type": "imu_orientation", "unit": "degrees"},
                  {"channel": 11, "column": "IMU_yaw", "name": "yaw", "type": "imu_orientation", "unit": "degrees"}])


def parse_info_line(line: str) -> dict:
    """``INFO|id|name|imu|wifi|fw[|rate]`` (USB) -> dict; ``{}`` when it is not an INFO line."""
    parts = str(line or "").strip().split("|")
    if len(parts) < 6 or parts[0] != "INFO":
        return {}
    info = {"device_id": parts[1], "device_name": parts[2], "imu_ready": parts[3] == "1", "firmware_version": parts[5]}
    if len(parts) > 6:
        try:
            info["sample_rate_hz"] = float(parts[6])
        except ValueError:
            pass
    return info


def build_metadata_json(*, session: str, label_map: dict, phase_labels: dict, **sections) -> dict:
    """Assemble the machine-readable session metadata.

    ``sections`` are the already-collected groups (software, device, protocol, environment, participant,
    electrode_placement, timing, files, ...). Anything missing stays absent rather than being guessed;
    empty values are written as null so "not recorded" is distinguishable from "zero".
    """
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if value in ("", None):
            return None
        return value

    meta = {"schema_version": 1, "session": session, "signal_mode": "raw",
            "channel_map": CHANNEL_MAP,
            "label_map": {"labels": label_map, "phases": phase_labels}}
    meta.update(sections)
    return clean(meta)
