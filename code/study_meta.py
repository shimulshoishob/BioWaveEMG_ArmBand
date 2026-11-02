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
