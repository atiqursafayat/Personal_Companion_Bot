"""
assistant_tools.py

Utilities the voice assistant can use for everyday tasks like speaker
volume control and saving a snapshot from the live camera feed.
"""

import os
import re
import time

from event_log_io import log_event
from frame_io import read_frame
from personal_memory import read_memory
from robot_state_io import read_field, write_field

DEFAULT_ASSISTANT_VOLUME = 70
SNAPSHOT_DIR = os.environ.get(
    "SNAPSHOT_DIR", os.path.join(os.getcwd(), "snapshots")
)


def clamp_volume(value):
    return max(0, min(100, int(round(value))))


def get_assistant_volume():
    stored = read_field(
        "assistant_volume", max_age_seconds=86400 * 365, default=DEFAULT_ASSISTANT_VOLUME
    )
    try:
        return clamp_volume(stored)
    except (TypeError, ValueError):
        return DEFAULT_ASSISTANT_VOLUME


def set_assistant_volume(value):
    volume = clamp_volume(value)
    write_field("assistant_volume", volume)
    return volume


def change_assistant_volume(delta):
    return set_assistant_volume(get_assistant_volume() + delta)


def get_display_name():
    memory = read_memory()
    return memory.get("preferred_name") or "there"


def save_snapshot():
    frame = read_frame(max_age_seconds=5.0)
    if frame is None:
        return None

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    snapshot_path = os.path.join(SNAPSHOT_DIR, f"snapshot-{timestamp}.jpg")
    with open(snapshot_path, "wb") as f:
        f.write(frame)

    write_field("latest_snapshot_path", snapshot_path)
    log_event(
        kind="snapshot_saved",
        severity="info",
        message=f"Snapshot saved to {os.path.basename(snapshot_path)}.",
        metadata={"path": snapshot_path},
    )
    return snapshot_path


def parse_volume_request(text):
    normalized = text.lower()
    number_match = re.search(r"(\d{1,3})", normalized)
    if number_match:
        return clamp_volume(int(number_match.group(1)))
    if "maximum" in normalized or "max volume" in normalized:
        return 100
    if "minimum" in normalized:
        return 0
    return None
