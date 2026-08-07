"""
robot_state_io.py

General-purpose shared state file used to pass sensor/robot readings
between other processes (vision, temperature sensor, ESP32 telemetry
bridge, etc.) and the voice process. Same file-based-handoff pattern as
mood_state_io.py, generalized to hold many named fields instead of just
mood, since the project now needs temperature, activity, and alert state
too.

Expected fields (writers TBD as those subsystems get built):
    temperature_c        -> float, from the MLX90614 IR sensor
    activity              -> str, from MediaPipe pose classification
                              ("standing" | "sitting" | "walking" | "fallen")
    fall_alert_active     -> bool, set True when a fall is detected,
                              cleared by the "I'm okay" voice command
    panic_alert_active    -> bool, set True by the "help me" voice command
    battery_pct           -> float, from ESP32 telemetry

Any process can write any field; readers just ask for the field they
need. Missing or stale fields return a default so callers degrade
gracefully instead of crashing.

Later, when the full multiprocessing supervisor exists, swap the
implementation of these functions to use a multiprocessing.Manager()
dict instead — callers don't need to change.
"""

import json
import os
import tempfile
import time

ROBOT_STATE_PATH = os.environ.get("ROBOT_STATE_PATH", "/tmp/robot_state.json")


def _load():
    try:
        with open(ROBOT_STATE_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return {}


def _save(data):
    directory = os.path.dirname(ROBOT_STATE_PATH) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, ROBOT_STATE_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def write_field(key, value):
    """Write a single field. Safe to call from any process."""
    data = _load()
    data[key] = {"value": value, "timestamp": time.time()}
    _save(data)


def write_fields(fields):
    """Write several fields atomically in one go (fewer file writes)."""
    data = _load()
    now = time.time()
    for key, value in fields.items():
        data[key] = {"value": value, "timestamp": now}
    _save(data)


def read_field(key, max_age_seconds=30, default=None):
    """
    Read a single field. Returns `default` if the field has never been
    written, or if the most recent write is older than max_age_seconds
    (treated as stale — e.g. the sensor process isn't running).
    """
    data = _load()
    entry = data.get(key)
    if not isinstance(entry, dict):
        return default
    timestamp = entry.get("timestamp")
    if not isinstance(timestamp, (int, float)):
        return default
    if time.time() - timestamp > max_age_seconds:
        return default
    return entry.get("value", default)
