"""
mood_state_io.py

Minimal cross-process handoff for the current detected mood, using a JSON
file as a poor-man's shared state. This is intentionally simple so the
vision script and the voice script can be developed/run independently.

Later, when you build the full multiprocessing supervisor, swap the
implementation of read_mood()/write_mood() to use a
multiprocessing.Manager().dict() instead — nothing else needs to change,
since both scripts only ever call these two functions.
"""

import json
import os
import tempfile
import time

MOOD_STATE_PATH = os.environ.get("MOOD_STATE_PATH", "/tmp/robot_mood_state.json")
MOOD_HISTORY_PATH = os.environ.get("MOOD_HISTORY_PATH", "/tmp/robot_mood_history.json")
MAX_MOOD_HISTORY = 180


def _atomic_write_json(path, payload):
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _read_history():
    try:
        with open(MOOD_HISTORY_PATH, "r") as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return []
    return payload if isinstance(payload, list) else []


def write_mood(mood, confidence=None):
    """Called by the vision process whenever it has a fresh mood reading."""
    payload = {
        "mood": mood,
        "confidence": confidence,
        "timestamp": time.time(),
    }
    _atomic_write_json(MOOD_STATE_PATH, payload)

    history = _read_history()
    history.append(payload)
    _atomic_write_json(MOOD_HISTORY_PATH, history[-MAX_MOOD_HISTORY:])


def read_mood(max_age_seconds=10):
    """
    Called by the voice process to get the latest mood reading.
    Returns (mood: str | None, confidence: float | None).
    Returns (None, None) if there's no reading yet, or it's stale
    (vision process not running / person not in frame recently).
    """
    try:
        with open(MOOD_STATE_PATH, "r") as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return None, None

    if not isinstance(payload, dict):
        return None, None
    timestamp = payload.get("timestamp")
    if not isinstance(timestamp, (int, float)):
        return None, None
    age = time.time() - timestamp
    if age > max_age_seconds:
        return None, None

    return payload.get("mood"), payload.get("confidence")


def read_mood_history(limit=30, max_age_seconds=1800):
    cutoff = time.time() - max_age_seconds
    history = []
    for entry in _read_history():
        if not isinstance(entry, dict):
            continue
        timestamp = entry.get("timestamp")
        if not isinstance(timestamp, (int, float)) or timestamp < cutoff:
            continue
        history.append(
            {
                "mood": entry.get("mood"),
                "confidence": entry.get("confidence"),
                "timestamp": timestamp,
            }
        )
    return history[-limit:]
