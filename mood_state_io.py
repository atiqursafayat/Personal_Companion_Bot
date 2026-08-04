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


def write_mood(mood, confidence=None):
    """Called by the vision process whenever it has a fresh mood reading."""
    payload = {
        "mood": mood,
        "confidence": confidence,
        "timestamp": time.time(),
    }
    # Write atomically so a reader never sees a half-written file.
    directory = os.path.dirname(MOOD_STATE_PATH) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, MOOD_STATE_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


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
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None

    age = time.time() - payload.get("timestamp", 0)
    if age > max_age_seconds:
        return None, None

    return payload.get("mood"), payload.get("confidence")
