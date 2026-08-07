"""
event_log_io.py

Tiny append-only event timeline for caregiver-facing incidents and
important system acknowledgements. This follows the same file-based
handoff style as robot_state_io.py so every process can contribute
timeline entries without a separate database.
"""

import json
import os
import tempfile
import time
import uuid

EVENT_LOG_PATH = os.environ.get("EVENT_LOG_PATH", "/tmp/robot_event_log.json")
MAX_EVENTS = 200


def _load_events():
    try:
        with open(EVENT_LOG_PATH, "r") as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return []

    return payload if isinstance(payload, list) else []


def _save_events(events):
    directory = os.path.dirname(EVENT_LOG_PATH) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(events[-MAX_EVENTS:], f)
        os.replace(tmp_path, EVENT_LOG_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def log_event(kind, message, severity="info", metadata=None):
    event = {
        "id": uuid.uuid4().hex,
        "kind": kind,
        "severity": severity,
        "message": message,
        "metadata": metadata or {},
        "timestamp": time.time(),
        "acknowledged": False,
        "acknowledged_at": None,
    }
    events = _load_events()
    events.append(event)
    _save_events(events)
    return event


def read_events(limit=25):
    events = _load_events()
    return list(reversed(events[-limit:]))


def acknowledge_event(event_id):
    events = _load_events()
    acknowledged_event = None
    now = time.time()
    for event in events:
        if event.get("id") != event_id:
            continue
        event["acknowledged"] = True
        event["acknowledged_at"] = now
        acknowledged_event = event
        break

    if acknowledged_event is None:
        return None

    _save_events(events)
    return acknowledged_event
