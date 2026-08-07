"""Persistent timers and reminders for the companion assistant."""

import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timedelta


DATA_DIRECTORY = os.environ.get(
    "COMPANION_DATA_DIRECTORY", os.path.join(os.path.dirname(__file__), ".companion_data")
)
REMINDERS_PATH = os.path.join(DATA_DIRECTORY, "reminders.json")


def _load():
    try:
        with open(REMINDERS_PATH, "r") as reminder_file:
            data = json.load(reminder_file)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _save(items):
    os.makedirs(DATA_DIRECTORY, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(dir=DATA_DIRECTORY)
    try:
        with os.fdopen(fd, "w") as reminder_file:
            json.dump(items, reminder_file, indent=2)
        os.replace(temporary_path, REMINDERS_PATH)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def add_item(message, due_at, kind="reminder"):
    items = _load()
    item = {
        "id": uuid.uuid4().hex,
        "kind": kind,
        "message": message,
        "due_at": due_at.timestamp(),
        "created_at": time.time(),
    }
    items.append(item)
    _save(items)
    return item


def pop_due_items(now=None):
    now_timestamp = (now or datetime.now().astimezone()).timestamp()
    items = _load()
    due = [item for item in items if item.get("due_at", float("inf")) <= now_timestamp]
    if due:
        due_ids = {item.get("id") for item in due}
        _save([item for item in items if item.get("id") not in due_ids])
    return sorted(due, key=lambda item: item.get("due_at", 0))


def list_items():
    return sorted(_load(), key=lambda item: item.get("due_at", 0))


def cancel_all():
    count = len(_load())
    _save([])
    return count


def next_clock_time(hour, minute, now=None):
    now = now or datetime.now().astimezone()
    due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if due <= now:
        due += timedelta(days=1)
    return due
