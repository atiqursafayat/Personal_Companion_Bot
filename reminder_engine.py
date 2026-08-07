"""
reminder_engine.py

File-backed reminder storage plus a small scheduler for one-time and daily
care reminders such as medicine, water, and sleep prompts.
"""

import json
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timedelta

from event_log_io import log_event
from robot_state_io import write_field

REMINDER_PATH = os.environ.get("REMINDER_PATH", "/tmp/robot_reminders.json")


def _load():
    try:
        with open(REMINDER_PATH, "r") as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        payload = []
    return payload if isinstance(payload, list) else []


def _save(reminders):
    directory = os.path.dirname(REMINDER_PATH) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(reminders, f)
        os.replace(tmp_path, REMINDER_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _dt_to_ts(dt):
    return dt.astimezone().timestamp()


def _format_label(reminder):
    due_ts = reminder.get("next_due_at")
    if not isinstance(due_ts, (int, float)):
        return reminder.get("text", "Reminder")
    when = datetime.fromtimestamp(due_ts).astimezone()
    prefix = "Every day at" if reminder.get("schedule_type") == "daily" else "At"
    return f"{reminder.get('text', 'Reminder')} · {prefix} {when.strftime('%I:%M %p').lstrip('0')}"


def parse_reminder_request(text):
    normalized = " ".join(text.lower().strip().split())

    daily = re.search(
        r"remind me every day at (\d{1,2})(?::(\d{2}))?\s*(am|pm)\s+to\s+(.+)$",
        normalized,
    )
    if daily:
        hour = int(daily.group(1))
        minute = int(daily.group(2) or 0)
        meridiem = daily.group(3)
        reminder_text = daily.group(4).strip(" .")
        scheduled_dt = _next_daily_occurrence(hour, minute, meridiem)
        return {
            "schedule_type": "daily",
            "text": reminder_text,
            "next_due_at": _dt_to_ts(scheduled_dt),
        }

    one_time = re.search(
        r"remind me at (\d{1,2})(?::(\d{2}))?\s*(am|pm)\s+to\s+(.+)$",
        normalized,
    )
    if one_time:
        hour = int(one_time.group(1))
        minute = int(one_time.group(2) or 0)
        meridiem = one_time.group(3)
        reminder_text = one_time.group(4).strip(" .")
        scheduled_dt = _next_one_time_occurrence(hour, minute, meridiem)
        return {
            "schedule_type": "once",
            "text": reminder_text,
            "next_due_at": _dt_to_ts(scheduled_dt),
        }

    quick_map = (
        ("take my medicine", "medicine"),
        ("drink water", "water"),
        ("go to sleep", "sleep"),
        ("wake up", "wake"),
    )
    for phrase, category in quick_map:
        match = re.search(
            rf"{re.escape(phrase)}(?: every day)? at (\d{{1,2}})(?::(\d{{2}}))?\s*(am|pm)",
            normalized,
        )
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2) or 0)
            meridiem = match.group(3)
            daily_flag = "every day" in normalized
            scheduled_dt = (
                _next_daily_occurrence(hour, minute, meridiem)
                if daily_flag
                else _next_one_time_occurrence(hour, minute, meridiem)
            )
            return {
                "schedule_type": "daily" if daily_flag else "once",
                "text": phrase,
                "category": category,
                "next_due_at": _dt_to_ts(scheduled_dt),
            }

    return None


def _normalize_hour(hour, meridiem):
    if hour < 1 or hour > 12:
        raise ValueError("hour out of range")
    if meridiem == "am":
        return 0 if hour == 12 else hour
    return 12 if hour == 12 else hour + 12


def _next_one_time_occurrence(hour, minute, meridiem):
    now = datetime.now().astimezone()
    scheduled = now.replace(
        hour=_normalize_hour(hour, meridiem),
        minute=minute,
        second=0,
        microsecond=0,
    )
    if scheduled <= now:
        scheduled += timedelta(days=1)
    return scheduled


def _next_daily_occurrence(hour, minute, meridiem):
    return _next_one_time_occurrence(hour, minute, meridiem)


def create_reminder(data):
    reminders = _load()
    reminder = {
        "id": uuid.uuid4().hex[:8],
        "text": data["text"],
        "category": data.get("category", "general"),
        "schedule_type": data["schedule_type"],
        "created_at": time.time(),
        "next_due_at": data["next_due_at"],
        "last_triggered_at": None,
        "active": True,
    }
    reminders.append(reminder)
    _save(reminders)
    log_event(
        kind="reminder_created",
        severity="info",
        message=f"Reminder created: {_format_label(reminder)}.",
        metadata={"id": reminder["id"], "text": reminder["text"]},
    )
    return reminder


def list_reminders(active_only=True):
    reminders = _load()
    if active_only:
        reminders = [item for item in reminders if item.get("active", True)]
    return sorted(reminders, key=lambda item: item.get("next_due_at") or float("inf"))


def cancel_reminder(reminder_id=None, text_match=None):
    reminders = _load()
    cancelled = None
    for reminder in reminders:
        if not reminder.get("active", True):
            continue
        if reminder_id and reminder.get("id") == reminder_id:
            reminder["active"] = False
            cancelled = reminder
            break
        if text_match and text_match in reminder.get("text", "").lower():
            reminder["active"] = False
            cancelled = reminder
            break
    if cancelled:
        _save(reminders)
        log_event(
            kind="reminder_cancelled",
            severity="info",
            message=f"Reminder cancelled: {_format_label(cancelled)}.",
            metadata={"id": cancelled["id"], "text": cancelled["text"]},
        )
    return cancelled


def describe_reminders(limit=5):
    reminders = list_reminders(active_only=True)[:limit]
    if not reminders:
        return "You have no active reminders right now."
    parts = []
    for reminder in reminders:
        due = datetime.fromtimestamp(reminder["next_due_at"]).astimezone()
        label = due.strftime("%I:%M %p").lstrip("0")
        prefix = "every day at" if reminder["schedule_type"] == "daily" else "at"
        parts.append(f"{reminder['text']} {prefix} {label}")
    return "Your reminders are: " + "; ".join(parts) + "."


def get_due_reminders(now_ts=None):
    now_ts = now_ts or time.time()
    reminders = _load()
    due = []
    changed = False
    for reminder in reminders:
        if not reminder.get("active", True):
            continue
        next_due_at = reminder.get("next_due_at")
        if not isinstance(next_due_at, (int, float)) or next_due_at > now_ts:
            continue
        due.append(reminder.copy())
        reminder["last_triggered_at"] = now_ts
        if reminder.get("schedule_type") == "daily":
            reminder["next_due_at"] = next_due_at + 86400
        else:
            reminder["active"] = False
        changed = True

    if changed:
        _save(reminders)
    return due


def publish_reminder_state(limit=5):
    reminders = list_reminders(active_only=True)[:limit]
    payload = []
    for reminder in reminders:
        payload.append(
            {
                "id": reminder["id"],
                "text": reminder["text"],
                "schedule_type": reminder["schedule_type"],
                "next_due_at": reminder["next_due_at"],
                "category": reminder.get("category", "general"),
            }
        )
    write_field("upcoming_reminders", payload)
    return payload
