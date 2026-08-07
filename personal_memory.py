"""
personal_memory.py

Small file-backed memory store for user preferences and comforting details.
The goal is not broad long-term memory; it is to keep a few high-value facts
that can make support responses feel more personal.
"""

import json
import os
import tempfile
import time

MEMORY_PATH = os.environ.get("PERSONAL_MEMORY_PATH", "/tmp/personal_memory.json")

DEFAULT_MEMORY = {
    "preferred_name": None,
    "favorite_music": [],
    "favorite_people": [],
    "comfort_topics": [],
    "notes": [],
    "updated_at": None,
}


def _load_memory():
    try:
        with open(MEMORY_PATH, "r") as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        payload = {}

    memory = dict(DEFAULT_MEMORY)
    if isinstance(payload, dict):
        memory.update(payload)
    return memory


def _save_memory(memory):
    memory = dict(DEFAULT_MEMORY, **memory)
    memory["updated_at"] = time.time()
    directory = os.path.dirname(MEMORY_PATH) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(memory, f)
        os.replace(tmp_path, MEMORY_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def read_memory():
    return _load_memory()


def remember_value(key, value, multi=False):
    memory = _load_memory()
    if multi:
        items = list(memory.get(key) or [])
        if value not in items:
            items.append(value)
        memory[key] = items[-5:]
    else:
        memory[key] = value
    _save_memory(memory)
    return memory


def describe_memory(memory=None):
    memory = memory or _load_memory()
    parts = []
    if memory.get("preferred_name"):
        parts.append(f"Preferred name: {memory['preferred_name']}.")
    if memory.get("favorite_music"):
        parts.append(
            "Favorite music: " + ", ".join(memory["favorite_music"][:3]) + "."
        )
    if memory.get("favorite_people"):
        parts.append(
            "Important people: " + ", ".join(memory["favorite_people"][:3]) + "."
        )
    if memory.get("comfort_topics"):
        parts.append(
            "Comfort topics: " + ", ".join(memory["comfort_topics"][:3]) + "."
        )
    if memory.get("notes"):
        parts.append("Notes: " + "; ".join(memory["notes"][:2]) + ".")
    return " ".join(parts).strip()
