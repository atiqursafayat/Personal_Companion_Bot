"""
robot_commands.py

Local voice commands covering the safety/health features from the
project proposal (fall override, panic alert, temperature check,
activity status). These register into the SAME command registry as
voice_commands.py (time/weather) — importing this module runs the
@register_command decorators below, which is enough to activate them.

These commands read sensor data via robot_state_io, which other
subsystems (MLX90614 driver, MediaPipe pose classifier, fall detector)
are expected to write to as they get built. Until those exist, reads
simply return "no recent reading" — safe, no crash.
"""

import re

from alert_logic import set_alert_state
from event_log_io import log_event
from personal_memory import remember_value
from robot_state_io import read_field
from voice_commands import normalize_text, register_command

IM_OKAY_PHRASES = (
    "i'm okay",
    "i am okay",
    "i'm fine",
    "i am fine",
    "cancel alert",
    "cancel the alert",
)
HELP_PHRASES = ("help me", "call for help", "emergency", "i need help")
TEMPERATURE_PHRASES = (
    "what's my temperature",
    "what is my temperature",
    "check my temperature",
    "how's my temperature",
)
STATUS_PHRASES = (
    "what's my status",
    "what is my status",
    "how long have i been sitting",
    "what am i doing",
    "what's my activity",
    "what is my activity",
)

FEVER_THRESHOLD_C = 37.8


# ---------------------------------------------------------------------------
# "I'm okay" — cancels a fall alert (doc section 4: Manual Fall Override)
# ---------------------------------------------------------------------------

@register_command("im_okay_override")
def handle_im_okay(text):
    normalized = normalize_text(text)
    if not any(phrase in normalized for phrase in IM_OKAY_PHRASES):
        return None

    fall_was_active = read_field(
        "fall_alert_active", max_age_seconds=3600, default=False
    )
    set_alert_state(
        "fall_alert_active",
        False,
        active_message="Possible fall detected.",
        clear_message="Fall alert cleared after user said they are okay.",
        metadata={"source": "voice_command", "command": "im_okay_override"},
    )

    if fall_was_active:
        return "Okay, I've cancelled the fall alert. Glad you're alright."
    return "Good to hear you're okay."


# ---------------------------------------------------------------------------
# "Help me" — manual panic trigger, voice fallback for the physical button
# ---------------------------------------------------------------------------

@register_command("emergency_help")
def handle_help(text):
    normalized = normalize_text(text)
    if not any(phrase in normalized for phrase in HELP_PHRASES):
        return None

    set_alert_state(
        "panic_alert_active",
        True,
        active_message="Emergency assistance requested by voice command.",
        clear_message="Emergency assistance request cleared.",
        metadata={"source": "voice_command", "command": "emergency_help"},
    )
    return "Sending an alert to your caregiver now. Help is on the way."


# ---------------------------------------------------------------------------
# Temperature check (MLX90614 reading)
# ---------------------------------------------------------------------------

@register_command("temperature_check")
def handle_temperature(text):
    normalized = normalize_text(text)
    if not any(phrase in normalized for phrase in TEMPERATURE_PHRASES):
        return None

    temperature_c = read_field("temperature_c", max_age_seconds=300)
    if temperature_c is None:
        return "I don't have a recent temperature reading yet."

    reply = f"Your temperature reads {temperature_c:.1f} degrees Celsius."
    if temperature_c >= FEVER_THRESHOLD_C:
        log_event(
            kind="temperature_c",
            severity="warning",
            message=f"High temperature reading detected: {temperature_c:.1f} degrees Celsius.",
            metadata={"source": "voice_command", "temperature_c": temperature_c},
        )
        reply += " That's a bit high — you might want to rest."
    return reply


# ---------------------------------------------------------------------------
# Activity/status check (MediaPipe pose classification)
# ---------------------------------------------------------------------------

@register_command("activity_status")
def handle_status(text):
    normalized = normalize_text(text)
    if not any(phrase in normalized for phrase in STATUS_PHRASES):
        return None

    activity = read_field("activity", max_age_seconds=60)
    if activity is None:
        return "I can't tell what you're doing right now."
    return f"Right now, I can see you're {activity.lower()}."


@register_command("remember_preference")
def handle_remember_preference(text):
    normalized = normalize_text(text)

    match = re.search(r"\bcall me\s+([a-z][a-z\s'-]{0,30})$", normalized)
    if match:
        preferred_name = match.group(1).strip().title()
        remember_value("preferred_name", preferred_name)
        return f"Okay, I'll call you {preferred_name}."

    match = re.search(
        r"\bmy favorite music is\s+([a-z0-9][a-z0-9\s&,'-]{0,50})$",
        normalized,
    )
    if match:
        music = match.group(1).strip().title()
        remember_value("favorite_music", music, multi=True)
        return f"I'll remember that you like {music}."

    match = re.search(
        r"\b(?:thinking about|talking about)\s+([a-z0-9][a-z0-9\s&,'-]{0,50})\s+helps me feel better$",
        normalized,
    )
    if match:
        topic = match.group(1).strip().title()
        remember_value("comfort_topics", topic, multi=True)
        return f"Thanks for telling me. I'll remember that {topic} helps you feel better."

    match = re.search(
        r"\b([a-z][a-z\s'-]{0,30})\s+is important to me$",
        normalized,
    )
    if match:
        person = match.group(1).strip().title()
        remember_value("favorite_people", person, multi=True)
        return f"I'll remember that {person} matters to you."

    match = re.search(
        r"\bremember that\s+([a-z0-9][a-z0-9\s,.'-]{0,80})$",
        normalized,
    )
    if match:
        note = match.group(1).strip()
        remember_value("notes", note, multi=True)
        return "Okay, I'll remember that."

    return None
