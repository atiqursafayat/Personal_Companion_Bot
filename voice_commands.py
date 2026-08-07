"""
voice_commands.py

Registry of "local" voice commands — questions the robot can answer directly
without calling the LLM (time, weather, etc.).

HOW TO ADD A NEW COMMAND
------------------------
1. Write a function that takes the raw heard text and returns either:
     - a string reply (if this command applies to the text), or
     - None (if it doesn't apply, so the next command gets a chance)
2. Decorate it with @register_command("some_name").
3. That's it — voice_assistant.py doesn't need to change at all.

Example:

    @register_command("joke")
    def handle_joke(text):
        if "tell me a joke" in normalize_text(text):
            return "Why did the robot go on a diet? Too many byte-sized snacks."
        return None
"""

import ast
import json
import operator
import os
import random
import re
import time
from datetime import datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import urlopen

from device_controls import (
    bluetooth_status,
    change_volume,
    change_brightness,
    control_media,
    delete_latest_picture,
    get_battery_text,
    get_brightness_text,
    get_cpu_temperature_text,
    get_storage_text,
    get_wifi_status,
    get_volume_text,
    lock_screen,
    open_application,
    open_website,
    power_action,
    request_picture,
    set_brightness,
    set_bluetooth_power,
    set_muted,
    set_volume,
    set_wifi_power,
    take_screenshot,
)
from mood_state_io import read_mood
from reminders import add_item, cancel_all, list_items, next_clock_time

WEATHER_LOCATION = os.environ.get("WEATHER_LOCATION", "Dhaka").strip()

WEATHER_CODE_DESCRIPTIONS = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "light snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "rain showers",
    81: "heavy rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
}

# Ordered list of (name, handler) pairs. Order = priority when matching.
_COMMANDS = []
_PENDING_CONFIRMATION = None
CONFIRMATION_TIMEOUT_SECONDS = 20


def register_command(name):
    """Decorator: adds a handler function to the command registry."""

    def decorator(func):
        _COMMANDS.append((name, func))
        return func

    return decorator


def normalize_text(text):
    return " ".join(text.lower().strip().split())


def request_confirmation(description, action):
    global _PENDING_CONFIRMATION
    _PENDING_CONFIRMATION = {
        "description": description,
        "action": action,
        "expires_at": time.monotonic() + CONFIRMATION_TIMEOUT_SECONDS,
    }
    return f"Do you really want me to {description}? Please say yes or no."


def handle_pending_confirmation(text):
    global _PENDING_CONFIRMATION
    if not _PENDING_CONFIRMATION:
        return None
    if time.monotonic() > _PENDING_CONFIRMATION["expires_at"]:
        _PENDING_CONFIRMATION = None
        return "That confirmation expired, so I didn't do anything."

    normalized = normalize_text(text).strip(" ?!.,")
    if normalized in {"yes", "yes please", "confirm", "do it", "go ahead"}:
        pending = _PENDING_CONFIRMATION
        _PENDING_CONFIRMATION = None
        return pending["action"]()
    if normalized in {"no", "no thanks", "cancel", "never mind", "don't do it"}:
        _PENDING_CONFIRMATION = None
        return "Cancelled."
    return "Please say yes to confirm or no to cancel."


# ---------------------------------------------------------------------------
# Time command
# ---------------------------------------------------------------------------

def is_time_question(text):
    normalized = normalize_text(text)
    return bool(re.search(r"\b(current\s+)?time\b", normalized))


def get_current_time_text():
    now = datetime.now().astimezone()
    return now.strftime("The current time is %I:%M %p on %A, %B %d.").replace(
        " 0", " "
    )


@register_command("time")
def handle_time(text):
    if is_time_question(text):
        return get_current_time_text()
    return None


# ---------------------------------------------------------------------------
# Date and day command
# ---------------------------------------------------------------------------

@register_command("date")
def handle_date(text):
    normalized = normalize_text(text)
    date_phrases = (
        "what is the date",
        "what's the date",
        "what date is it",
        "what day is it",
        "what is today",
        "today's date",
    )
    if any(phrase in normalized for phrase in date_phrases):
        now = datetime.now().astimezone()
        return now.strftime("Today is %A, %B %d, %Y.").replace(" 0", " ")
    return None


# ---------------------------------------------------------------------------
# Conversation commands
# ---------------------------------------------------------------------------

@register_command("greeting")
def handle_greeting(text):
    normalized = normalize_text(text).strip(" ?!.,")
    if normalized in {"hello", "hi", "hey", "good morning", "good afternoon", "good evening"}:
        return "Hello! It's good to hear from you. How can I help?"
    return None


@register_command("identity")
def handle_identity(text):
    normalized = normalize_text(text)
    if any(
        phrase in normalized
        for phrase in ("who are you", "what are you", "what is your name", "what's your name")
    ):
        return "I'm your personal companion assistant. I can listen, answer questions, and respond to how you're feeling."
    return None


@register_command("capabilities")
def handle_capabilities(text):
    normalized = normalize_text(text)
    if any(
        phrase in normalized
        for phrase in ("what can you do", "how can you help", "show me your commands", "list your commands")
    ):
        return (
            "I can answer questions, manage timers and reminders, control volume, "
            "brightness, Bluetooth, Wi-Fi and media, take pictures or screenshots, "
            "open apps and websites, and report battery, temperature, and storage."
        )
    return None


@register_command("thanks")
def handle_thanks(text):
    normalized = normalize_text(text).strip(" ?!.,")
    if normalized in {"thanks", "thank you", "thank you very much", "thanks a lot"}:
        return "You're welcome. I'm always happy to help."
    return None


@register_command("joke")
def handle_joke(text):
    normalized = normalize_text(text)
    if "tell me a joke" not in normalized and normalized.strip(" ?!.,") != "joke":
        return None
    return random.choice(
        (
            "Why did the robot go on vacation? It needed to recharge.",
            "Why was the computer cold? It left its Windows open.",
            "I told my robot to take a break. It said it needed one byte first.",
        )
    )


@register_command("mood")
def handle_mood(text):
    normalized = normalize_text(text)
    if not any(
        phrase in normalized
        for phrase in ("how do i look", "how am i feeling", "what is my mood", "what's my mood")
    ):
        return None

    mood, confidence = read_mood()
    if not mood:
        return "I can't read your mood right now. Please face the camera and try again."

    reply = f"You seem {mood.lower()} right now"
    if confidence is not None:
        reply += f", with about {confidence:.0f} percent confidence"
    return reply + "."


# ---------------------------------------------------------------------------
# Device control commands
# ---------------------------------------------------------------------------

@register_command("picture")
def handle_picture(text):
    normalized = normalize_text(text)
    if any(
        phrase in normalized
        for phrase in ("take a picture", "take my picture", "take a photo", "take my photo")
    ):
        try:
            return request_picture()
        except OSError:
            return "I couldn't send the picture request to the camera."
    return None


@register_command("volume")
def handle_volume(text):
    normalized = normalize_text(text).strip(" ?!.,")

    if normalized in {"mute", "mute volume", "mute the volume", "mute system audio"}:
        return set_muted(True)
    if normalized in {"unmute", "unmute volume", "unmute the volume", "unmute system audio"}:
        return set_muted(False)
    if any(phrase in normalized for phrase in ("what is the volume", "what's the volume", "volume level")):
        return get_volume_text()

    level_match = re.search(
        r"(?:set|change|turn) (?:the )?(?:system )?volume (?:to )?(\d{1,3})(?: percent)?",
        normalized,
    )
    if level_match:
        requested_level = int(level_match.group(1))
        if requested_level > 100:
            return "Please choose a volume level between zero and one hundred percent."
        return set_volume(requested_level)

    if normalized in {"volume up", "turn up the volume", "increase the volume"}:
        return change_volume("up")
    if normalized in {"volume down", "turn down the volume", "decrease the volume"}:
        return change_volume("down")
    return None


@register_command("bluetooth")
def handle_bluetooth(text):
    normalized = normalize_text(text).strip(" ?!.,")
    if normalized in {
        "turn on bluetooth",
        "turn bluetooth on",
        "bluetooth on",
        "enable bluetooth",
    }:
        return set_bluetooth_power(True)
    if normalized in {
        "turn off bluetooth",
        "turn bluetooth off",
        "bluetooth off",
        "disable bluetooth",
    }:
        return request_confirmation(
            "turn Bluetooth off", lambda: set_bluetooth_power(False)
        )
    if normalized in {"bluetooth status", "is bluetooth on", "is bluetooth off"}:
        return bluetooth_status()
    return None


@register_command("brightness")
def handle_brightness(text):
    normalized = normalize_text(text).strip(" ?!.,")
    if any(
        phrase in normalized
        for phrase in ("what is the brightness", "what's the brightness", "brightness level")
    ):
        return get_brightness_text()
    level_match = re.search(
        r"(?:set|change|turn) (?:the )?(?:screen )?brightness (?:to )?(\d{1,3})(?: percent)?",
        normalized,
    )
    if level_match:
        level = int(level_match.group(1))
        if not 1 <= level <= 100:
            return "Please choose a brightness level between one and one hundred percent."
        return set_brightness(level)
    if normalized in {"brightness up", "turn up the brightness", "increase brightness"}:
        return change_brightness("up")
    if normalized in {"brightness down", "turn down the brightness", "decrease brightness"}:
        return change_brightness("down")
    return None


@register_command("battery")
def handle_battery(text):
    normalized = normalize_text(text)
    if any(phrase in normalized for phrase in ("battery level", "battery status", "how much battery")):
        return get_battery_text()
    return None


@register_command("wifi")
def handle_wifi(text):
    normalized = (
        normalize_text(text)
        .strip(" ?!.,")
        .replace("wi-fi", "wifi")
        .replace("wi fi", "wifi")
    )
    if normalized in {"wifi status", "is wifi on", "am i connected to wifi"}:
        return get_wifi_status()
    if normalized in {"turn on wifi", "turn wifi on", "enable wifi", "wifi on"}:
        return set_wifi_power(True)
    if normalized in {"turn off wifi", "turn wifi off", "disable wifi", "wifi off"}:
        return request_confirmation("turn Wi-Fi off", lambda: set_wifi_power(False))
    return None


@register_command("open")
def handle_open(text):
    normalized = normalize_text(text).strip(" ?!.,")
    website_match = re.fullmatch(r"open (?:the )?(?:website )?([a-z0-9.-]+(?: dot [a-z]+)?)", normalized)
    if website_match:
        destination = website_match.group(1).replace(" dot ", ".")
        app_names = {"browser", "calculator", "files", "file manager", "terminal", "text editor"}
        if destination in app_names:
            return open_application(destination)
        return open_website(destination)
    return None


@register_command("screenshot")
def handle_screenshot(text):
    normalized = normalize_text(text)
    if any(phrase in normalized for phrase in ("take a screenshot", "capture the screen", "take screenshot")):
        return take_screenshot()
    return None


@register_command("lock")
def handle_lock(text):
    normalized = normalize_text(text).strip(" ?!.,")
    if normalized in {"lock screen", "lock the screen", "lock my computer", "lock the computer"}:
        return request_confirmation("lock the screen", lock_screen)
    return None


@register_command("media")
def handle_media(text):
    normalized = normalize_text(text).strip(" ?!.,")
    if normalized in {"play", "pause", "play music", "pause music", "play pause"}:
        return control_media("play-pause")
    if normalized in {"next", "next track", "next song", "skip song"}:
        return control_media("next")
    if normalized in {"previous", "previous track", "previous song", "go back a song"}:
        return control_media("previous")
    return None


@register_command("system_status")
def handle_system_status(text):
    normalized = normalize_text(text)
    if any(phrase in normalized for phrase in ("cpu temperature", "system temperature", "computer temperature")):
        return get_cpu_temperature_text()
    if any(phrase in normalized for phrase in ("storage status", "disk space", "free storage", "storage space")):
        return get_storage_text()
    return None


@register_command("power")
def handle_power(text):
    normalized = normalize_text(text).strip(" ?!.,")
    if normalized in {"shut down", "shutdown", "shut down the computer", "turn off the computer"}:
        return request_confirmation("shut down the computer", lambda: power_action("poweroff"))
    if normalized in {"restart", "reboot", "restart the computer", "reboot the computer"}:
        return request_confirmation("restart the computer", lambda: power_action("reboot"))
    return None


@register_command("delete_picture")
def handle_delete_picture(text):
    normalized = normalize_text(text).strip(" ?!.,")
    if normalized in {"delete the latest picture", "delete my last picture", "delete the last photo"}:
        return request_confirmation("delete the latest picture", delete_latest_picture)
    return None


# ---------------------------------------------------------------------------
# Persistent timers and reminders
# ---------------------------------------------------------------------------

_SMALL_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "ninety": 90,
}


def _parse_number(value):
    value = value.strip().replace("-", " ")
    if value.isdigit():
        return int(value)
    total = 0
    for word in value.split():
        if word not in _SMALL_NUMBERS:
            return None
        total += _SMALL_NUMBERS[word]
    return total if total > 0 else None


def _parse_duration(value):
    match = re.fullmatch(r"(.+?)\s+(seconds?|minutes?|hours?)", value.strip())
    if not match:
        return None
    amount = _parse_number(match.group(1))
    if amount is None or amount <= 0:
        return None
    unit = match.group(2)
    seconds = amount
    if unit.startswith("minute"):
        seconds *= 60
    elif unit.startswith("hour"):
        seconds *= 3600
    return timedelta(seconds=seconds)


def _format_due_time(due):
    return due.strftime("%I:%M %p").lstrip("0")


@register_command("timer")
def handle_timer(text):
    normalized = normalize_text(text).strip(" ?!.,")
    match = re.fullmatch(r"set (?:a )?timer for (.+)", normalized)
    if not match:
        return None
    duration = _parse_duration(match.group(1))
    if not duration:
        return "Please give the timer duration in seconds, minutes, or hours."
    due = datetime.now().astimezone() + duration
    add_item("Your timer is finished.", due, kind="timer")
    return f"Timer set for {match.group(1)}."


@register_command("reminder")
def handle_reminder(text):
    normalized = normalize_text(text).strip(" ?!.,")
    if normalized in {"list reminders", "what are my reminders", "list my reminders"}:
        items = list_items()
        if not items:
            return "You have no pending reminders."
        descriptions = []
        for item in items[:5]:
            due = datetime.fromtimestamp(item["due_at"]).astimezone()
            descriptions.append(f"{item['message']} at {_format_due_time(due)}")
        return "Your reminders are: " + "; ".join(descriptions) + "."
    if normalized in {"cancel all reminders", "delete all reminders", "clear all reminders"}:
        return request_confirmation(
            "cancel all reminders",
            lambda: f"Cancelled {cancel_all()} reminders.",
        )

    match = re.fullmatch(r"remind me to (.+?) (in|at) (.+)", normalized)
    if not match:
        return None
    message, mode, when = match.groups()
    now = datetime.now().astimezone()
    if mode == "in":
        duration = _parse_duration(when)
        if not duration:
            return "Please give the reminder delay in seconds, minutes, or hours."
        due = now + duration
    else:
        clock_text = when.replace(".", "")
        clock_match = re.fullmatch(
            r"([a-z]+|\d{1,2})(?::(\d{2}))?\s*(am|pm)?", clock_text
        )
        if not clock_match:
            return "Please give a time such as 8 PM or 8:30 AM."
        hour = _parse_number(clock_match.group(1))
        minute = int(clock_match.group(2) or 0)
        meridiem = clock_match.group(3)
        if hour is None or minute > 59 or (meridiem and not 1 <= hour <= 12) or (not meridiem and hour > 23):
            return "That reminder time isn't valid."
        if meridiem:
            hour = hour % 12 + (12 if meridiem == "pm" else 0)
        due = next_clock_time(hour, minute, now)
    add_item(message, due, kind="reminder")
    return f"I'll remind you to {message} at {_format_due_time(due)}."


# ---------------------------------------------------------------------------
# Safe arithmetic command
# ---------------------------------------------------------------------------

_MATH_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate_math(node):
    if isinstance(node, ast.Expression):
        return _evaluate_math(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _MATH_OPERATORS:
        return _MATH_OPERATORS[type(node.op)](_evaluate_math(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _MATH_OPERATORS:
        left = _evaluate_math(node.left)
        right = _evaluate_math(node.right)
        if abs(left) > 1e12 or abs(right) > 1e12:
            raise ValueError("number is too large")
        return _MATH_OPERATORS[type(node.op)](left, right)
    raise ValueError("unsupported expression")


@register_command("calculator")
def handle_calculator(text):
    normalized = normalize_text(text).strip(" ?!.,")
    match = re.fullmatch(r"(?:what is|what's|calculate|compute)\s+(.+)", normalized)
    if not match:
        return None

    expression = match.group(1)
    replacements = (
        ("divided by", "/"),
        ("multiplied by", "*"),
        ("times", "*"),
        ("plus", "+"),
        ("minus", "-"),
        ("modulo", "%"),
        ("mod", "%"),
    )
    for phrase, symbol in replacements:
        expression = expression.replace(phrase, symbol)

    if not re.fullmatch(r"[0-9+\-*/%().\s]+", expression):
        return None

    try:
        result = _evaluate_math(ast.parse(expression, mode="eval"))
    except (ArithmeticError, SyntaxError, ValueError, OverflowError):
        return "I couldn't calculate that expression."

    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"The answer is {result}."


# ---------------------------------------------------------------------------
# Weather command
# ---------------------------------------------------------------------------

def is_weather_question(text):
    normalized = normalize_text(text)
    return "weather" in normalized or "temperature" in normalized


def extract_weather_location(text):
    match = re.search(
        r"(?:weather|temperature)(?:\s+like)?(?:\s+in|\s+for)?\s+(.+)$",
        normalize_text(text),
    )
    if match:
        location = match.group(1).strip(" ?.,!")
        if location:
            return location
    return WEATHER_LOCATION or None


def get_weather_description(weather_code):
    return WEATHER_CODE_DESCRIPTIONS.get(weather_code, "unknown conditions")


def fetch_json(url):
    try:
        with urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError):
        return None


def get_current_weather_text(location):
    if not location:
        return (
            "I need a location for weather. Set WEATHER_LOCATION or ask "
            "for weather in a specific city."
        )

    search_url = (
        "https://geocoding-api.open-meteo.com/v1/search"
        f"?name={quote_plus(location)}&count=1&language=en&format=json"
    )
    search_data = fetch_json(search_url)
    if not search_data:
        return f"I couldn't reach the weather service for {location}."

    results = search_data.get("results") or []
    if not results:
        return f"I couldn't find weather data for {location}."

    place = results[0]
    latitude = place["latitude"]
    longitude = place["longitude"]
    place_name = ", ".join(
        part
        for part in [place.get("name"), place.get("admin1"), place.get("country")]
        if part
    )

    forecast_url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        "&current=temperature_2m,wind_speed_10m,weather_code"
        "&temperature_unit=celsius&wind_speed_unit=kmh"
    )
    forecast_data = (fetch_json(forecast_url) or {}).get("current") or {}

    temperature = forecast_data.get("temperature_2m")
    wind_speed = forecast_data.get("wind_speed_10m")
    weather_code = forecast_data.get("weather_code")
    description = get_weather_description(weather_code)

    if temperature is None:
        return f"I couldn't read the current weather for {place_name}."

    if wind_speed is not None:
        return (
            f"Right now in {place_name}, it's {temperature:.1f} degrees Celsius "
            f"with {description}. Wind is {wind_speed:.1f} kilometers per hour."
        )
    return f"Right now in {place_name}, it's {temperature:.1f} degrees Celsius with {description}."


@register_command("weather")
def handle_weather(text):
    if is_weather_question(text):
        return get_current_weather_text(extract_weather_location(text))
    return None


# ---------------------------------------------------------------------------
# Dispatcher — this is the only thing voice_assistant.py needs to import
# ---------------------------------------------------------------------------

def answer_local_question(text):
    """
    Try every registered command in order. Return the first non-None reply,
    or None if no local command matched (caller should fall back to the LLM).
    """
    confirmation_reply = handle_pending_confirmation(text)
    if confirmation_reply is not None:
        return confirmation_reply

    for _name, handler in _COMMANDS:
        reply = handler(text)
        if reply:
            return reply
    return None
