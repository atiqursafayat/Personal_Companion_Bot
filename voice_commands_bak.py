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

import json
import os
import re
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import urlopen

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


def register_command(name):
    """Decorator: adds a handler function to the command registry."""

    def decorator(func):
        _COMMANDS.append((name, func))
        return func

    return decorator


def normalize_text(text):
    return " ".join(text.lower().strip().split())


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
    for _name, handler in _COMMANDS:
        reply = handler(text)
        if reply:
            return reply
    return None
