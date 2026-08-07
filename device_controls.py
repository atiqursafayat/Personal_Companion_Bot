"""Small, local Linux device-control helpers used by voice commands."""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


PICTURE_REQUEST_PATH = os.environ.get(
    "PICTURE_REQUEST_PATH", "/tmp/robot_picture_request.json"
)
CAPTURE_DIRECTORY = os.environ.get(
    "CAPTURE_DIRECTORY", os.path.join(os.path.dirname(__file__), "captures")
)


def _run(command):
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _audio_command(*arguments):
    if shutil.which("wpctl"):
        return _run(["wpctl", *arguments])
    return None


def get_volume_text():
    result = _audio_command("get-volume", "@DEFAULT_AUDIO_SINK@")
    if result and result.returncode == 0:
        match = re.search(r"Volume:\s+([0-9.]+)", result.stdout)
        if match:
            level = round(float(match.group(1)) * 100)
            muted = "MUTED" in result.stdout.upper()
            return f"System volume is {level} percent" + (" and muted." if muted else ".")
    return "I couldn't read the system volume. This control requires PipeWire wpctl."


def set_volume(level):
    level = max(0, min(100, int(level)))
    result = _audio_command("set-volume", "@DEFAULT_AUDIO_SINK@", f"{level}%")
    if result and result.returncode == 0:
        return f"System volume is now {level} percent."
    return "I couldn't change the system volume. This control requires PipeWire wpctl."


def change_volume(direction, amount=10):
    suffix = "+" if direction == "up" else "-"
    result = _audio_command(
        "set-volume", "-l", "1.0", "@DEFAULT_AUDIO_SINK@", f"{amount}%{suffix}"
    )
    if result and result.returncode == 0:
        return f"Volume turned {direction}."
    return "I couldn't change the system volume. This control requires PipeWire wpctl."


def set_muted(muted):
    value = "1" if muted else "0"
    result = _audio_command("set-mute", "@DEFAULT_AUDIO_SINK@", value)
    if result and result.returncode == 0:
        return "System audio muted." if muted else "System audio unmuted."
    return "I couldn't change mute. This control requires PipeWire wpctl."


def bluetooth_status():
    result = _run(["bluetoothctl", "show"])
    if not result or result.returncode != 0:
        return "I couldn't read Bluetooth status."
    powered = re.search(r"Powered:\s+(yes|no)", result.stdout, re.IGNORECASE)
    if not powered:
        return "I couldn't read Bluetooth status."
    return f"Bluetooth is {'on' if powered.group(1).lower() == 'yes' else 'off'}."


def set_bluetooth_power(enabled):
    state = "on" if enabled else "off"
    result = _run(["bluetoothctl", "power", state])
    if result and result.returncode == 0 and "failed" not in result.stdout.lower():
        return f"Bluetooth turned {state}."
    return f"I couldn't turn Bluetooth {state}. Check Bluetooth permissions and service status."


def get_brightness_text():
    result = _run(["brightnessctl", "-m"])
    if result and result.returncode == 0:
        fields = result.stdout.strip().split(",")
        if len(fields) >= 4:
            return f"Screen brightness is {fields[3].strip()}."
    return "I couldn't read screen brightness. This control requires brightnessctl."


def set_brightness(level):
    level = max(1, min(100, int(level)))
    result = _run(["brightnessctl", "set", f"{level}%"])
    if result and result.returncode == 0:
        return f"Screen brightness is now {level} percent."
    return "I couldn't change screen brightness. This control requires brightnessctl permission."


def change_brightness(direction, amount=10):
    value = f"{amount}%+" if direction == "up" else f"{amount}%-"
    result = _run(["brightnessctl", "set", value])
    if result and result.returncode == 0:
        return f"Brightness turned {direction}."
    return "I couldn't change screen brightness. This control requires brightnessctl permission."


def get_battery_text():
    batteries = sorted(Path("/sys/class/power_supply").glob("BAT*"))
    if not batteries:
        return "I couldn't find a system battery."
    try:
        level = (batteries[0] / "capacity").read_text().strip()
        status = (batteries[0] / "status").read_text().strip().lower()
        return f"Battery is at {level} percent and is {status}."
    except OSError:
        return "I couldn't read the battery status."


def get_wifi_status():
    result = _run(["nmcli", "-t", "-f", "WIFI", "general"])
    if result and result.returncode == 0:
        enabled = result.stdout.strip().lower() == "enabled"
        if not enabled:
            return "Wi-Fi is off."
        connection = _run(["nmcli", "-t", "-f", "ACTIVE,SSID", "device", "wifi"])
        if connection and connection.returncode == 0:
            for line in connection.stdout.splitlines():
                if line.startswith("yes:"):
                    return f"Wi-Fi is on and connected to {line.split(':', 1)[1]}."
        return "Wi-Fi is on but not connected."
    return "I couldn't read Wi-Fi status. This control requires NetworkManager nmcli."


def set_wifi_power(enabled):
    state = "on" if enabled else "off"
    result = _run(["nmcli", "radio", "wifi", state])
    if result and result.returncode == 0:
        return f"Wi-Fi turned {state}."
    return f"I couldn't turn Wi-Fi {state}. This control requires NetworkManager permission."


APPLICATIONS = {
    "browser": ("xdg-open", "https://www.google.com"),
    "calculator": ("gnome-calculator",),
    "files": ("xdg-open", os.path.expanduser("~")),
    "file manager": ("xdg-open", os.path.expanduser("~")),
    "terminal": ("x-terminal-emulator",),
    "text editor": ("gedit",),
}


def open_application(name):
    command = APPLICATIONS.get(name.lower())
    if not command:
        return "I can open the browser, calculator, files, terminal, or text editor."
    if not shutil.which(command[0]):
        return f"I couldn't find the {name} application."
    try:
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"Opening {name}."
    except OSError:
        return f"I couldn't open {name}."


def open_website(address):
    address = address.strip().lower().replace(" ", "")
    if not address:
        return "Please tell me which website to open."
    if "." not in address:
        address += ".com"
    if not address.startswith(("http://", "https://")):
        address = "https://" + address
    parsed = urlparse(address)
    if parsed.scheme != "https" or not parsed.hostname:
        return "I can only open a valid secure website address."
    result = _run(["xdg-open", address])
    if result and result.returncode == 0:
        return f"Opening {parsed.hostname}."
    return "I couldn't open that website."


def take_screenshot():
    os.makedirs(CAPTURE_DIRECTORY, exist_ok=True)
    filename = datetime.now().strftime("screenshot_%Y%m%d_%H%M%S.png")
    destination = os.path.join(CAPTURE_DIRECTORY, filename)
    candidates = (
        ("gnome-screenshot", "-f", destination),
        ("spectacle", "-b", "-n", "-o", destination),
        ("scrot", destination),
        ("grim", destination),
    )
    for command in candidates:
        if not shutil.which(command[0]):
            continue
        result = _run(list(command))
        if result and result.returncode == 0 and os.path.exists(destination):
            return f"Screenshot saved as {filename}."
    return "I couldn't take a screenshot. Install a supported screenshot tool."


def lock_screen():
    commands = (("loginctl", "lock-session"), ("xdg-screensaver", "lock"))
    for command in commands:
        if shutil.which(command[0]):
            result = _run(list(command))
            if result and result.returncode == 0:
                return "Locking the screen."
    return "I couldn't lock the screen."


def control_media(action):
    if action not in {"play-pause", "next", "previous"}:
        return "Unsupported media command."
    result = _run(["playerctl", action])
    if result and result.returncode == 0:
        labels = {"play-pause": "Toggling playback.", "next": "Playing the next track.", "previous": "Playing the previous track."}
        return labels[action]
    return "I couldn't control media playback. This control requires playerctl."


def get_cpu_temperature_text():
    for temperature_file in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            temperature = float(temperature_file.read_text().strip())
            if temperature > 1000:
                temperature /= 1000
            if 0 < temperature < 150:
                return f"CPU temperature is about {temperature:.1f} degrees Celsius."
        except (OSError, ValueError):
            continue
    return "I couldn't read the CPU temperature."


def get_storage_text():
    usage = shutil.disk_usage(os.path.dirname(__file__))
    gibibyte = 1024 ** 3
    free = usage.free / gibibyte
    total = usage.total / gibibyte
    used_percent = round(usage.used / usage.total * 100)
    return f"Storage is {used_percent} percent used, with {free:.1f} of {total:.1f} gigabytes free."


def power_action(action):
    if action not in {"poweroff", "reboot"}:
        return "Unsupported power action."
    result = _run(["systemctl", action])
    if result and result.returncode == 0:
        return "Shutting down." if action == "poweroff" else "Restarting now."
    return "I couldn't complete that power command. Check system permissions."


def delete_latest_picture():
    files = sorted(
        Path(CAPTURE_DIRECTORY).glob("picture_*.jpg"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not files:
        return "There are no saved pictures to delete."
    try:
        files[0].unlink()
        return f"Deleted {files[0].name}."
    except OSError:
        return "I couldn't delete the latest picture."


def request_picture():
    """Ask the running mood tracker to save its next camera frame."""
    directory = os.path.dirname(PICTURE_REQUEST_PATH) or "."
    payload = {"requested_at": time.time()}
    fd, temporary_path = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "w") as request_file:
            json.dump(payload, request_file)
        os.replace(temporary_path, PICTURE_REQUEST_PATH)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise
    return "Taking a picture now."


def save_requested_picture(frame, cv2_module):
    """Save a pending picture request using the mood tracker's current frame."""
    try:
        with open(PICTURE_REQUEST_PATH, "r") as request_file:
            request = json.load(request_file)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    if time.time() - request.get("requested_at", 0) > 30:
        os.remove(PICTURE_REQUEST_PATH)
        return None

    os.makedirs(CAPTURE_DIRECTORY, exist_ok=True)
    filename = datetime.now().strftime("picture_%Y%m%d_%H%M%S.jpg")
    destination = os.path.join(CAPTURE_DIRECTORY, filename)
    if not cv2_module.imwrite(destination, frame):
        return None

    os.remove(PICTURE_REQUEST_PATH)
    return destination
