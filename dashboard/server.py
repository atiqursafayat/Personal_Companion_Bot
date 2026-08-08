"""
dashboard/server.py

Local web dashboard for the Personal Companion Bot.

Serves a real-time status page showing the live camera feed (MJPEG),
current mood, system status, reminders, and recent captures.

Endpoints
---------
GET /              — Dashboard HTML page
GET /feed          — MJPEG camera stream (reads LATEST_FRAME_PATH)
GET /events        — SSE stream: mood + reminders + system status (1 Hz)
GET /api/captures  — JSON list of saved captures
GET /captures/{f}  — Serve a capture file
GET /static/{path} — Static assets

Run standalone (for development):
    uvicorn dashboard.server:app --host 0.0.0.0 --port 8080 --reload
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import aiofiles
from fastapi import FastAPI
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Paths (all configurable via environment variables)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LATEST_FRAME_PATH = Path(
    os.environ.get("LATEST_FRAME_PATH", "/tmp/robot_latest_frame.jpg")
)
MOOD_STATE_PATH = Path(
    os.environ.get("MOOD_STATE_PATH", "/tmp/robot_mood_state.json")
)
CAPTURE_DIRECTORY = Path(
    os.environ.get("CAPTURE_DIRECTORY", PROJECT_ROOT / "captures")
)
REMINDERS_PATH = Path(
    os.environ.get(
        "COMPANION_DATA_DIRECTORY",
        str(PROJECT_ROOT / ".companion_data"),
    )
) / "reminders.json"

DASHBOARD_HOST = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "8080"))

STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Companion Bot Dashboard", docs_url=None, redoc_url=None)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json_file(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _get_mood_data() -> dict:
    payload = _read_json_file(MOOD_STATE_PATH)
    if not isinstance(payload, dict):
        return {"mood": None, "confidence": None, "age_seconds": None, "online": False}

    age = time.time() - payload.get("timestamp", 0)
    return {
        "mood": payload.get("mood"),
        "confidence": payload.get("confidence"),
        "age_seconds": round(age, 1),
        "online": age < 10,
    }


def _get_reminders() -> list:
    data = _read_json_file(REMINDERS_PATH)
    if not isinstance(data, list):
        return []
    now = time.time()
    pending = [r for r in data if r.get("due_at", 0) > now]
    return sorted(pending, key=lambda r: r.get("due_at", 0))[:10]


def _run(command: list) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None


def _get_system_status() -> dict:
    status: dict = {}

    # Battery
    batteries = sorted(Path("/sys/class/power_supply").glob("BAT*"))
    if batteries:
        try:
            status["battery_percent"] = int(
                (batteries[0] / "capacity").read_text().strip()
            )
            status["battery_status"] = (batteries[0] / "status").read_text().strip().lower()
        except OSError:
            pass

    # CPU temperature
    for temp_file in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            raw = float(temp_file.read_text().strip())
            temp_c = raw / 1000 if raw > 1000 else raw
            if 0 < temp_c < 150:
                status["cpu_temp_celsius"] = round(temp_c, 1)
                break
        except (OSError, ValueError):
            continue

    # Disk
    try:
        usage = shutil.disk_usage(PROJECT_ROOT)
        gib = 1024 ** 3
        status["disk_used_percent"] = round(usage.used / usage.total * 100)
        status["disk_free_gb"] = round(usage.free / gib, 1)
        status["disk_total_gb"] = round(usage.total / gib, 1)
    except OSError:
        pass

    # Wi-Fi
    result = _run(["nmcli", "-t", "-f", "WIFI", "general"])
    if result and result.returncode == 0:
        wifi_enabled = result.stdout.strip().lower() == "enabled"
        status["wifi_enabled"] = wifi_enabled
        if wifi_enabled:
            conn = _run(["nmcli", "-t", "-f", "ACTIVE,SSID", "device", "wifi"])
            if conn and conn.returncode == 0:
                for line in conn.stdout.splitlines():
                    if line.startswith("yes:"):
                        status["wifi_ssid"] = line.split(":", 1)[1]
                        break

    return status


def _list_captures() -> list:
    if not CAPTURE_DIRECTORY.is_dir():
        return []
    files = []
    for path in sorted(
        CAPTURE_DIRECTORY.iterdir(),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:20]:
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            stat = path.stat()
            files.append(
                {
                    "name": path.name,
                    "size_bytes": stat.st_size,
                    "captured_at": stat.st_mtime,
                }
            )
    return files


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    async with aiofiles.open(html_path, "r") as f:
        return HTMLResponse(content=await f.read())


@app.get("/feed")
async def camera_feed():
    """
    MJPEG stream. Reads the latest annotated frame written by target_mood_fixed.py
    and rebroadcasts it at up to 15 fps. If no frame is available, sends a
    placeholder JPEG so the browser <img> tag doesn't break.
    """

    async def generate():
        while True:
            if LATEST_FRAME_PATH.exists():
                try:
                    async with aiofiles.open(LATEST_FRAME_PATH, "rb") as f:
                        frame_bytes = await f.read()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + frame_bytes
                        + b"\r\n"
                    )
                except OSError:
                    pass

            await asyncio.sleep(1 / 15)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/events")
async def sse_events():
    """
    Server-Sent Events: pushes a JSON payload every second containing
    mood, reminders, and system status.
    """

    async def generate():
        while True:
            payload = {
                "mood": _get_mood_data(),
                "reminders": _get_reminders(),
                "system": _get_system_status(),
                "captures_count": len(_list_captures()),
                "timestamp": time.time(),
            }
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/captures")
async def api_captures():
    return JSONResponse(_list_captures())


@app.get("/captures/{filename}")
async def serve_capture(filename: str):
    # Prevent path traversal: allow only safe filenames
    if not re.fullmatch(r"[\w\-. ]+\.(jpg|jpeg|png)", filename, re.IGNORECASE):
        return JSONResponse({"error": "Invalid filename."}, status_code=400)
    path = CAPTURE_DIRECTORY / filename
    if not path.is_file():
        return JSONResponse({"error": "Not found."}, status_code=404)
    return FileResponse(path)


# ---------------------------------------------------------------------------
# Entry point (used by main.py subprocess launcher)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # Ensure the project root is on sys.path so the dashboard package is
    # importable when this script is launched directly by main.py as a
    # subprocess (cwd = PROJECT_ROOT, but the package may not be on the path).
    _project_root = str(PROJECT_ROOT)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    print(
        f"[DASHBOARD] Starting on http://{DASHBOARD_HOST}:{DASHBOARD_PORT}",
        flush=True,
    )
    # Pass the app object directly rather than a module string so uvicorn
    # does not attempt a fresh import of 'dashboard.server' (which would
    # fail if sys.path is not yet updated at that point).
    uvicorn.run(
        app,
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        log_level="warning",
    )
