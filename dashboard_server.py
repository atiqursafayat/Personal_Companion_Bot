"""
dashboard_server.py

Lightweight local web dashboard for the companion robot.

Routes:
    "/"             -> the dashboard page (camera feed + live status panel)
    "/video_feed"   -> MJPEG stream, relaying frames published by the
                       vision process (target_mood_tracker.py)
    "/api/state"    -> JSON snapshot of everything currently known about
                       the robot (mood, temperature, activity, alerts, battery)

Run on the Pi with:
    python3 dashboard_server.py
Then open http://<pi-ip-address>:8080 from any browser on the same network.

IMPORTANT: this process does NOT open the camera itself. Only
target_mood_tracker.py (the vision process) owns cv2.VideoCapture --
most USB webcams, the C270 included, can't be opened by two independent
processes at once. This process just reads whatever frame the vision
process last published via frame_io.py, so the feed you see here already
has the tracking box and mood label drawn on it.

Design note: this is otherwise a READ-ONLY viewer. It doesn't own any
state itself -- it reads from the same shared-state files the rest of the
stack already writes to (mood_state_io.py, robot_state_io.py), the same
way voice_assistant.py does. Nothing here needs to change when a new
sensor writer gets built; add the field to get_full_state() and it shows up.
"""

import base64
import time

from flask import Flask, Response, jsonify, render_template, request

from event_log_io import acknowledge_event, log_event, read_events
from frame_io import read_frame
from mood_state_io import read_mood, read_mood_history
from robot_state_io import read_field
from reminder_engine import list_reminders, publish_reminder_state

# dashboard.html lives in the project root, so point Flask templates there.
app = Flask(__name__, template_folder=".")

STREAM_FPS_CAP = 15
FRAME_INTERVAL_SECONDS = 1 / STREAM_FPS_CAP
MOOD_SCORE = {
    "HAPPY": 2,
    "SURPRISED": 1,
    "NEUTRAL": 0,
    "DISGUSTED": -1,
    "FEARFUL": -1,
    "ANGRY": -2,
    "SAD": -2,
}

# Valid 1x1 JPEG fallback. Keeping it embedded means the dashboard API can
# start even on machines where the optional vision dependencies are absent.
_PLACEHOLDER_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPyF//9oADAMBAAIAAwAAABD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/EH//xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/EH//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/EH//2Q=="
)


def _generate_mjpeg():
    while True:
        frame = read_frame()
        if frame is None:
            frame = _PLACEHOLDER_JPEG

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(FRAME_INTERVAL_SECONDS)


@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def get_full_state():
    """
    Aggregate everything the dashboard displays into one dict. This is the
    single place that decides which fields exist and how stale a reading
    is allowed to be before it's treated as "unknown" -- keep this in sync
    as new sensors get added to the stack.
    """
    mood, mood_confidence = read_mood()
    mood_history = read_mood_history(limit=24, max_age_seconds=1800)
    reminders = publish_reminder_state()

    return {
        "timestamp": time.time(),
        "mood": {"value": mood, "confidence": mood_confidence},
        "mood_trends": summarize_mood_history(mood_history),
        "voice_status": read_field("voice_status", max_age_seconds=30),
        "assistant_output": read_field("assistant_output", max_age_seconds=300),
        "temperature_c": read_field("temperature_c", max_age_seconds=300),
        "activity": read_field("activity", max_age_seconds=60),
        "battery_pct": read_field("battery_pct", max_age_seconds=30),
        "fall_alert_active": read_field(
            "fall_alert_active", max_age_seconds=3600, default=False
        ),
        "panic_alert_active": read_field(
            "panic_alert_active", max_age_seconds=3600, default=False
        ),
        "mood_escalation_active": read_field(
            "mood_escalation_active", max_age_seconds=3600, default=False
        ),
        "mood_escalation_summary": read_field(
            "mood_escalation_summary", max_age_seconds=3600
        ),
        "upcoming_reminders": reminders,
        "events": read_events(limit=12),
    }


def summarize_mood_history(history):
    if not history:
        return {
            "points": [],
            "dominant": None,
            "summary": "No recent mood trend yet",
            "direction": "steady",
        }

    counts = {}
    points = []
    scores = []
    for entry in history:
        mood = entry.get("mood")
        if mood:
            counts[mood] = counts.get(mood, 0) + 1
        points.append(
            {
                "mood": mood,
                "confidence": entry.get("confidence"),
                "timestamp": entry.get("timestamp"),
            }
        )
        scores.append(MOOD_SCORE.get(mood, 0))

    dominant = max(counts.items(), key=lambda item: item[1])[0] if counts else None
    midpoint = max(1, len(scores) // 2)
    first_avg = sum(scores[:midpoint]) / len(scores[:midpoint])
    second_avg = sum(scores[midpoint:]) / len(scores[midpoint:])

    if second_avg - first_avg >= 0.5:
        direction = "lifting"
    elif second_avg - first_avg <= -0.5:
        direction = "heavier"
    else:
        direction = "steady"

    summary_map = {
        "lifting": "Mood seems to be improving",
        "heavier": "Mood seems more strained lately",
        "steady": "Mood has been fairly steady",
    }
    return {
        "points": points,
        "dominant": dominant,
        "summary": summary_map[direction],
        "direction": direction,
    }


@app.route("/api/state")
def api_state():
    response = jsonify(get_full_state())
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/api/health")
def api_health():
    """Small, dependency-free liveness endpoint for the UI and monitoring."""
    response = jsonify({"status": "ok", "timestamp": time.time()})
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/api/events/<event_id>/ack")
def api_acknowledge_event(event_id):
    event = acknowledge_event(event_id)
    if event is None:
        response = jsonify({"ok": False, "error": "event_not_found"})
        response.status_code = 404
        return response

    log_event(
        kind="event_acknowledged",
        severity="info",
        message=f"A timeline event was acknowledged: {event.get('message', 'Unknown event')}",
        metadata={"event_id": event_id, "source": "dashboard"},
    )
    response = jsonify({"ok": True, "event": event})
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/")
def index():
    return render_template("dashboard.html")


if __name__ == "__main__":
    # host="0.0.0.0" so it's reachable from other devices on the LAN,
    # not just localhost on the Pi itself.
    app.run(host="0.0.0.0", port=8080, threaded=True)
