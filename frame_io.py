"""
frame_io.py

Hands the latest camera frame (already JPEG-encoded) from the vision
process to the dashboard process, without both processes trying to open
the physical camera at the same time. Most USB webcams -- the Logitech
C270 included -- don't support two independent OpenCV VideoCapture
handles open simultaneously, so only ONE process may own cv2.VideoCapture.
That owner is target_mood_tracker.py; dashboard_server.py just reads
whatever frame was last written here.

Same atomic-write-to-a-file pattern as mood_state_io.py / robot_state_io.py,
just storing raw JPEG bytes instead of JSON.
"""

import os
import tempfile
import time

FRAME_PATH = os.environ.get("FRAME_PATH", "/tmp/robot_latest_frame.jpg")
DEFAULT_MAX_FRAME_AGE_SECONDS = 2.0


def write_frame(jpeg_bytes):
    """Called by the vision process after encoding each annotated frame."""
    directory = os.path.dirname(FRAME_PATH) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(jpeg_bytes)
        os.replace(tmp_path, FRAME_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def read_frame(max_age_seconds=DEFAULT_MAX_FRAME_AGE_SECONDS):
    """
    Called by the dashboard process. Returns JPEG bytes, or None if there's
    no frame yet or the most recent one is stale (vision process isn't
    running / camera disconnected).
    """
    try:
        mtime = os.path.getmtime(FRAME_PATH)
    except OSError:
        return None

    if time.time() - mtime > max_age_seconds:
        return None

    try:
        with open(FRAME_PATH, "rb") as f:
            return f.read()
    except OSError:
        return None
