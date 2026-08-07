"""Raspberry Pi 5 target tracking and mood detection.

This keeps the Pi 5 implementation's NCNN/ByteTrack pipeline while matching
the current project's headless dashboard and mood-state integrations.
"""

import os
import time

# The Pi 5 runs both inference backends on its CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
from ultralytics import YOLO

from frame_io import write_frame
from mood_state_io import write_mood


MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "./yolov8n_ncnn_model")
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PERSON_CONFIDENCE = 0.4
EMOTION_INTERVAL = 5
MAX_MISSING_FRAMES = 30
JPEG_QUALITY = 80

# Position and apparent-distance thresholds tuned for a 640x480 C270 feed.
LEFT_BOUNDARY = 250
RIGHT_BOUNDARY = 390
TOO_CLOSE_HEIGHT = 360
TOO_FAR_HEIGHT = 130

MOOD_NAMES = {
    "angry": "ANGRY",
    "disgust": "DISGUSTED",
    "disgusted": "DISGUSTED",
    "fear": "FEARFUL",
    "fearful": "FEARFUL",
    "happy": "HAPPY",
    "neutral": "NEUTRAL",
    "sad": "SAD",
    "surprise": "SURPRISED",
    "surprised": "SURPRISED",
}


def normalize_mood(raw_mood):
    """Translate DeepFace labels to the mood names used by this project."""
    if not isinstance(raw_mood, str):
        return None
    return MOOD_NAMES.get(raw_mood.strip().lower())


def analyze_emotion(head_crop):
    """Return a normalized mood and DeepFace confidence for a head crop."""
    try:
        # Lazy import avoids initializing TensorFlow before NCNN/YOLO.
        from deepface import DeepFace

        analysis = DeepFace.analyze(
            img_path=head_crop,
            actions=["emotion"],
            enforce_detection=False,
            # The crop is already the target's head region. Skipping another
            # detector is substantially lighter and avoids cascade-file issues.
            detector_backend="skip",
            silent=True,
        )
        if isinstance(analysis, list):
            if not analysis:
                return None, 0.0
            analysis = analysis[0]
        if not isinstance(analysis, dict):
            return None, 0.0

        dominant_emotion = analysis.get("dominant_emotion")
        normalized_mood = normalize_mood(dominant_emotion)
        if normalized_mood is None:
            return None, 0.0

        scores = analysis.get("emotion", {})
        confidence = float(scores.get(dominant_emotion, 0.0))
        return normalized_mood, confidence
    except Exception as exc:
        print(f"[VISION] Emotion analysis failed: {exc}", flush=True)
        return None, 0.0


def mood_color(mood):
    if mood == "HAPPY":
        return (0, 255, 0)
    if mood in {"SAD", "FEARFUL"}:
        return (255, 191, 0)
    if mood in {"ANGRY", "DISGUSTED"}:
        return (0, 80, 255)
    if mood == "SURPRISED":
        return (255, 255, 0)
    return (220, 220, 220)


def publish_frame(frame):
    """Publish an annotated JPEG for dashboard_server.py."""
    ok, buffer = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not ok:
        return
    try:
        write_frame(buffer.tobytes())
    except Exception as exc:
        print(f"[VISION] Frame publish error: {exc}", flush=True)


def main():
    print("[VISION] Loading YOLO NCNN model for Raspberry Pi 5...", flush=True)
    model = YOLO(MODEL_PATH, task="detect")
    print("[VISION] YOLO NCNN model loaded.", flush=True)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("[VISION] ERROR: Could not open camera.", flush=True)
        raise SystemExit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    target_id = None
    missing_frames = 0
    current_mood = "Analyzing..."
    frame_counter = 0
    previous_time = time.time()

    print("[VISION] Starting Pi 5 vision loop in headless mode.", flush=True)
    print("[VISION] Frames will be published through frame_io.", flush=True)

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[VISION] Failed to grab camera frame.", flush=True)
                break

            frame_counter += 1
            frame_height, frame_width = frame.shape[:2]

            results = model.track(
                source=frame,
                persist=True,
                classes=[0],
                conf=PERSON_CONFIDENCE,
                imgsz=320,
                tracker="bytetrack.yaml",
                verbose=False,
            )
            boxes = results[0].boxes
            target_found = False

            if boxes is not None and boxes.id is not None:
                coordinates = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.int().cpu().numpy()

                if target_id is None and len(track_ids) > 0:
                    # Preserve the original Pi behavior: lock the first track.
                    target_id = int(track_ids[0])
                    print(f"[LOCK] Target acquired! Tracking ID: {target_id}", flush=True)

                for box, track_id in zip(coordinates, track_ids):
                    if int(track_id) != target_id:
                        continue

                    target_found = True
                    missing_frames = 0
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_width, x2), min(frame_height, y2)

                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    if bbox_width <= 0 or bbox_height <= 0:
                        target_found = False
                        continue

                    center_x = x1 + bbox_width // 2
                    if center_x < LEFT_BOUNDARY:
                        horizontal_position = "LEFT"
                    elif center_x > RIGHT_BOUNDARY:
                        horizontal_position = "RIGHT"
                    else:
                        horizontal_position = "CENTER"

                    if bbox_height > TOO_CLOSE_HEIGHT:
                        distance_status = "TOO CLOSE"
                    elif bbox_height < TOO_FAR_HEIGHT:
                        distance_status = "TOO FAR"
                    else:
                        distance_status = "OK DISTANCE"

                    if frame_counter % EMOTION_INTERVAL == 0:
                        head_y2 = min(y2, y1 + int(bbox_height * 0.40))
                        head_crop = frame[y1:head_y2, x1:x2]
                        if (
                            head_crop.size > 0
                            and head_crop.shape[0] > 20
                            and head_crop.shape[1] > 20
                        ):
                            mood, confidence = analyze_emotion(head_crop)
                            if mood is not None:
                                current_mood = mood
                                try:
                                    write_mood(current_mood, confidence)
                                except Exception as exc:
                                    print(f"[VISION] Mood publish error: {exc}", flush=True)
                                print(
                                    f"[VISION] Mood: {current_mood} "
                                    f"({confidence:.1f}%)",
                                    flush=True,
                                )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"TARGET {target_id}: {horizontal_position} | {distance_status}",
                        (x1, max(y1 - 25, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"MOOD: {current_mood}",
                        (x1, max(y1 - 5, 40)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        mood_color(current_mood),
                        2,
                    )
                    break

            if not target_found and target_id is not None:
                missing_frames += 1
                cv2.putText(
                    frame,
                    f"SEARCHING TARGET {target_id} "
                    f"({missing_frames}/{MAX_MISSING_FRAMES})",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )
                if missing_frames > MAX_MISSING_FRAMES:
                    print("[VISION] Target lost. Resetting target lock.", flush=True)
                    target_id = None
                    missing_frames = 0
                    current_mood = "Analyzing..."

            cv2.line(frame, (LEFT_BOUNDARY, 0), (LEFT_BOUNDARY, frame_height), (80, 80, 80), 1)
            cv2.line(frame, (RIGHT_BOUNDARY, 0), (RIGHT_BOUNDARY, frame_height), (80, 80, 80), 1)

            current_time = time.time()
            elapsed = current_time - previous_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            previous_time = current_time
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            publish_frame(frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[VISION] Pi 5 vision loop stopped.", flush=True)


if __name__ == "__main__":
    main()
