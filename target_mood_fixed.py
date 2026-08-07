import os
import time

# Raspberry Pi 5: force CPU inference and reduce TensorFlow console noise.
# These must be set before DeepFace/TensorFlow is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
from ultralytics import YOLO

from device_controls import save_requested_picture
from mood_state_io import write_mood


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "./yolov8n_ncnn_model"

DEFAULT_CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

PERSON_CONFIDENCE = 0.4

MAX_MISSING_FRAMES = 30

# Run emotion detection once every 1.5 seconds instead of every 5 frames.
# This is much lighter on the Raspberry Pi 5.
EMOTION_INTERVAL_SECONDS = 1.5

LEFT_BOUNDARY = 250
RIGHT_BOUNDARY = 390

TOO_CLOSE_HEIGHT = 360
TOO_FAR_HEIGHT = 130


# ============================================================
# Mood helpers
# ============================================================

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
    if not isinstance(raw_mood, str):
        return None

    return MOOD_NAMES.get(raw_mood.strip().lower())


def analyze_emotion(head_crop):
    """
    Analyze emotion from the already-cropped head region.

    DeepFace is imported lazily so TensorFlow does not initialize
    before YOLO/NCNN.
    """

    try:
        # IMPORTANT:
        # Lazy import prevents TensorFlow from initializing at startup.
        from deepface import DeepFace

        analysis = DeepFace.analyze(
            img_path=head_crop,
            actions=["emotion"],
            enforce_detection=False,

            # We already provide a head crop.
            # This also avoids the missing OpenCV Haar cascade problem.
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

        mood = normalize_mood(dominant_emotion)

        if mood is None:
            return None, 0.0

        emotion_scores = analysis.get("emotion", {})

        confidence = emotion_scores.get(
            dominant_emotion,
            0.0
        )

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        return mood, confidence

    except Exception as exc:
        # NEVER silently hide DeepFace errors while debugging.
        print(
            f"[DEEPFACE ERROR] "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        return None, 0.0


def get_mood_color(mood):
    if mood == "HAPPY":
        return (0, 255, 0)

    if mood in {"SAD", "FEARFUL"}:
        return (255, 191, 0)

    if mood in {"ANGRY", "DISGUSTED"}:
        return (0, 80, 255)

    if mood == "SURPRISED":
        return (255, 255, 0)

    if mood == "NEUTRAL":
        return (220, 220, 220)

    return (0, 255, 255)


# ============================================================
# Load YOLO
# ============================================================

print("[VISION] Loading YOLO NCNN model...", flush=True)

model = YOLO(
    MODEL_PATH,
    task="detect"
)

print("[VISION] YOLO model loaded.", flush=True)


# ============================================================
# Camera
# ============================================================

# V4L2 is preferable on Raspberry Pi/Linux.
cap = cv2.VideoCapture(
    DEFAULT_CAMERA_INDEX,
    cv2.CAP_V4L2
)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open the default camera "
        f"(OpenCV index {DEFAULT_CAMERA_INDEX}). "
        "Check that the camera is connected and not in use."
    )


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)


# ============================================================
# Tracking state
# ============================================================

target_id = None

missing_frames = 0

current_mood = "Analyzing..."

last_emotion_time = 0.0

previous_time = time.monotonic()


print("[VISION] Vision loop started.", flush=True)


# ============================================================
# Main loop
# ============================================================

try:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret or frame is None:
            print(
                "[VISION] Failed to grab camera frame.",
                flush=True,
            )
            break


        # IMPORTANT:
        # Use actual camera dimensions rather than assuming 640x480.
        frame_height, frame_width = frame.shape[:2]


        # ----------------------------------------------------
        # Requested picture capture
        # ----------------------------------------------------

        try:

            saved_picture = save_requested_picture(
                frame,
                cv2
            )

            if saved_picture:

                print(
                    f"[CAMERA] Picture saved to "
                    f"{saved_picture}",
                    flush=True,
                )

        except Exception as exc:

            print(
                f"[CAMERA ERROR] {exc}",
                flush=True,
            )


        # ----------------------------------------------------
        # YOLO + ByteTrack
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Process tracked people
        # ----------------------------------------------------

        if (
            boxes is not None
            and boxes.id is not None
        ):

            coordinates = (
                boxes.xyxy
                .cpu()
                .numpy()
            )

            track_ids = (
                boxes.id
                .int()
                .cpu()
                .numpy()
            )


            # -----------------------------------------------
            # Acquire target
            # -----------------------------------------------

            if (
                target_id is None
                and len(track_ids) > 0
            ):

                target_id = int(
                    track_ids[0]
                )

                print(
                    f"[LOCK] Target acquired! "
                    f"Tracking ID: {target_id}",
                    flush=True,
                )


            # -----------------------------------------------
            # Find locked target
            # -----------------------------------------------

            for box, track_id in zip(
                coordinates,
                track_ids
            ):

                if int(track_id) != target_id:
                    continue


                target_found = True

                missing_frames = 0


                x1, y1, x2, y2 = map(
                    int,
                    box
                )


                # Clamp coordinates to REAL frame dimensions.
                x1 = max(
                    0,
                    min(
                        x1,
                        frame_width - 1
                    )
                )

                y1 = max(
                    0,
                    min(
                        y1,
                        frame_height - 1
                    )
                )

                x2 = max(
                    0,
                    min(
                        x2,
                        frame_width
                    )
                )

                y2 = max(
                    0,
                    min(
                        y2,
                        frame_height
                    )
                )


                bbox_width = (
                    x2 - x1
                )

                bbox_height = (
                    y2 - y1
                )


                # Invalid box protection
                if (
                    bbox_width <= 0
                    or bbox_height <= 0
                ):

                    target_found = False
                    continue


                center_x = (
                    x1
                    + bbox_width // 2
                )


                # -------------------------------------------
                # Horizontal position
                # -------------------------------------------

                if center_x < LEFT_BOUNDARY:

                    horizontal_pos = "LEFT"

                elif center_x > RIGHT_BOUNDARY:

                    horizontal_pos = "RIGHT"

                else:

                    horizontal_pos = "CENTER"


                # -------------------------------------------
                # Apparent distance
                # -------------------------------------------

                if bbox_height > TOO_CLOSE_HEIGHT:

                    distance_status = "TOO CLOSE"

                elif bbox_height < TOO_FAR_HEIGHT:

                    distance_status = "TOO FAR"

                else:

                    distance_status = "OK DISTANCE"


                # -------------------------------------------
                # Emotion detection
                # -------------------------------------------

                current_time = time.monotonic()

                if (
                    current_time
                    - last_emotion_time
                    >= EMOTION_INTERVAL_SECONDS
                ):

                    last_emotion_time = (
                        current_time
                    )


                    # Top 40% of tracked person
                    head_y2 = min(
                        y2,
                        y1
                        + int(
                            bbox_height
                            * 0.40
                        )
                    )


                    head_crop = frame[
                        y1:head_y2,
                        x1:x2
                    ]


                    # Protect DeepFace from invalid/tiny crops.
                    if (
                        head_crop.size > 0
                        and head_crop.shape[0] > 20
                        and head_crop.shape[1] > 20
                    ):

                        mood, confidence = (
                            analyze_emotion(
                                head_crop
                            )
                        )


                        if mood is not None:

                            current_mood = mood


                            try:

                                write_mood(
                                    current_mood,
                                    confidence
                                )

                            except Exception as exc:

                                print(
                                    "[MOOD WRITE ERROR] "
                                    f"{exc}",
                                    flush=True,
                                )


                            print(
                                f"[MOOD] "
                                f"{current_mood} "
                                f"({confidence:.1f}%)",
                                flush=True,
                            )


                # -------------------------------------------
                # Bounding box
                # -------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 0),
                    2,
                )


                status_line1 = (
                    f"TARGET {target_id}: "
                    f"{horizontal_pos} | "
                    f"{distance_status}"
                )

                status_line2 = (
                    f"MOOD: {current_mood}"
                )


                cv2.putText(
                    frame,
                    status_line1,
                    (
                        x1,
                        max(
                            y1 - 25,
                            20
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                )


                cv2.putText(
                    frame,
                    status_line2,
                    (
                        x1,
                        max(
                            y1 - 5,
                            40
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    get_mood_color(
                        current_mood
                    ),
                    2,
                )


                # We only care about the locked target.
                break


        # ----------------------------------------------------
        # Target lost
        # ----------------------------------------------------

        if (
            not target_found
            and target_id is not None
        ):

            missing_frames += 1


            cv2.putText(
                frame,
                (
                    f"SEARCHING TARGET "
                    f"{target_id} "
                    f"({missing_frames}/"
                    f"{MAX_MISSING_FRAMES})"
                ),
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )


            if (
                missing_frames
                > MAX_MISSING_FRAMES
            ):

                print(
                    "[LOCK] Target lost. "
                    "Resetting lock.",
                    flush=True,
                )

                target_id = None

                missing_frames = 0

                current_mood = (
                    "Analyzing..."
                )


        # ----------------------------------------------------
        # Center alignment lines
        # ----------------------------------------------------

        cv2.line(
            frame,
            (
                LEFT_BOUNDARY,
                0
            ),
            (
                LEFT_BOUNDARY,
                frame_height
            ),
            (80, 80, 80),
            1,
        )


        cv2.line(
            frame,
            (
                RIGHT_BOUNDARY,
                0
            ),
            (
                RIGHT_BOUNDARY,
                frame_height
            ),
            (80, 80, 80),
            1,
        )


        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        current_time = time.monotonic()

        elapsed = (
            current_time
            - previous_time
        )

        fps = (
            1.0 / elapsed
            if elapsed > 0
            else 0.0
        )

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


        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            "Target Human & Mood Tracker",
            frame
        )


        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        # Q = quit
        if key == ord("q"):
            break


        # R = reset target
        elif key == ord("r"):

            print(
                "[LOCK] Manual reset.",
                flush=True,
            )

            target_id = None

            missing_frames = 0

            current_mood = (
                "Analyzing..."
            )


finally:

    cap.release()

    cv2.destroyAllWindows()

    print(
        "[VISION] Vision loop stopped.",
        flush=True,
    )