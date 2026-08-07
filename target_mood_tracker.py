import os

# ================================================================
# CPU / Runtime Configuration
# ================================================================

# Disable CUDA because this laptop is running inference on CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Reduce TensorFlow / backend logging noise.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ================================================================
# Imports
# ================================================================

import cv2
import time

from ultralytics import YOLO

from mood_reactions import normalize_mood
from mood_state_io import write_mood
from frame_io import write_frame


# ================================================================
# Configuration
# ================================================================

# YOLO NCNN model
MODEL_PATH = "./yolov8n_ncnn_model"

# Logitech C270
CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Person detection confidence threshold
PERSON_CONFIDENCE = 0.4

# Run emotion analysis every N frames
EMOTION_INTERVAL = 5

# Number of consecutive missing frames
# before target lock is completely reset.
MAX_MISSING_FRAMES = 30

# IoU threshold used to match the previously locked target
# with a new YOLO detection.
TARGET_IOU_THRESHOLD = 0.1

# Target label
TARGET_NUMBER = 1


# ================================================================
# Load YOLO
# ================================================================

print("[VISION] Loading YOLO NCNN model...", flush=True)

model = YOLO(
    MODEL_PATH,
    task="detect"
)

print(
    "[VISION] YOLO NCNN model loaded.",
    flush=True
)


# ================================================================
# Configure Webcam
# ================================================================

# Use Linux V4L2 backend.
# We deliberately do not force MJPG here.
cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_V4L2
)

if not cap.isOpened():

    print(
        "[VISION] ERROR: Could not open camera.",
        flush=True
    )

    raise SystemExit(1)


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)


print(
    "[VISION] Camera opened.",
    flush=True
)


# ================================================================
# Tracking / Target State
# ================================================================

# We do NOT use YOLO's tracker.
# This is a lightweight target lock based on bounding-box IoU.

target_locked = False

previous_box = None

missing_frames = 0


# ================================================================
# Mood State
# ================================================================

current_mood = "Analyzing..."

frame_counter = 0


# ================================================================
# FPS State
# ================================================================

prev_time = time.time()


# ================================================================
# Helper Functions
# ================================================================

def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union between two bounding boxes.

    box format:
        (x1, y1, x2, y2)
    """

    if box_a is None or box_b is None:
        return 0.0


    ax1, ay1, ax2, ay2 = box_a

    bx1, by1, bx2, by2 = box_b


    # Intersection coordinates
    inter_x1 = max(
        ax1,
        bx1
    )

    inter_y1 = max(
        ay1,
        by1
    )

    inter_x2 = min(
        ax2,
        bx2
    )

    inter_y2 = min(
        ay2,
        by2
    )


    # Intersection dimensions
    inter_width = max(
        0,
        inter_x2 - inter_x1
    )

    inter_height = max(
        0,
        inter_y2 - inter_y1
    )


    intersection_area = (
        inter_width *
        inter_height
    )


    # Box A area
    area_a = (
        max(0, ax2 - ax1) *
        max(0, ay2 - ay1)
    )


    # Box B area
    area_b = (
        max(0, bx2 - bx1) *
        max(0, by2 - by1)
    )


    # Union
    union_area = (
        area_a +
        area_b -
        intersection_area
    )


    if union_area <= 0:
        return 0.0


    return (
        intersection_area /
        union_area
    )


def select_target(
    detections,
    previous_box
):
    """
    Select the target person.

    Priority:
    1. Match the previous target using IoU.
    2. If no previous target exists, select the largest person.
    3. If the target cannot be matched, fall back to the largest person.

    Each detection:
        (x1, y1, x2, y2, confidence)
    """

    if not detections:
        return None


    # ============================================================
    # If target was already locked,
    # try to maintain target identity using IoU.
    # ============================================================

    if previous_box is not None:

        best_iou = 0.0

        best_match = None


        for detection in detections:

            x1, y1, x2, y2, confidence = detection


            current_box = (
                x1,
                y1,
                x2,
                y2
            )


            iou = calculate_iou(
                previous_box,
                current_box
            )


            if iou > best_iou:

                best_iou = iou

                best_match = detection


        # Target successfully matched
        if (
            best_match is not None
            and best_iou >= TARGET_IOU_THRESHOLD
        ):

            return best_match


    # ============================================================
    # If target cannot be matched,
    # select the largest detected person.
    # ============================================================

    largest_area = 0

    largest_detection = None


    for detection in detections:

        x1, y1, x2, y2, confidence = detection


        width = max(
            0,
            x2 - x1
        )

        height = max(
            0,
            y2 - y1
        )


        area = (
            width *
            height
        )


        if area > largest_area:

            largest_area = area

            largest_detection = detection


    return largest_detection


def analyze_emotion(
    head_crop
):
    """
    Run DeepFace emotion analysis.

    DeepFace is imported lazily so TensorFlow does not initialize
    before YOLO detection starts.
    """

    try:

        # Lazy import
        from deepface import DeepFace


        analysis = DeepFace.analyze(

            img_path=head_crop,

            actions=[
                "emotion"
            ],

            enforce_detection=False,

            # We already pass a head crop, so skip an extra face detector pass.
            # This avoids runtime failures when OpenCV cascade files are absent.
            detector_backend="skip",

            silent=True
        )


        # DeepFace commonly returns a list.
        # Handle both list and dictionary safely.
        if isinstance(
            analysis,
            list
        ):

            if len(analysis) == 0:

                return None, 0.0

            analysis = analysis[0]


        dominant_emotion = (
            analysis.get(
                "dominant_emotion"
            )
        )


        if not dominant_emotion:

            return None, 0.0


        emotion_scores = (
            analysis.get(
                "emotion",
                {}
            )
        )


        confidence = float(

            emotion_scores.get(

                dominant_emotion,

                0.0
            )
        )


        return (
            dominant_emotion.upper(),
            confidence
        )


    except Exception as e:

        print(
            "[VISION] Emotion analysis failed:",
            e,
            flush=True
        )

        return None, 0.0


# ================================================================
# Main Vision Loop
# ================================================================

print(
    "[VISION] Starting Laptop Vision Loop.",
    flush=True
)

print(
    "[VISION] Headless mode enabled.",
    flush=True
)

print(
    "[VISION] Frames will be published through frame_io.",
    flush=True
)


try:

    while cap.isOpened():


        # ========================================================
        # Capture Frame
        # ========================================================

        ret, frame = cap.read()


        if not ret or frame is None:

            print(
                "[VISION] Failed to grab camera frame.",
                flush=True
            )

            break


        frame_counter += 1


        # ========================================================
        # YOLO PERSON DETECTION
        #
        # IMPORTANT:
        # Do NOT use model.track().
        # The tracking backend was causing the segfault.
        # ========================================================

        results = model(

            frame,

            imgsz=320,

            classes=[0],

            conf=PERSON_CONFIDENCE,

            verbose=False
        )


        boxes = results[0].boxes


        detections = []


        # ========================================================
        # Extract Person Detections
        # ========================================================

        if (
            boxes is not None
            and len(boxes) > 0
        ):

            coords = (
                boxes.xyxy
                .cpu()
                .numpy()
            )


            confidences = (
                boxes.conf
                .cpu()
                .numpy()
            )


            for box, confidence in zip(
                coords,
                confidences
            ):

                x1, y1, x2, y2 = map(
                    int,
                    box
                )


                # Keep bounding box inside image
                x1 = max(
                    0,
                    x1
                )

                y1 = max(
                    0,
                    y1
                )

                x2 = min(
                    frame.shape[1],
                    x2
                )

                y2 = min(
                    frame.shape[0],
                    y2
                )


                detections.append(

                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        float(confidence)
                    )
                )


        # ========================================================
        # Select Target
        # ========================================================

        target_box = select_target(

            detections,

            previous_box
        )


        target_found = (
            target_box is not None
        )


        # ========================================================
        # TARGET FOUND
        # ========================================================

        if target_found:


            x1, y1, x2, y2, confidence = (
                target_box
            )


            # ----------------------------------------------------
            # Save current target box
            # ----------------------------------------------------

            previous_box = (

                x1,
                y1,

                x2,
                y2
            )


            # ----------------------------------------------------
            # Target Lock
            # ----------------------------------------------------

            if not target_locked:

                target_locked = True

                print(

                    "[LOCK] Target acquired! "
                    f"Target {TARGET_NUMBER}",

                    flush=True
                )


            # Reset missing counter
            missing_frames = 0


            # ----------------------------------------------------
            # Bounding Box Dimensions
            # ----------------------------------------------------

            bbox_width = (
                x2 -
                x1
            )

            bbox_height = (
                y2 -
                y1
            )


            # ----------------------------------------------------
            # Target Center
            # ----------------------------------------------------

            center_x = (

                x1 +

                (
                    bbox_width //
                    2
                )
            )


            # ====================================================
            # Spatial Position
            # ====================================================

            if center_x < 250:

                horizontal_pos = (
                    "LEFT"
                )

            elif center_x > 390:

                horizontal_pos = (
                    "RIGHT"
                )

            else:

                horizontal_pos = (
                    "CENTER"
                )


            # ====================================================
            # Distance Evaluation
            # ====================================================

            if bbox_height > 360:

                distance_status = (
                    "TOO CLOSE"
                )

            elif bbox_height < 130:

                distance_status = (
                    "TOO FAR"
                )

            else:

                distance_status = (
                    "OK DISTANCE"
                )


            # ====================================================
            # Mood Detection
            # ====================================================

            if (

                frame_counter %
                EMOTION_INTERVAL

                == 0

            ):


                # ------------------------------------------------
                # Extract top 40% of person bounding box
                # ------------------------------------------------

                head_y2 = (

                    y1 +

                    int(

                        bbox_height *

                        0.40
                    )
                )


                head_crop = frame[

                    y1:
                    head_y2,

                    x1:
                    x2
                ]


                # ------------------------------------------------
                # Validate crop
                # ------------------------------------------------

                if (

                    head_crop.size > 0

                    and

                    head_crop.shape[0] > 20

                    and

                    head_crop.shape[1] > 20

                ):


                    print(

                        "[VISION] Running "
                        "emotion analysis...",

                        flush=True
                    )


                    mood, confidence = (
                        analyze_emotion(
                            head_crop
                        )
                    )


                    # ------------------------------------------------
                    # Update mood only if valid
                    # ------------------------------------------------

                    normalized_mood = normalize_mood(mood)

                    if normalized_mood is not None:

                        current_mood = (
                            normalized_mood
                        )


                        try:

                            write_mood(

                                current_mood,

                                confidence
                            )

                        except Exception as e:

                            print(

                                "[VISION] "
                                "Mood publish error:",

                                e,

                                flush=True
                            )


                        print(

                            "[VISION] Mood: "

                            f"{current_mood} "

                            f"({confidence:.1f}%)",

                            flush=True
                        )


            # ====================================================
            # Target Info Overlay
            # ====================================================

            status_line1 = (

                f"TARGET {TARGET_NUMBER}: "

                f"{horizontal_pos} | "

                f"{distance_status}"
            )


            status_line2 = (

                f"MOOD: "

                f"{current_mood}"
            )


            # ----------------------------------------------------
            # Person Bounding Box
            # ----------------------------------------------------

            cv2.rectangle(

                frame,

                (
                    x1,
                    y1
                ),

                (
                    x2,
                    y2
                ),

                (255, 255, 0),

                2
            )


            # ----------------------------------------------------
            # Position / Distance Text
            # ----------------------------------------------------

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

                2
            )


            # ----------------------------------------------------
            # Mood Color
            # ----------------------------------------------------

            if current_mood == "HAPPY":
                mood_color = (0, 255, 0)
            elif current_mood in {"SAD", "FEARFUL"}:
                mood_color = (255, 191, 0)
            elif current_mood in {"ANGRY", "DISGUSTED"}:
                mood_color = (0, 80, 255)
            elif current_mood == "SURPRISED":
                mood_color = (255, 255, 0)
            else:
                mood_color = (220, 220, 220)


            # ----------------------------------------------------
            # Mood Text
            # ----------------------------------------------------

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

                mood_color,

                2
            )


        # ========================================================
        # TARGET NOT FOUND
        # ========================================================

        else:


            if target_locked:

                missing_frames += 1


                # ------------------------------------------------
                # Searching Status
                # ------------------------------------------------

                cv2.putText(

                    frame,

                    (
                        f"SEARCHING TARGET "
                        f"{TARGET_NUMBER} "
                        f"({missing_frames}/"
                        f"{MAX_MISSING_FRAMES})"
                    ),

                    (
                        20,
                        80
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (
                        0,
                        165,
                        255
                    ),

                    2
                )


                # ------------------------------------------------
                # Reset Target
                # ------------------------------------------------

                if (

                    missing_frames >

                    MAX_MISSING_FRAMES

                ):


                    print(

                        "[VISION] Target lost. "
                        "Resetting target lock.",

                        flush=True
                    )


                    target_locked = False

                    previous_box = None

                    missing_frames = 0

                    current_mood = (
                        "Analyzing..."
                    )


        # ========================================================
        # Center Alignment Lines
        # ========================================================

        cv2.line(

            frame,

            (
                250,
                0
            ),

            (
                250,
                FRAME_HEIGHT
            ),

            (
                80,
                80,
                80
            ),

            1
        )


        cv2.line(

            frame,

            (
                390,
                0
            ),

            (
                390,
                FRAME_HEIGHT
            ),

            (
                80,
                80,
                80
            ),

            1
        )


        # ========================================================
        # FPS Calculation
        # ========================================================

        curr_time = time.time()


        elapsed_time = (

            curr_time -

            prev_time
        )


        if elapsed_time > 0:

            fps = (

                1.0 /

                elapsed_time
            )

        else:

            fps = 0.0


        prev_time = curr_time


        # ========================================================
        # FPS Overlay
        # ========================================================

        cv2.putText(

            frame,

            f"FPS: {fps:.1f}",

            (
                20,
                40
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (
                0,
                255,
                0
            ),

            2
        )


        # ========================================================
        # Publish Annotated Frame
        #
        # Headless operation:
        # There is NO cv2.imshow().
        # The dashboard receives JPEG bytes through frame_io.
        # ========================================================

        ok, buffer = cv2.imencode(

            ".jpg",

            frame,

            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),

                80
            ]
        )


        if ok:

            try:

                write_frame(

                    buffer.tobytes()
                )

            except Exception as e:

                # A dropped frame should never
                # stop the vision loop.

                print(

                    "[VISION] Frame publish "
                    f"error: {e}",

                    flush=True
                )


finally:

    # ============================================================
    # Cleanup
    # ============================================================

    cap.release()


    print(

        "[VISION] Vision loop stopped.",

        flush=True
    )
