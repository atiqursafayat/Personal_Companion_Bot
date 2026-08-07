import os

# ------------------------------------------------------------------
# Force CPU execution
# ------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import time

from ultralytics import YOLO
from mood_state_io import write_mood


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MODEL_PATH = "yolov8n.pt"

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

EMOTION_INTERVAL = 5

# Minimum person detection confidence
PERSON_CONFIDENCE = 0.4

# Number of frames allowed without finding target
MAX_MISSING_FRAMES = 30


# ------------------------------------------------------------------
# Load YOLO
# ------------------------------------------------------------------
print("[VISION] Loading YOLO...", flush=True)

model = YOLO(MODEL_PATH)

print("[VISION] YOLO loaded.", flush=True)


# ------------------------------------------------------------------
# Open webcam
# ------------------------------------------------------------------
cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_V4L2
)

if not cap.isOpened():
    print("[VISION] ERROR: Could not open webcam.", flush=True)
    raise SystemExit(1)


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)


print("[VISION] Camera opened.", flush=True)


# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------
current_mood = "Analyzing..."

frame_counter = 0

missing_frames = 0

target_locked = False

# Previous target bounding box
previous_box = None


# ------------------------------------------------------------------
# Helper: calculate IoU
# ------------------------------------------------------------------
def calculate_iou(box_a, box_b):

    if box_a is None or box_b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)

    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width *
        intersection_height
    )

    area_a = (
        (ax2 - ax1) *
        (ay2 - ay1)
    )

    area_b = (
        (bx2 - bx1) *
        (by2 - by1)
    )

    union_area = (
        area_a +
        area_b -
        intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


# ------------------------------------------------------------------
# Start vision loop
# ------------------------------------------------------------------
print(
    "[VISION] Starting Laptop Vision Loop.",
    flush=True
)

print(
    "[VISION] Press 'q' on the preview window to exit.",
    flush=True
)


try:

    while cap.isOpened():

        # ----------------------------------------------------------
        # Read frame
        # ----------------------------------------------------------
        ret, frame = cap.read()

        if not ret or frame is None:

            print(
                "[VISION] Failed to grab camera frame.",
                flush=True
            )

            break


        frame_counter += 1


        # ----------------------------------------------------------
        # YOLO Detection
        #
        # IMPORTANT:
        # We use model() instead of model.track()
        # ----------------------------------------------------------
        results = model(
            frame,
            imgsz=320,
            classes=[0],          # Person class
            conf=PERSON_CONFIDENCE,
            device="cpu",
            verbose=False
        )


        boxes = results[0].boxes


        detected_boxes = []


        # ----------------------------------------------------------
        # Extract detected people
        # ----------------------------------------------------------
        if boxes is not None and len(boxes) > 0:

            coords = boxes.xyxy.cpu().numpy()

            confidences = boxes.conf.cpu().numpy()


            for box, confidence in zip(
                coords,
                confidences
            ):

                x1, y1, x2, y2 = map(
                    int,
                    box
                )

                detected_boxes.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        float(confidence)
                    )
                )


        target_box = None


        # ----------------------------------------------------------
        # Target selection
        #
        # If we already have a target:
        #   Try to find the detection with highest IoU
        #
        # Otherwise:
        #   Select the largest person
        # ----------------------------------------------------------
        if len(detected_boxes) > 0:


            if previous_box is not None:

                best_iou = 0.0

                best_detection = None


                for detection in detected_boxes:

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

                        best_detection = detection


                # Target found again
                if best_detection is not None and best_iou > 0.1:

                    target_box = best_detection


            # ------------------------------------------------------
            # If target could not be matched,
            # select largest person
            # ------------------------------------------------------
            if target_box is None:

                largest_area = 0

                for detection in detected_boxes:

                    x1, y1, x2, y2, confidence = detection

                    width = x2 - x1
                    height = y2 - y1

                    area = width * height


                    if area > largest_area:

                        largest_area = area

                        target_box = detection


        # ----------------------------------------------------------
        # Process target
        # ----------------------------------------------------------
        if target_box is not None:

            x1, y1, x2, y2, confidence = target_box


            # Keep coordinates inside image
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


            previous_box = (
                x1,
                y1,
                x2,
                y2
            )


            if not target_locked:

                target_locked = True

                print(
                    "[LOCK] Target acquired!",
                    flush=True
                )


            missing_frames = 0


            # ------------------------------------------------------
            # Calculate head region
            # ------------------------------------------------------
            bbox_height = y2 - y1

            head_y2 = (
                y1 +
                int(
                    bbox_height *
                    0.45
                )
            )


            head_crop = frame[
                y1:head_y2,
                x1:x2
            ]


            # ------------------------------------------------------
            # Emotion analysis
            #
            # Import DeepFace only when needed.
            # This avoids loading TensorFlow at startup.
            # ------------------------------------------------------
            if (
                frame_counter %
                EMOTION_INTERVAL
                == 0
            ):

                if (
                    head_crop.size > 0
                    and head_crop.shape[0] > 20
                    and head_crop.shape[1] > 20
                ):

                    try:

                        print(
                            "[VISION] Running emotion analysis...",
                            flush=True
                        )


                        # Lazy import
                        from deepface import DeepFace


                        analysis = DeepFace.analyze(

                            img_path=head_crop,

                            actions=[
                                "emotion"
                            ],

                            enforce_detection=False,

                            detector_backend="skip",

                            silent=True
                        )


                        # DeepFace may return list
                        if isinstance(
                            analysis,
                            list
                        ):

                            analysis = analysis[0]


                        dominant_emotion = (
                            analysis.get(
                                "dominant_emotion"
                            )
                        )


                        if dominant_emotion:

                            current_mood = (
                                dominant_emotion.upper()
                            )


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


                            write_mood(

                                current_mood,

                                confidence
                            )


                            print(

                                f"[VISION] Mood: "
                                f"{current_mood} "
                                f"({confidence:.1f}%)",

                                flush=True
                            )


                    except Exception as e:

                        print(

                            "[VISION] Emotion "
                            f"analysis error: {e}",

                            flush=True
                        )


            # ------------------------------------------------------
            # Draw target box
            # ------------------------------------------------------
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


            # ------------------------------------------------------
            # Draw mood
            # ------------------------------------------------------
            cv2.putText(

                frame,

                f"MOOD: {current_mood}",

                (
                    x1,
                    max(
                        y1 - 10,
                        30
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 255),

                2
            )


        # ----------------------------------------------------------
        # Target lost
        # ----------------------------------------------------------
        else:

            if target_locked:

                missing_frames += 1


                if (
                    missing_frames >
                    MAX_MISSING_FRAMES
                ):

                    print(

                        "[VISION] Target lost.",

                        flush=True
                    )


                    target_locked = False

                    previous_box = None

                    missing_frames = 0

                    current_mood = (
                        "Analyzing..."
                    )


        # ----------------------------------------------------------
        # Show preview
        # ----------------------------------------------------------
        cv2.imshow(

            "Robot Vision - Laptop Preview",

            frame
        )


        if (
            cv2.waitKey(1) &
            0xFF
        ) == ord("q"):

            break


finally:

    cap.release()

    cv2.destroyAllWindows()

    print(
        "[VISION] Vision loop stopped.",
        flush=True
    )