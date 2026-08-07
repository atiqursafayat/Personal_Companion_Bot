import cv2
import time
from ultralytics import YOLO
from deepface import DeepFace

from device_controls import save_requested_picture
from mood_state_io import write_mood

# 1. Load YOLOv8 Nano NCNN model for person detection
model = YOLO("./yolov8n_ncnn_model", task="detect")

# 2. Configure the operating system's default camera.
# OpenCV index 0 represents the default/first video capture device.
DEFAULT_CAMERA_INDEX = 0
cap = cv2.VideoCapture(DEFAULT_CAMERA_INDEX)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

if not cap.isOpened():
    raise RuntimeError(
        "Could not open the default camera (OpenCV index 0). "
        "Check that a default camera is connected and not exclusively in use."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Tracking & Mood Variables
target_id = None
missing_frames = 0
MAX_MISSING_FRAMES = 30

current_mood = "Analyzing..."
frame_counter = 0
EMOTION_INTERVAL = 5  # Run mood analysis every 5 frames to save CPU

prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab camera frame.")
        break

    frame_counter += 1

    saved_picture = save_requested_picture(frame, cv2)
    if saved_picture:
        print(f"[CAMERA] Picture saved to {saved_picture}")

    # Track Humans (class 0 = person)
    results = model.track(
        source=frame,
        persist=True,
        classes=[0],
        imgsz=320,
        tracker="bytetrack.yaml",
        verbose=False
    )

    boxes = results[0].boxes
    target_found = False

    if boxes is not None and boxes.id is not None:
        coords = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.int().cpu().numpy()

        # Lock onto the first person detected
        if target_id is None and len(track_ids) > 0:
            target_id = int(track_ids[0])
            print(f"[LOCK] Target acquired! Tracking ID: {target_id}")

        # Search for our locked target
        for box, tid in zip(coords, track_ids):
            if int(tid) == target_id:
                target_found = True
                missing_frames = 0

                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(FRAME_WIDTH, x2), min(FRAME_HEIGHT, y2)

                bbox_width = x2 - x1
                bbox_height = y2 - y1
                center_x = x1 + (bbox_width // 2)

                # --- Spatial Position (Left / Right / Center) ---
                if center_x < 250:
                    horizontal_pos = "LEFT"
                elif center_x > 390:
                    horizontal_pos = "RIGHT"
                else:
                    horizontal_pos = "CENTER"

                # --- Distance Evaluation ---
                if bbox_height > 360:
                    distance_status = "TOO CLOSE"
                elif bbox_height < 130:
                    distance_status = "TOO FAR"
                else:
                    distance_status = "OK DISTANCE"

                # --- Mood Detection on Upper Body/Head Area ---
                if frame_counter % EMOTION_INTERVAL == 0:
                    # Isolate top 40% of the bounding box (where the head/face is located)
                    head_y2 = y1 + int(bbox_height * 0.40)
                    head_crop = frame[y1:head_y2, x1:x2]

                    if head_crop.size > 0:
                        try:
                            # DeepFace analyzes the crop directly with built-in enforcement bypass
                            analysis = DeepFace.analyze(
                                img_path=head_crop,
                                actions=['emotion'],
                                enforce_detection=False,
                                detector_backend='opencv',  # Lightest internal backend
                                silent=True
                            )
                            current_mood = analysis[0]['dominant_emotion'].upper()
                            emotion_scores = analysis[0].get('emotion', {})
                            confidence = emotion_scores.get(analysis[0]['dominant_emotion'])
                            write_mood(current_mood, confidence)
                        except Exception:
                            pass

                # --- Visual Overlays ---
                # Draw person box (Cyan)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                # Display Target Info & Mood
                status_line1 = f"TARGET {target_id}: {horizontal_pos} | {distance_status}"
                status_line2 = f"MOOD: {current_mood}"

                cv2.putText(frame, status_line1, (x1, max(y1 - 25, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                
                mood_color = (0, 255, 0) if current_mood == "HAPPY" else (0, 255, 255)
                cv2.putText(frame, status_line2, (x1, max(y1 - 5, 40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, mood_color, 2)
                break

    # Reset target lock if lost
    if not target_found and target_id is not None:
        missing_frames += 1
        cv2.putText(
            frame, f"SEARCHING TARGET {target_id} ({missing_frames}/{MAX_MISSING_FRAMES})", 
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
        )
        if missing_frames > MAX_MISSING_FRAMES:
            target_id = None
            missing_frames = 0
            current_mood = "Analyzing..."

    # Center alignment lines
    cv2.line(frame, (250, 0), (250, FRAME_HEIGHT), (80, 80, 80), 1)
    cv2.line(frame, (390, 0), (390, FRAME_HEIGHT), (80, 80, 80), 1)

    # FPS counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Target Human & Mood Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        target_id = None
        missing_frames = 0
        current_mood = "Analyzing..."

cap.release()
cv2.destroyAllWindows()
