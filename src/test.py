import cv2
import time

from src.detection.yolo_detector import YoloBallDetector
from src.tracking.tracker import BallTracker
from src.association.data_association import associate_ball

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\cricket player train\ball\runs\detect\ball_stump_test\weights\best.pt"
VIDEO_PATH = r"C:\cricket-ai\data\samples\test6.mp4"

METERS_PER_PIXEL = 20.12 / 520   # calibrate later

# ---------------- INIT ----------------
detector = YoloBallDetector(
    model_path=MODEL_PATH,
    conf=0.2,
    ball_class_id=0
)

tracker = BallTracker()

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "❌ Failed to open video"

prev_time = time.time()

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------- REAL TIME DELTA --------
    now = time.time()
    dt = now - prev_time
    prev_time = now
    dt = max(0.001, min(dt, 0.1))  # clamp

    predicted = None
    if tracker.initialized:
        predicted = tracker.predict(dt)

    # -------- DETECT --------
    detections = detector.detect(frame, predicted)

    # draw YOLO detections
    for det in detections:
        cx, cy, x1, y1, x2, y2, conf = det
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    # -------- TRACKER UPDATE --------
    if detections:
        tracker.missed_frames = 0

        if not tracker.initialized:
            cx, cy, *_ = detections[0]
            tracker.update((cx, cy))
        else:
            matched = associate_ball(detections, predicted)
            if matched:
                cx, cy, *_ = matched
                tracker.update((cx, cy))
    else:
        tracker.missed_frames += 1

    # reset if lost
    if tracker.missed_frames > 15:
        tracker.reset()

    # -------- DRAW TRACKED STATE --------
    if tracker.initialized:
        x, y = tracker.get_position()

        # tracked ball
        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)

        # speed
        speed_now = tracker.get_speed_kmph(METERS_PER_PIXEL)
        speed_max = tracker.update_max_speed(METERS_PER_PIXEL)

        cv2.putText(
            frame,
            f"Speed: {speed_now:.1f} km/h",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Max: {speed_max:.1f} km/h",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        # -------- BOUNCE + PITCH TYPE --------
        if tracker.detect_bounce():
            tracker.pitch_type = tracker.classify_pitch(frame.shape[0])

        if tracker.pitch_type:
            cv2.putText(
                frame,
                tracker.pitch_type,
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )

        # bounce marker
        if tracker.bounce_y is not None:
            cv2.circle(frame, (x, tracker.bounce_y), 8, (0, 0, 255), 2)

    # -------- DISPLAY --------
    cv2.imshow("Cricket Ball Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
