import cv2
import time
from src.detection.yolo_detector import YoloBallDetector
from src.tracking.tracker import BallTracker
from src.association.data_association import associate_ball

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\cricket player train\ball\runs\detect\ball_stump_test\weights\best.pt"
VIDEO_PATH = r"C:\cricket-ai\data\samples\test3.mp4"
METERS_PER_PIXEL = 20.12 / 520

def perspective_scale(y, h):
    return 1.0 + 1.5 * (y / h)

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

    now = time.time()
    dt = max(0.001, now - prev_time)
    prev_time = now

    predicted = tracker.predict() if tracker.initialized else None
    detections = detector.detect(frame, predicted)

    # draw detections
    for det in detections:
        cx, cy, x1, y1, x2, y2, conf = det
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.circle(frame, (cx, cy), 3, (0,0,255), -1)

    # update tracker
    if detections:
        tracker.missed_frames = 0
        if not tracker.initialized:
            cx, cy, *_ = detections[0]
            tracker.update((cx, cy))
        else:
            match = associate_ball(detections, predicted)
            if match:
                cx, cy, *_ = match
                tracker.update((cx, cy))
    else:
        tracker.missed_frames += 1

    if tracker.missed_frames > 15:
        tracker.reset()

    # ---------------- DISPLAY ----------------
    if tracker.initialized:
        x, y = tracker.get_position()
        cv2.circle(frame, (x, y), 6, (0,0,255), -1)

        scale = perspective_scale(y, frame.shape[0])
        speed = tracker.get_speed_kmph(METERS_PER_PIXEL, scale)

        cv2.putText(frame, f"Speed: {speed:.1f} km/h",
                    (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        if tracker.release_speed:
            cv2.putText(frame, f"Release: {tracker.release_speed:.1f} km/h",
                        (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.putText(frame, f"Max: {tracker.max_speed:.1f} km/h",
                    (20,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        if tracker.detect_bounce():
            tracker.pitch_type = tracker.classify_pitch(frame.shape[0])
            print(f"🏏 BOUNCE → {tracker.pitch_type}")

        if tracker.pitch_type:
            cv2.putText(frame, tracker.pitch_type,
                        (20,130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

    cv2.imshow("Cricket Ball Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
