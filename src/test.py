import cv2
from src.detection.yolo_detector import YoloBallDetector
from src.tracking.tracker import BallTracker
from src.association.data_association import associate_ball

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\cricket player train\ball\runs\detect\ball_stump_test\weights\best.pt"
VIDEO_PATH = r"C:\cricket-ai\data\samples\test3.mp4"

# ⚠️ FIXED CALIBRATION (YOU MUST MEASURE THIS ONCE)
# Example: pitch visible ≈ 1050 px
METERS_PER_PIXEL = 20.12 / 1050

# gentle depth compensation
def perspective_scale(y, h):
    return 0.9 + 0.6 * (y / h)   # max ≈ 1.5

# ---------------- INIT ----------------
detector = YoloBallDetector(
    model_path=MODEL_PATH,
    conf=0.15,
    ball_class_id=0
)

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened()

FPS = int(cap.get(cv2.CAP_PROP_FPS))
tracker = BallTracker(fps=FPS)

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    predicted = tracker.predict() if tracker.initialized else None
    detections = detector.detect(frame, predicted)

    # draw detections
    for cx, cy, x1, y1, x2, y2, conf in detections:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.circle(frame, (cx,cy), 3, (0,0,255), -1)

    # tracker update
    if detections:
        tracker.missed_frames = 0
        matched = associate_ball(detections, predicted)
        cx, cy, *_ = matched if matched else detections[0]
        tracker.update((cx, cy))
    else:
        tracker.missed_frames += 1

    if tracker.missed_frames > 20:
        tracker.reset()

    # ---------------- DRAW TRACKED ----------------
    if tracker.initialized:
        x, y = tracker.get_position()
        cv2.circle(frame, (x,y), 6, (0,0,255), -1)

        scale = perspective_scale(y, frame.shape[0])
        speed = tracker.get_speed_kmph(METERS_PER_PIXEL, scale)
        max_speed = tracker.update_max_speed(speed)

        cv2.putText(frame, f"Speed: {speed:.1f} km/h",
                    (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.putText(frame, f"Max: {max_speed:.1f} km/h",
                    (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        if tracker.detect_bounce():
            tracker.pitch_type = tracker.classify_pitch(frame.shape[0])

        if tracker.pitch_type:
            cv2.putText(frame, tracker.pitch_type,
                        (20,95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

    cv2.imshow("Cricket Ball Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
