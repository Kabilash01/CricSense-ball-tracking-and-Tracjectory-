import cv2
from src.detection.yolo_detector import YoloBallDetector
from src.tracking.tracker import BallTracker
from src.association.data_association import associate_ball

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\cricket player train\ball\runs\detect\ball_stump_test\weights\best.pt"
VIDEO_PATH = r"C:\cricket-ai\data\samples\test6.mp4"

METERS_PER_PIXEL = 20.12 / 490  # 🔧 adjust 520 to your measured pitch length
PERSPECTIVE_SCALE = 3.8      # 🔧 monocular depth compensation

# ---------------- INIT ----------------
detector = YoloBallDetector(
    model_path=MODEL_PATH,
    conf=0.01,
    ball_class_id=0
)

tracker = BallTracker(fps=30)

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "❌ Failed to open video"

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    predicted = None

    # 1️⃣ PREDICT (ALWAYS if initialized)
    if tracker.initialized:
        predicted = tracker.predict()

    # 2️⃣ DETECT
    detections = detector.detect(frame, predicted)

    # 3️⃣ DRAW YOLO DETECTIONS (GREEN BOX + BLUE CENTER)
    for det in detections:
        cx, cy, x1, y1, x2, y2, conf = det
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

    # 4️⃣ TRACKER INIT / UPDATE
    if not tracker.initialized:
        if detections:
            cx, cy, *_ = detections[0]
            tracker.update((cx, cy))
    else:
        matched = associate_ball(detections, predicted)
        if matched:
            cx, cy, *_ = matched
            tracker.update((cx, cy))

    # 5️⃣ DRAW TRACKED BALL + SPEED + BOUNCE
    if tracker.initialized:
        # --- tracked position ---
        x, y = tracker.get_position()
        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)

        # --- speed ---
        speed_now = tracker.get_speed_kmph(METERS_PER_PIXEL) * PERSPECTIVE_SCALE
        speed_max = tracker.update_max_speed(METERS_PER_PIXEL) * PERSPECTIVE_SCALE

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

        # --- bounce detection ---
        if tracker.detect_bounce():
            cv2.putText(
                frame,
                "BOUNCE",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

    # 6️⃣ SHOW
    cv2.imshow("Cricket Ball Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
