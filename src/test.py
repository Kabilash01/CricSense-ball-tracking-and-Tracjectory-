import cv2
from src.detection.yolo_detector import YoloBallDetector
from src.tracking.tracker import BallTracker
from src.association.data_association import associate_ball

MODEL_PATH = r"C:\cricket player train\ball\runs\detect\ball_stump_test\weights\best.pt"
VIDEO_PATH = r"C:\cricket-ai\data\samples\test3.mp4"
detector = YoloBallDetector(
    MODEL_PATH,
    conf=0.01,
    ball_class_id=0
)

tracker = BallTracker(fps=30)
cap = cv2.VideoCapture(VIDEO_PATH)

assert cap.isOpened(), "Video not opened"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1️⃣ DETECT
    detections = detector.detect(frame)

    print("detections:", len(detections), "tracker initialized:", tracker.initialized)

    # 2️⃣ ALWAYS DRAW YOLO DETECTIONS (VISUAL ONLY)
    for det in detections:
        cx, cy, x1, y1, x2, y2, conf = det
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.circle(frame, (cx, cy), 3, (255,0,0), -1)

    # 3️⃣ TRACKING LOGIC
    if not tracker.initialized:
        if detections:
            cx, cy, *_ = detections[0]
            tracker.update((cx, cy))
    else:
        predicted = tracker.predict()
        matched = associate_ball(detections, predicted)

        if matched:
            cx, cy, *_ = matched
            tracker.update((cx, cy))

    # 4️⃣ DRAW TRACKED BALL (RED DOT)
    if tracker.initialized:
        x, y = tracker.get_position()
        cv2.circle(frame, (x, y), 6, (0,0,255), -1)

    cv2.imshow("Ball Tracking (VISUAL FIXED)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
