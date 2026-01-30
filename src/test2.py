import cv2
import time
import json

from src.detection.yolo_detector import YoloBallDetector
from src.tracking.tracker import BallTracker
from src.association.data_association import associate_ball
from src.detection.boundary_detector import BoundaryLineDetector
from src.events.boundary_logic import intersects
from src.events.boundary_event import classify_boundary

# ---------------- CONFIG ----------------
BALL_MODEL_PATH = r"C:\CricketSense-Ball\ball_test\weights\best.pt"
BOUNDARY_MODEL_PATH = r"C:\CrickeSense-train\Boundary\runs\detect\boundary_detect\weights\best.pt"
VIDEO_PATH = r"C:\CricketSense\data\samples\test6.mp4"

OUTPUT_JSON = "ball_analysis.json"
EVENTS_JSON = "events.json"

METERS_PER_PIXEL = 18.5 / 520

def perspective_scale(y, h):
    return 1.0 + 0.6 * (y / h)

# ---------------- EVENT BUFFERS ----------------
events = []
deliveries = []

# ---------------- INIT ----------------
ball_detector = YoloBallDetector(
    model_path=BALL_MODEL_PATH,
    conf=0.25,
    ball_class_id=0
)

boundary_detector = BoundaryLineDetector(
    model_path=BOUNDARY_MODEL_PATH,
    conf=0.35
)

tracker = BallTracker()

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "❌ Failed to open video"

fps = cap.get(cv2.CAP_PROP_FPS) or 30
ball_id = 0
boundary_fired = False

# FPS display
fps_frames, fps_time, display_fps = 0, time.time(), 0

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS calc
    fps_frames += 1
    if time.time() - fps_time >= 1.0:
        display_fps = fps_frames
        fps_frames = 0
        fps_time = time.time()

    # ---------------- BALL DETECTION ----------------
    predicted = tracker.predict() if tracker.initialized else None
    ball_detections = ball_detector.detect(frame, predicted)

    # ---------------- BOUNDARY DETECTION ----------------
    boundary_boxes = boundary_detector.detect(frame)

    # Draw boundary boxes (cyan)
    for bx1, by1, bx2, by2, _ in boundary_boxes:
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255,255,0), 2)

    # Draw ball detections
    for cx, cy, x1, y1, x2, y2, _ in ball_detections:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.circle(frame, (cx,cy), 3, (0,0,255), -1)

    # ---------------- TRACKER UPDATE ----------------
    if ball_detections:
        tracker.missed_frames = 0

        if not tracker.initialized:
            ball_id += 1
            cx, cy, *_ = ball_detections[0]
            tracker.update((cx, cy))
        else:
            match = associate_ball(ball_detections, predicted)
            if match:
                cx, cy, *_ = match
                tracker.update((cx, cy))
    else:
        tracker.missed_frames += 1

    # ---------------- BOUNDARY EVENT LOGIC ----------------
    if tracker.initialized and not boundary_fired:
        x, y = tracker.get_position()
        ball_box = (x-6, y-6, x+6, y+6)

        for boundary_box in boundary_boxes:
            if intersects(ball_box, boundary_box):
                boundary_type = classify_boundary(
                    tracker.get_speed_kmph(METERS_PER_PIXEL),
                    tracker.has_bounced
                )

                event = {
                    "ball_id": ball_id,
                    "event": "boundary",
                    "type": boundary_type,
                    "timestamp_sec": round(cap.get(cv2.CAP_PROP_POS_MSEC)/1000, 2)
                }

                events.append(event)
                boundary_fired = True

                print("🏏 EVENT:", event)

                cv2.putText(
                    frame,
                    boundary_type,
                    (frame.shape[1]//2 - 80, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.2,
                    (0,0,255),
                    4
                )
                break

    # ---------------- DELIVERY END ----------------
    if tracker.missed_frames > 15 and tracker.initialized:
        deliveries.append({
            "ball_id": ball_id,
            "max_speed_kmph": round(tracker.max_speed, 2),
            "pitch_type": tracker.pitch_type,
            "bounce_y_px": tracker.bounce_y
        })

        tracker.reset()
        boundary_fired = False

    # ---------------- DRAW TRACKER ----------------
    if tracker.initialized:
        x, y = tracker.get_position()
        cv2.circle(frame, (x,y), 6, (0,0,255), -1)

        scale = perspective_scale(y, frame.shape[0])
        speed = tracker.get_speed_kmph(METERS_PER_PIXEL, scale)

        cv2.putText(frame, f"Speed: {speed:.1f} km/h",
                    (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.putText(frame, f"Max: {tracker.max_speed:.1f} km/h",
                    (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        if tracker.detect_bounce():
            tracker.pitch_type = tracker.classify_pitch(frame.shape[0])

        if tracker.pitch_type:
            cv2.putText(frame, tracker.pitch_type,
                        (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

    cv2.putText(frame, f"FPS: {display_fps}",
                (20,140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("CricketSense | Ball + Boundary", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- SAVE JSON ----------------
cap.release()
cv2.destroyAllWindows()

with open(OUTPUT_JSON, "w") as f:
    json.dump(deliveries, f, indent=4)

with open(EVENTS_JSON, "w") as f:
    json.dump(events, f, indent=4)

print(f"✅ Saved {len(events)} events → {EVENTS_JSON}")
print(f"✅ Saved {len(deliveries)} deliveries → {OUTPUT_JSON}")
