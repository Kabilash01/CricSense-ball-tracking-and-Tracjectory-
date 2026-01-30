import cv2
import time
import json
import os

from src.detection.yolo_detector import YoloBallDetector
from src.tracking.tracker import BallTracker
from src.association.data_association import associate_ball
from src.events.ball_bat import BallBatContact
from src.events.boundary import BoundaryDetector

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\CricketSense-Ball\ball_test\weights\best.pt"
VIDEO_PATH = r"C:\CricketSense\data\samples\test6.mp4"

OUTPUT_JSON = "ball_analysis.json"
EVENTS_JSON = "events.json"

METERS_PER_PIXEL = 18.5 / 520

def perspective_scale(y, h):
    y_norm = y / h
    return 1.0 + 0.6 * y_norm

# ---------------- GLOBAL EVENT BUFFERS ----------------
events = []                # all match events
current_ball_events = []   # per-delivery events

# ---------------- INIT ----------------
detector = YoloBallDetector(
    model_path=MODEL_PATH,
    conf=0.2,
    ball_class_id=0
)

tracker = BallTracker()
contact_detector = BallBatContact()
boundary_detector = BoundaryDetector()

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "❌ Failed to open video"

# FPS
fps_frames = 0
fps_time = time.time()
display_fps = 0
fps = cap.get(cv2.CAP_PROP_FPS) or 30

# Storage
deliveries = []
ball_id = 0
frame_idx = 0

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # FPS calc
    fps_frames += 1
    if time.time() - fps_time >= 1.0:
        display_fps = fps_frames
        fps_frames = 0
        fps_time = time.time()

    predicted = tracker.predict() if tracker.initialized else None
    detections = detector.detect(frame, predicted)

    # draw detections
    for det in detections:
        cx, cy, x1, y1, x2, y2, conf = det
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.circle(frame, (cx, cy), 3, (0,0,255), -1)

    # ---------- TRACKER UPDATE ----------
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

    # ---------- BALL–BAT CONTACT ----------
    if tracker.initialized:
        x, y = tracker.get_position()

        try:
            contact, conf = contact_detector.detect(
                ball_pos=(x, y),
                bat_box=batter_bbox,
                velocity=(tracker.vx, tracker.vy)
            )

            if contact:
                event = {
                    "ball_id": ball_id,
                    "event": "ball_bat_contact",
                    "frame": frame_idx,
                    "confidence": round(conf, 2),
                    "timestamp_sec": round(frame_idx / fps, 2)
                }

                current_ball_events.append(event)
                print("EVENT:", event)

        except NameError:
            pass

    # ---------- BOUNDARY DETECTION ----------
    if tracker.initialized:
        boundary = boundary_detector.detect(
            tracker.trajectory,
            frame.shape[0],
            tracker.has_bounced
        )

        if boundary:
            event = {
                "ball_id": ball_id,
                "event": "boundary",
                "type": boundary,
                "frame": frame_idx,
                "timestamp_sec": round(frame_idx / fps, 2)
            }

            current_ball_events.append(event)
            print("EVENT:", event)

            cv2.putText(
                frame,
                boundary,
                (frame.shape[1] // 2 - 60, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 0, 255),
                4
            )

    # ---------- DELIVERY END ----------
    if tracker.missed_frames > 15 and tracker.initialized:
        ball_id += 1

        deliveries.append({
            "ball_id": ball_id,
            "release_speed_kmph": round(tracker.release_speed or 0, 2),
            "max_speed_kmph": round(tracker.max_speed, 2),
            "pitch_type": tracker.pitch_type,
            "bounce_y_px": tracker.bounce_y,
            "release_point": {
                "x": tracker.release_point[0],
                "y": tracker.release_point[1]
            }
        })

        # 🔥 FLUSH EVENTS FOR THIS BALL
        if current_ball_events:
            events.extend(current_ball_events)
            current_ball_events = []

        tracker.reset()
        boundary_detector.reset()

    # ---------- DRAW ----------
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

        if tracker.pitch_type:
            cv2.putText(frame, tracker.pitch_type,
                        (20,130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

    cv2.putText(frame, f"FPS: {display_fps}",
                (20,170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Cricket Ball Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------- SAVE JSON (AFTER VIDEO ENDS) ----------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(deliveries, f, indent=4)

with open(EVENTS_JSON, "w") as f:
    json.dump(events, f, indent=4)

print(f"✅ Saved {len(events)} events to {EVENTS_JSON}")
print(f"✅ Saved {len(deliveries)} deliveries to {OUTPUT_JSON}")
