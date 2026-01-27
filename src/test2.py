import cv2
import time
import json
from collections import defaultdict

from src.detection.yolo_detector import YoloBallDetector
from src.tracking.tracker import BallTracker
from src.association.data_association import associate_ball

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\CricketSense-Ball\ball_test\weights\best.pt"
#MODEL_PATH = r"C:\CrickeSense-train\ball\runs\detect\ball_stump_test\weights\best.pt"
VIDEO_PATH = r"C:\CricketSense\data\samples\test3.mp4"
OUTPUT_VIDEO = "output_tracking.mp4"
OUTPUT_JSON = "ball_tracking.json"

METERS_PER_PIXEL = 20.12 / 520  # pitch calibration

# ---------------- PERSPECTIVE SCALE ----------------
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

fps_video = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_video,
    (w, h)
)

# ---------------- FPS COUNTER ----------------
fps_counter_time = time.time()
fps_counter_frames = 0
display_fps = 0

# ---------------- JSON STORAGE ----------------
json_output = {
    "video": VIDEO_PATH,
    "deliveries": []
}

current_delivery = {
    "frames": [],
    "release_speed": None,
    "max_speed": 0.0,
    "bounce_frame": None,
    "pitch_type": None
}

prev_time = time.time()
frame_id = 0

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # ---------- FPS ----------
    fps_counter_frames += 1
    if time.time() - fps_counter_time >= 1.0:
        display_fps = fps_counter_frames
        fps_counter_frames = 0
        fps_counter_time = time.time()

    # ---------- REAL DT ----------
    now = time.time()
    dt = max(0.001, now - prev_time)
    prev_time = now

    predicted = tracker.predict() if tracker.initialized else None
    detections = detector.detect(frame, predicted)

    # ---------- DRAW DETECTIONS ----------
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
            match = associate_ball(detections, predicted)
            if match:
                cx, cy, *_ = match
                tracker.update((cx, cy))
    else:
        tracker.missed_frames += 1

    # ---------- DELIVERY RESET ----------
    if tracker.missed_frames > 15:
        if current_delivery["frames"]:
            json_output["deliveries"].append(current_delivery)

        current_delivery = {
            "frames": [],
            "release_speed": None,
            "max_speed": 0.0,
            "bounce_frame": None,
            "pitch_type": None
        }
        tracker.reset()

    # ---------- VISUALS + LOGIC ----------
    if tracker.initialized:
        x, y = tracker.get_position()
        cv2.circle(frame, (x, y), 6, (0,0,255), -1)

        scale = perspective_scale(y, frame.shape[0])
        speed = tracker.get_speed_kmph(METERS_PER_PIXEL, scale)

        current_delivery["max_speed"] = max(
            current_delivery["max_speed"], speed
        )

        if tracker.release_speed is None:
            tracker.release_speed = speed
            current_delivery["release_speed"] = speed

        # ---------- TEXT ----------
        cv2.putText(frame, f"Speed: {speed:.1f} km/h",
                    (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.putText(frame, f"Release: {tracker.release_speed:.1f} km/h",
                    (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.putText(frame, f"Max: {current_delivery['max_speed']:.1f} km/h",
                    (20,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # ---------- BOUNCE ----------
        if tracker.detect_bounce():
            tracker.pitch_type = tracker.classify_pitch(frame.shape[0])
            current_delivery["bounce_frame"] = frame_id
            current_delivery["pitch_type"] = tracker.pitch_type
            print(f"🏏 BOUNCE → {tracker.pitch_type}")

        if tracker.pitch_type:
            cv2.putText(frame, tracker.pitch_type,
                        (20,130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

        # ---------- JSON FRAME ----------
        current_delivery["frames"].append({
            "frame": frame_id,
            "x": int(x),
            "y": int(y),
            "speed_kmph": round(speed, 2)
        })

    # ---------- FPS DISPLAY ----------
    cv2.putText(frame, f"FPS: {display_fps}",
                (20,170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # ---------- SHOW & SAVE ----------
    writer.write(frame)
    cv2.imshow("Cricket Ball Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------- FINAL SAVE ----------------
if current_delivery["frames"]:
    json_output["deliveries"].append(current_delivery)

with open(OUTPUT_JSON, "w") as f:
    json.dump(json_output, f, indent=2)

cap.release()
writer.release()
cv2.destroyAllWindows()

print("✅ Output video saved:", OUTPUT_VIDEO)
print("✅ JSON saved:", OUTPUT_JSON)
