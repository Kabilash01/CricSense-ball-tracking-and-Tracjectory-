from ultralytics import YOLO

class YoloBallDetector:
    def __init__(self, model_path, conf=0.01, ball_class_id=0):
        self.model = YOLO(model_path)
        self.conf = conf
        self.ball_class_id = ball_class_id

    def detect(self, frame):
        detections = []
        results = self.model(frame, conf=self.conf, verbose=False)[0]

        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id != self.ball_class_id:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            conf = float(box.conf[0])

            detections.append((cx, cy, x1, y1, x2, y2, conf))

        return detections
