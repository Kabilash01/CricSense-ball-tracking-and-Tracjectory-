import math
import time
from collections import deque

class BallTracker:
    def __init__(self, window=6):
        # stores (x, y, t)
        self.positions = deque(maxlen=window)

        self.initialized = False
        self.vx = 0.0
        self.vy = 0.0
        self.instant_speed = 0.0

        # speed states
        self.release_speed = None
        self.max_speed = 0.0

        # bounce / pitch
        self.has_bounced = False
        self.bounce_y = None
        self.pitch_type = None

        # lifecycle
        self.missed_frames = 0

    # ---------------- CORE ----------------
    def reset(self):
        self.__init__()

    def update(self, pos):
        x, y = pos
        t = time.time()
        self.positions.append((x, y, t))

        if len(self.positions) < 2:
            self.initialized = True
            return

        x1, y1, t1 = self.positions[-2]
        x2, y2, t2 = self.positions[-1]

        dt = max(t2 - t1, 1e-3)

        self.vx = (x2 - x1) / dt
        self.vy = (y2 - y1) / dt
        self.instant_speed = math.sqrt(self.vx**2 + self.vy**2)

        self.initialized = True

    def predict(self):
        if not self.initialized or not self.positions:
            return None

        x, y, _ = self.positions[-1]
        return int(x + self.vx * 0.03), int(y + self.vy * 0.03)

    def get_position(self):
        x, y, _ = self.positions[-1]
        return int(x), int(y)

    # ---------------- SPEED ----------------
    def get_speed_kmph(self, meters_per_pixel, perspective_scale=1.0):
        speed = self.instant_speed * meters_per_pixel * 3.6 * perspective_scale

        # physical safeguard (fast bowling ceiling)
        speed = min(speed, 155.0)

        # lock release speed once
        if self.release_speed is None and speed > 80:
            self.release_speed = speed

        self.max_speed = max(self.max_speed, speed)
        return speed

    # ---------------- BOUNCE ----------------
    def detect_bounce(self):
        if len(self.positions) < 3:
            return False

        p = list(self.positions)
        _, y1, _ = p[-3]
        _, y2, _ = p[-2]
        _, y3, _ = p[-1]

        # down → flatten → up
        if not self.has_bounced and (y2 > y1 and y3 < y2):
            self.has_bounced = True
            self.bounce_y = y2
            return True

        return False

    # ---------------- PITCH ----------------
    def classify_pitch(self, frame_height):
        if self.bounce_y is None:
            return None

        y = self.bounce_y / frame_height

        if y > 0.75:
            return "YORKER"
        elif y > 0.55:
            return "FULL"
        elif y > 0.35:
            return "GOOD"
        else:
            return "SHORT"
