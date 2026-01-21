import math
from collections import deque

class BallTracker:
    def __init__(self, fps=30, window=5):
        self.fps = fps
        self.dt = 1.0 / fps

        # position history (for stable velocity)
        self.positions = deque(maxlen=window)

        self.initialized = False

        # velocity
        self.vx = 0.0
        self.vy = 0.0

        # speed
        self.max_speed = 0.0
        self.prev_speed = None

        # bounce / pitch
        self.has_bounced = False
        self.bounce_y = None
        self.pitch_type = None

        # lifecycle
        self.missed_frames = 0

    # ---------------- core ----------------
    def reset(self):
        self.__init__(self.fps)

    def update(self, pos):
        self.positions.append(pos)

        if len(self.positions) < 2:
            self.initialized = True
            return

        (x1, y1) = self.positions[-2]
        (x2, y2) = self.positions[-1]

        self.vx = (x2 - x1) * self.fps
        self.vy = (y2 - y1) * self.fps

        self.initialized = True

    def predict(self):
        if not self.initialized or len(self.positions) == 0:
            return None

        x, y = self.positions[-1]
        return (
            int(x + self.vx * self.dt),
            int(y + self.vy * self.dt)
        )

    def get_position(self):
        return self.positions[-1]

    # ---------------- speed ----------------
    def get_speed_kmph(self, meters_per_pixel, perspective_scale=1.0):
        speed_px = math.sqrt(self.vx**2 + self.vy**2)

        speed = speed_px * meters_per_pixel * 3.6 * perspective_scale

        # 🚨 HARD PHYSICAL LIMIT
        speed = min(speed, 160.0)

        return speed

    def update_max_speed(self, speed):
        if speed > self.max_speed:
            self.max_speed = speed
        return self.max_speed

    # ---------------- bounce ----------------
    def detect_bounce(self):
        if not self.initialized:
            return False

        speed = math.sqrt(self.vx**2 + self.vy**2)

        if self.prev_speed is None:
            self.prev_speed = speed
            return False

        speed_drop = self.prev_speed - speed

        print(f"speed_drop={speed_drop:.2f}, vy={self.vy:.2f}")

        if (
            not self.has_bounced and
            speed_drop > 15.0 and        # realistic drop
            abs(self.vy) < 200           # vertical flattening
        ):
            self.has_bounced = True
            self.bounce_y = self.positions[-1][1]
            print("✅ BOUNCE DETECTED")
            return True

        self.prev_speed = speed
        return False

    # ---------------- pitch ----------------
    def classify_pitch(self, frame_height):
        if self.bounce_y is None:
            return None

        y_norm = self.bounce_y / frame_height

        if y_norm > 0.75:
            return "YORKER"
        elif y_norm > 0.55:
            return "FULL"
        elif y_norm > 0.35:
            return "GOOD"
        else:
            return "SHORT"
