import numpy as np
from filterpy.kalman import KalmanFilter
import math
import time 

class BallTracker:
    def __init__(self, fps=30):
        self.fps = fps
        self.dt = 1.0 / fps

        # Kalman Filter: [x, y, vx, vy]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.zeros((4, 1))

        self.kf.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Motion-friendly tuning
        self.kf.P *= 500.0
        self.kf.R *= 1.0
        self.kf.Q *= 1.0

        self.initialized = False

        # ---- bounce & speed state ----
        self.prev_speed = None
        self.frame_count = 0
        self.last_bounce_frame = -100
        self.last_bounce_position = None
        self.missed_frames = 0
        self.bounce_y = None
        self.pitch_type = None
         


        # ---- per-delivery speed ----
        self.max_speed_kmph = 0.0

    # ---------------- core ----------------
    def reset_delivery(self):
        self.max_speed_kmph = 0.0
        self.prev_speed = None

    def update(self, measurement):
        x, y = measurement

        if not self.initialized:
            self.kf.x = np.array([[x], [y], [0], [0]])
            self.initialized = True
            self.reset_delivery()
        else:
            self.kf.update(np.array([[x], [y]]))

    def predict(self, dt):
        self.kf.F[0, 2] = dt
        self.kf.F[1, 3] = dt
        self.kf.predict()
        return self.get_position()


    def get_position(self):
        return int(self.kf.x[0, 0]), int(self.kf.x[1, 0])

    def get_velocity(self):
        vx, vy = self.kf.x[2, 0], self.kf.x[3, 0]
        speed_px_per_sec = math.sqrt(vx * vx + vy * vy)
        return vx, vy, speed_px_per_sec

    def get_speed_kmph(self, meters_per_pixel):
        _, _, speed_px = self.get_velocity()
        return speed_px * meters_per_pixel * 3.6

    def update_max_speed(self, meters_per_pixel):
        speed = self.get_speed_kmph(meters_per_pixel)
        if speed > self.max_speed_kmph:
            self.max_speed_kmph = speed
        return self.max_speed_kmph

    # ---------------- bounce detection ----------------
    def detect_bounce(self):
        self.frame_count += 1

        _, _, speed = self.get_velocity()

        # 🔒 FIRST FRAME GUARD (CRITICAL FIX)
        if self.prev_speed is None:
            self.prev_speed = speed
            return False

        speed_drop = self.prev_speed - speed
        bounce = False

        # Robust bounce condition
        if (
            speed_drop > 8.0 and
            speed < self.prev_speed and
            self.frame_count - self.last_bounce_frame > 12
        ):
            bounce = True
            self.last_bounce_frame = self.frame_count
            self.last_bounce_position = self.get_position()

        self.prev_speed = speed
        return bounce
    def classify_pitch(self, frame_height):
        """
        Classify pitch length based on bounce Y position
        """
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

    
    def reset(self):
        self.initialized = False
        self.x = None
        self.y = None
        self.vx = 0.0
        self.vy = 0.0

        self.prev_speed = None
        self.max_speed = 0.0
        self.missed_frames = 0     
        self.bounce_y = None
        self.pitch_type = None

