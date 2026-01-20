import numpy as np
from filterpy.kalman import KalmanFilter
import math

class BallTracker:
    def __init__(self, fps=30):
        self.fps = fps
        self.dt = 1.0 / fps

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # State: [x, y, vx, vy]
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

        self.kf.P *= 500.
        self.kf.R *= 1.0
        self.kf.Q *= 1.0
        self.max_speed_kmph = 0.0


        self.initialized = False
        
        self.prev_vy = None
        self.frame_count = 0
        self.last_bounce_frame = -100
        

    def update(self, measurement):
        x, y = measurement
        if not self.initialized:
            self.kf.x = np.array([[x], [y], [0], [0]])
            self.initialized = True
        else:
            self.kf.update(np.array([[x], [y]]))

    def predict(self):
        self.kf.predict()
        return self.get_position()

    def get_position(self):
        x, y = self.kf.x[0,0], self.kf.x[1,0]
        return int(x), int(y)

    def get_velocity(self):
        vx, vy = self.kf.x[2,0], self.kf.x[3,0]
        speed_px_per_sec = math.sqrt(vx**2 + vy**2)
        return vx, vy, speed_px_per_sec
    
    def detect_bounce(self):
        self.frame_count += 1

        vx, vy, speed = self.get_velocity()

        bounce = False

        if self.prev_vy is not None:
            # Vertical velocity inversion
            if self.prev_vy > 0 and vy < 0:
                # Cooldown to avoid double triggers
                if self.frame_count - self.last_bounce_frame > 10:
                    bounce = True
                    self.last_bounce_frame = self.frame_count

        self.prev_vy = vy
        return bounce
    
    def get_speed_kmph(self, meters_per_pixel):
        _, _, speed_px_per_sec = self.get_velocity()
        speed_mps = speed_px_per_sec * meters_per_pixel
        return speed_mps * 3.6
    
    def update_max_speed(self, meters_per_pixel):
        speed = self.get_speed_kmph(meters_per_pixel)
        if speed > self.max_speed_kmph:
            self.max_speed_kmph = speed
        return self.max_speed_kmph

