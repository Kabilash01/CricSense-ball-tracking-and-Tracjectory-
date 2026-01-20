import numpy as np

class BallTracker:
    def __init__(self, fps=30):
        dt = 1 / fps

        self.x = np.zeros((4, 1))  # [x, y, vx, vy]
        self.P = np.eye(4) * 500

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        self.Q = np.eye(4) * 0.5
        self.R = np.eye(2) * 10

        self.initialized = False

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.get_position()

    def update(self, measurement):
        if not self.initialized:
            self.x[:2] = np.array(measurement).reshape(2, 1)
            self.initialized = True
            return self.get_position()

        z = np.array(measurement).reshape(2, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.get_position()

    def get_position(self):
        return int(self.x[0, 0]), int(self.x[1, 0])
