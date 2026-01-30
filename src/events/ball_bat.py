import math

class BallBatContact:
    def __init__(self):
        self.triggered = False
        self.prev_speed = None

    def reset(self):
        self.triggered = False
        self.prev_speed = None

    def detect(self, ball_pos, bat_box, velocity):
        """
        ball_pos: (x, y)
        bat_box: (x1, y1, x2, y2)
        velocity: (vx, vy)
        """
        if self.triggered or bat_box is None:
            return False, 0.0

        bx, by = ball_pos
        x1, y1, x2, y2 = bat_box
        vx, vy = velocity

        # --- Distance ball → bat ---
        dx = max(x1 - bx, 0, bx - x2)
        dy = max(y1 - by, 0, by - y2)
        dist = math.sqrt(dx*dx + dy*dy)

        speed = math.sqrt(vx*vx + vy*vy)

        if self.prev_speed is None:
            self.prev_speed = speed
            return False, 0.0

        speed_drop = self.prev_speed - speed

        # --- Contact condition ---
        if dist < 15 and speed_drop > 0.2 * self.prev_speed:
            self.triggered = True
            confidence = min(1.0, 0.6 + speed_drop / self.prev_speed)
            return True, confidence

        self.prev_speed = speed
        return False, 0.0
