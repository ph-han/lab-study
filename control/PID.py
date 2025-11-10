class PIDController:
    def __init__(self, KP, KI, KD):
        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.integral = 0.0
        self.prev_error = 0.0

    def control(self, curr, target, dt):
        error = target - curr
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        output = self.KP * error + self.KI * self.integral + self.KD * derivative
        self.prev_error = error
        return output