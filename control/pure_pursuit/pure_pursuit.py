import numpy as np
import matplotlib.pyplot as plt

from Car import Car

# KP = 1.0
# KD = 1.0
# KI = 1.0

# K_LD = 0.5


class PIDController:
    def __init__(self, KP, KI, KD):
        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.integral = 0.0
        self.prev_error = 0.0

    def control(self, curr, target, dt):
        error = target - curr
        print(f"error: {error}")
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        output = self.KP * error + self.KI * self.integral + self.KD * derivative
        print(f"output: {output}")
        self.prev_error = error
        return output


class PurePursuit:
    def __init__(self, wheel_base, max_steer, K_LD=0.5, min_ld=2.0):
        self.wheel_base = wheel_base
        self.max_steer = max_steer
        self.K_LD = K_LD
        self.min_ld = min_ld
        self.prev_wp_idx = 0

    def find_lookahead_point(self, curr, traj):
        ld = max(self.K_LD * curr[3], self.min_ld)
        search_window = 100
        start = max(0, self.prev_wp_idx)
        end = min(len(traj[0]), self.prev_wp_idx + search_window)

        dx = traj[0][start:end] - curr[0]
        dy = traj[1][start:end] - curr[1]
        d = np.hypot(dx, dy)

        idx = start + np.argmin(abs(d - ld))
        self.prev_wp_idx = idx
        return idx

    def find_lookahead_point_frenet(self, curr_s, curr_d, traj_s, traj_x, traj_y):
        ld = max(self.K_LD * curr_s, self.min_ld)  # 또는 curr 속도로 계산
        s_target = curr_s + ld
        idx = np.searchsorted(traj_s, s_target)
        if idx >= len(traj_s):
            idx = len(traj_s) - 1
        return idx

    def steer_control(self, curr, traj):
        ld_idx = self.find_lookahead_point(curr, traj)
        dx = traj[0][ld_idx] - curr[0]
        dy = traj[1][ld_idx] - curr[1]
        plt.plot(traj[0][ld_idx], traj[1][ld_idx], "rx")

        alpha = np.arctan2(dy, dx) - curr[2]
        delta = np.arctan2(2 * self.wheel_base * np.sin(alpha), np.hypot(dx, dy))
        delta = np.clip(delta, -self.max_steer, self.max_steer)
        return delta

    def steer_control_frenet(self, curr, traj):
        ld_idx = self.find_lookahead_point_frenet(curr.s, curr.d, traj.s, traj.xlist, traj.ylist)
        dx = traj.xlist[ld_idx] - curr[0]
        dy = traj.ylist[ld_idx] - curr[1]

        alpha = np.arctan2(dy, dx) - curr[2]
        delta = np.arctan2(2 * self.wheel_base * np.sin(alpha), np.hypot(dx, dy))
        delta = np.clip(delta, -self.max_steer, self.max_steer)
        return delta


if __name__ == "__main__":
    cx = np.linspace(0, 150, 200)
    cy = 5 * np.sin(0.3 * cx)

    car = Car(x=0.0, y=-3.0, yaw=np.deg2rad(10))

    pid = PIDController(KP=1.0, KI=0.1, KD=0.05)
    pp = PurePursuit(Car.WHEEL_BASE, Car.MAX_STEER, K_LD=0.5, min_ld=2.0)

    dt = 0.1
    sim_time = 20
    target_speed = 8.0  # [m/s]

    for _ in np.arange(0, sim_time, dt):
        plt.cla()
        plt.plot(cx, cy, "k--", label="Reference Path")
        plt.title(f"Steer: {car.steer:.2f}, Speed: {car.v:.2f}")
        car.draw()
        if pp.prev_wp_idx == len(cx) - 1:
            break
        delta = pp.steer_control((car.x, car.y, car.yaw, car.v), (cx, cy))
        accel = pid.control(car.v, target_speed, dt)
        car.action(delta, accel, dt)
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.pause(dt)
    plt.show()
