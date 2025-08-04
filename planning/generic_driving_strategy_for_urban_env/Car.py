from math import cos, sin, tan, pi
import numpy as np
import matplotlib.pyplot as plt


class Car:
    """
    Car

    The unit is in meters.
    This model is based on the IONIQ 5.

    """
    # The unit is in meters.
    OVERALL_LENGTH = 4.635
    FRONT_OVERHANG = 0.845
    REAR_OVERHANG = 0.79
    OVERALL_WIDTH = 1.890
    TRACK_WIDTH  = 1.647
    WHEEL_BASE = 3.0
    TR = 0.24
    TW = 0.48

    MAX_STEER = np.deg2rad(40)  # rad
    SPEED = 1.0

    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = Car.pi_2_pi(yaw)
        self.steer = 0.0

    @staticmethod
    def point_in_polygon(px, py, poly_x, poly_y):
        n = len(poly_x)
        inside = False
        j = n - 1
        for i in range(n):
            if ((poly_y[i] > py) != (poly_y[j] > py)) and \
                    (px < (poly_x[j] - poly_x[i]) * (py - poly_y[i]) / (poly_y[j] - poly_y[i]) + poly_x[i]):
                inside = not inside
            j = i
        return inside

    def is_collision(self, map, nx, ny, nyaw):
        half_width = self.OVERALL_WIDTH / 2

        car = np.array([
            [-self.REAR_OVERHANG, -self.REAR_OVERHANG, self.WHEEL_BASE + self.FRONT_OVERHANG,
             self.WHEEL_BASE + self.FRONT_OVERHANG, -self.REAR_OVERHANG],
            [half_width, -half_width, -half_width, half_width, half_width]
        ], dtype=np.float32)

        body_rot = np.array([
            [cos(nyaw), -sin(nyaw)],
            [sin(nyaw), cos(nyaw)]
        ], dtype=np.float32)

        car_pos = np.dot(body_rot, car) + np.array([[nx], [ny]])
        car_x = car_pos[0]
        car_y = car_pos[1]

        min_x = max(int(np.floor(car_x.min())), 0)
        max_x = min(int(np.ceil(car_x.max())) + 1, map.x_width)
        min_y = max(int(np.floor(car_y.min())), 0)
        max_y = min(int(np.ceil(car_y.max())) + 1, map.y_width)

        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                if map.obstacle_map[x][y]:
                    if Car.point_in_polygon(x, y, car_x, car_y):
                        return True

        return False

    @staticmethod
    def pi_2_pi(angle):
        return (angle + pi) % (2 * pi) - pi

    @staticmethod
    def action(x, y, yaw, steer_angle, speed=1.0):
        x += speed * cos(yaw + steer_angle)
        y += speed * sin(yaw + steer_angle)
        yaw += Car.pi_2_pi(speed * tan(steer_angle) / Car.WHEEL_BASE)

        return x, y, yaw

    def display_arrow(self, color="red"):
        arrow_length = self.WHEEL_BASE * 0.5
        dx = arrow_length * cos(self.yaw)
        dy = arrow_length * sin(self.yaw)

        plt.arrow(self.x, self.y, dx, dy,
                  head_width=0.3, head_length=0.4,
                  fc=color, ec=color)

    def display_wheels(self, steer_rot, body_rot):
        quarter_wheel_width = self.TW / 4
        wheel = np.array([
            [-self.TR, -self.TR, self.TR, self.TR, -self.TR],
            [quarter_wheel_width, -quarter_wheel_width, -quarter_wheel_width, quarter_wheel_width, quarter_wheel_width]
        ])

        front_wheel_dir = np.dot(steer_rot, wheel)
        rear_wheel_dir = np.dot(body_rot, wheel)

        front_left_wheel = front_wheel_dir \
                           + np.dot(body_rot, np.array([[self.WHEEL_BASE], [self.TRACK_WIDTH / 2]])) \
                           + np.array([[self.x], [self.y]])
        front_right_wheel = front_wheel_dir \
                            + np.dot(body_rot, np.array([[self.WHEEL_BASE], [-self.TRACK_WIDTH / 2]])) \
                           + np.array([[self.x], [self.y]])

        rear_left_wheel = rear_wheel_dir \
                          + np.dot(body_rot, np.array([[0.0], [self.TRACK_WIDTH / 2]])) \
                           + np.array([[self.x], [self.y]])
        rear_right_wheel = rear_wheel_dir \
                           + np.dot(body_rot, np.array([[0.0], [-self.TRACK_WIDTH / 2]])) \
                           + np.array([[self.x], [self.y]])


        plt.plot(front_left_wheel[0, :], front_left_wheel[1, :], "black")
        plt.plot(front_right_wheel[0, :], front_right_wheel[1, :], "black")
        plt.plot(rear_left_wheel[0, :], rear_left_wheel[1, :], "black")
        plt.plot(rear_right_wheel[0, :], rear_right_wheel[1, :], "black")

    def display_car(self, body_rot):
        half_width = self.OVERALL_WIDTH / 2

        car = np.array([
            [-self.REAR_OVERHANG, -self.REAR_OVERHANG, self.WHEEL_BASE + self.FRONT_OVERHANG,
             self.WHEEL_BASE + self.FRONT_OVERHANG, -self.REAR_OVERHANG],
            [half_width, -half_width, -half_width, half_width, half_width]
        ], dtype=np.float32)

        car_pos = np.dot(body_rot, car) + np.array([[self.x], [self.y]])
        plt.plot(car_pos[0, :], car_pos[1, :], "black")

    def draw(self):
        body_rot = np.array([
            [cos(self.yaw), -sin(self.yaw)],
            [sin(self.yaw), cos(self.yaw)]
        ], dtype=np.float32)

        steer_yaw = self.yaw + self.steer
        steer_rot = np.array([
            [cos(steer_yaw), -sin(steer_yaw)],
            [sin(steer_yaw), cos(steer_yaw)]
        ], dtype=np.float32)

        self.display_car(body_rot)
        self.display_wheels(steer_rot, body_rot)
        self.display_arrow()

from scipy.interpolate import CubicSpline

def poly_test(nx, ny, nyaw):
    half_width = Car.OVERALL_WIDTH / 2

    car = np.array([
        [-Car.REAR_OVERHANG, -Car.REAR_OVERHANG, Car.WHEEL_BASE + Car.FRONT_OVERHANG,
         Car.WHEEL_BASE + Car.FRONT_OVERHANG, -Car.REAR_OVERHANG],
        [half_width, -half_width, -half_width, half_width, half_width]
    ], dtype=np.float32)

    body_rot = np.array([
        [cos(nyaw), -sin(nyaw)],
        [sin(nyaw), cos(nyaw)]
    ], dtype=np.float32)

    car_pos = np.dot(body_rot, car) + np.array([[nx], [ny]])
    car_x = car_pos[0]
    car_y = car_pos[1]

    print(f"TEST {Car.point_in_polygon(10, 11, car_x, car_y)}")

if __name__ == "__main__":

    ioniq5 = Car(x=10.0, y=10.0, yaw=np.deg2rad(90))
    ioniq5.draw()
    plt.show()
    poly_test(ioniq5.x, ioniq5.y, ioniq5.yaw)
    # trajectory_x = [ioniq5.x]
    # trajectory_y = [ioniq5.y]
    # trajectory_yaw = [ioniq5.yaw]
    # trajectory_steer = [ioniq5.steer]
    #
    # for steer in np.arange(-ioniq5.MAX_STEER, ioniq5.MAX_STEER, np.deg2rad(5)):
    #     nx, ny, nyaw = Car.action(ioniq5.x, ioniq5.y, ioniq5.yaw, steer)
    #     ioniq5.x, ioniq5.y, ioniq5.yaw = nx, ny, nyaw
    #     trajectory_x.append(nx)
    #     trajectory_y.append(ny)
    #     trajectory_yaw.append(nyaw)
    #     trajectory_steer.append(steer)
    #
    # s = np.arange(len(trajectory_x))
    # spline_x = CubicSpline(s, trajectory_x)
    # spline_y = CubicSpline(s, trajectory_y)
    # spline_yaw = CubicSpline(s, trajectory_yaw)
    # spline_steer = CubicSpline(s, trajectory_steer)
    # s_fine = np.linspace(0, len(trajectory_x) - 1, 500)
    #
    # ioniq5 = Car(x=spline_x(0), y=spline_y(0), yaw=np.deg2rad(270))
    # plt.figure(figsize=(10, 10))
    #
    # plt.plot(spline_x(s_fine), spline_y(s_fine), "--", color="gray", label="planned path")
    #
    # for t in range(len(s_fine)):
    #     ioniq5.x = spline_x(s_fine[t])
    #     ioniq5.y = spline_y(s_fine[t])
    #     ioniq5.yaw = spline_yaw(s_fine[t])
    #     ioniq5.steer = spline_steer(s_fine[t])
    #
    #     plt.cla()
    #     plt.plot(spline_x(s_fine), spline_y(s_fine), "--", color="gray", label="planned path")
    #     plt.xlim(min(trajectory_x) - 2, max(trajectory_x) + 2)
    #     plt.ylim(min(trajectory_y) - 2, max(trajectory_y) + 2)
    #     plt.grid(True)
    #     plt.axis("equal")
    #     ioniq5.draw()
    #     plt.legend()
    #     plt.pause(0.01)
    #
    # plt.show()


