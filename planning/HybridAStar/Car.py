'''
TODO
- [] 자동차 모델 정의
  - [] 차의 크기(가로, 세로), 조향각의 최대각 설정, 바퀴 위치
  - [] 차 움직임 함수
  - [] 차량 충돌 검사 함수
  - [] 차량 그려주는 함수
'''

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

    MAX_STEER = 0.6  # rad
    SPEED = 1.0

    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.steer = 0.0

    def check_collision(self):
        pass


    def action(self, steer_angle, speed=SPEED):
        self.steer = steer_angle
        self.x += speed * cos(self.yaw)
        self.y += speed * sin(self.yaw)
        self.yaw += speed * tan(steer_angle) / self.WHEEL_BASE
        # self.draw()
        return self.x, self.y, self.yaw

    def display_arrow(self):
        arrow_length = self.WHEEL_BASE * 0.5
        dx = arrow_length * cos(self.yaw)
        dy = arrow_length * sin(self.yaw)

        plt.arrow(self.x, self.y, dx, dy,
                  head_width=0.3, head_length=0.4,
                  fc='red', ec='red')

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

if __name__ == "__main__":
    ioniq5 = Car(0, 1, np.deg2rad(0))
    plt.xlim()
    for ang in np.linspace(-ioniq5.MAX_STEER, ioniq5.MAX_STEER):
        plt.cla()
        plt.xlim(-40, 40)
        plt.ylim(-40, 40)
        ioniq5.action(ang)
        plt.pause(0.01)

    plt.show()

