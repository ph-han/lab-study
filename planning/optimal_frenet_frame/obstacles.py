from math import cos, sin, tan, pi
import numpy as np
import matplotlib.pyplot as plt
import random

from IDM import IDMVehicle
from config import V_MAX, ACC_MAX, DESIRED_SPEED
from frenet import frenet2world

class StaticBox:
    # The unit is in meters.
    OVERALL_LENGTH = 3
    OVERALL_WIDTH = 1.890

    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.width = self.OVERALL_WIDTH
        self.height = self.OVERALL_LENGTH
        self.yaw = Car.pi_2_pi(yaw)


    @staticmethod
    def pi_2_pi(angle):
        return (angle + pi) % (2 * pi) - pi

class BaseCar:
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

    BUBBLE_R = (WHEEL_BASE) / 2

    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = self.pi_2_pi(yaw)

    @staticmethod
    def pi_2_pi(angle):
        return (angle + pi) % (2 * pi) - pi

class StaticCar(BaseCar):
    OVERALL_LENGTH = 6.9 # Override for StaticCar
    def __init__(self, x, y, yaw):
        super().__init__(x, y, yaw)
        self.width = self.OVERALL_WIDTH
        self.height = self.OVERALL_LENGTH


class Car(BaseCar):
    def __init__(self, x, y, yaw, s=0, d=0):
        self.x = x
        self.y = y
        self.s = s
        self.d = d
        self.yaw = Car.pi_2_pi(yaw)
        self.steer = 0.0

        v0 = random.randint(4, 15)
        self.idm = IDMVehicle(s, v=0, v0=v0, a_max=ACC_MAX, s0=self.OVERALL_LENGTH, length=self.OVERALL_LENGTH)

    def update_state(self, npcs, cxlist, cylist, cslist, dt=0.1):
        # s좌표 기준 정렬
        npcs.sort(key=lambda car: car['object'].s)

        # 가장 가까운 앞차 찾기
        leader = None
        min_gap = float('inf')
        for npc in npcs:
            if npc['object'].d == self.d and npc['object'].s > self.s:
                gap = npc['object'].s - self.s
                if gap < min_gap:
                    min_gap = gap
                    leader = npc['object'].idm

        self.s = self.idm.get_s()
        if self.s < 300:
            self.idm.update_acceleration(leader)
            self.idm.update_state()
        self.x, self.y, self.yaw = frenet2world(self.idm.get_s(), self.d, cxlist, cylist, cslist)
       

    @staticmethod
    def action(x, y, yaw, steer_angle, speed=1.0):
        x += speed * cos(yaw + steer_angle)
        y += speed * sin(yaw + steer_angle)
        yaw += Car.pi_2_pi(speed * tan(steer_angle) / Car.WHEEL_BASE)

        return x, y, yaw

    def display_arrow(self, ax, color="red"):
        arrow_length = self.WHEEL_BASE * 0.5
        dx = arrow_length * cos(self.yaw)
        dy = arrow_length * sin(self.yaw)

        ax.arrow(self.x, self.y, dx, dy,
                  head_width=0.3, head_length=0.4,
                  fc=color, ec=color)

    def display_wheels(self, steer_rot, body_rot, ax):
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


        ax.plot(front_left_wheel[0, :], front_left_wheel[1, :], "black")
        ax.plot(front_right_wheel[0, :], front_right_wheel[1, :], "black")
        ax.plot(rear_left_wheel[0, :], rear_left_wheel[1, :], "black")
        ax.plot(rear_right_wheel[0, :], rear_right_wheel[1, :], "black")

    def display_car(self, body_rot, ax):
        half_width = self.OVERALL_WIDTH / 2

        car = np.array([
            [-self.REAR_OVERHANG, -self.REAR_OVERHANG, self.WHEEL_BASE + self.FRONT_OVERHANG,
             self.WHEEL_BASE + self.FRONT_OVERHANG, -self.REAR_OVERHANG],
            [half_width, -half_width, -half_width, half_width, half_width]
        ], dtype=np.float32)

        car_pos = np.dot(body_rot, car) + np.array([[self.x], [self.y]])
        ax.plot(car_pos[0, :], car_pos[1, :], "black")

    def draw(self, ax):
        body_rot = np.array([
            [cos(self.yaw), -sin(self.yaw)],
            [sin(self.yaw), cos(self.yaw)]
        ], dtype=np.float32)

        steer_yaw = self.yaw + self.steer
        steer_rot = np.array([
            [cos(steer_yaw), -sin(steer_yaw)],
            [sin(steer_yaw), cos(steer_yaw)]
        ], dtype=np.float32)

        self.display_car(body_rot, ax)
        self.display_wheels(steer_rot, body_rot, ax)
        self.display_arrow(ax)
        circle1 = plt.Circle((self.x, self.y), self.BUBBLE_R, fill=False, color="blue")
        ax.add_artist(circle1)
        circle2 = plt.Circle((self.x + Car.WHEEL_BASE * np.cos(self.yaw), self.y + Car.WHEEL_BASE * np.sin(self.yaw)), self.BUBBLE_R, fill=False, color="blue")
        ax.add_artist(circle2)
        
