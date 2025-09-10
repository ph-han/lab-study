import numpy as np

class Quintic:
    def __init__(self, qi_0, qi_1, qi_2, qt_0, qt_1, qt_2, tt):
        self.qt = np.array([qt_0, qt_1, qt_2])

        self.c0 = qi_0
        self.c1 = qi_1
        self.c2 = qi_2 * 0.5

        self.c012 = np.array([self.c0, self.c1, self.c2])

        self.m1 = np.array([
            [1, tt, tt**2],
            [0, 1, 2*tt],
            [0, 0, 2]
        ])

        self.m2 = np.array([
            [tt**3, tt**4, tt**5],
            [3*(tt**2), 4*(tt**3), 5*(tt**4)],
            [6*tt, 12*(tt**2), 20*(tt**3)]
        ])

        self.c3, self.c4, self.c5 = np.linalg.inv(self.m2) @ (self.qt - (self.m1 @ self.c012))

    def get_position(self, t):
        return self.c5*(t**5) + self.c4*(t**4) + self.c3*(t**3) + self.c2*(t**2) + self.c1*t + self.c0

    def get_velocity(self, t):
        return 5*self.c5*(t**4) + 4*self.c4*(t**3) + 3*self.c3*(t**2) + 2*self.c2*t+ self.c1

    def get_acceleration(self, t):
        return 20 * self.c5 * (t ** 3) + 12 * self.c4 * (t ** 2) + 6 * self.c3 * t + 2 * self.c2

    def get_jerk(self, t):
        return 60*self.c5*(t**2) + 24*self.c4*t + 6*self.c3

class Quartic:
    def __init__(self, qi_0, qi_1, qi_2, qt_1, qt_2, tt):
        self.qt = np.array([qt_1, qt_2])

        self.c0 = qi_0
        self.c1 = qi_1
        self.c2 = qi_2 * 0.5

        self.c12 = np.array([self.c1, self.c2])

        self.m1 = np.array([
            [1, 2 * tt],
            [0, 2]
        ])

        self.m2 = np.array([
            [3 * (tt ** 2), 4 * (tt ** 3)],
            [6 * tt, 12 * (tt ** 2)]
        ])

        self.c3, self.c4 = np.linalg.inv(self.m2) @ (self.qt - (self.m1 @ self.c12))

    def get_position(self, t):
        return self.c4 * (t ** 4) + self.c3 * (t ** 3) + self.c2 * (t ** 2) + self.c1 * t + self.c0

    def get_velocity(self, t):
        return 4 * self.c4 * (t ** 3) + 3 * self.c3 * (t ** 2) + 2 * self.c2 * t + self.c1

    def get_acceleration(self, t):
        return 12 * self.c4 * (t ** 2) + 6 * self.c3 * t + 2 * self.c2

    def get_jerk(self, t):
        return 24 * self.c4 * t + 6 * self.c3

if __name__ == "__main__":
    quintic = Quintic(0, 0, 0, 3, 2, 1, 2)

    print(f"pos = {quintic.get_position(1)}, velo = {quintic.get_velocity(1)}, accel = {quintic.get_acceleration(1)}")