import numpy as np

class Quintic:
    def __init__(self, t, d_0, dd_0, ddd_0, d_1, dd_1, ddd_1):
        self.t = t
        self.c0 = d_0
        self.c1 = dd_0
        self.c2 = 2 * ddd_0

        self.m1 = np.array([
            [1, t, t]
        ])

class Quartic:
    def __init__(self):
        pass