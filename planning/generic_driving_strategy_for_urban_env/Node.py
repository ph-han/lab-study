import numpy as np

class Node:
    def __init__(self, s=0, v=0, t=0, g=0.0, h=0.0, pidx=-1):
        self.s = s
        self.v = v
        self.t = t
        self.g = g
        self.h = h
        self.pidx = pidx

    def __sub__(self, other):
        return np.array([self.s - other.s, self.t - other.t, self.v - other.v])