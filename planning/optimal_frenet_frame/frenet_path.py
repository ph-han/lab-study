import numpy as np

class FrenetPath:
    def __init__(self):
        # time list
        self.t = []

        # lateral traj
        self.d0 = [] # position
        self.d1 = [] # velocity
        self.d2 = [] # acceleration
        self.dj = [] # jerk

        # longitudinal traj
        self.s0 = []  # position
        self.s1 = []  # velocity
        self.s2 = []  # acceleration
        self.sj = []  # jerk

        # cost
        self.lat_cost = 0
        self.lon_cost = 0
        self.tot_cost = 0

        # world_cood
        self.xlist = []
        self.ylist = []
        self.yawlist = []
        self.ds = []
        self.kappa = []