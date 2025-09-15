# common config
GEN_T_STEP = 0.05

# lateral final state position config
DT_0_MIN = -3.5
DT_0_MAX = 3.5
DT_0_STEP = 3.5

# longitunial final state position config
ST_0_MIN = 0
ST_0_MAX = 20
ST_0_STEP = 2
GAP = 4
STOP_POS = 60

# longitunial final state speed config
ST_1_MIN = -3
ST_1_MAX = 3
ST_1_STEP = 1

# terminal time config
TT_MIN = 1
TT_MAX = 3
TT_STEP = 0.5

# plot config
SHOW_LATERAL_PLOT = True
SHOW_OPT_LATERAL_PLOT = True
SHOW_LONGITUDINAL_PLOT = True
SHOW_OPT_LONGITUDINAL_PLOT = True
SHOW_ALL_FRENET_PATH = True
SHOW_OPT_PATH = True
SHOW_VALID_PATH = True

# cost config
K_J   = 0.01
K_T   = 0.10
K_D   = 10.0
K_S   = 10.0
K_LAT = 1.0
K_LON = 1.0


DESIRED_LAT_POS = 0
DESIRED_SPEED = 20
DESIRED_DISTACE = 16
V_MAX = 20
ACC_MAX = 4
K_MAX = 3
