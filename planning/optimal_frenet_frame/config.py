# common config
GEN_T_STEP = 0.05

# lateral final state position config
DT_0_MIN = -3.5
DT_0_MAX = 3.5
DT_0_STEP = 3.5

# longitunial final state position config
ST_0_MIN = 0
ST_0_MAX = 10
ST_0_STEP = 1
GAP = 2
STOP_POS = 150

# longitunial final state speed config
ST_1_MIN = 0
ST_1_MAX = 20
ST_1_STEP = 2

# terminal time config
TT_MIN = 1
TT_MAX = 3
TT_STEP = 0.5

# plot config
SHOW_LATERAL_PLOT = False
SHOW_OPT_LATERAL_PLOT = False
SHOW_LONGITUDINAL_PLOT = False
SHOW_OPT_LONGITUDINAL_PLOT = False
SHOW_ALL_FRENET_PATH = False
SHOW_OPT_PATH = False
SHOW_VALID_PATH = False

# cost config
K_J   = 0.01
K_T   = 0.10
K_D   = 10.0
K_S   = 10.0
K_LAT = 1.0
K_LON = 1.5


DESIRED_LAT_POS = 0
DESIRED_SPEED = 20
DESIRED_DISTACE = 16
V_MAX = 20
ACC_MAX = 20
K_MAX = 20
