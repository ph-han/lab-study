from enum import Enum

class DrivingMode(Enum):
    VELOCITY_KEEPING = 1
    STOPPING = 2
    MERGING = 3
    FOLLOWING = 4

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
STOP_POS = 200

# longitunial final state speed config
ST_1_MIN = -6
ST_1_MAX = 6
ST_1_STEP = 2

# terminal time config
V_KEEP_TT_MIN = 2
V_KEEP_TT_MAX = 4
STOP_TT_MIN = 5
STOP_TT_MAX = 7
FOLLOWING_TT_MIN = 3
FOLLOWING_TT_MAX = 6
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
K_J   = 0.1
K_T   = 0.10
K_D   = 3.0
K_S   = 1.0
K_LAT = 1.0
K_LON = 1.0


DESIRED_LAT_POS = 0
FINAL_DESIRED_SPEED = 20
DESIRED_SPEED = 5
DESIRED_DISTACE = 16
V_MAX = 20
ACC_MAX = 4
K_MAX = 3