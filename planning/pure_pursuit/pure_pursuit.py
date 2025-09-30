import numpy as np
import matplotlib.pyplot as plt

def pid_accel_control(curr, target):
    '''
    TODO
    
    '''

def pursuit_steer_control(curr, traj):
    '''
    TODO
    curr = (position, yaw, velocity)
    traj = list
    
    1. waypoint를 찾아 (lookahead distance 만큼 떨어진)
    2. tan(theta) = L/R, R = Ld / (2 * sin(alpha)) (Ld: k_ld * velocity, alpha = atan2(y2 - y1, x2 - x1) - curr_yaw)
    3. delta = atan2(2 * L * sin(alpha), Ld) ==> steer angle
    4. limit steer angle
    '''


if __name__ == "__main__":
    center_line_xlist = np.linspace(10, 30, 100)
    center_line_ylist = 0.1 * (center_line_xlist**2)
    plt.plot(center_line_xlist, center_line_ylist)
    plt.axis('equal')
    plt.grid(True)
    plt.show()
