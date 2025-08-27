import numpy as np
import matplotlib.pyplot as plt

def find_closest_waypoint(curr_x, curr_y, center_line_xlist, center_line_ylist):
    xlist = np.array(center_line_xlist)
    ylist = np.array(center_line_ylist)

    distance_list = np.hypot(xlist - curr_x, ylist - curr_y)
    closest_wp = np.argmin(distance_list)
    return closest_wp

def get_next_waypoint(curr_x, curr_y, center_line_xlist, center_line_ylist):
    closest_wp = find_closest_waypoint(curr_x, curr_y, center_line_xlist, center_line_ylist)
    return closest_wp

def world2frenet(curr_x, curr_y, center_line_xlist, center_line_ylist):
    next_wp = get_next_waypoint(curr_x, curr_y, center_line_xlist, center_line_ylist)
    prev_wp = next_wp - 1

    nx, ny = center_line_xlist[next_wp], center_line_ylist[next_wp]

    ego_vec = [nx - curr_x, ny - curr_y, 0]
    traj_vec = [center_line_xlist[next_wp + 1] - nx, center_line_ylist[next_wp + 1] - ny]

    
    return frenet_s, frenet_d

def frenet2world(curr_s, curr_d):
    pass

if __name__ == "__main__":
    pass
