import numpy as np
import matplotlib.pyplot as plt

import config

def show_coord_transformation(ego, world, center_line):
    ego_x, ego_y = ego
    world_x, world_y = world
    center_line_xlist, center_line_ylist = center_line

    plt.plot(ego_x, ego_y, 'xb')
    plt.plot(world_x, world_y, 'xr')
    plt.plot(center_line_xlist, center_line_ylist)
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def show_lateral_traj(traj, dt_0, tt, is_end):
    tlist = np.linspace(0, tt, 500)
    dlist = [traj.get_position(t) for t in tlist]
    if tt != config.TT_MAX:
        tlist = tlist.tolist() + [config.TT_MAX]
        dlist.append(dt_0)
    plt.plot(tlist, dlist, '-', color="#1E6EF4")
    plt.xlabel('t [sec]')
    plt.ylabel('d [m]')
    if is_end:
        plt.grid(True)
        plt.show()

def show_longitudinal_traj(traj, st_1, tt, is_end):
    tlist = np.linspace(0, tt, 500)
    slist = [traj.get_velocity(t) for t in tlist]
    if tt != config.TT_MAX:
        tlist = tlist.tolist() + [config.TT_MAX]
        slist.append(st_1)
    plt.plot(tlist, slist, '-', color="#1E6EF4")
    plt.xlabel('t [sec]')
    plt.ylabel('dot s [m/s]')
    if is_end:
        plt.grid(True)
        plt.show()