import numpy as np
import matplotlib.pyplot as plt

import config

def show_coord_transformation(ego, world, center_line):
    plt.figure(4)
    ego_x, ego_y = ego
    world_x, world_y = world
    center_line_xlist, center_line_ylist = center_line

    plt.plot(ego_x, ego_y, 'xb')
    plt.plot(world_x, world_y, 'xr')
    plt.plot(center_line_xlist, center_line_ylist)
    plt.axis('equal')
    plt.grid(True)

def show_lateral_traj(traj, dt_0, tt):
    plt.figure(2)
    tlist = np.linspace(0, tt, 500)
    dlist = [traj.get_position(t) for t in tlist]
    if tt != config.TT_MAX:
        tlist = tlist.tolist() + [config.TT_MAX]
        dlist.append(dt_0)
    plt.plot(tlist, dlist, '-', color="#1E6EF4")
    plt.title("lateral trajectories")
    plt.xlabel('t [sec]')
    plt.ylabel('d [m]')
    plt.grid(True)

def show_opt_lateral_traj(opt_traj):
    plt.figure(2)
    dlist, tlist = opt_traj
    plt.plot(tlist, dlist, '-', color="#6cf483", lw=3)
    plt.grid(True)

def show_longitudinal_traj(traj, st_1, tt):
    plt.figure(3)
    tlist = np.linspace(0, tt, 500)
    slist = [traj.get_velocity(t) for t in tlist]
    if tt != config.TT_MAX:
        tlist = tlist.tolist() + [config.TT_MAX]
        slist.append(st_1)
    plt.plot(tlist, slist, '-', color="#1E6EF4")
    plt.title("longitudinal trajectories")
    plt.xlabel('t [sec]')
    plt.ylabel(r"$\dot{s}$ m/s")
    plt.grid(True)

def show_opt_longitudinal_traj(opt_traj):
    plt.figure(3)
    s_d_list, tlist = opt_traj
    plt.plot(tlist, s_d_list, '-', color="#6cf483", lw=3)
    plt.grid(True)

def show_frenet_path_in_world(xlist, ylist):
    plt.figure(4)
    plt.plot(xlist, ylist, '-', color="#bbc5c6")
    plt.axis('equal')
    plt.grid(True)

def show_frenet_valid_path_in_world(xlist, ylist):
    plt.figure(4)
    plt.plot(xlist, ylist, '-', color="#1E6EF4")
    plt.axis('equal')
    plt.grid(True)

def show_opt_traj(opt_traj):
    plt.figure(4)
    plt.plot(opt_traj.xlist, opt_traj.ylist, '-', color="#6cf483", lw=2)
    plt.grid(True)

def show():
    plt.show()