import numpy as np
import matplotlib.pyplot as plt

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
    if tt != 3.0:
        tlist = tlist.tolist() + [3.0]
        dlist.append(dt_0)
    plt.plot(tlist, dlist, '-', color="#1E6EF4")
    plt.xlabel('t [sec]')
    plt.ylabel('d [m]')
    if is_end:
        plt.grid(True)
        plt.show()