import numpy as np
import matplotlib.pyplot as plt

import figuare
from config import *
from frenet_path import FrenetPath
from polynomial import Quartic, Quintic


def find_closest_waypoint(curr_x, curr_y, center_line_xlist, center_line_ylist):
    xlist = np.array(center_line_xlist)
    ylist = np.array(center_line_ylist)

    distance_list = np.hypot(xlist - curr_x, ylist - curr_y)
    closest_wp = np.argmin(distance_list)
    return closest_wp

def get_next_waypoint(curr_x, curr_y, center_line_xlist, center_line_ylist):
    closest_wp = find_closest_waypoint(curr_x, curr_y, center_line_xlist, center_line_ylist)

    # loop until the next waypoint is ahead
    while True:
        if closest_wp == len(center_line_xlist) - 1:
            break
        traj_vec = np.array([center_line_xlist[closest_wp + 1] - center_line_xlist[closest_wp],
                            center_line_ylist[closest_wp + 1] - center_line_ylist[closest_wp]])
        ego_vec = np.array([curr_x - center_line_xlist[closest_wp], curr_y - center_line_ylist[closest_wp]])

        # check if waypoint is ahead of ego vehicle.
        is_waypoint_ahead = np.sign(np.dot(ego_vec, traj_vec)) 
        if is_waypoint_ahead < 0:
            break

        closest_wp += 1

    return closest_wp

def world2frenet(curr_x, curr_y, center_line_xlist, center_line_ylist):
    next_wp = get_next_waypoint(curr_x, curr_y, center_line_xlist, center_line_ylist)
    prev_wp = next_wp - 1


    ego_vec = np.array([
        curr_x - center_line_xlist[prev_wp], 
        curr_y - center_line_ylist[prev_wp]
    ])
    traj_vec = np.array([
        center_line_xlist[next_wp] - center_line_xlist[prev_wp],
        center_line_ylist[next_wp] - center_line_ylist[prev_wp]
    ])

    ego_proj_vec = ((ego_vec @ traj_vec) / (traj_vec @ traj_vec)) * traj_vec
    frenet_d_sign = np.sign(traj_vec[0] * ego_vec[1] - traj_vec[1] * ego_vec[0])    

    frenet_d = frenet_d_sign * np.hypot(ego_proj_vec[0] - ego_vec[0], ego_proj_vec[1] - ego_vec[1])

    frenet_s = 0
    for i in range(prev_wp):
        frenet_s += np.hypot(
            center_line_xlist[i + 1] - center_line_xlist[i],
            center_line_ylist[i + 1] - center_line_ylist[i]
        )

    frenet_s += np.hypot(ego_proj_vec[0], ego_proj_vec[1])

    return frenet_s, frenet_d

def frenet2world(curr_s, curr_d, center_line_xlist, center_line_ylist, center_line_slist):
    next_wp = 0

    while curr_s > center_line_slist[next_wp] and next_wp + 1 < len(center_line_slist):
        next_wp += 1

    wp = 0 if next_wp - 1 < 0 else next_wp - 1

    print(f"wp: {wp}, next_wp: {next_wp}")

    dx = center_line_xlist[next_wp] - center_line_xlist[wp]
    dy = center_line_ylist[next_wp] - center_line_ylist[wp]

    heading = np.arctan2(dy, dx)

    seg_s = curr_s - center_line_slist[wp]
    print(f"seg_s: {seg_s}")
    seg_vec = np.array([
        center_line_xlist[wp] + seg_s * np.cos(heading),
        center_line_ylist[wp] + seg_s * np.sin(heading)
        ])

    vertical_heading = heading + (np.pi / 2)
    world_x = seg_vec[0] + curr_d * np.cos(vertical_heading)
    world_y = seg_vec[1] + curr_d * np.sin(vertical_heading)

    return world_x, world_y, heading

def generate_lateral_movement(di_0, di_1, di_2, dt_1, dt_2):
    opt_lat_cost = np.inf
    opt_lat_traj = None
    for dt_0 in np.arange(DT_0_MIN, DT_0_MAX + DT_0_STEP, DT_0_STEP):
        for tt in np.arange(TT_MIN, TT_MAX + TT_STEP, TT_STEP):
            lat_traj = Quintic(di_0, di_1, di_2, dt_0, dt_1, dt_2, tt)

            if SHOW_LATERAL_PLOT:
                figuare.show_lateral_traj(lat_traj, dt_0, tt, not SHOW_OPT_LATERAL_PLOT and tt == TT_MAX and dt_0 == DT_0_MAX)

            t_list = [t for t in np.arange(0.0, tt, GEN_T_STEP)]
            d0_list = [lat_traj.get_position(t) for t in t_list]
            d1_list = [lat_traj.get_velocity(t) for t in t_list]
            d2_list = [lat_traj.get_acceleration(t) for t in t_list]
            dj_list = [lat_traj.get_jerk(t) for t in t_list]

            for t in np.arange(tt, TT_MAX + GEN_T_STEP, GEN_T_STEP):
                t_list.append(t)
                d0_list.append(d0_list[-1])
                d1_list.append(d1_list[-1])
                d2_list.append(d2_list[-1])
                dj_list.append(dj_list[-1])

            # print(f"{t}, {dt_0} : {d0_list[-1]}")
            # print(np.sum(np.power(dj_list, 2)))
            d_diff = (d0_list[-1] - DESIRED_LAT_POS)**2
            lat_cost = K_J * sum(np.power(dj_list, 2)) + K_T * 1 + K_D * d_diff

            if opt_lat_cost > lat_cost:
                opt_lat_cost = lat_cost
                opt_lat_traj = (d0_list, t_list)

    if SHOW_OPT_LATERAL_PLOT:
        figuare.show_opt_lateral_traj(opt_lat_traj, SHOW_OPT_LATERAL_PLOT)


def generate_longitudinal_movement(si_0, si_1, si_2, st_1, st_2):
    for st_1 in np.arange(ST_1_MIN, ST_1_MAX + ST_1_STEP, ST_1_STEP):
        for tt in np.arange(TT_MIN, TT_MAX + TT_STEP, TT_STEP):
            long_traj = Quartic(si_0, si_1, si_2, st_1, st_2, tt)

            if SHOW_LONGITUDINAL_PLOT:
                figuare.show_longitudinal_traj(long_traj, st_1, tt, tt==TT_MAX and st_1 == ST_1_MAX)

def generate_frenet_trajectory():
    frenet_paths = []

    pass

if __name__ == "__main__":
    center_line_xlist = np.linspace(10, 50, 100)
    center_line_ylist = 0.1 * (center_line_xlist**2)
    center_line_slist = [world2frenet(rx, ry, center_line_xlist, center_line_ylist)[0] for (rx, ry) in list(zip(center_line_xlist, center_line_ylist))]

    ego_x, ego_y = 10, 10
    frenet_s, frenet_d = world2frenet(ego_x, ego_y, center_line_xlist, center_line_ylist)
    generate_lateral_movement(frenet_d, 1, 0, 0, 0)
    generate_longitudinal_movement(frenet_s, 18, 0, 16, 0)
    world_x, world_y, heading = frenet2world(frenet_s, frenet_d, center_line_xlist, center_line_ylist, center_line_slist)
    print(f"frenet coordinate (s, d): ({frenet_s}, {frenet_d})")
    print(f"world coordinate (x, y): ({world_x}, {world_y})")
    # figuare.show_coord_transformation((ego_x, ego_y), (world_x, world_y), (center_line_xlist, center_line_ylist))
