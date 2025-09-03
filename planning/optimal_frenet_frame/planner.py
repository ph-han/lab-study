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

    # print(f"wp: {wp}, next_wp: {next_wp}")

    dx = center_line_xlist[next_wp] - center_line_xlist[wp]
    dy = center_line_ylist[next_wp] - center_line_ylist[wp]

    heading = np.arctan2(dy, dx)

    seg_s = curr_s - center_line_slist[wp]
    # print(f"seg_s: {seg_s}")
    seg_vec = np.array([
        center_line_xlist[wp] + seg_s * np.cos(heading),
        center_line_ylist[wp] + seg_s * np.sin(heading)
        ])

    vertical_heading = heading + (np.pi / 2)
    world_x = seg_vec[0] + curr_d * np.cos(vertical_heading)
    world_y = seg_vec[1] + curr_d * np.sin(vertical_heading)

    return world_x, world_y, heading

def generate_lateral_movement(di_0, di_1, di_2, dt_1, dt_2, tt): # current function is for high speed movement
    trajectories = []
    for dt_0 in np.arange(DT_0_MIN, DT_0_MAX + DT_0_STEP, DT_0_STEP):
        lat_traj = Quintic(di_0, di_1, di_2, dt_0, dt_1, dt_2, tt)
        
        if SHOW_LATERAL_PLOT:
            figuare.show_lateral_traj(lat_traj, dt_0, tt)

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
        
        generated_traj = {
            't': t_list,
            'd0': d0_list,
            'd1': d1_list,
            'd2': d2_list,
            'jerk': dj_list

        }
        trajectories.append(generated_traj)
    return trajectories


def generate_longitudinal_movement(si_0, si_1, si_2, st_1, st_2, tt): # current function is for velocity keeping

    trajectories = []
    for st_1 in np.arange(ST_1_MIN, ST_1_MAX + ST_1_STEP, ST_1_STEP):
        long_traj = Quartic(si_0, si_1, si_2, st_1, st_2, tt)

        if SHOW_LONGITUDINAL_PLOT:
            figuare.show_longitudinal_traj(long_traj, st_1, tt)

        t_list = [t for t in np.arange(0.0, tt, GEN_T_STEP)]
        s0_list = [long_traj.get_position(t) for t in t_list]
        s1_list = [long_traj.get_velocity(t) for t in t_list]
        s2_list = [long_traj.get_acceleration(t) for t in t_list]
        sj_list = [long_traj.get_jerk(t) for t in t_list]

        for t in np.arange(tt, TT_MAX + GEN_T_STEP, GEN_T_STEP):
            t_list.append(t)
            _s = s0_list[-1] + s1_list[-1] * GEN_T_STEP
            s0_list.append(_s)
            s1_list.append(s1_list[-1])
            s2_list.append(s2_list[-1])
            sj_list.append(sj_list[-1])

        generated_traj = {
            't': t_list,
            's0': s0_list,
            's1': s1_list,
            's2': s2_list,
            'jerk': sj_list

        }
        trajectories.append(generated_traj)
    return trajectories

def generate_frenet_trajectory(lat_state, lon_state):
    frenet_paths = []

    opt_lon_cost = np.inf
    opt_lon_traj = None
    opt_lat_cost = np.inf
    opt_lat_traj = None
    for tt in np.arange(TT_MIN, TT_MAX + TT_STEP, TT_STEP):
        lat_traj_list = generate_lateral_movement(*lat_state, tt)
        lon_traj_list = generate_longitudinal_movement(*lon_state, tt)

        for lat_traj in lat_traj_list:
            for lon_traj in lon_traj_list:
                fp = FrenetPath()

                fp.t = lat_traj['t']
                fp.d0 = lat_traj['d0']
                fp.d1 = lat_traj['d1']
                fp.d2 = lat_traj['d2']
                fp.dj = lat_traj['jerk']

                fp.s0 = lon_traj['s0']
                fp.s1 = lon_traj['s1']
                fp.s2 = lon_traj['s2']
                fp.sj = lon_traj['jerk']

                d_diff = (lat_traj['d0'][-1] - DESIRED_LAT_POS)**2
                lat_cost = K_J * sum(np.power(lat_traj['jerk'], 2)) + K_T * 1 + K_D * d_diff
                fp.lat_cost = lat_cost

                v_diff = (lon_traj['s1'][-1] - DESIRED_SPEED) ** 2
                lon_cost = K_J * sum(np.power(lon_traj['jerk'], 2)) + K_T * 1 + K_S * v_diff
                fp.lon_cost = lon_cost

                fp.tot_cost = K_LAT * fp.lat_cost + K_LON * fp.lon_cost
                frenet_paths.append(fp)

                if opt_lat_cost > lat_cost:
                    opt_lat_cost = lat_cost
                    opt_lat_traj = (lat_traj['d0'], lat_traj['t'])

                if opt_lon_cost > lon_cost:
                    opt_lon_cost = lon_cost
                    opt_lon_traj = (lon_traj['s1'], lat_traj['t'])
    
    if SHOW_OPT_LATERAL_PLOT:
        figuare.show_opt_lateral_traj(opt_lat_traj)

    if SHOW_OPT_LONGITUDINAL_PLOT:
        figuare.show_opt_longitudinal_traj(opt_lon_traj)

    return frenet_paths

def frenet_paths_to_world(frenet_paths, center_line_xlist, center_line_ylist, center_line_slist):
    for fp in frenet_paths:
        for s, d in zip(fp.s0, fp.d0):
            x, y, _ = frenet2world(s, d, center_line_xlist, center_line_ylist, center_line_slist)
            fp.xlist.append(x)
            fp.ylist.append(y)

        for i in range(len(fp.xlist) - 1):
            dx = fp.xlist[i + 1] - fp.xlist[i]
            dy = fp.ylist[i + 1] - fp.ylist[i]
            fp.yawlist.append(np.arctan2(dy, dx))
            fp.ds.append(np.hypot(dx, dy))

        fp.yawlist.append(fp.yawlist[-1])
        fp.ds.append(fp.ds[-1])

        for i in range(len(fp.yawlist) - 1):
            yaw_diff = fp.yawlist[i + 1] - fp.yawlist[i]
            yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
            fp.kappa.append(yaw_diff / fp.ds[i])

        figuare.show_frenet_path_in_world(fp.xlist, fp.ylist)
    
    return frenet_paths

def check_valid_path(paths, obs, center_line_xlist, center_line_ylist, center_line_slist):
    valid_paths = []
    for path in paths:
        acc_squared = [(abs(a_s**2 + a_d**2)) for (a_s, a_d) in zip(path.s2, path.d2)]
        if any([v > V_MAX for v in path.s1]):
            continue
        elif any([acc > ACC_MAX**2 for acc in acc_squared]):
            continue
        elif any([abs(kappa) > K_MAX for kappa in path.kappa]):
            continue
        # elif collision_check():
        #     continue
        
        valid_paths.append(path)
    for path in valid_paths:
        figuare.show_frenet_valid_path_in_world(path.xlist, path.ylist)
    return valid_paths


if __name__ == "__main__":
    center_line_xlist = np.linspace(10, 30, 100)
    center_line_ylist = 0.1 * (center_line_xlist**2)
    center_line_slist = [world2frenet(rx, ry, center_line_xlist, center_line_ylist)[0] for (rx, ry) in list(zip(center_line_xlist, center_line_ylist))]

    ego_x, ego_y = 10, 10
    frenet_s, frenet_d = world2frenet(ego_x, ego_y, center_line_xlist, center_line_ylist)
    fplist = generate_frenet_trajectory((frenet_d, 1, 0, 0, 0), (frenet_s, 18, 0, 16, 0))
    fplist = frenet_paths_to_world(fplist, center_line_xlist, center_line_ylist, center_line_slist)
    check_valid_path(fplist, None, center_line_xlist, center_line_ylist, center_line_slist)
    world_x, world_y, _ = frenet2world(frenet_s, frenet_d, center_line_xlist, center_line_ylist, center_line_slist)
    print(f"frenet coordinate (s, d): ({frenet_s}, {frenet_d})")
    print(f"world coordinate (x, y): ({world_x}, {world_y})")
    figuare.show_coord_transformation((ego_x, ego_y), (world_x, world_y), (center_line_xlist, center_line_ylist))
    figuare.show()
