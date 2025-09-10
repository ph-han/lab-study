import figure
from config import *
from frenet import *
from frenet_path import FrenetPath
from polynomial import Quartic, Quintic
from Car import Car

def generate_lateral_movement(di_0, di_1, di_2, dt_1, dt_2, tt): # current function is for high speed movement
    trajectories = []
    for dt_0 in np.arange(DT_0_MIN, DT_0_MAX + DT_0_STEP, DT_0_STEP):
        lat_traj = Quintic(di_0, di_1, di_2, dt_0, dt_1, dt_2, tt)
        
        if SHOW_LATERAL_PLOT:
            figure.show_lateral_traj(lat_traj, dt_0, tt)

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


def generate_velocity_keeping(si_0, si_1, si_2, st_1, st_2, tt): # current function is for velocity keeping

    trajectories = []
    for st_1 in np.arange(ST_1_MIN, ST_1_MAX + ST_1_STEP, ST_1_STEP):
        long_traj = Quartic(si_0, si_1, si_2, st_1, st_2, tt)

        if SHOW_LONGITUDINAL_PLOT:
            figure.show_longitudinal_traj(long_traj, st_1, tt)

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

def generate_follwing_merging_and_stopping(si_0, si_1, si_2, st_1, st_2, tt): # current function is for velocity keeping

    trajectories = []
    for st_0 in np.arange(si_0 + ST_0_MIN, min(STOP_POS, si_0 + ST_0_MAX + ST_0_STEP), ST_0_STEP):
        long_traj = Quintic(si_0, si_1, si_2, st_0, st_1, st_2, tt)

        if SHOW_LONGITUDINAL_PLOT:
            figure.show_longitudinal_traj(long_traj, st_1, tt)

        t_list = [t for t in np.arange(0.0, tt, GEN_T_STEP)]
        s0_list = [long_traj.get_position(t) for t in t_list]
        s1_list = [long_traj.get_velocity(t) for t in t_list]
        s2_list = [long_traj.get_acceleration(t) for t in t_list]
        sj_list = [long_traj.get_jerk(t) for t in t_list]

        for t in np.arange(tt, TT_MAX + GEN_T_STEP, GEN_T_STEP):
            t_list.append(t)
            s0_list.append(s0_list[-1])
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

def generate_frenet_trajectory(lat_state, lon_state, opt_d, velocity_keeping=True):
    frenet_paths = []

    opt_lon_cost = np.inf
    opt_lon_traj = None
    opt_lat_cost = np.inf
    opt_lat_traj = None
    for tt in np.arange(TT_MIN, TT_MAX + TT_STEP, TT_STEP):
        lat_traj_list = generate_lateral_movement(*lat_state, tt)
        if velocity_keeping:
            lon_traj_list = generate_velocity_keeping(*lon_state, tt)
        else:
            lon_traj_list = generate_follwing_merging_and_stopping(*lon_state, tt)

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

                d_diff = (lat_traj['d0'][-1] - opt_d)**2
                lat_cost = K_J * sum(np.power(lat_traj['jerk'], 2)) + K_T * 1 + K_D * d_diff
                fp.lat_cost = lat_cost

                if velocity_keeping:
                    v_diff = (lon_traj['s1'][-1] - DESIRED_SPEED) ** 2
                    lon_cost = K_J * sum(np.power(lon_traj['jerk'], 2)) + K_T * 1 + K_S * v_diff
                else:
                    s_diff = (lon_traj['s0'][-1] - STOP_POS)**2
                    print(s_diff, lon_traj['s0'][-1])
                    lon_cost = K_J * sum(np.power(lon_traj['jerk'], 2)) + K_T * tt + K_S * s_diff
                fp.lon_cost = lon_cost

                fp.tot_cost = K_LAT * fp.lat_cost + K_LON * fp.lon_cost
                frenet_paths.append(fp)

                if opt_lat_cost > lat_cost:
                    opt_lat_cost = lat_cost
                    opt_lat_traj = (lat_traj['d0'], lat_traj['t'])

                if opt_lon_cost > lon_cost:
                    opt_lon_cost = lon_cost
                    if velocity_keeping:
                        opt_lon_traj = (lon_traj['s1'], lat_traj['t'])
                    else:
                        opt_lon_traj = (lon_traj['s0'], lat_traj['t'])
    
    if SHOW_OPT_LATERAL_PLOT:
        figure.show_opt_lateral_traj(opt_lat_traj)

    if SHOW_OPT_LONGITUDINAL_PLOT:
        figure.show_opt_longitudinal_traj(opt_lon_traj)

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
            if fp.ds[i] > 1e-6:
                fp.kappa.append(yaw_diff / fp.ds[i])
            else:
                fp.kappa.append(0.0)

        if SHOW_ALL_FRENET_PATH:
            figure.show_frenet_path_in_world(fp.xlist, fp.ylist)
    
    return frenet_paths

def check_collision(path, obstacles):
    for obstacle in obstacles:
        obj = obstacle['object']

        if obstacle['type'] != 'vehicle':
            obs_r = 0.5 * np.hypot(obj.width, obj.height)
            for x, y in zip(path.xlist, path.ylist):
                d = np.hypot(obj.x - x, obj.y - y)
                if d < obs_r + Car.BUBBLE_R:
                    return True

        else:
            obs_rear_x = obj.x
            obs_rear_y = obj.y
            obs_front_x = obj.x + obj.WHEEL_BASE * np.cos(obj.yaw)
            obs_front_y = obj.y + obj.WHEEL_BASE * np.sin(obj.yaw)
            gap = 0

            for x, y, yaw in zip(path.xlist, path.ylist, path.yawlist):
                ego_rear_x, ego_rear_y = x, y
                ego_front_x = x + Car.WHEEL_BASE * np.cos(yaw)
                ego_front_y = y + Car.WHEEL_BASE * np.sin(yaw)

                d_rr = np.hypot(ego_rear_x - obs_rear_x, ego_rear_y - obs_rear_y)
                d_rf = np.hypot(ego_rear_x - obs_front_x, ego_rear_y - obs_front_y)
                d_fr = np.hypot(ego_front_x - obs_rear_x, ego_front_y - obs_rear_y)
                d_ff = np.hypot(ego_front_x - obs_front_x, ego_front_y - obs_front_y)
                # print(f"[DEBUG] d_rr: {d_rr}, d_rf: {d_rf}, d_fr: {d_fr}, d_ff: {d_ff}")

                if (d_rr + gap < Car.BUBBLE_R + obj.BUBBLE_R or
                    d_rf + gap < Car.BUBBLE_R + obj.BUBBLE_R or
                    d_fr + gap < Car.BUBBLE_R + obj.BUBBLE_R or
                    d_ff + gap < Car.BUBBLE_R + obj.BUBBLE_R):
                    return True
    return False

def is_in_road(path, boundaries, center_line_xlist, center_line_ylist):
    frenet_boundaries = [world2frenet(0, boundery, center_line_xlist, center_line_ylist)[1] for boundery in [5.25, -5.25]]
    road_d_min, road_d_max = np.min(frenet_boundaries), np.max(frenet_boundaries)
    for d in path.d0:
        if d <= road_d_min or road_d_max <= d:
            return False
    return True

def check_valid_path(paths, obs, road_boundaries, center_line_xlist, center_line_ylist):
    valid_paths = []
    for path in paths:
        acc_squared = [(abs(a_s**2 + a_d**2)) for (a_s, a_d) in zip(path.s2, path.d2)]
        if any([v > V_MAX for v in path.s1]):
            continue
        elif any([acc > ACC_MAX**2 for acc in acc_squared]):
            continue
        elif any([abs(kappa) > K_MAX for kappa in path.kappa]):
            continue
        elif obs and check_collision(path, obs):
            continue
        elif road_boundaries and not is_in_road(path, road_boundaries, center_line_xlist, center_line_ylist):
            continue
        
        valid_paths.append(path)
    if SHOW_VALID_PATH:
        for path in valid_paths:
            figure.show_frenet_valid_path_in_world(path.xlist, path.ylist)
    return valid_paths

def generate_opt_path(valid_paths):
    if not valid_paths:
        return []
    opt_path = min(valid_paths, key=lambda p: p.tot_cost)
    if SHOW_OPT_PATH:
        figure.show_opt_traj(opt_path)
    return opt_path


if __name__ == "__main__":
    center_line_xlist = np.linspace(10, 30, 100)
    center_line_ylist = 0.1 * (center_line_xlist**2)
    center_line_slist = [world2frenet(rx, ry, center_line_xlist, center_line_ylist)[0] for (rx, ry) in list(zip(center_line_xlist, center_line_ylist))]

    ego_x, ego_y = 10, 10
    frenet_s, frenet_d = world2frenet(ego_x, ego_y, center_line_xlist, center_line_ylist)
    fplist = generate_frenet_trajectory((frenet_d, 1, 0, 0, 0), (frenet_s, 18, 0, 16, 0), 2)
    fplist = frenet_paths_to_world(fplist, center_line_xlist, center_line_ylist, center_line_slist)
    valid_paths = check_valid_path(fplist, None, None, center_line_xlist, center_line_ylist)
    print(valid_paths)
    generate_opt_path(valid_paths)
    figure.show_coord_transformation(None, None, (center_line_xlist, center_line_ylist))
    figure.show()
