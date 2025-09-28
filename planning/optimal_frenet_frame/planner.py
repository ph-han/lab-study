import bisect
import figure
from config import *
from frenet import *
from frenet_path import FrenetPath
from polynomial import Quartic, Quintic
from obstacles import Car

def generate_lateral_movement(di_0, di_1, di_2, dt_1, dt_2, tt, tt_max, dt_0_candidates):
    trajectories = []
    for dt_0 in dt_0_candidates:
        lat_traj = Quintic(di_0, di_1, di_2, dt_0, dt_1, dt_2, tt)
        
        if SHOW_LATERAL_PLOT:
            figure.show_lateral_traj(lat_traj, dt_0, tt, tt_max)

        t_list = [t for t in np.arange(0.0, tt, GEN_T_STEP)]
        d0_list = [lat_traj.get_position(t) for t in t_list]
        d1_list = [lat_traj.get_velocity(t) for t in t_list]
        d2_list = [lat_traj.get_acceleration(t) for t in t_list]
        dj_list = [lat_traj.get_jerk(t) for t in t_list]

        for t in np.arange(tt, tt_max + GEN_T_STEP, GEN_T_STEP):
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


def generate_longitudinal_movement_using_quartic(si_0, si_1, si_2, st_2, tt, tt_max, st_1_candidates):

    trajectories = []
    for st_1 in st_1_candidates:
        long_traj = Quartic(si_0, si_1, si_2, st_1, st_2, tt)

        if SHOW_LONGITUDINAL_PLOT:
            figure.show_longitudinal_traj(long_traj, st_1, tt, tt_max)

        t_list = [t for t in np.arange(0.0, tt, GEN_T_STEP)]
        s0_list = [long_traj.get_position(t) for t in t_list]
        s1_list = [long_traj.get_velocity(t) for t in t_list]
        s2_list = [long_traj.get_acceleration(t) for t in t_list]
        sj_list = [long_traj.get_jerk(t) for t in t_list]

        for t in np.arange(tt, tt_max + GEN_T_STEP, GEN_T_STEP):
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

def generate_longitudinal_movement_using_quintic(si_0, si_1, si_2, st_1, st_2, tt, tt_max, st_0_candidates):

    trajectories = []

    for st_0 in st_0_candidates:
        long_traj = Quintic(si_0, si_1, si_2, st_0, st_1, st_2, tt)

        if SHOW_LONGITUDINAL_PLOT:
            figure.show_longitudinal_traj(long_traj, st_0, tt, tt_max, False)

        t_list = [t for t in np.arange(0.0, tt, GEN_T_STEP)]
        s0_list = [long_traj.get_position(t) for t in t_list]
        s1_list = [long_traj.get_velocity(t) for t in t_list]
        s2_list = [long_traj.get_acceleration(t) for t in t_list]
        sj_list = [long_traj.get_jerk(t) for t in t_list]

        for t in np.arange(tt, tt_max + GEN_T_STEP, GEN_T_STEP):
            t_list.append(t)
            s0_list.append(s0_list[-1])
            s1_list.append(s1_list[-1])
            s2_list.append(s2_list[-1])
            sj_list.append(sj_list[-1])

        generated_traj = {
            't': t_list,
            's0': np.round(s0_list, 3).tolist(),
            's1': np.round(s1_list, 3).tolist(),
            's2': np.round(s2_list, 3).tolist(),
            'jerk': np.round(sj_list, 3).tolist()

        }
        trajectories.append(generated_traj)
    return trajectories

def generate_velocity_keeping_trajectories_in_frenet(lat_state, lon_state, opt_d, desired_speed):
    frenet_paths = []

    opt_lon_cost = np.inf
    opt_lon_traj = None
    opt_lat_cost = np.inf
    opt_lat_traj = None

    v_keep_tt_max = V_KEEP_TT_MAX
    v_keep_tt_min = V_KEEP_TT_MIN

    if desired_speed == 0:
        desired_speed_list = sorted(set(np.arange( 5 * (lon_state[1] // 5), 0, -5)) | {desired_speed})
        if lon_state[1] < 3:
            v_keep_tt_min = 0.5
            v_keep_tt_max = 3.0 
    else:
        desired_speed_list = sorted(set(np.arange(5, desired_speed + 5, 5)) | {desired_speed})
    curr_desired_speed_idx = min(bisect.bisect_left(desired_speed_list, lon_state[1]), len(desired_speed_list) - 1)
    curr_desired_speed = desired_speed_list[curr_desired_speed_idx]
    dt_0_candidates = [DT_0_MIN, 0, DT_0_MAX]
    st_1_candidates = np.arange(curr_desired_speed + ST_1_MIN, curr_desired_speed + ST_1_MAX + ST_1_STEP, ST_1_STEP)
    for tt in np.arange(v_keep_tt_min, v_keep_tt_max + TT_STEP, TT_STEP):
        lat_traj_list = generate_lateral_movement(*lat_state, tt, v_keep_tt_max, dt_0_candidates)
        lon_traj_list = generate_longitudinal_movement_using_quartic(*lon_state, tt, v_keep_tt_max, st_1_candidates)

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

                d_diff = (lat_traj['d0'][-1] - opt_d) ** 2
                lat_cost = K_J * sum(np.power(lat_traj['jerk'], 2)) + K_T * tt + K_D * d_diff
                fp.lat_cost = lat_cost

                v_diff = (lon_traj['s1'][-1] - desired_speed) ** 2
                lon_cost = K_J * sum(np.power(lon_traj['jerk'], 2)) + K_T * tt + K_S * v_diff

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
        figure.show_opt_lateral_traj(opt_lat_traj)

    if SHOW_OPT_LONGITUDINAL_PLOT:
        figure.show_opt_longitudinal_traj(opt_lon_traj)

    return frenet_paths

def generate_stopping_trajectories_in_frenet(lat_state, lon_state, opt_d):
    frenet_paths = []

    dt_0_candidates = np.arange(DT_0_MIN, DT_0_MAX + DT_0_STEP, DT_0_STEP)
    st_0_candidates = [STOP_POS - GAP]

    remaining_s = (STOP_POS - GAP) - lon_state[0]

    dynamic_tt_max = STOP_TT_MAX
    dynamic_tt_min = STOP_TT_MIN
    if remaining_s <= 5:
        dynamic_tt_max = 3.0
        dynamic_tt_min = 0.5
    elif remaining_s <= 15:
        dynamic_tt_max = 6.0
        dynamic_tt_min = 3.0
    elif remaining_s <= 25:
        dynamic_tt_max = 8.0
        dynamic_tt_min = 5.0
    elif lon_state[1] >= 12:
        dynamic_tt_max = 14.0
        dynamic_tt_min = 10.0

    for tt in np.arange(dynamic_tt_min, dynamic_tt_max + TT_STEP, TT_STEP):
        lat_traj_list = generate_lateral_movement(*lat_state, tt, dynamic_tt_max, dt_0_candidates)
        lon_traj_list = generate_longitudinal_movement_using_quintic(*lon_state, tt, dynamic_tt_max, st_0_candidates)

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
                lat_cost = K_J * sum(np.power(lat_traj['jerk'], 2)) + K_T * tt + K_D * d_diff
                fp.lat_cost = lat_cost

                s_diff = (lon_traj['s0'][-1] - (STOP_POS - GAP))**2
                lon_cost = K_J * sum(np.power(lon_traj['jerk'], 2)) + K_T * tt + 50 * K_S * s_diff

                fp.lon_cost = lon_cost

                fp.tot_cost = K_LAT * fp.lat_cost + K_LON * fp.lon_cost
                frenet_paths.append(fp)

    return frenet_paths

def generate_following_trajectories_in_frenet(lat_state, lon_state, lv, opt_d):
    frenet_paths = []

    dt_0_candidates = [DT_0_MIN, 0, DT_0_MAX]
    safe_d = 5
    st_0 = max(lv.s - (safe_d + 1.5 * lv.idm.v), lv.s - 6)
    print(f"follwing safe_d = {st_0}")
    st_0_candidates = [st_0 - 5, st_0, st_0 + 5]

    for tt in np.arange(FOLLOWING_TT_MIN, FOLLOWING_TT_MAX + TT_STEP, TT_STEP):
        lat_traj_list = generate_lateral_movement(*lat_state, tt, FOLLOWING_TT_MAX, dt_0_candidates)
        lon_traj_list = generate_longitudinal_movement_using_quintic(*lon_state, tt, FOLLOWING_TT_MAX, st_0_candidates)

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
                lat_cost = K_J * sum(np.power(lat_traj['jerk'], 2)) + K_T * tt + K_D * d_diff
                fp.lat_cost = lat_cost

                s_diff = (lon_traj['s0'][-1] - st_0)**2
                lon_cost = K_J * sum(np.power(lon_traj['jerk'], 2)) + K_T * tt + K_S * s_diff

                fp.lon_cost = lon_cost

                fp.tot_cost = K_LAT * fp.lat_cost + K_LON * fp.lon_cost
                frenet_paths.append(fp)

    return frenet_paths

def frenet_paths_to_world(frenet_paths, center_line_xlist, center_line_ylist, center_line_slist):
    for fp in frenet_paths:
        center_headings = []
        for s, d in zip(fp.s0, fp.d0):
            x, y, heading = frenet2world(s, d, center_line_xlist, center_line_ylist, center_line_slist)
            fp.xlist.append(x)
            fp.ylist.append(y)
            center_headings.append(heading)

        if len(fp.xlist) < 2:
            fallback_yaw = center_headings[-1] if center_headings else 0.0
            fp.yawlist.append(fallback_yaw)
            fp.ds.append(0.0)
        else:
            for i in range(len(fp.xlist) - 1):
                dx = fp.xlist[i + 1] - fp.xlist[i]
                dy = fp.ylist[i + 1] - fp.ylist[i]
                step = np.hypot(dx, dy)
                if step < 1e-4:
                    # fall back to the center-line heading when the step is nearly zero to avoid yaw spikes at stop
                    fallback_idx = min(i + 1, len(center_headings) - 1)
                    yaw = center_headings[fallback_idx]
                else:
                    yaw = np.arctan2(dy, dx)
                fp.yawlist.append(yaw)
                fp.ds.append(step)

        if fp.yawlist:
            fp.yawlist.append(fp.yawlist[-1])
        else:
            fallback_yaw = center_headings[-1] if center_headings else 0.0
            fp.yawlist.append(fallback_yaw)

        if fp.ds:
            fp.ds.append(fp.ds[-1])
        else:
            fp.ds.append(0.0)

        # fp.kappa = [0.0]
        # for i in range(1, len(fp.yawlist) - 1):
        #     yaw_diff = fp.yawlist[i + 1] - fp.yawlist[i - 1]
        #     yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
        #     ds_sum = fp.ds[i] + fp.ds[i - 1]
        #     if ds_sum > 1e-6:
        #         fp.kappa.append(yaw_diff / ds_sum)
        #     else:
        #         fp.kappa.append(0.0)
        # fp.kappa.append(0.0)

        # 1차 미분 (속도)
        xd = np.gradient(fp.xlist, GEN_T_STEP)
        yd = np.gradient(fp.ylist, GEN_T_STEP)

        # 2차 미분 (가속도)
        xdd = np.gradient(xd, GEN_T_STEP)
        ydd = np.gradient(yd, GEN_T_STEP)

        # 곡률 계산
        num = xd * ydd - yd * xdd
        den = (xd**2 + yd**2)**1.5
        fp.kappa = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-4)

        if SHOW_ALL_FRENET_PATH:
            figure.show_frenet_path_in_world(fp.xlist, fp.ylist)
    
    return frenet_paths

def check_collision(path, obstacles):
    if any(np.array(path.d0) > 4.25) or any(np.array(path.d0) < -4.25):
        return True
    
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

                if (d_rr + gap < Car.BUBBLE_R + obj.BUBBLE_R or
                    d_rf + gap < Car.BUBBLE_R + obj.BUBBLE_R or
                    d_fr + gap < Car.BUBBLE_R + obj.BUBBLE_R or
                    d_ff + gap < Car.BUBBLE_R + obj.BUBBLE_R):
                    return True
    return False

def is_forward_motion(path, eps=1e-4):

    s = np.array(path.s0, dtype=float)

    # 너무 짧으면 그냥 True
    if len(s) <= 1:
        return True
    
    v = np.array(path.s1, dtype=float)
    if np.all(np.abs(v) < 0.05):   # 전체가 거의 0 속도
        return True

    # s(t)가 단조 증가하는지 검사
    if np.any(np.diff(s) < -eps):
        return False

    v = np.array(path.s1, dtype=float)

    # 작은 값은 0으로 처리
    v[(v >= 0) & (v < 0.1)] = 0.0

    # 음수 속도만 진짜 후진으로 판단
    if np.any(v < -eps):
        return False

    return True


def is_in_road(path, boundaries, center_line_xlist, center_line_ylist):
    frenet_boundaries = [world2frenet(0, boundery, center_line_xlist, center_line_ylist)[1] for boundery in [5.25, -5.25]]
    road_d_min, road_d_max = np.min(frenet_boundaries), np.max(frenet_boundaries)
    for d in path.d0:
        if d <= road_d_min or road_d_max <= d:
            return False
    return True

def check_valid_path(paths, obs, road_boundaries, center_line_xlist, center_line_ylist):
    valid_paths = []
    cv, ca, ck, cb, cc = 0, 0, 0, 0, 0
    for path in paths:
        acc_squared = [(a_s**2 + a_d**2) for (a_s, a_d) in zip(path.s2, path.d2)]
        if any([v > V_MAX for v in path.s1]):
            cv += 1
            continue
        elif any([acc > ACC_MAX**2 for acc in acc_squared]):
            ca += 1
            continue
        elif any([abs(kappa) > K_MAX for kappa in path.kappa]):
            ck += 1
            # print(path.ds)
            # print(path.kappa)
            continue
        elif not is_forward_motion(path):
            cb += 1
            continue
        elif obs and check_collision(path, obs):
            cc += 1
            continue
        valid_paths.append(path)
    if SHOW_VALID_PATH:
        for path in valid_paths:
            figure.show_frenet_valid_path_in_world(path.xlist, path.ylist)
    print(f"tot: {len(paths)} | cv : {cv}, ca : {ca}, ck : {ck}, cb: {cb}, cc : {cc}")
    
    return valid_paths

def generate_opt_path(valid_paths):
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
    fplist = generate_stopping_trajectories_in_frenet((frenet_d, 1, 0, 0, 0), (frenet_s, 18, 0, 16, 0), 2)
    fplist = frenet_paths_to_world(fplist, center_line_xlist, center_line_ylist, center_line_slist)
    valid_paths = check_valid_path(fplist, None, None, center_line_xlist, center_line_ylist)
    generate_opt_path(valid_paths)
    figure.show_coord_transformation(None, None, (center_line_xlist, center_line_ylist))
    figure.show()
