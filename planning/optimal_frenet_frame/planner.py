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


def generate_longitudinal_movement(si_0, si_1, si_2, st_1, st_2, tt): # current function is for velocity keeping

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

        # figure.show_frenet_path_in_world(fp.xlist, fp.ylist)
    
    return frenet_paths

def check_collision(path, obstacles):
    for obstacle in obstacles:
        is_collision = False
        if obstacle['type'] != 'vehicle':
            obj = obstacle['object']
            obs_r = np.hypot(obj.width, obj.height) * 0.5

            distance_list = [np.hypot(obj.x - x, obj.y - y) for x, y in zip(path.xlist, path.ylist)]
            is_collision = any([d < obs_r + Car.BUBBLE_FRONT_R for d in distance_list])
        else: # vehicle
            obj = obstacle['object']
            rear_distance_list = [np.hypot(obj.x - x, obj.y - y) for x, y in zip(path.xlist, path.ylist)]
            rear_distance_list += [np.hypot(obj.x + obj.WHEEL_BASE * np.cos(obj.yaw) - x, obj.y + np.sin(obj.yaw) - y) for x, y in zip(path.xlist, path.ylist)]
            is_rear_collision = any([d < obj.BUBBLE_FRONT_R + Car.BUBBLE_REAR_R for d in rear_distance_list])
            front_distance_list = [np.hypot(obj.x - x + Car.WHEEL_BASE * np.cos(yaw), obj.y - y + np.sin(yaw)) for x, y, yaw in zip(path.xlist, path.ylist, path.yawlist)]
            front_distance_list += [np.hypot(obj.x + obj.WHEEL_BASE * np.cos(obj.yaw) - x + Car.WHEEL_BASE * np.cos(yaw), obj.y + np.sin(obj.yaw) - y + np.sin(yaw)) for x, y, yaw in zip(path.xlist, path.ylist, path.yawlist)]
            is_front_collision = any([d < obj.BUBBLE_REAR_R + Car.BUBBLE_FRONT_R for d in front_distance_list])

            # print(is_rear_collision, is_front_collision)
            is_collision = is_rear_collision or is_front_collision

        if is_collision:
            return True
    return False

def is_in_road(path, boundaries, center_line_xlist, center_line_ylist):

    # for boundery in boundaries:
    #     print(boundery[0])
    frenet_boundaries = [world2frenet(0, boundery[0], center_line_xlist, center_line_ylist)[1] for boundery in boundaries]
    road_d_min, road_d_max = np.min(frenet_boundaries), np.max(frenet_boundaries)
    # print(f"{road_d_max}. {road_d_min}")
    for d in path.d0:
        if d < road_d_min or road_d_max < d:
            return False
    return True

def check_valid_path(paths, obs, road_boundaries, center_line_xlist, center_line_ylist):
    valid_paths = []
    for path in paths:
        acc_squared = [(abs(a_s**2 + a_d**2)) for (a_s, a_d) in zip(path.s2, path.d2)]
        # if not is_in_road(path, road_boundaries, center_line_xlist, center_line_ylist):
        #     continue
        if any([v > V_MAX for v in path.s1]):
            continue
        elif any([acc > ACC_MAX**2 for acc in acc_squared]):
            continue
        elif any([abs(kappa) > K_MAX for kappa in path.kappa]):
            continue
        elif check_collision(path, obs):
            continue
        
        valid_paths.append(path)
    # for path in valid_paths:
    #     figure.show_frenet_valid_path_in_world(path.xlist, path.ylist)
    return valid_paths

def generate_opt_path(valid_paths):
    if not valid_paths:
        return []
    opt_path = min(valid_paths, key=lambda p: p.tot_cost)
    # figuare.show_opt_traj(opt_path)
    return opt_path


if __name__ == "__main__":
    from sim import generate_road
    # center_line_xlist = np.linspace(10, 30, 100)
    # center_line_ylist = 0.1 * (center_line_xlist**2)
    # center_line_slist = [world2frenet(rx, ry, center_line_xlist, center_line_ylist)[0] for (rx, ry) in list(zip(center_line_xlist, center_line_ylist))]
    road = generate_road(lane_num=3, lane_width=3.5, road_length=300, curved=True)
    ego_x, ego_y = 0, 1.75
    frenet_s, frenet_d = world2frenet(ego_x, ego_y, road['center_xlist'], road['center_ylist'])
    # world_x, world_y, _ = frenet2world(frenet_s, frenet_d, center_line_xlist, center_line_ylist, center_line_slist)
    print(f"frenet coordinate (s, d): ({frenet_s}, {frenet_d})")
    # print(f"world coordinate (x, y): ({world_x}, {world_y})")
    # # figuare.show_coord_transformation((ego_x, ego_y), (world_x, world_y), (center_line_xlist, center_line_ylist))
    # fplist = generate_frenet_trajectory((frenet_d, 1, 0, 0, 0), (frenet_s, 18, 0, 16, 0))
    # fplist = frenet_paths_to_world(fplist, center_line_xlist, center_line_ylist, center_line_slist)
    # valid_paths = check_valid_path(fplist, None, center_line_xlist, center_line_ylist, center_line_slist)
    # generate_opt_path(valid_paths)
    
    # figuare.show()
