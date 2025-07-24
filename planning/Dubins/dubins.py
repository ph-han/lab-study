import numpy as np
import math
import matplotlib.pyplot as plt
from Car import Car

def _calc_angle(start_pos, end_pos):
    det = start_pos[0] * end_pos[1] - start_pos[1] * end_pos[0]
    dot = np.dot(start_pos, end_pos)
    return np.arctan2(det, dot)

def _LRL(di, r_turn):
    _, _c1, _, _c2 = di
    dx = _c2['x'] - _c1['x']
    dy = _c2['y'] - _c1['y']
    distance = np.hypot(dx, dy)
    # plt.plot([_c1['x'], _c2['x']], [_c1['y'], _c2['y']], color='blue', linewidth=1)

    p1 = px1, py1 = (_c1['x'], _c1['y'])
    p2 = px2, py2 = (_c2['x'], _c2['y'])

    if (distance / (4*r_turn) > 1.0):
        return None

    theta = _calc_angle([1, 0], [dx, dy]) - math.acos(distance / (4*r_turn))

    _c3 = {
        'x': px1 + (2*r_turn)*math.cos(theta),
        'y': py1 + (2*r_turn)*math.sin(theta)
    }
    distance_of_c3_to_c2 = np.hypot(_c3['x'] - _c2['x'], _c3['y'] - _c2['y'])
    if (2*r_turn + 1e-4 < distance_of_c3_to_c2 or distance_of_c3_to_c2 < 2*r_turn - + 1e-4):
        return None
    # plt.plot([_c1['x'], _c3['x']], [_c1['y'], _c3['y']], color='blue', linewidth=1)
    p3 = [_c3['x'], _c3['y']]
    # plt.plot(p3[0], p3[1], 'xg')
    # circle = plt.Circle((p3[0], p3[1]), r_turn, edgecolor='cyan', facecolor='none')

    # ax = plt.gca()
    # ax.add_patch(circle)

    v2 = np.array(p1) - np.array(p3)
    v2 = (v2 / np.hypot(v2[0], v2[1])) * r_turn
    v3 = np.array(p2) - np.array(p3)
    v3 = (v3 / np.hypot(v3[0], v3[1])) * r_turn

    pt1 = p3 + v2
    pt2 = p3 + v3

    plt.plot(pt1[0], pt1[1], 'xb')
    plt.plot(pt2[0], pt2[1], 'xr')

    alpha_start_pos = [_c1['state'][0] - _c1['x'], _c1['state'][1] - _c1['y']]
    alpha_end_pos = [pt1[0] - _c1['x'], pt1[1] - _c1['y']]

    beta_start_pos = [pt1[0] - _c3['x'], pt1[1] - _c3['y']]
    beta_end_pos = [pt2[0] - _c3['x'], pt2[1] - _c3['y']]

    gamma_start_pos = [pt2[0] - _c2['x'], pt2[1] - _c2['y']]
    gamma_end_pos = [((_c2['state'][0] - _c2['x'])),
                     ((_c2['state'][1] - _c2['y']))]
    # angle
    alpha = _calc_angle(alpha_start_pos, alpha_end_pos)
    a = 2 * np.pi + alpha if alpha < 0 else alpha
    beta = _calc_angle(beta_start_pos, beta_end_pos)
    b = -2 * np.pi + beta if beta > 0 else beta
    gamma = _calc_angle(gamma_start_pos, gamma_end_pos)
    g = 2 * np.pi + gamma if gamma < 0 else gamma
    return [['l', a, _c1], ['r', b, _c3], ['l', g, _c2]]

def _RLR(di, r_turn):
    _c1, _, _c2, _ = di
    dx = _c2['x'] - _c1['x']
    dy = _c2['y'] - _c1['y']
    distance = np.hypot(dx, dy)
    # plt.plot([_c1['x'], _c2['x']], [_c1['y'], _c2['y']], color='blue', linewidth=1)

    p1 = px1, py1 = (_c1['x'], _c1['y'])
    p2 = px2, py2 = (_c2['x'], _c2['y'])
    if distance / (4 * r_turn) > 1.0:
        return None
    # print("asf: ", distance / (4 * r_turn) )
    theta = _calc_angle([1, 0], [dx, dy]) - math.acos(distance / (4 * r_turn))
    # print("theta ok")
    _c3 = {
        'x': px1 + (2 * r_turn) * math.cos(-theta),
        'y': py1 + (2 * r_turn) * math.sin(-theta)
    }
    distance_of_c3_to_c2 = np.hypot(_c3['x'] - _c2['x'], _c3['y'] - _c2['y'])
    if (2 * r_turn + 1e-4 < distance_of_c3_to_c2 or distance_of_c3_to_c2 < 2 * r_turn - + 1e-4):
        return None
    # print("asdf")
    plt.plot([_c1['x'], _c3['x']], [_c1['y'], _c3['y']], color='blue', linewidth=1)
    p3 = [_c3['x'], _c3['y']]
    # plt.plot(p3[0], p3[1], 'xg')
    # circle = plt.Circle((p3[0], p3[1]), r_turn, edgecolor='cyan', facecolor='none')
    #
    # ax = plt.gca()
    # ax.add_patch(circle)

    v2 = np.array(p1) - np.array(p3)
    v2 = (v2 / np.hypot(v2[0], v2[1])) * r_turn
    v3 = np.array(p2) - np.array(p3)
    v3 = (v3 / np.hypot(v3[0], v3[1])) * r_turn

    pt1 = p3 + v2
    pt2 = p3 + v3

    plt.plot(pt1[0], pt1[1], 'xb')
    plt.plot(pt2[0], pt2[1], 'xr')

    alpha_start_pos = [_c1['state'][0] - _c1['x'], _c1['state'][1] - _c1['y']]
    alpha_end_pos = [pt1[0] - _c1['x'], pt1[1] - _c1['y']]

    beta_start_pos = [pt1[0] - _c3['x'], pt1[1] - _c3['y']]
    beta_end_pos = [pt2[0] - _c3['x'], pt2[1] - _c3['y']]

    gamma_start_pos = [pt2[0] - _c2['x'], pt2[1] - _c2['y']]
    gamma_end_pos = [((_c2['state'][0] - _c2['x'])),
                     ((_c2['state'][1] - _c2['y']))]
    # angle
    alpha = _calc_angle(alpha_start_pos, alpha_end_pos)
    a = -2 * np.pi + alpha if alpha > 0 else alpha
    beta = _calc_angle(beta_start_pos, beta_end_pos)
    b = 2 * np.pi + beta if beta < 0 else beta
    gamma = _calc_angle(gamma_start_pos, gamma_end_pos)
    g = -2 * np.pi + gamma if gamma > 0 else gamma
    return [['r', a, _c1], ['l', b, _c3], ['r', g, _c2]]

def _LSL(di, r_turn):
    _, _c1, _, _c2 = di
    dx = _c2['x'] - _c1['x']
    dy = _c2['y'] - _c1['y']
    distance = np.hypot(dx, dy)
    # plt.plot([_c1['x'], _c2['x']], [_c1['y'], _c2['y']], color='blue', linewidth=1)

    hat_v = {'x': dx / distance, 'y': dy / distance}
    hat_n = {'x': hat_v['y'], 'y': -hat_v['x']}

    pot1 = {'x': _c1['x'] + hat_n['x'] * r_turn, 'y': _c1['y'] + hat_n['y'] * r_turn}
    pot2 = {'x': _c2['x'] + hat_n['x'] * r_turn, 'y': _c2['y'] + hat_n['y'] * r_turn}
    # print(f"pot2 : {pot2['x']}, {pot2['y']}")
    #
    # plt.plot(pot1['x'], pot1['y'], '.b')
    # plt.plot(pot2['x'], pot2['y'], '.r')
    # plt.plot([pot1['x'], pot2['x']], [pot1['y'], pot2['y']], color='green')

    alpha_start_pos = [_c1['state'][0] - _c1['x'], _c1['state'][1] - _c1['y']]
    alpha_end_pos = [pot1['x'] - _c1['x'], pot1['y'] - _c1['y']]

    gamma_start_pos = [pot2['x'] - _c2['x'], pot2['y'] - _c2['y']]
    gamma_end_pos = [((_c2['state'][0] - _c2['x'])),
                     ((_c2['state'][1] - _c2['y']))]

    alpha = _calc_angle(alpha_start_pos, alpha_end_pos)
    a = 2*np.pi + alpha if alpha < 0 else alpha
    d = np.hypot(pot2['x'] - pot1['x'], pot2['y'] - pot1['y'])
    gamma = _calc_angle(gamma_start_pos, gamma_end_pos)
    g = 2 * np.pi + gamma if gamma < 0 else gamma

    return [['l', a, _c1], ['s', d], ['l', g, _c2]]

def _LSR(di, r_turn):
    _, _c1, _c2, _ = di
    dx = _c2['x'] - _c1['x']
    dy = _c2['y'] - _c1['y']
    distance = np.hypot(dx, dy)
    # plt.plot([_c1['x'], _c2['x']], [_c1['y'], _c2['y']], color='blue', linewidth=1)
    c = -(2 * r_turn) / distance
    if 1 - c ** 2 < 0:
        return None
    hat_v = {'x': dx / distance, 'y': dy / distance}
    hat_n = {'x': hat_v['x'] * c - hat_v['y'] * math.sqrt(1 - c**2),
             'y': hat_v['x'] * math.sqrt(1 - c**2) + hat_v['y'] * c}

    pot1 = {'x': _c1['x'] - hat_n['x'] * r_turn, 'y': _c1['y'] - hat_n['y'] * r_turn}
    pot2 = {'x': _c2['x'] + hat_n['x'] * r_turn, 'y': _c2['y'] + hat_n['y'] * r_turn}

    #
    # plt.plot(pot1['x'], pot1['y'], '.b')
    # plt.plot(pot2['x'], pot2['y'], '.r')
    # plt.plot([pot1['x'], pot2['x']], [pot1['y'], pot2['y']], color='green')

    alpha_start_pos = [_c1['state'][0] - _c1['x'], _c1['state'][1] - _c1['y']]
    alpha_end_pos = [pot1['x'] - _c1['x'], pot1['y'] - _c1['y']]

    gamma_start_pos = [pot2['x'] - _c2['x'], pot2['y'] - _c2['y']]
    gamma_end_pos = [((_c2['state'][0] - _c2['x'])),
                     ((_c2['state'][1] - _c2['y']))]

    alpha = _calc_angle(alpha_start_pos, alpha_end_pos)
    a = 2 * np.pi + alpha if alpha < 0 else alpha
    d = np.hypot(pot2['x'] - pot1['x'], pot2['y'] - pot1['y'])
    gamma = _calc_angle(gamma_start_pos, gamma_end_pos)
    g = -2 * np.pi + gamma if gamma > 0 else gamma

    return [['l', a, _c1], ['s', d], ['r', g, _c2]]

def _RSL(di, r_turn):
    _c1, _, _, _c2 = di
    dx = _c2['x'] - _c1['x']
    dy = _c2['y'] - _c1['y']
    distance = np.hypot(dx, dy)
    # plt.plot([_c1['x'], _c2['x']], [_c1['y'], _c2['y']], color='blue', linewidth=1)
    c = (2 * r_turn) / distance
    if 1 - c ** 2 < 0:
        return None
    hat_v = {'x': dx / distance, 'y': dy / distance}
    hat_n = {'x': hat_v['x'] * c - hat_v['y'] * math.sqrt(1 - c ** 2),
             'y': hat_v['x'] * math.sqrt(1 - c ** 2) + hat_v['y'] * c}

    pot1 = {'x': _c1['x'] + hat_n['x'] * r_turn, 'y': _c1['y'] + hat_n['y'] * r_turn}
    pot2 = {'x': _c2['x'] - hat_n['x'] * r_turn, 'y': _c2['y'] - hat_n['y'] * r_turn}

    # print(pot1)
    # print(pot2)

    # plt.plot(pot1['x'], pot1['y'], '.b')
    # plt.plot(pot2['x'], pot2['y'], '.r')
    # plt.plot([pot1['x'], pot2['x']], [pot1['y'], pot2['y']], color='green')

    alpha_start_pos = [_c1['state'][0] - _c1['x'], _c1['state'][1] - _c1['y']]
    alpha_end_pos = [pot1['x'] - _c1['x'], pot1['y'] - _c1['y']]

    gamma_start_pos = [pot2['x'] - _c2['x'], pot2['y'] - _c2['y']]
    gamma_end_pos = [((_c2['state'][0] - _c2['x'])),
                     ((_c2['state'][1] - _c2['y']))]

    alpha = _calc_angle(alpha_start_pos, alpha_end_pos)
    a = -2 * np.pi + alpha if alpha > 0 else alpha
    d = np.hypot(pot2['x'] - pot1['x'], pot2['y'] - pot1['y'])
    gamma = _calc_angle(gamma_start_pos, gamma_end_pos)
    g = 2 * np.pi + gamma if gamma < 0 else gamma

    return [['r', a, _c1], ['s', d], ['l', g, _c2]]

def _RSR(di, r_turn):
    _c1, _, _c2, _ = di
    dx = _c2['x'] - _c1['x']
    dy = _c2['y'] - _c1['y']
    distance = np.hypot(dx, dy)
    # plt.plot([_c1['x'], _c2['x']], [_c1['y'], _c2['y']], color='blue', linewidth=1)

    hat_v = {'x': dx / distance, 'y': dy / distance}
    hat_n = {'x': -hat_v['y'], 'y': hat_v['x']}

    pot1 = {'x': _c1['x'] + hat_n['x'] * r_turn, 'y': _c1['y'] + hat_n['y'] * r_turn}
    pot2 = {'x': _c2['x'] + hat_n['x'] * r_turn, 'y': _c2['y'] + hat_n['y'] * r_turn}
    # print(f"pot2 : {pot2['x']}, {pot2['y']}")
    #
    # plt.plot(pot1['x'], pot1['y'], '.b')
    # plt.plot(pot2['x'], pot2['y'], '.r')
    # plt.plot([pot1['x'], pot2['x']], [pot1['y'], pot2['y']], color='green')

    alpha_start_pos = [_c1['state'][0] - _c1['x'], _c1['state'][1] - _c1['y']]
    alpha_end_pos = [pot1['x'] - _c1['x'], pot1['y'] - _c1['y']]

    gamma_start_pos = [pot2['x'] - _c2['x'], pot2['y'] - _c2['y']]
    gamma_end_pos = [((_c2['state'][0] - _c2['x'])),
                     ((_c2['state'][1] - _c2['y']))]

    alpha =  _calc_angle(alpha_start_pos, alpha_end_pos)
    a = -2*np.pi + alpha if alpha > 0 else alpha
    d = np.hypot(pot2['x'] - pot1['x'], pot2['y'] - pot1['y'])
    gamma = _calc_angle(gamma_start_pos, gamma_end_pos)
    g = -2*np.pi + gamma if gamma > 0 else gamma

    return [['r', a, _c1], ['s', d], ['r', g, _c2]]

def _dubins_in(curr_state, goal_state, r_turn):
    cx, cy, cyaw = curr_state
    gx, gy, gyaw = goal_state

    # curr_state right, left circle point direction
    cr_dir, cl_dir = cyaw - np.deg2rad(90), cyaw + np.deg2rad(90)

    # goal_state right, left circle point direction
    gr_dir, gl_dir = gyaw - np.deg2rad(90), gyaw + np.deg2rad(90)

    cr = {'x': cx + r_turn * math.cos(cr_dir), 'y': cy + r_turn * math.sin(cr_dir), 'state': curr_state}
    cl = {'x': cx + r_turn * math.cos(cl_dir), 'y': cy + r_turn * math.sin(cl_dir), 'state': curr_state}

    gr = {'x': gx + r_turn * math.cos(gr_dir), 'y': gy + r_turn * math.sin(gr_dir), 'state': goal_state}
    gl = {'x': gx + r_turn * math.cos(gl_dir), 'y': gy + r_turn * math.sin(gl_dir), 'state': goal_state}

    return cr, cl, gr, gl

def _calc_cost(path, r_turn):
    cost = 0
    for p in path:
        if p[0] == 's':
            cost += p[1]
        else:
            cost += abs(np.rad2deg(p[1])) * r_turn
    return cost

def _get_opt_path(paths, r_turn):
    opt_cost = _calc_cost(paths[0][:], r_turn)
    opt_idx = 0
    num_of_possible_path = len(paths)

    for idx in range(1, num_of_possible_path):
        curr_cost = _calc_cost(paths[idx][:], r_turn)
        if curr_cost >= opt_cost:
            continue
        opt_cost = curr_cost
        opt_idx = idx

    return paths[opt_idx][:], opt_cost

def rotate_point(x, y, c, angle_rad):
    cx, cy = c['x'], c['y']
    dx = x - cx
    dy = y - cy

    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    x_new = cos_theta * dx - sin_theta * dy + cx
    y_new = sin_theta * dx + cos_theta * dy + cy
    return x_new, y_new

def gen_path(start_pos, dubins_params):
    path_x = [start_pos[0]]
    path_y = [start_pos[1]]
    path_yaw = [start_pos[2]]
    yaw = start_pos[2]

    for p in dubins_params:
        if p[0] == 's':
            print(p[1])
            ds = p[1] / 300
            base_x, base_y = path_x[-1], path_y[-1]
            for d in np.arange(1, p[1], ds):
                path_x.append(base_x + d * np.cos(yaw))
                path_y.append(base_y + d * np.sin(yaw))
                path_yaw.append(yaw)
            path_x.append(base_x + p[1] * np.cos(yaw))
            path_y.append(base_y + p[1] * np.sin(yaw))
            path_yaw.append(yaw)
        else:
            step = np.deg2rad(1)
            goal_angle = round(np.rad2deg(np.pi * 2 + p[1] if p[0] == 'l' and p[1] < 0 else p[1]))
            base_x, base_y = path_x[-1], path_y[-1]
            print(f"goal_angle = {goal_angle} {p[0]}")
            for y in np.arange(0, goal_angle + step, step if p[0] == 'l' else -step):
                nx, ny = rotate_point(base_x, base_y, p[2], np.deg2rad(y))
                path_x.append(nx)
                path_y.append(ny)
                yaw = np.arctan2(path_y[-1] - path_y[-2], path_x[-1] - path_x[-2])
                path_yaw.append(yaw)
        # plt.plot(path_x, path_y, '-r')
        # plt.axis('equal')
        # plt.pause(1)

    return path_x, path_y, path_yaw

def dubins_path(curr_state, goal_state, r_turn):
    di = _dubins_in(curr_state, goal_state, r_turn)

    dubins_words = [_LSL, _LSR, _RSL, _RSR, _RLR, _LRL]
    color_set = ['r', 'b', 'g', 'y', 'c', 'm']
    paths = []
    for idx, word in enumerate(dubins_words):
        path = word(di, r_turn)
        if path is None:
            print("is none!")
            continue
        paths.append(path)
        px, py, pyaw = gen_path(curr_state, path)
        plt.plot(px, py, f"-{color_set[idx]}")
        plt.pause(1)
        plt.axis('equal')

    opt_path, opt_path_cost = _get_opt_path(paths, r_turn)
    return opt_path, opt_path_cost

if __name__ == "__main__":
    curr_state = [0, 0, np.deg2rad(100)]
    goal_state = [4, -2, np.deg2rad(140)]
    car = Car(*curr_state)
    car.display_arrow('black')
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.axis('equal')
    # dubins_path(curr_state, goal_state)
    arrow_length = 2
    dx = arrow_length * math.cos(goal_state[2])
    dy = arrow_length * math.sin(goal_state[2])
    plt.arrow(goal_state[0], goal_state[1], dx, dy,
              head_width=0.3, head_length=0.4,
              fc="blue", ec="blue")
    r_turn =Car.WHEEL_BASE / math.tan(Car.MAX_STEER)
    opt_path, opt_cost = dubins_path(curr_state, goal_state, r_turn)
    print("cost = ", opt_cost)

    px, py, pyaw = gen_path(curr_state, opt_path)
    count = 0
    for x, y, yaw in zip(px, py, pyaw):
        if count % 10 == 0:
            car.x, car.y, car.yaw = x, y, yaw
            plt.cla()
            plt.plot(px, py, "-k")
            car.draw()
            plt.pause(0.001)
            plt.axis('equal')
        count+=1
    plt.axis('equal')
    plt.show()
