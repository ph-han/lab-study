import math
import numpy as np
import random
import matplotlib.pyplot as plt
import sim
from Node import Node
from Car import Car

DELTA_T = 1.0

def generate_random_vehicles(start_num, num_vehicles, max_distance=100, max_time=20):
    events = {}
    vehicles = []
    max_tries = num_vehicles * 10  # Try at most 10 times per vehicle
    tries = 0

    while len(vehicles) < num_vehicles and tries < max_tries:
        # Generate random properties
        start_t = random.randint(0, int(max_time * 0.7))  # Start within the first 70% of the timeline
        begin_distance = random.uniform(10, max_distance)  # Don't spawn right at the beginning
        obj_len = 4.7  # Standard object length

        # Simple duration for the event
        end_t = start_t + random.randint(5, 15)

        # Collision check
        is_collision = False
        for v in vehicles:
            # Check for overlap in space and time
            if (abs(v['begin_distance'] - begin_distance) < v['obj_len'] + obj_len) and (
                    max(v['start_t'], start_t) < min(v['end_t'], end_t)):
                is_collision = True
                break

        tries += 1
        if not is_collision:
            new_vehicle = {
                "type": "dynamic",
                "start_t": start_t,
                "end_t": end_t,
                "begin_distance": begin_distance,
                "following_distance": 8,
                "vs": end_t,
                "obj_len": obj_len,
                "gap": 0
            }
            vehicles.append(new_vehicle)

    # Sort vehicles by start_t
    vehicles.sort(key=lambda v: v['begin_distance'])

    # Create events dictionary from sorted vehicles
    for i, vehicle in enumerate(vehicles):
        vehicle_num = i + start_num
        vehicle['name'] = f"vehicle_{vehicle_num}"
        events[f"e{vehicle_num + 1}"] = vehicle
    return events

def transition_model(s, v, t, a):
    m1 = np.array([
        [1, DELTA_T, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, DELTA_T]
    ], dtype=np.float32)

    m2 = np.array([(DELTA_T ** 2) / 2, DELTA_T, 0], dtype=np.float32)

    v = np.array([s, v, t, 1])
    return (m1 @ v.T) + (m2 * a)

def ccw(a, b, c):
    v1 = b - a
    v2 = c - a
    res = v1[0] * v2[1] - v1[1] * v2[0]
    if res < 0:
        return -1
    elif res == 0:
        return 0
    else:
        return 1

def is_collision_static(curr, target, es):
    # p1 = np.array([np.float64(3.7200000000000006), np.float64(3.0)])
    # p2 = np.array([np.float64(3.960000000000001), np.float64(4.0)])
    # p3 = np.array([4,  4.6])
    # p4 = np.array([4, 0])
    p1 = np.array([curr.s, curr.t])
    p2 = np.array([target.s, target.t])
    p3 = np.array([es['begin_distance'], es['end_t']])
    p4 = np.array([es['begin_distance'], es['start_t']])
    # plt.plot((p3[0], p4[0]), (p3[1], p4[1]))
    # plt.plot((p1[0], p2[0]), (p1[1], p2[1]))
    # plt.show()

    d1 = ccw(p1, p2, p3)
    d2 = ccw(p1, p2, p4)
    d3 = ccw(p3, p4, p1)
    d4 = ccw(p3, p4, p2)

    # 일반 교차
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True

    # collinear(일직선) 케이스: 구간 겹침 확인
    def on_segment(a, b, c):
        return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and
                min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))

    if d1 == 0 and on_segment(p1, p2, p3): return True
    if d2 == 0 and on_segment(p1, p2, p4): return True
    if d3 == 0 and on_segment(p3, p4, p1): return True
    if d4 == 0 and on_segment(p3, p4, p2): return True

    return False

def get_desired_speed(s):
    desired_speed = [30] * 100000 # 1km 전 구간 제한 30km/h
    return desired_speed[round(s)]

def is_in_rect(p, p1, p2, p3, p4):
    d1 = ccw(p1, p2, p)
    d2 = ccw(p2, p3, p)
    d3 = ccw(p3, p4, p)
    d4 = ccw(p4, p1, p)

    is_neg = (d1 < 0) or (d2 < 0) or (d3 < 0) or (d4 < 0)
    is_posi = (d1 > 0) or (d2 > 0) or (d3 > 0) or (d4 > 0)
    return not (is_neg and is_posi)

def is_in_cost_map(target, event):
    p1 = np.array([event['begin_distance'] - event['following_distance'], event['start_t']])
    p2 = np.array([event['begin_distance'], event['start_t']])
    p3 = np.array([p1[0] + event['vs'], event['end_t']])
    p4 = np.array([p2[0] + event['vs'], event['end_t']])

    p = np.array([target.s, target.t])

    a = event['end_t'] / (event['end_t'] - event['start_t'])
    b = p2[0] - a * p2[1]

    s = a * target.t + b
    if abs(s - target.s) < event['following_distance']:
        # print("s: ", s, ", t: ", abs(s - target.s))
        # print("test : ", 300 / abs(s - target.s))
        # return 3000 / abs(s - target.s)
        return 1500 / abs(s - target.s)
    return 0
    # return is_in_rect(p, p, p2, p4, p3)

def is_collision_vehicles(target, event):
    p2 = np.array([event['begin_distance'], event['start_t']])
    p5 = np.array([p2[0] + event['obj_len'], event['start_t']])
    p4 = np.array([p2[0] + event['vs'], event['end_t']])
    p6 = np.array([p4[0] + event['obj_len'], event['end_t']])

    p = np.array([target.s, target.t])
    d1 = ccw(p2, p5, p)
    d2 = ccw(p5, p6, p)
    d3 = ccw(p6, p4, p)
    d4 = ccw(p4, p2, p)
    return is_in_rect(p, p2, p5, p6, p4) or (ccw(p5, p6, p) < 0)

def is_collision_dynamic(target, event):
    cost = is_in_cost_map(target, event)
    if is_in_cost_map(target, event):
        return cost
    elif is_collision_vehicles(target, event):
        return np.inf
    else:
        return 0


def calc_desired_v_cost(target):
    v = target.v
    des_v = get_desired_speed(target.s)
    if v > des_v:
        return (v - des_v) * (v - des_v)
    elif v == des_v:
        return 0
    else:
        return (des_v - v) / 2

def calc_a_cost(a):
    return abs(a)

def calc_event_cost(curr, target, events):
    total_cost = 0
    for event_key, event in events.items():
        if not event:
            continue

        if not (event['start_t'] <= target.t <= event['end_t'] or event['start_t'] <= curr.t <= event['end_t']):
             if not (curr.t < event['start_t'] and target.t > event['end_t']):
                continue

        if event['type'] == 'static':
            if is_collision_static(curr, target, event):
                return np.inf
        elif event['type'] == 'dynamic':
            total_cost += is_collision_dynamic(target, event)
    return total_cost

def calc_cost(curr, a, target, event):
    cv = calc_desired_v_cost(target)
    ca = calc_a_cost(a)
    ce = calc_event_cost(curr, target, event)
    return cv + ca + ce

def set_of_action():
    return [-2, -1, 0, 1]

def get_grid_idx(s, t, a, ds=0.5, dt=0.5):
    s_idx = int(s / ds)
    t_idx = int(t / dt)
    return (s_idx, t_idx)

def get_result_path(closed, g_node):
    # plt.figure(2).clf()
    rs, rv, rt = [], [], []
    # cost = []

    curr = g_node
    while curr.pidx != -1:
        # sim.expansion_pos(curr.s, curr.t, 'r')
        # sim.pause(0.1)
        rs.append(curr.s)
        rv.append(curr.v)
        rt.append(curr.t)
        # cost.append(curr.g)
        curr = closed[curr.pidx]

    rs.append(curr.s)
    rv.append(curr.v)
    rt.append(curr.t)
    # cost.append(curr.g)
    # plt.figure(2)
    # plt.plot(rt[::-1], cost[::-1])
    # plt.show()
    return rs[::-1], rv[::-1], rt[::-1]

def planning(start_state, events, horizen=13):
    w = 0.0
    s, v, t = start_state
    s_node = Node(s, v, t)
    g_node = Node()

    open_set = {}
    closed_set = {}

    open_set[get_grid_idx(s, t, 0)] = s_node

    while open_set:
        curr_id = min(open_set, key=lambda o: open_set[o].g + w * open_set[o].h)
        curr = open_set[curr_id]

        # print(f"current g cost; {curr.g} | {curr.s}, {curr.t}")
        # if curr_event_key in ['e1', 'e2']:
        #     print(f"--- \n{events[curr_event_key]}")

        del open_set[curr_id]
        closed_set[curr_id] = curr

        # draw.expansion_pos(curr.s, curr.t)
        # draw.pause(0.01)

        if curr.t > horizen:
            g_node = curr
            break

        for a in set_of_action():
            ns, nv, nt = transition_model(curr.s, curr.v, curr.t, a)

            if ns < curr.s or nt < curr.t:
                continue

            nidx = get_grid_idx(ns, nt, a)
            # if nidx in closed_set:
            #     continue

            next_node = Node(ns, nv, nt, pidx=curr_id)
            n_g_cost = calc_cost(curr, a, next_node, events)
            next_node.g = curr.g + n_g_cost

            if nidx in closed_set:
                if closed_set[nidx].g > next_node.g:
                    del closed_set[nidx]
                    open_set[nidx] = next_node
            elif nidx in open_set:
                if open_set[nidx].g > next_node.g:
                    open_set[nidx] = next_node
            else:
                open_set[nidx] = next_node
    # print(closed_set)
    return get_result_path(closed_set, g_node)



if __name__ == "__main__":
    print(is_collision_static(None, None, None))
