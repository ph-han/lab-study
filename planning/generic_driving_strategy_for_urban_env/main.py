import numpy as np
import json
import draw
from Node import Node
from Car import Car
from IDM import IDMVehicle

DELTA_T = 1.0

def transition_model(s, v, t, a):
    '''
    transition model

    :param s: current state travel distance
    :param v: current state velocity
    :param t: current state time
    :param a: acceleration in set of actions
    :return: next state x_(i+1)
    '''
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
    p1 = np.array([curr.s, curr.t])
    p2 = np.array([target.s, target.t])
    p3 = np.array([es['begin_distance'], es['end_t']])
    p4 = np.array([es['begin_distance'], es['start_t']])

    p1p2 = ccw(p1, p2, p3) * ccw(p1, p2, p4)
    p3p4 = ccw(p3, p4, p1) * ccw(p3, p4, p2)

    if p1p2 == 0 and p3p4 == 0:
        if p1.any() > p2.any():
            p1, p2 = p2, p1
        if p3.any() > p4.any():
            p3, p4 = p4, p3
        return p3.any() < p2.any() and p1.any() < p4.any()
    return p1p2 <= 0 and p3p4 <= 0

def get_desired_speed(s):
    desired_speed = [30] * 1000 # 1km 전 구간 제한 30km/h
    return desired_speed[round(s)]

def is_in_rect(p, p1, p2, p3, p4):
    d1 = ccw(p1, p2, p)
    d2 = ccw(p2, p4, p)
    d3 = ccw(p4, p3, p)
    d4 = ccw(p3, p1, p)

    is_neg = (d1 < 0) or (d2 < 0) or (d3 < 0) or (d4 < 0)
    is_posi = (d1 > 0) or (d2 > 0) or (d3 > 0) or (d4 > 0)
    # print(d1, d2, d3, d4)
    return not (is_neg and is_posi)

def is_in_cost_map(target, event):
    # 속도는 1로 가정
    p1 = np.array([event['begin_distance'] - event['following_distance'], event['start_t']])
    p2 = np.array([event['begin_distance'], event['start_t']])
    p3 = np.array([p1[0] + event['end_t'], event['end_t']])
    p4 = np.array([p2[0] + event['end_t'], event['end_t']])

    p = np.array([target.s, target.t])
    # p = target
    return is_in_rect(p, p1, p2, p3, p4)

def is_collision_vehicles(target, event):
    p2 = np.array([event['begin_distance'], event['start_t']])
    p4 = np.array([p2[0] + event['end_t'], event['end_t']])
    p5 = np.array([p2[0] + event['obj_len'], event['start_t']])
    p6 = np.array([p4[0] + event['obj_len'], event['end_t']])

    p = np.array([target.s, target.t])

    return is_in_rect(p, p2, p5, p6, p4) or (ccw(p, p5, p6) < 0)

def is_collision_dynamic(target, event):
    if is_in_cost_map(target, event):
        return 100
    elif is_collision_vehicles(target, event):
        return np.inf
    else:
        return 0


def calc_desired_v_cost(target):
    '''
    the cost for any deviation to the desired speed.

    :param next: next state information
    :return: cost
    '''

    v = target.v
    des_v = get_desired_speed(target.s)
    if v > des_v:
        return (v - des_v) * (v - des_v)
    elif v == des_v:
        return 0
    else:
        return (des_v - v) / 2

def calc_a_cost(a):
    '''
    the cost for taking action a

    :param a: action a
    :return: cost
    '''
    return abs(a)

def calc_event_cost(curr, target, event):
    '''
    the cost for a collision while traversing from x_i to x_{i+1}

    :param curr: current state
    :param target: next target state
    :param event: event information
    :return: cost
    '''

    if not event:
        return 0
    event_type = event['type']
    cost = 0
    if event_type == 'static':
        cost += np.inf if is_collision_static(curr, target, event) else 0
    else: # dynamic event
        cost += is_collision_dynamic(target, event)

    return cost

def calc_cost(curr, a, target, event):
    '''
    The step cost c(xᵢ,a,xᵢ+₁,E) is the cost for taking action a in state xᵢ to traverse to state Xᵢ+1.

    :param curr: current state
    :param a: current selected action
    :param target: next target state
    :param event: event information
    :return: calculate cost current state to next state
    '''

    cv = calc_desired_v_cost(target)
    ca = calc_a_cost(a)
    ce = calc_event_cost(curr, target, event)
    print(f"cv = {cv}, ca = {ca}, ce = {ce} | tot = {cv + ca + ce}")
    return cv + ca + ce

def set_of_action():
    return [-2, -1, 0, 1]

def get_grid_idx(s, t, a):
    return t * 41 + a * 37 + s * 31

def get_result_path(closed, g_node):
    rs, rv, rt = [], [], []

    curr = g_node
    while curr.pidx != -1:
        draw.expansion_pos(curr.s, curr.t, 'r')
        draw.pause(0.1)
        print(f"{curr.s}, {curr.v}, {curr.t}")
        rs.append(curr.s)
        rv.append(curr.v)
        rt.append(curr.t)
        curr = closed[curr.pidx]

    rs.append(curr.s)
    rv.append(curr.v)
    rt.append(curr.t)

    return rs[::-1], rv[::-1], rt[::-1]

def planning(start_state, events, horizen=13):
    w = 0.0
    s, v, t = start_state
    s_node = Node(s, v, t)
    g_node = Node()

    open_set = {}
    closed_set = {}

    open_set[get_grid_idx(s, t, 0)] = s_node
    iter_event = iter(events)
    curr_event_key = next(iter_event)
    while open_set:
        curr_id = min(open_set, key=lambda o: open_set[o].g + w * open_set[o].h)
        curr = open_set[curr_id]

        del open_set[curr_id]
        closed_set[curr_id] = curr

        # draw.expansion_pos(curr.s, curr.t)
        # draw.pause(0.01)

        if curr.t > horizen:
            g_node.s = closed_set[curr.pidx].s
            g_node.v = closed_set[curr.pidx].v
            g_node.t = closed_set[curr.pidx].t
            g_node.pidx = closed_set[curr.pidx].pidx
            break

        print(f"{curr_id} : s, v, t = {curr.s}, {curr.v}, {curr.t} | cost = {curr.g} |event = {curr_event_key}")
        if events[curr_event_key] and events[curr_event_key]['end_t'] <= curr.t:
            try:
                curr_event_key = next(iter_event)
                print(f"--- {curr_event_key}")
            except StopIteration:
                curr_event_key = 'none'
                events['none'] = None

        for a in set_of_action():
            ns, nv, nt = transition_model(curr.s, curr.v, curr.t, a)

            if ns < curr.s or nt < curr.t:
                continue

            # ngap = abs(events[curr_event_key]['begin_distance'] - ns)
            # if events[curr_event_key]['end_t'] > nt and ngap < events[curr_event_key]['gap']:
            #     continue

            nidx = get_grid_idx(ns, nt, a)
            if nidx in closed_set:
                continue

            next_node = Node(ns, nv, nt, pidx=curr_id)
            n_g_cost = calc_cost(curr, a, next_node, events[curr_event_key])
            next_node.g = curr.g + n_g_cost

            # if nidx in closed_set and closed_set[nidx].g > next_node.g:
            #     del closed_set[nidx]

            if nidx in open_set and open_set[nidx].g > next_node.g:
                del open_set[nidx]

            if nidx not in open_set:
                open_set[nidx] = next_node
    # print(closed_set)
    return get_result_path(closed_set, g_node)

if __name__ == "__main__":
    with open("urban_env.json") as json_file:
        event_json_data = json.load(json_file)

    # print(is_collision_static(None, None, None))
    draw.distance_time(event_json_data)
    # print(is_in_cost_map(np.array([28, 8]), event_json_data["vehicles"]))
    rs, rv, rt = planning([0, 0, 0], event_json_data, 25)
    draw.planning_res(rs, rt)
    draw.show()
    ego = Car(0, 0, 0)
    vehicle = Car(event_json_data["e3"]["begin_distance"], 0, 0)
    draw.simulation([rs, rv, rt], ego, vehicle, event_json_data)
