import json

import numpy as np
import random
import sim
from Node import Node
from Car import Car
from IDM import IDMVehicle
import matplotlib.pyplot as plt

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
        obj_len = 4  # Standard object length

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
                # name will be assigned later
                "type": "dynamic",
                "start_t": start_t,
                "end_t": end_t,
                "begin_distance": begin_distance,
                "following_distance": 8,
                "obj_len": obj_len,
                "idm": IDMVehicle(begin_distance, 0, 20),
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
    p1 = np.array([curr.s, curr.t])
    p2 = np.array([target.s, target.t])
    p3 = np.array([es['begin_distance'], es['end_t']])
    p4 = np.array([es['begin_distance'], es['start_t']])

    p1p2 = ccw(p1, p2, p3) * ccw(p1, p2, p4)
    p3p4 = ccw(p3, p4, p1) * ccw(p3, p4, p2)

    if p1p2 == 0 and p3p4 == 0:
        if p1[1] > p2[1]: p1, p2 = p2, p1
        if p3[1] > p4[1]: p3, p4 = p4, p3

        return p2[1] >= p3[1] and p4[1] >= p1[1]
    return p1p2 <= 0 and p3p4 <= 0

def get_desired_speed(s):
    desired_speed = [30] * 1000 # 1km 전 구간 제한 30km/h
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
    # 속도는 1로 가정
    p1 = np.array([event['begin_distance'] - event['following_distance'], event['start_t']])
    p2 = np.array([event['begin_distance'], event['start_t']])
    p3 = np.array([p1[0] + event['end_t'], event['end_t']])
    p4 = np.array([p2[0] + event['end_t'], event['end_t']])



    p = np.array([target.s, target.t])

    return is_in_rect(p, p, p2, p4, p3)

def is_collision_vehicles(target, event):
    p2 = np.array([event['begin_distance'], event['start_t']])
    p5 = np.array([p2[0] + event['obj_len'], event['start_t']])
    p4 = np.array([p2[0] + event['end_t'], event['end_t']])
    p6 = np.array([p4[0] + event['obj_len'], event['end_t']])

    p = np.array([target.s, target.t])
    d1 = ccw(p2, p5, p)
    d2 = ccw(p5, p6, p)
    d3 = ccw(p6, p4, p)
    d4 = ccw(p4, p2, p)
    is_zero = d1 == 0 or d2 == 0 or d3 == 0 or d4 == 0
    return is_zero or (is_in_rect(p, p2, p5, p6, p4) or (ccw(p5, p6, p) < 0))

def is_collision_dynamic(target, event):
    if is_in_cost_map(target, event):
        return 300
    elif is_collision_vehicles(target, event):
        return 99999
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

def calc_event_cost(curr, target, event):
    if not event:
        return 0
    event_type = event['type']
    cost = 0
    if event_type == 'static':
        cost += 99999 if is_collision_static(curr, target, event) else 0
    else: # dynamic event
        cost += is_collision_dynamic(target, event)

    return cost

def calc_cost(curr, a, target, event):
    cv = calc_desired_v_cost(target)
    ca = calc_a_cost(a) * 0.25
    ce = calc_event_cost(curr, target, event)
    return cv + ca + ce

def set_of_action():
    return [-2, -1, 0, 1]

def get_grid_idx(s, t, a):
    return t * 41 + s * 31

def get_result_path(closed, g_node):
    rs, rv, rt = [], [], []

    curr = g_node
    while curr.pidx != -1:
        sim.expansion_pos(curr.s, curr.t, 'r')
        sim.pause(0.1)
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

        print(f"current g cost; {curr.g} | {curr.s}, {curr.t}")
        print(f"--- {curr_event_key}")

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

        # print(f"{curr_id} : s, v, t = {curr.s}, {curr.v}, {curr.t} | cost = {curr.g} |event = {curr_event_key}")
        if events.get(curr_event_key) and events[curr_event_key]['end_t'] < curr.t:
            try:
                curr_event_key = next(iter_event)
                # print(f"--- {curr_event_key}")
            except StopIteration:
                curr_event_key = 'none'
                events['none'] = None

        for a in set_of_action():
            ns, nv, nt = transition_model(curr.s, curr.v, curr.t, a)

            if ns < curr.s or nt < curr.t:
                continue

            nidx = get_grid_idx(ns, nt, a)
            # if nidx in closed_set:
            #     continue

            next_node = Node(ns, nv, nt, pidx=curr_id)
            n_g_cost = calc_cost(curr, a, next_node, events.get(curr_event_key))
            next_node.g = curr.g + n_g_cost

            if nidx in closed_set and closed_set[nidx].g > next_node.g:
                del closed_set[nidx]

            if nidx in open_set and open_set[nidx].g > next_node.g:
                del open_set[nidx]

            if nidx not in open_set:
                open_set[nidx] = next_node
    # print(closed_set)
    return get_result_path(closed_set, g_node)

if __name__ == "__main__":
    # is_collision_vehicles(None, None)

    with open('urban_env.json', 'r') as json_file:
        event_json_data = json.load(json_file)

    # Keep only static events from the file
    static_events = {k: v for k, v in event_json_data.items() if v.get('type') == 'static'}

    # Generate random dynamic vehicles
    random_vehicles = generate_random_vehicles(len(static_events), 3)

    # Combine static and dynamic events
    event_json_data = static_events
    event_json_data.update(random_vehicles)

    sim.distance_time(event_json_data)

    ego = Car(0, 0, 0)

    # Create a list of NPC cars from the generated events
    npcs = []
    if event_json_data:
        for event_key, event_value in event_json_data.items():
            if event_value and event_value.get('type') == 'dynamic':
                npcs.append(Car(event_value["begin_distance"], 0, 0))


    rs, rv, rt = planning([0, 0, 0], event_json_data, 30)
    sim.planning_res(rs, rt)
    sim.show()

    sim.simulation([rs, rv, rt], ego, npcs, event_json_data)
