import numpy as np
import json
import draw

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

    return m1 * [s, v, t, 1] + m2 * a

def get_desired_speed(s):
    pass

def calc_desired_v_cost(next):
    '''
    the cost for any deviation to the desired speed.

    :param next: next state information
    :return: cost
    '''

    v = next.v
    des_v = get_desired_speed(next.s)
    if v > des_v:
        return (v - des_v) ** 2
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

def calc_event_cost(curr, next, event):
    '''
    the cost for a collision while traversing from x_i to x_{i+1}

    :param curr: current state
    :param next: next state
    :param event: event information
    :return: cost
    '''
    return 0

def calc_cost(curr, a, next, event):
    '''
    The step cost c(xᵢ,a,xᵢ+₁,E) is the cost for taking action a in state xᵢ to traverse to state Xᵢ+1.

    :param curr: current state
    :param a: current selected action
    :param next: next state
    :param event: event information
    :return: calculate cost current state to next state
    '''

    cv = calc_desired_v_cost(next)
    ca = calc_a_cost(a)
    ce = calc_event_cost(curr, next, event)

    return cv + ca + ce

def set_of_action():
    return [-2 , -1, 0, 1]

def planning(events, horizen=13):
    pass

if __name__ == "__main__":
    with open("urban_env.json") as json_file:
        event_json_data = json.load(json_file)

    planning(event_json_data)
    draw.distance_time(event_json_data)
    draw.show()
