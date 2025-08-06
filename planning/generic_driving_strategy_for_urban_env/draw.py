import numpy as np
import matplotlib.pyplot as plt

def distance_time(events):
    iter_event = iter(events)
    while True:
        try:
            e_key = next(iter_event)
        except StopIteration:
            break
    # traffic_red_sign = events['traffic_red_sign']
    # crossing_pedestrian = events['crossing_pedestrian']
    # vehicles = events['vehicles']

        print(f"{e_key} event type : {events[e_key]['type']}")
        type = events[e_key]['type']

        if type == "static":
            start_t = events[e_key]['start_t']
            end_t = events[e_key]['end_t']
            y = np.arange(start_t, end_t + 1)
            x = [events[e_key]['begin_distance']] * len(y)
            plt.plot(x, y, '-r')
        else:
            p1 = (events[e_key]['begin_distance'] - events[e_key]['following_distance'], events[e_key]['start_t'])
            p2 = (events[e_key]['begin_distance'] , events[e_key]['start_t'])
            p3 = (p1[0] + events[e_key]['end_t'], events[e_key]['end_t'])
            p4 = (p2[0] + events[e_key]['end_t'], events[e_key]['end_t'])

            plt.plot((p1[0], p2[0]), (p1[1], p2[1]), '-c')
            plt.plot((p1[0], p3[0]), (p1[1], p3[1]), '-c')
            plt.plot((p4[0], p2[0]), (p4[1], p2[1]), '-c')
            plt.plot((p3[0], p4[0]), (p3[1], p4[1]), '-c')

            p5 = (p2[0] + events[e_key]['obj_len'], events[e_key]['start_t'])
            p6 = (p4[0] + events[e_key]['obj_len'], events[e_key]['end_t'])

            plt.plot((p2[0], p5[0]), (p2[1], p5[1]), '-b')
            plt.plot((p6[0], p5[0]), (p6[1], p5[1]), '-b')
            plt.plot((p2[0], p4[0]), (p2[1], p4[1]), '-b')
            plt.plot((p4[0], p6[0]), (p4[1], p6[1]), '-b')

    plt.xticks(np.arange(0, 51, 10))
    plt.xlabel("distance [m]")
    plt.yticks(np.arange(0, 16, 5))
    plt.ylabel("time [s]")
    plt.grid(True)

def planning_res(rs, rt):
    plt.plot(rs, rt, '-ob')

def expansion_pos(s, t, color='g'):
    plt.plot(s, t, f'x{color}')

def pause(sec):
    plt.pause(sec)

def show():
    plt.show()