import numpy as np
import matplotlib.pyplot as plt

def distance_time(events):
    traffic_red_sign = events['traffic_red_sign']
    crossing_pedestrian = events['crossing_pedestrian']

    print(f"traffic_red_sign event type : {traffic_red_sign['type']}")
    for event_data in traffic_red_sign['data']:
        begin_at = event_data['begin']
        duration = event_data['duration']
        y = np.arange(begin_at, begin_at + duration + 1)
        x = [event_data['begin_distance']] * len(y)
        plt.plot(x, y, '-r')

    print(f"crossing pedestrian event type : {crossing_pedestrian['type']}")
    for event_data in crossing_pedestrian['data']:
        begin_at = event_data['begin']
        duration = event_data['duration']
        y = np.arange(begin_at, begin_at + duration + 1)
        x = [event_data['begin_distance']] * len(y)
        plt.plot(x, y, '-r')

    plt.xticks(np.arange(0, 51, 10))
    plt.xlabel("distance [m]")
    plt.yticks(np.arange(0, 16, 5))
    plt.ylabel("time [s]")
    plt.grid(True)

def pause(sec):
    plt.pause(sec)

def show():
    plt.show()