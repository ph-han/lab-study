import matplotlib.pyplot as plt
import numpy as np
import random

import frenet
from sim import Simulator, generate_road
from Car import Car

'''
TODO
1. 장애물 충돌 여부 체크
2. 시뮬레이션 (시나리오 별)
    - [x] velocity keeping
    - [x] avoidance
    - [] merging
    - [] stop
'''

import random

def spawn_frenet_npcs(cxlist, cylist, cslist, num_npcs=7, road_length=150, lane_num=3, lane_width=3.5, min_gap=5.0):
    npcs = []
    slist = []

    for i in range(num_npcs):
        lane = random.randint(0, lane_num - 1)
        d = (lane - (lane_num - 1) / 2) * lane_width

        while True:
            s = random.uniform(5, road_length)
            if all(abs(s - s0) >= min_gap for s0 in slist):
                break

        slist.append(s)
        x, y, yaw = frenet.frenet2world(s, d, cxlist, cylist, cslist)

        npc = {
            'type': 'vehicle',
            'object': Car(x, y, yaw, s, d)
        }
        npcs.append(npc)

    npcs.sort(key=lambda car: car['object'].s)
    return npcs


if __name__ == "__main__":
    road = generate_road(lane_num=3, lane_width=3.5, road_length=300, curved=False)
    ego = Car(0, 0, 0)
    slist = [
        frenet.world2frenet(rx, ry, road['center_xlist'], road['center_ylist'])[0] \
        for (rx, ry) in zip(road['center_xlist'], road['center_ylist'])
    ]
    x, y, _ = frenet.frenet2world(60, 0, road['center_xlist'], road['center_ylist'], slist)
    npc = Car(x, y, np.arcsin(np.sin(59)))
    obstacles = [
    #     {
    #         "type": 'vehicle',
    #         "object": npc
    #     },
    ]

    for npc in spawn_frenet_npcs(road['center_xlist'], road['center_ylist'], slist):
        obstacles.append(npc)

    print(obstacles)
    fig, ax = plt.subplots(figsize=(10,6))
    sim = Simulator(obstacles, road, ego)
    sim.simple_example(ax)
    plt.show()