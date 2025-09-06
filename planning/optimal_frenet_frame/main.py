import matplotlib.pyplot as plt
import numpy as np

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

if __name__ == "__main__":
    road = generate_road(lane_num=3, lane_width=3.5, road_length=300, curved=True)
    ego = Car(0, 3.5, 0)
    slist = [
        frenet.world2frenet(rx, ry, road['center_xlist'], road['center_ylist'])[0] \
        for (rx, ry) in zip(road['center_xlist'], road['center_ylist'])
    ]
    x, y, _ = frenet.frenet2world(60, 0, road['center_xlist'], road['center_ylist'], slist)
    npc = Car(x, y, np.arcsin(np.sin(59)))
    obstacles = [
        {
            "type": 'vehicle',
            "object": npc
        }
    ]
    fig, ax = plt.subplots(figsize=(10,6))
    sim = Simulator(obstacles, road, ego)
    sim.simple_example(ax)
    plt.show()