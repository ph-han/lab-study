import matplotlib.pyplot as plt
import numpy as np
import random

import frenet
from sim import Simulator, generate_road, spawn_frenet_npcs
from obstacles import Car, StaticCar, StaticBox
from config import STOP_POS

'''
TODO
1. 장애물 충돌 여부 체크
2. 시뮬레이션 (시나리오 별)
    - [x] velocity keeping
    - [x] avoidance
    - [] merging
    - [] stop
'''

def demo_static_obstacle_avoidancea_and_velocity_keeping():
    road = generate_road(lane_num=3, lane_width=3.5, road_length=300, curved=False)
    ego = Car(0, 3.5, 0)
    obstacles = [
        {
            "type": 'static',
            "object": StaticBox(60, 3.5, 0)
        },
        {
            "type": 'static',
            "object": StaticCar(90, -2, 0)
        },
        {
            "type": 'static',
            "object": StaticBox(170, 3.5, 0)
        }
    ]
    fig, ax = plt.subplots(figsize=(10,6))
    sim = Simulator(obstacles, road, ego)
    sim.run(ax)
    plt.show()

def demo_dynamic_obstacle_advoidance_and_curved_road_velocity_keeping():
    road = generate_road(lane_num=3, lane_width=3.5, road_length=300, curved=True)
    ego = Car(0, 0, 0)
    slist = [
        frenet.world2frenet(rx, ry, road['center_xlist'], road['center_ylist'])[0] \
        for (rx, ry) in zip(road['center_xlist'], road['center_ylist'])
    ]
    x, y, _ = frenet.frenet2world(60, 0, road['center_xlist'], road['center_ylist'], slist)
    obstacles = []

    for npc in spawn_frenet_npcs(road['center_xlist'], road['center_ylist'], slist):
        obstacles.append(npc)

    fig, ax = plt.subplots(figsize=(10,6))
    sim = Simulator(obstacles, road, ego)
    sim.run(ax)
    plt.show()

def demo_stopping():
    road = generate_road(lane_num=3, lane_width=3.5, road_length=300, curved=False)
    ego = Car(0, 0, 0)
    fig, ax = plt.subplots(figsize=(10,6))
    sim = Simulator(None, road, ego, False)
    sim.run(ax)
    plt.show()


if __name__ == "__main__":
    # demo_static_obstacle_avoidancea_and_velocity_keeping()
    demo_dynamic_obstacle_advoidance_and_curved_road_velocity_keeping()
    # demo_stopping()
