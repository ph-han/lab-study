import planner
import numpy as np
import matplotlib.pyplot as plt
from frenet import *

from Car import Car

def generate_road(lane_num=3, lane_width=3.5, road_length=100, curved=False, amp=5, freq=0.05, num_points=1000):
    """
    도로 좌표 데이터를 생성하는 함수
    Made by GhatGPT-5
    """
    x = np.linspace(0, road_length, num_points)

    # 도로 중심선
    if curved:
        y_centerline = amp * np.sin(freq * x)   # 곡선 도로
    else:
        y_centerline = np.zeros_like(x)         # 직선 도로

    # 도로 경계선
    half_width = (lane_num / 2) * lane_width
    boundaries = [y_centerline + offset for offset in np.linspace(-half_width, half_width, lane_num+1)]

    # 차선 중앙선
    centers = [y_centerline + offset for offset in np.linspace(-(lane_num-1)/2*lane_width, (lane_num-1)/2*lane_width, lane_num)]

    return {
        "center_xlist": x,
        "center_ylist": y_centerline,
        "boundaries": boundaries
    }


def plot_road(ax, road_data, lane_num=3):
    """
    도로 좌표 데이터를 받아서 그림
    Made by GhatGPT-5
    """
    x = road_data["center_xlist"]

    # 도로 경계선 그리기
    for i, y in enumerate(road_data["boundaries"]):
        if i == 0 or i == len(road_data["boundaries"]) - 1:
            ax.plot(x, y, 'k-', linewidth=2)   # 바깥 경계: 실선
        else:
            ax.plot(x, y, 'k--', linewidth=1)  # 차선 분리선: 점선

    ax.set_aspect('equal')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"{lane_num}-lane Road Map")
    ax.grid(True, linestyle=':')


class Simulator:
    def __init__(self, obs, road, ego):
        self.obs = obs
        self.road = road
        self.ego = ego
        self.center_line_xlist = road['center_xlist']
        self.center_line_ylist = road['center_ylist']
        self.center_line_slist = [
            world2frenet(rx, ry, self.center_line_xlist, self.center_line_ylist)[0] \
                for (rx, ry) in list(zip(self.center_line_xlist, self.center_line_ylist))
            ]
        
    def draw_valid_paths_and_opt_path(self, ax, paths, opt_path):
        for path in paths:
            ax.plot(path.xlist, path.ylist, '-', color="#1E6EF4")
        ax.plot(opt_path.xlist, opt_path.ylist, '-', color="#6cf483")

    def move_car(self, ax, opt_path):
        plot_road(ax, road, lane_num=3)
        for x, y, yaw in zip(opt_path.xlist, opt_path.ylist, opt_path.yawlist):
            ego.x, ego.y, ego.yaw = x, y, yaw
            ego.draw(ax)
            
            plt.pause(0.01)

    def simple_example(self, ax):
        frenet_s, frenet_d = world2frenet(ego.x, ego.y, self.center_line_xlist, self.center_line_ylist)
        print(f"{frenet_s}, {frenet_d}")
        fplist = planner.generate_frenet_trajectory((frenet_d, 1, 0, 0, 0), (frenet_s, 6, 0, 0, 0))
        fplist = planner.frenet_paths_to_world(fplist, self.center_line_xlist, self.center_line_ylist, self.center_line_slist)
        valid_paths = planner.check_valid_path(fplist, self.obs, self.center_line_xlist, self.center_line_ylist, self.center_line_slist)
        opt_path = planner.generate_opt_path(valid_paths)
        self.draw_valid_paths_and_opt_path(ax, valid_paths, opt_path)
        self.move_car(ax, opt_path)
        # ego.draw(ax)
        

if __name__ == "__main__":
    ego = Car(0, 3.5, 0)
    road = generate_road(lane_num=3, lane_width=3.5, road_length=100, curved=False)
    fig, ax = plt.subplots(figsize=(10,6))
    sim = Simulator(None, road, ego)
    sim.simple_example(ax)
    plt.show()