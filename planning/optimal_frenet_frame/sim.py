import numpy as np
import matplotlib.pyplot as plt

import planner
from frenet import *
from config import *
from Car import Car


def generate_road(lane_num=3, lane_width=3.5, road_length=100, curved=False, amp=5, freq=0.05, num_points=1000):
    """
    도로 좌표 데이터를 생성하는 함수
    Made by GhatGPT-5
    """
    x = np.linspace(-2, road_length, num_points)

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


def plot_road(ax, road_data):
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
            ax.plot(path.xlist, path.ylist, '-', color="#A2C0F5")
        ax.plot(opt_path.xlist, opt_path.ylist, '-', color="#6cf483")

    def draw_obstacles(self, ax):
        for o in self.obs:
            if o['type'] == 'vehicle':
                print(o['object'].x, o['object'].y, o['object'].idm.v)
                o['object'].update_state(self.obs, self.center_line_xlist, self.center_line_ylist, self.center_line_slist)
                o['object'].draw(ax)
            else:
                half_width = o['object'].width / 2
                half_height = o['object'].height / 2
                p1 = [o['object'].x - half_width, o['object'].y - half_height]
                p2 = [o['object'].x + half_width, o['object'].y - half_height]
                p3 = [o['object'].x + half_width, o['object'].y + half_height]
                p4 = [o['object'].x - half_width, o['object'].y + half_height]
                xs = [p1[0], p2[0], p3[0], p4[0]]
                ys = [p1[1], p2[1], p3[1], p4[1]]

                ax.fill(xs, ys, color="skyblue", alpha=0.7)
                ax.plot(xs + [xs[0]], ys + [ys[0]], "k-")

    def move_car(self, ax, opt_path):
        for x, y, yaw in zip(opt_path.xlist, opt_path.ylist, opt_path.yawlist):
            ax.clear()
            self.ego.x, self.ego.y, self.ego.yaw = x, y, yaw
            self.ego.draw(ax)
            ax.plot(opt_path.xlist, opt_path.ylist, '-', color="#6cf483")
            plot_road(ax, self.road)
            
            plt.pause(0.01)

    def simple_example(self, ax):
        s0, d0 = world2frenet(self.ego.x, self.ego.y, self.center_line_xlist, self.center_line_ylist)
        s1, s2, d1, d2 = 5, 0, 0, 0
        lane_num = 3
        for i in range(500):
            ax.figure.canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            # fplist = planner.generate_frenet_trajectory((d0, d1, d2, 0, 0), (s0, s1, s2, 0, 0))
            # fplist = planner.frenet_paths_to_world(fplist, self.center_line_xlist, self.center_line_ylist, self.center_line_slist)
            # valid_paths = planner.check_valid_path(fplist, self.obs, self.road['boundaries'], self.center_line_xlist, self.center_line_ylist)
            # opt_path = planner.generate_opt_path(valid_paths)
            # if not opt_path:
            #     print(f"{i} step error no path!")
            #     break
            # s0 = opt_path.s0[1]
            # s1 = opt_path.s1[1]
            # s2 = opt_path.s2[1]
            # d0 = opt_path.d0[1]
            # d1 = opt_path.d1[1]
            # d2 = opt_path.d2[1]
            # opt_d = opt_path.d0[1]
            # print(f"current opt d : {opt_d}, desired d: {DESIRED_LAT_POS}")
            ax.cla()
            # self.ego.x, self.ego.y, self.ego.yaw = opt_path.xlist[0], opt_path.ylist[0], opt_path.yawlist[0]
            # ax.text(self.ego.x + Car.WHEEL_BASE // 2,
            #         self.ego.y + Car.OVERALL_WIDTH + 2,
            #         f'ego', ha='center', va='top')
            self.draw_obstacles(ax)
            self.ego.draw(ax)
            # self.draw_valid_paths_and_opt_path(ax, valid_paths, opt_path)
            plot_road(ax, self.road)
            ax.set_title(f"{lane_num}-lane Road Map | ego speed :{s1:.2f} m/s, desired speed: {DESIRED_SPEED} m/s")
            plt.pause(0.1)

