import numpy as np
import random
import matplotlib.pyplot as plt

import planner
from frenet import *
from config import *
from obstacles import Car

def spawn_frenet_npcs(cxlist, cylist, cslist, num_npcs=5, road_length=80, lane_num=3, lane_width=3.5, min_gap=5.0):
    npcs = []
    slist = []

    random.seed(25)
    for i in range(num_npcs):
        lane = random.randint(0, lane_num - 1)
        d = (lane - (lane_num - 1) / 2) * lane_width

        while True:
            s = random.uniform(5, road_length)
            if all(abs(s - s0) >= min_gap for s0 in slist):
                break

        slist.append(s)
        x, y, yaw = frenet2world(s, d, cxlist, cylist, cslist)

        npc = {
            'type': 'vehicle',
            'object': Car(x, y, yaw, s, d)
        }
        npcs.append(npc)

    npcs.sort(key=lambda car: car['object'].s)
    return npcs

def generate_road(lane_num=3, lane_width=3.5, road_length=100, curved=False, amp=5, freq=0.05, num_points=1000):
    """
    Made by GhatGPT-5
    """
    x = np.linspace(-2, road_length, num_points)

    if curved:
        y_centerline = amp * np.sin(freq * x)
    else:
        y_centerline = np.zeros_like(x)

    half_width = (lane_num / 2) * lane_width
    boundaries = [y_centerline + offset for offset in np.linspace(-half_width, half_width, lane_num+1)]

    centers = [y_centerline + offset for offset in np.linspace(-(lane_num-1)/2*lane_width, (lane_num-1)/2*lane_width, lane_num)]

    return {
        "center_xlist": x,
        "center_ylist": y_centerline,
        "boundaries": boundaries
    }


def plot_road(ax, road_data):
    """
    Made by GhatGPT-5
    """
    x = road_data["center_xlist"]

    for i, y in enumerate(road_data["boundaries"]):
        if i == 0 or i == len(road_data["boundaries"]) - 1:
            ax.plot(x, y, 'k-', linewidth=2)
        else:
            ax.plot(x, y, 'k--', linewidth=1)

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
        # self.velocity_keeping=velocity_keeping
        
    def draw_valid_paths_and_opt_path(self, ax, paths, opt_path):
        for path in paths:
            ax.plot(path.xlist, path.ylist, '-', color="#A2C0F5")
        ax.plot(opt_path.xlist, opt_path.ylist, '-', color="#6cf483")

    def draw_obstacles(self, ax):
        if not self.obs:
            return
        
        for o in self.obs:
            if o['type'] == 'vehicle':
                o['object'].draw(ax)
                o['object'].update_state(self.obs, self.center_line_xlist, self.center_line_ylist, self.center_line_slist, dt=0.01)
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

    def find_leading_vehicle(self, obs, ego_s, ego_d, lane_width=3.5, lane_num=3):
        leading_vehicle = None
        min_dist = np.inf
        
        ego_lane = int(round(ego_d / lane_width + (lane_num - 1) / 2))

        for o in obs:
            if o['type'] != 'vehicle':
                continue
            obj = o['object']

            obj_lane = int(round(obj.d / lane_width + (lane_num - 1) / 2))

            if obj_lane != ego_lane:
                continue

            dist = obj.s - ego_s
            if dist > 0 and dist < min_dist:
                min_dist = dist
                leading_vehicle = obj

        return leading_vehicle


    def get_opt_traj(self, d0, d1, d2, s0, s1, s2, opt_d, mode):
        fplist = []
        if mode == DrivingMode.VELOCITY_KEEPING:
            # pass
            fplist = planner.generate_velocity_keeping_trajectories_in_frenet((d0, d1, d2, 0, 0), (s0, s1, s2, 0), opt_d, FINAL_DESIRED_SPEED)
        elif mode == DrivingMode.STOPPING:
            # pass
            fplist = planner.generate_stopping_trajectories_in_frenet((d0, d1, d2, 0, 0), (s0, s1, s2, 0, 0), opt_d)
            # fplist += planner.generate_velocity_keeping_trajectories_in_frenet((d0, d1, d2, 0, 0), (s0, s1, s2, 0), opt_d, 0)
        elif mode == DrivingMode.MERGING:
            pass
            # fplist = planner.generate_merging_trajectories_in_frenet((d0, d1, d2, 0, 0), (s0, s1, s2, 0, 0), opt_d)
        elif mode == DrivingMode.FOLLOWING:
            pass
            # leading_vehicle = self.find_leading_vehicle(self.obs, s0, d0)
            # if not leading_vehicle:
            #     return None 
            # lon_state = (s0, s1, s2, leading_vehicle.idm.v, leading_vehicle.idm.a)
            # fplist = planner.generate_following_trajectories_in_frenet((d0, d1, d2, 0, 0), lon_state, leading_vehicle, leading_vehicle.d)
        else:
            print("[ERROR] Wrong mode input!")
            return None
        
        fplist = planner.frenet_paths_to_world(fplist, self.center_line_xlist, self.center_line_ylist, self.center_line_slist)
        print(f"[{mode.value}]: {len(fplist)}")
        valid_paths = planner.check_valid_path(fplist, self.obs, self.road['boundaries'], self.center_line_xlist, self.center_line_ylist)
        print(f"after check contraint[{mode.value}]: {len(valid_paths)}")
        if not valid_paths:
            return None
        opt_path = planner.generate_opt_path(valid_paths)
        return opt_path, valid_paths

    def run(self, name, ax):
        s0, d0 = world2frenet(self.ego.x, self.ego.y, self.center_line_xlist, self.center_line_ylist)
        s1, s2, d1, d2 = 0, 0, 0, 0
        opt_d = d0
        lane_num = 3
        no_new_path_cnt = 0
        opt_traj = None
        for i in range(1000):
            if SHOW_ALL_FRENET_PATH:
                plt.figure(2).clf()
                plt.figure(3).clf()
                plt.figure(4).clf()
            
            new_opt_traj_cand = [self.get_opt_traj(d0, d1, d2, s0, s1, s2, opt_d, mode) for mode in DrivingMode]
            new_opt_traj, valid_paths = min(filter(None, new_opt_traj_cand), key=lambda o: o[0].sj[0], default=(None, []))
            # print(new_opt_traj_cand.index(new_opt_traj))
            no_new_path_cnt = no_new_path_cnt + 1 if not new_opt_traj else 0

            if new_opt_traj:
                opt_traj = new_opt_traj
            if not opt_traj:
                print("[ERROR] Can't play simulation")
                return
            s0 = opt_traj.s0[1 + no_new_path_cnt]
            s1 = opt_traj.s1[1 + no_new_path_cnt]
            s2 = opt_traj.s2[1 + no_new_path_cnt]
            d0 = opt_traj.d0[1 + no_new_path_cnt]
            d1 = opt_traj.d1[1 + no_new_path_cnt]
            d2 = opt_traj.d2[1 + no_new_path_cnt]
            opt_d = opt_traj.d0[-1]

            ax.figure.canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            ax.cla()
            self.ego.x, self.ego.y, self.ego.yaw = opt_traj.xlist[0 + no_new_path_cnt], opt_traj.ylist[0 + no_new_path_cnt], opt_traj.yawlist[0 + no_new_path_cnt]
            self.ego.draw(ax)
            self.draw_valid_paths_and_opt_path(ax, valid_paths, opt_traj)
            self.draw_obstacles(ax)
            plot_road(ax, self.road)
            ax.plot([STOP_POS, STOP_POS], [-5.25, 5.25], '-r', lw=3)
            ax.set_title(f"{name}\nStep: {i} | ego [S]: {self.ego.x:.2f} m, [V]: {s1:.2f} m/s")
            ax.set_xlim(self.ego.x - 10, self.ego.x + 60)
            plt.pause(0.1)
            # input("test: ")


