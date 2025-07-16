'''
TODO
- [] Node 정의
    - [] x, y, yaw, r (= forward 1, backward 0), cost, parent_idx
- [] Map 정의
    - [] 장애물 정보, 맴 크기, 해상도
- [] HybridAStar 정의
    - [] 맵, 자동차, 시작점, 도착점 (방향까지)
    - [] expansion 방식 정의
    - [] heuristic fucntion
        - [] non-holonomic-without-obstacle (reeds-shepp action)
            - [] 4D 정보 사용 (x, y, ywa, r)
            - [] 시작 초기 한번에 다 계산해두면 됨
        - [] holonomic-with-obstacle (dp - dijkstra)
            - [] 2D 정보 사용 (x, y)
        - [] 둘중에 큰걸 사용
    - [] analytic expansions
        - [] reeds-shepp를 이용해서 탐색속도 향상 시킴
            - [] 장애물 맵에 대해 충돌이 없는 것이 확인되면 해당 경로가 자식 노드로 추가됨.
            - [] 매번 계산을 하게되면 비용이 많이 들기 때문에 매 N번째 노드일때 수행 (목표에 가까워 질 수록 자주 분석적 확장을 하게됨.)

    - [] planning
'''

import math
import heapq
import itertools


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pprint import pprint
from Car import Car

class Node:
    def __init__(self, x, y, yaw, x_list, y_list, yaw_list, r=1, cost=0.0, p_idx=-1):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.r = r
        self.cost = cost
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.p_idx = p_idx

    def __repr__(self):
        return (f"Node(\n"
                f"  x={self.x},\n"
                f"  y={self.y},\n"
                f"  yaw={self.yaw},\n"
                f"  r={self.r},\n"
                f"  cost={self.cost},\n"
                f"  p_idx={self.p_idx},\n"
                f"  x_list={self.x_list},\n"
                f"  y_list={self.y_list},\n"
                f"  yaw_list={self.yaw_list}\n"
                f")")

class Node2D:
    def __init__(self, x, y, cost=0.0, p_idx=-1):
        self.x = x
        self.y = y
        self.cost = cost
        self.p_idx = p_idx

    def __repr__(self):
        return f"(x, y, yaw) = {self.x , self.y}"

class Map:
    def __init__(self, ox_list, oy_list):
        self.ox_list = ox_list
        self.oy_list = oy_list
        self.max_x, self.min_x = round(max(ox_list)), round(min(ox_list))
        self.max_y, self.min_y = round(max(oy_list)), round(min(oy_list))
        self.min_yaw = round(-math.pi / np.deg2rad(5)) - 1
        self.max_yaw = round(math.pi / np.deg2rad(5))
        self.yaw_width = round(self.max_yaw - self.min_yaw)
        self.x_width = round((self.max_x - self.min_x))
        self.y_width = round((self.max_y - self.min_y))
        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                         for _ in range(self.x_width)]
        for ix in range(self.x_width):
            for iy in range(self.y_width):
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - ix, ioy - iy)
                    if d <= 1.0:
                        self.obstacle_map[ix][iy] = True
                        break
        # self.obstacle_kd_tree = cKDTree(np.vstack((ox_list, oy_list)).T)
        self.draw()

    def verify_node(self, x, y, resolution):
        return (self.min_x < x * resolution + self.min_x < self.max_x) and (self.min_y < y * resolution + self.min_y < self.max_y)

    def check_obstacle(self, node, resolution):
        x = node.x * resolution + self.min_x
        y = node.y * resolution + self.min_y
        return self.obstacle_map[x][y]

    def get_grid_idx(self, cx, cy, resolution):
        width = round(self.x_width / resolution)
        # height = round((self.max_y - self.min_y) / resolution)
        return round((cy - self.min_y) * width + (cx - self.min_x))


    @staticmethod
    def draw():
        plt.plot(ox, oy, ".k")
        plt.grid(True)
        plt.axis("equal")

class HybridAStar:
    def __init__(self, grid_map, car, start, goal, xy_resolution, yaw_resolution):
        self.holonomic_table = None
        self.non_holonomic_table = None
        self.map = grid_map
        self.vehicle = car
        self.start = start
        self.goal = goal
        self.xy_resolution = xy_resolution
        self.yaw_resolution = yaw_resolution

    def planning(self):
        start_node = Node(round((self.start[0] - self.map.min_x) / self.xy_resolution),
                          round((self.start[1] - self.map.min_y) / self.xy_resolution),
                          round(self.start[2] / self.yaw_resolution),
                          [self.start[0]], [self.start[1]], [self.start[2]])

        goal_node = Node(round((self.goal[0] - self.map.min_x) / self.xy_resolution),
                         round((self.goal[1] - self.map.min_y) / self.xy_resolution),
                         round(self.goal[2] / self.yaw_resolution),
                         [self.goal[0]], [self.goal[1]], [self.goal[2]])
        open_set = {}
        closed_set = {}

        self.non_holonomic_table = self.non_holonomic_without_obstacles(start_node, goal_node)
        # pprint(self.non_holonomic_table, sort_dicts=False)
        self.holonomic_table = self.holonomic_with_obstacles(start_node, goal_node)
        pprint(self.holonomic_table, sort_dicts=True)

        map_width_reso = round(self.map.x_width / self.xy_resolution)
        map_height_reso = round(self.map.y_width / self.xy_resolution)

        start_id = start_node.x + start_node.y * map_width_reso + start_node.yaw * map_width_reso * map_height_reso
        open_set[start_id] = start_node

        while open_set:
            curr_id = min(open_set, key=lambda o: open_set[o].cost)
            curr = open_set[curr_id]

            self.vehicle.x, self.vehicle.y, self.vehicle.yaw = curr.x * self.xy_resolution, curr.y * self.xy_resolution, curr.yaw * self.yaw_resolution
            # self.vehicle.draw()
            # print(curr.yaw_list)
            self.vehicle.display_arrow("green")
            # plt.plot(curr.x, curr.y, "xc")
            if len(closed_set.keys()) % 1000 == 0:
                plt.pause(0.001)

            del open_set[curr_id]
            closed_set[curr_id] = curr

            if self.is_goal(curr.x, curr.y, curr.yaw, goal_node):
                goal_node.p_idx = curr_id
                goal_node.x_list = curr.x_list[:]
                goal_node.y_list = curr.y_list[:]
                goal_node.yaw_list = curr.yaw_list[:]
                print("Find Path!")
                break

            for angle, direction in self.bicycle_action_command():
                next_id, next_node = self.get_next_node(curr_id, curr, goal_node, angle, direction)
                if next_node is None:
                    continue

                if next_id in closed_set and closed_set[next_id].cost > next_node.cost:
                    del closed_set[next_id]

                # print(next_node.x_list)
                if next_id in open_set and open_set[next_id].cost > next_node.cost:
                    del open_set[next_id]

                if next_id not in open_set and next_id not in closed_set:
                    open_set[next_id] = next_node

        return goal_node.x_list, goal_node.y_list, goal_node.yaw_list

    def get_next_node(self, curr_id, curr, goal_node, angle, direction):
        # Penalty
        reverse_penalty = 10.0
        change_direction_penalty = 30.0
        steering_penalty = 5
        map_width_reso = round(self.map.x_width / self.xy_resolution)
        map_height_reso = round(self.map.y_width / self.xy_resolution)

        nx, ny, nyaw = self.vehicle.action(curr.x_list[-1], curr.y_list[-1], curr.yaw_list[-1], angle, direction)

        if not ((self.map.min_x < round(nx) < self.map.max_x)
                and (self.map.min_y < round(ny) < self.map.max_y)):
            return -1, None

        if self.vehicle.is_collision(self.map, nx, ny, nyaw):
            return -1, None

        new_cost = 1

        if direction == -1:
            new_cost *= reverse_penalty

        if curr.r != direction:
            new_cost += change_direction_penalty

        new_cost += abs(angle) * steering_penalty
        new_cost += abs(curr.yaw - nyaw) * steering_penalty

        next_node_id_2d = round(nx / self.xy_resolution) + round(ny / self.xy_resolution) * map_width_reso
        next_node_id = next_node_id_2d + round(nyaw / self.yaw_resolution) * map_width_reso * map_height_reso

        if next_node_id not in self.non_holonomic_table:
            self.non_holonomic_table[next_node_id] = 0.0
        if next_node_id_2d not in self.holonomic_table:
            self.holonomic_table[next_node_id_2d] = 0.0

        new_cost += max(self.non_holonomic_table[next_node_id], self.holonomic_table[next_node_id_2d])
        # print(new_cost)

        next_node = Node(round(nx / self.xy_resolution), round(ny / self.xy_resolution), round(nyaw / self.yaw_resolution),
                         curr.x_list[:] + [nx],
                         curr.y_list[:] + [ny],
                         curr.yaw_list[:] + [nyaw],
                         direction, new_cost, curr_id)
        # print(next_node_id)
        return next_node_id, next_node


    def is_goal(self, cx, cy, cyaw, goal_node):
        xy_diff = math.hypot(cx - goal_node.x, cy - goal_node.y)
        yaw_diff = abs(cyaw - goal_node.yaw)

        return xy_diff < 1 and yaw_diff < np.deg2rad(5) / self.yaw_resolution

    def non_holonomic_without_obstacles(self, start_node, goal_node):
        # Penalty
        reverse_penalty = 10.0
        change_direction_penalty = 50.0
        steering_penalty = 1.5
        move_cost = 1.0
        map_width_reso = round(self.map.x_width / self.xy_resolution)
        map_height_reso = round(self.map.y_width / self.xy_resolution)

        open_set = {} # key: (x, y)
        curr_pos = {}
        closed_set = {}

        curr_idx = start_node.x + start_node.y * self.map.x_width + start_node.yaw * self.map.x_width * self.map.y_width
        curr_pos[curr_idx] = [start_node.x_list[-1], start_node.y_list[-1], start_node.yaw_list[-1], start_node.r]

        open_set[curr_idx] = 0.0
        while open_set:
            curr_idx_list = list(open_set.keys())
            curr_cost = open_set[curr_idx_list[0]]

            del open_set[curr_idx_list[0]]
            closed_set[curr_idx_list[0]] = curr_cost

            curr_x, curr_y, curr_yaw, curr_dir = curr_pos[curr_idx_list[0]]
            if self.is_goal(round(curr_x / self.xy_resolution), round(curr_y / self.xy_resolution), round(curr_yaw / self.yaw_resolution), goal_node):
                break
            # plt.plot(round(curr_x), round(curr_y), 'xb')
            # if len(open_set) % 1000 == 0:
            #     plt.pause(0.0001)
            # print(len(open_set))
            for angle, direction in self.bicycle_action_command():
                nx, ny, nyaw = self.vehicle.action(curr_x, curr_y, curr_yaw, angle, direction)
                real_pos = (nx, ny, nyaw, direction)
                nx = round(nx / self.xy_resolution)
                ny = round(ny / self.xy_resolution)
                nyaw = round(nyaw / self.yaw_resolution)

                if not ((self.map.min_x / self.xy_resolution < nx < self.map.max_x / self.xy_resolution)
                        and (self.map.min_y / self.xy_resolution < ny < self.map.max_y / self.xy_resolution)):
                    continue

                next_idx = nx + ny * map_width_reso + nyaw * map_width_reso * map_height_reso
                if next_idx in closed_set:
                    continue

                new_cost = 1

                if direction == -1:
                    new_cost *= reverse_penalty

                if curr_dir != direction:
                    new_cost += change_direction_penalty

                new_cost += abs(angle) * steering_penalty
                new_cost += abs(curr_yaw - nyaw) * 2
                new_cost += self.manhattan(goal_node, nx, ny)

                if next_idx in open_set and open_set[next_idx] < new_cost:
                    continue

                curr_pos[next_idx] = [*real_pos]
                open_set[next_idx] = new_cost

        return closed_set

    def bicycle_action_command(self):
        forward, backward = 1, -1
        angle_step = np.deg2rad(10) # deg 10

        for angle in np.arange(-self.vehicle.MAX_STEER, self.vehicle.MAX_STEER, angle_step):
            yield angle, forward
            yield angle, backward

    @staticmethod
    def manhattan(node, nx, ny):
        return abs(node.x - nx) + abs(node.y - ny)

    def holonomic_with_obstacles(self, car_node, goal_node):
        action_command = [
            [0, 1, 1],
            [0, -1, 1],
            [1, 0, 1],
            [-1, 0, 1],
            [1, 1, math.sqrt(2)],
            [1, -1, math.sqrt(2)],
            [-1, 1, math.sqrt(2)],
            [-1, -1, math.sqrt(2)]
        ]

        cost_weight = 2.0

        open_set = dict()
        closed_set = dict()

        open_set[self.map.get_grid_idx(goal_node.x, goal_node.y, self.xy_resolution)] = goal_node
        while open_set:
            curr_id = min(open_set, key=lambda o: open_set[o].cost)
            curr = open_set[curr_id]

            del open_set[curr_id]

            closed_set[curr_id] = curr.cost

            for motion in action_command:
                mx, my, cost = motion
                neighbor_node = Node2D(curr.x + mx,
                                     curr.y + my,
                                     curr.cost + cost)

                neighbor_id = self.map.get_grid_idx(neighbor_node.x, neighbor_node.y, self.xy_resolution)

                if not self.map.verify_node(neighbor_node.x, neighbor_node.y, self.xy_resolution):
                    continue

                if self.map.check_obstacle(neighbor_node, self.xy_resolution):
                    continue

                if neighbor_id in closed_set and closed_set[neighbor_id] > neighbor_node.cost:
                    del closed_set[neighbor_id]

                if neighbor_id in open_set and open_set[neighbor_id].cost > neighbor_node.cost:
                    del open_set[neighbor_id]

                if neighbor_id not in open_set and neighbor_id not in closed_set:
                    open_set[neighbor_id] = neighbor_node

        return closed_set

    def analystic_expansion(self):
        pass

if __name__ == "__main__":
    start = [10.0, 10.0, np.deg2rad(90.0)]
    goal = [50.0, 50.0, np.deg2rad(0.0)]
    ioniq5 = Car(*start)

    ox, oy = [], []

    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)


    grid_map = Map(ox, oy)

    hybrid_a_star = HybridAStar(grid_map, ioniq5, start, goal, 2, np.deg2rad(15))
    r_x, r_y, r_yaw = hybrid_a_star.planning()
    plt.plot(r_x, r_y, "-r")
    print(r_x, r_y, r_yaw)
    # plt.cla()
    for x, y, yaw in zip(r_x, r_y, r_yaw):
        ioniq5.x, ioniq5.y, ioniq5.yaw = x, y, yaw
        plt.cla()
        grid_map.draw()
        plt.plot(r_x, r_y, "-r")
        ioniq5.draw()
        plt.pause(0.01)
    plt.show()
