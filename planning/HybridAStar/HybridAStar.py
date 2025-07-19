import math
import numpy as np
import matplotlib.pyplot as plt
import heapq as hq

from Car import Car
class Node:
    def __init__(self, x, y, yaw, rx, ry, ryaw, r=1, g=0.0, h= 0.0, cost=0.0, p_idx=-1):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.r = r
        self.g = g
        self.h = h
        self.f = g + h
        self.rx = rx
        self.ry = ry
        self.ryaw = ryaw
        self.p_idx = p_idx

class Node2D:
    def __init__(self, x, y, f=0.0, p_idx=-1):
        self.x = x
        self.y = y
        self.f = f
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

    def get_grid2D_idx(self, node, resolution):
        width = round(self.x_width / resolution)
        return round(node.y * width + node.x)

    def get_grid3D_idx(self, node, resolution):
        width = round(self.x_width / resolution)
        height = round(self.y_width / resolution)
        return round(node.x + node.y * width + node.yaw * width * height)


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
        # setting start node and goal node
        start_node = Node(round((self.start[0] - self.map.min_x) / self.xy_resolution),
                          round((self.start[1] - self.map.min_y) / self.xy_resolution),
                          round(self.start[2] / self.yaw_resolution),
                          self.start[0], self.start[1], self.start[2])

        goal_node = Node(round((self.goal[0] - self.map.min_x) / self.xy_resolution),
                         round((self.goal[1] - self.map.min_y) / self.xy_resolution),
                         round(self.goal[2] / self.yaw_resolution),
                         self.goal[0], self.goal[1], self.goal[2])
        
        # penalty constants
        move_cost = 1
        reverse_move_cost = 10
        change_direction_cost = 50
        steering_cost = 1.5
        diff_yaw_cost = 2
        
        # calculated heurstic tables
        self.non_holonomic_table = self.non_holonomic_without_obstacles(start_node, goal_node)
        self.holonomic_table = self.holonomic_with_obstacles(goal_node)
        
        open_set, closed_set = {}, {}
        open_set[self.map.get_grid3D_idx(start_node, self.xy_resolution)] = start_node

        while open_set:
            curr_id = min(open_set, key=lambda o:open_set[o].f)
            curr = open_set[curr_id]

            del open_set[curr_id]
            closed_set[curr_id] = curr

            if self.is_goal(curr.x, curr.y, curr.yaw, goal_node):
                goal_node.p_idx = curr_id
                break

            print(f"current state = x : {curr.rx}, y : {curr.ry}, yaw : {curr.ryaw}")
            self.vehicle.x = curr.x * self.xy_resolution
            self.vehicle.y = curr.y * self.xy_resolution
            self.vehicle.yaw = curr.yaw * self.yaw_resolution
            self.vehicle.display_arrow('green')
            if len(open_set) % 100 == 0:
                plt.pause(0.01)

            for steer, direction in self.bicycle_action_command():
                rx, ry, ryaw = self.vehicle.action(curr.rx, curr.ry, curr.ryaw, steer, direction)

                nx = round(rx / self.xy_resolution)
                ny = round(ry / self.xy_resolution)
                nyaw = round(ryaw / self.yaw_resolution)

                if not self.map.verify_node(nx, ny, self.xy_resolution):
                    continue

                if self.vehicle.is_collision(self.map, rx, ry, ryaw):
                    continue

                # calculating g cost (including change direction, sterring angle etc..)
                next_g_cost = move_cost
                next_g_cost *= (direction == -1) * reverse_move_cost + (direction == 1) * 1
                next_g_cost += (direction != curr.r) * change_direction_cost
                next_g_cost += abs(steer) * steering_cost
                next_g_cost += abs(curr.yaw - nyaw) * diff_yaw_cost
                next_g_cost += curr.g

                next_node = Node(nx, ny, nyaw, rx, ry, ryaw, direction, next_g_cost, p_idx=curr_id)

                h_idx = self.map.get_grid2D_idx(next_node, self.xy_resolution)
                nh_idx = self.map.get_grid3D_idx(next_node, self.xy_resolution)

                if nh_idx in closed_set:
                    continue

                if h_idx not in self.holonomic_table:
                    self.holonomic_table[h_idx] = 100000
                if nh_idx not in self.non_holonomic_table:
                    self.non_holonomic_table[nh_idx] = 100000

                next_node.h = max(self.holonomic_table[h_idx], self.non_holonomic_table[nh_idx])

                next_node.f = next_g_cost + next_node.h

                if nh_idx not in open_set or open_set[nh_idx].f > next_node.f:
                    open_set[nh_idx] = next_node

        return self.get_final_path(closed_set, goal_node)

    def get_final_path(self, closed_set, goal_node):
        rx, ry, ryaw = [], [], []

        p_idx = goal_node.p_idx
        while p_idx != -1:
            curr = closed_set[p_idx]
            rx.append(curr.rx)
            ry.append(curr.ry)
            ryaw.append(curr.ryaw)
            p_idx = curr.p_idx

        return rx[::-1], ry[::-1], ryaw[::-1]

    def is_goal(self, cx, cy, cyaw, goal_node):
        xy_diff = math.hypot(cx - goal_node.x, cy - goal_node.y)
        yaw_diff = abs(cyaw - goal_node.yaw)

        return xy_diff < 1 and yaw_diff < np.deg2rad(5) / self.yaw_resolution

    def non_holonomic_without_obstacles(self, start_node, goal_node): # calculating from goal_node to all node
        # result of non-holonomic heuristic table
        cost_table = {}

        # penalty constants
        move_cost = 1
        reverse_move_cost = 10
        change_direction_cost = 50
        steering_cost = 1.5
        diff_yaw_cost = 2

        # for exploring the map
        init_idx = self.map.get_grid3D_idx(goal_node, self.xy_resolution)
        init_cost = 0.0
        open_set = {}
        pq = []
        hq.heappush(pq, (init_cost, init_idx))
        open_set[init_idx] = goal_node

        # explore the map
        while open_set:
            curr_cost, curr_idx = hq.heappop(pq) # choose lowest cost in pq
            
            if curr_idx not in open_set:
                continue
            curr = open_set[curr_idx]

            del open_set[curr_idx]
            cost_table[curr_idx] = curr_cost

            # end exploration
            if self.is_goal(curr.x, curr.y, curr.yaw, start_node):
                break

            # test code
            # print(f"current state = x : {curr.rx}, y : {curr.ry}, yaw : {curr.ryaw}")
            # self.vehicle.x = curr.x * self.xy_resolution
            # self.vehicle.y = curr.y * self.xy_resolution
            # self.vehicle.yaw = curr.yaw * self.yaw_resolution
            # self.vehicle.display_arrow('green')
            # if len(open_set) % 1000 == 0:
            #     plt.pause(0.01)
            
            # expanding
            for steer, direction in self.bicycle_action_command():
                rx, ry, ryaw = self.vehicle.action(curr.rx, curr.ry, curr.ryaw, 
                                                   steer, direction * self.xy_resolution)
                
                nx = round(rx / self.xy_resolution)
                ny = round(ry / self.xy_resolution)
                nyaw = round(ryaw / self.yaw_resolution)

                if not self.map.verify_node(nx, ny, self.xy_resolution):
                    continue

                # calculating moving cost (including change direction, sterring angle etc..)
                next_cost = move_cost
                next_cost *= (direction == 1) * reverse_move_cost + (direction == -1) * 1
                next_cost += (direction != curr.r) * change_direction_cost
                next_cost += abs(steer) * steering_cost
                next_cost += abs(goal_node.yaw - nyaw) * diff_yaw_cost
                next_cost += curr_cost

                next_node = Node(nx, ny, nyaw, rx, ry, ryaw, direction, next_cost, p_idx=curr_idx)
                next_idx = self.map.get_grid3D_idx(next_node, self.xy_resolution)

                if next_idx in cost_table:
                    continue

                if next_idx not in open_set or open_set[next_idx].g > next_cost:
                    open_set[next_idx] = next_node
                    hq.heappush(pq, (next_cost, next_idx))

        return cost_table

    def bicycle_action_command(self):
        forward, backward = (1, -1)
        angle_step = np.deg2rad(10) # deg 10

        for angle in np.arange(-self.vehicle.MAX_STEER, self.vehicle.MAX_STEER + angle_step, angle_step):
            yield angle, forward
            yield angle, backward

    @staticmethod
    def manhattan(node, nx, ny):
        return abs(node.x - nx) + abs(node.y - ny)

    def holonomic_with_obstacles(self, goal_node):
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

        open_set[self.map.get_grid2D_idx(goal_node, self.xy_resolution)] = goal_node
        while open_set:
            curr_id = min(open_set, key=lambda o: open_set[o].f)
            curr = open_set[curr_id]

            del open_set[curr_id]

            closed_set[curr_id] = curr.f

            for motion in action_command:
                mx, my, cost = motion
                neighbor_node = Node2D(curr.x + mx,
                                     curr.y + my,
                                     curr.f + cost_weight * cost)

                neighbor_id = self.map.get_grid2D_idx(neighbor_node, self.xy_resolution)

                if not self.map.verify_node(neighbor_node.x, neighbor_node.y, self.xy_resolution):
                    continue

                if self.map.check_obstacle(neighbor_node, self.xy_resolution):
                    continue

                if neighbor_id in closed_set and closed_set[neighbor_id] > neighbor_node.f:
                    del closed_set[neighbor_id]

                if neighbor_id in open_set and open_set[neighbor_id].f > neighbor_node.f:
                    del open_set[neighbor_id]

                if neighbor_id not in open_set and neighbor_id not in closed_set:
                    open_set[neighbor_id] = neighbor_node

        return closed_set

    def analystic_expansion(self):
        pass

if __name__ == "__main__":
    start = [10.0, 10.0, np.deg2rad(90.0)]
    goal = [55.0, 30.0, np.deg2rad(90.0)]
    arrow_length = 2 * 0.5
    dx = arrow_length * math.cos(goal[2])
    dy = arrow_length * math.sin(goal[2])

    plt.arrow(goal[0], goal[1], dx, dy,
              head_width=0.3, head_length=0.4,
              fc="blue", ec="blue")
    ioniq5 = Car(*start)

    ox, oy = [], []

    # for i in range(60):
    #     ox.append(i)
    #     oy.append(0.0)
    # for i in range(60):
    #     ox.append(60.0)
    #     oy.append(i)
    # for i in range(61):
    #     ox.append(i)
    #     oy.append(60.0)
    # for i in range(61):
    #     ox.append(0.0)
    #     oy.append(i)
    # for i in range(40):
    #     ox.append(20.0)
    #     oy.append(i)
    # for i in range(40):
    #     ox.append(40.0)
    #     oy.append(60.0 - i)

    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(40):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(40.0)
    for i in range(41):
        ox.append(0.0)
        oy.append(i)
    
    for i in range(25):
        ox.append(50.0)
        oy.append(i)
    
    for i in range(10):
        ox.append(50.0 + i)
        oy.append(25)
    
    for i in range(10):
        ox.append(50.0 + i)
        oy.append(35)
    
    for i in range(40 - 35):
        ox.append(50.0)
        oy.append(35 + i)


    grid_map = Map(ox, oy)
    plt.pause(0.1)

    hybrid_a_star = HybridAStar(grid_map, ioniq5, start, goal, 2, np.deg2rad(15))
    r_x, r_y, r_yaw = hybrid_a_star.planning()
    plt.plot(r_x, r_y, "-r")
    if len(r_x) == 1:
        print("ERROR!")
    print(r_x, r_y, r_yaw)

    for x, y, yaw in zip(r_x, r_y, r_yaw):
        ioniq5.x, ioniq5.y, ioniq5.yaw = x, y, yaw
        plt.cla()
        grid_map.draw()
        plt.plot(r_x, r_y, "-r")
        ioniq5.draw()
        plt.pause(0.1)
    plt.show()
