import math
import numpy as np
import matplotlib.pyplot as plt
import heapq as hq

from Car import Car
class Node:
    def __init__(self, x, y, yaw, x_list, y_list, yaw_list, r=1, g=0.0, h= 0.0, cost=0.0, p_idx=-1):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.r = r
        self.g = g
        self.h = h
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

    def get_grid2D_idx(self, cx, cy, resolution):
        width = round(self.x_width / resolution)
        return round(cy * width + cx)

    def get_grid3D_idx(self, cx, cy, cyaw, resolution):
        width = round(self.x_width / resolution)
        height = round(self.y_width / resolution)
        return round(cx + cy * width + cyaw * width * height)


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
        self.holonomic_table = self.holonomic_with_obstacles(start_node, goal_node)

        # start_id = self.map.get_grid3D_idx(start_node.x, start_node.y, start_node.yaw, self.xy_resolution)
        start_id = self.map.get_grid2D_idx(start_node.x, start_node.y, self.xy_resolution)
        open_set[start_id] = start_node

        while open_set:
            curr_id = min(open_set, key=lambda o: open_set[o].cost)
            curr = open_set[curr_id]
            print(f"current cost : g = {curr.g}, h = {curr.h}, f = {curr.cost}")
            self.vehicle.x, self.vehicle.y, self.vehicle.yaw = curr.x * self.xy_resolution, curr.y * self.xy_resolution, curr.yaw * self.yaw_resolution
            print(f"current pos : x = {self.vehicle.x}, y = {self.vehicle.y}, yaw = {self.vehicle.yaw}")
            self.vehicle.display_arrow("green")
            if len(closed_set.keys()) % 1 == 0:
                plt.pause(0.1)
            # plt.pause(0.1)

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

                if next_id in open_set and open_set[next_id].cost > next_node.cost:
                    del open_set[next_id]

                if next_id not in open_set and next_id not in closed_set:
                    open_set[next_id] = next_node

        return goal_node.x_list, goal_node.y_list, goal_node.yaw_list

    def get_next_node(self, curr_id, curr, goal_node, angle, direction):

        reverse_penalty = 10.0
        change_direction_penalty = 150.0
        steering_penalty = 2

        nx, ny, nyaw = self.vehicle.action(curr.x_list[-1], curr.y_list[-1], curr.yaw_list[-1], angle, direction)

        if not ((self.map.min_x < round(nx) < self.map.max_x)
                and (self.map.min_y < round(ny) < self.map.max_y)):
            return -1, None

        if self.vehicle.is_collision(self.map, nx, ny, nyaw):
            return -1, None

        g = 1

        if direction == -1:
            g *= reverse_penalty

        if curr.r != direction:
            g += change_direction_penalty

        g += abs(curr.yaw - nyaw) * steering_penalty
        g += curr.g

        next_node_id_2d = self.map.get_grid2D_idx(
            round(nx / self.xy_resolution),
            round(ny / self.xy_resolution),
            self.xy_resolution)
        next_node_id = self.map.get_grid3D_idx(
            round(nx / self.xy_resolution),
            round(ny / self.xy_resolution),
            round(nyaw / self.yaw_resolution),
            self.xy_resolution)

        # if next_node_id not in self.non_holonomic_table:
        #     self.non_holonomic_table[next_node_id] = 100000
        if next_node_id_2d not in self.non_holonomic_table:
            self.non_holonomic_table[next_node_id_2d] = 0.0
        if next_node_id_2d not in self.holonomic_table:
            self.holonomic_table[next_node_id_2d] = 100000.0

        h = max(self.non_holonomic_table[next_node_id_2d], self.holonomic_table[next_node_id_2d])
        # h = self.holonomic_table[next_node_id_2d]

        next_node = Node(round(nx / self.xy_resolution), round(ny / self.xy_resolution), round(nyaw / self.yaw_resolution),
                         curr.x_list[:] + [nx],
                         curr.y_list[:] + [ny],
                         curr.yaw_list[:] + [nyaw],
                         direction, g, h, g + h, curr_id)
        return next_node_id_2d, next_node


    def is_goal(self, cx, cy, cyaw, goal_node):
        xy_diff = math.hypot(cx - goal_node.x, cy - goal_node.y)
        yaw_diff = abs(cyaw - goal_node.yaw)

        return xy_diff < 1 and yaw_diff < np.deg2rad(5) / self.yaw_resolution

    def non_holonomic_without_obstacles(self, start_node, goal_node):
        # Penalty
        move_cost = self.xy_resolution
        reverse_penalty = 10.0

        open_set = {} # key: (x, y)
        cost_map = {}
        curr_pos = {}
        closed_set = {}

        # curr_idx = self.map.get_grid3D_idx(start_node.x, start_node.y, start_node.yaw, self.xy_resolution)
        curr_idx = self.map.get_grid2D_idx(goal_node.x, goal_node.y, self.xy_resolution)
        curr_pos[curr_idx] = [goal_node.x_list[-1], goal_node.y_list[-1], goal_node.yaw_list[-1], goal_node.r]

        # open_set[curr_idx] = 0.0
        # hq.heappush(open_set, (0.0, curr_idx))
        open_set[curr_idx] = 0.0
        while open_set:
        # for _ in range(2):
            curr_idx = min(open_set, key=lambda o: open_set[o])
            curr_cost = open_set[curr_idx]
            # curr_cost, curr_idx = hq.heappop(open_set)
            del open_set[curr_idx]
            closed_set[curr_idx] = curr_cost


            curr_x, curr_y, curr_yaw, curr_dir = curr_pos[curr_idx]
            if round(curr_x / self.xy_resolution) == start_node.x and round(curr_y / self.xy_resolution) == start_node.y:
                break
            # plt.plot(round(curr_x), round(curr_y), 'xb')
            # print("curr cost = ", curr_cost, f", curr x = {curr_x}, y = {curr_y}, yaw = {curr_yaw}, direct = {curr_dir}")
            if len(closed_set) % 500 == 0:
                print("calculating... non holonomic heuristic ")
                # plt.pause(0.001)
            print(len(closed_set))
            # plt.cla()
            for angle, direction in self.bicycle_action_command():
                nx, ny, nyaw = self.vehicle.action(curr_x, curr_y, curr_yaw, angle, direction)

                # plt.plot(round(nx), round(ny), 'xb')
                # plt.pause(1)
                real_pos = (nx, ny, nyaw, direction)
                nx = round(nx / self.xy_resolution)
                ny = round(ny / self.xy_resolution)
                nyaw = round(nyaw / self.yaw_resolution)

                if not ((self.map.min_x / self.xy_resolution < nx < self.map.max_x / self.xy_resolution)
                        and (self.map.min_y / self.xy_resolution < ny < self.map.max_y / self.xy_resolution)):
                    continue

                # next_idx = self.map.get_grid3D_idx(nx, ny, nyaw, self.xy_resolution)
                next_idx = self.map.get_grid2D_idx(nx, ny, self.xy_resolution)

                new_cost = move_cost
                new_cost *= (direction == 1) * reverse_penalty + 1
                new_cost += (direction == curr_dir) * 50
                new_cost += curr_cost
                new_cost += np.rad2deg(abs(angle))
                new_cost += abs(goal_node.yaw - nyaw) * 2
                # print(f"steer angle = {np.rad2deg(angle)}, cost = {new_cost}")
                if next_idx in closed_set and closed_set[next_idx] > new_cost:
                    del closed_set[next_idx]

                if next_idx in open_set and open_set[next_idx] > new_cost:
                    del open_set[next_idx]
                    del curr_pos[next_idx]

                if next_idx not in open_set:
                    open_set[next_idx] = new_cost
                    # hq.heappush(open_set, (new_cost, next_idx))
                    curr_pos[next_idx] = [*real_pos]

        return closed_set

    def bicycle_action_command(self):
        forward, backward = (self.xy_resolution, -self.xy_resolution)
        angle_step = np.deg2rad(10) # deg 10

        for angle in np.arange(-self.vehicle.MAX_STEER, self.vehicle.MAX_STEER + angle_step, angle_step):
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

        open_set[self.map.get_grid2D_idx(goal_node.x, goal_node.y, self.xy_resolution)] = goal_node
        while open_set:
            curr_id = min(open_set, key=lambda o: open_set[o].cost)
            curr = open_set[curr_id]

            del open_set[curr_id]

            closed_set[curr_id] = curr.cost

            for motion in action_command:
                mx, my, cost = motion
                neighbor_node = Node2D(curr.x + mx,
                                     curr.y + my,
                                     curr.cost + cost_weight * cost)

                neighbor_id = self.map.get_grid2D_idx(neighbor_node.x, neighbor_node.y, self.xy_resolution)

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
    goal = [50.0, 50.0, np.deg2rad(90.0)]
    arrow_length = 2 * 0.5
    dx = arrow_length * math.cos(goal[2])
    dy = arrow_length * math.sin(goal[2])

    plt.arrow(goal[0], goal[1], dx, dy,
              head_width=0.3, head_length=0.4,
              fc="blue", ec="blue")
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

    # for i in range(60):
    #     ox.append(i)
    #     oy.append(0.0)
    # for i in range(40):
    #     ox.append(60.0)
    #     oy.append(i)
    # for i in range(61):
    #     ox.append(i)
    #     oy.append(40.0)
    # for i in range(41):
    #     ox.append(0.0)
    #     oy.append(i)
    #
    # for i in range(25):
    #     ox.append(50.0)
    #     oy.append(i)
    #
    # for i in range(10):
    #     ox.append(50.0 + i)
    #     oy.append(25)
    #
    # for i in range(10):
    #     ox.append(50.0 + i)
    #     oy.append(35)
    #
    # for i in range(40 - 35):
    #     ox.append(50.0)
    #     oy.append(35 + i)


    grid_map = Map(ox, oy)
    plt.pause(0.1)

    hybrid_a_star = HybridAStar(grid_map, ioniq5, start, goal, 2, np.deg2rad(5))
    r_x, r_y, r_yaw = hybrid_a_star.planning() # [10.0, 11.0, 11.417638143600186, 12.839678026658769, 13.096269380693549, 13.624620200223486, 15.269265120883384, 15.012739916853038, 13.046738024179838, 11.080736131506638, 9.114734238833439, 7.141706636823612, 8.303618683617117, 9.099985275008668, 10.086715986098582, 11.36054211364293, 12.806409172703827, 14.791639018192821, 12.859033674194697, 10.866391907139358, 9.371990856165878, 7.390253902426876, 5.408516948687874, 3.7170817214087473, 5.25364544116095, 6.544556846226355, 8.461845555899533, 9.888782188670671, 10.937205085296856, 12.787790505557336, 14.787773781790854, 12.80812759734355, 10.951400351692206, 12.230699402495539, 14.055840736150463, 15.582659500686246, 15.13172023080849, 13.139084181224616, 11.146448131640742, 9.153812082056868, 7.342404206862857, 8.940275523488463, 10.771507962997209, 12.369379279622816, 14.20061171913156, 15.798483035757165, 17.62971547526591, 19.227586791891518, 21.215534006818515, 23.2146993425942, 25.213864678369887, 27.213030014145573, 29.21219534992126, 31.211360685696945, 33.21052602147263, 34.912966447045086, 35.79884219029713, 34.17589340476394, 32.43798053831059, 33.48932917460464, 33.41293760507477, 33.33654603554489, 33.26015446601502, 33.18376289648514, 33.10737132695527, 33.03097975742539, 33.302791671906476, 33.34034251822102, 33.14311400426277, 33.407814545075304, 33.25271538999497, 35.2398631547221, 37.23981402226296, 39.23976488980382, 41.21176635301868, 42.59268346384715, 43.92093191759076, 44.76236959055745, 45.47836699035672, 45.97040783608053, 46.23170077916227, 47.78096646019102, 49.14043807247112, 50.49990968475122, 51.27568096107365, 51.23799673619405, 50.78357928578467, 50.88701606016795, 50.876535840620114, 50.86605562107228, 50.85557540152444, 50.8450951819766], [10.0, 11.732050807568877, 13.687959389535372, 15.094303078322268, 17.07777500326464, 19.0067243057353, 20.144767839589605, 18.16128735828983, 17.79408638061494, 17.426885402940048, 17.059684425265157, 17.387040625309314, 19.014910520453313, 20.849522264696603, 22.589166625993474, 24.131037890653847, 25.512873064443614, 25.755488934863795, 25.24067189517893, 25.069269479249126, 23.740072165510732, 23.4704082150367, 23.200744264562665, 24.2680056533139, 25.5482290459846, 27.075824458478387, 27.645037954242284, 29.046412938340318, 30.74958921513317, 31.508096698497276, 31.516275621791014, 31.231668902960926, 31.97501534658999, 33.512348732233356, 34.3301868043383, 35.62201669262136, 33.67351630132499, 33.50204742980957, 33.33057855829416, 33.15910968677874, 34.00693131288008, 35.209764325191074, 36.01387112934762, 37.21670414165862, 38.020810945815164, 39.22364395812616, 40.02775076228271, 41.230583774593704, 41.01134447459482, 41.0691195577439, 41.126894640892985, 41.18466972404207, 41.24244480719115, 41.30021989034023, 41.35799497348931, 42.40761233109002, 44.20071694809307, 43.031940990290906, 42.04216380899408, 40.34079199631978, 38.34225144679251, 36.34371089726524, 34.34517034773797, 32.3466297982107, 30.34808924868343, 28.34954869915616, 26.368105214434447, 24.36845776202163, 22.378206292161146, 20.395800273280667, 18.401823279408028, 18.17545243071068, 18.161433606846074, 18.147414782981468, 18.480896760123358, 19.92764066732364, 21.42288514403229, 23.237267309539984, 25.104711478312023, 27.043240767783654, 29.026098805540187, 30.290921263273496, 31.75783534917101, 33.224749435068524, 35.06816445259254, 37.06780939587503, 39.01550156374023, 41.01282498114293, 43.01279752220399, 45.01277006326505, 47.012742604326114, 49.012715145387176], [np.float64(1.5707963267948966), np.float64(1.1858961473351464), np.float64(1.3034474678074566), np.float64(0.9185472883477064), np.float64(1.303447467807457), np.float64(1.303447467807457), np.float64(0.7440477136892705), np.float64(0.1846479595710835), np.float64(0.1846479595710835), np.float64(0.1846479595710835), np.float64(0.1846479595710835), np.float64(0.42729478241521823), np.float64(0.8121949618749689), np.float64(1.054841784719104), np.float64(1.054841784719104), np.float64(0.9372904642467943), np.float64(0.8197391437744845), np.float64(0.260339389656298), np.float64(0.260339389656298), np.float64(0.37789071012860775), np.float64(0.13524388728447256), np.float64(0.13524388728447256), np.float64(0.13524388728447256), np.float64(0.6946436414026591), np.float64(0.6946436414026591), np.float64(0.8121949618749693), np.float64(0.4272947824152191), np.float64(0.6699416052593543), np.float64(0.9125884281034895), np.float64(0.5276882486437393), np.float64(0.14278806918398912), np.float64(0.14278806918398912), np.float64(0.5276882486437393), np.float64(0.7703350714878745), np.float64(0.5276882486437398), np.float64(0.64523956911605), np.float64(0.08583981499786297), np.float64(0.08583981499786297), np.float64(0.08583981499786297), np.float64(0.08583981499786297), np.float64(0.47073999445761316), np.float64(0.5882913149299234), np.float64(0.4707399944576136), np.float64(0.5882913149299238), np.float64(0.47073999445761405), np.float64(0.5882913149299243), np.float64(0.4707399944576145), np.float64(0.5882913149299247), np.float64(0.02889156081173816), np.float64(0.02889156081173816), np.float64(0.02889156081173816), np.float64(0.02889156081173816), np.float64(0.02889156081173816), np.float64(0.02889156081173816), np.float64(0.02889156081173816), np.float64(0.4137917402714888), np.float64(0.9731914943896758), np.float64(1.2158383172338105), np.float64(1.775238071351997), np.float64(1.5325912485078619), np.float64(1.5325912485078619), np.float64(1.5325912485078619), np.float64(1.5325912485078619), np.float64(1.5325912485078619), np.float64(1.5325912485078619), np.float64(1.5325912485078619), np.float64(1.4150399280355517), np.float64(1.2974886075632415), np.float64(1.1799372870909313), np.float64(0.7950371076311806), np.float64(0.23563735351299364), np.float64(-0.007009469331141105), np.float64(-0.007009469331141105), np.float64(-0.007009469331141105), np.float64(0.1105418511411691), np.float64(0.6699416052593561), np.float64(0.7874929257316663), np.float64(1.0301397485758015), np.float64(1.1476910690481117), np.float64(1.265242389520422), np.float64(1.382793709992732), np.float64(0.8233939558745456), np.float64(0.8233939558745456), np.float64(0.8233939558745456), np.float64(1.0660407787186807), np.float64(1.4509409581784314), np.float64(1.6935877810225666), np.float64(1.5760364605502568), np.float64(1.5760364605502568), np.float64(1.5760364605502568), np.float64(1.5760364605502568), np.float64(1.5760364605502568)]# hybrid_a_star.planning()
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
        plt.pause(1)
    plt.show()
