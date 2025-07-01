import sys
import os

from pytesseract.pytesseract import is_valid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
import matplotlib.pyplot as plt
from common.GridMap2D import GridMap2D
from common.Node import Node

class AStar:
    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.motions = [
            [0, 1, 1],
            [0, -1, 1],
            [1, 0, 1],
            [-1, 0, 1],
            [1, 1, math.sqrt(2)],
            [1, -1, math.sqrt(2)],
            [-1, 1, math.sqrt(2)],
            [-1, -1, math.sqrt(2)],
        ]

    def planning(self, start_pos, goal_pos):
        start_node = Node(self.calc_xy_index(start_pos[0], self.grid_map.min_x),
                        self.calc_xy_index(start_pos[1], self.grid_map.min_y))
        goal_node = Node(self.calc_xy_index(goal_pos[0], self.grid_map.min_x),
                          self.calc_xy_index(goal_pos[1], self.grid_map.min_y))

        open_set = dict()
        closed_set = dict()

        open_set[self.calc_grid_index(start_node)] = start_node
        while len(open_set):
            curr_id = min(open_set, key=lambda o: open_set[o].g + open_set[o].h)
            curr = open_set[curr_id]

            # show graph
            # pragma: no cover
            plt.plot(self.grid_map.calc_grid_position(curr.x, self.grid_map.min_x),
                     self.grid_map.calc_grid_position(curr.y, self.grid_map.min_y), "xc")
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(
                                             0) if event.key == 'escape' else None])
            if len(closed_set.keys()) % 10 == 0:
                plt.pause(0.001)

            del open_set[curr_id]
            closed_set[curr_id] = curr

            if curr.x == goal_node.x and curr.y == goal_node.y:
                goal_node.p_idx = curr_id
                break

            for motion in self.motions:
                mx, my, cost = motion
                neighbor_node = Node(curr.x + mx,
                                     curr.y + my,
                                     curr.g + cost,
                                     self.heuristic(goal_node, curr.x + mx, curr.y + my),
                                     curr_id)

                neighbor_id = self.calc_grid_index(neighbor_node)

                if not self.grid_map.verify_node(neighbor_node):
                    continue

                if neighbor_id in closed_set and closed_set[neighbor_id].g > neighbor_node.g:
                    del closed_set[neighbor_id]

                if neighbor_id in open_set and open_set[neighbor_id].g > neighbor_node.g:
                    del open_set[neighbor_id]

                if neighbor_id not in open_set and neighbor_id not in closed_set:
                    open_set[neighbor_id] = neighbor_node

        return self.final_path(goal_node, closed_set)

    def final_path(self, goal, closed_set):
        # generate final course
        rx, ry = [self.grid_map.calc_grid_position(goal.x, self.grid_map.min_x)], [
            self.grid_map.calc_grid_position(goal.y, self.grid_map.min_y)]
        parent_index = goal.p_idx

        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.grid_map.calc_grid_position(n.x, self.grid_map.min_x))
            ry.append(self.grid_map.calc_grid_position(n.y, self.grid_map.min_y))
            parent_index = n.p_idx


        return rx, ry

    @staticmethod
    def heuristic(goal, nx, ny):
        return abs(goal.x - nx) + abs(goal.y - ny)

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.grid_map.resolution)

    def calc_grid_index(self, node):
        return ((node.y - self.grid_map.min_y) *
                self.grid_map.x_width + (node.x - self.grid_map.min_x))


if __name__ == "__main__":

    # start and goal position
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]
    grid_size = 2.0  # [m]
    robot_radius = 1.0  # [m]

    # set obstacle positions
    obs = []
    start = -10
    end = 60

    # outline
    for i in range(start, end):
        obs.append((i, float(start)))
    for i in range(start, end):
        obs.append((float(end), i))
    for i in range(start, end + 1):
        obs.append((i, float(end)))
    for i in range(start, end + 1):
        obs.append((float(start), i))

    for i in range(-10, 40):
        obs.append((20.0, i))
    for i in range(0, 40):
        obs.append((40.0, 60.0 - i))
    grid_map = GridMap2D(2.0, 1.0, obs)
    plt.grid(True)
    plt.axis("equal")
    # sx = float(input("start x: "))
    # sy = float(input("start y: "))
    # gx = float(input("goal x: "))
    # gy = float(input("goal y: "))
    plt.plot(sx, sy, "og")
    plt.plot(gx, gy, "xb")

    a_star = AStar(grid_map)
    rx, ry = a_star.planning((sx, sy), (gx, gy))
    print(rx, ry)
    plt.plot(rx, ry, "-r")
    plt.pause(0.001)
    plt.show()