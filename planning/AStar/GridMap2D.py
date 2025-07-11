import random
import math
import matplotlib.pyplot as plt

class GridMap2D:
    def __init__(self, resolution, rr, obs=None):
        self.grid_map = None
        self.resolution = resolution
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.x_width, self.y_width = 0, 0
        self.rr = rr
        self.gen_grid_map(obs)

    def gen_grid_map(self, obs):
        if obs is None:
            obs = self.gen_random_obs()

        ox, oy = zip(*obs)
        plt.plot(ox, oy, ".k")
        plt.grid(True)
        plt.axis("equal")
        plt.savefig('grid.png')
        plt.plot(ox, oy, ".k")
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print(f"min (x,y) = ({self.min_x}, {self.min_y}), max (x,y) = ({self.max_x}, {self.max_y})")

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # obstacle map generation
        self.grid_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.grid_map[ix][iy] = True
                        break

    def calc_grid_position(self, index, start):
        return index * self.resolution + start

    @staticmethod
    def gen_random_obs(num_walls=2, wall_length=30):
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

        # wall
        for _ in range(num_walls):
            start_x = random.randint(start + 5, end - 5)
            start_y = random.randint(start + 5, end - 5)

            direction = random.choice(['h', 'v'])

            for i in range(wall_length):
                obs.append((start_x + i, start_y)) if direction == 'h' else obs.append((start_x, start_y + i))

        return obs

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.grid_map[node.x][node.y]:
            return False

        return True

