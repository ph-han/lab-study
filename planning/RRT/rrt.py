import random
import numpy as np
import matplotlib.pyplot as plt
import cspace
import time

class Node:
    def __init__(self, position, parent=None, cost=0):
        self.x, self.y = position
        self.parent = parent
        self.cost = cost

class RRT:
    def __init__(self, init_pos, goal_pos, iter_num, cspace, robot_radius=1, expand_size=2, seed=None, do_plot=True):
        self.init_x, self.init_y = init_pos
        self.goal_x, self.goal_y = goal_pos
        self.iter_num = iter_num
        self.cspace = cspace
        self.is_goal = False
        self.robot_radius = robot_radius
        self.expand_size = expand_size
        self.seed = seed
        self.do_plot = do_plot

        self.paths = []

        if self.do_plot:
            plt.plot(self.init_x, self.init_y, 'ob')
            plt.plot(self.goal_x, self.goal_y, 'xr')
    

    def check_goal(self, curr):
        return curr.x == self.goal_x and curr.y == self.goal_y
    
    def get_random_node(self):
        if random.randint(0, 100) > 30:
            x = random.uniform(self.cspace.min_x, self.cspace.max_x)
            y = random.uniform(self.cspace.min_y, self.cspace.max_y)
        else:
            x = self.goal_x
            y = self.goal_y
        node = Node((x, y))
        return node
    
    def get_nearest_node(self, rand):
        nearest_node_dist = np.inf
        nearest_node = None
        for node in self.paths:
            dist = np.hypot(node.x - rand.x, node.y - rand.y)
            if nearest_node_dist > dist:
                nearest_node = node
                nearest_node_dist = dist
        
        return nearest_node
    
    def steer(self, rand, near):
        dist = np.hypot(rand.x - near.x, rand.y - near.y)
        theta = np.arctan2(rand.y - near.y, rand.x - near.x)
        if  dist < self.expand_size:
            new_cost = near.cost + dist
            return Node((rand.x, rand.y), parent=near, cost=new_cost)
        
        new_x = near.x + self.expand_size * np.cos(theta)
        new_y = near.y + self.expand_size * np.sin(theta)
        new_cost = near.cost + self.expand_size
        return Node((new_x, new_y), parent=near, cost=new_cost)
    
    def is_collision(self, node):
        is_outside = node.x < self.cspace.min_x or node.x > self.cspace.max_x or node.y < self.cspace.min_y  or node.y > self.cspace.max_y

        if is_outside:
            return True

        for obs in self.cspace.obstacles:
            dist = np.hypot(node.x - obs[0], node.y - obs[1])

            if dist < self.robot_radius + obs[2]:
                return True
            
        return False

    def planning(self):
        init_node = Node((self.init_x, self.init_y))

        self.paths.append(init_node)
        for it in range(self.iter_num):
            rand = self.get_random_node()
            near = self.get_nearest_node(rand)

            new = self.steer(rand, near)

            if self.is_collision(new):
                continue

            self.paths.append(new)
            self.plot_explore_edge(near, new)
            
            if self.check_goal(new):
                self.is_goal = True
                # print(f"Find path!")
                break
            
        if not self.is_goal:
            print(f"No path... more iteration")

        return self.paths[-1]
    
    def plot_explore_edge(self, near, new):
        if not self.do_plot or near is None:
            return

        plt.plot([near.x, new.x], [near.y, new.y], '-g', alpha=0.4, zorder=0)
        plt.title("RRT")
        plt.axis('equal')
        plt.pause(0.001)
    
def extract_path(final_node):
    xlist = []
    ylist = []

    curr = final_node
    while curr:
        xlist.append(curr.x)
        ylist.append(curr.y)
        curr = curr.parent

    return xlist[::-1], ylist[::-1]


def plot_final_path(final_node, style='-r', **kwargs):
    xlist, ylist = extract_path(final_node)
    plt.plot(xlist, ylist, style, **kwargs)


def run_rrt(iter_num=1000, seed=None, do_plot=True):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    init_pos = (-15, -10)
    goal_pos = (7, 8)
    # goal_pos = (10, 15)
    obstacleList = [
        (5, 5, 1), (3, 6, 2), (3, 8, 2),
        (3, 10, 2), (7, 5, 2), (9, 5, 2), 
        (8, 10, 0.3), (0, -1, 1), (-10, 10, 2),
        (10, -15, 1), (-10, -8, 3), (15, -7, 2),
        (3, -13, 2), (5, -7, 2), (-13, 2, 2)
    ]
    cspace_size = (20, 20)
    cspace_obj = cspace.CSpace(cspace_size, obstacleList) # only 2 dimension

    if do_plot:
        cspace_obj.plot()

    rrt = RRT(init_pos, goal_pos, iter_num, cspace_obj, seed=seed, do_plot=do_plot)
    final_node = rrt.planning()

    if do_plot:
        plt.title("RRT")
        plt.axis('equal')

    return final_node, rrt


if __name__ == "__main__":
    num_runs = 1
    seeds = []
    results = []

    for i in range(num_runs):
        seed = i * 10
        seeds.append(seed)
        final_node, _ = run_rrt(seed=seed, do_plot=(i == 0))
        results.append(final_node.cost)

        path_x, path_y = extract_path(final_node)
        plt.plot(path_x, path_y, "-", zorder=2, linewidth=1.5)

    plt.title("RRT multiple runs")
    plt.axis("equal")
    plt.show()

    print("seeds:", seeds)
    print("final costs:", results)
    
    
    x = list(range(num_runs))
    plt.plot(x, results, linewidth=2, marker='o')

    plt.title("RRT Costs", fontsize=14)
    plt.xlabel("Iteration", fontsize=10)
    plt.ylabel("Cost", fontsize=10)
    plt.axis("equal")

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
