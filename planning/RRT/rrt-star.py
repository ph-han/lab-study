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

    def is_same(self, other, eps=1e-6):
        return abs(self.x - other.x) < eps and abs(self.y - other.y) < eps

class RRTStar:
    def __init__(self, seed, init_pos, goal_pos, iter_num, cspace, near_distance=7, robot_radius=1, expand_size=2):
        self.init_x, self.init_y = init_pos
        self.goal_x, self.goal_y = goal_pos
        self.iter_num = iter_num
        self.near_distance = near_distance
        self.cspace = cspace
        self.is_goal = False
        self.robot_radius = robot_radius
        self.expand_size = expand_size
        self.seed = seed
        self.paths = []

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
    
    def steer(self, near, rand):
        dist = np.hypot(rand.x - near.x, rand.y - near.y)
        theta = np.arctan2(rand.y - near.y, rand.x - near.x)
        if  dist <= self.expand_size:
            new_cost = near.cost + dist
            return Node((rand.x, rand.y), parent=near, cost=new_cost), dist
        
        new_x = near.x + self.expand_size * np.cos(theta)
        new_y = near.y + self.expand_size * np.sin(theta)
        new_cost = near.cost + self.expand_size
        return Node((new_x, new_y), parent=near, cost=new_cost), self.expand_size
    

    def is_collision(self, node):
        is_outside = node.x < self.cspace.min_x or node.x > self.cspace.max_x or node.y < self.cspace.min_y  or node.y > self.cspace.max_y

        if is_outside:
            return True

        for obs in self.cspace.obstacles:
            dist = np.hypot(node.x - obs[0], node.y - obs[1])

            if dist < self.robot_radius + obs[2]:
                return True
            
        return False
    
    def get_near_ids(self, new):
        node_idxs = []

        for node in self.paths:
            dist = np.hypot(node.x - new.x, node.y - new.y)
            if dist <= self.near_distance:
                node_idxs.append(self.paths.index(node))

        return node_idxs
    
    def choose_parent(self, near_by_vertices, nearest, new):
        candi_parent = nearest
        cost_min = new.cost

        for near_id in near_by_vertices:
            near = self.paths[near_id]
            t_node, t_cost = self.steer(near, new)
            if not self.is_collision(t_node) and new.is_same(t_node):
                new_cost = near.cost + t_cost
                if new_cost < cost_min and t_node.cost < new.cost:
                    candi_parent = near
                    cost_min = new_cost

        return candi_parent, cost_min

    def update_subtree_cost(self, node):
        for child in self.paths:
            if child.parent is node:
                edge_cost = np.hypot(child.x - node.x, child.y - node.y)
                child.cost = node.cost + edge_cost
                self.update_subtree_cost(child)

    def rewire(self, near_by_vertices, parent, new):
        for near_id in near_by_vertices:
            near = self.paths[near_id]
            if near.is_same(parent):
                continue

            t_node, t_cost = self.steer(new, near)
            if not self.is_collision(t_node) and t_node.is_same(near) and t_cost + new.cost < near.cost:
                near.parent = new
                near.cost = t_cost + new.cost
                self.update_subtree_cost(near)

    def planning(self, is_rewiring=True, is_break=False):
        init_node = Node((self.init_x, self.init_y))

        best_node = None
        self.paths.append(init_node)
        for it in range(self.iter_num):
            rand = self.get_random_node()
            nearest = self.get_nearest_node(rand)

            new, _ = self.steer(nearest, rand)

            if not self.is_collision(new):
                near_by_vertices = self.get_near_ids(new)
                parent, cost = self.choose_parent(near_by_vertices, nearest, new)
                print(f"({nearest.x}, {nearest.y}) -> ({parent.x}, {parent.y})")
                new.parent = parent
                new.cost = cost
                self.paths.append(new)
                if is_rewiring:
                    self.rewire(near_by_vertices, parent, new)
                self.plot_explore_edge(new)
            
            if self.check_goal(new) and (best_node is None or best_node.cost > new.cost):
                self.is_goal = True
                best_node = new
                if is_break:
                    print(f"Find path!")
                    break
            
        if not self.is_goal:
            print(f"No path... more iteration")

        return best_node
    

    def plot_explore_edge(self, new):
        if new.parent is None:
            return

        plt.title("RRT Star")
        plt.axis('equal')
        plt.plot(
            [new.parent.x, new.x],
            [new.parent.y, new.y],
            f"-g", alpha=0.3, zorder=0)
        plt.pause(0.001)

    
def plot_final_path(final_node):
    xlist = []
    ylist = []
    
    curr = final_node
    while curr:
        xlist.append(curr.x)
        ylist.append(curr.y)
        curr = curr.parent

    plt.plot(xlist[::-1], ylist[::-1], '-r')

def run_rrt_star(seed=None, do_plot=True, is_rewiring=True, is_break=False):
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
    cspace_obj = cspace.CSpace(cspace_size, obstacleList)

    if do_plot:
        cspace_obj.plot()

    rrt = RRTStar(seed, init_pos, goal_pos, 1000, cspace_obj)
    final_node = rrt.planning(is_rewiring=is_rewiring, is_break=is_break)

    return final_node, rrt 

def test_rrt_star(seed=None, do_plot=True):
    final_node, rrt = run_rrt_star(seed=seed, do_plot=do_plot)
    path_x, path_y = [], []
    node = final_node
    while node is not None:
        path_x.append(node.x)
        path_y.append(node.y)
        node = node.parent

    path_x = path_x[::-1]
    path_y = path_y[::-1]
    plt.plot(path_x, path_y, "-g", alpha=0.5, linewidth=1.5, label=f"original (cost: {final_node.cost:.2f})")

    final_node, rrt = run_rrt_star(seed=seed, do_plot=do_plot, is_rewiring=False)
    path_x, path_y = [], []
    node = final_node
    while node is not None:
        path_x.append(node.x)
        path_y.append(node.y)
        node = node.parent

    path_x = path_x[::-1]
    path_y = path_y[::-1]
    plt.plot(path_x, path_y, "-r", alpha=0.5, linewidth=1.5, label=f"no rewire (cost: {final_node.cost:.2f})")

    final_node, rrt = run_rrt_star(seed=seed, do_plot=do_plot, is_break=True)
    path_x, path_y = [], []
    node = final_node
    while node is not None:
        path_x.append(node.x)
        path_y.append(node.y)
        node = node.parent

    path_x = path_x[::-1]
    path_y = path_y[::-1]
    plt.plot(path_x, path_y, "-b", alpha=0.5, linewidth=1.5, label=f"break (cost: {final_node.cost:.2f})")


        


if __name__ == "__main__":
    num_runs = 1
    results = []


    for i in range(num_runs):
        seed = 0
        final_node, rrt = run_rrt_star(seed=seed, do_plot=True)
        results.append(final_node.cost)

        path_x, path_y = [], []
        node = final_node
        while node is not None:
            path_x.append(node.x)
            path_y.append(node.y)
            node = node.parent

        print(f"{i+1} step done")
        path_x = path_x[::-1]
        path_y = path_y[::-1]
        plt.plot(path_x, path_y, "-r", zorder=2, linewidth=1.5)

        plt.title("RRT* multiple runs")
        plt.axis("equal")
        plt.pause(0.001)

    # test_rrt_star(83, True)
    # plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.title("RRT*")
    # plt.title("RRT* original, no rewire, break (seed=30)")
    plt.axis("equal")
    plt.show()

    # print("final costs:", results)
    # x = list(range(num_runs))
    # plt.plot(x, results, linewidth=2, marker='o')
    # plt.title("RRT-star Costs", fontsize=14)
    # plt.xlabel("Iteration", fontsize=10)
    # plt.ylabel("Cost", fontsize=10)
    # plt.axis("equal")

    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # plt.show()


