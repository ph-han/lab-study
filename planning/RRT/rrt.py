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
    def __init__(self, init_pos, goal_pos, iter_num, cspace, robot_radius=1, expand_size=2):
        self.init_x, self.init_y = init_pos
        self.goal_x, self.goal_y = goal_pos
        self.iter_num = iter_num
        self.cspace = cspace
        self.is_goal = False
        self.robot_radius = robot_radius
        self.expand_size = expand_size

        self.rrt_paths = []

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
        for node in self.rrt_paths:
            dist = np.hypot(node.x - rand.x, node.y - rand.y)
            if nearest_node_dist > dist:
                nearest_node = node
                nearest_node_dist = dist
        
        return nearest_node
    
    def select_input(self, rand, near):
        dist = np.hypot(rand.x - near.x, rand.y - near.y)
        theta = np.arctan2(rand.y - near.y, rand.x - near.x)
        if  dist < self.expand_size:
            return Node((rand.x, rand.y), parent=near, cost=dist)
        
        new_x = near.x + self.expand_size * np.cos(theta)
        new_y = near.y + self.expand_size * np.sin(theta)
        new = Node((new_x, new_y), near, self.expand_size)
        return new
    

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

        self.rrt_paths.append(init_node)
        for it in range(self.iter_num):
            print(f"Step {it + 1} start!")
            rand = self.get_random_node()
            near = self.get_nearest_node(rand)

            new = self.select_input(rand, near)

            if self.is_collision(new):
                continue

            self.rrt_paths.append(new)
            plt.plot([near.x, new.x], [near.y, new.y], '-g')
            plt.title("RRT")
            plt.axis('equal')
            plt.pause(0.001)
            
            if self.check_goal(new):
                self.is_goal = True
                print(f"Find path!")
                break
            
        if not self.is_goal:
            print(f"No path... more iteration")

        return self.rrt_paths[-1]
    
def plot_final_path(final_node):
    xlist = []
    ylist = []
    
    curr = final_node
    while curr:
        xlist.append(curr.x)
        ylist.append(curr.y)
        curr = curr.parent

    plt.plot(xlist[::-1], ylist[::-1], '-r')


if __name__ == "__main__":
    init_pos = (-15, -10)
    goal_pos = (7, 8)
    obstacleList = [
        (5, 5, 1), (3, 6, 2), (3, 8, 2),
        (3, 10, 2), (7, 5, 2), (9, 5, 2), 
        (8, 10, 0.3), (0, -1, 1), (-10, 10, 2),
        (10, -15, 1), (-10, -8, 3), (15, -7, 2),
        (3, -13, 2), (5, -7, 2), (-13, 2, 2)
    ]
    cspace_size = (20, 20)
    cspace = cspace.CSpace(cspace_size, obstacleList) # only 2 dimension
    cspace.plot()

    rrt = RRT(init_pos, goal_pos, 1000, cspace)
    final_node = rrt.planning()
    print(final_node.x, final_node.y)
    plot_final_path(final_node)

    plt.title("RRT")
    plt.axis('equal')
    plt.show()