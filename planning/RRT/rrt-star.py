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
    def __init__(self, init_pos, goal_pos, iter_num, cspace, near_distance=5, robot_radius=1, expand_size=2):
        self.init_x, self.init_y = init_pos
        self.goal_x, self.goal_y = goal_pos
        self.iter_num = iter_num
        self.near_distance = near_distance
        self.cspace = cspace
        self.is_goal = False
        self.robot_radius = robot_radius
        self.expand_size = expand_size

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
    
    def steer(self, rand, near):
        dist = np.hypot(rand.x - near.x, rand.y - near.y)
        theta = np.arctan2(rand.y - near.y, rand.x - near.x)
        if  dist <= self.expand_size:
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
    
    def get_near(self, new):
        nodes = []

        for node in self.paths:
            dist = np.hypot(node.x - new.x, node.y - new.y)
            if dist <= self.near_distance:
                nodes.append(node)

        return nodes
    
    def choose_parent(self, near_by_vertices, nearest, new):
        candi_parent = nearest
        cost_min = nearest.cost + new.cost

        for near in near_by_vertices:
            candi = self.steer(new, near)
            if candi and not self.is_collision(candi):
                new_cost = near.cost + candi.cost
                if new_cost < cost_min and candi.cost < new.cost:
                    candi_parent = near
                    cost_min = new_cost

        return candi_parent


    def rewire(self, near_by_vertices, parent, new):
        pass

    def planning(self):
        init_node = Node((self.init_x, self.init_y))

        self.paths.append(init_node)
        for it in range(self.iter_num):
            rand = self.get_random_node()
            nearest = self.get_nearest_node(rand)

            new = self.steer(rand, nearest)

            if not self.is_collision(new):
                near_by_vertices = self.get_near()
                parent = self.choose_parent(near_by_vertices, nearest, new)
                self.rewire(near_by_vertices, parent, new)
                plt.plot([nearest.x, new.x], [nearest.y, new.y], '-g')
                plt.title("RRT Star")
                plt.axis('equal')
                plt.pause(0.001)
            
            if self.check_goal(new):
                self.is_goal = True
                print(f"Find path!")
                break
            
        if not self.is_goal:
            print(f"No path... more iteration")

        return self.paths[-1]
    
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

    plt.title("RRT Star")
    plt.axis('equal')
    plt.show()
