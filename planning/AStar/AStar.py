import math
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y, cost, p_idx=-1):
        self.x = x
        self.y = y
        self.cost = cost
        self.p_idx = p_idx # parent index

    def __str__(self):
        return f"x: {self.x}, y: {self.y}, cost: {self.cost}, p_idx: {self.p_idx}"


class AStar:
    def __init__(self):
        pass

    def planning(self, start_pos, goal_pos):
        pass

if __name__ == "__main__":
    a_star = AStar()
    a_star.planning((0, 0), (10, 10))