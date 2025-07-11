class Node:
    def __init__(self, x, y, g=0.0, h=0.0, p_idx=-1):
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.p_idx = p_idx # parent index

    def __str__(self):
        return f"x: {self.x}, y: {self.y}, g: {self.g}, h: {self.h},  p_idx: {self.p_idx}"