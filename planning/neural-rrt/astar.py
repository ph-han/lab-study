import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class Node:
    def __init__(self, x, y, g=0.0, h=0.0, p_idx=-1):
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.p_idx = p_idx # parent index

    def __str__(self):
        return f"x: {self.x}, y: {self.y}, g: {self.g}, h: {self.h},  p_idx: {self.p_idx}"

class AStar:
    def __init__(self, map_data, clearance, step_size):
        self.y_width, self.x_width = map_data.shape
        self.grid_map, self.sx, self.sy, self.gx, self.gy = self.read_grid_map(map_data, self.x_width, self.y_width)
        self.clearance = clearance
        self.step_size = step_size
        self.motions = [
            [0, step_size, step_size],
            [0, -step_size, step_size],
            [step_size, 0, step_size],
            [-step_size, 0, step_size],
            [step_size, step_size, math.sqrt(2) * step_size],
            [step_size, -step_size, math.sqrt(2) * step_size],
            [-step_size, step_size, math.sqrt(2) * step_size],
            [-step_size, -step_size, math.sqrt(2) * step_size],
        ]

    def read_grid_map(self, map_data, x_width, y_width):
        grid_map = np.zeros((y_width, x_width))
        sx, sy, gx, gy = -1, -1, -1, -1

        for y in range(y_width):
            for x in range(x_width):
                if map_data[y, x] == 1:
                    grid_map[y, x] = 1
                elif map_data[y, x] == 2:
                    sx, sy = x, y
                elif map_data[y, x] == 3:
                    gx, gy = x, y
                        
        return grid_map, sx, sy, gx, gy

    def planning(self):
        start_node = Node(self.sx, self.sy, 0.0, self.heuristic(self.gx, self.gy, self.sx, self.sy))
        goal_node = Node(self.gx, self.gy, 0.0, 0.0)

        open_set = dict()
        closed_set = dict()

        open_set[self.calc_grid_index(start_node)] = start_node
        while len(open_set):
            curr_id = min(open_set, key=lambda o: open_set[o].g + open_set[o].h, default=None)
            if curr_id is None:
                return [], []
            
            curr = open_set[curr_id]

            # plt.plot(curr.x, curr.y, "xc")
            # # for stopping simulation with the esc key.
            # plt.gcf().canvas.mpl_connect('key_release_event',
            #                              lambda event: [exit(
            #                                  0) if event.key == 'escape' else None])
            # if len(closed_set.keys()) % 10 == 0:
            #     plt.pause(0.01)

            del open_set[curr_id]
            closed_set[curr_id] = curr

            if self.check_goal(curr, goal_node):
                goal_node.p_idx = curr_id
                break

            for motion in self.motions:
                mx, my, cost = motion
                neighbor_node = Node(curr.x + mx,
                                     curr.y + my,
                                     curr.g + cost,
                                     self.heuristic(goal_node.x, goal_node.y, curr.x + mx, curr.y + my),
                                     curr_id)

                neighbor_id = self.calc_grid_index(neighbor_node)

                if not self.verify_node(neighbor_node):
                    continue

                if neighbor_id in closed_set:
                    continue

                if neighbor_id in open_set and open_set[neighbor_id].g > neighbor_node.g:
                    del open_set[neighbor_id]

                if neighbor_id not in open_set and neighbor_id not in closed_set:
                    open_set[neighbor_id] = neighbor_node
        rx, ry = self.final_path(goal_node, closed_set)
        return self.get_gt_map(rx, ry)
    
    def check_goal(self, curr, goal_node):
        return np.hypot(curr.x - goal_node.x, curr.y - goal_node.y) <= self.clearance


    def final_path(self, goal, closed_set):
        rx, ry = [goal.x], [goal.y]
        parent_index = goal.p_idx

        while parent_index != -1:
            if parent_index not in closed_set:
                print(f"error: {parent_index}, {n.x} {n.y}")
                break
            n = closed_set[parent_index]
            rx.append(n.x)
            ry.append(n.y)
            parent_index = n.p_idx

        if len(rx) == 1:
            return [], []

        return rx, ry
    
    def get_gt_map(self, rx, ry):
        if not rx:
            return None, rx, ry
        gt_map = np.zeros((self.y_width, self.x_width), dtype=np.uint8)
        
        for x, y in zip(rx, ry):
            for i in range(x - 1, x + 2):
                for j in range(y - 1, y + 2):
                    if 1 <= i < self.x_width - 1 and 1 <= j < self.y_width - 1:
                        gt_map[j, i] = 1
        return gt_map, rx, ry
    
    def heuristic(self, gx, gy, nx, ny):
        return np.hypot(gx - nx, gy - ny)

    def calc_grid_index(self, node):
        return node.y * self.x_width + node.x
    
    def verify_node(self, node):
        px = node.x
        py = node.y
        
        if px < 0:
            return False
        elif py < 0:
            return False
        elif px >= self.x_width:
            return False
        elif py >= self.y_width:
            return False

        for i in range(px - self.clearance, px + self.clearance + 1):
            for j in range(py - self.clearance, py + self.clearance + 1):
                if 0 <= i < self.x_width and 0 <= j < self.y_width:
                    if self.grid_map[j, i]:
                        return False
                else:
                    return False

        return True

if __name__ == "__main__":
    # 1. Load a map image for testing
    print("Start A* path planning simulation with a map image...")
    # 테스트하고 싶은 맵 이미지 파일 경로를 지정하세요.
    map_path = "./dataset/train/maps/007118.png" 
    try:
        map_image = Image.open(map_path).convert('L') # Grayscale로 열기
        map_data = np.array(map_image)
    except FileNotFoundError:
        print(f"오류: 맵 파일을 찾을 수 없습니다. 경로: {map_path}")
        print("데이터셋이 생성되었는지, 파일 경로가 올바른지 확인해주세요.")
        exit()


    # 2. Set parameters and run A*
    clearance = 4
    step_size = 2
    a_star = AStar(map_data, clearance, step_size)
    plt.figure(figsize=(8,8))
        # Draw map, obstacles, start and goal
    plt.imshow(map_data, cmap='gray_r', origin='lower')
    plt.plot(a_star.sx, a_star.sy, "og", label='Start')
    plt.plot(a_star.gx, a_star.gy, "xb", label='Goal')
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    _, rx, ry = a_star.planning()

    # 3. Visualize the result
    if rx: # if path is found
        
        
        # Draw path
        plt.plot(rx, ry, "-r", label='A* Path')
        plt.legend()
        plt.show()
    else:
        print("Path not found")



