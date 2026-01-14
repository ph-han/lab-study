import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import time
import os
from collections import deque  # 비재귀적 업데이트를 위해 추가
from train_recon2 import NeuralRRTStarNet
from tqdm.auto import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Node:
    def __init__(self, position, parent=None, cost=0):
        self.x, self.y = position
        self.parent = parent
        self.cost = cost
        self.children = []  # 효율적인 비용 전파를 위해 자식 노드 리스트 관리

    def is_same(self, other, eps=1e-6):
        return abs(self.x - other.x) < eps and abs(self.y - other.y) < eps

class NeuralRRTStar:
    def __init__(self, seed, clearance, step_size, iter_num, cspace_img_path, is_neural_mode=True, near_distance=15, robot_radius=1, expand_size=2):
        self.clearance = clearance
        self.step_size = step_size
        self.init_x, self.init_y = None, None
        self.goal_x, self.goal_y = None, None
        self.iter_num = iter_num
        self.near_distance = near_distance  # 최대 탐색 반경(초기값)
        self.cspace_img_path = cspace_img_path
        self.is_goal = False
        self.robot_radius = robot_radius
        self.expand_size = expand_size
        self.seed = seed
        self.is_neural_mode = is_neural_mode
        self.paths = []
        
        # 동적 반경 계산을 위한 상수 (환경 크기에 따라 조절 가능)
        self.gamma_rrt = 20.0 

    def neural_model(self, model, device, is_draw):
        resize = transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)
        map_image = Image.open(self.cspace_img_path).convert('L')
        
        map_image = resize(map_image)
        map_numpy = np.array(map_image)
        self.grid_map = map_numpy.copy()
        h, w = map_numpy.shape
        rgb = np.ones((h, w, 3), dtype=np.float32)

        rgb[map_numpy == 1] = [0.0, 0.0, 0.0]
        rgb[map_numpy == 2] = [1.0, 0.0, 0.0]
        rgb[map_numpy == 3] = [0.0, 1.0, 0.0]

        rgb = rgb.transpose(2, 0, 1)
        input_map = torch.from_numpy(rgb).unsqueeze(0).to(device)
        input_sc = torch.tensor([self.clearance, self.step_size], dtype=torch.float32).to(device)

        total_time = 0
        if self.is_neural_mode:
            with torch.no_grad():
                start = time.perf_counter()
                output, _ = model(input_map, input_sc)
                end = time.perf_counter()
                total_time = end - start

        im_np  = input_map.squeeze(0).cpu().permute(1, 2, 0).numpy()
        out = None

        start_mask = (im_np[...,0] == 1.0) & (im_np[...,1] == 0.0) & (im_np[...,2] == 0.0)
        goal_mask  = (im_np[...,0] == 0.0) & (im_np[...,1] == 1.0) & (im_np[...,2] == 0.0)

        start_coords = np.argwhere(start_mask)
        goal_coords  = np.argwhere(goal_mask)

        self.init_y, self.init_x = start_coords[0]
        self.goal_y, self.goal_x = goal_coords[0]

        self.plot_map = im_np.copy()
        if self.is_neural_mode:
            out = output.squeeze().squeeze().cpu().numpy()
            if is_draw:
                plt.cla()
                plt.imshow(im_np, interpolation='nearest')
                alpha = out.copy()
                alpha[out < 0.5] = 0.0
                alpha = np.clip(alpha * 2.0, 0.0, 0.8)
                plt.imshow(out, cmap='plasma', alpha=alpha, interpolation='bilinear')
        
        if is_draw:
            plt.plot(self.init_x, self.init_y, 'ob')
            plt.plot(self.goal_x, self.goal_y, 'xr')
            
        return out, total_time

    def prepare_non_uniform(self, threshold=0.5):
        h, w = self.non_uniform_map.shape
        flat = self.non_uniform_map.reshape(-1)
        if flat.sum() == 0:
            flat = np.ones_like(flat)
        flat[flat < threshold] = 0
        self.flat_prob = flat / flat.sum()
        self.h, self.w = h, w

    def sample_from_non_uniform_map(self):
        idx = np.random.choice(len(self.flat_prob), p=self.flat_prob)
        return idx // self.w, idx % self.w

    def get_random_node(self):
        if self.is_neural_mode and random.random() >= 0.5:
            y, x = self.sample_from_non_uniform_map()
        else:
            if random.random() > 0.1:
                x, y = random.uniform(1, 223), random.uniform(1, 223)
            else:
                x, y = self.goal_x, self.goal_y
        return Node((x, y))

    def get_nearest_node(self, rand):
        # 팁: KD-Tree를 쓰면 더 빨라지지만, 현재는 리스트 순회 유지
        distances = [np.hypot(node.x - rand.x, node.y - rand.y) for node in self.paths]
        return self.paths[np.argmin(distances)]

    def steer(self, near, rand):
        dist = np.hypot(rand.x - near.x, rand.y - near.y)
        if dist <= self.expand_size:
            return Node((rand.x, rand.y), parent=near, cost=near.cost + dist), dist
        
        theta = np.arctan2(rand.y - near.y, rand.x - near.x)
        new_x = near.x + self.expand_size * np.cos(theta)
        new_y = near.y + self.expand_size * np.sin(theta)
        return Node((new_x, new_y), parent=near, cost=near.cost + self.expand_size), self.expand_size

    def is_collision(self, node):
        if not (0 <= node.x < 224 and 0 <= node.y < 224): return True
        x, y = int(round(node.x)), int(round(node.y))
        clr = int(round(self.clearance))
        
        # 슬라이싱을 이용한 충돌 체크 최적화
        y_start, y_end = max(0, y-clr), min(224, y+clr+1)
        x_start, x_end = max(0, x-clr), min(224, x+clr+1)
        if np.any(self.grid_map[y_start:y_end, x_start:x_end] == 1):
            return True
        return False

    def get_near_ids(self, new):
        n = len(self.paths)
        r = min(self.gamma_rrt * np.sqrt(np.log(n) / n), self.near_distance)
        
        node_idxs = []
        for i, node in enumerate(self.paths):
            if np.hypot(node.x - new.x, node.y - new.y) <= r:
                node_idxs.append(i)
        return node_idxs

    def choose_parent(self, near_by_vertices, nearest, new):
        candi_parent, cost_min = nearest, new.cost
        for near_id in near_by_vertices:
            near = self.paths[near_id]
            dist = np.hypot(near.x - new.x, near.y - new.y)
            if not self.is_collision(new) and near.cost + dist < cost_min:
                candi_parent, cost_min = near, near.cost + dist
        return candi_parent, cost_min

    def update_subtree_cost(self, start_node):
        queue = deque([start_node])
        while queue:
            curr = queue.popleft()
            for child in curr.children:
                dist = np.hypot(child.x - curr.x, child.y - curr.y)
                child.cost = curr.cost + dist
                queue.append(child)

    def rewire(self, near_by_vertices, parent, new):
        for near_id in near_by_vertices:
            near = self.paths[near_id]
            if near is parent:
                continue

            dist = np.hypot(new.x - near.x, new.y - near.y)
            if new.cost + dist < near.cost:
                if not self.is_collision(near):
                    if near.parent:
                        near.parent.children.remove(near)
                    near.parent = new
                    near.cost = new.cost + dist
                    new.children.append(near)
                    self.update_subtree_cost(near)

    def planning(self, model=None, device=None, is_rewiring=True, is_break=False, is_draw=False):
        if self.is_neural_mode:
            assert model is not None and device is not None
            self.non_uniform_map, gpu_time = self.neural_model(model, device, is_draw)
        else:
            _, gpu_time = self.neural_model(None, None, is_draw)
            self.non_uniform_map = np.ones_like(self.grid_map, dtype=np.float32)
        return None, None
        self.prepare_non_uniform()
        init_node = Node((self.init_x, self.init_y))
        self.paths = [init_node]
        best_node = None
        
        start = time.perf_counter()
        for _ in range(self.iter_num):
            rand = self.get_random_node()
            nearest = self.get_nearest_node(rand)
            new, _ = self.steer(nearest, rand)

            if not self.is_collision(new):
                near_ids = self.get_near_ids(new)
                parent, cost = self.choose_parent(near_ids, nearest, new)
                
                new.parent = parent
                new.cost = cost
                parent.children.append(new) # 자식 관계로 등록
                self.paths.append(new)
                
                if is_rewiring:
                    self.rewire(near_ids, parent, new)
                
                if is_draw:
                    self.plot_explore_edge(new)

                # Goal Check
                dist_to_goal = np.hypot(new.x - self.goal_x, new.y - self.goal_y)
                if dist_to_goal <= self.expand_size:
                    final_node = Node((self.goal_x, self.goal_y), parent=new, cost=new.cost + dist_to_goal)
                    if not self.is_collision(final_node):
                        if best_node is None or final_node.cost < best_node.cost:
                            best_node = final_node
                            self.is_goal = True
                            if is_break: break

        end = time.perf_counter()
        # print(f"cpu time : {end - start}, gpu time : {gpu_time}")
        return best_node, (end - start + gpu_time)

    def plot_explore_edge(self, new):
        if new.parent:
            plt.plot([new.parent.x, new.x], [new.parent.y, new.y], "-g", alpha=0.3)
            if len(self.paths) % 100 == 0: plt.pause(0.001)
            # plt.pause(0.001)

def plot_final_path(final_node):
    xlist, ylist = [], []
    curr = final_node
    while curr:
        xlist.append(curr.x)
        ylist.append(curr.y)
        curr = curr.parent
    plt.plot(xlist[::-1], ylist[::-1], '-r', linewidth=2)

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

if __name__ == "__main__":
    # 설정값
    ITER_MAX = 7000
    EXPAND_SIZE = 4
    CLEARANCE = 1
    
    set_seed(42)
    device = get_device()
    print(f"device: {device}")
    model = NeuralRRTStarNet().to(device)
    state = torch.load("best_neural_rrt_star_net_iou_2.pth", map_location="cpu")
    if next(iter(state)).startswith("module."):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # map_path = "./dataset/test/maps/000726.png"
    map_path = "./dataset/test/maps/custom_map.png"
    planner = NeuralRRTStar(1, CLEARANCE, EXPAND_SIZE, ITER_MAX, map_path, is_neural_mode=True, expand_size=EXPAND_SIZE)
    
    best_node, total_time = planner.planning(model=model, device=device, is_rewiring=True, is_break=True, is_draw=True)
    
    if best_node:
        print(f"Success! Time: {total_time:.4f}s, Nodes: {len(planner.paths)}, Cost: {best_node.cost:.2f}")
        plt.imshow(planner.plot_map)
        plot_final_path(best_node)
        plt.show()
    else:
        plt.show()
        print("Failed to find path.")
