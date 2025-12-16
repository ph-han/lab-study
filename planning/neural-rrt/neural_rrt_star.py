import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import time
import os
from train import NeuralRRTStarNet
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

    def is_same(self, other, eps=1e-6):
        return abs(self.x - other.x) < eps and abs(self.y - other.y) < eps

class NeuralRRTStar:
    def __init__(self, seed, clearance, step_size, iter_num, cspace_img_path, is_neural_mode=True, near_distance=7, robot_radius=1, expand_size=2):
        self.clearance = clearance
        self.step_size = step_size
        self.init_x, self.init_y = None, None
        self.goal_x, self.goal_y = None, None
        self.iter_num = iter_num
        self.near_distance = near_distance
        self.cspace_img_path = cspace_img_path
        self.is_goal = False
        self.robot_radius = robot_radius
        self.expand_size = expand_size
        self.seed = seed
        self.is_neural_mode = is_neural_mode
        self.paths = []

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

            # print(f"model inference time: {end - start:.6f} sec")
        im_np  = input_map.squeeze(0).cpu().permute(1, 2, 0).numpy()
        out = None

        start_mask = (im_np[...,0] == 1.0) & (im_np[...,1] == 0.0) & (im_np[...,2] == 0.0)
        goal_mask  = (im_np[...,0] == 0.0) & (im_np[...,1] == 1.0) & (im_np[...,2] == 0.0)

        start_coords = np.argwhere(start_mask)
        goal_coords  = np.argwhere(goal_mask)

        self.init_y, self.init_x = start_coords[0]
        self.goal_y, self.goal_x = goal_coords[0]

        self.plot_map = im_np.copy()
        # draw map info and predicted probaility
        if self.is_neural_mode:
            out = output.squeeze().squeeze().cpu().numpy()
            if is_draw:
                plt.cla()
                im = plt.imshow(im_np, interpolation='nearest')
                alpha = out.copy()
                alpha[out < 0.5] = 0.0
                alpha = np.clip(alpha * 2.0, 0.0, 0.8)

                plt.imshow(out, cmap='plasma', alpha=alpha, interpolation='bilinear')
                cbar = plt.colorbar(im)
                cbar.set_label("Predicted sampling probability")
        if is_draw:
            if not self.is_neural_mode:
                plt.imshow(im_np, interpolation='nearest')
            plt.plot(self.init_x, self.init_y, 'ob')
            plt.plot(self.goal_x, self.goal_y, 'xr')
        return out, total_time
    
    def check_goal(self, curr):
        return curr.x == self.goal_x and curr.y == self.goal_y
    
    def prepare_non_uniform(self, threshold=0.5):
        h, w = self.non_uniform_map.shape
        flat = self.non_uniform_map.reshape(-1)
        flat[flat < threshold] = 0

        print("sum:", flat.sum())
        print("min:", flat.min(), "max:", flat.max())

        if flat.sum() == 0:
            flat = np.ones_like(flat)
    
        prob = flat / flat.sum()

        self.flat_prob = prob
        self.h, self.w = h, w

    def sample_from_non_uniform_map(self):
        idx = np.random.choice(len(self.flat_prob), p=self.flat_prob)

        row = idx // self.w
        col = idx % self.w
        return row, col

    def get_random_node(self):

        if self.is_neural_mode and random.random() > 0.5:
            y, x = self.sample_from_non_uniform_map()
        else:
            if random.randint(0, 100) > 10:
                x = random.uniform(1, 224)
                y = random.uniform(1, 224)
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
        is_outside = node.x <= 0 or node.x >= 223 or node.y <= 0 or node.y >= 223

        if is_outside:
            return True

        x = int(round(node.x))
        y = int(round(node.y))
        clr = int(round(self.clearance))

        for i in range(x - clr, x + clr + 1):
            for j in range(y - clr, y + clr + 1):
                if 0 <= i < 224 and 0 <= j < 224:
                    if self.grid_map[j, i] == 1:
                        return True
                else:
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

    def planning(self, model=None, device=None, is_rewiring=True, is_break=False, is_draw=False):

        if self.is_neural_mode:
            assert model is not None and device is not None, "Neural mode needs model & device"
            self.non_uniform_map, gpu_time = self.neural_model(model, device, is_draw)
        else:
            _, gpu_time = self.neural_model(None, None, is_draw)
            self.non_uniform_map = np.ones_like(self.grid_map, dtype=np.float32)
        # return None, None
        self.prepare_non_uniform()
        init_node = Node((self.init_x, self.init_y))
        best_node = None
        self.paths = [init_node]
        
        start = time.perf_counter()
        for _ in range(self.iter_num):
            rand = self.get_random_node()
            nearest = self.get_nearest_node(rand)

            new, _ = self.steer(nearest, rand)

            if not self.is_collision(new):
                near_by_vertices = self.get_near_ids(new)
                parent, cost = self.choose_parent(near_by_vertices, nearest, new)
                new.parent = parent
                new.cost = cost
                self.paths.append(new)
                if is_rewiring:
                    self.rewire(near_by_vertices, parent, new)
                if is_draw:
                    self.plot_explore_edge(new)

            if self.check_goal(new) and (best_node is None or best_node.cost > new.cost):
                self.is_goal = True
                best_node = new
                if is_break:
                    break
        end = time.perf_counter()
        if not self.is_goal:
            return None, None
        cpu_time = end - start
        return best_node, (cpu_time + gpu_time)
    

    def plot_explore_edge(self, new):
        if new.parent is None:
            return

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

if __name__ == "__main__":
    neural_mode = True
    plt.title("Neural RRT Star")
    plt.axis('off')
    set_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralRRTStarNet().to(device)
    model.load_state_dict(torch.load("best_neural_rrt_star_net_loss.pth"))
    model.eval()
    print("Using device:", device)
    plt.cla()
    map_path = "./dataset/test/maps/000726.png"
    # map_path = "./dataset/test/maps/000455.png"
    # map_path = "./dataset/test/maps/custom_map.png"
    neural_planner = NeuralRRTStar(42, 1, 4, 7000, map_path,
                                   is_neural_mode=neural_mode, expand_size=4)
    neural_node, tot_time_neural = neural_planner.planning(model=model, device=device, is_rewiring=True, is_break=True, is_draw=True)
    print(f"total time: {tot_time_neural:.6f}sec, node: {len(neural_planner.paths)}, cost: {neural_node.cost:.2f}")
    
    if neural_node:
        plot_final_path(neural_node)
    plt.show()