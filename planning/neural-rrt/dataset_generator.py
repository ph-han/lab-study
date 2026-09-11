import csv
import random
import numpy as np
from PIL import Image
from astar import AStar

DATASET_ROOT_SRC = "./dataset/"

def gen_meta_file():
    metas = {}
    for split in ["train", "valid", "test"]:
        f = open(DATASET_ROOT_SRC + split + "/meta.csv", "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["map_id", "sample_id", "clearance", "step_size",
                        "map_path", "gt_path"])
        metas[split] = (f, writer)

    return metas

META_FILES = gen_meta_file()

def map_generator(seed, x_size, y_size):
    random.seed(seed)
    np.random.seed(seed)

    human_visual = False

    map_list = []
    ori_map_data = np.zeros((x_size, y_size), dtype=np.uint8)

    # map outline
    for x in range(x_size):
        for y in range(y_size):
            if x == 0 or x == x_size - 1 or y == 0 or y == y_size - 1: 
                ori_map_data[y, x] = 1 * (80 if human_visual else 1)

    # set random obstacles
    obs_num = random.randint(2, 10)
    for _ in range(obs_num):
        obs_size = random.randint(5, 80)
        obs_x = random.randint(1, x_size - obs_size - 1)
        obs_y = random.randint(1, y_size - obs_size - 1)
        half_size = obs_size // 2
        for x in range(obs_x - half_size, obs_x + half_size + 1):
            for y in range(obs_y - half_size, obs_y + half_size + 1):
                ori_map_data[y, x] = 1 * (80 if human_visual else 1)

    # set random start/goal point (12 diff)
    for _ in range(12):
        map_data = ori_map_data.copy()
        min_dist = random.randint(35, 100)
        start_x = random.randint(1, x_size - 2)
        start_y = random.randint(1, y_size - 2)
        while map_data[start_y, start_x] != 0:
            start_x = random.randint(1, x_size - 2)
            start_y = random.randint(1, y_size - 2)

        map_data[start_y, start_x] = 2 * (80 if human_visual else 1)

        goal_x = random.randint(1, x_size - 2)
        goal_y = random.randint(1, y_size - 2)
        dist = np.hypot(goal_x - start_x, goal_y - start_y)
        while map_data[goal_y, goal_x] != 0 or dist < min_dist:
            goal_x = random.randint(1, x_size - 2)
            goal_y = random.randint(1, y_size - 2)
            dist = np.hypot(goal_x - start_x, goal_y - start_y)
        map_data[goal_y, goal_x] = 3 * (80 if human_visual else 1)
        map_list.append(map_data)


    return map_list

def make_split_map(total_size: int, seed: int = 42):
    random.seed(seed)
    indices = list(range(total_size))
    random.shuffle(indices)

    n_train = total_size * 8 // 10
    n_valid = total_size * 1 // 10

    idx2split = {}
    for i in indices[:n_train]:
        idx2split[i] = "train"
    for i in indices[n_train:n_train + n_valid]:
        idx2split[i] = "valid"
    for i in indices[n_train + n_valid:]:
        idx2split[i] = "test"
    return idx2split

def get_dataset_path(idx: int, split_map: dict) -> str:
    return split_map[idx]

def data2img(idx, data, split_map):
    split_path = get_dataset_path(idx, split_map)
    save_path = DATASET_ROOT_SRC + split_path
    _, meta_csv = META_FILES[split_path]

    map_img = Image.fromarray(data, mode='L')
    map_img.save(f'{save_path}/maps/{idx:06d}.png')
    print(f"{idx}th map saved")

    id = 0
    for c in [1, 2, 4, 6]:
        for s in [1, 2, 4, 6]:
            gt_astar = AStar(data, c, s)
            gt_data, _, _ = gt_astar.planning()
            if gt_data is None:
                continue
            gt_img = Image.fromarray((gt_data * 255).astype(np.uint8), mode='L')
            gt_img.save(f'{save_path}/gts/{idx:06d}_{id}.png')
            meta_csv.writerow([
                idx, id, c, s, 
                f'{save_path}/maps/{idx:06d}.png', 
                f'{save_path}/gts/{idx:06d}_{id}.png'
            ])
            id += 1

def main():
    seeds = list(range(1, 10000))
    x_size = 201
    y_size = 201

    base_idx = 0
    split_map = make_split_map(10000 * 12)
    for seed in seeds:
        map_data_list = map_generator(seed, x_size=x_size, y_size=y_size)
        for idx, map_data in enumerate(map_data_list):
            data2img(base_idx + idx, map_data, split_map)
        base_idx += len(map_data_list)

    for f, _ in META_FILES.values():
        f.close()


if __name__ == "__main__":
    main()
