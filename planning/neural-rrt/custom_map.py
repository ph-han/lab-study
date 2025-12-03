import numpy as np
from PIL import Image


x_size, y_size = 201, 201
human_visual=False
ori_map_data = np.zeros((x_size, y_size), dtype=np.uint8)
# map outline
for x in range(x_size):
    for y in range(y_size):
        if x == 0 or x == x_size - 1 or y == 0 or y == y_size - 1: 
            ori_map_data[y, x] = 1

# set random obstacles
for x in range(x_size):
    for y in range(y_size):
        if (x == 65 or x == 64 or x==66 or x == 63 or x == 68 or x==67) and y > 100:
            ori_map_data[y, x] = 1

for x in range(x_size):
    for y in range(y_size):
        if (x == 130 or x == 131 or x == 129 or x == 128 or x == 132 or x == 133) and y <= 100:
            ori_map_data[y, x] = 1

# set random start/goal point (12 diff)
start_y, start_x = 180, 30
goal_y, goal_x = 20, 180
ori_map_data[start_y, start_x] = 2
ori_map_data[goal_y, goal_x] = 3

map_img = Image.fromarray(ori_map_data, mode='L')
map_img.save(f'./dataset/test/maps/custom_map.png')
