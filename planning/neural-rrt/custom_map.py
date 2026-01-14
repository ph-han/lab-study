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

# for x in range(x_size):
#     for y in range(y_size):
#         if x in list(range(50, 80)) and y < 70:
#             ori_map_data[y, x] = 1

# for x in range(x_size):
#     for y in range(y_size):
#         if x in list(range(50, 80)) and y > 80:
#             ori_map_data[y, x] = 1

# for x in range(x_size):
#     for y in range(y_size):
#         if x in list(range(120, 150)) and y <= 150:
#             ori_map_data[y, x] = 1

# for x in range(x_size):
#     for y in range(y_size):
#         if x in list(range(120, 150)) and y > 160:
#             ori_map_data[y, x] = 1

# for x in range(x_size):
#     for y in range(y_size):
#         if x in list(range(10, 30)) and y in list(range(120, 140)):
#             ori_map_data[y, x] = 1

# for x in range(x_size):
#     for y in range(y_size):
#         if x in list(range(0, 30)) and y in list(range(40, 60)):
#             ori_map_data[y, x] = 1

# for x in range(x_size):
#     for y in range(y_size):
#         if x in list(range(45, 60)) and y in list(range(90, 190)):
#             ori_map_data[y, x] = 1

# for x in range(x_size):
#     for y in range(y_size):
#         if x in list(range(80, 90)) and y in list(range(90, 180)):
#             ori_map_data[y, x] = 1

# for x in range(x_size):
#     for y in range(y_size):
#         if x in list(range(100, 110)) and y in list(range(130, 140)):
#             ori_map_data[y, x] = 1
# for x in range(x_size):
#     for y in range(y_size):
#         if x in list(range(105, 120)) and y in list(range(50, 80)):
#             ori_map_data[y, x] = 1


def add_rect(m, x0, x1, y0, y1, val=1):
    """
    [x0, x1) , [y0, y1) 범위를 직사각형으로 채움 (end는 미포함)
    """
    m[y0:y1, x0:x1] = val


# --- 왼쪽 큰 세로 장애물 (위/아래가 끊겨있는 형태) ---
add_rect(ori_map_data, 45, 75, 0, 45)       # 위쪽 블록
add_rect(ori_map_data, 39, 80, 66, 201)     # 아래쪽 블록

# --- 오른쪽 큰 세로 장애물 (중간에 왼쪽으로 튀어나온 '턱' + 아래쪽 분리) ---
add_rect(ori_map_data, 126, 156, 0, 164)    # 메인 세로 블록
add_rect(ori_map_data, 119, 126, 37, 56)    # 중간 왼쪽으로 튀어나온 부분(턱)
add_rect(ori_map_data, 126, 156, 186, 201)  # 아래쪽 분리 블록

# --- 작은 정사각형 장애물들 3개 ---
add_rect(ori_map_data, 11, 28, 98, 119)     # 왼쪽 작은 블록
add_rect(ori_map_data, 90, 113, 98, 119)    # 가운데 작은 블록
add_rect(ori_map_data, 172, 192, 123, 143)  # 오른쪽 작은 블록
add_rect(ori_map_data, 35, 45, 125, 150)  # 오른쪽 작은 블록
add_rect(ori_map_data, 80, 90, 70, 90)  # 오른쪽 작은 블록

# set random start/goal point (12 diff)
start_y, start_x = 180, 10
goal_y, goal_x = 20, 180
ori_map_data[start_y, start_x] = 2
ori_map_data[goal_y, goal_x] = 3

map_img = Image.fromarray(ori_map_data, mode='L')
map_img.save(f'./dataset/test/maps/custom_map.png')
