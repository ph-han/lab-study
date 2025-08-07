import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Car import Car
import json

def distance_time(events):
    iter_event = iter(events)
    while True:
        try:
            e_key = next(iter_event)
        except StopIteration:
            break
    # traffic_red_sign = events['traffic_red_sign']
    # crossing_pedestrian = events['crossing_pedestrian']
    # vehicles = events['vehicles']

        print(f"{events[e_key]['name']} event type : {events[e_key]['type']}")
        type = events[e_key]['type']

        if type == "static":
            start_t = events[e_key]['start_t']
            end_t = events[e_key]['end_t']
            y = np.arange(start_t, end_t + 1)
            x = [events[e_key]['begin_distance']] * len(y)
            plt.plot(x, y, '-r')
        else:
            p1 = (events[e_key]['begin_distance'] - events[e_key]['following_distance'], events[e_key]['start_t'])
            p2 = (events[e_key]['begin_distance'] , events[e_key]['start_t'])
            p3 = (p1[0] + events[e_key]['end_t'], events[e_key]['end_t'])
            p4 = (p2[0] + events[e_key]['end_t'], events[e_key]['end_t'])

            plt.plot((p1[0], p2[0]), (p1[1], p2[1]), '-c')
            plt.plot((p1[0], p3[0]), (p1[1], p3[1]), '-c')
            plt.plot((p4[0], p2[0]), (p4[1], p2[1]), '-c')
            plt.plot((p3[0], p4[0]), (p3[1], p4[1]), '-c')

            p5 = (p2[0] + events[e_key]['obj_len'], events[e_key]['start_t'])
            p6 = (p4[0] + events[e_key]['obj_len'], events[e_key]['end_t'])

            plt.plot((p2[0], p5[0]), (p2[1], p5[1]), '-b')
            plt.plot((p6[0], p5[0]), (p6[1], p5[1]), '-b')
            plt.plot((p2[0], p4[0]), (p2[1], p4[1]), '-b')
            plt.plot((p4[0], p6[0]), (p4[1], p6[1]), '-b')

    plt.xticks(np.arange(0, 51, 10))
    plt.xlabel("distance [m]")
    plt.yticks(np.arange(0, 16, 5))
    plt.ylabel("time [s]")
    plt.grid(True)

def display_traffic_light(ax, event, road_y_bottom):
    w, h = 2, 4.5
    x, y = event['begin_distance'] - w / 2, road_y_bottom - h - 1
    traffic_light = plt.Rectangle((x, y), w, h, color="black", zorder=1)
    ax.add_patch(traffic_light)

    lights = []
    for i in range(3):
        light = plt.Circle((event['begin_distance'], y + 1 + i * 1.25), 0.5, color='gray', zorder=2)
        ax.add_patch(light)
        lights.append(light)

    return lights


state = -1
idx = 0
def simulation(path, car, events):
    # road_y_top = 2
    # road_y_bottom = -2

    fig, ax = plt.subplots()

    # # 도로 선
    # ax.plot([-10, 1000], [road_y_top, road_y_top], 'k')
    # ax.plot([-10, 1000], [road_y_bottom, road_y_bottom], 'k')

    car.draw(ax)


    def update(frame):
        global state, idx
        # ✨ 1. [수정] 이전 프레임 지우기
        # 이 한 줄이 잔상을 모두 제거합니다.
        ax.clear()

        # ✨ 2. [추가] 배경(도로) 다시 그리기
        # 화면을 지웠으니, 배경인 도로 선을 매번 다시 그려줍니다.
        road_y_top = 2
        road_y_bottom = -2
        ax.plot([-10, 1000], [road_y_top, road_y_top], 'k')
        ax.plot([-10, 1000], [road_y_bottom, road_y_bottom], 'k')

        # print(frame, len(path[0]))
        if frame < len(path[0]):
            # 현재 프레임(시간)에 맞는 위치 정보를 path에서 가져옵니다.
            # path가 {'x': 값, 'y': 값} 형태의 딕셔너리 리스트라고 가정합니다.
            position_data = list(zip(*path))
            new_x = position_data[frame][0]
            # new_y = position_data[frame][1]

            # car 객체의 위치를 업데이트합니다.
            car.x = new_x - Car.FRONT_OVERHANG - Car.WHEEL_BASE
            # car.y = new_y

            car.draw(ax)

        # 0: red, 1: yellow, 2: green
        traffic_light_lights = []
        event_key_list = list(events.keys())

        ek = event_key_list[idx]
        if frame >= events[ek]['end_t'] and len(event_key_list) - 1 > idx:
            idx += 1

        if events[ek]['type'] == "static":
            if events[ek]['name'] == "red_sign":
                lights = display_traffic_light(ax, events[ek], road_y_bottom)
                traffic_light_lights.append(lights)
                if state != 0 and events[ek]['end_t'] > frame >= events[ek]['start_t']:
                    state = 0
                elif state == 0 and frame == events[ek]['end_t']:
                    state = 2

            elif  events[ek]['name'] == "crossing_p":
                start_point = -2.5
                if events[ek]['end_t'] >= frame >= events[ek]['start_t']:
                    diff = (events[ek]["end_t"] - events[ek]["start_t"])
                    print(start_point + (frame - events[ek]["start_t"]) * (5 / diff))
                    ax.plot(events[ek]["begin_distance"], start_point + (frame - events[ek]["start_t"]) * (5 / diff), 'ob')

        print(frame, state, ek)
        for lights in traffic_light_lights:
            # 초기화
            for light in lights:
                light.set_color('gray')

            if state == 0:
                lights[2].set_color('red')
            elif state == 1:
                lights[1].set_color('yellow')
            elif state == 2:
                lights[0].set_color('green')


        ax.set_aspect('equal')

        ax.set_xlim(-10, 30 + frame)
        ax.set_ylim(-10, 10)

    ani = FuncAnimation(fig, update, frames=len(path[0]), interval=1000)
    plt.show()

def planning_res(rs, rt):
    plt.plot(rs, rt, '-ob')

def expansion_pos(s, t, color='g'):
    plt.plot(s, t, f'x{color}')

def pause(sec):
    plt.pause(sec)

def show():
    plt.show()

if __name__ == '__main__':
    with open("urban_env.json", 'r') as json_file:
        event_json_data = json.load(json_file)
    car = Car(0, 0, 0)
    simulation(None, car, event_json_data)
    show()