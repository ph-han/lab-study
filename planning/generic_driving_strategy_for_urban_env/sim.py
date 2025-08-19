import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Car import Car
from IDM import IDMVehicle
import json

def distance_time(events):
    if not events:
        return
    iter_event = iter(events)
    while True:
        try:
            e_key = next(iter_event)
        except StopIteration:
            break

        event_data = events.get(e_key)
        if not event_data:
            continue

        print(f"{event_data.get('name', 'N/A')} event type : {event_data.get('type', 'N/A')}")
        type = event_data.get('type')

        if type == "static":
            start_t = event_data.get('start_t', 0)
            end_t = event_data.get('end_t', 0)
            y = np.arange(start_t, end_t + 1)
            x = [event_data.get('begin_distance', 0)] * len(y)
            plt.plot(x, y, '-r')
        elif type == "dynamic":
            p1 = (event_data['begin_distance'] - event_data['following_distance'], event_data['start_t'])
            p2 = (event_data['begin_distance'], event_data['start_t'])
            p3 = (p1[0] + event_data['end_t'], event_data['end_t'])
            p4 = (p2[0] + event_data['end_t'], event_data['end_t'])

            plt.plot((p1[0], p2[0]), (p1[1], p2[1]), '-c')
            plt.plot((p1[0], p3[0]), (p1[1], p3[1]), '-c')
            plt.plot((p4[0], p2[0]), (p4[1], p2[1]), '-c')
            plt.plot((p3[0], p4[0]), (p3[1], p4[1]), '-c')

            p5 = (p2[0] + event_data['obj_len'], event_data['start_t'])
            p6 = (p4[0] + event_data['obj_len'], event_data['end_t'])

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


class UrbanSimulator:
    def __init__(self, fig, ax, path, ego, npcs, events):
        self.fig = fig
        self.ax = ax
        self.path = path
        self.ego = ego
        self.npcs = npcs
        self.events = events
        
        self.road_y_top = 2
        self.road_y_bottom = -2
        
        self.traffic_light_state = -1

        self.npc_idms = []
        dynamic_events = [e for e in events.values() if e and e.get('type') == 'dynamic']

        for npc_car in self.npcs:
            matching_event = None
            for event in dynamic_events:
                if abs(event['begin_distance'] - npc_car.x) < 0.1:
                    matching_event = event
                    break

            if matching_event:
                start_t = matching_event.get('start_t', 0)
                end_t = matching_event.get('end_t', 0)

                npc_start_velocity = path[1][start_t] if start_t < len(path[1]) else 0
                npc_end_velocity = path[1][end_t] if end_t < len(path[1]) else path[1][-1]

                self.npc_idms.append((IDMVehicle(npc_car.x, npc_start_velocity, v0=npc_end_velocity), start_t, end_t))
            else:
                self.npc_idms.append(None)

        self.ego_idm = IDMVehicle(ego.x, path[1][0])
        self.path = self.upsample_data(20) # Changed for smoother animation

    def upsample_data(self, ms):
        upsample_s = []
        upsample_v = []
        upsample_t = []

        if not self.path or not self.path[0]:
            return [[], [], []]

        points_per_step = 1000 // ms
        for idx in range(len(self.path[0]) - 1):
            diff_s = self.path[0][idx + 1] - self.path[0][idx]
            diff_v = self.path[1][idx + 1] - self.path[1][idx]
            diff_t = self.path[2][idx + 1] - self.path[2][idx]
            for new_idx in range(points_per_step):
                upsample_s.append(self.path[0][idx] + (diff_s / points_per_step) * new_idx)
                upsample_v.append(self.path[1][idx] + (diff_v / points_per_step) * new_idx)
                upsample_t.append(self.path[2][idx] + (diff_t / points_per_step) * new_idx)
        
        upsample_s.append(self.path[0][-1])
        upsample_v.append(self.path[1][-1])
        upsample_t.append(self.path[2][-1])

        return [upsample_s, upsample_v, upsample_t]

    def update(self, frame):
        if frame >= len(self.path[0]):
            return

        # Time in seconds, based on 50 FPS (1000ms / 20ms interval)
        sec = frame / 50.0

        self._draw_background()
        self._update_ego_position(frame)
        self._update_npcs(sec)
        self._handle_static_events(sec)
        self._update_plot_view(frame, sec)

    def _draw_background(self):
        self.ax.clear()
        self.ax.plot([-10, 1000], [self.road_y_top, self.road_y_top], 'k')
        self.ax.plot([-10, 1000], [self.road_y_bottom, self.road_y_bottom], 'k')

    def _update_ego_position(self, frame):
        new_x = self.path[0][frame]
        self.ego.x = new_x - Car.FRONT_OVERHANG - Car.WHEEL_BASE
        self.ego.draw(self.ax)

    def _update_npcs(self, sec):

        all_vehicles = list(zip(self.npcs, self.npc_idms))
        
        for i, (npc_car, npc_idm) in enumerate(zip(self.npcs, self.npc_idms)):
            if not npc_idm:
                continue

            if npc_idm[1] < sec:
                leader = None
                min_dist = float('inf')
                for other_car, other_idm in all_vehicles:
                    if npc_car is not other_car and sec < other_idm[1]:
                        dist = other_car.x - npc_car.x
                        if 0 < dist < min_dist:
                            min_dist = dist
                            leader = other_idm[0]


                if sec < npc_idm[2] and self.ego.x < npc_car.x:
                    npc_idm[0].update_acceleration(leader=leader)
                    npc_idm[0].update_state(dt=0.02)
                    npc_car.x = npc_idm[0].x - Car.FRONT_OVERHANG - Car.WHEEL_BASE
                    npc_car.draw(self.ax)

    def _handle_static_events(self, sec):
        for event in self.events.values():
            if event and event.get('type') == 'static':
                traffic_light_lights = []
                self._handle_static_event(sec, event, traffic_light_lights)
                self._update_traffic_light_colors(traffic_light_lights)

    def _handle_static_event(self, sec, event, traffic_light_lights):
        if event['name'] == "red_sign":
            lights = display_traffic_light(self.ax, event, self.road_y_bottom)
            traffic_light_lights.append(lights)
            if self.traffic_light_state != 0 and event['start_t'] <= sec < event['end_t']:
                self.traffic_light_state = 0
            elif self.traffic_light_state == 0 and sec >= event['end_t']:
                self.traffic_light_state = 2
        
        elif event['name'] == "crossing_p":
            if event['start_t'] <= sec < event['end_t']:
                start_point = -2.5
                duration = event["end_t"] - event["start_t"]
                time_elapsed = sec - event["start_t"]
                pedestrian_y = start_point + time_elapsed * (5 / duration) if duration > 0 else start_point
                self.ax.plot(event["begin_distance"], pedestrian_y, 'ob')

    def _update_traffic_light_colors(self, traffic_light_lights):
        for lights in traffic_light_lights:
            for light in lights:
                light.set_color('gray')
            if self.traffic_light_state == 0:
                lights[2].set_color('red')
            elif self.traffic_light_state == 1:
                lights[1].set_color('yellow')
            elif self.traffic_light_state == 2:
                lights[0].set_color('green')

    def _update_plot_view(self, frame, sec):
        self.ax.set_aspect('equal')
        self.ax.set_title(f'{sec:.1f}s | ego : {self.path[0][frame]:.2f} m, {self.path[1][frame]:.2f} m/s')
        self.ax.set_xlim(-7 + self.path[0][frame], 50 + self.path[0][frame])
        self.ax.set_ylim(-10, 10)

def simulation(path, ego, npcs, events):
    fig, ax = plt.subplots()
    
    if not path or not path[0]:
        print("Warning: Empty path provided to simulation.")
        return

    simulator = UrbanSimulator(fig, ax, path, ego, npcs, events)

    ani = FuncAnimation(fig, simulator.update, frames=len(simulator.path[0]), interval=20, repeat=False)
    plt.show()

def planning_res(rs, rt):
    plt.plot(rs, rt, '-ob')
    plt.savefig('distance_time.jpg')

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
    # This is just for standalone testing of draw.py, main logic is in main.py
    # simulation(None, car, [], event_json_data) 
    show()
