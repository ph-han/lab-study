import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Car import Car
from IDM import IDMVehicle
from driving_planning import planning
import json

def distance_time(axes, events):
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

        # print(f"{event_data.get('name', 'N/A')} event type : {event_data.get('type', 'N/A')}")
        type = event_data.get('type')

        if type == "static":
            start_t = event_data.get('start_t', 0)
            end_t = event_data.get('end_t', 0)
            y = np.arange(start_t, end_t + 1)
            x = [event_data.get('begin_distance', 0)] * len(y)
            axes.plot(x, y, '-r')
        elif type == "dynamic":
            p1 = (event_data['begin_distance'] - event_data['following_distance'], event_data['start_t'])
            p2 = (event_data['begin_distance'], event_data['start_t'])
            p3 = (p1[0] + event_data['end_t'], event_data['end_t'])
            p4 = (p2[0] + event_data['end_t'], event_data['end_t'])

            axes.plot((p1[0], p2[0]), (p1[1], p2[1]), '-c')
            axes.plot((p1[0], p3[0]), (p1[1], p3[1]), '-c')
            axes.plot((p4[0], p2[0]), (p4[1], p2[1]), '-c')
            axes.plot((p3[0], p4[0]), (p3[1], p4[1]), '-c')

            p5 = (p2[0] + event_data['obj_len'], event_data['start_t'])
            p6 = (p4[0] + event_data['obj_len'], event_data['end_t'])

            axes.plot((p2[0], p5[0]), (p2[1], p5[1]), '-b')
            axes.plot((p6[0], p5[0]), (p6[1], p5[1]), '-b')
            axes.plot((p2[0], p4[0]), (p2[1], p4[1]), '-b')
            axes.plot((p4[0], p6[0]), (p4[1], p6[1]), '-b')

    # axes.xticks(np.arange(0, 101, 10))
    axes.set_xlabel("distance [m]")
    # axes.yticks(np.arange(0, 31, 5))
    axes.set_ylabel("time [s]")
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlim(0, 60)
    axes.set_ylim(0, 25)
    axes.grid(True)

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
    def __init__(self, fig, axes, path, ego, npcs, events):
        self.fig = fig
        self.ax0 = axes[0]
        self.ax1 = axes[1]
        self.path = path
        self.ego = ego
        self.npcs = npcs
        self.events = events
        self.c_events = events
        
        self.road_y_top = 2
        self.road_y_bottom = -2
        
        self.traffic_light_state = -1

        self.npc_idms = []
        dynamic_events = [e for e in events.values() if e and e.get('type') == 'dynamic']

        for npc_car, npc_event_key in self.npcs:
            matching_event = None
            for event in dynamic_events:
                if abs(event['begin_distance'] - npc_car.x) < 0.1:
                    matching_event = event
                    break

            if matching_event:
                start_t = matching_event.get('start_t', 0)
                end_t = matching_event.get('end_t', 0)

                npc_start_velocity = end_t / (end_t - start_t)
                npc_end_velocity = npc_start_velocity

                self.npc_idms.append((IDMVehicle(npc_car.x, npc_start_velocity, v0=npc_end_velocity), start_t, end_t))
            else:
                self.npc_idms.append(None)

        self.ego_idm = IDMVehicle(ego.x, path[1][0])
        self.path = self.upsample_data(self.path, 20) # Changed for smoother animation

    def upsample_data(self, path, ms):
        upsample_s = []
        upsample_v = []
        upsample_t = []

        if not path or not path[0]:
            return [[], [], []]

        points_per_step = 1000 // ms
        for idx in range(len(path[0]) - 1):
            diff_s = path[0][idx + 1] - path[0][idx]
            diff_v = path[1][idx + 1] - path[1][idx]
            diff_t = path[2][idx + 1] - path[2][idx]
            for new_idx in range(points_per_step):
                upsample_s.append(path[0][idx] + (diff_s / points_per_step) * new_idx)
                upsample_v.append(path[1][idx] + (diff_v / points_per_step) * new_idx)
                upsample_t.append(path[2][idx] + (diff_t / points_per_step) * new_idx)
        
        upsample_s.append(path[0][-1])
        upsample_v.append(path[1][-1])
        upsample_t.append(path[2][-1])

        return [upsample_s, upsample_v, upsample_t]

    def _update_path(self, frame):
        # curr_state = (self.path[0][frame], self.path[1][frame], frame // 50)
        curr_state = (0, self.path[1][frame], 0)
        # print("curr frame: ", frame, len(self.path[1]))
        new_events = self.events.copy()
        for key, event in self.events.items():
            if self.events[key] and self.events[key]['start_t'] == self.events[key]['end_t']:
                del new_events[key]
        if len(new_events) == 0:
            new_events = {'none': None}
        self.events = new_events
        new_rs, new_rv, new_rt = planning(curr_state, self.events, 13)
        
        self.ax0.clear()
        distance_time(self.ax0, self.events)
        draw_rs = np.array(new_rs)
        draw_rt = np.array(new_rt)
        self.ax0.plot(draw_rs, draw_rt, '-ob')
        
        new_rs, new_rv, new_rt = self.upsample_data([new_rs, new_rv, new_rt], 20)
        self.path[0] = self.path[0][:frame + 1] + new_rs
        self.path[1] = self.path[1][:frame + 1] + new_rv
        self.path[2] = self.path[2][:frame + 1] + new_rt


    def update(self, frame):
        if frame >= len(self.path[0]):
            return

        sec = frame / 50.0

        if frame % 5 == 0:
            self._update_path(frame)
        self._draw_background()
        self._update_ego_position(frame)
        self._update_npcs(sec)
        self._handle_static_events(sec)
        self._update_events(frame)
        self._update_plot_view(frame, sec)

    def _draw_background(self):
        self.ax1.clear()
        self.ax1.plot([-10, 1000], [self.road_y_top, self.road_y_top], 'k')
        self.ax1.plot([-10, 1000], [self.road_y_bottom, self.road_y_bottom], 'k')

    def _update_ego_position(self, frame):
        if self.ego.x == 0:
            new_x = self.path[0][frame]
        else:
            new_x = max(self.ego.x + Car.FRONT_OVERHANG + Car.WHEEL_BASE, 0) + self.path[0][frame]
        self.ego.x = new_x - Car.FRONT_OVERHANG - Car.WHEEL_BASE
        self.ax1.text(self.ego.x + Car.WHEEL_BASE // 2, self.ego.y + Car.OVERALL_WIDTH + 2, f'ego0', ha='center',
                     va='top')
        self.ego.draw(self.ax1)

    def _update_events(self, frame):
        _updated_events = self.events.copy()

        for key, obs in self.events.items():
            if key == 'none' or not obs:
                continue
            # print(obs)
            if obs['end_t'] < 0:
                del _updated_events[key]
                continue
            if self.c_events[key]['begin_distance'] < self.ego.x + Car.FRONT_OVERHANG + Car.WHEEL_BASE:
                del _updated_events[key]
                print("del")
                continue

            _updated_event = obs.copy()
            _updated_event['start_t'] = max(self.c_events[key]['start_t'] - frame / 50, 0) 
            _updated_event['end_t'] = max(self.c_events[key]['end_t'] - frame / 50, 0) 
            _updated_event['begin_distance'] = max(self.c_events[key]['begin_distance'] - (self.ego.x + Car.FRONT_OVERHANG + Car.WHEEL_BASE), 0)
            _updated_events[key] = _updated_event

        self.events = _updated_events

    def _update_npcs(self, sec):
        for i, (npc_info, npc_idm) in enumerate(zip(self.npcs, self.npc_idms)):
            if not npc_idm or not (npc_info[1] in self.events):
                continue
            leader = None
            if i < len(self.npc_idms) - 1 and sec < self.npc_idms[i + 1][1]:
                leader = self.npc_idms[i + 1][0]
            if npc_idm[1] <= sec <= npc_idm[2] :
                npc_idm[0].update_acceleration(leader=leader)
                npc_idm[0].update_state(dt=0.02)
                # diff = self.events[npc_info[1]]['end_t'] - self.events[npc_info[1]]['start_t']
                # vs = npc_idm[0].v * diff
                # if vs != self.events[npc_info[1]]['vs']:
                #     self.events[npc_info[1]]['start_t'] = int(sec)
                #     self.events[npc_info[1]]['vs'] = vs
            npc_info[0].x = npc_idm[0].x - Car.FRONT_OVERHANG - Car.WHEEL_BASE
            self.ax1.text(npc_info[0].x + Car.WHEEL_BASE // 2, npc_info[0].y + npc_info[0].OVERALL_WIDTH + 2, f'npc_{i} | {npc_idm[0].v:.2f}m/s', ha='center', va='top')
            npc_info[0].draw(self.ax1)

    def _handle_static_events(self, sec):
        for event in self.c_events.values():
            if event and event.get('type') == 'static':
                traffic_light_lights = []
                self._handle_static_event(sec, event, traffic_light_lights)
                self._update_traffic_light_colors(traffic_light_lights)

    def _handle_static_event(self, sec, event, traffic_light_lights):
        if event['name'] == "red_sign":
            lights = display_traffic_light(self.ax1, event, self.road_y_bottom)
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
                self.ax1.plot(event["begin_distance"], pedestrian_y, 'ob')

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
        self.ax1.set_aspect('equal')
        self.ax1.set_title(f'{sec:.1f}s | ego : {self.ego.x + Car.FRONT_OVERHANG + Car.WHEEL_BASE:.2f} m, {self.path[1][frame]:.2f} m/s')
        self.ax1.set_xlim(-7 + self.ego.x, 60 + self.ego.x)
        self.ax1.set_ylim(-10, 10)

def simulation(ego, npcs, events):
    # path = planning([0, 0, 0], events, 30)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    #
    # distance_time(axes[0], events)
    # axes[0].plot(path[0], path[2], '-ob')

    simulator = UrbanSimulator(fig, axes, [[0], [0], [0]], ego, npcs, events)

    ani = FuncAnimation(fig, simulator.update, frames=3000, interval=20, repeat=False)
    plt.tight_layout()
    plt.show()

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
    # This is just for standalone testing of sim.py, main logic is in main.py
    # simulation(None, car, [], event_json_data) 
    show()
