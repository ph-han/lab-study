from collections import defaultdict

from racetrack_lab.tracks import TrackMap

class Bellman:
    def __init__(self, track: TrackMap):
        self.grid = track.grid
        self.grid_h = track.height
        self.grid_w = track.width

        # 상태가치 추정치
        self.V = defaultdict(lambda: 0)

        # 행동 정책 확률
        pi = defaultdict(lambda: {
            0: 0.25, # FORWARD
            1: 0.25, # BACKWARD
            2: 0.25, # UP
            3: 0.25  # DOWN
        })

        # 보상 정책
        self.reward_rule = {
            0: -1,      # WALL
            1: 0,       # TRACK
            2: -0.1,    # START
            3: 3,       # FINISH
            4: -0.05    # MOVE
        }

        self.action_move_in_map = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def next_state(self, curr_state, action):
        move_cmd = self.action_move_in_map[action]

        new_y, new_x = curr_state[0] + move_cmd[0], curr_state[1] + move_cmd[1]

        if new_y >= self.grid_h or new_y < 0 or new_x >= self.grid_w or new_x < 0:
            return curr_state

        if self.grid[new_y][new_x] == 0: # WALL
            return curr_state

        return new_y, new_x
        

    def states(self):
        for h in range(self.grid_h):
            for w in range(self.grid_w):
                yield (h, w)

