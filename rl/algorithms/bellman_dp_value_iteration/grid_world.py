from config import UP, DOWN, LEFT, RIGHT, \
                   UP_ACTION, DOWN_ACTION, LEFT_ACTION, RIGHT_ACTION

from racetrack_lab.schema import WALL, TRACK, START, FINISH


class GridWorldEnv:
    def __init__(self, map_config):
        self.gridworld: list[list[int]] = map_config.grid
        self.height: int = map_config.height
        self.width: int = map_config.width

        self.reward_rule: dict = {
            WALL: -1.0,
            TRACK: -0.05,
            START: -0.05,
            FINISH: 1.0
        }

        self.actions = [UP, DOWN, LEFT, RIGHT]

        self.finish_states = map_config.cells_of(FINISH)

    def next_state(self, state: tuple[int, int], action: int) -> tuple[int, int]:
        action_move_in_grid = [UP_ACTION, DOWN_ACTION, LEFT_ACTION, RIGHT_ACTION]

        curr_y, curr_x = state
        dy, dx = action_move_in_grid[action]

        next_state_y, next_state_x = curr_y + dy, curr_x + dx

        if next_state_y >= self.height or next_state_y < 0 \
            or next_state_x >= self.width or next_state_x < 0: # check out of bound
            return state

        if self.gridworld[(next_state_y, next_state_x)] == WALL:
            return state

        return next_state_y, next_state_x
            

    def reward(self, state: tuple[int, int], action: int, next_state: tuple[int, int]) -> float:
        if state == next_state:
            return self.reward_rule[WALL]
        
        grid_type = self.gridworld[next_state]
        return self.reward_rule[grid_type]

    def states(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.gridworld[(y, x)] != WALL:
                    yield y, x