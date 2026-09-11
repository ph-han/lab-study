import numpy as np

from racetrack_lab.tracks import get_track

track = get_track("EASY")

track_grid = track.grid



def eval_policy_onestep(pi, V, env, gamma=0.9):
    for state in env.states():
        pass