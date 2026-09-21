"""Make a demo run for the viewer from a text map.

    python tools/make_room_run.py              # uses MAP below -> runs/room
    python tools/make_room_run.py my_room      # -> runs/my_room

Edit MAP, REWARD, GAMMA and SWEEPS below and run it again. The map is plain
text, top row first::

    .  drivable     #  wall     S  start     G  goal (terminal)

WHAT THIS IS: demo data so the screen has something to show -- the values are
produced by the straightforward synchronous evaluation in this file, under a
uniform policy, and the run is labelled [DEMO]. It is not a result from your
engine. Once your loop calls ``run.backup(...)``, point the viewer at your own
run instead and this file stops mattering.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racetrack_lab import rooms
from racetrack_lab.live import LiveRun
from racetrack_lab.schema import FINISH, START, TRACK, WALL

# ---------------------------------------------------------------------------
# EDIT ME
# ---------------------------------------------------------------------------
MAP = """
. . . G
. . # .
S . . .
"""

# Your rule, from algorithms/bellman_dp/track_grid_world.py.
REWARD = {WALL: -1.0, TRACK: -0.05, START: -0.1, FINISH: 1.0}

GAMMA = 0.9
SWEEPS = 8

# Seconds to wait after each sweep. 0 finishes instantly; raise it and the app
# fills in while this script runs -- that is what "live" means here.
DELAY = 0.0

# The action set never changes; a blocked move keeps the state and takes
# REWARD[WALL]. (dx, dy) with y upward -- same convention as the viewer.
ACTIONS = {"→": (1, 0), "←": (-1, 0), "↑": (0, 1), "↓": (0, -1)}
# ---------------------------------------------------------------------------


def main(run_name: str = "room") -> None:
    track = rooms.parse(MAP, name=run_name)
    W, H = track.width, track.height
    goal = rooms.cells(track, FINISH)[0]
    drivable = [(x, y) for y in range(H) for x in range(W) if int(track.grid[y][x]) != WALL]
    states = [s for s in drivable if s != goal]

    def step(s, d):
        nx, ny = s[0] + d[0], s[1] + d[1]
        if not (0 <= nx < W and 0 <= ny < H) or int(track.grid[ny][nx]) == WALL:
            return s, REWARD[WALL], True
        return (nx, ny), REWARD[int(track.grid[ny][nx])], False

    print(rooms.show(track))
    print(f"{W} x {H} · goal {goal} · states {len(states)}")

    run = LiveRun(
        Path("runs") / run_name, track,
        scalar_label="V(s)  [DEMO]",
        note=f"demo data · uniform policy · gamma {GAMMA} · " +
             ", ".join(f"{k}:{v}" for k, v in REWARD.items()),
    )
    run.trace(coords="xy", gamma=GAMMA, actions=ACTIONS)

    V = {s: 0.0 for s in drivable}
    run.publish(scalar_field=dict(V), metrics={"sweep": 0}, label="V0")

    for k in range(1, SWEEPS + 1):
        new, delta = dict(V), 0.0
        for s in states:
            terms, total = [], 0.0
            prob = 1.0 / len(ACTIONS)
            for a, d in ACTIONS.items():
                ns, r, blocked = step(s, d)
                terms.append({"action": a, "prob": prob, "next_state": ns, "reward": r,
                              "next_value": V[ns], "terminal": ns == goal, "blocked": blocked})
                total += prob * (r + GAMMA * V[ns])
            run.backup(state=s, value_before=V[s], terms=terms, value_after=total, sweep=k)
            new[s] = total
            delta = max(delta, abs(total - V[s]))
        V = new
        V[goal] = 0.0
        run.publish(scalar_field=dict(V), metrics={"sweep": k, "delta": delta}, label=f"sweep {k}")
        print(f"  sweep {k}: max|dV| = {delta:.5f}")
        if DELAY:
            time.sleep(DELAY)

    run.close()

    for y in range(H - 1, -1, -1):
        print("  ", "  ".join(
            "   wall" if int(track.grid[y][x]) == WALL
            else ("      G" if (x, y) == goal else "%7.4f" % V[(x, y)])
            for x in range(W)))
    print(f"wrote runs/{run_name} · {len(states) * SWEEPS} backup events")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "room")
