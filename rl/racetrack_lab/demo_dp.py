"""The viewer's own evaluator, so picking a map is enough to see something.

This is DEMO data: a plain synchronous policy evaluation under a uniform
policy, run in-process, recording every backup as it goes. It exists so the
screen has something real to draw before your engine is wired in -- values it
produces are labelled ``[DEMO]`` and are not a result from
``algorithms/bellman_dp``. To see your own numbers instead, publish a run with
``LiveRun`` or return ``(session, trace)`` from ``connect.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .rooms import cells, parse
from .schema import FINISH, START, TRACK, WALL, TrackMap
from .trace import Trace, TraceRecorder

# The reward rule this project uses (algorithms/bellman_dp/track_grid_world.py).
DEFAULT_REWARD: dict[int, float] = {WALL: -1.0, TRACK: -0.05, START: -0.1, FINISH: 1.0}

# The action set never changes; (dx, dy) with y upward.
ACTIONS: dict[str, tuple[int, int]] = {"→": (1, 0), "←": (-1, 0), "↑": (0, 1), "↓": (0, -1)}


@dataclass
class Result:
    track: TrackMap
    plane: dict[tuple[int, int], float]
    frames: list[tuple[int, dict[tuple[int, int], float]]] = field(default_factory=list)
    trace: Trace = field(default_factory=Trace)
    deltas: list[float] = field(default_factory=list)


def evaluate(
    map_text: str,
    *,
    gamma: float = 0.9,
    sweeps: int = 8,
    reward: Mapping[int, float] | None = None,
    name: str = "room",
) -> Result:
    """Evaluate the uniform policy on ``map_text`` and record every backup."""
    r = dict(DEFAULT_REWARD if reward is None else reward)
    track = parse(map_text, name=name)
    W, H = track.width, track.height
    goal = cells(track, FINISH)[0]
    drivable = [(x, y) for y in range(H) for x in range(W) if int(track.grid[y][x]) != WALL]
    states = [s for s in drivable if s != goal]

    def step(s: tuple[int, int], d: tuple[int, int]):
        nx, ny = s[0] + d[0], s[1] + d[1]
        if not (0 <= nx < W and 0 <= ny < H) or int(track.grid[ny][nx]) == WALL:
            return s, r[WALL], True                      # bounced: stay, take the penalty
        return (nx, ny), r[int(track.grid[ny][nx])], False

    rec = TraceRecorder(coords="xy", gamma=gamma, actions=ACTIONS)
    prob = 1.0 / len(ACTIONS)
    V: dict[tuple[int, int], float] = {s: 0.0 for s in drivable}
    frames: list[tuple[int, dict[tuple[int, int], float]]] = [(0, dict(V))]
    deltas: list[float] = []

    for k in range(1, max(0, sweeps) + 1):
        new, delta = dict(V), 0.0
        for s in states:
            terms, total = [], 0.0
            for a, d in ACTIONS.items():
                ns, rew, blocked = step(s, d)
                terms.append({"action": a, "prob": prob, "next_state": ns, "reward": rew,
                              "next_value": V[ns], "terminal": ns == goal, "blocked": blocked})
                total += prob * (rew + gamma * V[ns])
            rec.backup(state=s, value_before=V[s], terms=terms, value_after=total, sweep=k)
            new[s] = total
            delta = max(delta, abs(total - V[s]))
        V = new
        V[goal] = 0.0
        frames.append((k, dict(V)))
        deltas.append(delta)

    return Result(track=track, plane=dict(V), frames=frames, trace=rec.trace(), deltas=deltas)
