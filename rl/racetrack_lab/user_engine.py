"""Run the code in ``algorithms/bellman_dp_*`` and record it.

Nothing here evaluates anything. Your functions do the work exactly as you
wrote them; this module only watches the environment they call. Every
``next_state`` / ``reward`` pair is a term of the backup in progress, the value
read for the next state is taken at the moment your loop is about to read it,
and the new value is taken after your loop has written it. So the panel shows
your arithmetic, including any bug in it -- if your numbers are wrong, the
screen is wrong in the same way, which is the point.

Two engines, one recorder:

``policy`` -- ``algorithms/bellman_dp_policy_iteration``. One round is one
*evaluation sweep* followed by one *greedy improvement*, because the viewer
needs a frame per sweep to animate; your ``policy_eval`` and ``policy_iter``
loop internally and would hand back a single finished table. The arithmetic in
each round is still yours -- ``eval_one_step`` and ``greedy_policy`` are called
as you wrote them. Only the evaluation is recorded; the improvement runs
against the bare environment so it cannot pollute the trace with backups that
were never meant to change a value.

``value`` -- ``algorithms/bellman_dp_value_iteration``. One round is one call
of your ``value_iter_onestep``, for the same reason. There is no policy inside
the sweep, so each backup is recorded as a ``max`` over the four action values
and the panel marks the winning action instead of weighting by probability.
The route drawn from the start cell is the viewer's own argmax over your final
``V`` -- your value iteration never builds a policy, and a route has to come
from somewhere.

Your files are imported, never edited.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import START, WALL, TrackMap
from .trace import Trace, TraceRecorder

_ROOT = Path(__file__).resolve().parents[1] / "algorithms"

ALGO_DIRS: dict[str, Path] = {
    "policy": _ROOT / "bellman_dp_policy_iteration",
    "value": _ROOT / "bellman_dp_value_iteration",
}

# Both packages spell their modules the same way, so a cached module from one
# would otherwise be handed to the other. They are purged before every run.
_MODULES = ("config", "grid_world", "agent", "policy_eval", "policy_iter", "value_iter")

ALGO_DIR = ALGO_DIRS["policy"]
ALGO_NAME = f"algorithms/{ALGO_DIR.name}"


def algo_name(algo: str) -> str:
    return f"algorithms/{ALGO_DIRS[algo].name}"


class _Cells(list):
    """A cell list that also answers to being called.

    Your policy iteration reads ``env.finish_states`` and your value iteration
    calls ``env.finish_states()``; both reach the same cells here instead of
    one of the two raising ``TypeError``.
    """

    def __call__(self) -> "_Cells":
        return self


class _TrackYX:
    """The track, speaking your coordinate order.

    ``TrackMap.cells_of`` answers in ``(x, y)`` while your ``states()`` yields
    ``(y, x)``, so ``state in env.finish_states`` could never be true and the
    goal was never terminal. Rather than ask you to change that, the track
    handed to your environment answers in ``(y, x)`` -- everything else passes
    through.
    """

    def __init__(self, track):
        self._t = track

    def cells_of(self, code):
        return _Cells((y, x) for (x, y) in self._t.cells_of(code))

    def __getattr__(self, name):
        return getattr(self._t, name)


def _grid(env) -> Any:
    """Your environment's map, under whichever name it keeps it."""
    for name in ("gridworld", "grid"):
        g = getattr(env, name, None)
        if g is not None:
            return g
    raise RuntimeError("환경에 grid 가 없습니다 (gridworld / grid 중 하나가 필요합니다).")


def _goals(env) -> set[tuple[int, int]]:
    for name in ("finish_states", "goal_states"):
        g = getattr(env, name, None)
        if callable(g) and not isinstance(g, (list, tuple, set)):
            try:
                g = g()
            except TypeError:
                g = None
        if g:
            return {tuple(s) for s in g}
    return set()


def _derive_vectors(env) -> dict:
    """Ask the environment which way each action goes, instead of guessing.

    One free move per action is enough: the delta of ``next_state`` is the
    direction, so a blocked move can still be drawn pointing that way even if
    the action encoding changes.
    """
    vec: dict = {}
    for a in env.actions:
        for s in env.states():
            ns = env.next_state(s, a)
            if ns != s:
                vec[a] = (int(ns[1]) - int(s[1]), int(ns[0]) - int(s[0]))  # (dx, dy)
                break
    return vec


@dataclass
class Result:
    track: TrackMap
    plane: dict[tuple[int, int], float]
    frames: list[tuple[int, dict[tuple[int, int], float]]] = field(default_factory=list)
    trace: Trace = field(default_factory=Trace)
    notes: list[str] = field(default_factory=list)
    # the greedy rollout from the start cell, in viewer (x, y) -- the one thing
    # on this screen that *does* answer to where the start is
    path: list[tuple[int, int]] = field(default_factory=list)
    start_value: float | None = None


def _to_viewer(state: Any, coords: str) -> tuple[int, int]:
    a, b = int(state[0]), int(state[1])
    return (b, a) if coords == "yx" else (a, b)


def _greedy_path(env: Any, pi: Any, start: Any, coords: str, limit: int = 400) -> list[tuple[int, int]]:
    """Follow the greedy action from ``start`` until the goal, a wall or a loop.

    Nothing is evaluated here either: the policy is the one your
    ``greedy_policy`` returned (or, for value iteration, the argmax over your
    final ``V``), and each step is your ``next_state``.
    """
    if start is None:
        return []
    goals, cur, seen = _goals(env), tuple(start), {tuple(start)}
    out = [_to_viewer(cur, coords)]
    for _ in range(limit):
        if cur in goals:
            break
        try:
            probs = pi[cur]
        except KeyError:
            break
        if not probs:
            break
        nxt = tuple(env.next_state(cur, max(probs, key=probs.get)))
        if nxt == cur or nxt in seen:      # bounced off a wall, or going in circles
            break
        cur = nxt
        seen.add(cur)
        out.append(_to_viewer(cur, coords))
    return out


def _greedy_from_values(env: Any, V: Any, gamma: float) -> dict:
    """The viewer's own argmax over your values, used only to draw the route.

    Your value iteration stores values and never a policy, so the route has to
    be read back out of ``V``. It is deliberately not recorded: the panel only
    ever shows backups your own code performed.
    """
    pi: dict = {}
    for s in env.states():
        best_a, best_q = None, None
        for a in env.actions:
            ns = env.next_state(s, a)
            q = env.reward(s, a, ns) + gamma * V[ns]
            if best_q is None or q > best_q:
                best_a, best_q = a, q
        pi[tuple(s)] = {a: (1.0 if a == best_a else 0.0) for a in env.actions}
    return pi


def _prepare(algo_dir: Path) -> None:
    """Point imports at one package and forget the other one's modules."""
    for d in ALGO_DIRS.values():
        s = str(d)
        while s in sys.path:
            sys.path.remove(s)
    sys.path.insert(0, str(algo_dir))
    for name in _MODULES:
        sys.modules.pop(name, None)


def _import(name: str):
    return importlib.import_module(name)


class _Probe:
    """Stands in front of the environment and writes down every question asked.

    Your loop asks ``next_state(s, a)`` then ``reward(s, a, s')`` for each
    action of a state, then stores the new value. A change of ``s`` means the
    previous backup is finished, so it can be closed with the value your code
    just wrote.
    """

    def __init__(self, env: Any, V: Any, pi: Any, gamma: float, rec: TraceRecorder, sweep: int,
                 goals: set[tuple[int, int]] | None = None, policy_version: int = 0,
                 phase: str = "policy_evaluation", aggregation: str = "policy_weighted_sum"):
        self._env, self._V, self._pi, self._g, self._rec = env, V, pi, gamma, rec
        self._sweep = sweep
        self._goals = goals or set()
        self._version = policy_version
        self._phase, self._agg = phase, aggregation
        self._state = None
        self._before = 0.0
        self._terms: list[dict[str, Any]] = []

    # -- the two calls we listen to ------------------------------------------
    def next_state(self, state, action):
        ns = self._env.next_state(state, action)
        if state != self._state:
            self.flush()
            self._state = state
            self._before = float(self._V[state])
        return ns

    def reward(self, state, action, next_state):
        r = self._env.reward(state, action, next_state)
        probs: Any = {}
        if self._pi is not None:            # value iteration has no policy to read
            try:
                probs = self._pi[state]
            except KeyError:
                probs = {}
        self._terms.append(
            {
                "action": action,
                "prob": float(probs[action]) if action in probs else 0.0,
                "next_state": next_state,
                "reward": float(r),
                # read now: this is the number your loop is about to use
                "next_value": float(self._V[next_state]),
                "terminal": tuple(next_state) in self._goals,
                "blocked": tuple(next_state) == tuple(state),
            }
        )
        return r

    def flush(self):
        """Close the open backup once your code has written the new value."""
        if self._state is None or not self._terms:
            self._state, self._terms = None, []
            return
        self._rec.backup(
            state=self._state,
            value_before=self._before,
            terms=self._terms,
            value_after=float(self._V[self._state]),
            sweep=self._sweep,
            phase=self._phase,
            aggregation=self._agg,
            policy_version=self._version,
        )
        self._state, self._terms = None, []

    def __getattr__(self, name):        # states(), actions, finish_states, gridworld, ...
        return getattr(self._env, name)


def _plane(V: Any, env: Any, coords: str) -> dict[tuple[int, int], float]:
    """Values for the viewer. Walls are evaluated like any other state -- that
    is your loop's business -- they just have nothing to show."""
    grid = _grid(env)
    out: dict[tuple[int, int], float] = {}
    for s in env.states():
        a, b = int(s[0]), int(s[1])
        if int(grid[a][b]) == WALL:
            continue
        out[(b, a) if coords == "yx" else (a, b)] = float(V[s])
    return out


def _make_env(module: Any, track: TrackMap) -> Any:
    """Build your environment, and say plainly when it cannot be built."""
    try:
        env = module.GridWorldEnv(_TrackYX(track))
    except TypeError as exc:
        raise RuntimeError(
            f"GridWorldEnv(track) 를 만들지 못했습니다: {exc}  "
            "(생성자가 __init__ 이 아니라 __init 으로 적혀 있으면 호출되지 않습니다.)"
        ) from exc
    for attr in ("states", "actions", "next_state", "reward"):
        if not hasattr(env, attr):
            raise RuntimeError(f"GridWorldEnv 에 {attr} 가 없습니다.")
    _grid(env)
    return env


def _start_cell(track: TrackMap, coords: str):
    starts = [(y, x) for (x, y) in track.cells_of(START)] if coords == "yx" else track.cells_of(START)
    return starts[0] if starts else None


def _finish(env: Any, track: TrackMap, agent: Any, pi: Any, coords: str,
            frames: list, rec: TraceRecorder, notes: list[str]) -> Result:
    """Everything the two engines share once the sweeps are done."""
    goals = _goals(env)
    if goals and not any(tuple(s) in goals for s in env.states()):
        notes.append(
            f"목표 {sorted(goals)[0]} 가 states() 와 좌표 순서가 달라 종료 상태로 인식되지 않습니다."
        )
    start = _start_cell(track, coords)
    path = _greedy_path(env, pi, start, coords)
    start_value = float(agent.V[start]) if start is not None else None
    if start is not None and len(path) < 2:
        notes.append(f"출발 칸 {_to_viewer(start, coords)} 에서 그리디 경로가 한 걸음도 나아가지 않습니다.")

    plane = frames[-1][1] if frames else {}
    return Result(track=track, plane=plane, frames=[(0, dict.fromkeys(plane, 0.0))] + frames,
                  trace=rec.trace(), notes=notes, path=path, start_value=start_value)


# ---------------------------------------------------------------------------
# policy iteration: evaluate, then improve, once per round
# ---------------------------------------------------------------------------
def _run_policy(track: TrackMap, *, sweeps: int, coords: str) -> Result:
    _prepare(ALGO_DIRS["policy"])
    name = algo_name("policy")
    try:
        gw_mod = _import("grid_world")
        agent_mod = _import("agent")
        eval_mod = _import("policy_eval")
        iter_mod = _import("policy_iter")
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI, not swallowed
        raise RuntimeError(f"{name} 를 불러오지 못했습니다: {exc}") from exc

    env = _make_env(gw_mod, track)
    goals = _goals(env)
    vec = _derive_vectors(env)
    agent = agent_mod.Agent("app")
    gamma = float(getattr(agent, "gamma", 0.9))

    rec = TraceRecorder(coords=coords, gamma=gamma, actions=vec)
    frames: list[tuple[int, dict[tuple[int, int], float]]] = []
    notes: list[str] = []
    stable_at: int | None = None

    try:
        for k in range(1, max(1, sweeps) + 1):
            probe = _Probe(env, agent.V, agent.pi, gamma, rec, sweep=k,
                           goals=goals, policy_version=k - 1)
            out = eval_mod.eval_one_step(agent.pi, agent.V, probe, gamma)
            if out is not None:
                agent.V = out
            probe.flush()
            frames.append((k, _plane(agent.V, env, coords)))

            # improvement, against the bare env so it records nothing
            new_pi = iter_mod.greedy_policy(agent.V, env, gamma)
            if stable_at is None and new_pi == agent.pi:
                stable_at = k
            agent.pi = new_pi
    except Exception as exc:  # noqa: BLE001 - your bug, shown as your bug
        raise RuntimeError(f"{name} 실행 중 {type(exc).__name__}: {exc}") from exc

    if stable_at is not None:
        notes.append(f"{stable_at} 바퀴째부터 그리디 정책이 더 바뀌지 않습니다.")

    return _finish(env, track, agent, agent.pi, coords, frames, rec, notes)


# ---------------------------------------------------------------------------
# value iteration: one max-backup sweep per round, no policy in the loop
# ---------------------------------------------------------------------------
def _run_value(track: TrackMap, *, sweeps: int, coords: str) -> Result:
    _prepare(ALGO_DIRS["value"])
    name = algo_name("value")
    try:
        gw_mod = _import("grid_world")
        agent_mod = _import("agent")
        vi_mod = _import("value_iter")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{name} 를 불러오지 못했습니다: {exc}") from exc

    step = getattr(vi_mod, "value_iter_onestep", None)
    if step is None:
        raise RuntimeError(f"{name}/value_iter.py 에 value_iter_onestep 이 없습니다.")

    env = _make_env(gw_mod, track)
    goals = _goals(env)
    vec = _derive_vectors(env)
    agent = agent_mod.Agent("app")
    gamma = float(getattr(agent, "gamma", 0.9))

    rec = TraceRecorder(coords=coords, gamma=gamma, actions=vec)
    frames: list[tuple[int, dict[tuple[int, int], float]]] = []
    notes: list[str] = []
    deltas: list[float] = []
    settled_at: int | None = None

    try:
        for k in range(1, max(1, sweeps) + 1):
            old = {tuple(s): float(agent.V[s]) for s in env.states()}
            probe = _Probe(env, agent.V, None, gamma, rec, sweep=k, goals=goals,
                           policy_version=0, phase="value_iteration", aggregation="max")
            out = step(agent.V, probe, gamma)
            if out is not None:
                agent.V = out
            probe.flush()
            frames.append((k, _plane(agent.V, env, coords)))

            delta = max((abs(float(agent.V[s]) - v) for s, v in old.items()), default=0.0)
            deltas.append(delta)
            if settled_at is None and delta < 1e-4:
                settled_at = k
    except Exception as exc:  # noqa: BLE001 - your bug, shown as your bug
        raise RuntimeError(f"{name} 실행 중 {type(exc).__name__}: {exc}") from exc

    if settled_at is not None:
        notes.append(
            f"{settled_at} 바퀴째에 최대 변화량이 1e-4 아래로 내려갑니다 (Δ {deltas[settled_at - 1]:.2e})."
        )
    elif deltas:
        notes.append(f"{len(deltas)} 바퀴를 돌아도 최대 변화량이 아직 Δ {deltas[-1]:.2e} 입니다.")

    # the route is the viewer's argmax over your final V -- value iteration
    # stores no policy, and the animation needs one
    pi = _greedy_from_values(env, agent.V, gamma)
    return _finish(env, track, agent, pi, coords, frames, rec, notes)


def run(track: TrackMap, *, sweeps: int = 8, coords: str = "yx", algo: str = "policy") -> Result:
    """Import your modules and run ``sweeps`` rounds of the chosen engine."""
    if algo not in ALGO_DIRS:
        raise RuntimeError(f"알 수 없는 엔진 '{algo}' (policy / value)")
    return (_run_value if algo == "value" else _run_policy)(track, sweeps=sweeps, coords=coords)
