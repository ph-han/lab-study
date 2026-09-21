"""The calculation channel: one Bellman backup, recorded so it can be replayed.

``LiveRun.publish`` carries *results* -- a value field, an arrow field, a few
metrics. That is enough to colour a map and no more: from a finished value
table you cannot recover which next state was read, what reward was collected,
or how four action results were combined. This module carries the *calculation*
instead, so the viewer can open one cell and show the arithmetic that produced
its number.

Your loop already has every piece at the moment it computes them::

    terms = []
    for action, prob in pi[state].items():
        next_state = env.next_state(state, action)
        reward = env.reward(state, action, next_state)
        terms.append({"action": action, "prob": prob, "next_state": next_state,
                      "reward": reward, "next_value": V[next_state]})
        ...
    run.backup(state=state, value_before=old, terms=terms, value_after=new, sweep=k)

Nothing here computes a value. ``backup`` records what you pass, recomputes the
aggregation from the same terms and stores whether the two agree, so a wrong
number shows up as a failed check rather than as a pretty picture.

Events land in ``backups.jsonl`` inside the run directory, one JSON object per
line, alongside the files :mod:`racetrack_lab.live` already writes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 2
BACKUPS = "backups.jsonl"
TRACE_META = "trace.json"

PHASES = ("policy_evaluation", "policy_improvement", "value_iteration", "rollout")
AGGREGATIONS = ("policy_weighted_sum", "max", "single")

_TOL = 1e-9


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def _xy(state: Any, coords: str) -> tuple[int, int]:
    """Normalise a state to viewer coordinates ``(x, y)``.

    ``coords`` says how *you* write states: ``"yx"`` for ``grid[y][x]`` indexing
    (the usual shape when you iterate rows), ``"xy"`` otherwise. It is recorded
    in the trace so the viewer never has to guess -- a mismatch here is the
    difference between a goal that is recognised and one that is not.
    """
    a, b = (int(state[0]), int(state[1]))
    return (b, a) if coords == "yx" else (a, b)


_ARROWS = {(1, 0): "→", (-1, 0): "←", (0, 1): "↑", (0, -1): "↓"}


def _glyph(action: Any, frm: tuple[int, int], to: tuple[int, int], vecs: Mapping[Any, Any] | None) -> str:
    """An arrow for the action that was *tried*.

    A blocked move lands back on ``frm``, so the travelled delta is (0, 0) and
    says nothing about which way the agent pushed. When the action vectors are
    known (``TraceWriter(actions=...)``) the arrow comes from the action itself
    and a bounce still shows its direction; otherwise it falls back to the
    delta, which is all the record contains.
    """
    if vecs:
        v = vecs.get(action)
        if v is None and not isinstance(action, str):
            v = vecs.get(str(action))
        if v is not None:
            return _ARROWS.get((int(v[0]), int(v[1])), "")
    return _ARROWS.get((to[0] - frm[0], to[1] - frm[1]), "")


class _Recorder:
    """Builds backup events. Subclasses decide where they go."""

    def __init__(
        self,
        *,
        coords: str = "yx",
        gamma: float | None = None,
        actions: Mapping[Any, Any] | Sequence[Any] | None = None,
    ) -> None:
        if coords not in ("yx", "xy"):
            raise ValueError("coords must be 'yx' or 'xy'")
        self.coords = coords
        self.gamma = gamma
        # action -> (dx, dy) in viewer coordinates, so a blocked move can still
        # be drawn pointing the way it was tried. e.g. utils.ACTION_VEC.
        self.vecs: dict[Any, tuple[int, int]] = {}
        if isinstance(actions, Mapping):
            for a, v in actions.items():
                self.vecs[a] = (int(v[0]), int(v[1]))
                self.vecs[str(a)] = (int(v[0]), int(v[1]))
        self._seq = 0
        self._actions_meta = (
            {str(a): list(v) for a, v in self.vecs.items()}
            if self.vecs
            else ([str(a) for a in actions] if actions is not None else None)
        )

    def _meta(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "coordinate_convention": self.coords,
            "gamma": self.gamma,
            "actions": self._actions_meta,
        }

    def _emit(self, event: dict[str, Any]) -> None:
        raise NotImplementedError

    # -- one recorded backup -------------------------------------------------
    def backup(
        self,
        *,
        state: Any,
        value_before: float,
        terms: Iterable[Mapping[str, Any]],
        value_after: float,
        gamma: float | None = None,
        sweep: int | None = None,
        phase: str = "policy_evaluation",
        aggregation: str = "policy_weighted_sum",
        policy_version: int = 0,
        source_snapshot_id: str | None = None,
    ) -> dict[str, Any]:
        g = self.gamma if gamma is None else gamma
        if g is None:
            raise ValueError("gamma is unknown: pass it to TraceWriter(...) or to backup(...)")
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}")

        s_xy = _xy(state, self.coords)
        out_terms: list[dict[str, Any]] = []
        for t in terms:
            n_xy = _xy(t["next_state"], self.coords)
            reward = float(t["reward"])
            next_value = float(t["next_value"])
            prob = float(t.get("prob", t.get("policy_probability", 0.0)))
            action_value = reward + g * next_value
            blocked = bool(t.get("blocked", n_xy == s_xy))
            out_terms.append(
                {
                    "action": str(t.get("action")),
                    "glyph": _glyph(t.get("action"), s_xy, n_xy, self.vecs),
                    "policy_probability": prob,
                    "outcomes": [
                        {
                            "probability": float(t.get("outcome_probability", 1.0)),
                            "next_state": list(n_xy),
                            "reward": reward,
                            "terminal": bool(t.get("terminal", False)),
                            "blocked": blocked,
                            "referenced_value": next_value,
                            "value_source": source_snapshot_id,
                        }
                    ],
                    "action_value": action_value,
                    "weighted_value": prob * action_value,
                    "self_loop": n_xy == s_xy,
                }
            )

        checks = self._check(out_terms, float(value_after), aggregation)
        event = {
            "schema_version": SCHEMA_VERSION,
            "event_seq": self._seq,
            "phase": phase,
            "policy_version": int(policy_version),
            "source_snapshot_id": source_snapshot_id,
            "target_sweep": sweep,
            "gamma": g,
            "state": list(s_xy),
            "value_before": float(value_before),
            "aggregation": aggregation,
            "action_terms": out_terms,
            "value_after": float(value_after),
            "checks": checks,
        }
        self._seq += 1
        self._emit(event)
        return event

    @staticmethod
    def _check(terms: list[dict[str, Any]], value_after: float, aggregation: str) -> dict[str, Any]:
        """Recompute the aggregation from the terms and compare. No opinions."""
        probs = sum(t["policy_probability"] for t in terms)
        if aggregation == "policy_weighted_sum":
            recomputed = sum(t["weighted_value"] for t in terms)
        elif aggregation == "max":
            recomputed = max((t["action_value"] for t in terms), default=0.0)
        else:
            recomputed = terms[0]["action_value"] if terms else 0.0
        return {
            "probability_sum": probs,
            "probability_sum_ok": abs(probs - 1.0) < 1e-6 if aggregation != "max" else None,
            "recomputed": recomputed,
            "matches_value_after": abs(recomputed - value_after) < _TOL,
            "error": recomputed - value_after,
            "n_terms": len(terms),
        }


class TraceWriter(_Recorder):
    """Appends backup events to ``<run_dir>/backups.jsonl``.

    Use this when the training loop and the viewer are separate processes --
    your script writes, the app tails.
    """

    def __init__(self, run_dir: str | os.PathLike, **kw: Any) -> None:
        super().__init__(**kw)
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._path = self.dir / BACKUPS
        self._path.write_text("", encoding="utf-8")
        (self.dir / TRACE_META).write_text(
            json.dumps(self._meta(), ensure_ascii=False), encoding="utf-8"
        )

    def _emit(self, event: dict[str, Any]) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")


class TraceRecorder(_Recorder):
    """Keeps backup events in memory -- no run directory, no files.

    For ``connect.py``, where your code runs inside the app::

        rec = TraceRecorder(coords="yx", gamma=0.9, actions=ACTION_VEC)
        ...                                  # your sweep, calling rec.backup(...)
        return build_session(track, scalar_field=V_xy), rec.trace()
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.events: list[dict[str, Any]] = []

    def _emit(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def trace(self) -> "Trace":
        return Trace(events=list(self.events), coords=self.coords, gamma=self.gamma)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
@dataclass
class Trace:
    """Backup events read back from a run directory."""

    events: list[dict[str, Any]] = field(default_factory=list)
    coords: str = "yx"
    gamma: float | None = None

    def __bool__(self) -> bool:
        return bool(self.events)

    @property
    def sweeps(self) -> list[int]:
        seen = sorted({e["target_sweep"] for e in self.events if e.get("target_sweep") is not None})
        return list(seen)

    def at(self, state: tuple[int, int] | None, *, sweep: int | None = None) -> dict[str, Any] | None:
        """The last recorded backup of ``state`` (optionally within one sweep)."""
        if state is None:
            return None
        want = [int(state[0]), int(state[1])]
        hit = None
        for e in self.events:
            if e["state"] != want:
                continue
            if sweep is not None and e.get("target_sweep") != sweep:
                continue
            hit = e
        return hit

    def failures(self) -> list[dict[str, Any]]:
        return [e for e in self.events if not e["checks"]["matches_value_after"]]


def read_trace(run_dir: str | os.PathLike) -> Trace:
    """Read ``backups.jsonl`` from a run directory. Missing file -> empty trace."""
    d = Path(run_dir)
    path = d / BACKUPS
    if not path.exists():
        return Trace()
    meta: dict[str, Any] = {}
    mpath = d / TRACE_META
    if mpath.exists():
        try:
            meta = json.loads(mpath.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            meta = {}
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except ValueError:
                continue  # a half-written last line while the writer is running
    return Trace(events=events, coords=meta.get("coordinate_convention", "yx"), gamma=meta.get("gamma"))
