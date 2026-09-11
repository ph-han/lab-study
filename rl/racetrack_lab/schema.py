"""Data containers the visualization understands.

This module is the *only* contract between whatever you implement and the
viewer. It computes nothing about decisions, values or dynamics -- it only
normalizes shapes so the renderers can stay dumb.

Conventions
-----------
* ``x`` is the column, ``y`` is the row, and ``y`` grows **upward** on screen.
* A grid is stored as ``grid[y, x]`` with ``y = 0`` at the bottom.
* A state is ``(x, y, vx, vy)``; ``(x, y)`` alone is also accepted everywhere.
* Field keys are plain tuples: ``(x, y)`` for a 2D field, ``(x, y, vx, vy)``
  for a velocity-dependent one. Nothing else is assumed.

Every ``from_any`` classmethod is deliberately permissive: pass tuples, lists,
dicts, numpy arrays or your own objects with matching attributes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# Cell codes -----------------------------------------------------------------
WALL = 0
TRACK = 1
START = 2
FINISH = 3

CELL_NAMES = {WALL: "Wall", TRACK: "Track", START: "Start", FINISH: "Finish"}
DRIVABLE_CODES = (TRACK, START, FINISH)


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrackMap:
    """A discrete 2D circuit."""

    grid: np.ndarray  # (height, width) int codes, grid[y, x], y=0 at bottom
    name: str = "Track"
    difficulty: str = ""
    note: str = ""

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    @property
    def drivable(self) -> np.ndarray:
        return np.isin(self.grid, DRIVABLE_CODES)

    @property
    def n_drivable(self) -> int:
        return int(self.drivable.sum())

    def code_at(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return int(self.grid[y, x])
        return WALL

    def is_drivable(self, x: int, y: int) -> bool:
        return self.code_at(x, y) in DRIVABLE_CODES

    def cells_of(self, code: int) -> list[tuple[int, int]]:
        ys, xs = np.where(self.grid == code)
        return [(int(x), int(y)) for x, y in zip(xs, ys)]


# ---------------------------------------------------------------------------
# Vehicle state
# ---------------------------------------------------------------------------
def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class VehicleState:
    """One point in time: where the car is and how fast it moves."""

    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    step: int | None = None
    action: tuple[float, float] | None = None
    reward: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def cell(self) -> tuple[int, int]:
        return int(round(self.x)), int(round(self.y))

    @property
    def heading_deg(self) -> float | None:
        """Clockwise degrees from screen-up, or None when standing still."""
        if self.speed == 0:
            return None
        return math.degrees(math.atan2(self.vx, self.vy))

    @classmethod
    def from_any(cls, obj: Any, step: int | None = None) -> "VehicleState":
        if isinstance(obj, VehicleState):
            if step is not None and obj.step is None:
                obj.step = step
            return obj

        if isinstance(obj, Mapping):
            data = dict(obj)
            pos = data.pop("position", None)
            if pos is None:
                pos = data.pop("pos", None)
            vel = data.pop("velocity", None)
            if vel is None:
                vel = data.pop("vel", None)
            x = data.pop("x", None)
            y = data.pop("y", None)
            vx = data.pop("vx", None)
            vy = data.pop("vy", None)
            if pos is not None:
                x, y = pos[0], pos[1]
            if vel is not None:
                vx, vy = vel[0], vel[1]
            action = data.pop("action", None)
            reward = data.pop("reward", None)
            return cls(
                x=_as_float(x),
                y=_as_float(y),
                vx=_as_float(vx),
                vy=_as_float(vy),
                step=data.pop("step", step),
                action=tuple(action) if action is not None else None,
                reward=None if reward is None else _as_float(reward),
                extras=data,
            )

        if isinstance(obj, (tuple, list, np.ndarray)):
            vals = [_as_float(v) for v in list(obj)[:4]]
            while len(vals) < 4:
                vals.append(0.0)
            return cls(x=vals[0], y=vals[1], vx=vals[2], vy=vals[3], step=step)

        if hasattr(obj, "x") and hasattr(obj, "y"):
            return cls(
                x=_as_float(getattr(obj, "x")),
                y=_as_float(getattr(obj, "y")),
                vx=_as_float(getattr(obj, "vx", 0.0)),
                vy=_as_float(getattr(obj, "vy", 0.0)),
                step=getattr(obj, "step", step),
            )
        raise TypeError(f"cannot read a vehicle state from {type(obj)!r}")


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------
@dataclass
class Trajectory:
    """An ordered sequence of states, e.g. one rollout."""

    states: list[VehicleState]
    label: str = "Trajectory"
    note: str = ""

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, i: int) -> VehicleState:
        return self.states[i]

    @property
    def xs(self) -> list[float]:
        return [s.x for s in self.states]

    @property
    def ys(self) -> list[float]:
        return [s.y for s in self.states]

    def at(self, index: int) -> VehicleState | None:
        if not self.states:
            return None
        return self.states[max(0, min(index, len(self.states) - 1))]

    @property
    def has_velocity(self) -> bool:
        """False for position-only states -- the UI then hides velocity readouts."""
        return any(s.vx or s.vy for s in self.states)

    @classmethod
    def from_any(
        cls,
        states: Iterable[Any],
        label: str = "Trajectory",
        metadata: Sequence[Mapping[str, Any]] | None = None,
        note: str = "",
    ) -> "Trajectory":
        parsed: list[VehicleState] = []
        for i, raw in enumerate(states):
            state = VehicleState.from_any(raw, step=i)
            if state.step is None:
                state.step = i
            if metadata is not None and i < len(metadata):
                meta = dict(metadata[i])
                if "action" in meta and state.action is None:
                    action = meta.pop("action")
                    state.action = tuple(action) if action is not None else None
                if "reward" in meta and state.reward is None:
                    state.reward = _as_float(meta.pop("reward"))
                state.extras.update(meta)
            parsed.append(state)
        return cls(states=parsed, label=label, note=note)


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------
def _split_key(key: Any) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """Split a field key into ``((x, y), (vx, vy) | None)``."""
    if isinstance(key, (tuple, list, np.ndarray)):
        parts = [int(round(_as_float(k))) for k in list(key)]
        if len(parts) >= 4:
            return (parts[0], parts[1]), (parts[2], parts[3])
        if len(parts) >= 2:
            return (parts[0], parts[1]), None
    raise TypeError(f"field key {key!r} is not (x, y) or (x, y, vx, vy)")


@dataclass
class _KeyedField:
    """Shared slicing behaviour for scalar and vector fields."""

    data: dict[tuple[int, int], Any]  # (x, y) -> payload, merged over slices
    label: str = ""
    note: str = ""
    velocity_keyed: bool = False
    _slices: dict[tuple[int, int], dict[tuple[int, int], Any]] = field(
        default_factory=dict, repr=False
    )

    @property
    def slices(self) -> list[tuple[int, int]]:
        """Available ``(vx, vy)`` slices, empty when the field is 2D."""
        return sorted(self._slices.keys())

    @property
    def vx_values(self) -> list[int]:
        return sorted({vx for vx, _ in self._slices})

    @property
    def vy_values(self) -> list[int]:
        return sorted({vy for _, vy in self._slices})

    def plane(self, slice_key: tuple[int, int] | None = None) -> dict[tuple[int, int], Any]:
        """The ``(x, y) -> payload`` plane for one velocity slice."""
        if not self.velocity_keyed:
            return self.data
        if slice_key is None:
            return {}
        return self._slices.get(tuple(slice_key), {})

    def __len__(self) -> int:
        if not self.velocity_keyed:
            return len(self.data)
        return sum(len(p) for p in self._slices.values())

    def all_values(self) -> list[Any]:
        """Every payload across every slice (``data`` alone merges slices)."""
        if not self.velocity_keyed:
            return list(self.data.values())
        return [v for plane in self._slices.values() for v in plane.values()]

    def value_bounds(self) -> tuple[float, float] | None:
        """Cached ``(min, max)`` over every slice; None when empty."""
        cached = getattr(self, "_bounds", None)
        if cached is None:
            values = [v for v in self.all_values() if isinstance(v, (int, float))]
            cached = (float(min(values)), float(max(values))) if values else ()
            object.__setattr__(self, "_bounds", cached)
        return cached or None


@dataclass
class ScalarField(_KeyedField):
    """A number per state -- values, returns, visit counts, errors, anything."""

    @classmethod
    def from_any(
        cls,
        data: Any,
        label: str = "Scalar field",
        note: str = "",
    ) -> "ScalarField | None":
        if data is None:
            return None

        planes: dict[tuple[int, int], dict[tuple[int, int], float]] = {}
        flat: dict[tuple[int, int], float] = {}
        velocity_keyed = False

        if isinstance(data, Mapping):
            for key, value in data.items():
                v = _as_float(value, float("nan"))
                if not math.isfinite(v):
                    continue
                xy, vel = _split_key(key)
                if vel is None:
                    flat[xy] = v
                else:
                    velocity_keyed = True
                    planes.setdefault(vel, {})[xy] = v
        else:
            arr = np.asarray(data, dtype=float)
            if arr.ndim != 2:
                raise ValueError(
                    "array scalar fields must be 2D and indexed [y, x]; use a dict "
                    "keyed by (x, y, vx, vy) for velocity-dependent data"
                )
            for (y, x), v in np.ndenumerate(arr):
                if math.isfinite(v):
                    flat[(int(x), int(y))] = float(v)

        merged = (
            {k: v for p in planes.values() for k, v in p.items()} if velocity_keyed else flat
        )
        return cls(
            data=merged,
            label=label,
            note=note,
            velocity_keyed=velocity_keyed,
            _slices=planes,
        )

    def value_range(
        self, slice_key: tuple[int, int] | None = None, global_scale: bool = True
    ) -> tuple[float, float]:
        source = self.data if global_scale else self.plane(slice_key)
        values = list(source.values())
        if not values:
            return (0.0, 1.0)
        lo, hi = float(min(values)), float(max(values))
        if lo == hi:
            return (lo - 0.5, hi + 0.5)
        return (lo, hi)


@dataclass
class VectorField(_KeyedField):
    """A 2D vector per state -- chosen accelerations, gradients, flows."""

    @classmethod
    def from_any(
        cls,
        data: Any,
        label: str = "Vector field",
        note: str = "",
    ) -> "VectorField | None":
        if data is None:
            return None
        if not isinstance(data, Mapping):
            raise TypeError(
                "vector fields must be a mapping: (x, y) or (x, y, vx, vy) -> (u, v)"
            )

        planes: dict[tuple[int, int], dict[tuple[int, int], tuple[float, float]]] = {}
        flat: dict[tuple[int, int], tuple[float, float]] = {}
        velocity_keyed = False

        for key, vec in data.items():
            if vec is None:
                continue
            parts = [_as_float(c, float("nan")) for c in list(vec)[:2]]
            if len(parts) < 2 or not all(math.isfinite(c) for c in parts):
                continue
            uv = (parts[0], parts[1])
            xy, vel = _split_key(key)
            if vel is None:
                flat[xy] = uv
            else:
                velocity_keyed = True
                planes.setdefault(vel, {})[xy] = uv

        merged = (
            {k: v for p in planes.values() for k, v in p.items()} if velocity_keyed else flat
        )
        return cls(
            data=merged,
            label=label,
            note=note,
            velocity_keyed=velocity_keyed,
            _slices=planes,
        )

    def max_magnitude(self) -> float:
        mags = [math.hypot(u, v) for u, v in self.data.values()]
        return max(mags) if mags else 1.0


# ---------------------------------------------------------------------------
# Iterations
# ---------------------------------------------------------------------------
@dataclass
class Iteration:
    """One snapshot of a run: whatever fields existed at that point."""

    index: int
    scalar: ScalarField | None = None
    vector: VectorField | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    label: str | None = None

    @property
    def title(self) -> str:
        return self.label or f"Iteration {self.index}"


# ---------------------------------------------------------------------------
# Session: the whole payload the viewer renders
# ---------------------------------------------------------------------------
@dataclass
class LabSession:
    """Everything the viewer knows. Every field except ``track`` is optional."""

    track: TrackMap
    trajectory: Trajectory | None = None
    scalar: ScalarField | None = None
    vector: VectorField | None = None
    iterations: list[Iteration] = field(default_factory=list)
    source: str = "none"  # "connected" | "live" | "demo" | "none"
    title: str = ""
    note: str = ""
    # Optional full-resolution metric history (``iterations`` may be subsampled):
    # [{"index": int, "label": str | None, "metrics": {name: value}}, ...]
    metrics_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_iterations(self) -> bool:
        return len(self.iterations) > 1

    def snapshot(self, index: int) -> Iteration | None:
        if not self.iterations:
            return None
        i = max(0, min(index, len(self.iterations) - 1))
        return self.iterations[i]

    def fields_at(
        self, iteration_index: int | None
    ) -> tuple[ScalarField | None, VectorField | None]:
        """Fields to draw: the snapshot's, falling back to session-level ones."""
        snap = self.snapshot(iteration_index) if iteration_index is not None else None
        scalar = (snap.scalar if snap else None) or self.scalar
        vector = (snap.vector if snap else None) or self.vector
        return scalar, vector


def build_session(
    track: TrackMap,
    trajectory: Any = None,
    scalar_field: Any = None,
    vector_field: Any = None,
    iterations: Sequence[Any] | None = None,
    source: str = "connected",
    scalar_label: str = "Scalar field",
    vector_label: str = "Vector field",
    trajectory_label: str = "Trajectory",
    title: str = "",
    note: str = "",
) -> LabSession:
    """Normalize loose data into a :class:`LabSession`.

    ``trajectory`` may be any iterable of states; ``scalar_field`` a mapping or
    2D array; ``vector_field`` a mapping to ``(u, v)``; ``iterations`` a sequence
    of :class:`Iteration` or of dicts with ``scalar`` / ``vector`` / ``metrics``.
    """
    traj: Trajectory | None
    if isinstance(trajectory, Trajectory):
        traj = trajectory
    elif trajectory is not None:
        traj = Trajectory.from_any(trajectory, label=trajectory_label)
    else:
        traj = None

    scalar = (
        scalar_field
        if isinstance(scalar_field, ScalarField)
        else ScalarField.from_any(scalar_field, label=scalar_label)
    )
    vector = (
        vector_field
        if isinstance(vector_field, VectorField)
        else VectorField.from_any(vector_field, label=vector_label)
    )

    snapshots: list[Iteration] = []
    for i, raw in enumerate(iterations or []):
        if isinstance(raw, Iteration):
            snapshots.append(raw)
            continue
        if isinstance(raw, Mapping):
            raw_scalar = raw.get("scalar")
            raw_vector = raw.get("vector")
            snapshots.append(
                Iteration(
                    index=int(raw.get("index", i)),
                    scalar=(
                        raw_scalar
                        if isinstance(raw_scalar, ScalarField)
                        else ScalarField.from_any(raw_scalar, label=scalar_label)
                    ),
                    vector=(
                        raw_vector
                        if isinstance(raw_vector, VectorField)
                        else VectorField.from_any(raw_vector, label=vector_label)
                    ),
                    metrics={
                        k: _as_float(v) for k, v in (raw.get("metrics") or {}).items()
                    },
                    label=raw.get("label"),
                )
            )
            continue
        raise TypeError(f"cannot read an iteration snapshot from {type(raw)!r}")

    return LabSession(
        track=track,
        trajectory=traj,
        scalar=scalar,
        vector=vector,
        iterations=snapshots,
        source=source,
        title=title,
        note=note,
    )
