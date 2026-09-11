"""DEMO DATA -- placeholder shapes used to exercise the viewer's UI.

Nothing in this file is an algorithm, a policy, a value function or a simulated
rollout. Everything is closed-form geometry:

* the trajectory is a **hand-authored polyline** through the corridor; velocity
  is just the difference between consecutive hand-placed positions,
* the scalar field is ``-distance to the finish`` with a small directional tilt,
  so different velocity slices look different,
* the vector field points at the finish, snapped to the 9 grid directions,
* the iteration history linearly blends a flat field into that geometry.

It exists so the layout, the slice selector, the playback and the empty states
can be checked without any experiment attached, and it is always labelled
``DEMO DATA`` on screen. Replace it by returning your own data from
``racetrack_lab/connect.py``.
"""

from __future__ import annotations

import math

from .schema import (
    FINISH,
    Iteration,
    LabSession,
    ScalarField,
    TrackMap,
    Trajectory,
    VectorField,
    VehicleState,
)

DEMO_BADGE = "DEMO DATA"

# Hand-placed waypoints per track: (x, y, steps_to_reach_from_previous).
# Fewer steps over a longer gap = the demo car is moving faster there.
_LINES: dict[str, list[tuple[int, int, int]]] = {
    # Each hop is chosen so the per-step displacement is an integer vector:
    # ease out of the start, cruise the straights, slow into the corners.
    "Easy": [
        (3, 4, 0), (5, 4, 2), (23, 4, 9), (27, 6, 2), (29, 10, 2),
        (30, 12, 1), (30, 18, 3), (30, 19, 1),
    ],
    "Medium": [
        (3, 4, 0), (5, 4, 2), (9, 4, 2), (11, 5, 1), (13, 9, 2),
        (13, 12, 2), (14, 12, 1), (22, 12, 4), (26, 13, 2), (27, 16, 2),
        (28, 20, 2), (30, 20, 1), (42, 20, 6), (43, 20, 1),
    ],
    "Hard": [
        (3, 4, 0), (5, 4, 2), (13, 4, 4), (15, 5, 1), (17, 6, 1),
        (21, 6, 2), (23, 5, 1), (27, 5, 2), (28, 6, 1), (28, 12, 3),
        (27, 12, 1), (13, 12, 7), (11, 13, 1), (10, 15, 1), (10, 19, 2),
        (11, 20, 1), (19, 19, 4), (21, 18, 1), (25, 18, 2), (27, 19, 1),
        (33, 20, 3), (34, 22, 1), (34, 26, 2), (33, 28, 1), (17, 28, 8),
        (15, 28, 1),
    ],
}


def _interpolate(waypoints: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    """Walk the hand-authored polyline, rounding to grid cells."""
    if not waypoints:
        return []
    path = [(waypoints[0][0], waypoints[0][1])]
    for (x0, y0, _), (x1, y1, steps) in zip(waypoints, waypoints[1:]):
        steps = max(1, steps)
        for i in range(1, steps + 1):
            t = i / steps
            cell = (round(x0 + (x1 - x0) * t), round(y0 + (y1 - y0) * t))
            if cell != path[-1]:
                path.append(cell)
    return path


def demo_trajectory(track: TrackMap) -> Trajectory:
    """A hand-drawn driving line, with velocity read off the position deltas."""
    path = _interpolate(_LINES.get(track.name, []))
    states: list[VehicleState] = []
    for i, (x, y) in enumerate(path):
        if i + 1 < len(path):
            vx, vy = path[i + 1][0] - x, path[i + 1][1] - y
        else:
            vx, vy = (0, 0)
        states.append(VehicleState(x=x, y=y, vx=vx, vy=vy, step=i))
    return Trajectory(
        states=states,
        label="Demo line",
        note="Hand-drawn polyline, not a rollout",
    )


def _finish_center(track: TrackMap) -> tuple[float, float]:
    cells = track.cells_of(FINISH)
    if not cells:
        return (track.width / 2, track.height / 2)
    return (
        sum(c[0] for c in cells) / len(cells),
        sum(c[1] for c in cells) / len(cells),
    )


def _slice_keys() -> list[tuple[int, int]]:
    return [(vx, vy) for vx in (-1, 0, 1, 2) for vy in (-1, 0, 1, 2)]


def demo_scalar_values(
    track: TrackMap, sharpness: float = 1.0
) -> dict[tuple[int, int, int, int], float]:
    """Raw ``{(x, y, vx, vy): value}`` behind :func:`demo_scalar_field`."""
    fx, fy = _finish_center(track)
    diag = math.hypot(track.width, track.height)
    values: dict[tuple[int, int, int, int], float] = {}
    for y in range(track.height):
        for x in range(track.width):
            if not track.is_drivable(x, y):
                continue
            dx, dy = fx - x, fy - y
            dist = math.hypot(dx, dy)
            norm = dist / max(diag, 1.0)
            ux, uy = (dx / dist, dy / dist) if dist else (0.0, 0.0)
            for vx, vy in _slice_keys():
                tilt = 0.9 * (vx * ux + vy * uy)
                values[(x, y, vx, vy)] = sharpness * (-28.0 * norm + tilt)
    return values


def demo_scalar_field(track: TrackMap, sharpness: float = 1.0) -> ScalarField:
    """``-distance to finish`` plus a directional tilt. Geometry, nothing else."""
    return ScalarField.from_any(
        demo_scalar_values(track, sharpness),
        label="Demo field",
        note="-distance to finish (geometric placeholder)",
    )


def demo_vector_values(track: TrackMap) -> dict[tuple[int, int, int, int], tuple[int, int]]:
    """Raw ``{(x, y, vx, vy): (u, v)}`` behind :func:`demo_vector_field`."""
    fx, fy = _finish_center(track)
    vectors: dict[tuple[int, int, int, int], tuple[int, int]] = {}
    for y in range(track.height):
        for x in range(track.width):
            if not track.is_drivable(x, y):
                continue
            dx, dy = fx - x, fy - y
            norm = max(abs(dx), abs(dy), 1e-9)
            snapped = (
                int(round(max(-1.0, min(1.0, dx / norm)))),
                int(round(max(-1.0, min(1.0, dy / norm)))),
            )
            for vx, vy in _slice_keys():
                vectors[(x, y, vx, vy)] = snapped
    return vectors


def demo_vector_field(track: TrackMap) -> VectorField:
    """Grid-snapped direction toward the finish. A compass, not a policy."""
    return VectorField.from_any(
        demo_vector_values(track),
        label="Demo directions",
        note="Compass toward the finish (geometric placeholder)",
    )


def demo_iterations(track: TrackMap, count: int = 12) -> list[Iteration]:
    """A flat field blended linearly into the geometric one, for the timeline."""
    vector = demo_vector_field(track)
    snapshots: list[Iteration] = []
    for k in range(count):
        progress = k / max(count - 1, 1)
        snapshots.append(
            Iteration(
                index=k,
                scalar=demo_scalar_field(track, sharpness=progress),
                vector=vector if k >= count // 3 else None,
                metrics={"blend": progress, "demo delta": 10.0 * math.exp(-0.45 * k)},
                label=f"Demo step {k}",
            )
        )
    return snapshots


def demo_session(track: TrackMap) -> LabSession:
    """A fully populated session, labelled as demo data everywhere it shows."""
    iterations = demo_iterations(track)
    return LabSession(
        track=track,
        trajectory=demo_trajectory(track),
        scalar=iterations[-1].scalar,
        vector=iterations[-1].vector,
        iterations=iterations,
        source="demo",
        title="Visualization preview",
        note=(
            "Placeholder geometry so the UI can be inspected before an "
            "experiment is connected."
        ),
    )
