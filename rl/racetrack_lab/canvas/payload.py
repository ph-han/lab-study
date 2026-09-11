"""Turn a :class:`LabSession` into the JSON the canvas component draws.

The payload has three parts so the frontend can update in place:

* ``structure`` -- track, mode, options, colors, labels. Changing it rebuilds
  the SVG.
* ``data`` -- trajectory and field frames, stamped with a ``rev``. A new rev
  with the same structure recolors cells and moves the car *without* a
  rebuild, which is what keeps live streaming smooth.
* ``state`` -- step, iteration, selected cell, follow/live flags.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import palette
from ..render.colors import pick_ramp
from ..schema import LabSession, ScalarField, VectorField, VehicleState

MODES = ("Track", "Value", "Action", "Trajectory")
_CRASH_KEYS = ("crashed", "crash", "collision", "off_track")


@dataclass
class CanvasOptions:
    """Display switches from the sidebar."""

    show_grid: bool = True
    show_labels: bool = True
    show_velocity: bool = True
    show_trajectory: bool = True
    show_remaining: bool = True
    show_numbers: bool = False
    colorscale_mode: str = "auto"
    global_scale: bool = True
    sampling: int = 0  # 0 = auto
    max_frames: int = 40  # iteration slider resolution
    slice_key: tuple[int, int] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanvasFacts:
    """What the Streamlit panels need to caption the canvas."""

    scalar_plane: dict[tuple[int, int], float] | None = None
    vector_plane: dict[tuple[int, int], tuple[float, float]] | None = None
    scalar_label: str | None = None
    vector_label: str | None = None
    value_range: tuple[float, float] | None = None
    ramp: list[str] | None = None
    diverging: bool = False
    coverage: tuple[int, int] | None = None
    arrow_counts: tuple[int, int] | None = None
    empty_reason: str | None = None
    n_iterations: int = 0
    frame_indices: list[int] = field(default_factory=list)


def _plane(field_obj: Any, slice_key: tuple[int, int] | None) -> dict:
    if field_obj is None:
        return {}
    if field_obj.velocity_keyed:
        return field_obj.plane(tuple(slice_key)) if slice_key is not None else {}
    return field_obj.plane()


def _state_dict(s: VehicleState) -> dict[str, Any]:
    d: dict[str, Any] = {"x": s.x, "y": s.y, "vx": s.vx, "vy": s.vy, "step": s.step}
    if s.action is not None and len(s.action) >= 2:
        d["action"] = [s.action[0], s.action[1]]
    if s.reward is not None:
        d["reward"] = s.reward
    if any(bool(s.extras.get(k)) for k in _CRASH_KEYS):
        d["crashed"] = True
    return d


def auto_sampling(n_drivable: int) -> int:
    return 1 if n_drivable <= 420 else 2


def _subsample(n: int, max_frames: int) -> list[int]:
    """Indices of at most ``max_frames`` frames, always including the last one."""
    if n <= max_frames:
        return list(range(n))
    idx = np.linspace(0, n - 1, num=max_frames).round().astype(int)
    return sorted(set(idx.tolist()))


def _sources(session: LabSession, attr: str) -> list[tuple[int, str, Any]]:
    """``(index, label, field)`` per iteration, or the session field alone."""
    if session.iterations:
        out = []
        last = None
        for snap in session.iterations:
            f = getattr(snap, attr) or last or getattr(session, attr)
            last = f
            out.append((snap.index, snap.title, f))
        return out
    return [(0, "", getattr(session, attr))]


def build_payload(
    session: LabSession,
    mode: str,
    options: CanvasOptions,
    step: int,
    iteration: int,
    selected: tuple[int, int] | None,
    *,
    live: bool = False,
    follow: bool = False,
    hold: bool = False,
    update_label: str | None = None,
) -> tuple[dict[str, Any], CanvasFacts]:
    """Assemble ``{"structure", "data", "state"}`` and the facts for captions."""
    track = session.track
    facts = CanvasFacts()

    # ---- fields --------------------------------------------------------------
    scalar_block = None
    vector_block = None

    if mode == "Value":
        sources = _sources(session, "scalar")
        pick = _subsample(len(sources), options.max_frames)
        chosen = [sources[i] for i in pick]
        facts.frame_indices = [idx for idx, _, _ in chosen]
        base = session.scalar if not session.iterations else (
            chosen[min(_pos(iteration, facts.frame_indices), len(chosen) - 1)][2] if chosen else None
        )
        facts.scalar_label = base.label if base else None
        frames = [
            [[x, y, float(v)] for (x, y), v in _plane(f, options.slice_key).items() if track.is_drivable(x, y)]
            for _, _, f in chosen
        ]
        if base is None and not any(f for _, _, f in chosen):
            facts.empty_reason = "no-scalar"
        elif not any(frames):
            facts.empty_reason = "empty-slice"
        else:
            if options.global_scale:
                vmin, vmax = _global_range([f for _, _, f in chosen])
            else:
                cur = frames[min(_pos(iteration, facts.frame_indices), len(frames) - 1)]
                vals = [v for (_, _, v) in cur] or [v for fr in frames for (_, _, v) in fr]
                vmin, vmax = min(vals), max(vals)
            if vmin == vmax:
                vmin, vmax = vmin - 0.5, vmax + 0.5
            ramp, diverging, vmin, vmax = pick_ramp(vmin, vmax, options.colorscale_mode)
            label = (base.label if base else None) or "Value"
            scalar_block = {
                "label": label,
                "note": base.note if base else "",
                "vmin": vmin,
                "vmax": vmax,
                "ramp": ramp,
                "diverging": diverging,
                "frames": frames,
                "frameLabels": [lbl for _, lbl, _ in chosen],
                "frameIndices": facts.frame_indices,
            }
            cur = frames[min(_pos(iteration, facts.frame_indices), len(frames) - 1)]
            plane = {(x, y): v for (x, y, v) in cur}
            facts.scalar_plane = plane
            facts.scalar_label = label
            facts.value_range = (vmin, vmax)
            facts.ramp = ramp
            facts.diverging = diverging
            facts.coverage = (len(plane), track.n_drivable)
        facts.n_iterations = len(sources)

    elif mode == "Action":
        sources = _sources(session, "vector")
        pick = _subsample(len(sources), options.max_frames)
        chosen = [sources[i] for i in pick]
        facts.frame_indices = [idx for idx, _, _ in chosen]
        base = session.vector if not session.iterations else (
            chosen[min(_pos(iteration, facts.frame_indices), len(chosen) - 1)][2] if chosen else None
        )
        facts.vector_label = base.label if base else None
        frames = [
            [[x, y, float(u), float(v)] for (x, y), (u, v) in _plane(f, options.slice_key).items() if track.is_drivable(x, y)]
            for _, _, f in chosen
        ]
        if base is None and not any(f for _, _, f in chosen):
            facts.empty_reason = "no-vector"
        elif not any(frames):
            facts.empty_reason = "empty-slice"
        else:
            sampling = options.sampling or auto_sampling(track.n_drivable)
            peak = max((math.hypot(u, v) for fr in frames for (_, _, u, v) in fr), default=1.0) or 1.0
            label = (base.label if base else None) or "Vector"
            vector_block = {
                "label": label,
                "note": base.note if base else "",
                "sampling": sampling,
                "peak": peak,
                "frames": frames,
                "frameLabels": [lbl for _, lbl, _ in chosen],
                "frameIndices": facts.frame_indices,
            }
            cur = frames[min(_pos(iteration, facts.frame_indices), len(frames) - 1)]
            plane = {(x, y): (u, v) for (x, y, u, v) in cur}
            facts.vector_plane = plane
            facts.vector_label = label
            drawn = sum(1 for (x, y) in plane if not (x % sampling or y % sampling))
            facts.arrow_counts = (drawn, len(plane))
        facts.n_iterations = len(sources)

    # ---- trajectory ----------------------------------------------------------
    traj_block = None
    traj = session.trajectory
    if traj and len(traj):
        traj_block = {
            "label": traj.label,
            "hasVelocity": traj.has_velocity,
            "states": [_state_dict(s) for s in traj.states],
        }

    # ---- assemble -------------------------------------------------------------
    slice_label = None
    if options.slice_key is not None:
        vx, vy = options.slice_key
        slice_label = f"vx = {vx:+d}   vy = {vy:+d}"

    structure = {
        "mode": mode,
        "track": {"name": track.name, "width": track.width, "height": track.height, "grid": track.grid.tolist()},
        "options": {
            "grid": options.show_grid,
            "labels": options.show_labels,
            "velocity": options.show_velocity,
            "trajectory": options.show_trajectory,
            "remaining": options.show_remaining,
            "numbers": options.show_numbers,
        },
        "colors": _colors(),
        "source": session.source,
        "live": live,
        "sliceLabel": slice_label,
        "hasScalar": scalar_block is not None,
        "hasVector": vector_block is not None,
        "hasTrajectory": traj_block is not None,
        "scalarLabel": scalar_block["label"] if scalar_block else None,
        "vectorLabel": vector_block["label"] if vector_block else None,
    }
    data = {"trajectory": traj_block, "scalar": scalar_block, "vector": vector_block, "updateLabel": update_label}
    data["rev"] = hashlib.blake2b(json.dumps(data, sort_keys=True, default=float).encode(), digest_size=8).hexdigest()

    frame_pos = _pos(iteration, facts.frame_indices) if facts.frame_indices else 0
    state = {
        "step": int(step),
        "iteration": int(frame_pos),
        "selected": list(selected) if selected is not None else None,
        "follow": bool(follow),
        "hold": bool(hold),
        "live": bool(live),
    }
    return {"structure": structure, "data": data, "state": state}, facts


def _pos(iteration_index: int, frame_indices: list[int]) -> int:
    """Position in the (possibly subsampled) frame list closest to an iteration index."""
    if not frame_indices:
        return 0
    if iteration_index in frame_indices:
        return frame_indices.index(iteration_index)
    # iteration_index may be a raw position (non-live) or a seq (live): nearest wins
    best = min(range(len(frame_indices)), key=lambda i: abs(frame_indices[i] - iteration_index))
    return best


def _global_range(fields: list[ScalarField | None]) -> tuple[float, float]:
    lo, hi = math.inf, -math.inf
    for f in fields:
        if f is None:
            continue
        bounds = f.value_bounds()
        if bounds is None:
            continue
        lo = min(lo, bounds[0])
        hi = max(hi, bounds[1])
    if not math.isfinite(lo):
        return (0.0, 1.0)
    return (lo, hi)


def _colors() -> dict[str, str]:
    return {
        "grass": palette.GRASS,
        "grassDot": palette.GRASS_DOT,
        "asphalt": palette.ASPHALT,
        "asphaltDim": palette.ASPHALT_DIM,
        "kerb": palette.KERB,
        "grid": palette.GRID,
        "start": palette.START,
        "startFill": palette.START_FILL,
        "finishDark": palette.FINISH_DARK,
        "finishLight": palette.FINISH_LIGHT,
        "car": palette.CAR,
        "carDark": palette.CAR_DARK,
        "carGlass": palette.CAR_GLASS,
        "carWheel": palette.CAR_WHEEL,
        "carLight": palette.CAR_LIGHT,
        "trail": palette.TRAIL,
        "trailCasing": palette.TRAIL_CASING,
        "velocity": palette.VELOCITY,
        "arrow": palette.ARROW,
        "selection": palette.SELECTION,
        "crash": palette.CRASH,
        "ink": palette.INK,
        "inkMuted": palette.INK_MUTED,
    }


__all__ = ["MODES", "CanvasOptions", "CanvasFacts", "build_payload", "auto_sampling", "VectorField"]
