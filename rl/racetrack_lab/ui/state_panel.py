"""Right-hand column: the paused step in detail, the clicked cell, the run."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ..schema import CELL_NAMES, LabSession, TrackMap, VehicleState
from . import components as c

_SKIP_EXTRAS = {"crashed", "crash", "collision", "off_track"}


def _fmt(v: float) -> str:
    return f"{v:g}"


def _fmt_pair(a: float, b: float) -> str:
    return f"({_fmt(a)}, {_fmt(b)})"


def _fmt_scalar(value: Any) -> str | None:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:g}" if abs(value) < 1e4 else f"{value:.3g}"
    if isinstance(value, str) and len(value) <= 14:
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return _fmt_pair(float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return None
    return None


def step_card(
    state: VehicleState | None, n_steps: int, playing: bool, has_velocity: bool = True
) -> None:
    """Details of the step the canvas is paused on (the HUD shows it live)."""
    if state is None:
        with c.panel("Current step"):
            st.markdown(
                '<div class="rvl-empty-body">No trajectory connected. Position, velocity '
                "and per-step data appear here once a rollout is provided.</div>",
                unsafe_allow_html=True,
            )
        return

    items: list[tuple[str, str] | tuple[str, str, str | None]] = [
        ("Position", _fmt_pair(state.x, state.y)),
    ]
    if has_velocity:
        items += [("Velocity", _fmt_pair(state.vx, state.vy)), ("Speed", f"{state.speed:.2f}")]
    items.append(("Step", f"{state.step if state.step is not None else '-'}", f"/ {max(n_steps - 1, 0)}"))
    if state.action is not None and len(state.action) >= 2:
        items.append(("Action", _fmt_pair(state.action[0], state.action[1])))
    if state.reward is not None:
        items.append(("Reward", _fmt(state.reward)))
    for key, value in list(state.extras.items()):
        if key in _SKIP_EXTRAS or len(items) >= 8:
            continue
        text = _fmt_scalar(value)
        if text is not None:
            items.append((key.replace("_", " ").title(), text))

    crashed = any(bool(state.extras.get(k)) for k in _SKIP_EXTRAS)
    note = "playing · live values in the canvas" if playing else ("collision" if crashed else "paused")
    c.card("Current step", items, note=note)


def rollout_card(session: LabSession) -> None:
    traj = session.trajectory
    if not traj or not len(traj):
        return
    speeds = [s.speed for s in traj.states]
    crashes = sum(1 for s in traj.states if any(bool(s.extras.get(k)) for k in _SKIP_EXTRAS))
    items = [("Steps", f"{len(traj)}")]
    if traj.has_velocity:
        items.append(("Peak speed", f"{max(speeds):.2f}"))
    if crashes:
        items.append(("Collisions", f"{crashes}"))
    c.card(traj.label or "Trajectory", items, note=traj.note)


def selected_cell_card(
    track: TrackMap,
    cell: tuple[int, int] | None,
    scalar_plane: dict[tuple[int, int], float] | None,
    vector_plane: dict[tuple[int, int], tuple[float, float]] | None,
    scalar_label: str | None,
    vector_label: str | None,
    slice_key: tuple[int, int] | None,
) -> None:
    """Inspector for the cell the user clicked on the canvas."""
    if cell is None:
        with c.panel("Selected cell"):
            st.markdown(
                '<div class="rvl-empty-body">Click any track cell to inspect it.</div>',
                unsafe_allow_html=True,
            )
        return

    x, y = cell
    items: list[tuple[str, str] | tuple[str, str, str | None]] = [
        ("Position", f"({x}, {y})"),
        ("Cell", CELL_NAMES.get(track.code_at(x, y), "Wall")),
    ]
    if scalar_plane and (x, y) in scalar_plane:
        items.append((scalar_label or "Value", f"{scalar_plane[(x, y)]:+.4g}"))
    if vector_plane and (x, y) in vector_plane:
        u, v = vector_plane[(x, y)]
        items.append((vector_label or "Vector", _fmt_pair(u, v)))

    note = f"v = ({slice_key[0]:+d}, {slice_key[1]:+d})" if slice_key else ""
    with c.panel("Selected cell", note=note):
        st.markdown(c.metrics_html(items, columns=2), unsafe_allow_html=True)
        if st.button("Clear selection", key="clear_selection", width="stretch"):
            st.session_state["selected_cell"] = None
            st.rerun()
