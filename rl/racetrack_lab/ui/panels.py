"""The detail band under the canvas: one panel per visualization mode."""

from __future__ import annotations

import streamlit as st

from ..canvas.payload import CanvasFacts
from ..render import PLOTLY_CONFIG, charts, css_gradient
from ..schema import LabSession, VehicleState
from ..tracks import TRACK_SPECS
from . import components as c
from . import controls

_EMPTY_TEXT = {
    "no-scalar": (
        "Value data is not available yet",
        "Return a <code>scalar_field</code> from <code>racetrack_lab/connect.py</code> "
        "&mdash; a mapping of <code>(x, y)</code> or <code>(x, y, vx, vy)</code> to a "
        "number, or a 2D array indexed <code>[y, x]</code> &mdash; and it is painted "
        "onto the circuit.",
    ),
    "no-vector": (
        "Action data is not available yet",
        "Return a <code>vector_field</code> from <code>racetrack_lab/connect.py</code> "
        "&mdash; a mapping of <code>(x, y)</code> or <code>(x, y, vx, vy)</code> to "
        "<code>(u, v)</code> &mdash; and it is drawn as arrows.",
    ),
    "empty-slice": (
        "This velocity slice is empty",
        "The field carries no entry for the selected <code>(vx, vy)</code>. "
        "Pick another slice.",
    ),
    "no-trajectory": (
        "No trajectory connected",
        "Return a <code>trajectory</code> from <code>racetrack_lab/connect.py</code> "
        "&mdash; any sequence of <code>(x, y)</code> or <code>(x, y, vx, vy)</code> "
        "states &mdash; to see the car drive, with playback and a speed profile.",
    ),
}


def empty(reason: str) -> None:
    title, body = _EMPTY_TEXT[reason]
    c.empty_state(title, body)


def _slice_card(state: VehicleState | None, field, what: str) -> None:
    keyed = field is not None and field.velocity_keyed
    note = f"{len(field.slices)} slices" if keyed else ""
    with c.panel("Velocity slice", note=note):
        if keyed:
            st.markdown(
                c.slice_readout((st.session_state.get("slice_vx"), st.session_state.get("slice_vy"))),
                unsafe_allow_html=True,
            )
            controls.velocity_slice_picker(field.vx_values, field.vy_values)
            controls.follow_vehicle_button(state, field)
        else:
            st.markdown(c.slice_readout(None), unsafe_allow_html=True)
            st.markdown(
                f'<div class="rvl-card-note" style="text-align:left">{what} is keyed by '
                "position only, so one plane shows all of it.</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------
def track_panel(session: LabSession) -> None:
    track = session.track
    spec = TRACK_SPECS.get(track.name)
    left, right = st.columns([1, 1.3], gap="medium")
    with left:
        c.card(
            "Circuit",
            [
                ("Track", track.name),
                ("Grid", f"{track.width} x {track.height}"),
                ("Drivable cells", f"{track.n_drivable}"),
                ("Start cells", f"{len(track.cells_of(2))}"),
                ("Finish cells", f"{len(track.cells_of(3))}"),
            ],
        )
    with right:
        with c.panel("Reading this track", note=track.difficulty):
            st.markdown(
                f'<div class="rvl-text"><b>{spec.shape if spec else ""}</b><br>{track.note}</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Value
# ---------------------------------------------------------------------------
def value_panel(session: LabSession, facts: CanvasFacts, state: VehicleState | None, iteration: int) -> None:
    scalar, _ = session.fields_at(iteration if session.iterations else None)
    if facts.empty_reason == "no-scalar":
        empty("no-scalar")
        return

    left, right = st.columns([1, 1.35], gap="medium")
    with left:
        _slice_card(state, scalar, facts.scalar_label or "This field")
        if facts.value_range is not None and facts.ramp:
            covered, total = facts.coverage or (0, 0)
            with c.panel(facts.scalar_label or "Scalar field", note=scalar.note if scalar else ""):
                st.markdown(
                    c.scale_bar(facts.value_range[0], facts.value_range[1], css_gradient(facts.ramp)),
                    unsafe_allow_html=True,
                )
                kind = "diverging, centred on zero" if facts.diverging else "one hue, light = low"
                st.markdown(
                    f'<div class="rvl-card-note" style="text-align:left;margin-top:.45rem">'
                    f"{covered} of {total} drivable cells carry a value in this slice · {kind}</div>",
                    unsafe_allow_html=True,
                )
    with right:
        if facts.empty_reason == "empty-slice":
            empty("empty-slice")
        elif facts.scalar_plane:
            with c.panel("Distribution", note="cells per value bucket, current slice"):
                st.plotly_chart(
                    charts.value_distribution(list(facts.scalar_plane.values()), facts.scalar_label or "value"),
                    config=PLOTLY_CONFIG,
                    key="value_hist",
                )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------
def action_panel(session: LabSession, facts: CanvasFacts, state: VehicleState | None, iteration: int) -> None:
    _, vector = session.fields_at(iteration if session.iterations else None)
    if facts.empty_reason == "no-vector":
        empty("no-vector")
        return

    left, right = st.columns([1, 1.35], gap="medium")
    with left:
        _slice_card(state, vector, facts.vector_label or "This field")
    with right:
        if facts.empty_reason == "empty-slice":
            empty("empty-slice")
            return
        drawn, total = facts.arrow_counts or (0, 0)
        with c.panel(facts.vector_label or "Vector field", note=vector.note if vector else ""):
            st.markdown(
                f'<div class="rvl-text">Showing <b>{drawn}</b> of <b>{total}</b> vectors in this slice. '
                "Arrow rotation is direction, arrow length is magnitude; a ring marks a zero vector.</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="margin-top:.5rem">'
                + c.chips(
                    [
                        ("density", str(st.session_state.get("opt_sampling", "Auto"))),
                        ("max magnitude", f"{vector.max_magnitude():.3g}" if vector else "-"),
                    ]
                )
                + "</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------
def trajectory_panel(session: LabSession, step: int) -> None:
    traj = session.trajectory
    if not traj or not len(traj):
        empty("no-trajectory")
        return

    left, right = st.columns([1.45, 1], gap="medium")
    with left:
        if traj.has_velocity:
            with c.panel("Speed profile", note=f"step {step} of {len(traj) - 1}"):
                st.plotly_chart(charts.speed_profile(traj, step), config=PLOTLY_CONFIG, key="speed_profile")
        else:
            with c.panel("Speed profile", note="position-only states"):
                st.markdown(
                    '<div class="rvl-text">This trajectory carries no velocity, so there is no speed to '
                    "plot. Pass <code>(x, y, vx, vy)</code> states to get the profile.</div>",
                    unsafe_allow_html=True,
                )
    with right:
        with c.panel("Step log", note="window around the current step"):
            window = 4
            start = max(0, step - window)
            end = min(len(traj), step + window + 1)
            rows = []
            for i in range(start, end):
                s = traj[i]
                row = {
                    "step": ("▸ " if i == step else "   ") + str(s.step if s.step is not None else i),
                    "pos": f"({s.x:g}, {s.y:g})",
                }
                if traj.has_velocity:
                    row["vel"] = f"({s.vx:g}, {s.vy:g})"
                    row["speed"] = f"{s.speed:.2f}"
                if s.action is not None and len(s.action) >= 2:
                    row["action"] = f"({s.action[0]:g}, {s.action[1]:g})"
                if s.reward is not None:
                    row["reward"] = f"{s.reward:g}"
                rows.append(row)
            st.dataframe(rows, hide_index=True, height=210)


# ---------------------------------------------------------------------------
# Iteration metrics (the iteration slider itself lives in the canvas)
# ---------------------------------------------------------------------------
def iteration_panel(session: LabSession, iteration: int) -> None:
    snapshots = session.iterations
    log = session.metrics_log or [
        {"index": s.index, "label": s.title, "metrics": s.metrics} for s in snapshots
    ]
    if len(log) < 2:
        return
    metrics = {k for row in log for k in (row.get("metrics") or {})}
    if not metrics:
        return
    snap = session.snapshot(iteration)
    names = sorted(metrics)
    with c.panel("Iteration metrics", note=f"{snap.title if snap else ''}  ·  {len(log)} updates"):
        cols = st.columns([1, 2.4], gap="medium")
        with cols[0]:
            if snap and snap.metrics:
                st.markdown(
                    c.metrics_html([(k, f"{v:.4g}") for k, v in list(snap.metrics.items())[:4]], columns=1, size="sm"),
                    unsafe_allow_html=True,
                )
            st.caption("Scrub iterations with the slider under the canvas.")
        with cols[1]:
            metric = st.selectbox("Metric", names, key="metric_name", label_visibility="collapsed")
            series = [
                (int(row.get("index", i)), float(row["metrics"][metric]))
                for i, row in enumerate(log)
                if metric in (row.get("metrics") or {})
            ]
            if series:
                target = snap.index if snap else series[-1][0]
                current = min(range(len(series)), key=lambda i: abs(series[i][0] - target))
                st.plotly_chart(
                    charts.metric_history(
                        [v for _, v in series],
                        [i for i, _ in series],
                        current=current,
                        label=metric,
                        height=140,
                    ),
                    config=PLOTLY_CONFIG,
                    key="metric_chart",
                )
