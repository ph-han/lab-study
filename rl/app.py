"""Racetrack Visual Lab -- run with ``streamlit run app.py``.

Reading order on screen: the circuit and the car first, the step readout
beside it, the detail band last. The canvas is a small custom component that
animates on its own; when a live run is being followed, the main area is a
fragment that polls the run folder and pushes fresh data into that canvas,
which glides the car and recolors the field in place.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from streamlit.errors import StreamlitAPIException

from racetrack_lab import connect, get_track
from racetrack_lab.canvas import build_payload, racetrack_canvas
from racetrack_lab.demo import demo_session
from racetrack_lab.live import LiveReader, build_live_session
from racetrack_lab.schema import LabSession
from racetrack_lab.ui import components as c
from racetrack_lab.ui import controls, panels, sidebar, state_panel, styles

st.set_page_config(
    page_title="Racetrack Visual Lab",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)
styles.inject()


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
def live_reader(path) -> LiveReader:
    readers = st.session_state.setdefault("live_readers", {})
    key = str(path)
    if key not in readers:
        readers[key] = LiveReader(path)
    return readers[key]


def resolve_session(live_cfg: sidebar.LiveConfig) -> tuple[LabSession, LiveReader | None, bool]:
    """Live run > connect.py > preview > nothing. Returns (session, reader, connected)."""
    track = get_track(st.session_state.get("track_name", "Easy"))
    if live_cfg.run is not None:
        reader = live_reader(live_cfg.run)
        if reader.exists():
            return (
                build_live_session(reader, track, max_frames=int(st.session_state.get("opt_frames", 40))),
                reader,
                False,
            )
    connected_session = connect.load_session(track)
    if connected_session is not None:
        connected_session.source = "connected"
        return connected_session, None, True
    if st.session_state.get("use_demo", True):
        return demo_session(track), None, False
    return LabSession(track=track, source="none"), None, False


def apply_canvas_event(event: Any) -> None:
    """Fold an event reported by the canvas into session state (once)."""
    if not isinstance(event, dict):
        return
    nonce = event.get("nonce")
    if nonce is None or nonce == st.session_state.get("canvas_nonce"):
        return
    st.session_state["canvas_nonce"] = nonce
    kind = event.get("type")
    if kind == "step":
        st.session_state["step"] = int(event.get("step", 0))
        st.session_state["playing"] = False
    elif kind == "play":
        st.session_state["playing"] = bool(event.get("playing"))
    elif kind == "select":
        cell = event.get("cell")
        st.session_state["selected_cell"] = tuple(int(v) for v in cell) if cell else None
    elif kind == "iteration":
        st.session_state["iteration_index"] = int(event.get("index", 0))
    if "hold" in event:
        st.session_state["live_hold"] = bool(event.get("hold"))
    try:
        st.rerun(scope="fragment")  # cheap: only the main area redraws
    except StreamlitAPIException:
        st.rerun()  # during a full-app run a fragment-scoped rerun is not allowed


def iteration_position(session: LabSession, index: int | None) -> int:
    """Position in ``session.iterations`` closest to an iteration index."""
    if not session.iterations:
        return 0
    if index is None:
        return len(session.iterations) - 1
    indices = [it.index for it in session.iterations]
    if index in indices:
        return indices.index(index)
    return min(range(len(indices)), key=lambda i: abs(indices[i] - index))


def live_status(reader: LiveReader) -> tuple[str, str]:
    """``(status, caption)`` for the header badge."""
    if reader.done():
        return "done", "finished"
    age = reader.age()
    if age is None:
        return "stale", "waiting for the first update"
    if age > 30:
        return "stale", f"last update {age:.0f} s ago"
    return "live", f"updated {age:.1f} s ago"


# ---------------------------------------------------------------------------
# Sidebar (outside the polling fragment)
# ---------------------------------------------------------------------------
live_cfg = sidebar.render_live()
session0, reader0, connected0 = resolve_session(live_cfg)
options = sidebar.render_display(session0, connected0)


# ---------------------------------------------------------------------------
# Main area -- polls the run folder while following a live run
# ---------------------------------------------------------------------------
def main_area() -> None:
    session, reader, is_connected = resolve_session(live_cfg)
    live = reader is not None
    follow = live and live_cfg.follow and not reader.done()
    hold = bool(st.session_state.get("live_hold", False))

    n_steps = len(session.trajectory) if session.trajectory else 0
    n_iterations = len(session.iterations)
    controls.ensure_state(n_steps, n_iterations)

    if follow and not hold:
        step = max(n_steps - 1, 0)
        iter_pos = max(n_iterations - 1, 0)
    else:
        step = int(st.session_state["step"])
        iter_pos = iteration_position(session, st.session_state.get("iteration_index"))
    iter_index = session.iterations[iter_pos].index if n_iterations else 0
    mode = st.session_state["mode"]
    vehicle_now = controls.current_state(session, step)

    scalar_field, vector_field = session.fields_at(iter_pos if n_iterations else None)
    active_field = scalar_field if mode == "Value" else vector_field if mode == "Action" else None
    slice_key = controls.seed_slice(
        active_field,
        hint=(vehicle_now.vx, vehicle_now.vy) if vehicle_now is not None else None,
    )
    options.slice_key = slice_key

    # ---- header -----------------------------------------------------------------
    head_left, head_right = st.columns([2.4, 1.35], vertical_alignment="bottom")
    with head_left:
        c.title_block("Racetrack Visual Lab", "Watch your racetrack MDP drive")
    with head_right:
        badges = [controls.data_badge(session.source)]
        if live:
            status, caption = live_status(reader)
            badges.append(c.badge(f"{reader.count()} updates · {caption}", "mono"))
        badges.append(
            c.badge(f"{session.track.width} × {session.track.height} · {session.track.n_drivable} cells", "mono")
        )
        if live:
            badges.append(c.badge(f"run {session.title} · track {session.track.name}", "mono"))
        c.meta_row(badges)
        if not live:
            controls.track_selector()

    if session.source == "demo":
        st.markdown(
            '<div class="rvl-text" style="margin:-0.2rem 0 0.2rem"><b>Preview.</b> Placeholder geometry, '
            "not a run. Publish a run with <code>racetrack_lab.live.LiveRun</code>, connect an experiment in "
            "<code>racetrack_lab/connect.py</code>, or switch the preview off in the sidebar.</div>",
            unsafe_allow_html=True,
        )

    # ---- canvas + step readout -------------------------------------------------
    canvas_col, panel_col = st.columns([2.85, 1], gap="medium")
    with canvas_col:
        mode = controls.mode_selector()
        update_label = None
        if live and session.iterations:
            update_label = session.iterations[-1].title
        payload, facts = build_payload(
            session,
            mode=mode,
            options=options,
            step=step,
            iteration=iter_index,
            selected=st.session_state.get("selected_cell"),
            live=live,
            follow=follow,
            hold=hold,
            update_label=update_label,
        )
        if live:
            payload["state"]["liveStatus"] = live_status(reader)[0]
            payload["state"]["pollMs"] = int(live_cfg.poll * 1000)
            payload["state"]["resumeNonce"] = int(st.session_state.get("live_resume_nonce", 1))
        event = racetrack_canvas(payload, key="racetrack_canvas")

    with panel_col:
        state_panel.step_card(
            vehicle_now,
            n_steps,
            bool(st.session_state.get("playing")) or (follow and not hold),
            has_velocity=bool(session.trajectory and session.trajectory.has_velocity),
        )
        state_panel.selected_cell_card(
            session.track,
            st.session_state.get("selected_cell"),
            facts.scalar_plane,
            facts.vector_plane,
            facts.scalar_label,
            facts.vector_label,
            slice_key,
        )
        state_panel.rollout_card(session)

    # ---- detail band -------------------------------------------------------------
    st.markdown('<div style="height:.2rem"></div>', unsafe_allow_html=True)
    if mode in ("Value", "Action") and n_iterations:
        panels.iteration_panel(session, iter_pos)
    if mode == "Track":
        panels.track_panel(session)
    elif mode == "Value":
        panels.value_panel(session, facts, vehicle_now, iter_pos)
    elif mode == "Action":
        panels.action_panel(session, facts, vehicle_now, iter_pos)
    elif mode == "Trajectory":
        panels.trajectory_panel(session, step)

    apply_canvas_event(event)


poll_every = live_cfg.poll if (reader0 is not None and live_cfg.follow and not reader0.done()) else None
st.fragment(run_every=poll_every)(main_area)()
