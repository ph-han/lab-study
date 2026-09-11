"""Sidebar: the live-run picker and display settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from ..canvas.payload import CanvasOptions
from ..live import list_runs
from ..schema import LabSession
from . import components as c

_SAMPLING = {"Auto": 0, "Every cell": 1, "Every 2nd cell": 2, "Every 3rd cell": 3}
_POLL = {"0.25 s": 0.25, "0.5 s": 0.5, "1 s": 1.0, "2 s": 2.0}
_OFF = "— off —"


@dataclass
class LiveConfig:
    root: str = "runs"
    run: Path | None = None
    follow: bool = True
    poll: float = 0.5


def render_live() -> LiveConfig:
    """Top of the sidebar: which run directory to tail, and how."""
    with st.sidebar:
        st.markdown('<div class="rvl-side-title">Live run</div>', unsafe_allow_html=True)
        root = st.text_input(
            "Runs folder",
            value="runs",
            key="live_root",
            help="Each LiveRun(path) you create becomes a folder here. Relative paths are resolved from the rl/ folder, on both sides.",
        )
        runs = list_runs(root)
        names = [_OFF] + [p.name for p in runs]
        if st.session_state.get("live_run") not in names:
            st.session_state["live_run"] = names[1] if len(names) > 1 else _OFF
        choice = st.selectbox(
            "Run",
            names,
            key="live_run",
            help="Newest first. Pick “off” to fall back to connect.py or the preview.",
        )
        follow = st.toggle(
            "Follow live updates",
            value=True,
            key="live_follow",
            help="Poll the run folder and glide the car / recolor the field as updates land.",
        )
        poll_label = st.selectbox("Poll every", list(_POLL), index=1, key="live_poll")
        if st.button("Reload now", icon=":material/refresh:", key="live_reload", width="stretch"):
            st.session_state["live_hold"] = False
            st.session_state["live_resume_nonce"] = int(st.session_state.get("live_resume_nonce", 1)) + 1
        st.caption("Publish from your training loop with racetrack_lab.live.LiveRun.")
        st.divider()

    run = next((p for p in runs if p.name == choice), None) if choice != _OFF else None
    return LiveConfig(root=root, run=run, follow=follow, poll=_POLL[poll_label])


def render_display(session: LabSession, connected: bool) -> CanvasOptions:
    """Rest of the sidebar: display settings only."""
    with st.sidebar:
        st.markdown('<div class="rvl-side-title">Display</div>', unsafe_allow_html=True)

        badge = (
            c.badge("Live", "live dot")
            if session.source == "live"
            else c.badge("Connected", "live dot")
            if session.source == "connected"
            else c.badge("Demo data", "demo dot")
            if session.source == "demo"
            else c.badge("No data", "none dot")
        )
        st.markdown(badge, unsafe_allow_html=True)
        if session.source in ("demo", "none") and not connected:
            st.toggle(
                "Show preview data",
                key="use_demo",
                help="Placeholder geometry for checking the UI. Turn it off to see the empty states.",
            )
            st.caption("Return a session from racetrack_lab/connect.py to plug in a run.")
        elif session.source == "connected":
            st.caption("Data supplied by racetrack_lab/connect.py.")
        else:
            st.caption(f"Tailing runs/{session.title}.")

        st.divider()
        st.markdown('<div class="rvl-side-title">Layers</div>', unsafe_allow_html=True)
        show_trajectory = st.toggle("Trail", value=True, key="opt_traj")
        show_remaining = st.toggle(
            "Remaining path",
            value=True,
            key="opt_remaining",
            disabled=not show_trajectory,
            help="Dotted preview of the part of the path not yet reached.",
        )
        show_velocity = st.toggle("Velocity arrow", value=True, key="opt_velocity")
        show_grid = st.toggle("Cell grid", value=True, key="opt_grid")
        show_labels = st.toggle("Start / finish labels", value=True, key="opt_labels")

        st.divider()
        st.markdown('<div class="rvl-side-title">Value overlay</div>', unsafe_allow_html=True)
        colorscale_mode = st.selectbox(
            "Color scale",
            ["auto", "sequential", "diverging"],
            index=0,
            key="opt_colorscale",
            help="auto: one hue for magnitude, blue/orange diverging only when the data straddles zero.",
        )
        scale_choice = st.selectbox(
            "Scale range",
            ["Whole field", "Current slice"],
            index=0,
            key="opt_scale",
            help="Whether the color range is shared across velocity slices and iterations.",
        )
        show_numbers = st.toggle("Cell numbers", value=False, key="opt_numbers", help="Readable on small tracks only.")
        max_frames = st.select_slider(
            "Iteration slider frames",
            options=[10, 20, 40, 80, 160],
            value=40,
            key="opt_frames",
            help="Long histories are thinned to this many snapshots for the slider (the newest is always kept).",
        )

        st.divider()
        st.markdown('<div class="rvl-side-title">Action overlay</div>', unsafe_allow_html=True)
        sampling_label = st.selectbox(
            "Arrow density",
            list(_SAMPLING.keys()),
            index=0,
            key="opt_sampling",
            help="Fewer arrows keeps a dense field readable.",
        )

        st.divider()
        st.caption("x runs right, y runs up, origin bottom-left. Fields are keyed by (x, y) or (x, y, vx, vy).")

    return CanvasOptions(
        show_grid=show_grid,
        show_labels=show_labels,
        show_velocity=show_velocity,
        show_trajectory=show_trajectory,
        show_remaining=show_remaining and show_trajectory,
        show_numbers=show_numbers,
        colorscale_mode=colorscale_mode,
        global_scale=(scale_choice == "Whole field"),
        sampling=_SAMPLING[sampling_label],
        max_frames=int(max_frames),
    )
