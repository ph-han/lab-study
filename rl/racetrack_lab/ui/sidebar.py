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

# What the radio says -> which engine app.py runs. Your two packages read the
# same maps, so switching is one click and no other control changes.
_ENGINES = {"정책 반복": "policy", "가치 반복": "value", "데모": "demo"}


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


# ---------------------------------------------------------------------------
# Map picker: the only choice that matters when the app computes it itself.
# ---------------------------------------------------------------------------
@dataclass
class MapConfig:
    text: str
    name: str
    gamma: float
    sweeps: int
    reward: dict
    engine: str = "policy"
    run: Path | None = None


def render_map() -> MapConfig:
    """Pick a room, or write one. Everything else has a sane default."""
    from .. import rooms
    from ..demo_dp import DEFAULT_REWARD
    from ..schema import FINISH, START, TRACK, WALL

    with st.sidebar:
        st.markdown('<div class="rvl-side-title">엔진</div>', unsafe_allow_html=True)
        engine = st.radio(
            "엔진", list(_ENGINES), horizontal=True, key="engine_mode",
            label_visibility="collapsed",
            help="정책 반복 = algorithms/bellman_dp_policy_iteration, "
                 "가치 반복 = algorithms/bellman_dp_value_iteration 을 그대로 실행하고 계산을 기록합니다.",
        )
        st.markdown('<div class="rvl-side-title">맵</div>', unsafe_allow_html=True)
        names = list(rooms.ROOMS) + ["직접 쓰기"]
        start = names.index("3r4c_wall") if "3r4c_wall" in names else 0
        choice = st.selectbox("방", names, index=start, key="map_choice", label_visibility="collapsed")
        default = rooms.ROOMS.get(choice, rooms.ROOMS["3r4c_wall"])
        text = st.text_area(
            "맵",
            value="\n".join(line.strip() for line in default.strip().splitlines()),
            key=f"map_text_{choice}",
            height=140,
            help=". 빈 칸   # 벽   S 시작   G 목표   ·   맨 윗줄이 맵의 위쪽입니다",
        )
        st.caption(". 빈 칸 · # 벽 · S 시작 · G 목표")

        with st.expander("규칙", expanded=False):
            gamma = st.slider("γ 할인율", 0.0, 0.99, 0.9, 0.01, key="map_gamma")
            sweeps = st.slider("평가 바퀴 수", 1, 40, 8, 1, key="map_sweeps")
            c1, c2 = st.columns(2)
            with c1:
                r_wall = st.number_input("벽 충돌", value=float(DEFAULT_REWARD[WALL]), step=0.05, key="r_wall")
                r_track = st.number_input("이동", value=float(DEFAULT_REWARD[TRACK]), step=0.05, key="r_track")
            with c2:
                r_start = st.number_input("시작 칸", value=float(DEFAULT_REWARD[START]), step=0.05, key="r_start")
                r_goal = st.number_input("목표 진입", value=float(DEFAULT_REWARD[FINISH]), step=0.5, key="r_goal")

        # Your own runs, when there are any. One control, no plumbing.
        found = list_runs("runs")
        run = None
        if found:
            st.markdown('<div class="rvl-side-title">내 실행</div>', unsafe_allow_html=True)
            labels = ["끄기 (맵 계산)"] + [p.name for p in found]
            pick = st.selectbox("실행", labels, key="own_run", label_visibility="collapsed",
                                help="LiveRun 으로 발행한 폴더. 고르면 맵 대신 그 실행을 읽습니다.")
            run = next((p for p in found if p.name == pick), None)
        st.divider()

    return MapConfig(
        text=text,
        name=choice if choice != "직접 쓰기" else "custom",
        gamma=float(gamma),
        sweeps=int(sweeps),
        reward={WALL: r_wall, TRACK: r_track, START: r_start, FINISH: r_goal},
        engine=_ENGINES.get(engine, "policy"),
        run=run,
    )
