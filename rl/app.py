"""Bellman DP Lab -- run with ``streamlit run app.py``.

Pick a map; the app evaluates it here and now and shows how each value was
made. One screen: the map on the left, the calculation behind the selected
cell on the right. Selecting, stepping and autoplay happen inside the view, so
Python only reruns when the map or the rules change.

The engine radio picks whose arithmetic is on screen: your policy iteration,
your value iteration (``algorithms/bellman_dp_*``, run and recorded by
``racetrack_lab.user_engine``), or the app's own demo evaluation (see
``racetrack_lab.demo_dp``). A live run, or a ``(session, trace)`` returned from
``racetrack_lab/connect.py``, takes precedence over all three.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from racetrack_lab import connect, demo_dp, get_track, rooms, user_engine
from racetrack_lab.live import LiveReader, build_live_session
from racetrack_lab.schema import LabSession
from racetrack_lab.trace import Trace, read_trace
from racetrack_lab.ui import bellman_view
from racetrack_lab.ui import components as c
from racetrack_lab.ui import sidebar, styles

st.set_page_config(
    page_title="Bellman DP Lab",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)
styles.inject()


@st.cache_data(show_spinner=False)
def evaluate(text: str, gamma: float, sweeps: int, reward_items: tuple, name: str):
    """Cached so dragging a slider recomputes once, not on every rerun."""
    return demo_dp.evaluate(text, gamma=gamma, sweeps=sweeps, reward=dict(reward_items), name=name)


@st.cache_data(show_spinner=False)
def run_user_code(text: str, sweeps: int, name: str, algo: str):
    """Your algorithms package, imported and run. Cached per map, sweeps and engine."""
    return user_engine.run(rooms.parse(text, name=name), sweeps=sweeps, algo=algo)


def plane_of(field_obj: Any) -> dict:
    if field_obj is None or getattr(field_obj, "velocity_keyed", False):
        return {}
    return field_obj.plane()


def fixed_range(frames, plane: dict) -> tuple[float, float]:
    """One colour range for the whole run, so a value never changes colour
    between sweeps -- the map would otherwise lie about what moved."""
    lo, hi = float("inf"), float("-inf")
    for _, p in list(frames) + [(0, plane)]:
        for v in p.values():
            lo, hi = min(lo, v), max(hi, v)
    if lo > hi:
        return (0.0, 1.0)
    if hi - lo < 1e-12:
        return (min(0.0, lo), max(1.0, hi))
    return (lo, hi)


# ---------------------------------------------------------------------------
# Sidebar: the map, and the rules behind a fold
# ---------------------------------------------------------------------------
cfg = sidebar.render_map()

# ---------------------------------------------------------------------------
# Your engine first (connect.py), the built-in evaluation otherwise
# ---------------------------------------------------------------------------
track = plane = frames = None
trace: Trace = Trace()
path: list = []
start_value: float | None = None
error: str | None = None
notes: list[str] = []

reader = LiveReader(cfg.run) if cfg.run is not None else None
connected = None if reader is not None and reader.exists() else connect.load_session(None)  # type: ignore[arg-type]

if reader is not None and reader.exists():
    session = build_live_session(reader, get_track("Easy"), max_frames=40)
    trace = read_trace(cfg.run)
    track = session.track
    scalar_field, _ = session.fields_at(None)
    plane = plane_of(scalar_field)
    frames = [
        (int(it.metrics.get("sweep", it.index)) if it.metrics else int(it.index),
         plane_of(session.fields_at(pos)[0]))
        for pos, it in enumerate(session.iterations)
    ]
    label, source = getattr(scalar_field, "label", None) or "V(s)", f"run {cfg.run.name}"
elif connected is not None:
    session: LabSession = connected[0] if isinstance(connected, tuple) else connected
    if isinstance(connected, tuple):
        trace = connected[1]
    track = session.track
    scalar_field, _ = session.fields_at(None)
    plane = plane_of(scalar_field)
    frames = [
        (int(it.metrics.get("sweep", it.index)) if it.metrics else int(it.index),
         plane_of(session.fields_at(pos)[0]))
        for pos, it in enumerate(session.iterations)
    ]
    label, source = getattr(scalar_field, "label", None) or "V(s)", "connect.py"
elif cfg.engine in ("policy", "value"):
    source = user_engine.algo_name(cfg.engine)
    # value iteration takes the max at every state, so its table is V*(s) from
    # the first sweep on -- the label says which equation the map is showing
    label = "V*(s)" if cfg.engine == "value" else "V(s)"
    try:
        res = run_user_code(cfg.text, cfg.sweeps, cfg.name, cfg.engine)
        track, plane, frames, trace = res.track, res.plane, res.frames, res.trace
        notes = res.notes
        path, start_value = res.path, res.start_value
    except (ValueError, RuntimeError) as exc:
        error = str(exc)
else:
    source = "demo"
    try:
        res = evaluate(cfg.text, cfg.gamma, cfg.sweeps, tuple(sorted(cfg.reward.items())), cfg.name)
        track, plane, frames, trace = res.track, res.plane, res.frames, res.trace
        label = "V(s)  [DEMO]"
    except ValueError as exc:
        label, error = "V(s)", str(exc)


# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------
head_left, head_right = st.columns([2.4, 1.35], vertical_alignment="bottom")
with head_left:
    c.title_block(
        "RL Visualization",
        "Bellman optimality equation · V(s) ← max_a [ r + γV(s′) ]"
        if cfg.engine == "value"
        else "Bellman expectation equation · V(s) ← Σ_a π(a|s) [ r + γV(s′) ]",
    )
with head_right:
    badges = []
    if error is None and track is not None:
        badges.append(c.badge(f"{track.width} × {track.height} · {track.n_drivable} cells", "mono"))
        # your engines take γ from your own Agent, not from the sidebar slider,
        # so the badge reports the one the trace was recorded with
        gamma_shown = cfg.gamma if source == "demo" else trace.gamma
        badges.append(c.badge(
            f"γ {gamma_shown} · {cfg.sweeps} sweeps" if gamma_shown is not None else f"{cfg.sweeps} sweeps",
            "mono"))
        if trace:
            bad = len(trace.failures())
            badges.append(c.badge(f"backup {len(trace.events)}", "mono"))
            badges.append(c.badge("검증 통과" if bad == 0 else f"검증 실패 {bad}", "mono"))
        badges.append(c.badge(source, "demo" if source == "demo" else "live"))
    c.meta_row(badges)

if notes:
    with st.expander(f"실행 점검 {len(notes)}건", expanded=False):
        for note in notes:
            st.markdown(f'<div class="rvl-text">· {note}</div>', unsafe_allow_html=True)

if error is not None:
    st.markdown(
        '<div class="rvl-empty"><div class="rvl-empty-title">맵을 읽을 수 없습니다.</div>'
        f'<div class="rvl-empty-body">{error}<br>'
        "<code>.</code> 빈 칸 · <code>#</code> 벽 · <code>S</code> 시작 · <code>G</code> 목표</div></div>",
        unsafe_allow_html=True,
    )
else:
    bellman_view.render(
        track,
        plane,
        trace,
        value_range=fixed_range(frames, plane),
        scalar_label=label,
        frames=frames,
        path=path,
        start_value=start_value,
    )
