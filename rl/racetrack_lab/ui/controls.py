"""Streamlit-side selectors. Playback lives inside the canvas component."""

from __future__ import annotations

import streamlit as st

from ..canvas.payload import MODES
from ..schema import LabSession, VehicleState
from ..tracks import TRACK_ORDER
from . import components as c


def ensure_state(n_steps: int, n_iterations: int) -> None:
    """Create every session key the UI relies on, and keep them in range."""
    defaults = {
        "track_name": "Easy",
        "mode": "Track",
        "step": 0,
        "iteration": max(n_iterations - 1, 0),
        "playing": False,
        "selected_cell": None,
        "canvas_nonce": None,
        "use_demo": True,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    st.session_state["step"] = max(0, min(int(st.session_state["step"]), max(n_steps - 1, 0)))
    st.session_state["iteration"] = max(
        0, min(int(st.session_state["iteration"]), max(n_iterations - 1, 0))
    )


def track_selector() -> str:
    choice = st.segmented_control(
        "Track",
        TRACK_ORDER,
        key="track_name",
        label_visibility="collapsed",
        width="stretch",
    )
    return choice or "Easy"


def mode_selector() -> str:
    choice = st.segmented_control(
        "Visualization mode",
        MODES,
        key="mode",
        label_visibility="collapsed",
        width="stretch",
    )
    return choice or "Track"


# ---------------------------------------------------------------------------
# Velocity slice
# ---------------------------------------------------------------------------
def _nearest(options: list[int], target: float) -> int:
    return min(options, key=lambda v: (abs(v - target), v))


def seed_slice(field, hint: tuple[float, float] | None = None) -> tuple[int, int] | None:
    """Resolve which ``(vx, vy)`` plane to show, before any widget is built.

    The first view defaults to ``hint`` (normally the car's own velocity) so
    the overlay is immediately relevant; after that the user's pick sticks.
    """
    if field is None or not getattr(field, "velocity_keyed", False):
        return None
    slices = field.slices
    vxs, vys = field.vx_values, field.vy_values
    if not slices or not vxs or not vys:
        return None

    if st.session_state.get("slice_vx") not in vxs or st.session_state.get("slice_vy") not in vys:
        target = hint or (0, 0)
        st.session_state["slice_vx"] = _nearest(vxs, target[0])
        st.session_state["slice_vy"] = _nearest(vys, target[1])

    key = (int(st.session_state["slice_vx"]), int(st.session_state["slice_vy"]))
    if key not in slices:
        key = min(slices, key=lambda s: abs(s[0] - key[0]) + abs(s[1] - key[1]))
        st.session_state["slice_vx"], st.session_state["slice_vy"] = key
    return key


def velocity_slice_picker(vx_values: list[int], vy_values: list[int]) -> tuple[int, int] | None:
    if not vx_values or not vy_values:
        return None
    if st.session_state.get("slice_vx") not in vx_values:
        st.session_state["slice_vx"] = _nearest(vx_values, 0)
    if st.session_state.get("slice_vy") not in vy_values:
        st.session_state["slice_vy"] = _nearest(vy_values, 0)
    vx = st.select_slider("vx", options=vx_values, key="slice_vx", format_func=lambda v: f"{v:+d}")
    vy = st.select_slider("vy", options=vy_values, key="slice_vy", format_func=lambda v: f"{v:+d}")
    return (int(vx), int(vy))


def _set_slice(vx: int, vy: int) -> None:
    st.session_state["slice_vx"] = vx
    st.session_state["slice_vy"] = vy


def follow_vehicle_button(state: VehicleState | None, field) -> None:
    """Jump the slice selector to the velocity the car has at the current step."""
    if state is None or field is None or not getattr(field, "velocity_keyed", False):
        return
    target = (int(round(state.vx)), int(round(state.vy)))
    available = target in field.slices
    st.button(
        f"Match the car  ({target[0]:+d}, {target[1]:+d})",
        icon=":material/my_location:",
        key="follow_vehicle",
        on_click=_set_slice,
        args=target,
        disabled=not available,
        width="stretch",
        help=(
            "Show the slice for the car's current velocity"
            if available
            else "The field has no slice for the car's current velocity"
        ),
    )


def data_badge(source: str) -> str:
    if source == "live":
        return c.badge("Live", "live dot")
    if source == "demo":
        return c.badge("Demo data", "demo dot")
    if source == "connected":
        return c.badge("Connected", "live dot")
    return c.badge("No data", "none dot")


def current_state(session: LabSession, step: int) -> VehicleState | None:
    if session.trajectory and len(session.trajectory):
        return session.trajectory.at(step)
    return None
