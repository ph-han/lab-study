"""Small supporting charts: one series each, no second axis, hover always on."""

from __future__ import annotations

import plotly.graph_objects as go

from .. import palette
from ..schema import Trajectory


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:.3f})"


def _shell(height: int, y_title: str, x_title: str, log_y: bool = False) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=height,
        margin=dict(l=44, r=14, t=10, b=34),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hovermode="x unified",
        font=dict(family=palette.FONT_UI, size=11, color=palette.INK_MUTED),
        hoverlabel=dict(
            bgcolor="rgba(43,42,51,0.94)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(family=palette.FONT_MONO, size=11, color="#FFFFFF"),
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=10, color=palette.INK_FAINT)),
            showgrid=False,
            zeroline=False,
            linecolor=palette.BORDER,
            tickfont=dict(family=palette.FONT_MONO, size=10, color=palette.INK_MUTED),
            showspikes=True,
            spikemode="across",
            spikethickness=1,
            spikecolor="rgba(43,42,51,0.25)",
            spikedash="solid",
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=10, color=palette.INK_FAINT)),
            gridcolor="rgba(43,42,51,0.07)",
            zeroline=False,
            tickfont=dict(family=palette.FONT_MONO, size=10, color=palette.INK_MUTED),
            type="log" if log_y else "linear",
            nticks=4,
        ),
    )
    return fig


def _marker(fig: go.Figure, x: float, y: float, color: str, label: str) -> None:
    fig.add_trace(
        go.Scatter(
            x=[x],
            y=[y],
            mode="markers",
            marker=dict(size=11, color="#FFFFFF", line=dict(width=3, color=color)),
            hovertemplate=label + "<extra></extra>",
            name="now",
        )
    )
    fig.add_vline(x=x, line=dict(color="rgba(43,42,51,0.18)", width=1))


def speed_profile(trajectory: Trajectory, current: int, height: int = 160) -> go.Figure:
    """Speed against step, with the current step called out."""
    fig = _shell(height, "speed", "step")
    steps = [s.step if s.step is not None else i for i, s in enumerate(trajectory.states)]
    speeds = [s.speed for s in trajectory.states]
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=speeds,
            mode="lines",
            line=dict(color=palette.TRAIL, width=2.5, shape="spline", smoothing=0.5),
            fill="tozeroy",
            fillcolor=_rgba(palette.TRAIL, 0.14),
            hovertemplate="speed %{y:.2f}<extra></extra>",
            name="speed",
        )
    )
    if speeds:
        i = max(0, min(current, len(speeds) - 1))
        _marker(fig, steps[i], speeds[i], palette.CAR, "step %{x} · speed %{y:.2f}")
    return fig


def metric_history(
    values: list[float],
    indices: list[int],
    current: int,
    label: str,
    height: int = 150,
) -> go.Figure:
    """One metric across iterations, with the viewed iteration called out."""
    positive = [v for v in values if v > 0]
    log_y = bool(positive) and len(positive) == len(values) and max(values) / min(positive) > 100
    fig = _shell(height, f"{label} (log)" if log_y else label, "iteration", log_y=log_y)
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=values,
            mode="lines",
            line=dict(color=palette.INK_SECONDARY, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(43,42,51,0.06)",
            hovertemplate=label + " %{y:.4g}<extra></extra>",
            name=label,
        )
    )
    if values:
        i = max(0, min(current, len(values) - 1))
        _marker(fig, indices[i], values[i], palette.CAR, "iteration %{x} · %{y:.4g}")
    return fig


def value_distribution(values: list[float], label: str, height: int = 160, bins: int = 28) -> go.Figure:
    """How the scalar field is distributed over the visible slice."""
    fig = _shell(height, "cells", label)
    fig.update_layout(hovermode="closest")
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=bins,
            marker=dict(color=_rgba("#8F4BC7", 0.75), line=dict(width=0)),
            hovertemplate=label + " %{x}<br>cells %{y}<extra></extra>",
            name=label,
        )
    )
    fig.update_traces(marker_cornerradius=4)
    return fig
