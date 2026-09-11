"""Small markup pieces the panels are assembled from."""

from __future__ import annotations

import html
from contextlib import contextmanager
from typing import Iterable, Iterator, Sequence

import streamlit as st

Metric = tuple[str, str] | tuple[str, str, str | None]


def _esc(text: object) -> str:
    return html.escape(str(text))


def badge(text: str, kind: str = "") -> str:
    classes = " ".join(c for c in ("rvl-badge", kind) if c)
    return f'<span class="{classes}">{_esc(text)}</span>'


def title_block(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="rvl-head-title">{_esc(title)}</div>'
        f'<div class="rvl-head-sub">{_esc(subtitle)}</div>',
        unsafe_allow_html=True,
    )


def meta_row(pieces: Iterable[str], align: str = "flex-end") -> None:
    st.markdown(
        f'<div class="rvl-head-meta" style="justify-content:{align}">'
        f'{"".join(pieces)}</div>',
        unsafe_allow_html=True,
    )


def rule(top: str = "0.4rem", bottom: str = "0.85rem") -> None:
    st.markdown(
        f'<div style="border-bottom:1px solid var(--rvl-border);'
        f'margin:{top} 0 {bottom}"></div>',
        unsafe_allow_html=True,
    )


@contextmanager
def panel(title: str, note: str = "") -> Iterator[None]:
    """A titled card that can hold widgets.

    Streamlit closes the HTML of every markdown block, so a card wrapping
    widgets has to be a real container rather than a stray opening ``<div>``.
    """
    note_html = f'<span class="rvl-card-note">{_esc(note)}</span>' if note else ""
    with st.container(border=True):
        st.markdown(
            f'<div class="rvl-card-head rvl-panel-head">'
            f'<span class="rvl-card-title">{_esc(title)}</span>{note_html}</div>',
            unsafe_allow_html=True,
        )
        yield


def metrics_html(items: Sequence[Metric], columns: int = 2, size: str = "") -> str:
    grid_class = "rvl-metrics" + (" one" if columns == 1 else "")
    cells = []
    for item in items:
        label, value = item[0], item[1]
        unit = item[2] if len(item) > 2 else None
        unit_html = f'<span class="rvl-metric-unit">{_esc(unit)}</span>' if unit else ""
        value_class = " ".join(c for c in ("rvl-metric-value", size) if c)
        cells.append(
            f'<div><div class="rvl-metric-label">{_esc(label)}</div>'
            f'<div class="{value_class}">{_esc(value)}{unit_html}</div></div>'
        )
    return f'<div class="{grid_class}">{"".join(cells)}</div>'


def card(title: str, items: Sequence[Metric], note: str = "", columns: int = 2) -> None:
    note_html = f'<span class="rvl-card-note">{_esc(note)}</span>' if note else ""
    st.markdown(
        f'<div class="rvl-card"><div class="rvl-card-head">'
        f'<span class="rvl-card-title">{_esc(title)}</span>{note_html}</div>'
        f"{metrics_html(items, columns=columns)}</div>",
        unsafe_allow_html=True,
    )


def legend(items: Iterable[tuple[str, str, str]]) -> None:
    """``(label, color, shape)`` with shape in ``block`` / ``line`` / ``ring``."""
    parts = []
    for label, color, shape in items:
        cls = "rvl-swatch" + ("" if shape == "block" else f" {shape}")
        parts.append(
            f'<span class="rvl-legend-item">'
            f'<span class="{cls}" style="background:{color}"></span>{_esc(label)}</span>'
        )
    st.markdown(f'<div class="rvl-legend">{"".join(parts)}</div>', unsafe_allow_html=True)


def empty_state(title: str, body_html: str) -> None:
    st.markdown(
        f'<div class="rvl-empty"><div class="rvl-empty-title">{_esc(title)}</div>'
        f'<div class="rvl-empty-body">{body_html}</div></div>',
        unsafe_allow_html=True,
    )


def slice_readout(slice_key: tuple[int, int] | None, prefix: str = "Viewing") -> str:
    if slice_key is None:
        return (
            f'<span class="rvl-slice">{_esc(prefix)} <b>all states</b> '
            "<span>(field has no velocity axis)</span></span>"
        )
    vx, vy = slice_key
    return (
        f'<span class="rvl-slice">{_esc(prefix)} '
        f"<b>vx = {vx:+d}</b> <b>vy = {vy:+d}</b></span>"
    )


def scale_bar(vmin: float, vmax: float, gradient: str) -> str:
    """``gradient`` is a CSS background value, e.g. from ``render.css_gradient``."""
    return (
        f'<span class="rvl-scalebar"><span>{vmin:+.3g}</span>'
        f'<span class="ramp" style="background:{gradient}"></span>'
        f"<span>{vmax:+.3g}</span></span>"
    )


def chips(pairs: Sequence[tuple[str, str]]) -> str:
    return " ".join(
        f'<span class="rvl-chip">{_esc(k)} {_esc(v)}</span>' for k, v in pairs
    )


