"""Streamlit-free helpers: supporting charts and color-scale decisions."""

from . import charts
from .colors import css_gradient, pick_ramp, sample_ramp

PLOTLY_CONFIG = {
    "displaylogo": False,
    "displayModeBar": False,
    "scrollZoom": False,
}

__all__ = ["charts", "css_gradient", "pick_ramp", "sample_ramp", "PLOTLY_CONFIG"]
