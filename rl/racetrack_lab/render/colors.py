"""Color-scale decisions shared by the canvas and the Streamlit-side captions."""

from __future__ import annotations

from .. import palette


def pick_ramp(vmin: float, vmax: float, mode: str = "auto") -> tuple[list[str], bool, float, float]:
    """Return ``(ramp, is_diverging, vmin, vmax)`` for a scalar field.

    ``auto`` picks the diverging ramp only when the data genuinely straddles
    zero (both arms at least 15% of the range); a diverging ramp is centred on
    zero so equal magnitudes get equal color strength.
    """
    diverging = False
    if mode == "diverging":
        diverging = True
    elif mode == "auto":
        span = vmax - vmin
        if span > 0 and vmin < 0 < vmax and min(-vmin, vmax) / span >= 0.15:
            diverging = True

    if diverging:
        bound = max(abs(vmin), abs(vmax)) or 1.0
        return palette.DIVERGING, True, -bound, bound
    return palette.SEQUENTIAL, False, vmin, vmax


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    h = color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def sample_ramp(ramp: list[str], t: float) -> str:
    """Piecewise-linear interpolation of a hex ramp at ``t`` in [0, 1]."""
    if not ramp:
        return "#000000"
    if len(ramp) == 1:
        return ramp[0]
    t = max(0.0, min(1.0, t))
    pos = t * (len(ramp) - 1)
    i = min(int(pos), len(ramp) - 2)
    f = pos - i
    a, b = _hex_to_rgb(ramp[i]), _hex_to_rgb(ramp[i + 1])
    r, g, bl = (round(a[k] + (b[k] - a[k]) * f) for k in range(3))
    return f"#{r:02X}{g:02X}{bl:02X}"


def css_gradient(ramp: list[str]) -> str:
    return "linear-gradient(90deg, " + ", ".join(ramp) + ")"
