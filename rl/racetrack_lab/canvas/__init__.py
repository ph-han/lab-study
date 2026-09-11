"""The racetrack canvas: a small bidirectional Streamlit component.

The frontend (``frontend/``) is plain HTML/CSS/JS -- no build step. It draws
the circuit as SVG, animates the car client-side so playback never reruns the
Python script, and reports clicks / transport changes back through
``setComponentValue``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

from .payload import MODES, CanvasFacts, CanvasOptions, build_payload

_FRONTEND = Path(__file__).parent / "frontend"


def _frontend_digest() -> str:
    """Short hash of the frontend sources.

    It becomes part of the component name, hence of the iframe URL, so a
    browser can never keep serving a stale ``canvas.js`` after an edit.
    """
    h = hashlib.blake2b(digest_size=5)
    for f in sorted(_FRONTEND.glob("*")):
        if f.is_file():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()


_component = components.declare_component(
    f"racetrack_canvas_{_frontend_digest()}", path=str(_FRONTEND)
)


def racetrack_canvas(payload: dict[str, Any], key: str = "racetrack_canvas") -> Any:
    """Mount the canvas and return the latest event it reported (or ``None``)."""
    return _component(payload=payload, key=key, default=None)


__all__ = ["MODES", "CanvasFacts", "CanvasOptions", "build_payload", "racetrack_canvas"]
