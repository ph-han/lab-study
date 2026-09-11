"""The three circuits, described as corridors and rasterized to a grid.

A track is authored as a list of inclusive rectangles in track coordinates
(``x`` right, ``y`` up, origin bottom-left) so the geometry stays readable and
editable. Difficulty comes from the shape -- corner count, corridor width,
pinch points -- not from the map size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .schema import FINISH, START, TRACK, WALL, TrackMap

Rect = tuple[int, int, int, int]  # (x0, y0, x1, y1), inclusive


@dataclass(frozen=True)
class TrackSpec:
    """Authoring format for a circuit."""

    name: str
    difficulty: str
    size: tuple[int, int]  # (width, height)
    corridors: tuple[Rect, ...]
    start: Rect
    finish: Rect
    obstacles: tuple[Rect, ...] = field(default_factory=tuple)
    note: str = ""
    shape: str = ""  # one-line description of the layout


def _paint(grid: np.ndarray, rect: Rect, code: int) -> None:
    x0, y0, x1, y1 = rect
    h, w = grid.shape
    x0, x1 = max(0, min(x0, x1)), min(w - 1, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(h - 1, max(y0, y1))
    grid[y0 : y1 + 1, x0 : x1 + 1] = code


def build_track(spec: TrackSpec) -> TrackMap:
    """Rasterize a :class:`TrackSpec` into a :class:`TrackMap`."""
    width, height = spec.size
    grid = np.full((height, width), WALL, dtype=int)
    for rect in spec.corridors:
        _paint(grid, rect, TRACK)
    for rect in spec.obstacles:
        _paint(grid, rect, WALL)
    _paint(grid, spec.start, START)
    _paint(grid, spec.finish, FINISH)
    return TrackMap(
        grid=grid,
        name=spec.name,
        difficulty=spec.difficulty,
        note=spec.note,
    )


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------
EASY = TrackSpec(
    name="Easy",
    difficulty="Easy",
    size=(40, 20),
    # one long straight along the bottom, one left-hand corner, one finish chute
    corridors=(
        (2, 2, 33, 7),    # bottom straight
        (27, 2, 33, 19),  # finish chute
    ),
    start=(2, 2, 2, 7),
    finish=(27, 18, 33, 19),
    shape="Long straight, one corner",
    note=(
        "One decision to read: where the car starts braking and turning for the "
        "single corner."
    ),
)

MEDIUM = TrackSpec(
    name="Medium",
    difficulty="Medium",
    size=(48, 30),
    # a climbing S: right, up, right, up, right
    corridors=(
        (2, 2, 16, 7),
        (11, 2, 16, 15),
        (11, 10, 30, 15),
        (25, 10, 30, 23),
        (25, 18, 44, 23),
    ),
    start=(2, 2, 2, 7),
    finish=(43, 18, 44, 23),
    shape="Two-corner S, four direction changes",
    note=(
        "Corners come in pairs, so the line taken through the first one decides "
        "how the second one can be driven."
    ),
)

HARD = TrackSpec(
    name="Hard",
    difficulty="Hard",
    size=(44, 32),
    # a serpentine with a 3-wide connector, two hairpins and two pinch points
    corridors=(
        (2, 2, 29, 6),     # bottom straight
        (27, 2, 29, 14),   # narrow connector, 3 cells wide
        (8, 10, 29, 14),   # back straight, driven right to left
        (8, 10, 12, 22),   # first hairpin exit
        (8, 18, 36, 22),   # third straight
        (32, 18, 36, 30),  # second hairpin exit
        (14, 26, 36, 30),  # finish straight, driven right to left
    ),
    obstacles=(
        (18, 2, 20, 4),    # chicane on the bottom straight
        (22, 20, 24, 22),  # pinch before the last hairpin
    ),
    start=(2, 2, 2, 6),
    finish=(14, 26, 15, 30),
    shape="Serpentine, 3-wide neck, two pinch points",
    note=(
        "Two hairpins and a three-cell neck: arriving fast is not the same as "
        "arriving on a line that still fits through."
    ),
)

TRACK_SPECS: dict[str, TrackSpec] = {
    "Easy": EASY,
    "Medium": MEDIUM,
    "Hard": HARD,
}
TRACK_ORDER = ("Easy", "Medium", "Hard")


def get_track(name: str) -> TrackMap:
    """Build the named track (``Easy`` / ``Medium`` / ``Hard``)."""
    spec = TRACK_SPECS.get(name, EASY)
    return build_track(spec)


def to_ascii(track: TrackMap) -> str:
    """Top-down ASCII dump, for eyeballing geometry in a terminal."""
    symbols = {WALL: ".", TRACK: " ", START: "S", FINISH: "F"}
    rows = []
    for y in range(track.height - 1, -1, -1):
        rows.append("".join(symbols[int(c)] for c in track.grid[y]))
    return "\n".join(rows)
