"""Small maps written as text, for the viewer's demo rooms.

A map is a block of characters, top row first -- the way you would draw it::

    . . G
    . # .
    . . .
    S . .

    .  drivable        #  wall
    S  start           G  goal (terminal)

Spaces are ignored, so ``..G`` and ``. . G`` are the same row. The first line
is the TOP of the room: internally ``grid[y][x]`` keeps y = 0 at the bottom, so
the last line becomes y = 0 and the coordinates you see in the viewer match the
picture (x to the right, y upward).
"""

from __future__ import annotations

import numpy as np

from .schema import FINISH, START, TRACK, WALL, TrackMap

CHARS = {".": TRACK, " ": TRACK, "#": WALL, "X": WALL, "S": START, "G": FINISH}


def parse(text: str, name: str = "Room", note: str = "") -> TrackMap:
    """Build a :class:`TrackMap` from a text map. Raises on a ragged or unknown map."""
    rows = [r.replace(" ", "") for r in text.strip().splitlines()]
    rows = [r for r in rows if r]
    if not rows:
        raise ValueError("the map is empty")
    width = len(rows[0])
    for i, r in enumerate(rows):
        if len(r) != width:
            raise ValueError(f"row {i + 1} has {len(r)} cells, the first row has {width}")
        for ch in r:
            if ch not in CHARS:
                raise ValueError(f"unknown character {ch!r}; use one of {''.join(sorted(CHARS))}")
    grid = np.array([[CHARS[ch] for ch in r] for r in reversed(rows)], dtype=int)  # bottom row first
    if not (grid == FINISH).any():
        raise ValueError("the map has no goal: mark one cell with G")
    return TrackMap(grid=grid, name=name, difficulty=f"{grid.shape[1]}x{grid.shape[0]}", note=note)


def cells(track: TrackMap, code: int) -> list[tuple[int, int]]:
    """``(x, y)`` of every cell with this code."""
    return [
        (x, y)
        for y in range(track.height)
        for x in range(track.width)
        if int(track.grid[y][x]) == code
    ]


def show(track: TrackMap) -> str:
    """The text map again, top row first -- handy for checking what was parsed."""
    back = {TRACK: ".", WALL: "#", START: "S", FINISH: "G"}
    return "\n".join(
        " ".join(back.get(int(track.grid[y][x]), "?") for x in range(track.width))
        for y in range(track.height - 1, -1, -1)
    )


# ---------------------------------------------------------------------------
# A few rooms to start from. Add your own; any text map works.
# ---------------------------------------------------------------------------
# Keys are rows x cols -- the shape you see when you read the text.
ROOMS: dict[str, str] = {
    "2r3c": """
        . . G
        S . .
    """,
    "3r4c_wall": """
        . . . G
        . . # .
        S . . .
    """,
    "5r5c_detour": """
        . . # . G
        . . # . .
        . . # . .
        . . # . .
        S . . . .
    """,
    "corridor": """
        S . . G
    """,
}
