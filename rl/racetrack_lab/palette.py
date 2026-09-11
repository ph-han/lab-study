"""Design tokens for Racetrack Visual Lab.

Warm paper, a toy-track circuit and one little car. Every color has exactly one
job: grass is off-track, lavender-grey is asphalt, white is the start line,
checkers are the finish, coral is the car, blue is where it has been.
"""

from __future__ import annotations

# ----------------------------------------------------------------------------
# Page & ink
# ----------------------------------------------------------------------------
PAGE = "#F7F3EA"          # warm cream
CARD = "#FFFFFF"
CARD_SOFT = "#FBF9F4"
BORDER = "#ECE6DA"
SHADOW = "0 8px 24px rgba(74, 52, 28, 0.08)"

INK = "#2B2A33"
INK_SECONDARY = "#5D5B6E"
INK_MUTED = "#8E8B9E"
INK_FAINT = "#B7B4C4"

# ----------------------------------------------------------------------------
# Circuit
# ----------------------------------------------------------------------------
GRASS = "#B9E2A0"         # off-track / not drivable
GRASS_DOT = "#A6D48C"
ASPHALT = "#DFDBE9"       # drivable
ASPHALT_DIM = "#D3CEE0"   # asphalt under a scalar overlay (unpainted cells)
KERB = "#FFFFFF"          # rounded white edge stroke
GRID = "rgba(43, 42, 51, 0.06)"

START = "#2EC4B6"         # teal outline around the white start line
START_FILL = "#FFFFFF"
FINISH_DARK = "#2B2A33"   # checkerboard
FINISH_LIGHT = "#FFFFFF"

CAR = "#FF5D5D"           # body
CAR_DARK = "#C0392B"      # outline / shading
CAR_GLASS = "#CFEFFF"
CAR_WHEEL = "#2B2A33"
CAR_LIGHT = "#FFE66D"

TRAIL = "#3A86FF"         # where the car has been
TRAIL_CASING = "#FFFFFF"
VELOCITY = "#2B2A33"      # velocity arrow ink
ARROW = "#2B2A33"         # vector field ink
SELECTION = "#FF9F1C"     # inspected cell ring
CRASH = "#E63946"

# ----------------------------------------------------------------------------
# Scalar-field ramps (light -> dark; low values recede into the light asphalt)
# ----------------------------------------------------------------------------
SEQUENTIAL = [
    "#FFF3C4", "#FFE08A", "#FFC466", "#FFA26B", "#FF7E8A",
    "#E96AA6", "#C05CC4", "#8F4BC7", "#5B2E9E", "#341B6B",
]

# Blue <-> warm grey <-> orange, for fields that straddle zero.
DIVERGING = [
    "#0B4FA8", "#2F76CF", "#6AA3E6", "#A9CCF2", "#D7E6F7",
    "#EFEDF3",
    "#FBDCC0", "#F9B98A", "#F58F55", "#E5652C", "#B84517",
]

# Semantic swatches for legends (kept in one place for the UI)
LEGEND = {
    "car": CAR,
    "trail": TRAIL,
    "start": START,
    "finish": FINISH_DARK,
    "grass": GRASS,
    "arrow": ARROW,
    "selection": SELECTION,
}

# ----------------------------------------------------------------------------
# Typography
# ----------------------------------------------------------------------------
FONT_UI = '"Nunito", "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, sans-serif'
FONT_MONO = (
    '"JetBrains Mono", "Cascadia Mono", ui-monospace, SFMono-Regular, Consolas, monospace'
)
GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2?family=Nunito:wght@500;700;800"
    "&family=JetBrains+Mono:wght@500;700&display=swap"
)
