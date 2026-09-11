"""One stylesheet for the Streamlit shell: warm paper, white rounded cards."""

from __future__ import annotations

import streamlit as st

from .. import palette

_CSS = f"""
<style>
@import url("{palette.GOOGLE_FONTS}");

:root {{
  --rvl-page: {palette.PAGE};
  --rvl-card: {palette.CARD};
  --rvl-card-soft: {palette.CARD_SOFT};
  --rvl-border: {palette.BORDER};
  --rvl-ink: {palette.INK};
  --rvl-ink-2: {palette.INK_SECONDARY};
  --rvl-ink-3: {palette.INK_MUTED};
  --rvl-ink-4: {palette.INK_FAINT};
  --rvl-car: {palette.CAR};
  --rvl-trail: {palette.TRAIL};
  --rvl-start: {palette.START};
  --rvl-sel: {palette.SELECTION};
  --rvl-shadow: {palette.SHADOW};
  --rvl-ui: {palette.FONT_UI};
  --rvl-mono: {palette.FONT_MONO};
}}

/* ---- shell ------------------------------------------------------------- */
html, body, [data-testid="stAppViewContainer"], .stApp, .stApp p, .stApp button {{ font-family: var(--rvl-ui); }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"], .block-container {{
  padding-top: 3rem; padding-bottom: 3rem; max-width: 1560px;
}}
[data-testid="stAppDeployButton"], [data-testid="stStatusWidget"] {{ display: none; }}
[data-testid="stVerticalBlock"] {{ gap: 0.7rem; }}
[data-testid="stHorizontalBlock"] {{ gap: 1rem; }}
[data-testid="stSidebarContent"] {{ padding-top: 1.2rem; }}
[data-testid="stSidebar"] {{ background: var(--rvl-card-soft); border-right: 1px solid var(--rvl-border); }}

/* ---- header ------------------------------------------------------------ */
.rvl-head-title {{
  font-size: 1.55rem; font-weight: 800; letter-spacing: -0.01em; color: var(--rvl-ink); line-height: 1.1;
}}
.rvl-head-title .dot {{ color: var(--rvl-car); }}
.rvl-head-sub {{ font-size: 0.82rem; color: var(--rvl-ink-3); margin-top: 0.2rem; font-weight: 600; }}
.rvl-head-meta {{ display: flex; align-items: center; gap: 0.45rem; flex-wrap: wrap; margin-bottom: 0.45rem; }}

/* ---- badges ------------------------------------------------------------ */
.rvl-badge {{
  display: inline-flex; align-items: center; gap: 0.4rem;
  font-family: var(--rvl-ui); font-weight: 800; font-size: 0.68rem; letter-spacing: 0.06em;
  text-transform: uppercase; padding: 0.28rem 0.7rem; border-radius: 999px;
  color: var(--rvl-ink-2); background: var(--rvl-card); box-shadow: var(--rvl-shadow); white-space: nowrap;
}}
.rvl-badge.mono {{ font-family: var(--rvl-mono); font-weight: 700; text-transform: none; letter-spacing: 0; }}
.rvl-badge.demo {{ color: #9A6700; background: #FFF1C4; }}
.rvl-badge.live {{ color: #0E7C66; background: #D9F5EE; }}
.rvl-badge.none {{ color: var(--rvl-ink-3); background: #EFEBF3; }}
.rvl-badge.dot::before {{ content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; display: inline-block; }}

/* Streamlit sizes a markdown block expecting a trailing <p> margin; our custom
   HTML has none, so the wrapper ends 1rem short and the next element overlaps. */
[data-testid="stMarkdownContainer"] > div:last-child,
[data-testid="stMarkdownContainer"] > span:last-child {{ margin-bottom: 1rem; }}

/* ---- cards ------------------------------------------------------------- */
.rvl-card {{ background: var(--rvl-card); border-radius: 18px; padding: 0.95rem 1.1rem 1rem; box-shadow: var(--rvl-shadow); }}
.rvl-card + .rvl-card {{ margin-top: 0.7rem; }}
.rvl-card-head {{ display: flex; align-items: center; justify-content: space-between; gap: 0.6rem; margin-bottom: 0.65rem; }}
.rvl-card-title {{ font-size: 0.68rem; letter-spacing: 0.13em; text-transform: uppercase; color: var(--rvl-ink-3); font-weight: 800; }}
.rvl-card-note {{ font-size: 0.7rem; color: var(--rvl-ink-4); font-weight: 600; text-align: right; }}
.rvl-panel-head {{ margin-bottom: 0.55rem; }}
[data-testid="stVerticalBlockBorderWrapper"] {{
  background: var(--rvl-card); border: 0 !important; border-radius: 18px; box-shadow: var(--rvl-shadow);
}}

/* ---- metrics ----------------------------------------------------------- */
.rvl-metrics {{ display: grid; gap: 0.6rem 0.8rem; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr)); }}
.rvl-metrics.one {{ grid-template-columns: 1fr; }}
.rvl-metric-label {{
  font-size: 0.64rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--rvl-ink-3); font-weight: 800; margin-bottom: 0.1rem;
}}
.rvl-metric-value {{
  font-family: var(--rvl-mono); font-size: 1.22rem; font-weight: 700; line-height: 1.2; color: var(--rvl-ink); font-variant-numeric: tabular-nums;
}}
.rvl-metric-value.sm {{ font-size: 0.96rem; }}
.rvl-metric-unit {{ font-family: var(--rvl-mono); font-size: 0.7rem; color: var(--rvl-ink-4); margin-left: 0.3rem; }}

/* ---- prose / empty ----------------------------------------------------- */
.rvl-text {{ font-size: 0.8rem; color: var(--rvl-ink-2); line-height: 1.6; font-weight: 600; }}
.rvl-text b {{ color: var(--rvl-ink); }}
.rvl-text code, .rvl-empty-body code {{
  font-family: var(--rvl-mono); font-size: 0.72rem; color: var(--rvl-ink); background: #F1EEF7; padding: 0.08rem 0.4rem; border-radius: 6px;
}}
.rvl-empty {{ border: 2px dashed var(--rvl-border); border-radius: 18px; padding: 1.1rem 1.2rem; background: var(--rvl-card-soft); }}
.rvl-empty-title {{ font-size: 0.88rem; color: var(--rvl-ink); font-weight: 800; margin-bottom: 0.3rem; }}
.rvl-empty-body {{ font-size: 0.78rem; color: var(--rvl-ink-3); line-height: 1.6; font-weight: 600; }}

/* ---- slice / chips / scale ---------------------------------------------- */
.rvl-slice {{ display: inline-flex; align-items: center; gap: 0.5rem; font-family: var(--rvl-mono); font-size: 0.8rem; color: var(--rvl-ink-2); }}
.rvl-slice b {{ color: var(--rvl-ink); background: #F1EEF7; border-radius: 8px; padding: 0.1rem 0.45rem; }}
.rvl-chip {{ font-family: var(--rvl-mono); font-size: 0.7rem; color: var(--rvl-ink-2); border-radius: 999px; padding: 0.18rem 0.6rem; background: #F1EEF7; }}
.rvl-scalebar {{ display: flex; align-items: center; gap: 0.5rem; font-family: var(--rvl-mono); font-size: 0.7rem; color: var(--rvl-ink-3); }}
.rvl-scalebar .ramp {{ height: 10px; width: 150px; border-radius: 6px; box-shadow: inset 0 0 0 1px rgba(43,42,51,.1); }}

/* ---- streamlit widget retouches --------------------------------------- */
[data-testid="stSegmentedControl"] [role="radiogroup"], [data-testid="stButtonGroup"] [role="radiogroup"] {{
  background: var(--rvl-card); border-radius: 999px; padding: 4px; box-shadow: var(--rvl-shadow); gap: 2px;
}}
[data-testid="stSegmentedControl"] button, [data-testid="stButtonGroup"] button {{ border-radius: 999px !important; border: 0 !important; }}
[data-testid="stSegmentedControl"] button p, [data-testid="stButtonGroup"] button p {{ font-size: 0.8rem; font-weight: 800; }}
[data-testid="stSegmentedControl"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[aria-checked="true"] {{
  background: var(--rvl-ink) !important;
}}
[data-testid="stSegmentedControl"] button[aria-checked="true"] p, [data-testid="stButtonGroup"] button[aria-checked="true"] p {{ color: #fff !important; }}
button[kind="secondary"], button[kind="primary"] {{ border-radius: 999px; font-weight: 800; }}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
  font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; color: var(--rvl-ink-3); font-weight: 800;
}}
[data-testid="stCaptionContainer"] p {{ font-size: 0.72rem; color: var(--rvl-ink-4); font-weight: 600; }}
[data-testid="stSidebar"] hr {{ margin: 0.6rem 0; border-color: var(--rvl-border); }}
.rvl-side-title {{ font-size: 0.95rem; font-weight: 800; color: var(--rvl-ink); margin: 0.2rem 0 0.35rem; }}
[data-baseweb="select"] > div {{ border-radius: 999px !important; }}
[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; }}
iframe {{ border-radius: 22px; }}

/* ---- narrow screens ---------------------------------------------------- */
@media (max-width: 1000px) {{
  [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]
    > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"] {{ min-width: 100% !important; flex: 1 1 100% !important; }}
  [data-testid="stMainBlockContainer"], .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
  .rvl-head-title {{ font-size: 1.3rem; }}
}}
</style>
"""


def inject() -> None:
    """Install the stylesheet. Safe to call on every rerun."""
    st.markdown(_CSS, unsafe_allow_html=True)
