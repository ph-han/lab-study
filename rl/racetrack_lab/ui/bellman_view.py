"""The redesigned main screen: the map and the Bellman calculation, together.

One iframe holds both, so selecting a cell, stepping through recorded backups
and autoplay never rerun the Python script -- Streamlit only redraws when the
run itself changes. Everything shown comes from the run: the value field it
published and the backup events it recorded. When a run recorded no
calculation, the panel says so instead of inventing one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import streamlit.components.v1 as components

from ..schema import FINISH, START, WALL, TrackMap
from ..trace import Trace

_HERE = Path(__file__).parent / "bellman"

SNIPPET = """terms = []
for action, prob in action_probs.items():
    next_state = env.next_state(state, action)
    reward = env.reward(state, action, next_state)
    terms.append({"action": action, "prob": prob, "next_state": next_state,
                  "reward": reward, "next_value": V[next_state]})
    new_V += prob * (reward + gamma * V[next_state])

run.backup(state=state, value_before=V[state], terms=terms,
           value_after=new_V, sweep=k)"""

_AGG_LABEL = {
    "policy_weighted_sum": "확률로 가중평균",
    "max": "최댓값",
    "single": "단일 행동",
}

MAX_EVENTS = 600  # newest events kept in the payload; the rest stay on disk


def _kinds(track: TrackMap) -> dict[str, str]:
    out: dict[str, str] = {}
    for y in range(track.height):
        for x in range(track.width):
            c = track.grid[y][x]
            out[f"{x},{y}"] = (
                "wall" if c == WALL else "goal" if c == FINISH else "start" if c == START else "track"
            )
    return out


def _cell_size(cols: int, rows: int, width: int, height: int) -> tuple[int, int]:
    gap = 10 if cols <= 8 else 3
    by_w = (width - gap * (cols - 1)) / max(1, cols)
    by_h = (height - gap * (rows - 1)) / max(1, rows)
    return max(10, min(190, int(min(by_w, by_h)))), gap


def _as_values(plane: Mapping[tuple[int, int], float] | None) -> dict[str, float]:
    return {f"{int(k[0])},{int(k[1])}": float(v) for k, v in (plane or {}).items()}


def build_payload(
    track: TrackMap,
    plane: Mapping[tuple[int, int], float] | None,
    trace: Trace,
    *,
    value_range: tuple[float, float] | None = None,
    scalar_label: str = "V(s)",
    frames: list[tuple[int, Mapping[tuple[int, int], float]]] | None = None,
    live: bool = False,
    path: list[tuple[int, int]] | None = None,
    start_value: float | None = None,
    map_width: int = 780,
    map_height: int = 520,
) -> dict[str, Any]:
    values = _as_values(plane)
    if value_range is None:
        vals = list(values.values()) or [0.0, 1.0]
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-12:
            lo, hi = min(0.0, lo), max(1.0, hi)
        value_range = (lo, hi)

    kinds = _kinds(track)
    shown = [e for e in trace.events
             if kinds.get(str(e["state"][0]) + "," + str(e["state"][1])) != "wall"]
    events = shown[-MAX_EVENTS:]
    cell, gap = _cell_size(track.width, track.height, map_width, map_height)
    return {
        "cols": track.width,
        "rows": track.height,
        "cell": cell,
        "gap": gap,
        "kind": kinds,
        "values": values,
        # value tables per sweep, so stepping through events shows the table the
        # backup actually read instead of the finished one
        "frames": [{"sweep": int(k), "values": _as_values(p)} for k, p in (frames or [])],
        "range": [float(value_range[0]), float(value_range[1])],
        "label": scalar_label,
        "events": events,
        "total": len(shown),
        "aggLabel": _AGG_LABEL,
        "askWith": "State value evaluation",
        "askEmpty": "계산 기록이 아직 없습니다",
        "subEmpty": "값 지도는 보이지만, 그 숫자가 어떻게 나왔는지는 기록되지 않았습니다.",
        "emptyTitle": "값과 정책만 기록되어 있어 계산 과정은 확인할 수 없습니다.",
        "emptyBody": "평가 루프 안에서 <code>run.backup(...)</code> 을 한 번 부르면 이 자리에 네 행동의 "
                     "다음 상태 · 보상 · 참조한 값 · 합산이 열립니다.",
        "live": bool(live),
        # the greedy rollout from the start cell; the only thing on this screen
        # that moves when the start moves
        "path": [[int(x), int(y)] for x, y in (path or [])],
        "startValue": None if start_value is None else float(start_value),
        "snippet": SNIPPET,
    }


def _shell(payload: dict[str, Any], height: int) -> str:
    css = (_HERE / "view.css").read_text(encoding="utf-8")
    js = (_HERE / "view.js").read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    ramp = "linear-gradient(90deg,#FFFDF6,#FFF2CC,#FFE2A2,#FFCA8B,#FFAE86,#FF8E86)"
    lo, hi = payload["range"]
    seen = len(payload["events"])
    total = payload["total"]
    more = "" if seen == total else f" · 최근 {seen}개만 표시"
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Nunito:wght@600;700;800&family=Gothic+A1:wght@500;700;800&family=JetBrains+Mono:wght@500;700&display=swap">
<style>{css}</style></head>
<body>
<div style="padding:2px 2px 8px">
  <div style="margin-bottom:10px">
    <h1 class="ask" id="ask">&nbsp;</h1>
    <div class="sub" id="sub">&nbsp;</div>
  </div>

  <div class="wrap">
    <section class="card mapcard">
      <div class="chead">
        <div class="ctitle">지도 · {payload["label"]}</div>
        <span class="cnote" id="mapNote">{payload["cols"]} × {payload["rows"]}</span>
        <span class="cnote" id="startNote"></span>
        <button class="mini hid" id="bReplay">경로 다시 보기</button>
      </div>
      <div class="stagewrap">
        <div class="stage" id="stage"></div>
        <div class="rbadge hid" id="rbadge"></div>
      </div>
      <div class="legend">
        <span><span class="ramp" style="display:inline-block;background:{ramp};vertical-align:-1px"></span>
          <span class="m">&nbsp;{lo:.4g}</span> – <span class="m">{hi:.4g}</span> 고정 범위</span>
        <span><span class="sw" style="border-color:#2563EB"></span>선택한 칸</span>
        <span><span class="sw" style="border-color:#7C3AED;border-style:dashed"></span>계산에 쓴 다음 값</span>
        <span><span class="sw" style="border-color:#B45309;background:#B45309"></span>즉시 보상</span>
        <span><span class="sw" style="border-color:#EA580C;background:#FDBA74"></span>출발 → 목표 (그리디 경로)</span>
        <span><span class="sw" style="border-color:#C6CFDA;background:#D3DAE4"></span>벽</span>
      </div>
    </section>

    <aside class="card panel">
      <div class="chead">
        <div class="ctitle">벨만 계산</div>
        <span class="cnote">backup {total}{more}</span>
      </div>
      <div id="panelBody"></div>
    </aside>
  </div>

  <div class="scrub">
    <div class="sbar"><div class="sfill" id="sfill"></div><div class="sticks" id="sticks"></div></div>
    <input type="range" id="scrub" min="0" max="0" step="1" value="0" disabled>
    <div class="slabel"><span id="slabelL">&nbsp;</span><span id="slabelR">&nbsp;</span></div>
  </div>

  <div class="foot">
    <div class="btns">
      <button class="btn" id="bFirst">처음</button>
      <button class="btn" id="bPrev">이전</button>
      <button class="btn primary" id="bPlay">자동 재생</button>
      <button class="btn" id="bNext">다음 계산</button>
      <span class="spd">속도
        <span class="seg">
          <button data-ms="1600">0.4×</button>
          <button data-ms="900">0.7×</button>
          <button data-ms="620" class="on">1×</button>
          <button data-ms="320">2×</button>
          <button data-ms="140">4×</button>
        </span>
      </span>
    </div>
    <div class="status">
      <span class="live hid" id="liveDot"></span>
      <span>값은 실행이 기록한 것 그대로입니다</span>
    </div>
  </div>
</div>
<div class="modal hid" id="modal">
  <div class="mbox">
    <div class="mkicker">계산 완료</div>
    <div class="mtitle">애니메이션을 시작할까요?</div>
    <div class="mbody">기록된 backup <b id="mCount">0</b>개를 끝까지 재생했습니다.
      출발 칸에서 그리디 정책을 따라 목표까지 가는 경로를 그려 드립니다.</div>
    <div class="mmeta m" id="mMeta"></div>
    <div class="mbtns">
      <button class="btn" id="mNo">나중에</button>
      <button class="btn primary" id="mYes">시작</button>
    </div>
  </div>
</div>
<script>window.BELLMAN_PAYLOAD = {data};</script>
<script>{js}</script>
</body></html>"""


def render(
    track: TrackMap,
    plane: Mapping[tuple[int, int], float] | None,
    trace: Trace,
    *,
    value_range: tuple[float, float] | None = None,
    scalar_label: str = "V(s)",
    frames: list[tuple[int, Mapping[tuple[int, int], float]]] | None = None,
    live: bool = False,
    path: list[tuple[int, int]] | None = None,
    start_value: float | None = None,
) -> None:
    """Draw the whole screen. Call inside a full-width Streamlit container."""
    payload = build_payload(
        track, plane, trace, value_range=value_range, scalar_label=scalar_label,
        frames=frames, live=live, path=path, start_value=start_value,
    )
    map_h = payload["rows"] * (payload["cell"] + payload["gap"]) - payload["gap"]
    # The panel can be taller than the map; give both room and let the frame
    # scroll rather than clip, so nothing can go missing without a scrollbar.
    height = max(640, map_h + 300)
    components.html(_shell(payload, height), height=height, scrolling=True)
