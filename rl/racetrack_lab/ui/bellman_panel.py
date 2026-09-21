"""The Bellman calculation panel: one cell's backup, opened up.

Reads what :mod:`racetrack_lab.trace` recorded and draws it. When a run has no
calculation events -- which is every run that only publishes fields -- it says
so and shows what to record instead of inventing the arithmetic.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from ..trace import Trace

_CSS = """
<style>
.bp-wrap{--bp-sel:#2563EB;--bp-ref:#7C3AED;--bp-rew:#B45309;--bp-ink:#2B2A33;--bp-ink2:#5D5B6E;--bp-ink3:#8E8B9E;
  --bp-line:#ECE6DA;--bp-soft:#F4F2EE;font-family:var(--rvl-ui,inherit)}
.bp-head{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-bottom:.55rem}
.bp-s{font-family:var(--rvl-mono,monospace);font-size:1.05rem;font-weight:700;color:var(--bp-ink)}
.bp-k{font-size:.7rem;font-weight:800;letter-spacing:.06em;color:var(--bp-ink3)}
.bp-row{border:1px solid var(--bp-line);border-radius:14px;padding:.5rem .65rem .6rem;margin-bottom:.4rem;background:#fff}
.bp-row.win{border-color:var(--bp-ref);background:#FAF7FF}
.bp-r1{display:flex;align-items:center;gap:.5rem}
.bp-a{width:26px;height:26px;border-radius:9px;background:var(--bp-soft);display:flex;align-items:center;
  justify-content:center;font-size:.95rem;font-weight:800;color:var(--bp-ink2);flex:0 0 auto}
.bp-nx{font-family:var(--rvl-mono,monospace);font-size:.74rem;font-weight:700;color:var(--bp-ink2);white-space:nowrap}
.bp-rw{font-family:var(--rvl-mono,monospace);font-size:.68rem;font-weight:700;padding:.1rem .45rem;border-radius:999px;
  background:var(--bp-soft);color:var(--bp-ink3)}
.bp-rw.pos{background:var(--bp-rew);color:#fff}
.bp-b{margin-left:auto;font-family:var(--rvl-mono,monospace);font-size:.95rem;font-weight:700;color:var(--bp-ink)}
.bp-track{margin-top:.45rem;height:8px;border-radius:99px;background:var(--bp-soft);position:relative;overflow:hidden}
.bp-track i{position:absolute;left:0;top:0;bottom:0;border-radius:99px;animation:bpGrow .5s cubic-bezier(.3,.9,.3,1)}
.bp-track i.q{background:#FFD59B}
.bp-track i.w{background:var(--bp-sel)}
@keyframes bpGrow{from{width:0 !important}}
.bp-res{border:1px solid #DCE6F6;background:#F8FAFF;border-radius:14px;padding:.6rem .7rem .7rem;margin-top:.55rem}
.bp-stack{height:13px;border-radius:99px;background:var(--bp-soft);display:flex;overflow:hidden}
.bp-stack i{display:block;height:100%;animation:bpGrow .55s cubic-bezier(.3,.9,.3,1)}
.bp-out{display:flex;align-items:baseline;gap:.6rem;margin-top:.5rem;flex-wrap:wrap}
.bp-big{font-family:var(--rvl-mono,monospace);font-size:1.75rem;font-weight:700;letter-spacing:-.03em;color:var(--bp-ink)}
.bp-from{font-family:var(--rvl-mono,monospace);font-size:.72rem;font-weight:700;color:var(--bp-ink3)}
.bp-d{font-family:var(--rvl-mono,monospace);font-size:.72rem;font-weight:700;border-radius:999px;padding:.12rem .5rem;
  background:#D9F5EE;color:#0E7C66}
.bp-chk{font-size:.7rem;font-weight:800;border-radius:999px;padding:.15rem .55rem;display:inline-block}
.bp-chk.ok{background:#D9F5EE;color:#0E7C66}
.bp-chk.no{background:#FDECEC;color:#B42318}
.bp-empty{border:2px dashed var(--bp-line);border-radius:16px;background:#FBFAF7;padding:.8rem .9rem}
.bp-empty b{color:var(--bp-ink)}
.bp-note{font-size:.76rem;font-weight:600;color:var(--bp-ink2);line-height:1.55}
.bp-code{font-family:var(--rvl-mono,monospace);font-size:.7rem;line-height:1.6;color:var(--bp-ink2);
  background:#fff;border:1px solid var(--bp-line);border-radius:10px;padding:.55rem .65rem;margin-top:.5rem;white-space:pre}
</style>
"""

_SNIPPET = """terms = []
for action, prob in action_probs.items():
    next_state = env.next_state(state, action)
    reward = env.reward(state, action, next_state)
    terms.append({"action": action, "prob": prob, "next_state": next_state,
                  "reward": reward, "next_value": V[next_state]})
    new_V += prob * (reward + gamma * V[next_state])
run.backup(state=state, value_before=old_value, terms=terms,
           value_after=new_V, sweep=k)"""

_TINTS = ["#93B4F5", "#B9CDF8", "#2563EB", "#5D8FEE", "#7FA6F2", "#A7C0F7"]


def _num(x: float) -> str:
    s = f"{x:.5f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _css_once() -> None:
    if not st.session_state.get("_bp_css"):
        st.markdown(_CSS, unsafe_allow_html=True)
        st.session_state["_bp_css"] = True


def _empty(title: str, body: str, snippet: bool = False) -> None:
    code = f'<div class="bp-code">{_SNIPPET}</div>' if snippet else ""
    st.markdown(
        f'<div class="bp-wrap"><div class="bp-empty"><div class="bp-note"><b>{title}</b><br>{body}</div>'
        f"{code}</div></div>",
        unsafe_allow_html=True,
    )


def render(trace: Trace, cell: Any, *, sweep: int | None = None) -> None:
    """Draw the calculation for ``cell``, or an honest empty state."""
    _css_once()

    if not trace:
        _empty(
            "값과 정책만 기록되어 있어 계산 과정은 확인할 수 없습니다.",
            "네 행동의 확률·전이·보상이 함께 기록되면 이 자리에 계산표가 열립니다. "
            "평가 루프 안에서 <code>run.backup(...)</code> 을 한 번 호출하면 됩니다.",
            snippet=True,
        )
        return

    if cell is None:
        _empty("칸을 하나 고르세요.", f"이 실행에는 backup 이벤트 {len(trace.events)}개가 기록되어 있습니다.")
        return

    ev = trace.at(tuple(cell), sweep=sweep)
    if ev is None:
        _empty(
            "이 실행에는 해당 상태의 값이 기록되지 않았습니다.",
            f"선택한 칸 ({int(cell[0])}, {int(cell[1])}) 의 backup 이벤트가 없습니다. "
            "평가 대상에서 빠졌거나, 아직 이번 바퀴에 도달하지 않았습니다.",
        )
        return

    terms = ev["action_terms"]
    checks = ev["checks"]
    agg = ev["aggregation"]
    best = max((t["action_value"] for t in terms), default=0.0)
    scale = max([abs(t["action_value"]) for t in terms] + [abs(ev["value_after"]), 1e-9])

    rows = []
    for t in terms:
        out = t["outcomes"][0]
        nx, ny = out["next_state"]
        where = "제자리" if t["self_loop"] else ("종료 " if out["terminal"] else "") + f"({nx}, {ny})"
        r = out["reward"]
        rw_cls = "pos" if r > 0 else ""
        qw = min(100.0, abs(t["action_value"]) / scale * 100.0)
        ww = min(100.0, abs(t["weighted_value"]) / scale * 100.0)
        win = agg == "max" and t["action_value"] >= best - 1e-10
        rows.append(
            f'<div class="bp-row {"win" if win else ""}">'
            f'<div class="bp-r1">'
            f'<div class="bp-a">{t["glyph"] or "·"}</div>'
            f'<div class="bp-nx">{where}</div>'
            f'<span class="bp-rw {rw_cls}">r {_num(r)}</span>'
            f'<span class="bp-nx" style="color:#7C3AED">V {_num(out["referenced_value"])}</span>'
            f'<span class="bp-b">{t["action_value"]:.4f}</span>'
            f"</div>"
            f'<div class="bp-track"><i class="q" style="width:{qw:.1f}%"></i>'
            f'<i class="w" style="width:{ww:.1f}%"></i></div>'
            f"</div>"
        )

    stack = "".join(
        f'<i style="width:{min(100.0, abs(t["weighted_value"]) / scale * 100.0):.1f}%;'
        f'background:{_TINTS[i % len(_TINTS)]}"></i>'
        for i, t in enumerate(terms)
    )
    delta = ev["value_after"] - ev["value_before"]
    ok = checks["matches_value_after"]
    chk = (
        f'<span class="bp-chk ok">재계산 일치</span>'
        if ok
        else f'<span class="bp-chk no">불일치 {checks["error"]:+.3e}</span>'
    )
    agg_label = {"policy_weighted_sum": "확률로 가중평균", "max": "최댓값", "single": "단일 행동"}[agg]
    sweep_txt = "" if ev.get("target_sweep") is None else f' · sweep {ev["target_sweep"]}'

    st.markdown(
        f'<div class="bp-wrap">'
        f'<div class="bp-head"><div class="bp-s">s = ({ev["state"][0]}, {ev["state"][1]})</div>'
        f'<div class="bp-k">γ {ev["gamma"]} · {agg_label}{sweep_txt}</div></div>'
        f'{"".join(rows)}'
        f'<div class="bp-res"><div class="bp-stack">{stack}</div>'
        f'<div class="bp-out"><div class="bp-big">{_num(ev["value_after"])}</div>'
        f'<span class="bp-from">이전 {_num(ev["value_before"])}</span>'
        f'<span class="bp-d">{delta:+.4g}</span>{chk}</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def summary_badges(trace: Trace) -> list[str]:
    """Short strings for the header: how much calculation this run recorded."""
    if not trace:
        return []
    bad = len(trace.failures())
    out = [f"backup {len(trace.events)}"]
    if trace.sweeps:
        out.append(f"sweep {trace.sweeps[0]}–{trace.sweeps[-1]}")
    out.append("검증 통과" if bad == 0 else f"검증 실패 {bad}")
    return out
