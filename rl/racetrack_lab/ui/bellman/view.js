/* The Bellman view: one map, one calculation panel, one transport.
 *
 * Everything here replays what the engine recorded -- it never computes a
 * value. PAYLOAD.events are backup events from backups.jsonl; PAYLOAD.values
 * is the value field the run published. Selecting, stepping and autoplay all
 * happen in the browser so Python only reruns when the run itself changes.
 */
(function () {
  try { main(); } catch (err) {
    var box = document.getElementById("panelBody") || document.body;
    box.innerHTML = '<div class="empty"><div class="t">화면 스크립트가 멈췄습니다.</div>' +
      '<div class="s">이 메시지를 그대로 알려주세요.</div><div class="code">' +
      String(err && err.stack ? err.stack : err) + "</div></div>";
    throw err;
  }

function main() {
  var P = window.BELLMAN_PAYLOAD;
  var S = {
    sel: null,        // [x, y] inspected cell
    act: -1,          // highlighted action row, -1 = none
    won: false,       // a max backup has already replayed its winning action
    ev: P.events.length ? (P.live ? P.events.length - 1 : 0) : -1,
    playing: false,
    timer: null,
    speed: 620        // ms per step, set by the speed control
  };
  var EL = { rect: {}, text: {} };

  var DIROF = { "→": [1, 0], "←": [-1, 0], "↑": [0, 1], "↓": [0, -1] };
  var RAMP = [[255, 253, 246], [255, 242, 204], [255, 226, 162], [255, 202, 139], [255, 174, 134], [255, 142, 134]];

  function $(id) { return document.getElementById(id); }
  function num(x) {
    if (x === null || x === undefined) return "—";
    var s = x.toFixed(5).replace(/0+$/, "").replace(/\.$/, "");
    return (s === "" || s === "-0") ? "0" : s;
  }
  function key(x, y) { return x + "," + y; }

  function colorFor(v) {
    if (v === null || v === undefined) return "#FFFFFF";
    var lo = P.range[0], hi = P.range[1];
    var t = hi > lo ? (v - lo) / (hi - lo) : 0;
    t = Math.max(0, Math.min(1, t));
    var f = t * (RAMP.length - 1), i = Math.floor(f), k = f - i;
    var a = RAMP[i], b = RAMP[Math.min(RAMP.length - 1, i + 1)];
    return "rgb(" + Math.round(a[0] + (b[0] - a[0]) * k) + "," +
      Math.round(a[1] + (b[1] - a[1]) * k) + "," +
      Math.round(a[2] + (b[2] - a[2]) * k) + ")";
  }

  // ---- geometry ------------------------------------------------------------
  var CS = P.cell, GAP = P.gap, PITCH = CS + GAP;
  function cx(x) { return x * PITCH + CS / 2; }
  function cy(y) { return (P.rows - 1 - y) * PITCH + CS / 2; }

  // ---- fitting the numbers inside the cells --------------------------------
  // One decimal count and one font size for the whole map, chosen so that the
  // longest value any frame will show still fits. Values change while playing,
  // so every frame is measured, not just the one on screen.
  function fmtAt(v, d) {
    if (v === null || v === undefined) return "";
    var s = v.toFixed(d).replace(/0+$/, "").replace(/\.$/, "");
    return (s === "" || s === "-0") ? "0" : s;
  }
  function planText() {
    var tables = [P.values];
    for (var i = 0; i < P.frames.length; i++) tables.push(P.frames[i].values);
    var avail = CS - 10, cap = Math.min(26, CS * 0.26);
    for (var d = 5; d >= 1; d--) {
      var longest = 1;
      for (var t = 0; t < tables.length; t++) {
        for (var k in tables[t]) {
          if (P.kind[k] === "wall" || P.kind[k] === "goal") continue;
          longest = Math.max(longest, fmtAt(tables[t][k], d).length);
        }
      }
      var fs = Math.min(cap, avail / (longest * 0.62));
      if (fs >= 11 || d === 1) return { d: d, fs: Math.max(8, fs) };
    }
    return { d: 1, fs: cap };
  }
  var TXT = planText();
  function mapText(v) { return fmtAt(v, TXT.d); }

  // ---- map -----------------------------------------------------------------
  function buildMap() {
    var w = P.cols * PITCH - GAP, h = P.rows * PITCH - GAP;
    var svg = ['<svg class="map" width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + ' ' + h + '">'];
    for (var y = 0; y < P.rows; y++) {
      for (var x = 0; x < P.cols; x++) {
        var k = key(x, y), kind = P.kind[k] || "wall";
        var v = P.values[k];
        var fill = kind === "wall" ? "#D3DAE4" : (kind === "goal" ? "#E4F5F0" : colorFor(v));
        var cls = "cell" + (kind === "wall" ? "" : " hit");
        svg.push('<rect class="' + cls + '" data-k="' + k + '" x="' + (x * PITCH) + '" y="' + ((P.rows - 1 - y) * PITCH) +
          '" width="' + CS + '" height="' + CS + '" rx="' + Math.min(14, CS * 0.16) +
          '" fill="' + fill + '" stroke="rgba(20,28,43,.08)"></rect>');
        if (kind === "goal") {
          svg.push('<text x="' + cx(x) + '" y="' + (cy(y) + CS * 0.12) + '" text-anchor="middle" font-size="' +
            Math.min(26, CS * 0.42) + '" fill="#0E7C66" font-weight="800">G</text>');
        } else if (kind !== "wall" && CS >= 34) {
          svg.push('<text data-t="' + k + '" x="' + cx(x) + '" y="' + (cy(y) + TXT.fs * 0.36) +
            '" text-anchor="middle" font-size="' + TXT.fs.toFixed(1) + '">' +
            mapText(v) + "</text>");
        }
        if (kind !== "wall" && CS >= 76) {
          svg.push('<text class="co" x="' + (x * PITCH + 9) + '" y="' + ((P.rows - y) * PITCH - GAP - 8) +
            '" font-size="10">' + x + "," + y + "</text>");
        }
      }
    }
    svg.push('<g id="trail"></g><g id="carg"></g><g id="pol"></g><g id="ov"></g>');
    svg.push("</svg>");
    $("stage").innerHTML = svg.join("");
    Array.prototype.forEach.call($("stage").querySelectorAll("rect[data-k]"), function (el) {
      EL.rect[el.getAttribute("data-k")] = el;
    });
    Array.prototype.forEach.call($("stage").querySelectorAll("text[data-t]"), function (el) {
      EL.text[el.getAttribute("data-t")] = el;
    });
    $("stage").addEventListener("click", function (e) {
      var t = e.target.getAttribute && e.target.getAttribute("data-k");
      if (!t) return;
      pause();
      S.sel = t.split(",").map(Number);
      S.act = -1;
      S.won = false;
      var at = eventFor(S.sel);
      if (at >= 0) S.ev = at;
      draw();
    });
  }

  // Which value table the map shows: a backup of sweep k reads the table left
  // by sweep k-1, so that is what belongs on the map while it runs.
  function tableFor(sweep) {
    if (sweep === null || sweep === undefined || !P.frames.length) return P.values;
    var best = null;
    for (var i = 0; i < P.frames.length; i++) {
      if (P.frames[i].sweep <= sweep && (best === null || P.frames[i].sweep > best.sweep)) best = P.frames[i];
    }
    return best ? best.values : P.values;
  }

  function applyValues(vals) {
    for (var k in EL.rect) {
      var kind = P.kind[k];
      if (kind === "wall" || kind === "goal") continue;
      var v = vals[k];
      EL.rect[k].setAttribute("fill", colorFor(v));
      if (EL.text[k]) EL.text[k].textContent = mapText(v);
    }
  }

  function eventFor(sel) {
    if (!sel) return -1;
    for (var i = P.events.length - 1; i >= 0; i--) {
      if (P.events[i].state[0] === sel[0] && P.events[i].state[1] === sel[1]) return i;
    }
    return -1;
  }

  // ---- overlays ------------------------------------------------------------
  function drawOverlay(ev) {
    var ov = $("ov"), pol = $("pol");
    if (!ov) return;
    ov.innerHTML = "";
    pol.innerHTML = "";
    $("rbadge").className = "rbadge hid";
    if (!S.sel) return;

    var sx = cx(S.sel[0]), sy = cy(S.sel[1]);
    var pad = 3;
    ov.innerHTML += '<rect class="selring" x="' + (S.sel[0] * PITCH - pad) + '" y="' + ((P.rows - 1 - S.sel[1]) * PITCH - pad) +
      '" width="' + (CS + pad * 2) + '" height="' + (CS + pad * 2) + '" rx="' + Math.min(16, CS * 0.18) + '"></rect>';

    if (!ev || S.act < 0 || !ev.action_terms[S.act]) return;
    var t = ev.action_terms[S.act], o = t.outcomes[0];
    var nx = o.next_state[0], ny = o.next_state[1];

    if (!(nx === S.sel[0] && ny === S.sel[1])) {
      ov.innerHTML += '<rect class="refring" x="' + (nx * PITCH - pad) + '" y="' + ((P.rows - 1 - ny) * PITCH - pad) +
        '" width="' + (CS + pad * 2) + '" height="' + (CS + pad * 2) + '" rx="' + Math.min(16, CS * 0.18) + '"></rect>';
      var ex = cx(nx), ey = cy(ny);
      var dx = ex - sx, dy = ey - sy, len = Math.hypot(dx, dy) || 1;
      var ux = dx / len, uy = dy / len, back = Math.min(CS * 0.42, len * 0.4);
      var ax = sx + ux * back, ay = sy + uy * back, bx = ex - ux * back, by = ey - uy * back;
      var shaft = Math.hypot(bx - ax, by - ay);
      ov.innerHTML += '<path class="arrow" d="M ' + ax + ' ' + ay + ' L ' + bx + ' ' + by +
        '" stroke-dasharray="' + shaft + '" style="--len:' + shaft + 'px;animation:draw .3s ease both"></path>';
      var hx = bx + ux * 12, hy = by + uy * 12, px = -uy, py = ux;
      ov.innerHTML += '<polygon class="ahead" points="' + hx + "," + hy + " " +
        (bx + px * 7) + "," + (by + py * 7) + " " + (bx - px * 7) + "," + (by - py * 7) + '"></polygon>';
      placeBadge((ax + bx) / 2, (ay + by) / 2, o.reward);
    } else {
      // Blocked: show which way it pushed, and the wall it hit.
      var d = DIROF[t.glyph];
      if (d) {
        var ux2 = d[0], uy2 = -d[1];                 // screen y grows downward
        var edge = CS / 2, out = edge - 4;
        var ax2 = sx + ux2 * (edge * 0.30), ay2 = sy + uy2 * (edge * 0.30);
        var bx2 = sx + ux2 * out, by2 = sy + uy2 * out;
        var len2 = Math.hypot(bx2 - ax2, by2 - ay2);
        ov.innerHTML += '<path class="bump" d="M ' + ax2 + " " + ay2 + " L " + bx2 + " " + by2 +
          '" stroke-dasharray="' + len2 + '" style="--len:' + len2 + 'px;animation:draw .3s ease both"></path>';
        var px2 = -uy2, py2 = ux2, w = edge * 0.42;
        ov.innerHTML += '<path class="wallbar" d="M ' + (bx2 + px2 * w) + " " + (by2 + py2 * w) +
          " L " + (bx2 - px2 * w) + " " + (by2 - py2 * w) + '"></path>';
        placeBadge(sx + ux2 * edge * 0.62, sy + uy2 * edge * 0.62, o.reward);
      } else {
        var r = Math.min(CS * 0.34, 30);
        ov.innerHTML += '<circle class="refring" cx="' + sx + '" cy="' + sy + '" r="' + r + '"></circle>';
        placeBadge(sx, sy - r - 14, o.reward);
      }
    }
  }

  function placeBadge(x, y, r) {
    var b = $("rbadge"), box = $("stage").getBoundingClientRect();
    var svg = $("stage").querySelector("svg").getBoundingClientRect();
    b.className = "rbadge" + (r > 0 ? "" : " zero");
    b.textContent = "r " + (r > 0 ? "+" : "") + num(r);
    b.style.left = (svg.left - box.left + x) + "px";
    b.style.top = (svg.top - box.top + y) + "px";
  }

  // ---- panel ---------------------------------------------------------------
  function drawPanel(ev) {
    var host = $("panelBody");
    if (!P.events.length) {
      host.innerHTML = '<div class="empty"><div class="t">' + P.emptyTitle + '</div><div class="s">' +
        P.emptyBody + '</div><div class="code">' + P.snippet + "</div></div>";
      return;
    }
    if (!ev) {
      host.innerHTML = '<div class="empty"><div class="t">이 실행에는 해당 상태의 값이 기록되지 않았습니다.</div>' +
        '<div class="s">칸을 다시 고르거나 재생으로 기록된 계산을 따라가 보세요. ' +
        '기록된 backup ' + P.events.length + "개.</div></div>";
      return;
    }
    var terms = ev.action_terms, agg = ev.aggregation;
    var best = -Infinity, scale = 1e-9;
    terms.forEach(function (t) {
      best = Math.max(best, t.action_value);
      scale = Math.max(scale, Math.abs(t.action_value), Math.abs(t.weighted_value));
    });
    scale = Math.max(scale, Math.abs(ev.value_after), Math.abs(ev.value_before));

    // A bar grown from the middle of the track: right for +, left for -.
    function bar(v, cls) {
      var w = Math.min(50, Math.abs(v) / scale * 50);
      var left = v >= 0 ? 50 : 50 - w;
      return '<i class="' + cls + (v < 0 ? " neg" : "") + '" style="left:' + left.toFixed(2) +
        "%;width:" + w.toFixed(2) + '%"></i>';
    }

    var html = '<div class="selrow"><div class="s">s = (' + ev.state[0] + ", " + ev.state[1] + ')</div>' +
      '<div class="t">γ ' + ev.gamma + " · " + P.aggLabel[agg] + (ev.target_sweep === null ? "" : " · sweep " + ev.target_sweep) + "</div></div>";

    terms.forEach(function (t, i) {
      var o = t.outcomes[0];
      var stay = o.blocked !== undefined
        ? o.blocked
        : (o.next_state[0] === ev.state[0] && o.next_state[1] === ev.state[1]);
      var where = stay
        ? "벽 충돌 · 제자리"
        : (o.terminal ? "종료 " : "") + "(" + o.next_state[0] + ", " + o.next_state[1] + ")";
      var win = agg === "max" && t.action_value >= best - 1e-10;
      html += '<div class="row ' + (i === S.act ? "on " : "") + (win ? "win" : "") + '" data-i="' + i + '">' +
        '<div class="r1"><div class="aico">' + (t.glyph || "·") + "</div>" +
        '<div class="nx">' + where + "</div>" +
        '<span class="rb ' + (o.reward > 0 ? "pos" : (o.reward < 0 ? "neg" : "")) + '">r ' +
        (o.reward > 0 ? "+" : "") + num(o.reward) + "</span>" +
        '<span class="vref">V ' + num(o.referenced_value) + "</span>" +
        '<span class="bn">' + t.action_value.toFixed(4) + "</span></div>" +
        '<div class="barrow"><div class="track">' + bar(t.action_value, "q") +
        '<i class="zero"></i></div>' +
        // a max backup has no policy to weight by: the winning action *is* the
        // contribution, so the column marks it instead of printing ×0%
        (agg === "max"
          ? '<span class="contrib">' + (win ? "<b>max</b>" : "") + "</span>"
          : '<span class="contrib">×' + Math.round(t.policy_probability * 100) + "% <b>" +
            (t.weighted_value >= 0 ? "+" : "") + t.weighted_value.toFixed(4) + "</b></span>") +
        "</div></div>";
    });

    var mk = 50 + Math.max(-50, Math.min(50, ev.value_before / scale * 50));
    var resbar = '<div class="track lg">' + bar(ev.value_after, "q") +
      '<i class="zero"></i><i class="mark" style="left:' + mk.toFixed(2) + '%"></i></div>';
    var c = ev.checks;
    var chk = c.matches_value_after
      ? '<span class="chk ok">재계산 일치</span>'
      : '<span class="chk no">불일치 ' + c.error.toExponential(2) + "</span>";
    var d = ev.value_after - ev.value_before;
    html += '<div class="res ' + (agg === "max" ? "max" : "") + '">' + resbar +
      '<div class="out"><div class="big">' + num(ev.value_after) + "</div>" +
      '<span class="from">이전 ' + num(ev.value_before) + "</span>" +
      '<span class="from">' + (d >= 0 ? "+" : "") + d.toPrecision(3) + "</span>" + chk + "</div></div>";

    if (!c.matches_value_after) {
      html += '<div class="tiny" style="color:#B42318">기록한 항으로 다시 계산한 값과 <b>value_after</b> 가 다릅니다. ' +
        "합산식이나 읽는 값 표를 확인하세요.</div>";
    }
    host.innerHTML = html;
    Array.prototype.forEach.call(host.querySelectorAll(".row"), function (el) {
      el.addEventListener("click", function () {
        pause();
        var i = +el.getAttribute("data-i");
        S.act = (S.act === i) ? -1 : i;
        draw();
      });
    });
  }


  // ---- the greedy path from the start cell ---------------------------------
  // The value map does not move when the start moves: V(s) is defined for every
  // cell on its own, and the start never enters the Bellman equation. The route
  // does move, so it is drawn here and driven continuously -- this is the part
  // of the screen that answers to where S is.
  var PATH = { pts: [], segs: [], total: 0, t0: 0, car: null, line: null,
               ready: false, running: false, asked: false };

  function carShape(u) {
    var w = u * 1.40, h = u * 0.84, tire = "#43302B";
    return '<g id="car" class="car">' +
      '<ellipse cx="0" cy="' + (h * 0.30).toFixed(1) + '" rx="' + (w * 0.56).toFixed(1) +
        '" ry="' + (h * 0.46).toFixed(1) + '" fill="rgba(20,28,43,.13)"></ellipse>' +
      '<rect x="' + (-w * 0.32).toFixed(1) + '" y="' + (-h * 0.64).toFixed(1) + '" width="' + (w * 0.28).toFixed(1) +
        '" height="' + (h * 1.28).toFixed(1) + '" rx="' + (h * 0.18).toFixed(1) + '" fill="' + tire + '"></rect>' +
      '<rect x="' + (w * 0.06).toFixed(1) + '" y="' + (-h * 0.64).toFixed(1) + '" width="' + (w * 0.28).toFixed(1) +
        '" height="' + (h * 1.28).toFixed(1) + '" rx="' + (h * 0.18).toFixed(1) + '" fill="' + tire + '"></rect>' +
      '<rect x="' + (-w / 2).toFixed(1) + '" y="' + (-h / 2).toFixed(1) + '" width="' + w.toFixed(1) +
        '" height="' + h.toFixed(1) + '" rx="' + (h * 0.44).toFixed(1) +
        '" fill="#FF7A45" stroke="#9A3412" stroke-width="' + Math.max(1, u * 0.09).toFixed(1) + '"></rect>' +
      '<rect x="' + (-w * 0.08).toFixed(1) + '" y="' + (-h * 0.29).toFixed(1) + '" width="' + (w * 0.30).toFixed(1) +
        '" height="' + (h * 0.58).toFixed(1) + '" rx="' + (h * 0.26).toFixed(1) + '" fill="#FFF6E8"></rect>' +
      '<circle cx="' + (w * 0.42).toFixed(1) + '" cy="0" r="' + (h * 0.12).toFixed(1) + '" fill="#FFE9A8"></circle>' +
      "</g>";
  }

  function buildPath() {
    var g = $("trail"), cg = $("carg"), note = $("startNote");
    if (!g || !cg) return;
    g.innerHTML = ""; cg.innerHTML = "";
    var raw = P.path || [];
    if (note) {
      note.textContent = (P.startValue === null || P.startValue === undefined)
        ? ""
        : "출발 V " + num(P.startValue) + " · " + Math.max(0, raw.length - 1) + "걸음";
    }
    if (raw.length < 2) return;

    PATH.pts = raw.map(function (q) { return [cx(q[0]), cy(q[1])]; });
    PATH.segs = []; PATH.total = 0;
    var d = "M " + PATH.pts[0][0].toFixed(1) + " " + PATH.pts[0][1].toFixed(1);
    for (var i = 1; i < PATH.pts.length; i++) {
      d += " L " + PATH.pts[i][0].toFixed(1) + " " + PATH.pts[i][1].toFixed(1);
      var L = Math.hypot(PATH.pts[i][0] - PATH.pts[i - 1][0], PATH.pts[i][1] - PATH.pts[i - 1][1]);
      PATH.segs.push(L); PATH.total += L;
    }
    var sw = Math.max(3, CS * 0.075).toFixed(1), r = Math.max(3.5, CS * 0.065).toFixed(1);
    g.innerHTML =
      '<path class="trailbg" d="' + d + '" style="stroke-width:' + sw + 'px"></path>' +
      '<path class="trailfg" id="trailfg" d="' + d + '" style="stroke-width:' + sw + 'px" stroke-dasharray="' +
        PATH.total.toFixed(1) + '" stroke-dashoffset="' + PATH.total.toFixed(1) + '"></path>' +
      '<circle class="startdot" cx="' + PATH.pts[0][0].toFixed(1) + '" cy="' + PATH.pts[0][1].toFixed(1) +
        '" r="' + r + '"></circle>';
    cg.innerHTML = carShape(Math.max(11, CS * 0.25));
    PATH.line = $("trailfg");
    PATH.car = $("car");
    PATH.ready = true;
    showRoute(false);          // drawn, but waiting to be asked for
  }

  function showRoute(on) {
    var g = $("trail"), cg = $("carg");
    if (!g || !cg) return;
    g.setAttribute("display", on ? "inline" : "none");
    cg.setAttribute("display", on ? "inline" : "none");
  }

  function pointAt(s) {
    var acc = 0;
    for (var i = 0; i < PATH.segs.length; i++) {
      if (s <= acc + PATH.segs[i] || i === PATH.segs.length - 1) {
        var f = PATH.segs[i] ? (s - acc) / PATH.segs[i] : 0;
        f = Math.max(0, Math.min(1, f));
        var a = PATH.pts[i], b = PATH.pts[i + 1];
        return {
          x: a[0] + (b[0] - a[0]) * f,
          y: a[1] + (b[1] - a[1]) * f,
          a: Math.atan2(b[1] - a[1], b[0] - a[0]) * 180 / Math.PI
        };
      }
      acc += PATH.segs[i];
    }
    return { x: PATH.pts[0][0], y: PATH.pts[0][1], a: 0 };
  }

  function startDrive() {
    if (!PATH.ready || PATH.running) return;
    PATH.running = true;
    PATH.t0 = 0;
    showRoute(true);
    var b = $("bReplay"); if (b) b.className = "mini";
    requestAnimationFrame(driveStep);
  }

  // One trip from the start cell to the goal, then the car parks there.
  var MS_PER_CELL = 460;
  function driveStep(ts) {
    if (!PATH.car || !PATH.total) { PATH.running = false; return; }
    if (!PATH.t0) PATH.t0 = ts;
    var dur = Math.max(700, (PATH.pts.length - 1) * MS_PER_CELL);
    var e = ts - PATH.t0;
    var t = Math.max(0, Math.min(1, e / dur));
    var k = t * t * (3 - 2 * t);                       // ease in, ease out
    var p = pointAt(k * PATH.total);
    PATH.car.setAttribute("transform",
      "translate(" + p.x.toFixed(2) + "," + p.y.toFixed(2) + ") rotate(" + p.a.toFixed(1) + ")");
    PATH.line.setAttribute("stroke-dashoffset", (PATH.total * (1 - k)).toFixed(1));
    if (t >= 1) { PATH.running = false; return; }      // parked at the goal
    requestAnimationFrame(driveStep);
  }

  // ---- "the calculation is done -- shall I drive it?" -----------------------
  function finished() {
    if (!PATH.ready || PATH.running || PATH.asked) return;
    PATH.asked = true;
    $("mCount").textContent = String(P.events.length);
    $("mMeta").innerHTML = (P.startValue === null || P.startValue === undefined)
      ? ""
      : '출발 칸 V <span class="m">' + num(P.startValue) + "</span> · <span class=\"m\">" +
        Math.max(0, (P.path || []).length - 1) + "</span>걸음";
    $("modal").className = "modal";
  }
  function closeModal() { $("modal").className = "modal hid"; }

  // ---- transport -----------------------------------------------------------
  function current() { return S.ev >= 0 && S.ev < P.events.length ? P.events[S.ev] : null; }
  function gotoEvent(i) {
    S.ev = Math.max(0, Math.min(P.events.length - 1, i));
    S.won = false;
    var ev = current();
    if (ev) { S.sel = ev.state.slice(); }
    draw();
  }

  // Which action a max backup settled on. Ties keep the first, the same way
  // Python's max does, so the arrow matches the number that was stored.
  function winIndex(ev) {
    var best = -Infinity, at = -1;
    ev.action_terms.forEach(function (t, i) {
      if (t.action_value > best) { best = t.action_value; at = i; }
    });
    return at;
  }
  function play() {
    if (!P.events.length) return;
    S.playing = true;
    if (S.timer) clearInterval(S.timer);
    S.timer = setInterval(function () {
      var ev = current();
      if (!ev) { pause(); return; }
      if (S.act < ev.action_terms.length - 1) { S.act += 1; draw(); return; }
      // a max backup gets one extra beat on the action it chose, so the jump
      // back to the winning arrow is the thing you see before the value lands
      if (ev.aggregation === "max" && !S.won) {
        S.won = true;
        S.act = winIndex(ev);
        draw();
        return;
      }
      if (S.ev >= P.events.length - 1) { pause(); draw(); finished(); return; }
      S.act = -1;
      gotoEvent(S.ev + 1);
    }, S.speed);
    draw();
  }
  function pause() {
    S.playing = false;
    if (S.timer) { clearInterval(S.timer); S.timer = null; }
    var b = $("bPlay"); if (b) b.textContent = "자동 재생";
  }

  // ---- scrubber ------------------------------------------------------------
  function buildScrub() {
    var n = P.events.length;
    var el = $("scrub");
    el.max = String(Math.max(0, n - 1));
    el.disabled = n < 2;
    if (n < 2) return;
    var ticks = "";
    for (var i = 1; i < n; i++) {          // a mark where each new sweep begins
      if (P.events[i].target_sweep !== P.events[i - 1].target_sweep) {
        ticks += '<i style="left:' + (i / (n - 1) * 100).toFixed(3) + '%"></i>';
      }
    }
    $("sticks").innerHTML = ticks;
    el.addEventListener("input", function () {
      pause();
      S.act = -1;
      gotoEvent(+el.value);
    });
  }

  function drawScrub(ev) {
    var n = P.events.length;
    var pos = n ? S.ev : 0;
    $("scrub").value = String(pos);
    $("sfill").style.width = (n > 1 ? pos / (n - 1) * 100 : 0).toFixed(2) + "%";
    $("slabelL").innerHTML = ev && ev.target_sweep !== null && ev.target_sweep !== undefined
      ? "sweep <span class='m'>" + ev.target_sweep + "</span> · s = (" + ev.state[0] + ", " + ev.state[1] + ")"
      : "&nbsp;";
    $("slabelR").innerHTML = n
      ? "<span class='m'>" + (pos + 1) + "</span> / " + n + (P.total > n ? " (최근 " + n + "개)" : "")
      : "기록된 계산 없음";
  }

  // ---- render --------------------------------------------------------------
  function draw() {
    var ev = current();
    if (S.sel && (!ev || ev.state[0] !== S.sel[0] || ev.state[1] !== S.sel[1])) {
      var at = eventFor(S.sel);
      ev = at >= 0 ? P.events[at] : null;
    }
    var sweep = ev && ev.target_sweep !== null && ev.target_sweep !== undefined ? ev.target_sweep - 1 : null;
    applyValues(tableFor(sweep));
    $("mapNote").textContent = sweep === null
      ? P.cols + " × " + P.rows
      : "읽는 표 V" + sweep + " · 쓰는 표 V" + (sweep + 1);
    drawPanel(ev);
    drawOverlay(ev);
    drawScrub(ev);
    $("bPlay").textContent = S.playing ? "일시정지" : "자동 재생";
    $("bPrev").disabled = S.ev <= 0;
    $("bNext").disabled = S.ev >= P.events.length - 1;
    $("bPlay").disabled = !P.events.length;
    $("liveDot").className = S.playing ? "live" : "live hid";
    if (ev) {
      $("ask").textContent = P.askWith;
      $("sub").textContent = ev.checks.matches_value_after
        ? "기록한 네 항으로 다시 계산한 값이 " + num(ev.value_after) + " 와 같습니다."
        : "기록한 항으로 다시 계산하면 " + num(ev.checks.recomputed) + " 입니다.";
    } else {
      $("ask").textContent = P.askEmpty;
      $("sub").textContent = P.subEmpty;
    }
  }

  buildMap();
  buildPath();
  buildScrub();
  if (P.events.length) { S.sel = P.events[S.ev].state.slice(); }
  $("bPrev").addEventListener("click", function () { pause(); S.act = -1; gotoEvent(S.ev - 1); });
  $("bNext").addEventListener("click", function () {
    pause(); S.act = -1; gotoEvent(S.ev + 1);
    if (S.ev >= P.events.length - 1) finished();
  });
  $("bPlay").addEventListener("click", function () { S.playing ? pause() : play(); });
  $("bFirst").addEventListener("click", function () { pause(); S.act = -1; gotoEvent(0); });
  Array.prototype.forEach.call(document.querySelectorAll(".seg button[data-ms]"), function (b) {
    b.addEventListener("click", function () {
      S.speed = +b.getAttribute("data-ms");
      Array.prototype.forEach.call(document.querySelectorAll(".seg button[data-ms]"), function (o) {
        o.className = o === b ? "on" : "";
      });
      if (S.playing) play();   // restart at the new cadence
    });
  });
  $("mYes").addEventListener("click", function () { closeModal(); startDrive(); });
  $("mNo").addEventListener("click", closeModal);
  $("modal").addEventListener("click", function (e) { if (e.target === $("modal")) closeModal(); });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closeModal();
    if (e.key === "Enter" && $("modal").className === "modal") { closeModal(); startDrive(); }
  });
  $("bReplay").addEventListener("click", function () { PATH.running = false; startDrive(); });
  draw();
}
})();
