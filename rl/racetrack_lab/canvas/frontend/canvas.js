/* Racetrack canvas -- draws the circuit as SVG and animates the car locally.
 *
 * Python sends {structure, data, state}.
 *   structure : track, mode, options, colors, labels  -> rebuild the SVG when it changes
 *   data      : trajectory + field frames, stamped with `rev` -> update in place when it changes
 *   state     : step, iteration, selected cell, follow / hold / live flags
 * Everything that moves runs on requestAnimationFrame inside this iframe, so
 * neither playback nor live streaming ever reruns Python for a frame. User
 * actions go back through setComponentValue as small events.
 */
(() => {
  "use strict";

  // ---------------------------------------------------------------------------
  // Streamlit component protocol (vanilla)
  // ---------------------------------------------------------------------------
  function send(type, extra) {
    window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type }, extra || {}), "*");
  }
  const Host = {
    ready: () => send("streamlit:componentReady", { apiVersion: 1 }),
    height: (h) => send("streamlit:setFrameHeight", { height: h }),
    value: (v) => send("streamlit:setComponentValue", { value: v, dataType: "json" }),
  };

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  const SPEED_MS = { 0.5: 1100, 1: 620, 2: 320, 4: 160 }; // ms per step
  const PAD = 1.5; // cells of breathing room around the grid

  const S = {
    structureKey: "",
    dataRev: "",
    scene: null,      // structure + data merged
    states: [],
    headings: [],
    t: 0,             // continuous step position
    playing: false,
    speed: 1,
    raf: 0,
    lastTs: 0,
    chase: null,      // {from, target, start, dur} while gliding to a live update
    iter: 0,          // position in the frame list
    iterPlaying: false,
    iterTimer: 0,
    selected: null,
    scrubbing: false,
    follow: false,
    hold: false,
    live: false,
    liveStatus: "",
    pollMs: 600,
    lastReported: { step: null, iter: null },
    W: 0,
    H: 0,
  };

  const $ = (id) => document.getElementById(id);
  const svg = $("svg");
  const NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, parent) {
    const node = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }
  function clear(node) { while (node && node.firstChild) node.removeChild(node.firstChild); }
  const fmt = (v) => (Number.isInteger(v) ? String(v) : (+v).toFixed(2).replace(/\.?0+$/, ""));
  const fmtPair = (a, b) => `(${fmt(a)}, ${fmt(b)})`;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const sx = (x) => x + 0.5;
  const sy = (y) => S.H - y - 0.5;
  const lastStep = () => Math.max(S.states.length - 1, 0);

  // ---------------------------------------------------------------------------
  // Colors
  // ---------------------------------------------------------------------------
  function hexToRgb(h) {
    h = h.replace("#", "");
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  function ramp(colors, t) {
    if (!colors || !colors.length) return "#000";
    if (colors.length === 1) return colors[0];
    t = clamp(t, 0, 1);
    const pos = t * (colors.length - 1);
    const i = Math.min(Math.floor(pos), colors.length - 2);
    const f = pos - i;
    const a = hexToRgb(colors[i]), b = hexToRgb(colors[i + 1]);
    const c = a.map((v, k) => Math.round(v + (b[k] - v) * f));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  // ---------------------------------------------------------------------------
  // Geometry helpers
  // ---------------------------------------------------------------------------
  function drivable(grid, x, y) {
    if (y < 0 || y >= grid.length || x < 0 || x >= grid[0].length) return false;
    return grid[y][x] !== 0;
  }
  function maskBoundary(test) {
    const parts = [];
    for (let y = 0; y < S.H; y++) {
      for (let x = 0; x < S.W; x++) {
        if (!test(x, y)) continue;
        const L = x, R = x + 1, T = S.H - y - 1, B = S.H - y;
        if (!test(x - 1, y)) parts.push(`M${L} ${T}V${B}`);
        if (!test(x + 1, y)) parts.push(`M${R} ${T}V${B}`);
        if (!test(x, y - 1)) parts.push(`M${L} ${B}H${R}`);
        if (!test(x, y + 1)) parts.push(`M${L} ${T}H${R}`);
      }
    }
    return parts.join("");
  }
  function regionCenter(cells) {
    let mx = 0, my = 0;
    cells.forEach(([x, y]) => { mx += x; my += y; });
    return [mx / cells.length, my / cells.length];
  }
  function labelAnchor(cells, isDriv) {
    const xs = cells.map((c) => c[0]), ys = cells.map((c) => c[1]);
    const [cx, cy] = regionCenter(cells);
    const minX = Math.min(...xs), minY = Math.min(...ys);
    const maxX = Math.max(...xs), maxY = Math.max(...ys);
    const midX = xs[Math.floor(xs.length / 2)], midY = ys[Math.floor(ys.length / 2)];
    let out;
    if (maxY - minY >= maxX - minX) out = isDriv(minX - 1, midY) ? [2.3, 0] : [-2.3, 0];
    else out = isDriv(midX, minY - 1) ? [0, 2.0] : [0, -2.0];
    const cand = [cx + out[0], cy + out[1]];
    const m = 1.0;
    const inside = cand[0] >= -m && cand[0] <= S.W - 1 + m && cand[1] >= -m && cand[1] <= S.H - 1 + m;
    return inside ? cand : [cx - out[0], cy - out[1]];
  }

  // ---------------------------------------------------------------------------
  // Scene build (structure)
  // ---------------------------------------------------------------------------
  const L = {};
  let cellIndex = {};
  let isDrivable = () => false;

  function frameSet() {
    const sc = S.scene;
    if (sc.mode === "Value" && sc.scalar) return sc.scalar;
    if (sc.mode === "Action" && sc.vector) return sc.vector;
    return null;
  }

  function buildScene() {
    const sc = S.scene, C = sc.colors, grid = sc.track.grid;
    S.W = sc.track.width; S.H = sc.track.height;
    isDrivable = (x, y) => drivable(grid, x, y);
    const isDriv = isDrivable;

    clear(svg);
    svg.setAttribute("viewBox", `${-PAD} ${-PAD} ${S.W + 2 * PAD} ${S.H + 2 * PAD}`);

    const defs = el("defs", null, svg);
    const pat = el("pattern", { id: "dots", width: 2, height: 2, patternUnits: "userSpaceOnUse", patternTransform: "rotate(30)" }, defs);
    el("circle", { cx: 0.5, cy: 0.5, r: 0.13, fill: C.grassDot }, pat);
    el("circle", { cx: 1.5, cy: 1.5, r: 0.13, fill: C.grassDot }, pat);
    const mk = el("marker", { id: "vel-head", viewBox: "0 0 10 10", refX: 8.5, refY: 5, markerWidth: 4, markerHeight: 4, orient: "auto-start-reverse", markerUnits: "strokeWidth" }, defs);
    el("path", { d: "M0 0.5 L10 5 L0 9.5 z", fill: C.velocity }, mk);
    const shadow = el("filter", { id: "soft", x: "-30%", y: "-30%", width: "160%", height: "160%" }, defs);
    el("feDropShadow", { dx: 0.05, dy: 0.12, stdDeviation: 0.12, "flood-color": "#2B2A33", "flood-opacity": 0.28 }, shadow);

    ["grass", "asphalt", "overlay", "grid", "kerb", "regions", "numbers", "vectors", "trail", "select", "car", "labels", "hit"]
      .forEach((n) => { L[n] = el("g", { id: "L-" + n }, svg); });

    el("rect", { x: -PAD - 1, y: -PAD - 1, width: S.W + 2 * PAD + 2, height: S.H + 2 * PAD + 2, fill: C.grass }, L.grass);
    el("rect", { x: -PAD - 1, y: -PAD - 1, width: S.W + 2 * PAD + 2, height: S.H + 2 * PAD + 2, fill: "url(#dots)", opacity: 0.55 }, L.grass);

    const painted = sc.mode === "Value" && sc.hasScalar;
    const fillA = painted ? C.asphaltDim : C.asphalt;
    el("path", { d: maskBoundary(isDriv), fill: "none", stroke: "#2B2A33", "stroke-width": 0.46, "stroke-linejoin": "round", "stroke-linecap": "round", opacity: 0.10, transform: "translate(0.05 0.14)" }, L.asphalt);
    for (let y = 0; y < S.H; y++) for (let x = 0; x < S.W; x++) {
      if (isDriv(x, y)) el("rect", { x, y: S.H - y - 1, width: 1, height: 1, fill: fillA }, L.asphalt);
    }
    el("path", { d: maskBoundary(isDriv), fill: "none", stroke: fillA, "stroke-width": 0.42, "stroke-linejoin": "round", "stroke-linecap": "round" }, L.asphalt);
    el("path", { d: maskBoundary(isDriv), fill: "none", stroke: C.kerb, "stroke-width": 0.16, "stroke-linejoin": "round", "stroke-linecap": "round" }, L.kerb);

    if (sc.options.grid) {
      const parts = [];
      for (let y = 0; y < S.H; y++) for (let x = 0; x < S.W; x++) {
        if (!isDriv(x, y)) continue;
        if (isDriv(x + 1, y)) parts.push(`M${x + 1} ${S.H - y - 1}V${S.H - y}`);
        if (isDriv(x, y + 1)) parts.push(`M${x} ${S.H - y - 1}H${x + 1}`);
      }
      el("path", { d: parts.join(""), stroke: C.grid, "stroke-width": 0.035, fill: "none" }, L.grid);
    }

    const startCells = [], finishCells = [];
    for (let y = 0; y < S.H; y++) for (let x = 0; x < S.W; x++) {
      if (grid[y][x] === 2) startCells.push([x, y]);
      if (grid[y][x] === 3) finishCells.push([x, y]);
    }
    startCells.forEach(([x, y]) => el("rect", { x, y: S.H - y - 1, width: 1, height: 1, fill: C.startFill, opacity: painted ? 0.55 : 1 }, L.regions));
    if (startCells.length) el("path", { d: maskBoundary((x, y) => grid[y] && grid[y][x] === 2), fill: "none", stroke: C.start, "stroke-width": 0.16, "stroke-linejoin": "round" }, L.regions);
    finishCells.forEach(([x, y]) => {
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
        const dark = (x * 2 + i + y * 2 + j) % 2 === 0;
        el("rect", { x: x + i / 2, y: S.H - y - 1 + j / 2, width: 0.5, height: 0.5, fill: dark ? C.finishDark : C.finishLight, opacity: painted ? 0.6 : 1 }, L.regions);
      }
    });
    if (finishCells.length) el("path", { d: maskBoundary((x, y) => grid[y] && grid[y][x] === 3), fill: "none", stroke: C.finishDark, "stroke-width": 0.12, "stroke-linejoin": "round" }, L.regions);

    if (sc.options.labels) {
      const pill = (cells, text, fill, ink) => {
        if (!cells.length) return;
        const [ax, ay] = labelAnchor(cells, isDriv);
        const g = el("g", { transform: `translate(${sx(ax)} ${sy(ay)})`, filter: "url(#soft)" }, L.labels);
        const w = text.length * 0.42 + 0.7;
        el("rect", { x: -w / 2, y: -0.42, width: w, height: 0.84, rx: 0.42, fill }, g);
        el("text", { x: 0, y: 0.2, "text-anchor": "middle", "font-size": 0.56, fill: ink, "letter-spacing": 0.04 }, g).textContent = text;
      };
      pill(startCells, "START", C.start, "#FFFFFF");
      pill(finishCells, "FINISH", C.finishDark, "#FFFFFF");
    }

    cellIndex = {};
    for (let y = 0; y < S.H; y++) for (let x = 0; x < S.W; x++) {
      if (!isDriv(x, y)) continue;
      el("rect", { x, y: S.H - y - 1, width: 1, height: 1, fill: "transparent", class: "cell hit", "data-x": x, "data-y": y }, L.hit);
    }

    placeHud(isDriv);
    applyData(true);
  }

  function placeHud(isDriv) {
    // Measure the readout in cells at the current zoom, then pick the corner
    // with the fewest track cells under it. If every corner is busy, fall
    // back to a low strip that only needs the bottom two rows.
    const hud = $("hud");
    const stageW = $("stage").clientWidth || 900;
    const cellPx = stageW / (S.W + 2 * PAD);
    const count = (x0, y0, bw, bh) => {
      let n = 0;
      for (let y = Math.max(0, y0); y < Math.min(S.H, y0 + bh); y++)
        for (let x = Math.max(0, x0); x < Math.min(S.W, x0 + bw); x++) if (isDriv(x, y)) n++;
      return n;
    };
    const cardW = Math.ceil(200 / cellPx) + 1, cardH = Math.ceil(150 / cellPx) + 1;
    const corners = [
      { name: "tr", n: count(S.W - cardW, S.H - cardH, cardW, cardH) },
      { name: "br", n: count(S.W - cardW, -1, cardW, cardH) },
      { name: "bl", n: count(0, -1, cardW, cardH) },
    ];
    corners.sort((a, b) => a.n - b.n);
    hud.classList.remove("tr", "br", "bl", "strip");
    if (corners[0].n <= 2) { hud.classList.add(corners[0].name); return; }
    const stripH = Math.ceil(52 / cellPx);
    const stripW = Math.ceil(520 / cellPx);
    const right = count(S.W - stripW, -1, stripW, stripH), left = count(0, -1, stripW, stripH);
    hud.classList.add("strip", right <= left ? "br" : "bl");
  }

  // ---------------------------------------------------------------------------
  // Data apply (trajectory + frames) -- also used for incremental live updates
  // ---------------------------------------------------------------------------
  function applyData(fresh) {
    const sc = S.scene;
    const prevN = S.states.length;
    S.states = (sc.trajectory && sc.trajectory.states) || [];
    S.headings = computeHeadings(S.states);
    buildTrail();
    buildCar();

    // scalar overlay rects: create any cell that appears in some frame
    if (sc.mode === "Value" && sc.scalar) {
      const fillA = sc.colors.asphaltDim;
      sc.scalar.frames.forEach((fr) => fr.forEach(([x, y]) => {
        const k = x + "," + y;
        if (!cellIndex[k]) cellIndex[k] = el("rect", { x, y: S.H - y - 1, width: 1, height: 1, fill: fillA, class: "value-cell" }, L.overlay);
      }));
    }

    // transport
    const n = S.states.length, has = n > 1;
    ["b-reset", "b-prev", "b-play", "b-next", "b-end", "scrub", "speed"].forEach((id) => { $(id).disabled = !has; });
    $("scrub").max = Math.max(n - 1, 0);
    $("bar").hidden = !n;

    // iteration bar
    const fs = frameSet();
    const nIter = fs ? fs.frames.length : 0;
    $("iterbar").hidden = nIter <= 1;
    $("iscrub").max = Math.max(nIter - 1, 0);

    // where to be
    const following = S.follow && !S.hold;
    if (following) {
      S.iter = Math.max(nIter - 1, 0);
      if (fresh) { S.t = lastStep(); }
      else if (n < prevN) { S.t = 0; chaseTo(lastStep()); }
      else chaseTo(lastStep());
    } else {
      S.t = clamp(S.t, 0, lastStep());
      S.iter = clamp(S.iter, 0, Math.max(nIter - 1, 0));
    }

    buildBadges();
    buildLegend();
    buildEmpty();
    applyIteration(false);
    drawSelection();
    drawFrame();
  }

  function computeHeadings(states) {
    const out = [];
    let prev = 0;
    for (let i = 0; i < states.length; i++) {
      const a = states[i], b = states[i + 1];
      let dx = 0, dy = 0;
      if (b) { dx = b.x - a.x; dy = b.y - a.y; }
      if (!dx && !dy) { dx = a.vx; dy = a.vy; }
      if (dx || dy) prev = Math.atan2(-dy, dx) * 180 / Math.PI;
      out.push(prev);
    }
    return out;
  }

  // ---- trail ---------------------------------------------------------------
  const T = {};
  function buildTrail() {
    const C = S.scene.colors, emph = S.scene.mode === "Trajectory";
    clear(L.trail);
    T.driven = null;
    if (!S.states.length || !S.scene.options.trajectory) return;
    T.remaining = el("path", { fill: "none", stroke: C.trail, "stroke-width": 0.11, "stroke-dasharray": "0.06 0.34", "stroke-linecap": "round", opacity: 0.55 }, L.trail);
    T.casing = el("path", { fill: "none", stroke: C.trailCasing, "stroke-width": emph ? 0.5 : 0.44, "stroke-linejoin": "round", "stroke-linecap": "round", opacity: 0.9 }, L.trail);
    T.driven = el("path", { fill: "none", stroke: C.trail, "stroke-width": emph ? 0.26 : 0.22, "stroke-linejoin": "round", "stroke-linecap": "round", opacity: 0.85 }, L.trail);
    T.recent = el("path", { fill: "none", stroke: C.trail, "stroke-width": emph ? 0.3 : 0.24, "stroke-linejoin": "round", "stroke-linecap": "round" }, L.trail);
    T.dots = el("g", null, L.trail);
    T.crash = el("g", null, L.trail);
    const s0 = S.states[0];
    el("circle", { cx: sx(s0.x), cy: sy(s0.y), r: 0.3, fill: "#fff", stroke: C.trail, "stroke-width": 0.12 }, L.trail);
    S.states.forEach((s) => {
      if (!s.crashed) return;
      const g = el("g", { transform: `translate(${sx(s.x)} ${sy(s.y)})` }, T.crash);
      el("circle", { r: 0.46, fill: C.crash, opacity: 0.18 }, g);
      el("path", { d: "M-0.26 -0.26 L0.26 0.26 M0.26 -0.26 L-0.26 0.26", stroke: C.crash, "stroke-width": 0.14, "stroke-linecap": "round" }, g);
    });
  }

  // ---- car -----------------------------------------------------------------
  const Car = {};
  function buildCar() {
    const C = S.scene.colors;
    clear(L.car);
    Car.g = null;
    if (!S.states.length) return;
    Car.vel = el("path", { fill: "none", stroke: C.velocity, "stroke-width": 0.16, "stroke-linecap": "round", "marker-end": "url(#vel-head)" }, L.car);
    Car.g = el("g", { filter: "url(#soft)" }, L.car);
    const g = el("g", { id: "car-body", transform: "scale(1.6)" }, Car.g);
    [[-0.34, -0.44], [0.3, -0.44], [-0.34, 0.28], [0.3, 0.28]].forEach(([x, y]) =>
      el("rect", { x, y, width: 0.34, height: 0.16, rx: 0.07, fill: C.carWheel }, g));
    el("rect", { x: -0.66, y: -0.36, width: 1.32, height: 0.72, rx: 0.3, fill: C.car, stroke: C.carDark, "stroke-width": 0.05 }, g);
    el("rect", { x: -0.3, y: -0.29, width: 0.44, height: 0.58, rx: 0.12, fill: "#FF8A8A" }, g);
    el("path", { d: "M0.14 -0.25 L0.44 -0.2 L0.44 0.2 L0.14 0.25 Z", fill: C.carGlass, stroke: C.carDark, "stroke-width": 0.03, "stroke-linejoin": "round" }, g);
    el("path", { d: "M-0.3 -0.22 L-0.5 -0.17 L-0.5 0.17 L-0.3 0.22 Z", fill: C.carGlass, opacity: 0.9 }, g);
    el("circle", { cx: 0.6, cy: -0.2, r: 0.085, fill: C.carLight }, g);
    el("circle", { cx: 0.6, cy: 0.2, r: 0.085, fill: C.carLight }, g);
    Car.stopped = el("circle", { r: 1.35, fill: "none", stroke: C.velocity, "stroke-width": 0.07, "stroke-dasharray": "0.2 0.16", opacity: 0.5 }, L.car);
  }

  // ---- badges / legend / empty --------------------------------------------
  function buildBadges() {
    const sc = S.scene, box = $("badges");
    box.innerHTML = "";
    const chip = (text, cls, id) => {
      const d = document.createElement("div");
      d.className = "chip " + (cls || "");
      if (id) d.id = id;
      d.textContent = text;
      box.appendChild(d);
      return d;
    };
    chip(sc.mode, "mode");
    if (sc.mode === "Value" && sc.scalar) chip(sc.scalar.label);
    if (sc.mode === "Action" && sc.vector) chip(sc.vector.label);
    if (sc.sliceLabel && (sc.mode === "Value" || sc.mode === "Action")) chip(sc.sliceLabel, "slice");
    if (sc.source === "demo") chip("DEMO DATA", "demo");
    if (sc.source === "connected") chip("CONNECTED", "live");
    if (sc.live) {
      chip("", "livepulse", "live-chip");
      if (sc.updateLabel) chip(sc.updateLabel, "update", "update-chip");
    }
    updateLiveChrome();
  }

  function updateLiveChrome() {
    const c = $("live-chip");
    if (c) {
      const status = S.liveStatus || "live";
      c.textContent = status === "done" ? "FINISHED" : status === "stale" ? "LIVE · waiting" : "LIVE";
      c.classList.toggle("done", status === "done");
      c.classList.toggle("stale", status === "stale");
    }
    const b = $("b-live");
    b.hidden = !(S.live && S.follow && S.hold);
  }

  function buildLegend() {
    const sc = S.scene, C = sc.colors, box = $("legend");
    box.innerHTML = "";
    const item = (html) => { const d = document.createElement("span"); d.className = "it"; d.innerHTML = html; box.appendChild(d); };
    if (S.states.length) {
      item(`<span class="sw car" style="background:${C.car}"></span>Car`);
      if (sc.options.trajectory) item(`<span class="sw line" style="background:${C.trail}"></span>Trail`);
    }
    item(`<span class="sw" style="background:#fff;box-shadow:inset 0 0 0 2px ${C.start}"></span>Start`);
    item(`<span class="sw check"></span>Finish`);
    item(`<span class="sw" style="background:${C.grass}"></span>Off-track`);
    if (sc.mode === "Value" && sc.scalar) {
      const g = `linear-gradient(90deg, ${sc.scalar.ramp.join(",")})`;
      item(`<span class="ramp"><span>${fmt(sc.scalar.vmin)}</span><span class="g" style="background:${g}"></span><span>${fmt(sc.scalar.vmax)}</span><span class="hint">${sc.scalar.label}</span></span>`);
    }
    if (sc.mode === "Action" && sc.vector) item(`<span class="hint">arrow = direction · length = magnitude · ring = zero</span>`);
    if (S.selected) item(`<span class="sw" style="background:#fff;box-shadow:inset 0 0 0 2px ${C.selection}"></span>Selected`);
  }

  function buildEmpty() {
    const sc = S.scene, box = $("empty");
    let msg = null;
    if (sc.mode === "Value" && !sc.scalar) msg = ["No value data yet", sc.live ? "Waiting for a scalar field from the run." : "Connect a scalar field to paint it onto the circuit."];
    if (sc.mode === "Action" && !sc.vector) msg = ["No action data yet", sc.live ? "Waiting for a vector field from the run." : "Connect a vector field to draw it as arrows."];
    if ((sc.mode === "Trajectory" || sc.mode === "Track") && !S.states.length) msg = ["No trajectory yet", sc.live ? "Waiting for states from the run." : "Connect a rollout to see the car drive."];
    box.hidden = !msg;
    if (msg) box.innerHTML = `<div class="t">${msg[0]}</div><div class="d">${msg[1]}</div>`;
  }

  // ---------------------------------------------------------------------------
  // Iteration frames
  // ---------------------------------------------------------------------------
  function applyIteration(report) {
    const sc = S.scene, fs = frameSet();
    if (fs) {
      S.iter = clamp(S.iter, 0, fs.frames.length - 1);
      const frame = fs.frames[S.iter] || [];
      if (sc.mode === "Value") {
        const seen = new Set();
        const { vmin, vmax } = sc.scalar;
        frame.forEach(([x, y, v]) => {
          const k = x + "," + y, r = cellIndex[k];
          if (!r) return;
          r.setAttribute("fill", ramp(sc.scalar.ramp, (v - vmin) / ((vmax - vmin) || 1)));
          r.dataset.v = v;
          seen.add(k);
        });
        for (const k in cellIndex) if (!seen.has(k)) { cellIndex[k].setAttribute("fill", sc.colors.asphaltDim); delete cellIndex[k].dataset.v; }
        drawNumbers(frame);
      } else {
        drawVectors(frame);
      }
      const lbl = (fs.frameLabels && fs.frameLabels[S.iter]) || `iteration ${S.iter}`;
      $("iterLabel").textContent = `${lbl}  ·  ${S.iter + 1} / ${fs.frames.length}`;
      $("iscrub").value = S.iter;
      if (report) reportValue({ type: "iteration", index: (fs.frameIndices && fs.frameIndices[S.iter]) ?? S.iter });
    }
  }

  function drawNumbers(frame) {
    clear(L.numbers);
    if (!S.scene.options.numbers) return;
    frame.forEach(([x, y, v]) => {
      const t = el("text", { x: sx(x), y: sy(y) + 0.12, "text-anchor": "middle", "font-size": 0.3, fill: "#2B2A33", opacity: 0.85 }, L.numbers);
      t.textContent = (+v).toFixed(Math.abs(v) < 10 ? 1 : 0);
    });
  }

  let vecIndex = {};
  function drawVectors(frame) {
    const sc = S.scene, C = sc.colors, samp = sc.vector.sampling || 1, peak = sc.vector.peak || 1;
    clear(L.vectors);
    vecIndex = {};
    frame.forEach(([x, y, u, v]) => {
      vecIndex[x + "," + y] = [u, v];
      if (x % samp || y % samp) return;
      const mag = Math.hypot(u, v);
      const g = el("g", { transform: `translate(${sx(x)} ${sy(y)})`, class: "vec" }, L.vectors);
      if (!mag) { el("circle", { r: 0.14, fill: "none", stroke: C.arrow, "stroke-width": 0.07, opacity: 0.8 }, g); return; }
      const len = (0.34 + 0.5 * (mag / peak)) * samp;
      const ang = Math.atan2(-v, u) * 180 / Math.PI;
      const a = el("g", { transform: `rotate(${ang})` }, g);
      el("path", { d: `M${-len / 2} 0 H${len / 2 - 0.22}`, stroke: C.arrow, "stroke-width": 0.09, "stroke-linecap": "round", opacity: 0.9 }, a);
      el("path", { d: `M${len / 2} 0 L${len / 2 - 0.3} -0.17 L${len / 2 - 0.3} 0.17 Z`, fill: C.arrow, opacity: 0.9 }, a);
    });
  }

  // ---------------------------------------------------------------------------
  // Frame drawing (car + trail + HUD)
  // ---------------------------------------------------------------------------
  function lerpAngle(a, b, f) {
    const d = ((b - a + 540) % 360) - 180;
    return a + d * f;
  }

  function drawFrame() {
    const n = S.states.length;
    const hud = $("hud");
    if (!n) { hud.hidden = true; return; }
    hud.hidden = false;
    S.t = clamp(S.t, 0, n - 1);
    const i = Math.floor(S.t), f = S.t - i;
    const a = S.states[i], b = S.states[Math.min(i + 1, n - 1)];
    const x = a.x + (b.x - a.x) * f, y = a.y + (b.y - a.y) * f;
    const heading = lerpAngle(S.headings[i], S.headings[Math.min(i + 1, n - 1)], f);

    const hasVel = !!(S.scene.trajectory && S.scene.trajectory.hasVelocity);
    if (Car.g) {
      Car.g.setAttribute("transform", `translate(${sx(x)} ${sy(y)}) rotate(${heading})`);
      const moving = a.vx || a.vy;
      Car.stopped.setAttribute("transform", `translate(${sx(x)} ${sy(y)})`);
      Car.stopped.setAttribute("visibility", hasVel && !moving ? "visible" : "hidden");
      if (S.scene.options.velocity && hasVel && moving) {
        const dir = Math.atan2(-a.vy, a.vx);
        const ox = Math.cos(dir) * 1.1, oy = Math.sin(dir) * 1.1;
        Car.vel.setAttribute("d", `M${sx(x) + ox} ${sy(y) + oy} L${sx(x) + a.vx} ${sy(y) - a.vy}`);
        Car.vel.setAttribute("visibility", Math.hypot(a.vx, a.vy) > 1.3 ? "visible" : "hidden");
      } else Car.vel.setAttribute("visibility", "hidden");
    }

    if (T.driven) {
      const pts = [];
      for (let k = 0; k <= i; k++) pts.push(`${sx(S.states[k].x)} ${sy(S.states[k].y)}`);
      pts.push(`${sx(x)} ${sy(y)}`);
      const d = "M" + pts.join(" L");
      T.casing.setAttribute("d", d);
      T.driven.setAttribute("d", d);
      const from = Math.max(0, i - 5);
      const rp = [];
      for (let k = from; k <= i; k++) rp.push(`${sx(S.states[k].x)} ${sy(S.states[k].y)}`);
      rp.push(`${sx(x)} ${sy(y)}`);
      T.recent.setAttribute("d", "M" + rp.join(" L"));
      if (S.scene.options.remaining && i < n - 1) {
        const rem = [`${sx(x)} ${sy(y)}`];
        for (let k = i + 1; k < n; k++) rem.push(`${sx(S.states[k].x)} ${sy(S.states[k].y)}`);
        T.remaining.setAttribute("d", "M" + rem.join(" L"));
      } else T.remaining.setAttribute("d", "");
      if (S.scene.mode === "Trajectory") {
        clear(T.dots);
        for (let k = 1; k <= i; k++) {
          const s = S.states[k];
          el("circle", { cx: sx(s.x), cy: sy(s.y), r: 0.14, fill: S.scene.colors.trail, stroke: "#fff", "stroke-width": 0.05, opacity: 0.35 + 0.65 * (k / Math.max(i, 1)) }, T.dots);
        }
      }
    }

    const flags = [];
    if (a.crashed) flags.push("collision");
    hud.innerHTML =
      `<div class="title">Current state${S.live && S.follow && !S.hold ? '<span class="livedot"></span>' : ""}</div>` +
      `<div class="row"><span class="k">Position</span><span class="v big">${fmtPair(a.x, a.y)}</span></div>` +
      (hasVel ? `<div class="row"><span class="k">Velocity</span><span class="v big">${fmtPair(a.vx, a.vy)}</span></div>` : "") +
      (hasVel ? `<div class="row"><span class="k">Speed</span><span class="v">${Math.hypot(a.vx, a.vy).toFixed(2)}</span></div>` : "") +
      `<div class="row"><span class="k">Step</span><span class="v">${i}<span class="dim">/ ${n - 1}</span></span></div>` +
      (a.action ? `<div class="row"><span class="k">Action</span><span class="v">${fmtPair(a.action[0], a.action[1])}</span></div>` : "") +
      (a.reward !== undefined ? `<div class="row"><span class="k">Reward</span><span class="v">${fmt(a.reward)}</span></div>` : "") +
      (flags.length ? `<div class="flag">✕ ${flags.join(", ")}</div>` : "");

    if (!S.scrubbing) $("scrub").value = i;
    $("stepLabel").textContent = `step ${i} / ${n - 1}`;
  }

  function drawSelection() {
    clear(L.select);
    if (!S.selected) return;
    const [x, y] = S.selected, C = S.scene.colors;
    el("rect", { x: x - 0.08, y: S.H - y - 1 - 0.08, width: 1.16, height: 1.16, rx: 0.3, fill: "none", stroke: C.selection, "stroke-width": 0.3, opacity: 0.35 }, L.select);
    el("rect", { x: x + 0.02, y: S.H - y - 1 + 0.02, width: 0.96, height: 0.96, rx: 0.22, fill: "none", stroke: C.selection, "stroke-width": 0.12 }, L.select);
  }

  // ---------------------------------------------------------------------------
  // Playback, chase (live catch-up), hold
  // ---------------------------------------------------------------------------
  function msPerStep() { return SPEED_MS[S.speed] || 620; }

  function play() {
    const n = S.states.length;
    if (n < 2) return;
    S.chase = null;
    if (S.t >= n - 1) S.t = 0;
    S.playing = true;
    S.lastTs = performance.now();
    $("b-play").classList.add("on");
    $("b-play").title = "Pause";
    cancelAnimationFrame(S.raf);
    S.raf = requestAnimationFrame(tick);
    reportValue({ type: "play", playing: true });
  }
  function pause(report = true) {
    S.playing = false;
    cancelAnimationFrame(S.raf);
    $("b-play").classList.remove("on");
    $("b-play").title = "Play";
    S.t = Math.round(S.t);
    drawFrame();
    if (report) reportStep();
  }
  function tick(ts) {
    if (!S.playing) return;
    const dt = Math.min(ts - S.lastTs, 100);
    S.lastTs = ts;
    S.t += dt / msPerStep();
    const n = S.states.length;
    if (S.t >= n - 1) { S.t = n - 1; drawFrame(); pause(); return; }
    drawFrame();
    S.raf = requestAnimationFrame(tick);
  }
  function jump(to) {
    pause(false);
    S.t = clamp(Math.round(to), 0, lastStep());
    drawFrame();
    reportStep();
  }

  function chaseTo(target) {
    if (S.playing) return;
    if (S.t >= target) { S.t = target; S.chase = null; drawFrame(); return; }
    S.chase = { from: S.t, target, start: performance.now(), dur: clamp(S.pollMs * 0.85, 250, 900) };
    cancelAnimationFrame(S.raf);
    S.raf = requestAnimationFrame(chaseTick);
  }
  function chaseTick(ts) {
    const c = S.chase;
    if (!c) return;
    const f = Math.min(1, (ts - c.start) / c.dur);
    const e = 1 - Math.pow(1 - f, 3);
    S.t = c.from + (c.target - c.from) * e;
    drawFrame();
    if (f < 1) S.raf = requestAnimationFrame(chaseTick);
    else S.chase = null;
  }

  function userTouched() {
    // Any manual transport use while following live pauses the following.
    if (S.follow && !S.hold) setHold(true);
  }
  function setHold(v) {
    S.hold = v;
    updateLiveChrome();
    drawFrame();
    if (!v) {
      S.chase = null;
      pause(false);
      const fs = frameSet();
      if (fs) { S.iter = fs.frames.length - 1; applyIteration(false); }
      chaseTo(lastStep());
      reportValue({ type: "hold", hold: false });
    }
  }

  function iterPlay() {
    const max = +$("iscrub").max;
    if (max < 1) return;
    S.iterPlaying = true;
    $("i-play").classList.add("on");
    if (S.iter >= max) { S.iter = 0; applyIteration(false); }
    clearInterval(S.iterTimer);
    S.iterTimer = setInterval(() => {
      if (S.iter >= max) { iterPause(); return; }
      S.iter += 1;
      applyIteration(false);
    }, 700);
  }
  function iterPause() {
    S.iterPlaying = false;
    clearInterval(S.iterTimer);
    $("i-play").classList.remove("on");
    applyIteration(true);
  }

  // ---------------------------------------------------------------------------
  // Reporting to Python
  // ---------------------------------------------------------------------------
  function reportValue(obj) { Host.value(Object.assign({ nonce: Date.now() + Math.random(), hold: S.hold }, obj)); }
  function reportStep() {
    const step = Math.round(S.t);
    S.lastReported.step = step;
    reportValue({ type: "step", step, playing: false });
  }

  // ---------------------------------------------------------------------------
  // Hover & click
  // ---------------------------------------------------------------------------
  const tip = $("tip");
  function cellFromEvent(ev) {
    const t = ev.target;
    if (!(t instanceof SVGElement) || !t.classList.contains("hit")) return null;
    return [+t.dataset.x, +t.dataset.y];
  }
  svg.addEventListener("mousemove", (ev) => {
    const c = cellFromEvent(ev);
    if (!c) { tip.hidden = true; return; }
    const [x, y] = c, sc = S.scene, code = sc.track.grid[y][x];
    const kind = code === 2 ? "Start" : code === 3 ? "Finish" : "Track";
    let html = `<div><b>(${x}, ${y})</b> <span class="k"></span>${kind}</div>`;
    if (sc.sliceLabel && (sc.mode === "Value" || sc.mode === "Action")) html += `<div><span class="k">slice</span><b>${sc.sliceLabel.replace(/\s+/g, " ")}</b></div>`;
    const r = cellIndex[x + "," + y];
    if (r && r.dataset.v !== undefined && sc.scalar) html += `<div><span class="k">${sc.scalar.label}</span><b>${fmt(+r.dataset.v)}</b></div>`;
    const vv = sc.mode === "Action" && vecIndex[x + "," + y];
    if (vv && sc.vector) html += `<div><span class="k">${sc.vector.label}</span><b>${fmtPair(vv[0], vv[1])}</b></div>`;
    tip.innerHTML = html;
    tip.hidden = false;
    const stage = $("stage").getBoundingClientRect();
    let left = ev.clientX - stage.left + 14, top = ev.clientY - stage.top + 14;
    if (left + tip.offsetWidth > stage.width - 8) left = ev.clientX - stage.left - tip.offsetWidth - 12;
    if (top + tip.offsetHeight > stage.height - 8) top = ev.clientY - stage.top - tip.offsetHeight - 12;
    tip.style.left = left + "px"; tip.style.top = top + "px";
  });
  svg.addEventListener("mouseleave", () => { tip.hidden = true; });
  svg.addEventListener("click", (ev) => {
    const c = cellFromEvent(ev);
    if (!c) return;
    S.selected = (S.selected && S.selected[0] === c[0] && S.selected[1] === c[1]) ? null : c;
    drawSelection();
    buildLegend();
    reportValue({ type: "select", cell: S.selected });
    resize();
  });

  // ---------------------------------------------------------------------------
  // Controls wiring
  // ---------------------------------------------------------------------------
  $("b-play").addEventListener("click", () => { userTouched(); S.playing ? pause() : play(); });
  $("b-reset").addEventListener("click", () => { userTouched(); jump(0); });
  $("b-end").addEventListener("click", () => { userTouched(); jump(lastStep()); });
  $("b-prev").addEventListener("click", () => { userTouched(); jump(Math.round(S.t) - 1); });
  $("b-next").addEventListener("click", () => { userTouched(); jump(Math.round(S.t) + 1); });
  $("scrub").addEventListener("input", (e) => { userTouched(); S.scrubbing = true; pause(false); S.chase = null; S.t = +e.target.value; drawFrame(); });
  $("scrub").addEventListener("change", () => { S.scrubbing = false; reportStep(); });
  $("speed").addEventListener("change", (e) => { S.speed = +e.target.value; });
  $("b-live").addEventListener("click", () => setHold(false));

  $("i-play").addEventListener("click", () => { userTouched(); S.iterPlaying ? iterPause() : iterPlay(); });
  $("i-prev").addEventListener("click", () => { userTouched(); iterPause(); S.iter = Math.max(0, S.iter - 1); applyIteration(true); });
  $("i-next").addEventListener("click", () => { userTouched(); iterPause(); S.iter = Math.min(+$("iscrub").max, S.iter + 1); applyIteration(true); });
  $("iscrub").addEventListener("input", (e) => { userTouched(); clearInterval(S.iterTimer); S.iterPlaying = false; $("i-play").classList.remove("on"); S.iter = +e.target.value; applyIteration(false); });
  $("iscrub").addEventListener("change", () => applyIteration(true));

  // ---------------------------------------------------------------------------
  // Sizing
  // ---------------------------------------------------------------------------
  function resize() {
    const root = $("root");
    const w = root.clientWidth || 800;
    if (S.W && S.H) {
      const h = Math.round(w * (S.H + 2 * PAD) / (S.W + 2 * PAD));
      svg.setAttribute("height", h);
      svg.style.height = h + "px";
    }
    Host.height(root.scrollHeight + 6);
  }
  window.addEventListener("resize", resize);

  // ---------------------------------------------------------------------------
  // Render entry
  // ---------------------------------------------------------------------------
  function onRender(args) {
    const payload = args && args.payload;
    if (!payload || !payload.structure) return;
    const st = payload.state || {};
    S.follow = !!st.follow;
    S.live = !!st.live;
    S.liveStatus = st.liveStatus || "";
    if (st.pollMs) S.pollMs = st.pollMs;
    if (st.resumeNonce && st.resumeNonce !== S.resumeNonce) {
      const first = S.resumeNonce === undefined;
      S.resumeNonce = st.resumeNonce;
      if (!first && S.hold) { S.hold = false; S.pendingResume = true; }
    }

    const sKey = JSON.stringify(payload.structure);
    const firstRender = S.structureKey === "";
    if (sKey !== S.structureKey) {
      const wasPlaying = S.playing;
      S.structureKey = sKey;
      S.dataRev = payload.data.rev;
      S.scene = Object.assign({}, payload.structure, payload.data);
      S.selected = st.selected || null;
      // hold is owned here; Python's copy lags one round trip, so only seed it once
      if (firstRender) S.hold = !!st.hold;
      S.iter = st.iteration || 0;
      if (!wasPlaying) S.t = st.step || 0;
      buildScene();
      if (wasPlaying && S.states.length > 1) { S.lastTs = performance.now(); S.raf = requestAnimationFrame(tick); } else S.playing = false;
    } else if (payload.data.rev !== S.dataRev) {
      S.dataRev = payload.data.rev;
      S.scene = Object.assign({}, payload.structure, payload.data);
      applyData(false);
    } else {
      const sel = st.selected || null;
      if (JSON.stringify(sel) !== JSON.stringify(S.selected)) { S.selected = sel; drawSelection(); buildLegend(); }
      const following = S.follow && !S.hold;
      if (!following && !S.playing && !S.chase && st.step !== undefined && st.step !== Math.round(S.t) && st.step !== S.lastReported.step) { S.t = st.step; drawFrame(); }
      if (!following && st.iteration !== undefined && st.iteration !== S.iter && !S.iterPlaying) { S.iter = st.iteration; applyIteration(false); }
      updateLiveChrome();
    }
    if (S.pendingResume) { S.pendingResume = false; setHold(false); }
    resize();
    setTimeout(resize, 60);
  }

  window.addEventListener("message", (ev) => {
    const d = ev.data;
    if (d && d.type === "streamlit:render") onRender(d.args);
  });
  Host.ready();
  Host.height(420);
})();
