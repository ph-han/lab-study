class Component extends DCLogic {
  // Fixtures: REDESIGN_SPEC.md 8.2 (example B). V[k] = table after sweep k.
  FIX() {
    return {
      order: ['0,0', '1,0', '2,0', '0,1', '1,1'],
      goal: '2,1', start: '0,0', acts: ['r', 'l', 'u', 'd'],
      V: [
        { '0,0': 0, '1,0': 0, '2,0': 0, '0,1': 0, '1,1': 0, '2,1': 0 },
        { '0,0': 0, '1,0': 0, '2,0': 0.25, '0,1': 0, '1,1': 0.25, '2,1': 0 },
        { '0,0': 0, '1,0': 0.1125, '2,0': 0.3625, '0,1': 0.05625, '1,1': 0.30625, '2,1': 0 }
      ],
      T: {
        '0,0': { r: { s: '1,0' }, l: { s: '0,0', b: 1 }, u: { s: '0,1' }, d: { s: '0,0', b: 1 } },
        '1,0': { r: { s: '2,0' }, l: { s: '0,0' }, u: { s: '1,1' }, d: { s: '1,0', b: 1 } },
        '2,0': { r: { s: '2,0', b: 1 }, l: { s: '1,0' }, u: { s: '2,1', rw: 1, g: 1 }, d: { s: '2,0', b: 1 } },
        '0,1': { r: { s: '1,1' }, l: { s: '0,1', b: 1 }, u: { s: '0,1', b: 1 }, d: { s: '0,0' } },
        '1,1': { r: { s: '2,1', rw: 1, g: 1 }, l: { s: '0,1' }, u: { s: '1,1', b: 1 }, d: { s: '1,0' } }
      }
    };
  }

  st() {
    return Object.assign(
      { k: 0, cursor: 0, p: 0, sel: '0,0', layer: 'value', play: true, tick: 0, pop: false },
      this.state || {}
    );
  }
  f(x) { const s = x.toFixed(5).replace(/0+$/, '').replace(/\.$/, ''); return s === '-0' ? '0' : s; }
  lab(k) { const p = k.split(','); return '(' + p[0] + ',' + p[1] + ')'; }
  ctr(k) { const p = k.split(',').map(Number); return { x: p[0] * 202 + 95, y: (1 - p[1]) * 202 + 95 }; }
  lerp(a, b, t) { return Math.round(a + (b - a) * t); }
  ramp(v) {
    const S = [[0, 255, 253, 246], [.2, 255, 242, 204], [.4, 255, 226, 162], [.6, 255, 202, 139], [.8, 255, 174, 134], [1, 255, 142, 134]];
    const x = Math.max(0, Math.min(1, v));
    for (let i = 1; i < S.length; i++) {
      if (x <= S[i][0]) {
        const a = S[i - 1], b = S[i], t = (x - a[0]) / (b[0] - a[0]);
        return 'rgb(' + this.lerp(a[1], b[1], t) + ',' + this.lerp(a[2], b[2], t) + ',' + this.lerp(a[3], b[3], t) + ')';
      }
    }
    return 'rgb(255,142,134)';
  }
  terms(key, V) {
    const F = this.FIX(), T = F.T[key];
    if (!T) return [];
    return F.acts.map((a) => {
      const t = T[a], rw = t.rw || 0, vn = V[t.s];
      return { a: a, next: t.s, blocked: !!t.b, goal: !!t.g, r: rw, vnext: vn, B: rw + 0.9 * vn, w: 0.25 * (rw + 0.9 * vn) };
    });
  }

  // ---- transport -----------------------------------------------------------
  stop() { if (this.t) { clearInterval(this.t); this.t = null; } }
  componentDidMount() { if (this.st().play) this.start(); }
  componentWillUnmount() { this.stop(); if (this.pt) clearTimeout(this.pt); }
  start() {
    this.stop();
    this.t = setInterval(() => this.beat(), 620);
  }
  beat() {
    const s = this.st(), F = this.FIX();
    if (s.p < 4) { this.setState({ p: s.p + 1, tick: s.tick + 1 }); return; }
    // the four action terms are in, record the cell
    if (s.cursor < 4) {
      this.setState({ cursor: s.cursor + 1, sel: F.order[s.cursor + 1], p: 0, tick: s.tick + 1 });
      return;
    }
    if (s.k < 2) { this.commit(s.k + 1); return; }
    this.stop();
    this.pt = setTimeout(() => { this.setState({ k: 0, cursor: 0, p: 0, sel: F.order[0] }); this.start(); }, 1500);
  }
  commit(nk) {
    const F = this.FIX();
    this.setState({ k: nk, cursor: 0, p: 0, sel: F.order[0], pop: true });
    if (this.pt) clearTimeout(this.pt);
    this.pt = setTimeout(() => this.setState({ pop: false }), 560);
  }
  hand(patch) { this.stop(); this.setState(Object.assign({ play: false }, patch)); }

  renderVals() {
    const s = this.st(), F = this.FIX(), ORDER = F.order, self = this;
    const GL = { r: '→', l: '←', u: '↑', d: '↓' };
    const k = s.k, done = k >= 2 && s.cursor === 0 && s.p === 0 && !s.play;
    const wk = Math.min(k + 1, 2);
    const Vold = F.V[wk - 1], Vnew = F.V[wk], Vmap = F.V[k];
    const cur = ORDER[s.cursor];
    const sel = F.T[s.sel] ? s.sel : cur;
    const rows = this.terms(sel, Vold);
    const act = s.p < 4 ? F.acts[s.p] : null;
    const ref = act ? F.T[sel][act] : null;

    // ---- cells ----
    const cells = ORDER.concat([F.goal]).map((key) => {
      const p = key.split(',').map(Number);
      const goal = key === F.goal;
      const idx = ORDER.indexOf(key);
      const has = idx > -1 && idx < s.cursor && k < 2;
      const cls = [];
      if (goal) cls.push('goal');
      if (key === sel) cls.push('sel');
      else if (ref && !ref.b && ref.s === key) cls.push('ref');
      else if (key === cur && k < 2) cls.push('next');
      if (s.pop && !goal) cls.push('pop');
      const pol = s.layer === 'policy';
      return {
        goalCls: goal ? '' : 'hid', stateCls: goal ? 'hid' : '',
        left: p[0] * 202, top: (1 - p[1]) * 202,
        bg: goal ? '#E4F5F0' : this.ramp(Vmap[key]),
        cls: cls.join(' '),
        value: this.f(Vmap[key]), co: this.lab(key),
        tag: key === F.start ? 'S' : '',
        au: pol ? '' : 'hid', ad: pol ? '' : 'hid', al: pol ? '' : 'hid', ar: pol ? '' : 'hid',
        pu: '25', pd: '25', pl: '25', pr: '25',
        newCls: has ? '' : 'hid', newValue: has ? this.f(F.V[k + 1][key]) : '',
        pick: () => self.hand({ sel: goal ? sel : key, p: 0 })
      };
    });

    // ---- movement arrow ----
    const D = { r: [1, 0], l: [-1, 0], u: [0, -1], d: [0, 1] };
    const c0 = this.ctr(sel);
    const arrow = { d: '', head: '', len: 1, anim: 'none', rx: -999, ry: -999, r: '', rcls: 'hid' };
    if (act) {
      const t = F.T[sel][act], d = D[act], px = -d[1], py = d[0];
      if (t.b) {
        const ax = c0.x + d[0] * 66 + px * 18, ay = c0.y + d[1] * 66 + py * 18;
        const bx = c0.x + d[0] * 104, by = c0.y + d[1] * 104;
        const cx = c0.x + d[0] * 66 - px * 18, cy = c0.y + d[1] * 66 - py * 18;
        arrow.d = 'M ' + ax + ' ' + ay + ' Q ' + bx + ' ' + by + ' ' + cx + ' ' + cy;
        arrow.head = (cx - d[0] * 15) + ',' + (cy - d[1] * 15) + ' ' + (cx + px * 9) + ',' + (cy + py * 9) + ' ' + (cx - px * 9) + ',' + (cy - py * 9);
        arrow.len = 96;
        arrow.rx = c0.x + d[0] * 66 + px * 50; arrow.ry = c0.y + d[1] * 66 + py * 50;
      } else {
        const c1 = this.ctr(t.s);
        const sx = c0.x + d[0] * 80, sy = c0.y + d[1] * 80;
        const ex = c1.x - d[0] * 80, ey = c1.y - d[1] * 80;
        arrow.d = 'M ' + sx + ' ' + sy + ' L ' + ex + ' ' + ey;
        arrow.head = (ex + d[0] * 15) + ',' + (ey + d[1] * 15) + ' ' + (ex + px * 9) + ',' + (ey + py * 9) + ' ' + (ex - px * 9) + ',' + (ey - py * 9);
        arrow.len = Math.abs(ex - sx) + Math.abs(ey - sy);
        arrow.rx = (sx + ex) / 2; arrow.ry = (sy + ey) / 2;
      }
      arrow.anim = s.tick % 2 ? 'drawA' : 'drawB';
      const rw = t.rw || 0;
      arrow.r = rw > 0 ? 'r +1' : 'r 0';
      arrow.rcls = rw > 0 ? '' : 'zero';
    }

    // ---- action rows ----
    const rowItems = rows.map((t, i) => ({
      g: GL[t.a],
      next: t.blocked ? '제자리' : (t.goal ? 'G 종료' : this.lab(t.next)),
      r: t.r > 0 ? '+1' : '0', rcls: t.r > 0 ? 'pos' : '',
      b: t.B.toFixed(4),
      wB: Math.round(t.B * 100),
      wW: i < s.p ? Math.round(t.w * 100) : 0,
      cls: i === s.p ? 'on' : (i < s.p ? '' : 'dim'),
      pick: () => self.hand({ p: i })
    }));

    // ---- running sum ----
    const TINT = ['#93B4F5', '#B9CDF8', '#2563EB', '#5D8FEE'];
    const partial = rows.slice(0, s.p).reduce((a, t) => a + t.w, 0);
    const full = s.p >= 4;
    const prev = Vold[sel];
    const res = {
      value: this.f(full ? Vnew[sel] : partial),
      from: full ? '이전 ' + this.f(prev) : (s.p + '/4 행동 반영'),
      delta: full ? (Vnew[sel] - prev >= 0 ? '+' : '') + this.f(Vnew[sel] - prev) : '',
      deltaCls: full ? '' : 'hid',
      anim: s.tick % 2 ? 'popUpA' : 'popUpB',
      say: act && F.T[sel][act].g ? '목표로 들어가는 행동에만 +1이 붙습니다.'
        : (act && F.T[sel][act].b ? '경계라서 제자리로 돌아옵니다. 보상은 0.'
          : (full ? '네 결과를 각각 25%씩 반영해 하나로 합칩니다.'
            : '다음 칸의 값 ' + this.f(rows[s.p] ? rows[s.p].vnext : 0) + ' 을 0.9배로 가져옵니다.'))
    };
    const stack = rows.map((t, i) => ({ w: i < s.p ? Math.round(t.w * 100) : 0, c: TINT[i] }));

    const dots = [0, 1, 2, 3, 4, 5, 6].map((i) => ({ cls: i === (k === 0 ? 0 : 2) ? 'on' : (i < (k === 0 ? 0 : 2) ? 'past' : '') }));

    return {
      dots: dots,
      dotLabel: k === 0 ? '한 칸 계산' : '반복 평가',
      ask: k === 0 ? '이 칸의 값은 어떻게 만들어질까?' : '옆 칸 값이 바뀌면 이 칸은?',
      sub: res.say,
      read: wk - 1, write: wk, cursor: s.cursor,
      cells: cells, arrow: arrow, rows: rowItems, stack: stack, res: res,
      sel: { label: this.lab(sel) },
      seg: { v: s.layer === 'value' ? 'on' : '', p: s.layer === 'policy' ? 'on' : '' },
      strip: ORDER.map((key, i) => {
        const has = i < s.cursor && k < 2;
        return { s: this.lab(key), v: has ? this.f(F.V[k + 1][key]) : '—', cls: has ? (i === s.cursor - 1 ? 'done now' : 'done') : '' };
      }),
      playLabel: s.play ? '일시정지' : '자동 재생',
      liveCls: s.play ? '' : 'hid',
      dis: { prev: k === 0 && s.cursor === 0 && s.p === 0 ? 'off' : '', next: k >= 2 ? 'off' : '' },
      layerValue: () => this.setState({ layer: 'value' }),
      layerPolicy: () => this.setState({ layer: 'policy' }),
      goPlay: () => {
        if (s.play) { this.stop(); this.setState({ play: false }); }
        else { this.setState({ play: true }); this.start(); }
      },
      goNext: () => {
        this.stop();
        if (s.p < 4) this.setState({ play: false, p: s.p + 1, tick: s.tick + 1 });
        else if (s.cursor < 4) this.setState({ play: false, cursor: s.cursor + 1, sel: ORDER[s.cursor + 1], p: 0 });
        else if (k < 2) { this.setState({ play: false }); this.commit(k + 1); }
      },
      goSweep: () => { this.stop(); this.setState({ play: false }); if (k < 2) this.commit(k + 1); },
      goPrev: () => {
        this.stop();
        if (s.p > 0) this.setState({ play: false, p: s.p - 1 });
        else if (s.cursor > 0) this.setState({ play: false, cursor: s.cursor - 1, sel: ORDER[s.cursor - 1], p: 4 });
        else if (k > 0) this.setState({ play: false, k: k - 1, cursor: 4, sel: ORDER[4], p: 4 });
      },
      goReset: () => { this.stop(); this.setState({ k: 0, cursor: 0, p: 0, sel: ORDER[0], play: true }); this.start(); }
    };
  }
}
