class Component extends DCLogic {
  // V2 = example-B values after two evaluation sweeps (spec 8.2). The candidate
  // action values are r + 0.9*V2(s'); (2,0) reproduces the block in spec 7.3.
  FIX() {
    return {
      order: ['0,0', '1,0', '2,0', '0,1', '1,1'],
      goal: '2,1', acts: ['r', 'l', 'u', 'd'],
      V2: { '0,0': 0, '1,0': 0.1125, '2,0': 0.3625, '0,1': 0.05625, '1,1': 0.30625, '2,1': 0 },
      T: {
        '0,0': { r: { s: '1,0' }, l: { s: '0,0', b: 1 }, u: { s: '0,1' }, d: { s: '0,0', b: 1 } },
        '1,0': { r: { s: '2,0' }, l: { s: '0,0' }, u: { s: '1,1' }, d: { s: '1,0', b: 1 } },
        '2,0': { r: { s: '2,0', b: 1 }, l: { s: '1,0' }, u: { s: '2,1', rw: 1, g: 1 }, d: { s: '2,0', b: 1 } },
        '0,1': { r: { s: '1,1' }, l: { s: '0,1', b: 1 }, u: { s: '0,1', b: 1 }, d: { s: '0,0' } },
        '1,1': { r: { s: '2,1', rw: 1, g: 1 }, l: { s: '0,1' }, u: { s: '1,1', b: 1 }, d: { s: '1,0' } }
      }
    };
  }

  st() { return Object.assign({ sel: '2,0', applied: false, play: true, tick: 0, pop: false }, this.state || {}); }
  f(x) { const s = x.toFixed(5).replace(/0+$/, '').replace(/\.$/, ''); return s === '-0' ? '0' : s; }
  lab(k) { const p = k.split(','); return '(' + p[0] + ',' + p[1] + ')'; }
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
  terms(key) {
    const F = this.FIX(), T = F.T[key];
    if (!T) return [];
    return F.acts.map((a) => {
      const t = T[a], rw = t.rw || 0, vn = F.V2[t.s];
      return { a: a, next: t.s, blocked: !!t.b, goal: !!t.g, r: rw, B: rw + 0.9 * vn };
    });
  }
  // spec 7.3: every action within 1e-10 of the max shares the probability
  winners(key) {
    const ts = this.terms(key);
    const top = ts.reduce((m, t) => Math.max(m, t.B), -Infinity);
    return ts.filter((t) => t.B > top - 1e-10).map((t) => t.a);
  }
  candidate(key) {
    const w = this.winners(key), p = {};
    this.FIX().acts.forEach((a) => { p[a] = w.indexOf(a) > -1 ? 1 / w.length : 0; });
    return p;
  }
  uniform() { return { r: .25, l: .25, u: .25, d: .25 }; }
  same(a, b) { return this.FIX().acts.every((k) => Math.abs(a[k] - b[k]) < 1e-10); }

  stop() { if (this.t) { clearInterval(this.t); this.t = null; } }
  componentDidMount() { if (this.st().play) this.start(); }
  componentWillUnmount() { this.stop(); if (this.pt) clearTimeout(this.pt); }
  start() { this.stop(); this.t = setInterval(() => this.beat(), 1500); }
  beat() {
    const s = this.st(), F = this.FIX();
    const i = F.order.indexOf(s.sel);
    if (i < F.order.length - 1) { this.setState({ sel: F.order[i + 1], tick: s.tick + 1 }); return; }
    this.flip(!s.applied, F.order[0]);
  }
  flip(applied, sel) {
    const s = this.st();
    this.setState({ applied: applied, sel: sel || s.sel, pop: true, tick: s.tick + 1 });
    if (this.pt) clearTimeout(this.pt);
    this.pt = setTimeout(() => this.setState({ pop: false }), 560);
  }
  hand(patch) { this.stop(); this.setState(Object.assign({ play: false }, patch)); }

  renderVals() {
    const s = this.st(), F = this.FIX(), ORDER = F.order, self = this;
    const GL = { r: '→', l: '←', u: '↑', d: '↓' };
    const applied = s.applied;
    const sel = F.T[s.sel] ? s.sel : ORDER[0];
    const current = (key) => (applied ? this.candidate(key) : this.uniform());
    const changed = ORDER.filter((key) => !this.same(current(key), this.candidate(key))).length;

    const cells = ORDER.concat([F.goal]).map((key) => {
      const p = key.split(',').map(Number);
      const goal = key === F.goal;
      const w = goal ? [] : this.winners(key);
      const cur = goal ? null : current(key);
      const pct = (a) => (cur[a] === 0 ? '0' : String(Math.round(cur[a] * 100)));
      const cls = [];
      if (goal) cls.push('goal');
      if (key === sel) cls.push('sel');
      if (s.pop && !goal) cls.push('pop');
      return {
        goalCls: goal ? '' : 'hid', stateCls: goal ? 'hid' : '',
        left: p[0] * 202, top: (1 - p[1]) * 202,
        bg: goal ? '#E4F5F0' : this.ramp(F.V2[key]),
        cls: cls.join(' '), value: this.f(F.V2[key]), co: this.lab(key),
        au: goal ? 'hid' : (cur.u === 0 ? 'zero' : ''), ad: goal ? 'hid' : (cur.d === 0 ? 'zero' : ''),
        al: goal ? 'hid' : (cur.l === 0 ? 'zero' : ''), ar: goal ? 'hid' : (cur.r === 0 ? 'zero' : ''),
        pu: goal ? '' : pct('u'), pd: goal ? '' : pct('d'), pl: goal ? '' : pct('l'), pr: goal ? '' : pct('r'),
        chipCls: goal ? 'hid' : (applied ? 'applied' : ''),
        chipGlyph: goal ? '' : w.map((a) => GL[a]).join(''),
        chipText: applied ? '적용' : '후보',
        pick: () => self.hand({ sel: goal ? sel : key })
      };
    });

    const ts = this.terms(sel), win = this.winners(sel);
    const rows = ts.map((t) => ({
      g: GL[t.a],
      next: t.blocked ? '제자리' : (t.goal ? 'G 종료' : this.lab(t.next)),
      r: t.r > 0 ? '+1' : '0', rcls: t.r > 0 ? 'pos' : '',
      b: t.B.toFixed(5), wB: Math.round(t.B * 100),
      cls: win.indexOf(t.a) > -1 ? 'win' : 'dim',
      pick: () => self.hand({ sel: sel })
    }));

    return {
      cells: cells, rows: rows,
      flow: [
        { t: 'π₀ 평가', cls: 'done' }, { t: '→', cls: 'sep' },
        { t: 'π₁ 로 개선', cls: applied ? 'done' : 'now' }, { t: '→', cls: 'sep' },
        { t: 'π₁ 평가', cls: applied ? 'now' : '' }, { t: '→', cls: 'sep' },
        { t: '변경 없을 때까지', cls: '' }
      ],
      ask: applied ? '새 정책도 다시 평가해야 할까?' : '더 좋은 방향은 어느 쪽일까?',
      sub: applied
        ? 'π 만 바뀌었습니다. 지도의 값은 아직 π₀ 를 평가한 V₂ 입니다.'
        : '평가에 쓰던 표를 그대로 두고, 평균 대신 가장 큰 값을 고릅니다.',
      mapNote: applied ? '적용됨 · 값은 재평가 전' : '점선 후보 · 아직 적용 전',
      sel: { label: this.lab(sel), v: this.f(F.V2[sel]) },
      res: {
        pick: win.map((a) => GL[a]).join('') + '  ' + Math.round(100 / win.length) + '%',
        state: applied ? '적용됨 π₁' : '미적용',
        anim: s.tick % 2 ? 'popUpA' : 'popUpB',
        say: win.length > 1
          ? '동점 행동이 ' + win.length + '개라 확률을 똑같이 나눕니다.'
          : '동점은 1e-10 안에서 판정하고, 반올림한 화면 숫자로 고르지 않습니다.'
      },
      diff: ORDER.map((key) => ({
        s: this.lab(key),
        v: this.winners(key).map((a) => GL[a]).join(''),
        cls: this.same(current(key), this.candidate(key)) ? '' : 'done'
      })),
      policyName: applied ? 'π₁' : 'π₀',
      changedCount: changed,
      playLabel: s.play ? '일시정지' : '자동 재생',
      liveCls: s.play ? '' : 'hid',
      applyLabel: applied ? '되돌리기' : '정책 적용',
      goPlay: () => {
        if (s.play) { this.stop(); this.setState({ play: false }); }
        else { this.setState({ play: true }); this.start(); }
      },
      goApply: () => { this.stop(); this.setState({ play: false }); this.flip(!applied, sel); },
      goReset: () => { this.setState({ sel: ORDER[0], applied: false, play: true }); this.start(); }
    };
  }
}
