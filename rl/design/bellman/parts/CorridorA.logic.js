class Component extends DCLogic {
  // Fixtures: REDESIGN_SPEC.md 8.1 (example A, policy = right 100%).
  FIX() {
    return {
      order: ['A', 'B', 'C', 'G'],
      changed: [null, 'C', 'B', 'A', 'A'],
      V: [
        { A: 0, B: 0, C: 0, G: 0 },
        { A: 0, B: 0, C: 1, G: 0 },
        { A: 0, B: 0.9, C: 1, G: 0 },
        { A: 0.81, B: 0.9, C: 1, G: 0 },
        { A: 0.81, B: 0.9, C: 1, G: 0 }
      ],
      T: { A: { s: 'B', r: 0 }, B: { s: 'C', r: 0 }, C: { s: 'G', r: 1, g: 1 } },
      sub: [
        '모든 칸의 초기 가치는 0입니다.',
        '목표로 들어가는 칸만 +1을 받습니다.',
        '그 값의 0.9배가 한 칸 뒤로 옵니다.',
        '또 한 칸 뒤로. 0.9 × 0.9 = 0.81.',
        '더 이상 변하지 않습니다.'
      ]
    };
  }

  st() { return Object.assign({ k: 0, sel: 'C', play: true, tick: 0 }, this.state || {}); }
  f(x) { const s = x.toFixed(5).replace(/0+$/, '').replace(/\.$/, ''); return s === '-0' ? '0' : s; }
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

  stop() { if (this.t) { clearInterval(this.t); this.t = null; } }
  componentDidMount() { if (this.st().play) this.start(); }
  componentWillUnmount() { this.stop(); }
  start() { this.stop(); this.t = setInterval(() => this.beat(), 1500); }
  beat() {
    const s = this.st(), F = this.FIX();
    const nk = s.k >= 4 ? 0 : s.k + 1;
    this.setState({ k: nk, sel: F.changed[nk] || 'C', tick: s.tick + 1 });
  }
  hand(patch) { this.stop(); this.setState(Object.assign({ play: false }, patch)); }

  renderVals() {
    const s = this.st(), F = this.FIX(), ORDER = F.order, self = this;
    const k = s.k, read = Math.max(0, k - 1);
    const Vmap = F.V[k], Vold = F.V[read];
    const sel = s.sel;
    const isGoal = sel === 'G';
    const has = k >= 1 && !isGoal;
    const t = isGoal ? null : F.T[sel];

    const cells = ORDER.map((key, i) => {
      const goal = key === 'G';
      const chg = k >= 1 && Math.abs(F.V[k][key] - F.V[k - 1][key]) > 1e-12;
      const cls = [];
      if (goal) cls.push('goal');
      if (key === sel) cls.push('sel');
      else if (t && t.s === key) cls.push('ref');
      if (chg && key === F.changed[k]) cls.push('pop');
      return {
        goalCls: goal ? '' : 'hid', stateCls: goal ? 'hid' : '',
        left: i * 202, bg: goal ? '#E4F5F0' : this.ramp(Vmap[key]),
        cls: cls.join(' '), value: this.f(Vmap[key]), name: key,
        tag: key === 'A' ? 'S' : '',
        chgCls: chg ? '' : 'hid',
        chg: chg ? '+' + this.f(F.V[k][key] - F.V[k - 1][key]) : '',
        pick: () => self.hand({ sel: key })
      };
    });

    const i = ORDER.indexOf(sel);
    const arrow = { d: '', head: '', len: 1, anim: 'none', rx: -999, r: '', rcls: 'hid' };
    if (!isGoal) {
      const sx = i * 202 + 95 + 80, ex = (i + 1) * 202 + 95 - 80;
      arrow.d = 'M ' + sx + ' 95 L ' + ex + ' 95';
      arrow.head = (ex + 15) + ',95 ' + ex + ',86 ' + ex + ',104';
      arrow.len = ex - sx;
      arrow.anim = s.tick % 2 ? 'drawA' : 'drawB';
      arrow.rx = (sx + ex) / 2;
      arrow.r = t.r > 0 ? 'r +1' : 'r 0';
      arrow.rcls = t.r > 0 ? '' : 'zero';
    }

    let calc = { cls: 'hid', next: '', r: '', rcls: '', b: '', wB: 0, result: '', delta: '', deltaCls: 'hid', anim: 'popUpA' };
    if (has) {
      const vn = Vold[t.s], B = (t.r || 0) + 0.9 * vn;
      const d = F.V[k][sel] - Vold[sel];
      calc = {
        cls: '', next: t.g ? 'G 종료' : t.s,
        r: t.r > 0 ? '+1' : '0', rcls: t.r > 0 ? 'pos' : '',
        b: B.toFixed(4), wB: Math.round(B * 100),
        result: this.f(F.V[k][sel]),
        delta: (d >= 0 ? '+' : '') + this.f(d), deltaCls: Math.abs(d) > 1e-12 ? '' : 'hid',
        anim: s.tick % 2 ? 'popUpA' : 'popUpB'
      };
    }

    return {
      k: k, cells: cells, arrow: arrow, calc: calc,
      dots: [0, 1, 2, 3, 4, 5, 6].map((d) => ({ cls: d === 0 ? 'on' : '' })),
      ask: k === 0 ? '값은 어디서부터 정해질까?' : '+1은 어떻게 뒤로 전달될까?',
      sub: F.sub[k],
      sel: { name: sel, prev: this.f(Vold[sel]) },
      steps: [0, 1, 2, 3, 4].map((d) => ({
        k: d, cls: d === k ? 'done now' : (d < k ? 'done' : ''),
        pick: () => self.hand({ k: d, sel: F.changed[d] || 'C' })
      })),
      empty: {
        cls: has ? 'hid' : '',
        text: isGoal ? '목표는 종료 상태. 값은 항상 0이고 +1은 들어가는 화살표에 붙습니다.' : '아직 계산 전. V₀ = 0.'
      },
      tail: k >= 4 ? '변화 없음 · 설정한 허용오차에서 종료' : '한 바퀴에 한 전이씩 전달됩니다. 모든 환경의 법칙은 아닙니다.',
      playLabel: s.play ? '일시정지' : '자동 재생',
      liveCls: s.play ? '' : 'hid',
      dis: { prev: k === 0 ? 'off' : '', next: k >= 4 ? 'off' : '' },
      goPlay: () => {
        if (s.play) { this.stop(); this.setState({ play: false }); }
        else { this.setState({ play: true }); this.start(); }
      },
      goNext: () => this.hand({ k: Math.min(4, k + 1), sel: F.changed[Math.min(4, k + 1)] || 'C', tick: s.tick + 1 }),
      goPrev: () => this.hand({ k: Math.max(0, k - 1), sel: F.changed[Math.max(0, k - 1)] || 'C', tick: s.tick + 1 }),
      goReset: () => { this.setState({ k: 0, sel: 'C', play: true, tick: s.tick + 1 }); this.start(); }
    };
  }
}
