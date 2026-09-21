class Component extends DCLogic {
  // Both maps come from the same table B1(s,a) = r + 0.9*V1(s'). The weighted
  // column reproduces V2 in spec 8.2; the max column is that table with max.
  FIX() {
    return {
      order: ['0,0', '1,0', '2,0', '0,1', '1,1'],
      goal: '2,1', acts: ['r', 'l', 'u', 'd'],
      V1: { '0,0': 0, '1,0': 0, '2,0': 0.25, '0,1': 0, '1,1': 0.25, '2,1': 0 },
      T: {
        '0,0': { r: { s: '1,0' }, l: { s: '0,0', b: 1 }, u: { s: '0,1' }, d: { s: '0,0', b: 1 } },
        '1,0': { r: { s: '2,0' }, l: { s: '0,0' }, u: { s: '1,1' }, d: { s: '1,0', b: 1 } },
        '2,0': { r: { s: '2,0', b: 1 }, l: { s: '1,0' }, u: { s: '2,1', rw: 1, g: 1 }, d: { s: '2,0', b: 1 } },
        '0,1': { r: { s: '1,1' }, l: { s: '0,1', b: 1 }, u: { s: '0,1', b: 1 }, d: { s: '0,0' } },
        '1,1': { r: { s: '2,1', rw: 1, g: 1 }, l: { s: '0,1' }, u: { s: '1,1', b: 1 }, d: { s: '1,0' } }
      }
    };
  }

  st() { return Object.assign({ sel: '2,0', mode: 'pi', play: true, tick: 0 }, this.state || {}); }
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
      const t = T[a], rw = t.rw || 0, vn = F.V1[t.s];
      return { a: a, next: t.s, blocked: !!t.b, goal: !!t.g, r: rw, B: rw + 0.9 * vn, w: 0.25 * (rw + 0.9 * vn) };
    });
  }
  wsum(k) { return this.terms(k).reduce((a, t) => a + t.w, 0); }
  wmax(k) { return this.terms(k).reduce((a, t) => Math.max(a, t.B), -Infinity); }
  winners(k) { const top = this.wmax(k); return this.terms(k).filter((t) => t.B > top - 1e-10).map((t) => t.a); }

  stop() { if (this.t) { clearInterval(this.t); this.t = null; } }
  componentDidMount() { if (this.st().play) this.start(); }
  componentWillUnmount() { this.stop(); }
  start() { this.stop(); this.t = setInterval(() => this.beat(), 2200); }
  beat() { const s = this.st(); this.setState({ mode: s.mode === 'pi' ? 'max' : 'pi', tick: s.tick + 1 }); }
  hand(patch) { this.stop(); this.setState(Object.assign({ play: false }, patch)); }

  renderVals() {
    const s = this.st(), F = this.FIX(), ORDER = F.order, self = this;
    const GL = { r: '→', l: '←', u: '↑', d: '↓' };
    const isMax = s.mode === 'max';
    const sel = F.T[s.sel] ? s.sel : ORDER[0];

    const grid = (valueOf, greedy) => ORDER.concat([F.goal]).map((key) => {
      const p = key.split(',').map(Number);
      const goal = key === F.goal;
      const v = goal ? 0 : valueOf(key);
      const w = goal ? [] : this.winners(key);
      const cls = [];
      if (goal) cls.push('goal');
      if (key === sel) cls.push('sel');
      return {
        left: p[0] * 114, top: (1 - p[1]) * 114,
        bg: goal ? '#E4F5F0' : this.ramp(v),
        cls: cls.join(' '),
        value: goal ? 'G' : this.f(v), co: goal ? '' : this.lab(key),
        gCls: greedy && !goal ? '' : 'hid',
        greedy: greedy && !goal ? w.map((a) => GL[a]).join('') + ' ' + Math.round(100 / w.length) + '%' : '',
        pick: () => self.hand({ sel: goal ? sel : key })
      };
    });

    const ts = this.terms(sel), win = this.winners(sel);
    const rows = ts.map((t) => ({
      g: GL[t.a],
      next: t.blocked ? '제자리' : (t.goal ? 'G 종료' : this.lab(t.next)),
      r: t.r > 0 ? '+1' : '0', rcls: t.r > 0 ? 'pos' : '',
      b: t.B.toFixed(4),
      wB: Math.round(t.B * 100),
      wW: isMax ? 0 : Math.round(t.w * 100),
      cls: isMax ? (win.indexOf(t.a) > -1 ? 'win' : 'dim') : ''
    }));

    const TINT = ['#93B4F5', '#B9CDF8', '#2563EB', '#5D8FEE'];
    const pi = this.wsum(sel), mx = this.wmax(sel), prev = F.V1[sel];
    const val = isMax ? mx : pi;

    return {
      cellsA: grid((k) => this.wsum(k), false),
      cellsB: grid((k) => this.wmax(k), true),
      rows: rows,
      dots: [0, 1, 2, 3, 4, 5, 6].map((i) => ({ cls: i === 5 ? 'on' : (i < 5 ? 'past' : '') })),
      sub: isMax ? '가장 큰 값 하나만 남깁니다.' : '네 결과를 25% 씩 섞습니다.',
      modeName: isMax ? '최댓값' : '평균',
      seg: { pi: isMax ? '' : 'on', max: isMax ? 'on' : '' },
      paneA: isMax ? '' : 'act',
      paneB: isMax ? 'act' : '',
      resCls: isMax ? 'max' : '',
      stack: ts.map((t, i) => ({ w: isMax ? (win.indexOf(t.a) > -1 ? Math.round(t.B * 100) : 0) : Math.round(t.w * 100), c: isMax ? '#7C3AED' : TINT[i] })),
      sel: { label: this.lab(sel), prev: this.f(prev), pi: this.f(pi), max: this.f(mx) },
      res: {
        value: this.f(val),
        label: isMax ? 'maxₐ B₁' : 'Σ π · B₁',
        delta: (val - prev >= 0 ? '+' : '') + this.f(val - prev),
        anim: s.tick % 2 ? 'popUpA' : 'popUpB',
        say: Math.abs(mx - pi) < 1e-12
          ? '이 칸은 두 방식의 결과가 같습니다.'
          : '같은 표에서 나온 두 숫자, 차이 ' + this.f(mx - pi) + '.'
      },
      playLabel: s.play ? '일시정지' : '자동 재생',
      liveCls: s.play ? '' : 'hid',
      setPi: () => this.hand({ mode: 'pi', tick: s.tick + 1 }),
      setMax: () => this.hand({ mode: 'max', tick: s.tick + 1 }),
      toggleMode: () => this.hand({ mode: isMax ? 'pi' : 'max', tick: s.tick + 1 }),
      goPlay: () => {
        if (s.play) { this.stop(); this.setState({ play: false }); }
        else { this.setState({ play: true }); this.start(); }
      },
      goReset: () => { this.setState({ sel: '2,0', mode: 'pi', play: true }); this.start(); }
    };
  }
}
