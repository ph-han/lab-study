class Component extends DCLogic {
  // pi1 = greedy on V2. Spec 7.3 supplies (2,0) directly; the other four are the
  // argmax of r + 0.9*V2(s') over the same table. V2 itself is spec 8.2.
  FIX() {
    return {
      order: ['0,0', '1,0', '2,0', '0,1', '1,1'],
      goal: '2,1', start: '0,0',
      V2: { '0,0': 0, '1,0': 0.1125, '2,0': 0.3625, '0,1': 0.05625, '1,1': 0.30625, '2,1': 0 },
      pi1: { '0,0': 'r', '1,0': 'r', '2,0': 'u', '0,1': 'r', '1,1': 'r' },
      trace: [
        { s: '0,0', a: 'r', r: 0, n: '1,0' },
        { s: '1,0', a: 'r', r: 0, n: '2,0' },
        { s: '2,0', a: 'u', r: 1, n: '2,1' }
      ]
    };
  }

  st() { return Object.assign({ step: 0, full: false, play: true, tick: 0 }, this.state || {}); }
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

  stop() { if (this.t) { clearInterval(this.t); this.t = null; } }
  componentDidMount() { if (this.st().play) this.start(); }
  componentWillUnmount() { this.stop(); if (this.pt) clearTimeout(this.pt); }
  start() { this.stop(); this.t = setInterval(() => this.beat(), 1150); }
  beat() {
    const s = this.st(), F = this.FIX();
    if (s.step < F.trace.length) { this.setState({ step: s.step + 1, tick: s.tick + 1 }); return; }
    this.stop();
    this.pt = setTimeout(() => { this.setState({ step: 0 }); this.start(); }, 1600);
  }
  hand(patch) { this.stop(); this.setState(Object.assign({ play: false }, patch)); }

  renderVals() {
    const s = this.st(), F = this.FIX(), ORDER = F.order;
    const GL = { r: '→', l: '←', u: '↑', d: '↓' };
    const DIR = { r: 'r', l: 'l', u: 'u', d: 'd' };
    const ROT = { r: 0, l: 180, u: -90, d: 90 };
    const step = s.step, done = step >= F.trace.length;
    const cur = step === 0 ? F.start : F.trace[step - 1].n;

    const cells = ORDER.concat([F.goal]).map((key) => {
      const p = key.split(',').map(Number);
      const goal = key === F.goal;
      const a = F.pi1[key];
      return {
        goalCls: goal ? '' : 'hid', stateCls: goal ? 'hid' : '',
        left: p[0] * 202, top: (1 - p[1]) * 202,
        bg: goal ? '#E4F5F0' : this.ramp(F.V2[key]),
        cls: goal ? 'goal' : '',
        value: this.f(F.V2[key]), co: this.lab(key),
        tag: key === F.start ? 'S' : '',
        dir: goal ? 'r' : DIR[a], glyph: goal ? '' : GL[a]
      };
    });

    const toPath = (list) => list.map((k, i) => {
      const c = this.ctr(k);
      return (i === 0 ? 'M ' : 'L ') + c.x + ' ' + c.y;
    }).join(' ');
    const walked = [F.start].concat(F.trace.slice(0, step).map((t) => t.n));
    const rest = [cur].concat(F.trace.slice(step).map((t) => t.n));
    const path = {
      done: walked.length > 1 ? toPath(walked) : '',
      future: s.full && rest.length > 1 ? toPath(rest) : ''
    };

    const c0 = this.ctr(cur);
    const last = step > 0 ? F.trace[step - 1] : null;
    let reward = { x: -999, y: -999, v: '', cls: 'hid' };
    if (last) {
      const a = this.ctr(last.s), b = this.ctr(last.n);
      reward = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, v: last.r > 0 ? 'r +1' : 'r 0', cls: last.r > 0 ? '' : 'zero' };
    }

    const taken = F.trace.slice(0, step);
    const disc = taken.reduce((acc, t, i) => acc + Math.pow(0.9, i) * t.r, 0);
    const plain = taken.reduce((acc, t) => acc + t.r, 0);

    return {
      cells: cells, path: path, reward: reward, step: step,
      dots: [0, 1, 2, 3, 4, 5, 6].map((i) => ({ cls: i === 6 ? 'on' : 'past' })),
      sub: done ? '목표에 도착했습니다. 이후 보상도 행동도 없습니다.'
        : (step === 0 ? '계산이 끝난 뒤, 같은 전이·보상 함수로 경로를 만듭니다.'
          : '한 걸음마다 실제 보상을 기록합니다.'),
      car: { x: c0.x, y: c0.y, rot: last ? ROT[last.a] : ROT[F.pi1[cur] || 'r'] },
      curLabel: cur === F.goal ? 'G (2,1) 도착' : 's = ' + this.lab(cur),
      seg: { full: s.full ? 'on' : '' },
      emptyCls: step > 0 ? 'hid' : '',
      steps: taken.map((t, i) => ({
        i: i + 1, s: this.lab(t.s), a: GL[t.a], r: t.r > 0 ? '+1' : '0',
        n: t.n === F.goal ? 'G' : this.lab(t.n),
        cls: i === step - 1 ? 'fresh' : ''
      })),
      ret: {
        discounted: this.f(disc), plain: '+' + plain,
        pct: Math.round(disc * 100),
        anim: s.tick % 2 ? 'popUpA' : 'popUpB'
      },
      termCls: done ? 'g' : '',
      termText: done ? '목표 도달로 종료' : '진행 중 · 최대 200',
      playLabel: s.play ? '일시정지' : '재생',
      liveCls: s.play ? '' : 'hid',
      dis: { prev: step === 0 ? 'off' : '', next: done ? 'off' : '' },
      toggleFull: () => this.setState({ full: !s.full }),
      goPrev: () => this.hand({ step: Math.max(0, step - 1), tick: s.tick + 1 }),
      goNext: () => this.hand({ step: Math.min(F.trace.length, step + 1), tick: s.tick + 1 }),
      goReset: () => { this.setState({ step: 0, play: true, tick: s.tick + 1 }); this.start(); },
      goPlay: () => {
        if (s.play) { this.stop(); this.setState({ play: false }); }
        else { this.setState({ play: true }); this.start(); }
      }
    };
  }
}
