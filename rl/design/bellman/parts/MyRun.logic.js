class Component extends DCLogic {
  // Two run states the screen must handle: what the current publish path
  // produces (values + one representative arrow) and what the data contract in
  // spec 12.2 unlocks. The detailed case replays the (2,0) backup from 8.2.
  DATA() {
    return {
      summary: {
        path: 'runs/policy_iter',
        sub: 'scalar_field · vector_field · metrics 만 기록',
        badge: '요약 보기', badgeCls: '',
        have: 1, items: ['값 스냅숏'],
        color: '#E8A33D',
        map: {
          title: '값 지도만 재생할 수 있습니다',
          body: '셀을 눌러도 계산을 펼칠 수 없습니다.'
        },
        sel: { label: '선택한 칸', value: '계산 기록 없음' },
        empty: {
          cls: '',
          text: '값과 정책만 기록되어 있어 계산 과정은 확인할 수 없습니다.',
          more: 'π 가 대표 행동 하나로 저장돼, 균등 정책도 한 방향처럼 보입니다.'
        },
        detailCls: 'hid',
        status: '설명 모드 비활성'
      },
      trace: {
        path: 'runs/bellman_room   (예시)',
        sub: 'ExperimentConfig · Snapshot · BackupEvent 까지 기록',
        badge: '설명 가능', badgeCls: 'q',
        have: 7, items: ['환경 설정', '값 스냅숏', 'π 분포', '전이 · 보상'],
        color: '#2563EB',
        map: {
          title: '이벤트 단위로 되감을 수 있습니다',
          body: '학습 화면과 같은 계산표가 실행 기록으로 열립니다.'
        },
        sel: { label: 's = (2,0)', value: 'value_before 0.25' },
        empty: { cls: 'hid', text: '', more: '' },
        detailCls: '',
        status: '설명 모드 활성'
      }
    };
  }

  // The recorded backup event, exactly as in spec 8.2 / 12.3.
  EV() {
    return [
      { g: '→', next: '제자리', r: 0, b: '0.2250', B: 0.225, w: 0.05625 },
      { g: '←', next: '(1,0)', r: 0, b: '0.0000', B: 0, w: 0 },
      { g: '↑', next: 'G 종료', r: 1, b: '1.0000', B: 1, w: 0.25 },
      { g: '↓', next: '제자리', r: 0, b: '0.2250', B: 0.225, w: 0.05625 }
    ];
  }

  st() { return Object.assign({ phase: 0, play: true, which: null }, this.state || {}); }

  stop() { if (this.t) { clearInterval(this.t); this.t = null; } }
  componentDidMount() { if (this.st().play) this.start(); }
  componentWillUnmount() { this.stop(); }
  start() { this.stop(); this.t = setInterval(() => { const s = this.st(); this.setState({ phase: (s.phase + 1) % 9 }); }, 900); }
  hand(patch) { this.stop(); this.setState(Object.assign({ play: false }, patch)); }

  renderVals() {
    const s = this.st(), D = this.DATA(), EV = this.EV(), self = this;
    const auto = s.phase <= 4 ? 'trace' : 'summary';
    const which = s.which || auto;
    const cur = D[which];
    const shown = which === 'trace' ? (s.which ? 4 : Math.min(s.phase, 4)) : 0;

    return {
      runs: ['summary', 'trace'].map((key) => ({
        path: D[key].path, sub: D[key].sub,
        badge: D[key].badge, badgeCls: D[key].badgeCls,
        cls: key === which ? 'on' : '',
        pick: () => self.hand({ which: key })
      })),
      phases: [
        { t: 'policy_evaluation', cls: 'now' }, { t: '→', cls: 'sep' },
        { t: 'policy_improvement', cls: '' }, { t: '→', cls: 'sep' },
        { t: 'value_iteration', cls: '' }, { t: '→', cls: 'sep' },
        { t: 'rollout', cls: '' }
      ],
      sub: which === 'trace'
        ? '네 행동의 전이 · 보상까지 기록되면 같은 설명을 열 수 있습니다.'
        : '지금 기록으로는 값 지도까지만 보여줄 수 있습니다.',
      cover: {
        have: cur.have, total: 8,
        pct: Math.round(cur.have / 8 * 100),
        color: cur.color,
        items: cur.items.map((t) => ({ t: t, cls: 'done' }))
      },
      map: cur.map, sel: cur.sel, empty: cur.empty,
      rows: EV.map((t, i) => ({
        g: t.g, next: t.next,
        r: t.r > 0 ? '+1' : '0', rcls: t.r > 0 ? 'pos' : '',
        b: t.b, wB: Math.round(t.B * 100),
        wW: i < shown ? Math.round(t.w * 100) : 0,
        cls: i < shown ? '' : 'dim'
      })),
      detail: {
        cls: cur.detailCls,
        value: shown >= 4 ? '0.3625' : EV.slice(0, shown).reduce((a, t) => a + t.w, 0).toFixed(5).replace(/0+$/, '').replace(/\.$/, ''),
        anim: s.phase % 2 ? 'popUpA' : 'popUpB'
      },
      statusText: cur.status,
      playLabel: s.play ? '일시정지' : '자동 재생',
      liveCls: s.play && !s.which ? '' : 'hid',
      goSwap: () => this.hand({ which: which === 'trace' ? 'summary' : 'trace' }),
      goPlay: () => {
        if (s.play) { this.stop(); this.setState({ play: false }); }
        else { this.setState({ play: true, which: null }); this.start(); }
      }
    };
  }
}
