import { ChangeDetectionStrategy, Component, afterNextRender, computed, signal } from '@angular/core';
import data from './data/model.json';
import eda from './data/eda.json';
import { LolModel, Match, logit, predict, terms } from './model';

const M = data as unknown as Omit<typeof data, 'slots'> & LolModel;
const E = eda;

export const LINKS = {
  author: 'Gustavo Martins Gripaldi',
  portfolio: 'https://g2portfolio.grovicdata.com/',
  project: 'https://g2portfolio.grovicdata.com/projetos/predictwin-lol-ia/',
  repo: 'https://github.com/G2Martins/PredictWin-LOL-IA',
  csv: 'https://github.com/G2Martins/PredictWin-LOL-IA/blob/main/assets/s11.csv',
};

const ALL = [...new Set(M.slots.flatMap((s) => Object.keys(s.coef)))].sort((a, b) => a.localeCompare(b, 'pt-BR'));
const mostPicked = (i: number) => Object.entries(M.slots[i].count).sort((a, b) => b[1] - a[1])[0][0];

type PresetKey = 'comum' | 'readme' | 'teste' | 'sortear';
const PRESETS: Record<Exclude<PresetKey, 'sortear'>, Match> = {
  comum: { team1: [0, 1, 2, 3, 4].map(mostPicked), team2: [5, 6, 7, 8, 9].map(mostPicked), minutes: 25 },
  // Exact example from main.py (its lists are not in role order, so some picks never appear in that column).
  readme: { team1: ['Elise', 'Darius', 'Yasuo', 'Blitzcrank', 'Ashe'], team2: ['Cassiopeia', 'Draven', 'Aatrox', 'Aurelion Sol', 'Ahri'], minutes: 30 },
  teste: { team1: M.testMatch.team1, team2: M.testMatch.team2, minutes: Math.round(M.testMatch.minutes * 10) / 10 },
};

const nf = (o: Intl.NumberFormatOptions) => new Intl.NumberFormat('pt-BR', o);
const PCT = [0, 1].map((d) => nf({ style: 'percent', minimumFractionDigits: d, maximumFractionDigits: d }));
const DEC = [0, 1, 2, 3, 4].map((d) => nf({ minimumFractionDigits: d, maximumFractionDigits: d }));
const SGN = [0, 1, 2, 3, 4].map((d) => nf({ minimumFractionDigits: d, maximumFractionDigits: d, signDisplay: 'exceptZero' }));
const INT = nf({ maximumFractionDigits: 1 });

const cmp = (x: string | number, y: string | number) =>
  typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y), 'pt-BR');

/** Linear scale helper for the SVG charts. */
const lin = (d0: number, d1: number, r0: number, r1: number) => (v: number) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);

// Decision plane: x = S (sum of the ten champion weights), y = minutes. 600 x 340 plot area.
const BW = 600, BH = 340;
const bnd = M.viz.boundary;
const bx = lin(bnd.s[0], bnd.s[bnd.s.length - 1], 0, BW);
const by = lin(bnd.minutes[0], bnd.minutes[bnd.minutes.length - 1], BH, 0);

@Component({
  selector: 'app-root',
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './app.html',
})
export class App {
  protected readonly links = LINKS;
  protected readonly m = M;
  protected readonly e = E;
  protected readonly year = 2026;
  protected readonly Math = Math;
  protected readonly roles = M.slots.slice(0, 5).map((s) => s.role);
  /** Per slot: champions seen in that CSV column. An unseen pick (README preset) is shown on its own, weight zero. */
  // ponytail: unseen champions are not offered (halves the DOM: ~750 fewer <option>s); only the README preset uses them.
  protected readonly options = M.slots.map((s) => ALL.filter((c) => s.count[c]));
  protected readonly known = (slot: number, c: string) => !!M.slots[slot].count[c];

  // ---------- Simulator ----------
  protected readonly team1 = signal([...PRESETS.comum.team1]);
  protected readonly team2 = signal([...PRESETS.comum.team2]);
  protected readonly minutes = signal(PRESETS.comum.minutes);
  protected readonly preset = signal<PresetKey | ''>('comum');

  private readonly match = computed<Match>(() => ({ team1: this.team1(), team2: this.team2(), minutes: this.minutes() }));
  protected readonly p = computed(() => predict(M, this.match()));
  protected readonly z = computed(() => logit(M, this.match()));
  protected readonly unknown = computed(() => terms(M, this.match()).filter((t) => t.slot >= 0 && !t.games).length);
  protected readonly contrib = computed(() => {
    const ts = terms(M, this.match());
    const max = Math.max(0.5, ...ts.map((t) => Math.abs(t.value)));
    return ts
      .map((t) => ({
        ...t,
        team: t.slot < 0 ? 0 : M.slots[t.slot].team,
        label: t.slot < 0 ? `Duração · ${INT.format(this.minutes())} min` : `${M.slots[t.slot].role} · ${t.champion}`,
        scale: Math.abs(t.value) / max,
      }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  });
  protected readonly top = computed(() => this.contrib().find((t) => t.value !== 0));
  /** Current composition on the decision plane (clamped to the drawn range). */
  protected readonly here = computed(() => {
    const s = this.z() - M.intercept - M.minutesCoef * this.minutes();
    const cs = Math.min(Math.max(s, bnd.s[0]), bnd.s[bnd.s.length - 1]);
    return { x: bx(cs), y: by(this.minutes()), s, clamped: cs !== s };
  });

  // ---------- Dashboard ----------
  protected readonly winRate = E.kpis.victories / E.kpis.rows;
  protected readonly hist = (() => {
    const max = Math.max(...E.lengthHist.map((h) => h.win + h.loss));
    return { max, bins: E.lengthHist.map((h) => ({ ...h, w: h.win / max, l: h.loss / max })) };
  })();
  protected readonly byLength = (() => {
    const max = Math.max(...E.winByLength.map((b) => b.rate));
    return { max, bins: E.winByLength.map((b) => ({ ...b, scale: b.rate / max })), base: this.winRate / max };
  })();
  protected readonly weeks = (() => {
    const max = Math.max(...E.timeline.map((w) => w.n));
    return E.timeline.map((w) => ({ ...w, scale: w.n / max }));
  })();
  protected readonly servers = (() => {
    const max = Math.max(...E.servers.map((s) => s.n));
    return E.servers.map((s) => ({ ...s, scale: s.n / max }));
  })();
  protected readonly picks = E.topPicks.map((r) => {
    const max = Math.max(...r.champions.map((c) => c.n));
    return { ...r, champions: r.champions.map((c) => ({ ...c, scale: c.n / max })) };
  });
  protected readonly perm = (() => {
    const max = Math.max(...E.permutation.items.map((i) => Math.abs(i.mean) + i.std));
    return E.permutation.items.map((i) => ({ ...i, scale: Math.abs(i.mean) / max, err: i.std / max }));
  })();
  protected readonly topCoef = (() => {
    const all = M.slots.flatMap((s) => Object.entries(s.coef).map(([c, v]) => ({ champion: c, role: s.role, team: s.team, v, games: s.count[c] })));
    const top = all.sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 10);
    const max = Math.abs(top[0].v);
    return top.map((t) => ({ ...t, scale: Math.abs(t.v) / max }));
  })();

  // Sortable / filterable sample table (first 200 rows, precomputed).
  protected readonly cols = E.table.columns;
  protected readonly q = signal('');
  protected readonly fResult = signal('');
  protected readonly fServer = signal('');
  protected readonly sortCol = signal(0);
  protected readonly sortDir = signal<1 | -1>(1);
  protected readonly page = signal(0);
  protected readonly pageSize = 15;
  private readonly filtered = computed(() => {
    const q = this.q().trim().toLowerCase();
    const res = this.fResult(), srv = this.fServer(), c = this.sortCol(), d = this.sortDir();
    return E.table.rows
      .filter((r) => (!res || r[14] === res) && (!srv || r[1] === srv) && (!q || r.slice(3, 13).some((x) => String(x).toLowerCase().includes(q))))
      .sort((a, b) => d * cmp(a[c], b[c]));
  });
  protected readonly total = computed(() => this.filtered().length);
  protected readonly pages = computed(() => Math.max(1, Math.ceil(this.total() / this.pageSize)));
  protected readonly rows = computed(() => {
    const p = Math.min(this.page(), this.pages() - 1);
    return this.filtered().slice(p * this.pageSize, (p + 1) * this.pageSize);
  });

  // ---------- Model ----------
  protected readonly matrices = (['test', 'train'] as const).map((k) => {
    const c = M.metrics[k].confusion;
    return {
      key: k,
      title: k === 'test' ? 'Teste' : 'Treino',
      n: M.metrics[k].n,
      rows: ['Derrota', 'Vitória'].map((label, r) => {
        const total = c[r][0] + c[r][1];
        return { label, cells: c[r].map((v, col) => ({ v, share: v / total, pred: col ? 'Vitória' : 'Derrota' })) };
      }),
    };
  });
  protected readonly perClass = (['Derrota', 'Vitória'] as const).map((k) => ({ label: k, ...M.metrics.test.perClass[k] }));
  // ROC and calibration share a 0..1 square drawn in a 300 x 300 box.
  protected readonly sq = (v: number) => v * 300;
  protected readonly rocPath = (pts: number[][]) => pts.map(([x, y], i) => `${i ? 'L' : 'M'}${(x * 300).toFixed(1)} ${(300 - y * 300).toFixed(1)}`).join(' ');
  protected readonly calibPath = M.viz.calibration.map((c, i) => `${i ? 'L' : 'M'}${(c.pred * 300).toFixed(1)} ${(300 - c.obs * 300).toFixed(1)}`).join(' ');
  protected readonly plane = (() => {
    const r1 = (v: number) => Math.round(v * 10) / 10;
    const cells = bnd.p.flatMap((row, j) =>
      row.map((p, i) => ({
        x: r1(bx(bnd.s[i])), y: r1(by(bnd.minutes[j + 1])),
        w: r1(bx(bnd.s[i + 1]) - bx(bnd.s[i]) + 0.6), h: r1(by(bnd.minutes[j]) - by(bnd.minutes[j + 1]) + 0.6),
        t1: p >= 0.5, o: Math.round(Math.abs(p - 0.5) * 2 * 42) / 100,
      })),
    );
    const points = bnd.points.map(([s, mi, y]) => ({ x: r1(bx(s)), y: r1(by(mi)), win: y === 1, s, mi }));
    return {
      w: BW,
      h: BH,
      cells1: cells.filter((c) => c.t1),
      cells2: cells.filter((c) => !c.t1),
      wins: points.filter((p) => p.win),
      losses: points.filter((p) => !p.win),
      // p = 0.5 where intercept + S + beta * minutes = 0.
      line: [8, 50].map((mi) => ({ x: r1(bx(-M.intercept - M.minutesCoef * mi)), y: r1(by(mi)) })),
      xTicks: [-4, -2, 0, 2, 4].map((v) => ({ v, x: bx(v) })),
      yTicks: [10, 20, 30, 40, 50].map((v) => ({ v, y: by(v) })),
    };
  })();

  protected readonly credits = ['Caio Silveira Guimarães', 'Gustavo Martins Gripaldi', 'Giovanna Monteiro Bispo', 'Mariana Monteiro Bispo', 'Mayron Malaquias Oliveira'];

  protected pct = (v: number, d = 1) => PCT[d].format(v);
  protected dec = (v: number, d = 2) => DEC[d].format(v);
  protected sgn = (v: number, d = 2) => SGN[d].format(v);
  protected int = (v: number) => INT.format(v);
  protected kb = (b: number) => (b > 1e6 ? `${DEC[1].format(b / 1e6)} MB` : `${DEC[0].format(Math.max(1, b / 1e3))} KB`);
  protected dateBr = (s: string) => s.slice(0, 10).split('-').reverse().join('/');

  constructor() {
    afterNextRender(() => {
      document.documentElement.classList.add('reveal-ready');
      const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && (e.target.classList.add('in'), io.unobserve(e.target))), { rootMargin: '0px 0px -8% 0px' });
      document.querySelectorAll('[data-reveal]').forEach((el) => io.observe(el));
    });
  }

  protected pick(team: 1 | 2, i: number, value: string) {
    (team === 1 ? this.team1 : this.team2).update((t) => t.map((c, j) => (j === i ? value : c)));
    this.preset.set('');
  }

  protected setMinutes(v: string) {
    this.minutes.set(Number(v));
    this.preset.set('');
  }

  protected load(key: PresetKey) {
    const x = key === 'sortear' ? this.random() : PRESETS[key];
    this.team1.set([...x.team1]);
    this.team2.set([...x.team2]);
    this.minutes.set(x.minutes);
    this.preset.set(key);
  }

  private random(): Match {
    const any = (i: number) => { const o = this.options[i]; return o[Math.floor(Math.random() * o.length)]; };
    return { team1: [0, 1, 2, 3, 4].map(any), team2: [5, 6, 7, 8, 9].map(any), minutes: 15 + Math.round(Math.random() * 250) / 10 };
  }

  protected sort(c: number) {
    if (this.sortCol() === c) this.sortDir.update((d) => (d === 1 ? -1 : 1));
    else { this.sortCol.set(c); this.sortDir.set(1); }
    this.page.set(0);
  }

  protected filter(sig: { set(v: string): void }, v: string) {
    sig.set(v);
    this.page.set(0);
  }
}
