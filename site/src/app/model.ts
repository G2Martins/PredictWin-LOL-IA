// Inference identical to sklearn's binary LogisticRegression.predict_proba over the one-hot features of
// export_model.py. No imports on purpose: scripts/check.ts runs this file directly in Node (type stripping)
// against fixtures.json, which holds sklearn's own probabilities.

export interface Slot {
  key: string;
  team: 1 | 2;
  role: string;
  coef: Record<string, number>;
  count: Record<string, number>;
}

export interface LolModel {
  slots: Slot[];
  minutesCoef: number;
  intercept: number;
}

export interface Match {
  team1: string[];
  team2: string[];
  minutes: number;
}

export interface Term {
  slot: number;
  champion: string;
  /** coef x value: positive pushes toward a Team 1 win. */
  value: number;
  /** Games where this champion appears in this column (0 = unknown to the model, ignored like handle_unknown='ignore'). */
  games: number;
}

/** One term per slot plus the game-length term (slot = -1). */
export function terms(m: LolModel, x: Match): Term[] {
  const picks = [...x.team1, ...x.team2];
  const out: Term[] = m.slots.map((s, i) => {
    const c = picks[i];
    const known = Object.hasOwn(s.coef, c);
    return { slot: i, champion: c, value: known ? s.coef[c] : 0, games: known ? s.count[c] : 0 };
  });
  out.push({ slot: -1, champion: '', value: m.minutesCoef * x.minutes, games: 0 });
  return out;
}

export function logit(m: LolModel, x: Match): number {
  return terms(m, x).reduce((z, t) => z + t.value, m.intercept);
}

/** P(Team 1 wins) = expit(intercept + sum of coef x value). */
export function predict(m: LolModel, x: Match): number {
  return 1 / (1 + Math.exp(-logit(m, x)));
}
