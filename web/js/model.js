// Runs the exported models in the browser. Output is always
// [home win, draw, away win] probabilities.

function forest(m, x) {
  // sklearn casts inputs to float32 before walking its trees
  const v = x.map(Math.fround);
  const out = [0, 0, 0];
  for (const t of m.trees) {
    let n = 0;
    while (t.f[n] !== -1) n = v[t.f[n]] <= t.t[n] ? t.l[n] : t.r[n];
    const p = t.p[n];
    out[0] += p[0];
    out[1] += p[1];
    out[2] += p[2];
  }
  return out.map((s) => s / m.trees.length);
}

function boosted(m, x) {
  const v = x.map(Math.fround);
  const margin = [...m.intercept];
  m.trees.forEach((t, i) => {
    let n = 0;
    // xgboost goes left when the value is strictly below the split
    while (t.l[n] !== -1) n = v[t.f[n]] < Math.fround(t.t[n]) ? t.l[n] : t.r[n];
    margin[m.classes[i]] += t.t[n];
  });
  const top = Math.max(...margin);
  const e = margin.map((z) => Math.exp(z - top));
  const sum = e[0] + e[1] + e[2];
  return e.map((z) => z / sum);
}

export function predictRaw(m, features) {
  const x = m.features.map((f) => features[f]);
  return m.type === "rf" ? forest(m, x) : boosted(m, x);
}

const EPS = 1e-6;
const logit = (p) => {
  const q = Math.min(Math.max(p, EPS), 1 - EPS);
  return Math.log(q / (1 - q));
};

function interp(x, xs, ys) {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  let lo = 0;
  let hi = xs.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  const span = xs[hi] - xs[lo];
  return span === 0 ? ys[hi] : ys[lo] + ((x - xs[lo]) / span) * (ys[hi] - ys[lo]);
}

// Same maths as apply_calibration in model.py
export function calibrate(p, cal, method) {
  if (method === "raw") return p;
  const q = p.map((pk, k) => {
    if (method === "platt") {
      const { a, b } = cal.platt[k];
      return 1 / (1 + Math.exp(-(a * logit(pk) + b)));
    }
    const { x, y } = cal.isotonic[k];
    return interp(pk, x, y);
  });
  const sum = q[0] + q[1] + q[2];
  return sum > 0 ? q.map((v) => v / Math.max(sum, EPS)) : p;
}

export function predict(model, calibration, method, features) {
  return calibrate(predictRaw(model, features), calibration, method);
}
