// Formatting helpers.

// Round three probabilities to whole percentages that still add up to 100
// (largest remainder), so the bar never reads 34 + 33 + 32.
export function percents(p) {
  const raw = p.map((v) => v * 100);
  const out = raw.map(Math.floor);
  let left = 100 - out.reduce((a, b) => a + b, 0);
  const order = raw.map((v, i) => [v - Math.floor(v), i]).sort((a, b) => b[0] - a[0]);
  for (const [, i] of order) {
    if (left <= 0) break;
    out[i] += 1;
    left -= 1;
  }
  return out;
}

export function fairOdds(p) {
  return p > 0 ? (1 / p).toFixed(2) : "n/a";
}

export function kickoff(ts) {
  const d = new Date(ts * 1000);
  const day = d.toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" });
  const time = d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
  return `${day}, ${time}`;
}

export function longDate(iso) {
  return new Date(`${iso}T12:00:00`).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
}

export function shortDate(iso) {
  return new Date(`${iso}T12:00:00`).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "2-digit" });
}

const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
const tweens = new WeakMap();

// count a number up or down instead of snapping
export function tween(el, to, fmt = (v) => `${Math.round(v)}%`, ms = 380) {
  const from = Number(el.dataset.v ?? to);
  el.dataset.v = String(to);
  if (reduce || from === to) {
    el.textContent = fmt(to);
    return;
  }
  cancelAnimationFrame(tweens.get(el));
  const t0 = performance.now();
  const step = (now) => {
    const t = Math.min(1, (now - t0) / ms);
    const e = 1 - (1 - t) ** 3;
    el.textContent = fmt(from + (to - from) * e);
    if (t < 1) tweens.set(el, requestAnimationFrame(step));
  };
  tweens.set(el, requestAnimationFrame(step));
}
