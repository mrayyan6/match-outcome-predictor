// The report charts. Plain DOM and SVG, nothing clever.

const SVG = "http://www.w3.org/2000/svg";

function svg(tag, attrs = {}, parent) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

// tooltip shared by everything, content always goes in as text

export function tip(event, lines) {
  const t = document.getElementById("tooltip");
  t.replaceChildren(...lines.map((line, i) => (i === 0 ? el("b", null, line) : el("div", null, line))));
  t.hidden = false;
  const box = t.getBoundingClientRect();
  let x = event.clientX + 14;
  let y = event.clientY + 14;
  if (x + box.width > innerWidth - 8) x = event.clientX - box.width - 14;
  if (y + box.height > innerHeight - 8) y = event.clientY - box.height - 14;
  t.style.left = `${Math.max(8, x)}px`;
  t.style.top = `${Math.max(8, y)}px`;
}

export function untip() {
  document.getElementById("tooltip").hidden = true;
}

export function withTip(node, lines) {
  node.addEventListener("pointermove", (e) => tip(e, typeof lines === "function" ? lines() : lines));
  node.addEventListener("pointerleave", untip);
  return node;
}

// three segment bar used in lists and fixture cards
export function miniBar(p, titles) {
  const bar = el("div", "mini-bar");
  ["h", "d", "a"].forEach((cls, i) => {
    const s = el("span", cls);
    s.style.flexGrow = String(p[i]);
    bar.appendChild(s);
  });
  if (titles) withTip(bar, titles);
  return bar;
}

// Log loss for each contender on a shared axis. Every model sits between
// about 0.95 and 1.1, so bars from zero would all look the same: this is a
// dot plot on a labelled, zoomed axis instead.
export function renderBaselines(container, rows) {
  container.replaceChildren();
  const values = rows.map((r) => r.value);
  const lo = Math.floor((Math.min(...values) - 0.01) * 50) / 50;
  const hi = Math.ceil((Math.max(...values) + 0.01) * 50) / 50;
  const width = container.clientWidth || 600;
  const left = Math.min(170, width * 0.34);
  const right = 20;
  const rowH = 40;
  const height = rows.length * rowH + 34;
  const x = (v) => left + ((v - lo) / (hi - lo)) * (width - left - right);

  const s = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Log loss by model, lower is better" }, container);
  s.style.height = `${height}px`;
  const axis = svg("g", { class: "axis" }, s);
  for (let v = lo; v <= hi + 1e-9; v += 0.02) {
    svg("line", { class: "gridline", x1: x(v), x2: x(v), y1: 4, y2: height - 26 }, axis);
    const t = svg("text", { x: x(v), y: height - 8, "text-anchor": "middle" }, axis);
    t.textContent = v.toFixed(2);
  }

  rows.forEach((r, i) => {
    const cy = 22 + i * rowH;
    const g = svg("g", {}, s);
    const name = svg("text", { x: 0, y: cy + 4, fill: "var(--ink)", "font-size": 14, "font-weight": r.current ? 600 : 400 }, g);
    name.textContent = r.label;
    const sub = svg("text", { x: 0, y: cy + 19, fill: "var(--ink-3)", "font-size": 12 }, g);
    sub.textContent = r.sub;
    svg("line", { x1: x(lo), x2: x(r.value), y1: cy, y2: cy, stroke: "var(--line-2)", "stroke-width": 2 }, g);
    svg("circle", { cx: x(r.value), cy, r: r.current ? 8 : 6, fill: r.current ? "var(--ink)" : "var(--draw)", stroke: "var(--card)", "stroke-width": 2 }, g);
    const val = svg("text", { x: x(r.value) + 13, y: cy + 4, fill: "var(--ink-2)", "font-size": 13 }, g);
    val.textContent = r.value.toFixed(3);
    const hit = svg("rect", { x: 0, y: cy - rowH / 2, width, height: rowH, fill: "transparent" }, g);
    withTip(hit, [r.label, `Log loss ${r.value.toFixed(3)}`, r.sub]);
  });
}

// 3x3 table of counts, shaded by the share of each real result
export function renderConfusion(container, cm) {
  container.replaceChildren();
  const labels = ["Home", "Draw", "Away"];
  const grid = el("div", "cm");
  grid.setAttribute("role", "table");
  grid.appendChild(el("div", "head", ""));
  labels.forEach((l) => grid.appendChild(el("div", "head", `Said ${l.toLowerCase()}`)));
  cm.forEach((row, i) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    grid.appendChild(el("div", "rowhead", labels[i]));
    row.forEach((n, j) => {
      const share = n / total;
      const cell = el("div", `cell${i === j ? " diag" : ""}`);
      // one hue ramp: ink mixed into the card colour by share of the row
      cell.style.background = `color-mix(in srgb, var(--ink) ${Math.round(share * 85)}%, var(--card-2))`;
      cell.style.color = share > 0.45 ? "var(--page)" : "var(--ink)";
      cell.append(el("b", null, String(n)), el("small", null, `${Math.round(share * 100)}%`));
      withTip(cell, [`Was ${labels[i].toLowerCase()}, model said ${labels[j].toLowerCase()}`, `${n} games, ${Math.round(share * 100)}% of real ${labels[i].toLowerCase()} results`]);
      grid.appendChild(cell);
    });
  });
  container.appendChild(grid);
}

// predicted probability against how often it happened
export function renderReliability(container, series) {
  container.replaceChildren();
  const legend = el("div", "legend");
  for (const s of series) {
    const item = el("span");
    const key = el("i");
    key.style.borderColor = s.color;
    item.append(key, s.label);
    legend.appendChild(item);
  }
  container.appendChild(legend);

  const width = container.clientWidth || 420;
  const size = Math.min(width, 380);
  const m = { l: 40, r: 10, t: 8, b: 34 };
  const w = size;
  const h = size * 0.85;
  const x = (v) => m.l + v * (w - m.l - m.r);
  const y = (v) => h - m.b - v * (h - m.t - m.b);
  const s = svg("svg", { viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": "Reliability diagram" }, container);
  s.style.height = `${h}px`;
  s.style.maxWidth = `${w}px`;

  const axis = svg("g", { class: "axis" }, s);
  for (let v = 0; v <= 1.0001; v += 0.2) {
    svg("line", { class: "gridline", x1: x(v), x2: x(v), y1: y(0), y2: y(1) }, axis);
    svg("line", { class: "gridline", x1: x(0), x2: x(1), y1: y(v), y2: y(v) }, axis);
    const tx = svg("text", { x: x(v), y: h - m.b + 16, "text-anchor": "middle" }, axis);
    tx.textContent = `${Math.round(v * 100)}%`;
    const ty = svg("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, axis);
    ty.textContent = `${Math.round(v * 100)}%`;
  }
  const xt = svg("text", { x: (x(0) + x(1)) / 2, y: h - 2, "text-anchor": "middle" }, axis);
  xt.textContent = "what it said";
  const yt = svg("text", { x: 11, y: (y(0) + y(1)) / 2, "text-anchor": "middle", transform: `rotate(-90 11 ${(y(0) + y(1)) / 2})` }, axis);
  yt.textContent = "what happened";
  svg("line", { class: "diagonal", x1: x(0), y1: y(0), x2: x(1), y2: y(1) }, s);

  for (const ser of series) {
    const pts = ser.points;
    const d = pts.map((p, i) => `${i ? "L" : "M"}${x(p.predicted).toFixed(1)},${y(p.actual).toFixed(1)}`).join("");
    svg("path", { class: "rel-line", d, stroke: ser.color }, s);
    for (const p of pts) {
      const c = svg("circle", { class: "rel-dot", cx: x(p.predicted), cy: y(p.actual), r: 4.5, fill: ser.color }, s);
      const hit = svg("circle", { cx: x(p.predicted), cy: y(p.actual), r: 12, fill: "transparent" }, s);
      withTip(hit, [ser.label, `Said ${Math.round(p.predicted * 100)}% on average`, `Happened ${Math.round(p.actual * 100)}% of the time`, `${p.n} predictions in this bucket`]);
      void c;
    }
  }
}

// horizontal bars, animated widths, rows keyed so the order can change
export function renderBars(container, rows, fmt) {
  const max = Math.max(...rows.map((r) => r.value), 1e-9);
  const existing = new Map([...container.children].map((n) => [n.dataset.key, n]));
  const frag = document.createDocumentFragment();
  for (const r of rows) {
    let row = existing.get(r.key);
    if (!row) {
      row = el("div", "hbar-row");
      row.dataset.key = r.key;
      const track = el("div", "track");
      track.appendChild(el("div", "fill"));
      row.append(el("span", "name"), track, el("span", "val"));
    }
    row.classList.toggle("current", Boolean(r.current));
    row.querySelector(".name").textContent = r.label;
    row.querySelector(".val").textContent = fmt(r.value);
    requestAnimationFrame(() => {
      row.querySelector(".fill").style.width = `${(Math.max(r.value, 0) / max) * 100}%`;
    });
    frag.appendChild(row);
  }
  container.replaceChildren(frag);
}
