import { predict } from "./model.js";
import { percents, fairOdds, kickoff, longDate, shortDate, tween } from "./format.js";
import { miniBar, withTip, renderBaselines, renderConfusion, renderReliability, renderBars, untip } from "./charts.js";

const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

const KNOBS = [
  { key: "form", label: "Points from last five", min: 0, max: 15, step: 1, fmt: (v) => `${Math.round(v)}` },
  { key: "gf", label: "Goals scored per game", min: 0, max: 4, step: 0.1, fmt: (v) => v.toFixed(1) },
  { key: "ga", label: "Goals conceded per game", min: 0, max: 4, step: 0.1, fmt: (v) => v.toFixed(1) },
  { key: "sotf", label: "Shots on target per game", min: 0, max: 10, step: 0.1, fmt: (v) => v.toFixed(1) },
  { key: "poss", label: "Possession", min: 30, max: 72, step: 1, fmt: (v) => `${Math.round(v)}%` },
  { key: "elo", label: "Elo rating", min: 1300, max: 1850, step: 5, fmt: (v) => `${Math.round(v)}` },
];

const CAL_NOTE = {
  raw: "Straight out of a model trained with balanced class weights. It calls draws, but its draw percentages run high.",
  platt: "Platt scaling: one logistic curve per outcome, fitted on seasons the model hadn't seen. Honest percentages, hardly any draws called.",
  isotonic: "Isotonic regression: a free-form staircase per outcome instead of a curve. Can bend to noise with this little data.",
};
const MODEL_NAME = { xgb: "XGBoost", rf: "Random forest" };

const state = { league: null, model: "xgb", cal: "platt", home: null, away: null, fixture: null, tweaks: { h: {}, a: {} } };
let meta;
const data = {};

// loading

async function json(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

async function loadLeague(league) {
  const key = meta.leagues[league].key;
  if (!data[league]) {
    const [xgb, calibration, teams, fixtures, results] = await Promise.all(
      ["xgb", "calibration", "teams", "fixtures", "results"].map((f) => json(`data/${key}/${f}.json`))
    );
    data[league] = { xgb, calibration, teams, fixtures, results };
  }
  if (state.model === "rf" && !data[league].rf) data[league].rf = await json(`data/${key}/rf.json`);
  return data[league];
}

const L = () => data[state.league];
const team = (name) => L().teams.teams[name];
const display = (name) => team(name)?.display ?? name;

// features

function derive(f) {
  f.gd_gap = f.h_gf - f.h_ga - (f.a_gf - f.a_ga);
  f.form_gap = f.h_form - f.a_form;
  f.elo_diff = f.h_elo - f.a_elo;
  return f;
}

function snapshotFeatures(home, away) {
  const T = L().teams;
  const h = T.teams[home];
  const a = T.teams[away];
  const f = { h2h: T.h2h[`${home}|${away}`] ?? 1 / 3, home_adv_team: h.home_adv, home_adv_league: T.homeAdvLeague };
  for (const k of ["gf", "ga", "sotf", "sota", "poss", "form", "elo"]) {
    f[`h_${k}`] = h[k];
    f[`a_${k}`] = a[k];
  }
  return derive(f);
}

function baseFeatures() {
  const fx = state.fixture != null ? L().fixtures[state.fixture] : null;
  if (fx && fx.home === state.home && fx.away === state.away) return { ...fx.features };
  return snapshotFeatures(state.home, state.away);
}

function currentFeatures() {
  const f = baseFeatures();
  for (const side of ["h", "a"]) {
    for (const [k, v] of Object.entries(state.tweaks[side])) f[`${side}_${k}`] = v;
  }
  return derive(f);
}

const tweaked = () => Object.keys(state.tweaks.h).length + Object.keys(state.tweaks.a).length > 0;

function probs(features) {
  const d = L();
  return predict(d[state.model], d.calibration[state.model], state.cal, features);
}

// controls

function setPressed(container, attr, value) {
  for (const b of container.querySelectorAll("button")) b.setAttribute(attr, String(b.dataset.value === value));
}

function renderSettings() {
  setPressed($("#league-switch"), "aria-pressed", state.league);
  setPressed($("#model-toggle"), "aria-checked", state.model);
  setPressed($("#cal-toggle"), "aria-checked", state.cal);
  $("#cal-note").textContent = CAL_NOTE[state.cal];
  const facts = $("#facts");
  facts.replaceChildren();
  const r = meta.report[state.league];
  const rows = [
    ["Form as of", longDate(meta.exported)],
    ["Trained on", `${meta.seasons.train[0]} to ${meta.seasons.train.at(-1)}`],
    ["Tested on", `${meta.seasons.test}, ${r[state.model][state.cal].n} games`],
    ["This season", `${meta.seasons.live}, ${r[state.model][state.cal].live?.n ?? 0} played`],
  ];
  for (const [k, v] of rows) facts.append(el("dt", null, k), el("dd", null, v));
}

// fixtures

function renderFixtures() {
  const rail = $("#fixtures");
  rail.replaceChildren();
  const fixtures = L().fixtures;
  $("#fixtures-note").textContent = fixtures.length ? "Click one to load it below" : "";
  if (!fixtures.length) {
    rail.appendChild(el("p", "muted", "No fixtures coming up. Pick any two teams below."));
    return;
  }
  fixtures.forEach((fx, i) => {
    const p = probs(fx.features);
    const pc = percents(p);
    const card = el("button", "fixture");
    card.type = "button";
    card.setAttribute("role", "listitem");
    card.setAttribute("aria-pressed", String(state.fixture === i && state.home === fx.home && state.away === fx.away));
    card.appendChild(el("span", "when", kickoff(fx.kickoff)));
    for (const [name, v] of [[fx.home, pc[0]], [fx.away, pc[2]]]) {
      const row = el("div", "side");
      row.append(el("span", null, display(name)), el("span", null, `${v}%`));
      card.appendChild(row);
    }
    card.appendChild(miniBar(p, [`${display(fx.home)} v ${display(fx.away)}`, `Home ${pc[0]}%, draw ${pc[1]}%, away ${pc[2]}%`]));
    card.addEventListener("click", () => {
      state.fixture = i;
      state.home = fx.home;
      state.away = fx.away;
      state.tweaks = { h: {}, a: {} };
      renderMatch();
      renderFixtures();
    });
    rail.appendChild(card);
  });
}

// the matchup

function fillSelects() {
  const names = Object.keys(L().teams.teams).sort((a, b) => display(a).localeCompare(display(b)));
  for (const [sel, value] of [[$("#home-select"), state.home], [$("#away-select"), state.away]]) {
    sel.replaceChildren(...names.map((n) => {
      const o = el("option", null, display(n));
      o.value = n;
      return o;
    }));
    sel.value = value;
  }
}

function confidence(p, names) {
  const order = [0, 1, 2].sort((a, b) => p[b] - p[a]);
  const [top, second] = order;
  const gap = p[top] - p[second];
  const colour = ["var(--home)", "var(--draw)", "var(--away)"][top];
  let text;
  let pips;
  if (top === 1) {
    text = "Leaning towards a draw";
    pips = gap > 0.08 ? 2 : 1;
  } else if (p[top] >= 0.6) {
    text = `${names[top]}, strong favourites`;
    pips = 3;
  } else if (gap >= 0.12) {
    text = `${names[top]}, favourites`;
    pips = 2;
  } else {
    text = "Too close to call";
    pips = 1;
  }
  const wrap = el("span", "pips");
  for (let i = 0; i < 3; i++) {
    const pip = el("i", i < pips ? "on" : "");
    if (i < pips) pip.style.background = colour;
    wrap.appendChild(pip);
  }
  return [wrap, el("span", null, text)];
}

function renderMatch() {
  const f = currentFeatures();
  const p = probs(f);
  const pc = percents(p);
  const names = [display(state.home), "Draw", display(state.away)];

  fillSelects();
  tween($("#pct-home"), pc[0]);
  tween($("#pct-draw"), pc[1]);
  tween($("#pct-away"), pc[2]);
  $("#who-home").textContent = `${names[0]} win`;
  $("#who-away").textContent = `${names[2]} win`;

  const segs = $("#bar").children;
  p.forEach((v, i) => (segs[i].style.flexGrow = String(v)));
  $("#bar").setAttribute("aria-label", `${names[0]} ${pc[0]}%, draw ${pc[1]}%, ${names[2]} ${pc[2]}%`);

  $("#confidence").replaceChildren(...confidence(p, names));
  const odds = $("#fair-odds");
  odds.replaceChildren("Fair odds ");
  p.forEach((v, i) => {
    odds.append(el("b", null, fairOdds(v)));
    if (i < 2) odds.append(" / ");
  });

  const fx = state.fixture != null ? L().fixtures[state.fixture] : null;
  $("#kickoff").textContent =
    fx && fx.home === state.home && fx.away === state.away
      ? `${kickoff(fx.kickoff)} your time, matchweek ${fx.round}`
      : `Made-up fixture: played today, with form as of ${longDate(meta.exported)}`;

  const note = $("#tweak-note");
  if (tweaked()) {
    const base = percents(probs(baseFeatures()));
    note.hidden = false;
    note.textContent = `With the real numbers it was ${base[0]}% / ${base[1]}% / ${base[2]}%.`;
  } else {
    note.hidden = true;
  }
  $("#reset-all").hidden = !tweaked();

  renderKnobs(f);
  renderTape(f);
  renderForm();
  renderH2H();
}

function renderKnobs(f) {
  const base = baseFeatures();
  for (const side of ["h", "a"]) {
    const col = $(side === "h" ? "#tweak-home" : "#tweak-away");
    const name = side === "h" ? state.home : state.away;
    const tone = side === "h" ? "var(--home)" : "var(--away)";
    // rebuild only when the team changes, so a slider being dragged keeps focus
    if (col.dataset.team !== name) {
      col.dataset.team = name;
      col.replaceChildren(el("h3", null, display(name)));
      for (const k of KNOBS) {
        const id = `knob-${side}-${k.key}`;
        const row = el("div", "knob");
        row.dataset.key = k.key;
        const label = el("label", null, k.label);
        label.htmlFor = id;
        const out = el("output");
        const track = el("div", "track");
        const real = el("span", "real");
        const input = el("input");
        Object.assign(input, { type: "range", id, min: k.min, max: k.max, step: k.step });
        input.style.setProperty("--tone", tone);
        input.addEventListener("input", () => {
          state.tweaks[side][k.key] = Number(input.value);
          renderMatch();
        });
        track.append(real, input);
        row.append(label, out, track);
        col.appendChild(row);
      }
    }
    for (const k of KNOBS) {
      const row = col.querySelector(`[data-key="${k.key}"]`);
      const input = row.querySelector("input");
      const value = f[`${side}_${k.key}`];
      const real = base[`${side}_${k.key}`];
      if (document.activeElement !== input) input.value = value;
      const pct = (v) => ((Math.min(Math.max(v, k.min), k.max) - k.min) / (k.max - k.min)) * 100;
      input.style.setProperty("--fill", `${pct(value)}%`);
      row.querySelector(".real").style.left = `calc(${pct(real)}% + ${(0.5 - pct(real) / 100) * 18}px)`;
      const changed = k.key in state.tweaks[side];
      row.classList.toggle("changed", changed);
      const out = row.querySelector("output");
      out.replaceChildren(k.fmt(value));
      if (changed) out.appendChild(el("small", null, `was ${k.fmt(real)}`));
    }
  }
}

// what the model sees, side by side
function renderTape(f) {
  const rows = [
    ["Form", "h_form", "a_form", true, (v) => v.toFixed(0)],
    ["Scored", "h_gf", "a_gf", true, (v) => v.toFixed(1)],
    ["Conceded", "h_ga", "a_ga", false, (v) => v.toFixed(1)],
    ["On target", "h_sotf", "a_sotf", true, (v) => v.toFixed(1)],
    ["Faced", "h_sota", "a_sota", false, (v) => v.toFixed(1)],
    ["Possession", "h_poss", "a_poss", true, (v) => `${v.toFixed(0)}%`],
    ["Elo", "h_elo", "a_elo", true, (v) => v.toFixed(0)],
  ];
  const tape = $("#tape");
  tape.replaceChildren();
  for (const [label, hk, ak, higher, fmt] of rows) {
    const hv = f[hk];
    const av = f[ak];
    // Elo differences are small next to the rating itself, so measure from 1300
    const floor = hk === "h_elo" ? 1300 : 0;
    const top = Math.max(hv - floor, av - floor, 1e-9);
    const row = el("div", "tape-row");
    if (hv !== av) row.classList.add((hv > av) === higher ? "better-home" : "better-away");
    const left = el("div", "lane left");
    const lspan = el("span");
    lspan.style.width = `${((hv - floor) / top) * 100}%`;
    left.appendChild(lspan);
    const right = el("div", "lane right");
    const rspan = el("span");
    rspan.style.width = `${((av - floor) / top) * 100}%`;
    right.appendChild(rspan);
    row.append(el("span", "v", fmt(hv)), left, el("span", "label", label), right, el("span", "v r", fmt(av)));
    tape.appendChild(row);
  }
  const extra = el("p", "muted");
  extra.style.marginTop = "8px";
  extra.textContent = `Averages over the last five league games. ${display(state.home)}'s home advantage: ${f.home_adv_team >= 0 ? "+" : ""}${f.home_adv_team.toFixed(2)} points a game better at home than away.`;
  tape.appendChild(extra);
}

function renderForm() {
  const wrap = $("#form-guide");
  wrap.replaceChildren();
  for (const name of [state.home, state.away]) {
    const box = el("div", "form-team");
    box.appendChild(el("h3", null, display(name)));
    const row = el("div", "form-row");
    const last = team(name).last;
    if (!last.length) row.appendChild(el("span", "chip none", "No league games in the last year"));
    for (const g of [...last].reverse()) {
      const r = g.gf > g.ga ? "W" : g.gf === g.ga ? "D" : "L";
      const chip = el("span", `chip ${r}`, r);
      const opp = display(g.opp);
      withTip(chip, [`${r === "W" ? "Won" : r === "D" ? "Drew" : "Lost"} ${g.gf}-${g.ga}`, `${g.venue === "H" ? "v" : "at"} ${opp}`, shortDate(g.date)]);
      row.appendChild(chip);
    }
    box.appendChild(row);
    wrap.appendChild(box);
  }
}

function renderH2H() {
  const wrap = $("#h2h");
  wrap.replaceChildren();
  const [a, b] = [state.home, state.away].sort();
  const games = L().teams.meetings[`${a}|${b}`] ?? [];
  if (!games.length) {
    wrap.appendChild(el("p", "muted", "They haven't met in the league in the last three seasons, so this counts as neutral."));
    return;
  }
  const tally = { [state.home]: 0, [state.away]: 0, draw: 0 };
  for (const g of games) tally[g.winner ?? "draw"] += 1;
  wrap.appendChild(
    el("p", "muted", `${display(state.home)} ${tally[state.home]} wins, ${tally.draw} draws, ${display(state.away)} ${tally[state.away]} wins.`)
  );
  const list = el("div", "h2h");
  for (const g of [...games].reverse()) {
    const m = el("div", "meeting");
    m.append(el("b", null, g.winner ? `${display(g.winner)} won` : "Draw"), el("span", null, shortDate(g.date)));
    list.appendChild(m);
  }
  wrap.appendChild(list);
}

// report

function renderReport() {
  const r = meta.report[state.league];
  const s = r[state.model][state.cal];
  const book = r.baselines.bookmakers;
  const home = r.baselines.always_home;
  $("#report-lede").textContent = `${MODEL_NAME[state.model]} with ${state.cal === "raw" ? "no calibration" : `${state.cal} calibration`}, tested on all ${s.n} games of ${meta.seasons.test}, a season it never saw. Trained on ${meta.seasons.train[0]} to ${meta.seasons.train.at(-1)}.`;

  const tiles = [
    ["Called right", `${Math.round(s.accuracy * 100)}%`, `bookmakers ${Math.round(book.accuracy * 100)}%, always home ${Math.round(home.accuracy * 100)}%`],
    ["Log loss", s.log_loss.toFixed(3), `bookmakers ${book.log_loss.toFixed(3)}, lower is better`],
    ["Draws caught", `${Math.round(s.recall[1] * 100)}%`, "of the games that really were draws"],
    ["Draws called", `${Math.round(s.predicted_share[1] * 100)}%`, `of all games, real share ${Math.round((s.confusion[1].reduce((a, b) => a + b, 0) / s.n) * 100)}%`],
  ];
  $("#tiles").replaceChildren(
    ...tiles.map(([label, value, note]) => {
      const t = el("div", "tile");
      t.append(el("span", "tile-label", label), el("span", "tile-value", value), el("span", "tile-note", note));
      return t;
    })
  );

  const rows = [
    { label: "Always home", value: home.log_loss, sub: `${Math.round(home.accuracy * 100)}% right` },
    ...["rf", "xgb"].map((k) => ({
      label: MODEL_NAME[k],
      value: r[k][state.cal].log_loss,
      sub: `${Math.round(r[k][state.cal].accuracy * 100)}% right`,
      current: k === state.model,
    })),
    { label: "Bookmakers", value: book.log_loss, sub: `${Math.round(book.accuracy * 100)}% right` },
  ];
  renderBaselines($("#baselines"), rows);
  renderConfusion($("#confusion"), s.confusion);
  renderReliability($("#reliability"), [
    { label: `${MODEL_NAME[state.model]}, ${state.cal}`, color: "var(--ink)", points: s.reliability },
    { label: "Bookmakers", color: "var(--draw)", points: book.reliability },
  ]);

  const imp = r.importance[state.model];
  const all = Object.entries(imp)
    .map(([k, v]) => ({ key: k, label: meta.featureNames[k], value: v }))
    .sort((a, b) => b.value - a.value);
  // a long tail of zeros tells you nothing, so only the ones that matter get a bar
  const shown = all.filter((b, i) => b.value >= 0.0005 || i < 3);
  const rest = all.slice(shown.length);
  renderBars($("#importance"), shown, (v) => (v < 0.0005 ? "~0" : v.toFixed(3)));
  $("#importance-rest").textContent = rest.length
    ? `The other ${rest.length} make next to no difference once these are known: ${rest.map((b) => b.label.toLowerCase()).join(", ")}.`
    : "";

  renderResultsList();
}

function outcomeIndex(result) {
  return { H: 0, D: 1, A: 2 }[result];
}

function resultItem(g, p, sub) {
  const li = el("li");
  const game = el("div", "game");
  const title = el("b", null, `${display(g.home)} ${g.score[0]}-${g.score[1]} ${display(g.away)}`);
  game.append(title, el("small", null, sub));
  const pc = percents(p);
  const called = p.indexOf(Math.max(...p)) === outcomeIndex(g.result);
  const mark = el("span", `mark ${called ? "hit" : "miss"}`, called ? "✓" : "✗");
  mark.setAttribute("aria-label", called ? "called it" : "missed it");
  li.append(game, miniBar(p, [`${display(g.home)} v ${display(g.away)}`, `Model had home ${pc[0]}%, draw ${pc[1]}%, away ${pc[2]}%`]), mark);
  return li;
}

function renderResultsList() {
  const key = `${state.model}_${state.cal}`;
  const res = L().results;
  const live = [...(res.live ?? [])].reverse();
  const hits = live.filter((g) => {
    const p = g.p[key];
    return p.indexOf(Math.max(...p)) === outcomeIndex(g.result);
  }).length;
  $("#live-note").textContent = live.length
    ? `${hits} of ${live.length} called right so far (${Math.round((hits / live.length) * 100)}%). Latest first.`
    : "Nothing played yet this season.";
  $("#live").replaceChildren(...live.slice(0, 12).map((g) => resultItem(g, g.p[key], shortDate(g.date))));

  const shocks = (res.test ?? [])
    .map((g) => ({ g, chance: g.p[key][outcomeIndex(g.result)] }))
    .sort((a, b) => a.chance - b.chance)
    .slice(0, 8);
  $("#upsets").replaceChildren(
    ...shocks.map(({ g, chance }) => resultItem(g, g.p[key], `${shortDate(g.date)}, model gave it ${Math.round(chance * 100)}%`))
  );
}

// wiring

function renderAll() {
  renderSettings();
  renderFixtures();
  renderMatch();
  renderReport();
  try {
    history.replaceState(null, "", `#${meta.leagues[state.league].key}/${state.model}/${state.cal}`);
  } catch (e) {
    // file:// or sandboxed, not a big deal
  }
}

async function setLeague(league) {
  state.league = league;
  await loadLeague(league);
  const fx = L().fixtures;
  const names = Object.keys(L().teams.teams);
  if (fx.length) {
    state.fixture = 0;
    state.home = fx[0].home;
    state.away = fx[0].away;
  } else {
    state.fixture = null;
    [state.home, state.away] = names;
  }
  state.tweaks = { h: {}, a: {} };
  for (const col of document.querySelectorAll(".tweak-col")) col.dataset.team = "";
  renderAll();
}

function wire() {
  const sw = $("#league-switch");
  for (const league of Object.keys(meta.leagues)) {
    const b = el("button", null, league);
    b.type = "button";
    b.dataset.value = league;
    b.addEventListener("click", () => setLeague(league));
    sw.appendChild(b);
  }
  for (const b of document.querySelectorAll("#model-toggle button")) {
    b.addEventListener("click", async () => {
      state.model = b.dataset.value;
      await loadLeague(state.league);
      renderAll();
    });
  }
  for (const b of document.querySelectorAll("#cal-toggle button")) {
    b.addEventListener("click", () => {
      state.cal = b.dataset.value;
      renderAll();
    });
  }
  $("#home-select").addEventListener("change", (e) => pickTeams(e.target.value, state.away));
  $("#away-select").addEventListener("change", (e) => pickTeams(state.home, e.target.value));
  $("#swap").addEventListener("click", () => pickTeams(state.away, state.home));
  $("#reset-all").addEventListener("click", () => {
    state.tweaks = { h: {}, a: {} };
    renderMatch();
  });

  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const show = (tab) => {
    for (const t of tabs) {
      const on = t === tab;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
    }
    untip();
    if (tab.id === "tab-report") renderReport(); // charts measure their width
  };
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => show(t));
    t.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
      show(next);
      next.focus();
    });
  });

  $("#theme-toggle").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try {
      localStorage.setItem("theme", root.dataset.theme);
    } catch (e) {
      // private window, fine for this visit
    }
  });

  let resizeTimer;
  addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => !$("#panel-report").hidden && renderReport(), 150);
  });
}

function pickTeams(home, away) {
  if (home === away) {
    // picking the same side twice just swaps them round
    [home, away] = [state.away, state.home];
  }
  state.home = home;
  state.away = away;
  const i = L().fixtures.findIndex((f) => f.home === home && f.away === away);
  state.fixture = i >= 0 ? i : null;
  state.tweaks = { h: {}, a: {} };
  renderMatch();
  renderFixtures();
}

async function init() {
  meta = await json("data/meta.json");
  wire();
  const [key, model, cal] = location.hash.slice(1).split("/");
  if (model === "xgb" || model === "rf") state.model = model;
  if (["raw", "platt", "isotonic"].includes(cal)) state.cal = cal;
  const league = Object.keys(meta.leagues).find((l) => meta.leagues[l].key === key) ?? Object.keys(meta.leagues)[0];
  $("#updated").textContent = `Numbers last rebuilt ${longDate(meta.exported)}.`;
  await setLeague(league);
}

init().catch((err) => {
  console.error(err);
  document.querySelector(".kickoff").textContent = "Couldn't load the data. Try a refresh.";
});
