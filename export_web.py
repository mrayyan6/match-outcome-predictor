"""Everything the site needs, as JSON under web/data/.

    <league>/rf.json          random forest trees, leaf values are class probabilities
    <league>/xgb.json         xgboost trees plus a per class starting margin
    <league>/calibration.json Platt and isotonic maps for both models
    <league>/teams.json       every current team's form as of today (for the simulator)
    <league>/fixtures.json    the next games with their features
    <league>/results.json     predictions for last season and this season so far
    report.json               metrics, confusion matrices, importance, reliability

    python export_web.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from data_loader import FEATURES, LEAGUES, PROCESSED, season_label
from model import FEATURE_NAMES, MODELS, REPORTS, apply_calibration, seasons, slug

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "web" / "data"


def dump(path: Path, payload, indent=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":") if indent is None else None, indent=indent), encoding="utf-8")


def export_forest(model) -> dict:
    trees = []
    for est in model.estimators_:
        t = est.tree_
        leaf = t.children_left == -1
        values = t.value[:, 0, :]
        probs = values / values.sum(axis=1, keepdims=True)
        trees.append(
            {
                "f": np.where(leaf, -1, t.feature).tolist(),
                # full precision, sklearn compares float32 inputs against these
                "t": t.threshold.tolist(),
                "l": t.children_left.tolist(),
                "r": t.children_right.tolist(),
                # only leaves need probabilities, internal nodes get null
                "p": [np.round(p, 5).tolist() if is_leaf else None for p, is_leaf in zip(probs, leaf)],
            }
        )
    return {"type": "rf", "features": FEATURES, "trees": trees}


def _walk_xgb(tree: dict, x: np.ndarray) -> float:
    node = 0
    left, right = tree["left_children"], tree["right_children"]
    cond, idx = tree["split_conditions"], tree["split_indices"]
    while left[node] != -1:
        node = left[node] if np.float32(x[idx[node]]) < np.float32(cond[node]) else right[node]
    return cond[node]  # xgboost stores leaf values in split_conditions


def export_xgb(model, X: pd.DataFrame) -> dict:
    booster = model.get_booster()
    raw = json.loads(booster.save_raw(raw_format="json"))
    gb = raw["learner"]["gradient_booster"]["model"]
    classes = gb["tree_info"]
    trees = [
        {
            "l": t["left_children"],
            "r": t["right_children"],
            "f": t["split_indices"],
            "t": t["split_conditions"],
        }
        for t in gb["trees"]
    ]
    # Work out the starting margin per class from the model itself rather
    # than trusting how this xgboost version stores base_score.
    sample = X.to_numpy(dtype=float)[:200]
    margins = booster.predict(xgb.DMatrix(sample, feature_names=FEATURES), output_margin=True)
    leaf_sums = np.zeros_like(margins)
    for tree, k in zip(gb["trees"], classes):
        leaf_sums[:, k] += [_walk_xgb(tree, row) for row in sample]
    offsets = margins - leaf_sums
    assert np.allclose(offsets, offsets[0], atol=1e-4), "intercept isn't constant, tree walk is wrong"
    return {"type": "xgb", "features": FEATURES, "classes": classes, "intercept": offsets[0].round(6).tolist(), "trees": trees}


def teams_payload(snapshot: dict, league: str) -> dict:
    snap = snapshot[league]
    return {
        "homeAdvLeague": snap["home_adv_league"],
        "teams": snap["teams"],
        "h2h": snap["h2h"],
        "meetings": snap["meetings"],
    }


def fixtures_payload(fixtures: pd.DataFrame, league: str) -> list[dict]:
    fx = fixtures[fixtures["league"] == league]
    out = []
    for r in fx.itertuples():
        out.append(
            {
                "kickoff": int(r.kickoff),
                "round": int(r.round),
                "home": r.home,
                "away": r.away,
                # h_elo and a_elo aren't model inputs (elo_diff is) but the
                # sliders need them to rebuild elo_diff
                "features": {f: round(float(getattr(r, f)), 4) for f in [*FEATURES, "h_elo", "a_elo"]},
            }
        )
    return out


def results_payload(preds: pd.DataFrame, league: str, calibration: dict) -> dict:
    """Last season and this season so far, every model and calibration."""
    lg = preds[preds["league"] == league]
    out = {}
    for part, frame in lg.groupby("split"):
        games = {}
        for kind, rows in frame.groupby("model"):
            raw = rows[["p_home", "p_draw", "p_away"]].to_numpy()
            for method in ("raw", "platt", "isotonic"):
                p = apply_calibration(raw, calibration[kind], method)
                for (idx, r), probs in zip(rows.iterrows(), p):
                    key = f"{r['date']}|{r['home']}|{r['away']}"
                    g = games.setdefault(key, {"date": r["date"], "home": r["home"], "away": r["away"],
                                               "score": [int(r["home_goals"]), int(r["away_goals"])],
                                               "result": r["result"], "p": {}})
                    g["p"][f"{kind}_{method}"] = [round(float(v), 4) for v in probs]
        out[part] = sorted(games.values(), key=lambda g: g["date"])
    return out


def main() -> None:
    split = seasons()
    matches = pd.read_csv(PROCESSED / "matches.csv", parse_dates=["date"])
    fixtures = pd.read_csv(PROCESSED / "fixtures.csv") if (PROCESSED / "fixtures.csv").exists() else pd.DataFrame()
    snapshot = json.loads((PROCESSED / "snapshot.json").read_text(encoding="utf-8"))
    report = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    preds = pd.read_csv(REPORTS / "predictions.csv")

    for league in LEAGUES:
        key = slug(league)
        train = matches[(matches["league"] == league) & matches["season"].isin(split["train"])]
        calibration = json.loads((MODELS / f"{key}_calibration.json").read_text(encoding="utf-8"))

        dump(OUT / key / "rf.json", export_forest(joblib.load(MODELS / f"{key}_rf.joblib")))
        dump(OUT / key / "xgb.json", export_xgb(joblib.load(MODELS / f"{key}_xgb.joblib"), train[FEATURES]))
        dump(OUT / key / "calibration.json", calibration)
        dump(OUT / key / "teams.json", teams_payload(snapshot, league))
        dump(OUT / key / "fixtures.json", fixtures_payload(fixtures, league) if len(fixtures) else [])
        dump(OUT / key / "results.json", results_payload(preds, league, calibration))

    live_season = split["live"]
    meta = {
        "exported": date.today().isoformat(),
        "seasons": report["seasons"],
        "features": FEATURES,
        "featureNames": FEATURE_NAMES,
        "leagues": {lg: {"key": slug(lg)} for lg in LEAGUES},
        "report": {lg: report[lg] for lg in LEAGUES},
        "liveSeason": season_label(live_season),
        "ranges": {
            f: [round(float(matches[f].quantile(0.005)), 2), round(float(matches[f].quantile(0.995)), 2)]
            for f in FEATURES
        },
    }
    dump(OUT / "meta.json", meta, indent=1)

    sizes = {p.relative_to(OUT).as_posix(): round(p.stat().st_size / 1024) for p in sorted(OUT.rglob("*.json"))}
    print("exported:", ", ".join(f"{k} {v}KB" for k, v in sizes.items()))


if __name__ == "__main__":
    main()
