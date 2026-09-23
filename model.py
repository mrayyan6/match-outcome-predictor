"""Train and evaluate the match outcome models.

Three way classification, 0 home win, 1 draw, 2 away win. Per league, a
Random Forest (class_weight="balanced") and XGBoost (multi:softprob, with
the same balanced weighting passed as sample weights).

Split by time, never at random:
    train   the five seasons before the test season
    test    the last completed season
    live    the current season so far, a bonus check on games since

About draws: they're roughly a quarter of results and an unweighted model
almost never predicts one. Balanced weights fix that, but they also pull
every probability towards the draw, so the raw numbers are off. Each model
therefore gets two calibration maps (Platt scaling and isotonic), fitted on
out of fold predictions from the training seasons only. The site lets you
flip between raw and calibrated to see the trade.

    python model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from data_loader import FEATURES, LEAGUES, current_season, load_matches, season_label

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

SEED = 11
TRAIN_SEASONS = 5
CLASSES = ["H", "D", "A"]
EPS = 1e-6


def slug(league: str) -> str:
    return league.lower().replace(" ", "_")


def seasons() -> dict[str, list[int] | int]:
    live = current_season()
    test = live - 1
    return {"train": list(range(test - TRAIN_SEASONS, test)), "test": test, "live": live}


def make_forest() -> RandomForestClassifier:
    # small-ish on purpose, the whole forest ships to the browser as JSON
    return RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=25,
        max_features="sqrt",
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def make_xgb() -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=220,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=6,
        reg_lambda=2.0,
        random_state=SEED,
        n_jobs=4,
        eval_metric="mlogloss",
    )


BUILDERS = {"rf": make_forest, "xgb": make_xgb}


def fit(kind: str, X: pd.DataFrame, y: np.ndarray):
    model = BUILDERS[kind]()
    if kind == "xgb":
        # XGBoost has no class_weight, balanced sample weights do the same job
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


# Calibration

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_calibration(probs: np.ndarray, y: np.ndarray) -> dict:
    """One-vs-rest maps per class, renormalised after applying."""
    platt, iso = [], []
    for k in range(3):
        target = (y == k).astype(int)
        lr = LogisticRegression(C=1e4).fit(_logit(probs[:, k]).reshape(-1, 1), target)
        platt.append({"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])})
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(probs[:, k], target)
        iso.append({"x": ir.X_thresholds_.round(5).tolist(), "y": ir.y_thresholds_.round(5).tolist()})
    return {"platt": platt, "isotonic": iso}


def apply_calibration(probs: np.ndarray, cal: dict, method: str) -> np.ndarray:
    if method == "raw":
        return probs
    out = np.zeros_like(probs)
    for k in range(3):
        if method == "platt":
            a, b = cal["platt"][k]["a"], cal["platt"][k]["b"]
            out[:, k] = 1 / (1 + np.exp(-(a * _logit(probs[:, k]) + b)))
        else:
            m = cal["isotonic"][k]
            out[:, k] = np.interp(probs[:, k], m["x"], m["y"])
    total = out.sum(axis=1, keepdims=True)
    return np.where(total > 0, out / np.maximum(total, EPS), probs)


def oof_for_calibration(df: pd.DataFrame, kind: str, train: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Walk forward through the training seasons: fit on everything before a
    season, predict that season. Only seasons with two or more before them."""
    probs, ys = [], []
    for s in train[2:]:
        past = df[df["season"].isin([t for t in train if t < s])]
        now = df[df["season"] == s]
        model = fit(kind, past[FEATURES], past["target"].to_numpy())
        probs.append(model.predict_proba(now[FEATURES]))
        ys.append(now["target"].to_numpy())
    return np.vstack(probs), np.concatenate(ys)


# Evaluation

def scores(y: np.ndarray, probs: np.ndarray) -> dict:
    probs = np.clip(probs, EPS, 1)
    probs = probs / probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    return {
        "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "log_loss": round(float(log_loss(y, np.clip(probs, EPS, 1), labels=[0, 1, 2])), 4),
        "recall": [round(float(v), 3) for v in recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)],
        "precision": [round(float(v), 3) for v in precision_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)],
        "predicted_share": [round(float((pred == k).mean()), 3) for k in range(3)],
        "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }


def reliability(y: np.ndarray, probs: np.ndarray, bins: int = 10) -> list[dict]:
    """Pool every (predicted probability, did it happen) pair over the three
    outcomes. A calibrated model sits on the diagonal."""
    p = probs.ravel()
    hit = (np.arange(3)[None, :] == y[:, None]).ravel()
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.sum() >= 5:
            out.append({"lo": round(lo, 2), "hi": round(hi, 2), "predicted": round(float(p[mask].mean()), 4),
                        "actual": round(float(hit[mask].mean()), 4), "n": int(mask.sum())})
    return out


def bookmaker_probs(df: pd.DataFrame) -> np.ndarray:
    inv = 1 / df[["odds_home", "odds_draw", "odds_away"]].to_numpy(dtype=float)
    return inv / inv.sum(axis=1, keepdims=True)  # strip the margin


FEATURE_NAMES = {
    "h_gf": "Home: goals scored", "h_ga": "Home: goals conceded", "h_sotf": "Home: shots on target",
    "h_sota": "Home: shots on target faced", "h_poss": "Home: possession", "h_form": "Home: form",
    "a_gf": "Away: goals scored", "a_ga": "Away: goals conceded", "a_sotf": "Away: shots on target",
    "a_sota": "Away: shots on target faced", "a_poss": "Away: possession", "a_form": "Away: form",
    "h2h": "Head to head", "home_adv_team": "Home advantage (team)", "home_adv_league": "Home advantage (league)",
    "gd_gap": "Goal difference gap", "form_gap": "Form gap", "elo_diff": "Elo gap",
}


def plot_confusion(report: dict, league: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, kind, title in zip(axes, ["rf", "xgb"], ["Random Forest", "XGBoost"]):
        cm = np.array(report[kind]["raw"]["confusion"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Home", "Draw", "Away"], yticklabels=["Home", "Draw", "Away"])
        ax.set(title=title, xlabel="predicted", ylabel="actual")
    fig.suptitle(f"{league}, {season_label(seasons()['test'])}")
    fig.tight_layout()
    fig.savefig(REPORTS / f"{slug(league)}_confusion.png", dpi=110)
    plt.close(fig)


def plot_importance(imp: dict, league: str) -> None:
    frame = pd.DataFrame(imp).rename(index=FEATURE_NAMES)
    order = frame.mean(axis=1).sort_values(ascending=False).index
    long = frame.reset_index(names="feature").melt(id_vars="feature", var_name="model", value_name="importance")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.barplot(data=long, y="feature", x="importance", hue="model", order=order, ax=ax)
    ax.set(title=f"{league}: what the models lean on", xlabel="increase in log loss when shuffled", ylabel="")
    fig.tight_layout()
    fig.savefig(REPORTS / f"{slug(league)}_importance.png", dpi=110)
    plt.close(fig)


def train_all(verbose: bool = True) -> dict:
    df = load_matches()
    split = seasons()
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    report = {"seasons": {k: (season_label(v) if isinstance(v, int) else [season_label(s) for s in v]) for k, v in split.items()}}
    predictions = []

    for league in LEAGUES:
        lg = df[df["league"] == league]
        train = lg[lg["season"].isin(split["train"])]
        test = lg[lg["season"] == split["test"]]
        live = lg[lg["season"] == split["live"]]
        y_train, y_test = train["target"].to_numpy(), test["target"].to_numpy()

        out = {"class_share_train": [round(float((y_train == k).mean()), 3) for k in range(3)]}
        importance, calibration = {}, {}

        for kind in BUILDERS:
            oof_p, oof_y = oof_for_calibration(lg, kind, split["train"])
            cal = fit_calibration(oof_p, oof_y)
            calibration[kind] = cal

            model = fit(kind, train[FEATURES], y_train)
            joblib.dump(model, MODELS / f"{slug(league)}_{kind}.joblib")

            raw_test = model.predict_proba(test[FEATURES])
            out[kind] = {}
            for method in ("raw", "platt", "isotonic"):
                p = apply_calibration(raw_test, cal, method)
                out[kind][method] = scores(y_test, p)
                out[kind][method]["reliability"] = reliability(y_test, p)
                if len(live):
                    out[kind][method]["live"] = scores(live["target"].to_numpy(), apply_calibration(model.predict_proba(live[FEATURES]), cal, method))

            perm = permutation_importance(model, test[FEATURES], y_test, scoring="neg_log_loss", n_repeats=15, random_state=SEED)
            importance[kind] = {f: round(float(v), 5) for f, v in zip(FEATURES, perm.importances_mean)}

            for part, frame in (("test", test), ("live", live)):
                if len(frame):
                    p = model.predict_proba(frame[FEATURES])
                    predictions.append(
                        frame[["league", "season", "date", "home", "away", "home_goals", "away_goals", "result"]]
                        .assign(model=kind, split=part, p_home=p[:, 0], p_draw=p[:, 1], p_away=p[:, 2])
                    )

        # baselines, on the same test season
        share = np.array(out["class_share_train"])
        out["baselines"] = {
            "always_home": scores(y_test, np.tile(np.array([0.999, 0.0005, 0.0005]), (len(y_test), 1))) | {
                "log_loss": round(float(log_loss(y_test, np.tile(share, (len(y_test), 1)), labels=[0, 1, 2])), 4)},
            "bookmakers": scores(y_test, bookmaker_probs(test)) | {"reliability": reliability(y_test, bookmaker_probs(test))},
        }
        out["importance"] = importance
        (MODELS / f"{slug(league)}_calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
        report[league] = out

        plot_confusion(out, league)
        plot_importance(importance, league)

        if verbose:
            print(f"\n{league}: train {len(train)}, test {len(test)} ({season_label(split['test'])}), live {len(live)}")
            b = out["baselines"]
            print(f"  always home     acc {b['always_home']['accuracy']:.3f}  logloss {b['always_home']['log_loss']:.3f}")
            print(f"  bookmakers      acc {b['bookmakers']['accuracy']:.3f}  logloss {b['bookmakers']['log_loss']:.3f}")
            for kind in BUILDERS:
                for method in ("raw", "platt", "isotonic"):
                    s = out[kind][method]
                    print(f"  {kind:4s} {method:9s} acc {s['accuracy']:.3f}  logloss {s['log_loss']:.3f}  draw recall {s['recall'][1]:.2f}  draws predicted {s['predicted_share'][1]:.2f}")

    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    pd.concat(predictions).to_csv(REPORTS / "predictions.csv", index=False, float_format="%.4f")
    return report


if __name__ == "__main__":
    train_all()
