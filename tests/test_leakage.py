"""The features for a match must only depend on matches played before it."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import data_loader as dl  # noqa: E402

PRIOR = {"gf": 1.0, "ga": 1.7, "sotf": 3.5, "sota": 5.0, "poss": 44.0, "pts": 1.0}


def toy_matches() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    teams = [f"Team {c}" for c in "ABCDEF"]
    rows = []
    day = pd.Timestamp("2022-08-06")
    for week in range(30):
        order = rng.permutation(teams)
        for i in range(0, 6, 2):
            hg, ag = rng.integers(0, 4, size=2)
            rows.append(
                {"league": "Test League", "season": 2022, "date": day, "home": order[i], "away": order[i + 1],
                 "home_goals": int(hg), "away_goals": int(ag), "result": "H" if hg > ag else "A" if ag > hg else "D",
                 "home_sot": int(hg) + 2, "away_sot": int(ag) + 2,
                 "home_possession": 50 + int(rng.integers(-10, 10)), "away_possession": np.nan}
            )
        day += pd.Timedelta(days=7)
    df = pd.DataFrame(rows)
    df["away_possession"] = 100 - df["home_possession"]
    return df


@pytest.fixture
def matches():
    path = dl.PROCESSED / "matches.csv"
    if path.exists():
        return dl.load_matches()
    return toy_matches()


def test_later_matches_do_not_change_earlier_features(matches):
    full, _ = dl.build_features(matches, PRIOR)
    cut = matches["date"].sort_values().iloc[len(matches) * 2 // 3]
    early = matches[matches["date"] < cut]
    part, _ = dl.build_features(early, PRIOR)
    pd.testing.assert_frame_equal(full.loc[part.index, dl.FEATURES], part[dl.FEATURES])


def test_a_match_does_not_see_its_own_result():
    df = toy_matches()
    before, _ = dl.build_features(df, PRIOR)
    flipped = df.copy()
    i = 40
    flipped.loc[i, ["home_goals", "away_goals"]] = [9, 0]
    after, _ = dl.build_features(flipped, PRIOR)
    # same features for the match itself...
    pd.testing.assert_series_equal(before.loc[i, dl.FEATURES], after.loc[i, dl.FEATURES])
    # ...but the home side's next game notices the 9-0
    home = df.loc[i, "home"]
    nxt = df[(df.index > i) & ((df["home"] == home) | (df["away"] == home))].index[0]
    assert not before.loc[nxt, dl.FEATURES].equals(after.loc[nxt, dl.FEATURES])


def test_first_game_uses_the_prior_only():
    df = toy_matches()
    feats, _ = dl.build_features(df, PRIOR)
    first = feats.iloc[0]
    assert first["h_gf"] == pytest.approx(PRIOR["gf"])
    assert first["a_form"] == pytest.approx(PRIOR["pts"] * dl.WINDOW)
    assert first["h2h"] == pytest.approx(1 / 3)


def test_form_is_points_from_last_five():
    df = toy_matches()
    feats, _ = dl.build_features(df, PRIOR)
    team = "Team A"
    games = df[(df["home"] == team) | (df["away"] == team)]
    sixth = games.index[5]
    pts = []
    for _, g in games.iloc[:5].iterrows():
        gf, ga = (g.home_goals, g.away_goals) if g.home == team else (g.away_goals, g.home_goals)
        pts.append(3 if gf > ga else 1 if gf == ga else 0)
    side = "h" if df.loc[sixth, "home"] == team else "a"
    assert feats.loc[sixth, f"{side}_form"] == sum(pts)
