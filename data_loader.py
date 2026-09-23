"""Match data and leak free rolling features.

Two sources:
  * football-data.co.uk: results, shots, shots on target and bookmaker odds,
    one CSV per league per season
  * Sofascore: ball possession for every match (football-data doesn't have
    it) and the fixture list for games that haven't been played yet

Sofascore is slow to crawl (one request per match), so everything it gives
us is kept in two small committed CSVs, data/processed/events.csv and
match_stats.csv. After the first run only new matches get fetched.

    python data_loader.py              fetch what's new, rebuild features
    python data_loader.py --offline    rebuild features from what's on disk
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from collections import defaultdict, deque
from datetime import date
from difflib import SequenceMatcher
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from curl_cffi import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

FOOTBALL_DATA = "https://www.football-data.co.uk/mmz4281"
SOFASCORE = "https://api.sofascore.com/api/v1"

LEAGUES = {
    "Premier League": {"fd": "E0", "sofascore": 17, "tz": "Europe/London"},
    "La Liga": {"fd": "SP1", "sofascore": 8, "tz": "Europe/Madrid"},
}

# football-data goes back further than we need: the extra seasons are only
# there so head to head records and rolling form have some history.
FIRST_SEASON = 2016
# Sofascore possession is complete from 2019/20 onwards.
POSSESSION_FROM = 2019


def current_season(today: date | None = None) -> int:
    """Start year of the season we're in. Seasons flip over in July."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def season_label(start: int) -> str:
    return f"{start}/{(start + 1) % 100:02d}"


# HTTP

_session: requests.Session | None = None


def _http() -> requests.Session:
    # Sofascore turns away plain python clients, a browser fingerprint works
    global _session
    if _session is None:
        _session = requests.Session(impersonate="chrome")
    return _session


class Blocked(RuntimeError):
    pass


def _get(url: str, pause: float = 0.6):
    for attempt in range(4):
        resp = _http().get(url, timeout=30)
        if resp.status_code in (200, 404):
            time.sleep(pause)
            return resp
        if resp.status_code in (403, 429) or resp.status_code >= 500:
            time.sleep(6 * (attempt + 1))
            continue
        resp.raise_for_status()
    raise Blocked(f"{url} kept returning {resp.status_code}")


# football-data.co.uk

def _fd_path(div: str, start: int) -> Path:
    return RAW / "football-data" / f"{div}_{start}.csv"


def fetch_results(league: str, start: int, force: bool = False) -> pd.DataFrame:
    div = LEAGUES[league]["fd"]
    path = _fd_path(div, start)
    if force or not path.exists():
        code = f"{start % 100:02d}{(start + 1) % 100:02d}"
        resp = _get(f"{FOOTBALL_DATA}/{code}/{div}.csv", pause=0.3)
        if resp.status_code == 404:
            return pd.DataFrame()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)

    raw = path.read_bytes().decode("latin-1")
    df = pd.read_csv(StringIO(raw), on_bad_lines="skip")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
    if df.empty:
        return df

    # older files call the market average BbAv*, newer ones Avg*
    odds = [("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA"), ("B365H", "B365D", "B365A")]
    oh, od, oa = next((c for c in odds if set(c) <= set(df.columns)), (None, None, None))

    out = pd.DataFrame(
        {
            "league": league,
            "season": start,
            "date": pd.to_datetime(df["Date"], dayfirst=True, format="mixed"),
            "home": df["HomeTeam"].str.strip(),
            "away": df["AwayTeam"].str.strip(),
            "home_goals": df["FTHG"].astype(int),
            "away_goals": df["FTAG"].astype(int),
            "result": df["FTR"].str.strip(),
            "home_shots": df.get("HS"),
            "away_shots": df.get("AS"),
            "home_sot": df.get("HST"),
            "away_sot": df.get("AST"),
            "odds_home": df[oh] if oh else np.nan,
            "odds_draw": df[od] if od else np.nan,
            "odds_away": df[oa] if oa else np.nan,
        }
    )
    return out


def load_all_results(fetch: bool = True) -> pd.DataFrame:
    now = current_season()
    frames = []
    for league in LEAGUES:
        for start in range(FIRST_SEASON, now + 1):
            # finished seasons never change, only the current one is refetched
            if fetch or _fd_path(LEAGUES[league]["fd"], start).exists():
                frames.append(fetch_results(league, start, force=fetch and start == now))
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    return df.sort_values(["date", "league", "home"]).reset_index(drop=True)


# Sofascore: fixtures and possession

EVENTS_CSV = PROCESSED / "events.csv"
STATS_CSV = PROCESSED / "match_stats.csv"


def _season_ids(tournament: int) -> dict[int, int]:
    cache = RAW / "sofascore" / f"seasons_{tournament}.json"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 7 * 86400:
        data = json.loads(cache.read_text(encoding="utf-8"))
    else:
        data = _get(f"{SOFASCORE}/unique-tournament/{tournament}/seasons").json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data), encoding="utf-8")
    out = {}
    for s in data["seasons"]:
        a, b = s["year"].split("/")
        out[2000 + int(a)] = s["id"]
    return out


def _parse_event(ev: dict, league: str, start: int, rnd: int) -> dict:
    return {
        "event_id": ev["id"],
        "league": league,
        "season": start,
        "round": rnd,
        "kickoff": ev["startTimestamp"],
        "home_sofa": ev["homeTeam"]["name"],
        "away_sofa": ev["awayTeam"]["name"],
        "status": ev["status"]["type"],
        "home_score": ev.get("homeScore", {}).get("current"),
        "away_score": ev.get("awayScore", {}).get("current"),
    }


def update_events(verbose: bool = True) -> pd.DataFrame:
    """Every league match from POSSESSION_FROM on, played or not."""
    known = pd.read_csv(EVENTS_CSV) if EVENTS_CSV.exists() else pd.DataFrame()
    now = current_season()
    rows = []
    for league, cfg in LEAGUES.items():
        ids = _season_ids(cfg["sofascore"])
        for start in range(POSSESSION_FROM, now + 1):
            have = known[(known.get("league") == league) & (known.get("season") == start)] if len(known) else known
            finished_before = start < now and len(have) >= 370
            if finished_before:
                rows.extend(have.to_dict("records"))
                continue
            if verbose:
                print(f"  fixtures {league} {season_label(start)}")
            for rnd in range(1, 39):
                data = _get(f"{SOFASCORE}/unique-tournament/{cfg['sofascore']}/season/{ids[start]}/events/round/{rnd}").json()
                rows.extend(_parse_event(ev, league, start, rnd) for ev in data.get("events", []))

    events = pd.DataFrame(rows).drop_duplicates("event_id", keep="last")
    events = events.sort_values(["league", "season", "kickoff"]).reset_index(drop=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    events.to_csv(EVENTS_CSV, index=False)
    return events


def _parse_stats(data: dict) -> dict:
    out = {"home_possession": np.nan, "away_possession": np.nan, "home_xg": np.nan, "away_xg": np.nan}
    for period in data.get("statistics", []):
        if period.get("period") != "ALL":
            continue
        for group in period["groups"]:
            for item in group["statisticsItems"]:
                if item["name"] == "Ball possession":
                    out["home_possession"], out["away_possession"] = item["homeValue"], item["awayValue"]
                elif item["name"] == "Expected goals":
                    out["home_xg"], out["away_xg"] = item["homeValue"], item["awayValue"]
    return out


def update_match_stats(events: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    stats = pd.read_csv(STATS_CSV) if STATS_CSV.exists() else pd.DataFrame(columns=["event_id"])
    done = set(stats["event_id"])
    todo = events[(events["status"] == "finished") & ~events["event_id"].isin(done)]["event_id"].tolist()
    if verbose and todo:
        print(f"  possession: {len(todo)} matches to fetch")

    new = []

    def save():
        nonlocal stats, new
        if new:
            stats = pd.concat([stats, pd.DataFrame(new)], ignore_index=True)
            stats.sort_values("event_id").to_csv(STATS_CSV, index=False)
            new = []

    try:
        for i, event_id in enumerate(todo, 1):
            resp = _get(f"{SOFASCORE}/event/{event_id}/statistics")
            row = _parse_stats(resp.json()) if resp.status_code == 200 else _parse_stats({})
            new.append({"event_id": event_id, **row})
            if i % 50 == 0:
                save()
                if verbose:
                    print(f"    {i}/{len(todo)}", flush=True)
    finally:
        save()
    return stats


# Joining the two sources

def _norm(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    name = re.sub(r"\b(fc|cf|afc|cd|ud|sd|rcd|ca|de|club)\b", " ", name)
    return re.sub(r"[^a-z]+", "", name)


def team_map(results: pd.DataFrame, events: pd.DataFrame) -> dict[tuple[str, str], str]:
    """football-data name -> Sofascore name, learned from the matches both have.

    football-data abbreviates ("Nott'm Forest", "Ath Madrid"), Sofascore
    doesn't. Pair up matches played on the same day with the same score,
    then let each football-data name vote for a Sofascore name.
    """
    ev = events[events["status"] == "finished"].copy()
    # football-data dates are local, so compare against the local kickoff day
    utc = pd.to_datetime(ev["kickoff"], unit="s", utc=True)
    ev["day"] = None
    for lg, cfg in LEAGUES.items():
        mask = ev["league"] == lg
        ev.loc[mask, "day"] = utc[mask].dt.tz_convert(cfg["tz"]).dt.date
    by_key = defaultdict(list)
    for r in ev.itertuples():
        by_key[(r.league, r.day, r.home_score, r.away_score)].append(r)

    votes = defaultdict(lambda: defaultdict(float))
    for r in results.itertuples():
        cands = by_key.get((r.league, r.date.date(), r.home_goals, r.away_goals), [])
        for c in cands:
            sim = SequenceMatcher(None, _norm(r.home), _norm(c.home_sofa)).ratio()
            sim += SequenceMatcher(None, _norm(r.away), _norm(c.away_sofa)).ratio()
            weight = 1.0 if len(cands) == 1 else sim / 2
            votes[(r.league, r.home)][c.home_sofa] += weight
            votes[(r.league, r.away)][c.away_sofa] += weight

    return {key: max(v, key=v.get) for key, v in votes.items()}


def attach_possession(results: pd.DataFrame, events: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    mapping = team_map(results, events)
    df = results.copy()
    df["home_sofa"] = [mapping.get((lg, t)) for lg, t in zip(df["league"], df["home"])]
    df["away_sofa"] = [mapping.get((lg, t)) for lg, t in zip(df["league"], df["away"])]
    # each pair meets once at each ground per season, so this key is unique
    # and survives postponements that move a game to another date
    ev = events.merge(stats, on="event_id", how="left")
    ev = ev[["league", "season", "home_sofa", "away_sofa", "event_id", "home_possession", "away_possession", "home_xg", "away_xg"]]
    return df.merge(ev, on=["league", "season", "home_sofa", "away_sofa"], how="left")


def fetch_everything(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if verbose:
        print("results from football-data.co.uk")
    results = load_all_results(fetch=True)
    if verbose:
        print("fixtures and possession from Sofascore")
    events = update_events(verbose)
    stats = update_match_stats(events, verbose)
    return results, events, stats


# Rolling features
#
# Everything below walks through the matches in date order. For each match
# it first asks the FormBook what it knows about both teams, and only then
# records the result. So a feature can only ever be built from matches that
# finished before kickoff. tests/test_leakage.py checks this.

WINDOW = 5                 # last five matches
LOOKBACK_DAYS = 365        # form from over a year ago doesn't count
H2H_DAYS = 3 * 365         # head to head over the last three seasons
HOME_ADV_MATCHES = 38      # a season's worth of a team's own games
LEAGUE_ADV_MATCHES = 380   # a season's worth of the whole league

STATS = ["gf", "ga", "sotf", "sota", "poss", "pts"]

# Elo, for the long run strength that five games of form can't see
ELO_START = 1500           # everyone in the first season of our data
ELO_PROMOTED = 1430        # a team we haven't seen before, i.e. promoted
ELO_K = 22
ELO_HOME = 55              # home advantage in rating points

FEATURES = [
    "h_gf", "h_ga", "h_sotf", "h_sota", "h_poss", "h_form",
    "a_gf", "a_ga", "a_sotf", "a_sota", "a_poss", "a_form",
    "h2h", "home_adv_team", "home_adv_league", "gd_gap", "form_gap", "elo_diff",
]
TARGET = {"H": 0, "D": 1, "A": 2}


def _points(gf: int, ga: int) -> int:
    return 3 if gf > ga else 1 if gf == ga else 0


class FormBook:
    """What we know about every team at a given moment, built one match at a time."""

    def __init__(self, prior: dict[str, float]):
        self.prior = prior
        self.games = defaultdict(list)      # team -> list of dicts, oldest first
        self.meetings = defaultdict(list)   # frozenset(team, team) -> [(date, winner or None)]
        self.league = deque(maxlen=LEAGUE_ADV_MATCHES + 20)  # (date, home pts, away pts)
        self.elo = {}
        self.first_day = None

    def rating(self, team: str, when: pd.Timestamp) -> float:
        if team in self.elo:
            r, last = self.elo[team]
            # back after a spell in the second tier: meet the promoted level halfway
            return (r + ELO_PROMOTED) / 2 if (when - last).days > LOOKBACK_DAYS else r
        early = self.first_day is None or (when - self.first_day).days < 60
        return ELO_START if early else ELO_PROMOTED

    def recent(self, team: str, when: pd.Timestamp) -> list[dict]:
        cutoff = when - pd.Timedelta(days=LOOKBACK_DAYS)
        return [g for g in self.games[team] if cutoff <= g["date"] < when][-WINDOW:]

    def team_form(self, team: str, when: pd.Timestamp) -> dict[str, float]:
        """Averages over the last five, topped up with the prior when a team
        has fewer than five (newly promoted, or the start of our data)."""
        past = self.recent(team, when)
        out = {"n": len(past)}
        for stat in STATS:
            vals = [g[stat] for g in past if not np.isnan(g[stat])]
            total = sum(vals) + self.prior[stat] * (WINDOW - len(vals))
            out[stat] = total / WINDOW
        out["form"] = out.pop("pts") * WINDOW  # points from the last five, 0 to 15
        return out

    def h2h(self, home: str, away: str, when: pd.Timestamp) -> float:
        """Home side's win rate against this opponent, smoothed so that no
        meetings reads as a neutral 1 in 3."""
        cutoff = when - pd.Timedelta(days=H2H_DAYS)
        games = [w for d, w in self.meetings[frozenset((home, away))] if cutoff <= d < when]
        wins = sum(w == home for w in games)
        return (wins + 1) / (len(games) + 3)

    def home_adv_team(self, team: str, when: pd.Timestamp) -> float:
        """Points per game at home minus away, over the last season or so,
        shrunk towards zero when there isn't much to go on."""
        cutoff = when - pd.Timedelta(days=2 * 365)
        past = [g for g in self.games[team] if cutoff <= g["date"] < when][-HOME_ADV_MATCHES:]
        home = [g["pts"] for g in past if g["venue"] == "H"]
        away = [g["pts"] for g in past if g["venue"] == "A"]
        if not home or not away:
            return 0.0
        n = len(past)
        return (np.mean(home) - np.mean(away)) * n / (n + 20)

    def home_adv_league(self, when: pd.Timestamp) -> float:
        # strictly earlier days only: games on the same day kick off in no
        # particular order in the data, so they can't count for each other
        past = [(h, a) for d, h, a in self.league if d < when][-LEAGUE_ADV_MATCHES:]
        if not past:
            return 0.4
        h, a = zip(*past)
        return float(np.mean(h) - np.mean(a))

    def features(self, home: str, away: str, when: pd.Timestamp) -> dict[str, float]:
        h = self.team_form(home, when)
        a = self.team_form(away, when)
        row = {f"h_{k}": v for k, v in h.items() if k != "n"}
        row.update({f"a_{k}": v for k, v in a.items() if k != "n"})
        row["h2h"] = self.h2h(home, away, when)
        row["home_adv_team"] = self.home_adv_team(home, when)
        row["home_adv_league"] = self.home_adv_league(when)
        row["gd_gap"] = (h["gf"] - h["ga"]) - (a["gf"] - a["ga"])
        row["form_gap"] = h["form"] - a["form"]
        row["h_elo"], row["a_elo"] = self.rating(home, when), self.rating(away, when)
        row["elo_diff"] = row["h_elo"] - row["a_elo"]
        row["h_n"], row["a_n"] = h["n"], a["n"]
        return row

    def _update_elo(self, m) -> None:
        rh, ra = self.rating(m.home, m.date), self.rating(m.away, m.date)
        expected = 1 / (1 + 10 ** ((ra - rh - ELO_HOME) / 400))
        actual = 1.0 if m.home_goals > m.away_goals else 0.5 if m.home_goals == m.away_goals else 0.0
        gd = abs(m.home_goals - m.away_goals)
        margin = 1 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        shift = ELO_K * margin * (actual - expected)
        self.elo[m.home] = (rh + shift, m.date)
        self.elo[m.away] = (ra - shift, m.date)

    def record(self, m) -> None:
        if self.first_day is None:
            self.first_day = m.date
        self._update_elo(m)
        hp, ap = _points(m.home_goals, m.away_goals), _points(m.away_goals, m.home_goals)
        poss_h = m.home_possession if pd.notna(m.home_possession) else np.nan
        poss_a = m.away_possession if pd.notna(m.away_possession) else np.nan
        self.games[m.home].append(
            {"date": m.date, "venue": "H", "opp": m.away, "gf": m.home_goals, "ga": m.away_goals,
             "sotf": m.home_sot, "sota": m.away_sot, "poss": poss_h, "pts": hp}
        )
        self.games[m.away].append(
            {"date": m.date, "venue": "A", "opp": m.home, "gf": m.away_goals, "ga": m.home_goals,
             "sotf": m.away_sot, "sota": m.home_sot, "poss": poss_a, "pts": ap}
        )
        winner = m.home if hp == 3 else m.away if ap == 3 else None
        self.meetings[frozenset((m.home, m.away))].append((m.date, winner))
        self.league.append((m.date, hp, ap))


def promoted_prior(matches: pd.DataFrame, before_season: int) -> dict[str, float]:
    """What a team with no recent top flight history tends to look like:
    the average of promoted sides' first five games, from seasons before
    `before_season` only, so the test season can't leak into it."""
    seen_until = {}
    samples = defaultdict(list)
    counts = defaultdict(int)
    promoted = set()
    for m in matches.itertuples():
        for team, gf, ga, sotf, sota, poss in (
            (m.home, m.home_goals, m.away_goals, m.home_sot, m.away_sot, m.home_possession),
            (m.away, m.away_goals, m.home_goals, m.away_sot, m.home_sot, m.away_possession),
        ):
            key = (m.league, team, m.season)
            if key not in counts:
                last = seen_until.get((m.league, team))
                fresh = last is None or (m.date - last).days > LOOKBACK_DAYS
                if fresh and FIRST_SEASON < m.season < before_season:
                    promoted.add(key)
            counts[key] += 1
            seen_until[(m.league, team)] = m.date
            if key in promoted and counts[key] <= WINDOW:
                samples["gf"].append(gf)
                samples["ga"].append(ga)
                samples["sotf"].append(sotf)
                samples["sota"].append(sota)
                samples["poss"].append(poss)
                samples["pts"].append(_points(gf, ga))
    return {k: float(np.nanmean(v)) for k, v in samples.items()}


def build_features(matches: pd.DataFrame, prior: dict[str, float]) -> tuple[pd.DataFrame, dict[str, FormBook]]:
    rows, books = [], {}
    for league, games in matches.groupby("league", sort=False):
        book = FormBook(prior)
        for m in games.sort_values("date", kind="stable").itertuples():
            rows.append({"idx": m.Index, **book.features(m.home, m.away, m.date)})
            book.record(m)
        books[league] = book
    feats = pd.DataFrame(rows).set_index("idx").sort_index()
    # rebuilding from a table that already has features: replace, don't collide
    stale = [c for c in feats.columns if c in matches.columns]
    return matches.drop(columns=stale).join(feats), books


def upcoming_fixtures(events: pd.DataFrame, results: pd.DataFrame, books: dict[str, FormBook], per_league: int = 10) -> pd.DataFrame:
    """Next few unplayed games per league, with features as of right now."""
    mapping = team_map(results, events)
    rows = []
    for league, book in books.items():
        back = {sofa: fd for (lg, fd), sofa in mapping.items() if lg == league}
        known = sorted({fd for (lg, fd) in mapping if lg == league})
        todo = events[(events["league"] == league) & (events["status"] == "notstarted")].sort_values("kickoff").head(per_league)
        for ev in todo.itertuples():
            home = back.get(ev.home_sofa) or _closest(ev.home_sofa, known)
            away = back.get(ev.away_sofa) or _closest(ev.away_sofa, known)
            when = pd.Timestamp(ev.kickoff, unit="s").normalize()
            rows.append(
                {"league": league, "season": ev.season, "round": ev.round, "kickoff": ev.kickoff,
                 "home": home, "away": away, "home_sofa": ev.home_sofa, "away_sofa": ev.away_sofa,
                 **book.features(home, away, when)}
            )
    return pd.DataFrame(rows)


def snapshot(events: pd.DataFrame, results: pd.DataFrame, books: dict[str, FormBook]) -> dict:
    """Every current team's form as of today, for the head to head simulator."""
    mapping = team_map(results, events)
    today = pd.Timestamp(date.today())
    season = current_season()
    out = {}
    for league, book in books.items():
        back = {sofa: fd for (lg, fd), sofa in mapping.items() if lg == league}
        known = sorted({fd for (lg, fd) in mapping if lg == league})
        cur = events[(events["league"] == league) & (events["season"] == season)]
        names = sorted({back.get(n) or _closest(n, known) for n in pd.concat([cur["home_sofa"], cur["away_sofa"]])})

        teams = {}
        for team in names:
            form = book.team_form(team, today)
            last = [
                {"date": g["date"].strftime("%Y-%m-%d"), "opp": g["opp"], "venue": g["venue"],
                 "gf": int(g["gf"]), "ga": int(g["ga"])}
                for g in book.recent(team, today)
            ]
            teams[team] = {
                "display": mapping.get((league, team), team),
                **{k: round(float(v), 4) for k, v in form.items()},
                "elo": round(float(book.rating(team, today)), 1),
                "home_adv": round(float(book.home_adv_team(team, today)), 4),
                "last": last,
            }
        h2h = {
            f"{h}|{a}": round(book.h2h(h, a, today), 4)
            for h in names for a in names if h != a
        }
        meetings = {}
        for h in names:
            for a in names:
                if h < a:
                    games = [
                        {"date": d.strftime("%Y-%m-%d"), "winner": w}
                        for d, w in book.meetings[frozenset((h, a))]
                        if d >= today - pd.Timedelta(days=H2H_DAYS)
                    ]
                    if games:
                        meetings[f"{h}|{a}"] = games
        out[league] = {"teams": teams, "h2h": h2h, "meetings": meetings, "home_adv_league": round(book.home_adv_league(today), 4)}
    return out


def _closest(name: str, options: list[str]) -> str:
    return max(options, key=lambda o: SequenceMatcher(None, _norm(name), _norm(o)).ratio())


def build_and_save(results: pd.DataFrame, events: pd.DataFrame, stats: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    matches = attach_possession(results, events, stats)
    matches = matches.sort_values(["date", "league", "home"]).reset_index(drop=True)

    have = matches["season"] >= POSSESSION_FROM
    missing = matches[have & matches["home_possession"].isna()]
    if verbose:
        print(f"{len(matches)} matches, possession missing for {len(missing)} since {season_label(POSSESSION_FROM)}")
        if len(missing):
            print("  e.g.", ", ".join(f"{r.home} v {r.away} {r.date.date()}" for r in missing.head(5).itertuples()))

    test_season = current_season() - 1
    prior = promoted_prior(matches, before_season=test_season)
    feats, books = build_features(matches, prior)
    feats["target"] = feats["result"].map(TARGET)

    fixtures = upcoming_fixtures(events, results, books)
    snap = snapshot(events, results, books)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    feats.to_csv(PROCESSED / "matches.csv", index=False, float_format="%.4f")
    fixtures.to_csv(PROCESSED / "fixtures.csv", index=False, float_format="%.4f")
    (PROCESSED / "prior.json").write_text(json.dumps({k: round(v, 4) for k, v in prior.items()}, indent=1), encoding="utf-8")
    (PROCESSED / "snapshot.json").write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    if verbose:
        print(f"wrote {len(feats)} matches and {len(fixtures)} upcoming fixtures")
    return feats


def load_matches() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "matches.csv", parse_dates=["date"])


def load_offline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_all_results(fetch=False), pd.read_csv(EVENTS_CSV), pd.read_csv(STATS_CSV)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="don't download anything")
    parser.add_argument("--fetch-only", action="store_true", help="download, skip the feature build")
    args = parser.parse_args()

    results, events, stats = load_offline() if args.offline else fetch_everything()
    if args.fetch_only:
        return
    build_and_save(results, events, stats)


if __name__ == "__main__":
    main()
