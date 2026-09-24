"""Генерира СИНТЕТИЧНИ данни в точния формат на football-data.co.uk и
football-data.org. Ползва се само за офлайн тест (`python run.py all --offline`).
Реалните данни се свалят автоматично без този скрипт."""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.data import season_codes, season_start_year  # noqa: E402

rng = np.random.default_rng(42)
SIZES = {"E0": 20, "E1": 24, "E2": 24, "E3": 24, "SP1": 20, "SP2": 22, "I1": 20, "I2": 20,
         "D1": 18, "D2": 18, "F1": 18, "F2": 18}
TODAY = dt.date.today()


def team_pool(country, divs):
    teams = {}
    for lvl, d in enumerate(divs):
        for k in range(SIZES[d]):
            name = f"{country[:3]} {d}-{k + 1:02d}"
            teams[name] = {"att": rng.normal(0.25 - 0.12 * lvl, 0.22),
                           "def": rng.normal(0.0 - 0.10 * lvl, 0.20),
                           "corn": rng.normal(0, 0.12), "div": d}
    return teams


def match_probs(lam, mu):
    g = np.arange(11)
    from scipy.stats import poisson
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    return np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum(), m[np.add.outer(g, g) > 2.5].sum()


def odds_cols(ph, pd_, pa, po):
    def price(p, margin=1.055):
        p = np.clip(p * np.exp(rng.normal(0, 0.06)), 0.02, 0.97)
        return round(1 / (p * margin), 2)
    h, d, a, o = price(ph), price(pd_), price(pa), price(po)
    u = price(1 - po)
    return {"B365H": h, "B365D": d, "B365A": a, "AvgH": h, "AvgD": d, "AvgA": a,
            "MaxH": round(h * 1.05, 2), "MaxD": round(d * 1.05, 2), "MaxA": round(a * 1.05, 2),
            "B365>2.5": o, "B365<2.5": u, "Avg>2.5": o, "Avg<2.5": u,
            "Max>2.5": round(o * 1.04, 2), "Max<2.5": round(u * 1.04, 2)}


def simulate_season(div, teams, start_year, played_until):
    names = list(teams)
    pairs = [(h, a) for h in names for a in names if h != a]
    rng.shuffle(pairs)
    start = dt.date(start_year, 8, 10)
    per_round = len(names) // 2
    played, future = [], []
    for i, (h, a) in enumerate(pairs):
        date = start + dt.timedelta(days=7 * (i // per_round) + int(rng.integers(0, 3)))
        th, ta = teams[h], teams[a]
        lam = np.exp(0.3 + th["att"] - ta["def"])
        mu = np.exp(0.05 + ta["att"] - th["def"])
        ph, pdr, pa, po = match_probs(lam, mu)
        row = {"Div": div, "Date": date.strftime("%d/%m/%Y"), "Time": "15:00",
               "HomeTeam": h, "AwayTeam": a, **odds_cols(ph, pdr, pa, po)}
        if date < played_until:
            hg, ag = rng.poisson(lam), rng.poisson(mu)
            row.update({"FTHG": hg, "FTAG": ag, "FTR": "H" if hg > ag else "D" if hg == ag else "A",
                        "HC": rng.poisson(np.exp(1.65 + th["corn"] + 0.3 * (th["att"] - ta["att"]))),
                        "AC": rng.poisson(np.exp(1.45 + ta["corn"] + 0.3 * (ta["att"] - th["att"])))})
            played.append(row)
        elif date <= TODAY + dt.timedelta(days=config.DAYS_AHEAD):
            future.append(row)
    return played, future


def table(rows, names):
    pts = {n: 0 for n in names}
    for r in rows:
        pts[r["HomeTeam"]] += {"H": 3, "D": 1, "A": 0}[r["FTR"]]
        pts[r["AwayTeam"]] += {"H": 0, "D": 1, "A": 3}[r["FTR"]]
    return sorted(names, key=lambda n: -pts[n])


def main():
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    seasons = season_codes()
    first = season_start_year() - len(seasons) + 1
    fixtures, top_teams = [], {}
    for country, divs in config.COUNTRIES.items():
        divs = list(divs)
        teams = team_pool(country, divs)
        for si, season in enumerate(seasons):
            year = first + si
            until = min(TODAY, dt.date(year + 1, 6, 1))
            tables = {}
            for d in divs:
                members = {n: t for n, t in teams.items() if t["div"] == d}
                played, future = simulate_season(d, members, year, until)
                pd.DataFrame(played).to_csv(config.RAW_DIR / f"{season}_{d}.csv", index=False)
                fixtures += future
                if played:
                    tables[d] = table(played, list(members))
            top_teams[year] = tables.get(divs[0], [])[:6] if tables.get(divs[0]) else []
            # промоции/изпадания между дивизиите
            if season != seasons[-1]:
                for up, down in zip(divs[:-1], divs[1:]):
                    for n in tables[up][-3:]:
                        teams[n]["div"] = down
                    for n in tables[down][:3]:
                        teams[n]["div"] = up
                for t in teams.values():   # леки промени в силата между сезоните
                    t["att"] += rng.normal(0, 0.08)
                    t["def"] += rng.normal(0, 0.08)
        top_teams[country] = teams
    pd.DataFrame(fixtures).to_csv(config.RAW_DIR / "fixtures.csv", index=False)

    # Шампионска лига (формат на football-data.org)
    all_teams = {}
    for c in config.COUNTRIES:
        all_teams.update(top_teams[c])
    for si in range(len(seasons)):
        year = first + si
        strong = sorted(all_teams, key=lambda n: -(all_teams[n]["att"] + all_teams[n]["def"]))[:36]
        matches = []
        start = dt.datetime(year, 9, 16, 19, 0)
        for rnd in range(8):
            order = list(rng.permutation(strong))
            for k in range(0, 36, 2):
                h, a = order[k], order[k + 1]
                when = start + dt.timedelta(days=7 * rnd + (k % 2))
                th, ta = all_teams[h], all_teams[a]
                lam = np.exp(0.2 + th["att"] - ta["def"])
                mu = np.exp(0.0 + ta["att"] - th["def"])
                done = when.date() < TODAY
                m = {"utcDate": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "status": "FINISHED" if done else "TIMED",
                     "homeTeam": {"name": h, "shortName": h}, "awayTeam": {"name": a, "shortName": a},
                     "score": {"fullTime": {"home": int(rng.poisson(lam)) if done else None,
                                            "away": int(rng.poisson(mu)) if done else None}}}
                matches.append(m)
        (config.RAW_DIR / f"CL_{year}.json").write_text(json.dumps({"matches": matches}), "utf-8")
    print(f"Синтетични данни записани в {config.RAW_DIR} ({len(fixtures)} предстоящи мача)")


if __name__ == "__main__":
    main()
