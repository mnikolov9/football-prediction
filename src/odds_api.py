"""Коефициенти за Лигата на нациите от The Odds API (the-odds-api.com).

Безплатният план дава 500 кредита на месец. Една заявка за двата пазара
(1X2 и голове) в два региона струва 4 кредита, т.е. ежедневните пускания
харчат ~130 кредита на месец. Ключът е в променливата ODDS_API_KEY.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import config
from src import data, teammatch

URL = ("https://api.the-odds-api.com/v4/sports/{sport}/odds/"
       "?apiKey={key}&regions=eu,uk&markets=h2h,totals&oddsFormat=decimal")


def fetch(sport: str, offline: bool = False) -> list[dict]:
    path = config.RAW_DIR / f"odds_{sport}.json"
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not offline and key:
        try:
            raw = data._http_get(URL.format(sport=sport, key=key))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        except RuntimeError as e:
            print(f"  ! Коефициенти {sport}: {e}")
    elif not offline:
        print("  ! Няма ODDS_API_KEY – без коефициенти за Лигата на нациите.")
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError:
        return []


def _summarize(event: dict) -> dict:
    """Средни и максимални коефициенти от всички букмейкъри."""
    home, away = event["home_team"], event["away_team"]
    prices = {k: [] for k in ("h", "d", "a", "o25", "u25")}
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            for o in mk.get("outcomes", []):
                if mk["key"] == "h2h":
                    k = "h" if o["name"] == home else "a" if o["name"] == away else "d"
                    prices[k].append(o["price"])
                elif mk["key"] == "totals" and float(o.get("point", 0)) == 2.5:
                    prices["o25" if o["name"] == "Over" else "u25"].append(o["price"])
    out = {}
    for k, v in prices.items():
        name = {"h": "h", "d": "d", "a": "a", "o25": "o25", "u25": "u25"}[k]
        out[f"odds_{name}"] = float(np.mean(v)) if v else np.nan
        out[f"max_{name}"] = float(np.max(v)) if v else np.nan
    return out


def attach(fixtures: pd.DataFrame, sport: str = "soccer_uefa_nations_league",
           offline: bool = False) -> pd.DataFrame:
    """Добавя коефициентите към мачовете (съпоставяне по имена и дата ±1 ден)."""
    events = fetch(sport, offline)
    if not events or fixtures.empty:
        return fixtures
    fx = fixtures.copy()
    teams = sorted(set(fx["home"]) | set(fx["away"]))
    matched = 0
    for ev in events:
        h = teammatch.match([data.INTL_ALIASES.get(ev["home_team"], ev["home_team"])], teams, cutoff=0.8)
        a = teammatch.match([data.INTL_ALIASES.get(ev["away_team"], ev["away_team"])], teams, cutoff=0.8)
        if not h or not a:
            continue
        day = pd.to_datetime(ev["commence_time"]).tz_convert(None).normalize()
        sel = (fx["home"] == h) & (fx["away"] == a) & (
            fx["date"].isna() | ((fx["date"] - day).abs() <= pd.Timedelta(days=1)))
        if sel.any():
            for col, val in _summarize(ev).items():
                fx.loc[sel, col] = val
            matched += 1
    print(f"  коефициенти за {matched} мача")
    return fx
