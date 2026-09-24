"""xG (очаквани голове) от Understat (understat.com).

Understat покрива само първите дивизии на Англия, Испания, Италия, Германия и
Франция (config.UNDERSTAT_LEAGUES). Данните се взимат от същите AJAX адреси,
които ползва самият сайт (неофициално API – може да се промени без
предупреждение). Ако свалянето не успее, се ползва кешът, а ако няма и кеш,
колоните с xG остават празни и моделът работи без тях.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
from src import data, teammatch

URL = "https://understat.com/getLeagueData/{league}/{year}"
HEADERS = {
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://understat.com/",
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
}


def _season_year(season: str) -> int:
    """'2526' -> 2025 (Understat обозначава сезона с годината, в която започва)."""
    return 2000 + int(str(season)[:2])


def fetch_season(league: str, year: int, current: bool, offline: bool = False) -> list[dict]:
    path = config.RAW_DIR / f"understat_{league}_{year}.json"
    # минали сезони се свалят веднъж, текущият – при всяко пускане
    if not offline and (current or not path.exists()):
        try:
            raw = data._http_get(URL.format(league=league, year=year), headers=HEADERS)
            json.loads(raw)                          # проверка, че е валиден JSON
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        except (RuntimeError, ValueError) as e:
            print(f"  ! xG {league} {year}: {e}")
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError:
        return []
    return (payload.get("dates") if isinstance(payload, dict) else payload) or []


def _parse(matches: list[dict]) -> pd.DataFrame:
    rows = []
    for m in matches:
        if str(m.get("isResult")).lower() not in ("true", "1"):
            continue
        try:
            rows.append({
                "date": pd.to_datetime(m["datetime"]).normalize(),
                "us_home": m["h"]["title"], "us_away": m["a"]["title"],
                "hxg": float(m["xG"]["h"]), "axg": float(m["xG"]["a"]),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return pd.DataFrame(rows, columns=["date", "us_home", "us_away", "hxg", "axg"])


def _map_names(us_names: set[str], teams: list[str]) -> dict[str, str]:
    out = {}
    for n in us_names:
        out[n] = n if n in teams else teammatch.match([n], teams, cutoff=0.8)
    return out


def attach(hist: pd.DataFrame, offline: bool = False) -> pd.DataFrame:
    """Добавя колоните hxg/axg към историята (NaN, където няма данни)."""
    hist = hist.copy()
    hist["hxg"] = np.nan
    hist["axg"] = np.nan
    if hist.empty or "season" not in hist:
        return hist
    current = data.season_codes(1)[0]
    for div, league in config.UNDERSTAT_LEAGUES.items():
        for season in sorted(hist.loc[hist["div"] == div, "season"].unique()):
            us = _parse(fetch_season(league, _season_year(season), season == current, offline))
            if us.empty:
                continue
            sel = (hist["div"] == div) & (hist["season"] == season)
            teams = sorted(set(hist.loc[sel, "home"]) | set(hist.loc[sel, "away"]))
            names = _map_names(set(us["us_home"]) | set(us["us_away"]), teams)
            missing = sorted(k for k, v in names.items() if v is None)
            if missing:
                print(f"  ! xG {league} {season}: несъпоставени отбори {missing}")
            us["home"] = us["us_home"].map(names)
            us["away"] = us["us_away"].map(names)
            us = us.dropna(subset=["home", "away"]).drop_duplicates(["home", "away"])
            # всяка двойка домакин–гост се среща веднъж в сезона
            key = hist.loc[sel, ["home", "away"]].reset_index()
            m = key.merge(us[["home", "away", "hxg", "axg"]], on=["home", "away"], how="inner")
            hist.loc[m["index"], "hxg"] = m["hxg"].to_numpy()
            hist.loc[m["index"], "axg"] = m["axg"].to_numpy()
            print(f"  xG {league} {season}: {len(m)} от {int(sel.sum())} мача")
    return hist
