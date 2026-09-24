"""Сваляне и нормализиране на данните.

Източници:
  * football-data.co.uk – исторически резултати, корнери и коефициенти за
    европейските първенства + fixtures.csv с предстоящите мачове и коефициенти.
  * football-data.org (API v4) – Шампионска лига (резултати и програма).
"""
from __future__ import annotations

import io
import json
import os
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config

FDCO_HISTORY = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
FDCO_FIXTURES = "https://www.football-data.co.uk/fixtures.csv"
INTL_RESULTS = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
FDORG_MATCHES = "https://api.football-data.org/v4/competitions/{code}/matches?season={year}"

HEADERS = {"User-Agent": "Mozilla/5.0 (football-predictor; personal project)"}

# Нормализирани колони, които ползват моделите
COLUMNS = ["date", "time", "div", "home", "away", "hg", "ag", "hc", "ac",
           "odds_h", "odds_d", "odds_a", "odds_o25", "odds_u25",
           "max_h", "max_d", "max_a", "max_o25", "max_u25"]

# Имена на колоните с коефициенти (по приоритет – средни от пазара, после Bet365)
ODDS_SOURCES = {
    "odds_h": ["AvgH", "BbAvH", "B365H"],
    "odds_d": ["AvgD", "BbAvD", "B365D"],
    "odds_a": ["AvgA", "BbAvA", "B365A"],
    "odds_o25": ["Avg>2.5", "BbAv>2.5", "B365>2.5"],
    "odds_u25": ["Avg<2.5", "BbAv<2.5", "B365<2.5"],
    "max_h": ["MaxH", "BbMxH"],
    "max_d": ["MaxD", "BbMxD"],
    "max_a": ["MaxA", "BbMxA"],
    "max_o25": ["Max>2.5", "BbMx>2.5"],
    "max_u25": ["Max<2.5", "BbMx<2.5"],
}


# --------------------------------------------------------------------------- #
# Помощни
# --------------------------------------------------------------------------- #
def season_start_year(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return today.year if today.month >= 7 else today.year - 1


def season_codes(n: int = config.N_SEASONS, today: dt.date | None = None) -> list[str]:
    """Кодове на сезоните във формат на football-data.co.uk, напр. '2627'."""
    start = season_start_year(today)
    return [f"{(y) % 100:02d}{(y + 1) % 100:02d}" for y in range(start - n + 1, start + 1)]


_failures: dict[str, int] = {}


def _http_get(url: str, headers: dict | None = None, retries: int = 3) -> bytes:
    host = url.split("/")[2]
    if _failures.get(host, 0) >= 3:      # сайтът е недостъпен – ползваме кеша
        raise RuntimeError(f"{host} е недостъпен, ползва се кешът ({url})")
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers={**HEADERS, **(headers or {})}, timeout=30)
            if r.status_code == 429:           # rate limit (football-data.org: 10/мин)
                time.sleep(15)
                continue
            r.raise_for_status()
            _failures[host] = 0
            return r.content
        except requests.RequestException as e:  # noqa: PERF203
            last = e
            time.sleep(2 * (i + 1))
    _failures[host] = _failures.get(host, 0) + 1
    raise RuntimeError(f"Неуспешно сваляне: {url} ({last})")


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    raise ValueError("Непознато кодиране на CSV")


def normalize_fdco(df: pd.DataFrame) -> pd.DataFrame:
    """Превръща CSV от football-data.co.uk в общия ни формат."""
    df = df.rename(columns=lambda c: str(c).strip().lstrip("﻿"))
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "Date"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce"),
        "time": df["Time"] if "Time" in df else "",
        "div": df["Div"],
        "home": df["HomeTeam"].astype(str).str.strip(),
        "away": df["AwayTeam"].astype(str).str.strip(),
        "hg": pd.to_numeric(df.get("FTHG"), errors="coerce") if "FTHG" in df else np.nan,
        "ag": pd.to_numeric(df.get("FTAG"), errors="coerce") if "FTAG" in df else np.nan,
        "hc": pd.to_numeric(df.get("HC"), errors="coerce") if "HC" in df else np.nan,
        "ac": pd.to_numeric(df.get("AC"), errors="coerce") if "AC" in df else np.nan,
    })
    for col, sources in ODDS_SOURCES.items():
        out[col] = np.nan
        for s in reversed(sources):          # последният има най-висок приоритет
            if s in df:
                vals = pd.to_numeric(df[s], errors="coerce")
                out[col] = vals.where(vals.notna(), out[col])
    out["time"] = out["time"].fillna("").astype(str)
    return out.dropna(subset=["date"])[COLUMNS]


# --------------------------------------------------------------------------- #
# football-data.co.uk
# --------------------------------------------------------------------------- #
def fetch_league_season(div: str, season: str, offline: bool = False) -> pd.DataFrame | None:
    path = config.RAW_DIR / f"{season}_{div}.csv"
    current = season == season_codes(1)[0]
    # минали сезони се свалят веднъж, текущият – при всяко пускане
    if not offline and (current or not path.exists()):
        try:
            raw = _http_get(FDCO_HISTORY.format(season=season, div=div))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        except RuntimeError as e:
            print(f"  ! {e}")
    if not path.exists():
        return None
    return normalize_fdco(_read_csv_bytes(path.read_bytes()))


def load_history(divs: list[str], seasons: list[str] | None = None,
                 offline: bool = False) -> pd.DataFrame:
    seasons = seasons or season_codes()
    frames = []
    for div in divs:
        for s in seasons:
            df = fetch_league_season(div, s, offline)
            if df is not None and len(df):
                df["season"] = s
                frames.append(df)
    if not frames:
        return pd.DataFrame(columns=COLUMNS + ["season"])
    hist = pd.concat(frames, ignore_index=True)
    hist = hist.dropna(subset=["hg", "ag"])
    return hist.sort_values("date").reset_index(drop=True)


def load_fixtures(divs: list[str], offline: bool = False) -> pd.DataFrame:
    path = config.RAW_DIR / "fixtures.csv"
    if not offline:
        try:
            raw = _http_get(FDCO_FIXTURES)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        except RuntimeError as e:
            print(f"  ! {e}")
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    fx = normalize_fdco(_read_csv_bytes(path.read_bytes()))
    return fx[fx["div"].isin(divs)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# football-data.org – Шампионска лига
# --------------------------------------------------------------------------- #
def _parse_fdorg(payload: dict) -> pd.DataFrame:
    rows = []
    for m in payload.get("matches", []):
        utc = pd.to_datetime(m["utcDate"])
        ft = (m.get("score") or {}).get("fullTime") or {}
        rows.append({
            "date": utc.tz_convert(None).normalize(),
            "time": utc.strftime("%H:%M") + " UTC",
            "div": config.CL_CODE,
            "home": (m.get("homeTeam") or {}).get("shortName") or (m.get("homeTeam") or {}).get("name"),
            "away": (m.get("awayTeam") or {}).get("shortName") or (m.get("awayTeam") or {}).get("name"),
            "hg": ft.get("home") if m.get("status") == "FINISHED" else np.nan,
            "ag": ft.get("away") if m.get("status") == "FINISHED" else np.nan,
            "status": m.get("status"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["home", "away"])
    for c in COLUMNS:
        if c not in df:
            df[c] = np.nan
    return df


def load_champions_league(offline: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Връща (история, предстоящи) за ШЛ."""
    key = os.environ.get("FOOTBALL_DATA_ORG_KEY", "").strip()
    start = season_start_year()
    frames = []
    for year in range(start - config.N_SEASONS + 1, start + 1):
        path = config.RAW_DIR / f"CL_{year}.json"
        current = year == start
        if not offline and key and (current or not path.exists()):
            try:
                raw = _http_get(FDORG_MATCHES.format(code=config.CL_CODE, year=year),
                                headers={"X-Auth-Token": key})
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                time.sleep(6.5)               # безплатният план: 10 заявки/мин
            except RuntimeError as e:
                print(f"  ! ШЛ {year}: {e}")
        if path.exists():
            frames.append(_parse_fdorg(json.loads(path.read_text("utf-8"))))
    if not frames:
        if not key and not offline:
            print("  ! Няма FOOTBALL_DATA_ORG_KEY – Шампионската лига се пропуска.")
        empty = pd.DataFrame(columns=COLUMNS)
        return empty, empty
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "home", "away"])
    hist = df[df["status"] == "FINISHED"].dropna(subset=["hg", "ag"])
    fut = df[df["status"].isin(["SCHEDULED", "TIMED"])]
    return (hist[COLUMNS].sort_values("date").reset_index(drop=True),
            fut[COLUMNS].sort_values("date").reset_index(drop=True))


def load_fdorg_fixtures(offline: bool = False) -> pd.DataFrame:
    """Предстоящи мачове от football-data.org за първите дивизии (без коефициенти).
    Колоните home_variants/away_variants пазят всички варианти на имената
    за съпоставяне с football-data.co.uk."""
    key = os.environ.get("FOOTBALL_DATA_ORG_KEY", "").strip()
    rows = []
    today = dt.date.today()
    dto = today + dt.timedelta(days=config.DAYS_AHEAD)
    for div, code in config.FDORG_LEAGUES.items():
        path = config.RAW_DIR / f"fdorg_fixtures_{code}.json"
        if not offline and key:
            url = (f"https://api.football-data.org/v4/competitions/{code}/matches"
                   f"?dateFrom={today.isoformat()}&dateTo={dto.isoformat()}")
            try:
                raw = _http_get(url, headers={"X-Auth-Token": key})
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                time.sleep(6.5)
            except RuntimeError as e:
                print(f"  ! Програма {code}: {e}")
        if not path.exists():
            continue
        for m in json.loads(path.read_text("utf-8")).get("matches", []):
            if m.get("status") not in ("SCHEDULED", "TIMED"):
                continue
            ht, at = m.get("homeTeam") or {}, m.get("awayTeam") or {}
            utc = pd.to_datetime(m["utcDate"])
            rows.append({
                "date": utc.tz_convert(None).normalize(), "time": utc.strftime("%H:%M") + " UTC",
                "div": div,
                "home_variants": [ht.get("shortName"), ht.get("name"), ht.get("tla")],
                "away_variants": [at.get("shortName"), at.get("name"), at.get("tla")],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Национални отбори – Лига на нациите
# --------------------------------------------------------------------------- #
def load_internationals(offline: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Връща (история на всички международни мачове, предстоящи мачове от Лигата на нациите)."""
    path = config.RAW_DIR / "international_results.csv"
    if not offline:
        try:
            raw = _http_get(INTL_RESULTS)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        except RuntimeError as e:
            print(f"  ! {e}")
    empty = pd.DataFrame(columns=COLUMNS + ["neutral", "wt"])
    if not path.exists():
        return empty, empty
    df = _read_csv_bytes(path.read_bytes())
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=config.INTL_YEARS)
    df = df[df["date"] >= cutoff].copy()
    is_nl = df["tournament"].astype(str).str.contains("Nations League", case=False)
    out = pd.DataFrame({
        "date": df["date"], "time": "",
        "div": np.where(is_nl, config.NL_CODE, "INT"),
        "home": df["home_team"].astype(str).str.strip(),
        "away": df["away_team"].astype(str).str.strip(),
        "hg": pd.to_numeric(df["home_score"], errors="coerce"),
        "ag": pd.to_numeric(df["away_score"], errors="coerce"),
        "neutral": df["neutral"].astype(str).str.upper().eq("TRUE"),
        "wt": np.where(df["tournament"].astype(str).eq("Friendly"), config.FRIENDLY_WEIGHT, 1.0),
    })
    for c in COLUMNS:
        if c not in out:
            out[c] = np.nan
    hist = out.dropna(subset=["hg", "ag"])
    # предстоящите мачове са редове без резултат
    fut = out[out["hg"].isna() & (out["div"] == config.NL_CODE)]
    if config.NL_MANUAL_FIXTURES.exists():
        man = pd.read_csv(config.NL_MANUAL_FIXTURES)
        man["date"] = pd.to_datetime(man["date"], errors="coerce")
        man["div"] = config.NL_CODE
        man["neutral"] = (man["neutral"].astype(str).str.upper().eq("TRUE")
                          if "neutral" in man else False)
        man["time"] = man["time"].fillna("").astype(str) if "time" in man else ""
        man["wt"] = 1.0
        for c in COLUMNS:
            if c not in man:
                man[c] = np.nan
        fut = pd.concat([fut, man[fut.columns]], ignore_index=True)
    fut = fut.dropna(subset=["date"]).drop_duplicates(["date", "home", "away"])
    return (hist.sort_values("date").reset_index(drop=True),
            fut.sort_values("date").reset_index(drop=True))
