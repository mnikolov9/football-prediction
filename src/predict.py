"""Основен процес: данни -> модели -> прогнози за предстоящите мачове."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

import config
from src import data, markets, names
from src.models import DixonColes, PoissonTeamModel

MIN_CORNER_ROWS = 300


def _f(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)


def fit_group(hist: pd.DataFrame, ref_date) -> dict:
    """Модели за една група (държава или ШЛ)."""
    dc = DixonColes().fit(hist, ref_date)
    cm = None
    corners = hist.dropna(subset=["hc", "ac"])
    if len(corners) >= MIN_CORNER_ROWS:
        cm = PoissonTeamModel().fit(corners, ref_date, hcol="hc", acol="ac")
    return {"goals": dc, "corners": cm}


def predict_match(row: pd.Series, models: dict) -> dict:
    dc, cm = models["goals"], models["corners"]
    neutral = bool(row.get("neutral", False)) if pd.notna(row.get("neutral", False)) else False
    lam, mu = dc.rates(row["home"], row["away"], neutral=neutral)
    m = markets.score_matrix(lam, mu, dc.rho)
    g = markets.goal_markets(m, lam, mu)

    res = {
        "date": row["date"].strftime("%Y-%m-%d"),
        "time": str(row.get("time") or ""),
        "div": row["div"],
        "league": config.DIV_NAMES.get(row["div"], row["div"]),
        "country": config.DIV_COUNTRY.get(row["div"], ""),
        "home": row["home"], "away": row["away"],
        "home_name": names.display(row["home"]), "away_name": names.display(row["away"]),
        **g,
        "low_confidence": bool(min(dc.n_matches.get(row["home"], 0),
                                   dc.n_matches.get(row["away"], 0)) < 8),
    }
    if cm is not None and cm.knows(row["home"]) and cm.knows(row["away"]):
        res.update(markets.corner_markets(*cm.rates(row["home"], row["away"])))

    # коефициенти, пазар и value
    odds = {k: _f(row.get(k)) for k in data.ODDS_SOURCES}
    res["odds"] = odds
    imp = markets.implied([odds["odds_h"], odds["odds_d"], odds["odds_a"]])
    res["market_1x2"] = imp
    imp_ou = markets.implied([odds["odds_o25"], odds["odds_u25"]])
    res["market_ou25"] = imp_ou

    vb = []
    im = imp or [None] * 3
    for sel, p, o, mx, pm in [("1", g["p_home"], odds["odds_h"], odds["max_h"], im[0]),
                              ("X", g["p_draw"], odds["odds_d"], odds["max_d"], im[1]),
                              ("2", g["p_away"], odds["odds_a"], odds["max_a"], im[2])]:
        v = markets.value_bet("1X2", sel, p, o, mx, pm)
        if v:
            vb.append(v)
    io_ = imp_ou or [None] * 2
    for sel, p, o, mx, pm in [("Над 2.5", g["over_2.5"], odds["odds_o25"], odds["max_o25"], io_[0]),
                              ("Под 2.5", g["under_2.5"], odds["odds_u25"], odds["max_u25"], io_[1])]:
        v = markets.value_bet("Голове", sel, p, o, mx, pm)
        if v:
            vb.append(v)
    if res["low_confidence"]:
        vb = []                         # не препоръчваме залози при малко данни
    res["value_bets"] = vb

    # основна препоръка 1X2
    probs = {"1": g["p_home"], "X": g["p_draw"], "2": g["p_away"]}
    res["pick"] = max(probs, key=probs.get)
    return res


def run(offline: bool = False, today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    ref = pd.Timestamp(today)
    horizon = ref + pd.Timedelta(days=config.DAYS_AHEAD)
    all_divs = [d for divs in config.COUNTRIES.values() for d in divs]

    print("Сваляне на програмата...")
    fixtures = data.load_fixtures(all_divs, offline)
    matches, ratings, histories = [], {}, []

    for country, divs in config.COUNTRIES.items():
        print(f"{country}: данни и модел...")
        hist = data.load_history(list(divs), offline=offline)
        hist = hist[hist["date"] < ref]
        if len(hist) < 200:
            print(f"  ! Недостатъчно данни за {country} ({len(hist)} мача)")
            continue
        histories.append(hist)
        models = fit_group(hist, ref)
        ratings[country] = models["goals"].ratings().head(40).round(3).to_dict("records")
        fx = fixtures[fixtures["div"].isin(divs) & (fixtures["date"] >= ref) & (fixtures["date"] <= horizon)]
        for _, row in fx.iterrows():
            matches.append(predict_match(row, models))

    print("Шампионска лига...")
    cl_hist, cl_fx = data.load_champions_league(offline)
    cl_hist = cl_hist[cl_hist["date"] < ref] if len(cl_hist) else cl_hist
    if len(cl_hist) >= 150:
        histories.append(cl_hist)
        models = fit_group(cl_hist, ref)
        cl_fx = cl_fx[(cl_fx["date"] >= ref) & (cl_fx["date"] <= horizon)]
        for _, row in cl_fx.iterrows():
            matches.append(predict_match(row, models))

    print("Лига на нациите...")
    intl_hist, nl_fx = data.load_internationals(offline)
    intl_hist = intl_hist[intl_hist["date"] < ref] if len(intl_hist) else intl_hist
    nl_fx = nl_fx[(nl_fx["date"] >= ref) & (nl_fx["date"] <= horizon)] if len(nl_fx) else nl_fx
    if len(intl_hist) >= 300:
        histories.append(intl_hist)
        dc = DixonColes(xi=config.INTL_TIME_DECAY_XI).fit(intl_hist, ref)
        models = {"goals": dc, "corners": None}
        rt = dc.ratings().head(40).round(3)
        rt["team"] = rt["team"].map(names.display)
        ratings["Национални отбори"] = rt.to_dict("records")
        for _, row in nl_fx.iterrows():
            matches.append(predict_match(row, models))
        print(f"  {len(nl_fx)} предстоящи мача")

    matches.sort(key=lambda m: (m["date"], m["time"], m["div"]))
    value = sorted(
        [{**v, **{k: m[k] for k in ("date", "time", "league", "home", "away", "home_name", "away_name")}}
         for m in matches for v in m["value_bets"]],
        key=lambda v: -v["edge"])

    result = {
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "matches": matches,
        "value_bets": value,
        "ratings": ratings,
    }
    all_hist = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    return result, all_hist
