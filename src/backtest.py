"""Walk-forward бектест: моделът се обучава само с мачове преди всеки период
и се сравнява с пазара (коефициентите без маржа)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import data, markets
from src.predict import fit_group


def _metrics(P, y):
    ll = -np.mean(np.log(np.clip((P * y).sum(1), 1e-9, 1)))
    br = np.mean(((P - y) ** 2).sum(1))
    return float(ll), float(br)


def backtest_country(hist: pd.DataFrame, test_days: int = 240, step: int = 7) -> dict | None:
    end = hist["date"].max()
    start = end - pd.Timedelta(days=test_days)
    rows = []
    t = start
    while t < end:
        train = hist[hist["date"] < t]
        test = hist[(hist["date"] >= t) & (hist["date"] < t + pd.Timedelta(days=step))]
        t += pd.Timedelta(days=step)
        if len(test) == 0 or len(train) < 300:
            continue
        models = fit_group(train, t)
        dc, cm = models["goals"], models["corners"]
        for _, r in test.iterrows():
            lam, mu = dc.rates(r.home, r.away)
            g = markets.goal_markets(markets.score_matrix(lam, mu, dc.rho), lam, mu)
            row = {"hg": r.hg, "ag": r.ag, "ph": g["p_home"], "pd": g["p_draw"], "pa": g["p_away"],
                   "po": g["over_2.5"], "pb": g["btts_yes"],
                   "oh": r.odds_h, "od": r.odds_d, "oa": r.odds_a, "oo": r.odds_o25, "ou": r.odds_u25,
                   "mh": r.max_h, "md": r.max_d, "ma": r.max_a, "mo": r.max_o25, "mu": r.max_u25}
            if cm is not None and pd.notna(r.hc) and cm.knows(r.home) and cm.knows(r.away):
                row["pc"] = markets.corner_markets(*cm.rates(r.home, r.away))["c_over_9.5"]
                row["tc"] = r.hc + r.ac
            rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    res = np.where(df.hg > df.ag, 0, np.where(df.hg == df.ag, 1, 2))
    y = np.eye(3)[res]
    P = df[["ph", "pd", "pa"]].to_numpy()
    out = {"n": len(df)}
    out["model_logloss"], out["model_brier"] = _metrics(P, y)
    out["acc_1x2"] = float((P.argmax(1) == res).mean())
    out["acc_ou25"] = float(((df.po > 0.5) == (df.hg + df.ag > 2.5)).mean())
    out["acc_btts"] = float(((df.pb > 0.5) == ((df.hg > 0) & (df.ag > 0))).mean())
    if "pc" in df:
        c = df.dropna(subset=["pc"])
        out["acc_corners95"] = float(((c.pc > 0.5) == (c.tc > 9.5)).mean())

    mk = df.dropna(subset=["oh", "od", "oa"])
    if len(mk):
        inv = 1 / mk[["oh", "od", "oa"]].to_numpy()
        M = inv / inv.sum(1, keepdims=True)
        ym = y[mk.index]
        out["market_logloss"], out["market_brier"] = _metrics(M, ym)
        # value залози със същите правила като на сайта
        profits = []
        for j, (i, r) in enumerate(mk.iterrows()):
            for k, (p, o, m) in enumerate([(r.ph, r.oh, r.mh), (r.pd, r.od, r.md), (r.pa, r.oa, r.ma)]):
                v = markets.value_bet("1X2", str(k), p, o, m, M[j, k])
                if v:
                    profits.append(v["odds"] - 1 if res[i] == k else -1)
            tot = r.hg + r.ag
            ou = markets.implied([r.oo, r.ou]) or [None, None]
            for over, p, o, m, pm in [(True, r.po, r.oo, r.mo, ou[0]), (False, 1 - r.po, r.ou, r.mu, ou[1])]:
                v = markets.value_bet("OU", "", p, o, m, pm)
                if v:
                    profits.append(v["odds"] - 1 if (tot > 2.5) == over else -1)
        if profits:
            out.update({"bets": len(profits), "roi": float(np.mean(profits)),
                        "profit": float(np.sum(profits))})
    return out


def backtest_nations_league(offline: bool = False, days: int = 900, step: int = 30) -> dict | None:
    """Walk-forward върху мачовете от Лигата на нациите (без коефициенти)."""
    from src.models import DixonColes
    hist, _ = data.load_internationals(offline)
    if len(hist) < 500:
        return None
    end = hist["date"].max()
    test_all = hist[(hist["div"] == config.NL_CODE) & (hist["date"] > end - pd.Timedelta(days=days))]
    rows, t = [], test_all["date"].min()
    while t is not None and t <= end:
        test = test_all[(test_all["date"] >= t) & (test_all["date"] < t + pd.Timedelta(days=step))]
        if len(test):
            dc = DixonColes(xi=config.INTL_TIME_DECAY_XI).fit(hist[hist["date"] < t], t)
            for _, r in test.iterrows():
                lam, mu = dc.rates(r.home, r.away, neutral=bool(r.get("neutral", False)))
                g = markets.goal_markets(markets.score_matrix(lam, mu, dc.rho), lam, mu)
                rows.append({"hg": r.hg, "ag": r.ag, "ph": g["p_home"], "pd": g["p_draw"],
                             "pa": g["p_away"], "po": g["over_2.5"], "pb": g["btts_yes"]})
        t += pd.Timedelta(days=step)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    res = np.where(df.hg > df.ag, 0, np.where(df.hg == df.ag, 1, 2))
    y = np.eye(3)[res]
    P = df[["ph", "pd", "pa"]].to_numpy()
    out = {"n": len(df)}
    out["model_logloss"], out["model_brier"] = _metrics(P, y)
    out["acc_1x2"] = float((P.argmax(1) == res).mean())
    out["acc_ou25"] = float(((df.po > 0.5) == (df.hg + df.ag > 2.5)).mean())
    out["acc_btts"] = float(((df.pb > 0.5) == ((df.hg > 0) & (df.ag > 0))).mean())
    return out


def run(offline: bool = False) -> dict:
    results = {}
    for country, divs in config.COUNTRIES.items():
        print(f"Бектест: {country}")
        hist = data.load_history(list(divs), offline=offline)
        if len(hist) < 600:
            continue
        r = backtest_country(hist)
        if r:
            results[country] = r
            print("  ", {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
    print("Бектест: Лига на нациите")
    r = backtest_nations_league(offline)
    if r:
        results["Лига на нациите"] = r
        print("  ", {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
    return results
