"""Банка: по един залог на ден – най-вероятният изход сред всички мачове за деня.

Кандидати са само пазарите с реални коефициенти: краен резултат (1, X, 2) и
над/под 2.5 гола, с коефициент поне BANK_MIN_ODDS. Избира се изходът с най-висока
вероятност по модела (при равенство – с по-високия коефициент). Ако и той е под
BANK_MIN_PROB, за деня няма залог. Залогът е BANK_DAILY_PCT % от банката.
Ползва се средният коефициент на пазара (реално достъпен при повечето букмейкъри).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import names

DAILY_LOG = config.LOG_DIR / "daily_bets.csv"
COLS = ["date", "div", "home", "away", "market", "selection", "p", "odds"]


def _ok(o) -> bool:
    return o is not None and np.isfinite(o) and o > 1 and o >= config.BANK_MIN_ODDS


def _valid(c: dict) -> bool:
    """Отговаря ли залогът на текущите правила (коефициент и вероятност)."""
    p = c.get("p")
    return (_ok(c.get("odds")) and p is not None and np.isfinite(p)
            and p >= config.BANK_MIN_PROB)


def _best(cands: list[dict]) -> dict | None:
    cands = [c for c in cands if _valid(c)]
    return max(cands, key=lambda c: (c["p"], c["odds"])) if cands else None


def _candidates(base: dict, p1, px, p2, po, o1, ox, o2, oo, ou) -> list[dict]:
    return [
        {**base, "market": "1X2", "selection": "1", "p": p1, "odds": o1},
        {**base, "market": "1X2", "selection": "X", "p": px, "odds": ox},
        {**base, "market": "1X2", "selection": "2", "p": p2, "odds": o2},
        {**base, "market": "Голове", "selection": "Над 2.5", "p": po, "odds": oo},
        {**base, "market": "Голове", "selection": "Под 2.5", "p": None if po is None else 1 - po, "odds": ou},
    ]


# --------------------------------------------------------------------------- #
# Реални прогнози
# --------------------------------------------------------------------------- #
def daily_picks(matches: list[dict]) -> dict[str, dict]:
    """date -> най-вероятният залог за деня (от прогнозите на сайта)."""
    by_day: dict[str, list[dict]] = {}
    for m in matches:
        if not m["date"]:
            continue
        o = m.get("odds") or {}
        base = {"date": m["date"], "div": m["div"], "home": m["home"], "away": m["away"]}
        by_day.setdefault(m["date"], []).extend(_candidates(
            base, m["p_home"], m["p_draw"], m["p_away"], m["over_2.5"],
            o.get("odds_h"), o.get("odds_d"), o.get("odds_a"), o.get("odds_o25"), o.get("odds_u25")))
    return {d: b for d, c in by_day.items() if (b := _best(c))}


def log_daily(picks: dict[str, dict], today: str):
    """Залогът за ден се „заключва“ в деня на мачовете: минали и днешни записи
    не се променят, бъдещите се заменят с най-новия избор."""
    old = pd.read_csv(DAILY_LOG, dtype={"date": str}) if DAILY_LOG.exists() else pd.DataFrame(columns=COLS)
    # минали дни – без промяна; днешният залог остава, само ако отговаря на
    # текущите правила (напр. след смяна на минималния коефициент)
    old_ok = old.apply(lambda r: _valid({"odds": float(r["odds"]), "p": float(r["p"])}), axis=1) \
        if len(old) else pd.Series(dtype=bool)
    keep = old[(old["date"] < today) | ((old["date"] == today) & old_ok)]
    have = set(keep["date"])
    new = pd.DataFrame([{k: p[k] for k in COLS} for d, p in sorted(picks.items())
                        if d >= today and d not in have], columns=COLS)
    out = pd.concat([keep, new], ignore_index=True) if len(keep) else new
    if out.empty:
        return
    DAILY_LOG.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values("date").to_csv(DAILY_LOG, index=False)


def _settle(market: str, selection: str, hg, ag) -> bool | None:
    if hg is None or pd.isna(hg):
        return None
    if market == "1X2":
        return ("1" if hg > ag else "X" if hg == ag else "2") == selection
    over = hg + ag > 2.5
    return over if selection.startswith("Над") else not over


def _record(r, hg, ag, league: str) -> dict:
    win = _settle(r["market"], r["selection"], hg, ag)
    return {"date": str(r["date"]), "league": league,
            "home": names.display(r["home"]), "away": names.display(r["away"]),
            "market": r["market"], "selection": r["selection"],
            "p": float(r["p"]), "odds": float(r["odds"]),
            "profit": None if win is None else (float(r["odds"]) - 1 if win else -1.0),
            "score": None if hg is None or pd.isna(hg) else f"{int(hg)}-{int(ag)}"}


def history(hist: pd.DataFrame) -> list[dict]:
    """Дневните залози от лога с резултата им (profit = печалба при залог 1)."""
    if not DAILY_LOG.exists():
        return []
    log = pd.read_csv(DAILY_LOG, dtype={"date": str})
    if log.empty:
        return []
    if len(hist):
        h = hist[["date", "div", "home", "away", "hg", "ag"]].copy()
        h["date"] = h["date"].dt.strftime("%Y-%m-%d")
        log = log.merge(h.drop_duplicates(["date", "div", "home", "away"]),
                        on=["date", "div", "home", "away"], how="left")
    else:
        log["hg"] = log["ag"] = np.nan
    return [_record(r, r["hg"], r["ag"], config.DIV_NAMES.get(r["div"], r["div"]))
            for _, r in log.sort_values("date").iterrows()]


# --------------------------------------------------------------------------- #
# Бектест – същото правило върху walk-forward прогнозите (без параметри за
# настройка, затова целият период е „извън извадката“)
# --------------------------------------------------------------------------- #
def backtest_daily(df: pd.DataFrame) -> list[dict]:
    """Най-добрият кандидат за всеки ден от прогнозите на една държава."""
    out = []
    if df.empty:
        return out
    for date, g in df.groupby(df["date"].dt.strftime("%Y-%m-%d")):
        cands = []
        for r in g.itertuples():
            base = {"date": date, "div": r.div, "home": r.home, "away": r.away, "hg": r.hg, "ag": r.ag}
            cands += _candidates(base, r.ph, r.pd, r.pa, r.po, r.oh, r.od, r.oa, r.oo, r.ou)
        b = _best(cands)
        if b:
            out.append(_record(b, b["hg"], b["ag"], config.DIV_NAMES.get(b["div"], b["div"])))
    return out


def best_per_day(records: list[dict]) -> list[dict]:
    """От кандидатите на всички държави – по един (най-вероятният) на ден."""
    best: dict[str, dict] = {}
    for r in records:
        b = best.get(r["date"])
        if b is None or (r["p"], r["odds"]) > (b["p"], b["odds"]):
            best[r["date"]] = r
    return [best[d] for d in sorted(best)]
