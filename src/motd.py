"""„Мач на деня“: един мач и един залог с коефициент поне MOTD_MIN_ODDS + анализ.

Кандидати са пазарите с реални коефициенти – краен резултат (1, X, 2) и над/под
2.5 гола. От тези с коефициент поне 2.00 се избира изходът с най-висока
вероятност по модела (при равенство – с по-голямо очаквано предимство).
За избрания мач се смята статистика от историята: форма, голове, xG, корнери,
почивка, мачове у дома/като гост и преките срещи.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import bank, names

LOG = config.LOG_DIR / "motd.csv"
COLS = bank.COLS


def _valid(c: dict) -> bool:
    o, p = c.get("odds"), c.get("p")
    return (o is not None and np.isfinite(o) and o >= config.MOTD_MIN_ODDS
            and p is not None and np.isfinite(p))


def daily_picks(matches: list[dict]) -> dict[str, dict]:
    by_day: dict[str, list[dict]] = {}
    for m in matches:
        if not m["date"]:
            continue
        o = m.get("odds") or {}
        base = {"date": m["date"], "div": m["div"], "home": m["home"], "away": m["away"]}
        by_day.setdefault(m["date"], []).extend(bank._candidates(
            base, m["p_home"], m["p_draw"], m["p_away"], m["over_2.5"],
            o.get("odds_h"), o.get("odds_d"), o.get("odds_a"), o.get("odds_o25"), o.get("odds_u25")))
    out = {}
    for d, cands in by_day.items():
        cands = [c for c in cands if _valid(c)]
        if cands:
            out[d] = max(cands, key=lambda c: (c["p"], c["p"] * c["odds"]))
    return out


def log_daily(picks: dict[str, dict], today: str):
    """Като при „Банка“: минали дни не се пипат, днешният избор се заключва
    (освен ако вече не отговаря на правилата), бъдещите се обновяват."""
    old = pd.read_csv(LOG, dtype={"date": str}) if LOG.exists() else pd.DataFrame(columns=COLS)
    ok = old.apply(lambda r: _valid({"odds": float(r["odds"]), "p": float(r["p"])}), axis=1) \
        if len(old) else pd.Series(dtype=bool)
    keep = old[(old["date"] < today) | ((old["date"] == today) & ok)]
    have = set(keep["date"])
    new = pd.DataFrame([{k: p[k] for k in COLS} for d, p in sorted(picks.items())
                        if d >= today and d not in have], columns=COLS)
    out = pd.concat([keep, new], ignore_index=True) if len(keep) else new
    if out.empty:
        return
    LOG.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values("date").to_csv(LOG, index=False)


# --------------------------------------------------------------------------- #
# Статистика
# --------------------------------------------------------------------------- #
def _group_hist(hist: pd.DataFrame, div: str) -> pd.DataFrame:
    """Историята на групата, в която е мачът (държава / ШЛ / национални отбори)."""
    if div in (config.NL_CODE, "INT"):
        divs = [config.NL_CODE, "INT"]
    elif div == config.CL_CODE:
        divs = [config.CL_CODE]
    else:
        divs = list(config.COUNTRIES.get(config.DIV_COUNTRY.get(div, ""), {div: ""}))
    return hist[hist["div"].isin(divs)].dropna(subset=["hg", "ag"])


def _rows(h: pd.DataFrame, team: str) -> pd.DataFrame:
    """Мачовете на отбора от негова гледна точка (gf/ga, venue, opp)."""
    hm = h[h["home"] == team]
    aw = h[h["away"] == team]
    get = lambda d, c: d[c] if c in d else pd.Series(np.nan, index=d.index)
    a = pd.DataFrame({"date": hm["date"], "opp": hm["away"], "venue": "Д", "gf": hm["hg"], "ga": hm["ag"],
                      "xgf": get(hm, "hxg"), "xga": get(hm, "axg"), "cf": get(hm, "hc"), "ca": get(hm, "ac"),
                      "div": hm["div"]})
    b = pd.DataFrame({"date": aw["date"], "opp": aw["home"], "venue": "Г", "gf": aw["ag"], "ga": aw["hg"],
                      "xgf": get(aw, "axg"), "xga": get(aw, "hxg"), "cf": get(aw, "ac"), "ca": get(aw, "hc"),
                      "div": aw["div"]})
    return pd.concat([a, b]).sort_values("date", ascending=False)


def _mean(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return None if s.empty else round(float(s.mean()), 2)


def _share(mask: pd.Series):
    return None if len(mask) == 0 else round(float(mask.mean()), 3)


def team_stats(h: pd.DataFrame, team: str, date: pd.Timestamp, venue: str) -> dict:
    r = _rows(h[h["date"] < date], team)
    last10, last5 = r.head(10), r.head(5)
    res = np.where(r.gf > r.ga, "П", np.where(r.gf == r.ga, "Р", "З"))
    r = r.assign(res=res)
    pts = {"П": 3, "Р": 1, "З": 0}
    side = r[r["venue"] == venue].head(10)
    tot10 = last10.gf + last10.ga
    # серия без загуба / без победа
    streak_kind, streak = None, 0
    for x in r["res"]:
        if streak_kind is None:
            streak_kind = "без загуба" if x != "З" else "без победа"
        if (streak_kind == "без загуба" and x == "З") or (streak_kind == "без победа" and x == "П"):
            break
        streak += 1
    return {
        "name": names.display(team),
        "last5": [{"date": d.strftime("%Y-%m-%d"), "opp": names.display(o), "venue": v,
                   "score": f"{int(gf)}-{int(ga)}", "res": x}
                  for d, o, v, gf, ga, x in r.head(5)[["date", "opp", "venue", "gf", "ga", "res"]].itertuples(index=False)],
        "form": "".join(r.head(5)["res"]),
        "ppg5": round(float(np.mean([pts[x] for x in r.head(5)["res"]])), 2) if len(last5) else None,
        "gf10": _mean(last10.gf), "ga10": _mean(last10.ga),
        "xgf10": _mean(last10.xgf), "xga10": _mean(last10.xga),
        "cf10": _mean(last10.cf), "ca10": _mean(last10.ca),
        "over25": _share(tot10 > 2.5), "btts": _share((last10.gf > 0) & (last10.ga > 0)),
        "clean": _share(last10.ga == 0), "no_goal": _share(last10.gf == 0),
        "side_n": int(len(side)), "side_ppg": round(float(np.mean([pts[x] for x in side["res"]])), 2) if len(side) else None,
        "side_gf": _mean(side.gf), "side_ga": _mean(side.ga),
        "rest": int((date - r["date"].iloc[0]).days) if len(r) else None,
        "streak": f"{streak} мача {streak_kind}" if streak >= 3 else None,
        "n": int(len(r)),
    }


def h2h(h: pd.DataFrame, home: str, away: str, date: pd.Timestamp) -> list[dict]:
    m = h[(h["date"] < date) & (((h["home"] == home) & (h["away"] == away)) |
                                ((h["home"] == away) & (h["away"] == home)))]
    return [{"date": r.date.strftime("%Y-%m-%d"), "home": names.display(r.home), "away": names.display(r.away),
             "score": f"{int(r.hg)}-{int(r.ag)}"} for r in m.sort_values("date", ascending=False).head(6).itertuples()]


def _pct(x):
    return f"{round(100 * x)}%"


def insights(pick: dict, m: dict, hs: dict, as_: dict, meet: list[dict]) -> list[str]:
    """Кратък текстов анализ от числата."""
    out = []
    p, o = pick["p"], pick["odds"]
    imp = 1 / o
    edge = p * o - 1
    out.append(f"Моделът дава {_pct(p)} за „{pick['selection']}“, а коефициент {o:.2f} предполага {_pct(imp)}"
               + (f" – предимство {edge:+.0%}." if edge > 0 else f" – пазарът е по-оптимистичен ({edge:+.0%})."))
    out.append(f"Очаквани голове: {hs['name']} {m['xg_home']:.2f} – {m['xg_away']:.2f} {as_['name']} "
               f"(общо {m['xg_home'] + m['xg_away']:.2f}). Най-вероятен резултат: {m['top_scores'][0]['score']}.")
    for t, label in ((hs, "домакин"), (as_, "гост")):
        if t["form"]:
            line = f"{t['name']}: форма {t['form']} ({t['ppg5']:.1f} т./мач в последните 5)"
            if t["side_ppg"] is not None and t["side_n"] >= 3:
                line += f", като {label} {t['side_ppg']:.1f} т./мач ({t['side_n']} мача)"
            out.append(line + ".")
        if t["streak"]:
            out.append(f"{t['name']} е {t['streak']} поред.")
        if t["xgf10"] is not None and t["gf10"] is not None and abs(t["gf10"] - t["xgf10"]) >= 0.35:
            more = t["gf10"] > t["xgf10"]
            out.append(f"{t['name']} вкарва {'повече' if more else 'по-малко'} от xG ({t['gf10']:.1f} срещу "
                       f"{t['xgf10']:.1f} на мач) – {'вероятен спад' if more else 'възможен подем'} напред.")
    ov = [t["over25"] for t in (hs, as_) if t["over25"] is not None]
    if ov:
        out.append(f"Над 2.5 гола в последните 10 мача: {hs['name']} {_pct(hs['over25'] or 0)}, "
                   f"{as_['name']} {_pct(as_['over25'] or 0)}.")
    rh, ra = hs["rest"], as_["rest"]
    if rh is not None and ra is not None and abs(rh - ra) >= 3 and min(rh, ra) <= 4:
        fresh = hs if rh > ra else as_
        out.append(f"{fresh['name']} има повече почивка ({max(rh, ra)} срещу {min(rh, ra)} дни).")
    if meet:
        w = d = 0
        for x in meet:
            a, b = map(int, x["score"].split("-"))
            mine, theirs = (a, b) if x["home"] == hs["name"] else (b, a)
            w += mine > theirs
            d += mine == theirs
        l_ = len(meet) - w - d
        pl = lambda n, one, many: f"{n} {one if n == 1 else many}"
        out.append(f"Преки срещи (последни {len(meet)}): {hs['name']} – {pl(w, 'победа', 'победи')}, "
                   f"{pl(d, 'равен', 'равни')}, {pl(l_, 'загуба', 'загуби')}.")
    if m.get("low_confidence"):
        out.append("Внимание: малко данни за поне един от отборите – прогнозата е по-несигурна.")
    return out


def _settle(market, selection, hg, ag):
    return bank._settle(market, selection, hg, ag)


def build(result: dict, hist: pd.DataFrame, today: str) -> dict:
    """Данните за таба: текущият мач на деня с анализ + история на изборите."""
    if not LOG.exists():
        return {"min_odds": config.MOTD_MIN_ODDS, "current": None, "history": []}
    log = pd.read_csv(LOG, dtype={"date": str})
    histo = []
    h = hist.copy()
    if len(h):
        k = h[["date", "div", "home", "away", "hg", "ag"]].copy()
        k["date"] = k["date"].dt.strftime("%Y-%m-%d")
        log = log.merge(k.drop_duplicates(["date", "div", "home", "away"]), on=["date", "div", "home", "away"], how="left")
    else:
        log["hg"] = log["ag"] = np.nan
    for _, r in log.sort_values("date").iterrows():
        histo.append(bank._record(r, r["hg"], r["ag"], config.DIV_NAMES.get(r["div"], r["div"])))
    # текущият: първият избор от днес нататък, за който имаме прогноза
    current = None
    by_key = {(m["date"], m["home"], m["away"]): m for m in result["matches"]}
    for _, r in log[log["date"] >= today].sort_values("date").iterrows():
        m = by_key.get((r["date"], r["home"], r["away"]))
        if m is None:
            continue
        g = _group_hist(h, r["div"]) if len(h) else h
        if r["div"] in config.UNDERSTAT_LEAGUES or config.DIV_COUNTRY.get(r["div"]) in config.COUNTRIES:
            try:                      # xG от кеша (сваля се при настройката на модела)
                from src import xg
                g = xg.attach(g, offline=True)
            except Exception:         # noqa: BLE001
                pass
        date = pd.Timestamp(r["date"])
        hs = team_stats(g, r["home"], date, "Д")
        as_ = team_stats(g, r["away"], date, "Г")
        meet = h2h(g, r["home"], r["away"], date)
        pick = {k: (float(r[k]) if k in ("p", "odds") else r[k]) for k in COLS}
        current = {"pick": pick, "match": m, "home": hs, "away": as_, "h2h": meet,
                   "edge": pick["p"] * pick["odds"] - 1,
                   "insights": insights(pick, m, hs, as_, meet)}
        break
    return {"min_odds": config.MOTD_MIN_ODDS, "current": current, "history": histo}
