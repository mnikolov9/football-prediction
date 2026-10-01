"""Track record: пази публикуваните прогнози и ги оценява след мачовете."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
from src import names

PRED_LOG = config.LOG_DIR / "predictions.csv"
BET_LOG = config.LOG_DIR / "value_bets.csv"
KEY = ["date", "div", "home", "away"]


def _load(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _save_merged(path, new: pd.DataFrame, today: str, keys: set | None = None):
    """keys – всички мачове в текущата прогноза. Записите за тях от днес нататък се
    заменят с най-новите; така value залог, който вече не е value, отпада."""
    old = _load(path)
    if len(old):
        if keys is None:
            keys = set(map(tuple, new[KEY].astype(str).to_numpy())) if len(new) else set()
        stale = old["date"].astype(str).ge(today) & old[KEY].astype(str).apply(tuple, axis=1).isin(keys)
        old = old[~stale]
    out = pd.concat([old, new], ignore_index=True) if len(old) else new
    if out.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def log_predictions(result: dict, today: str):
    rows, bets = [], []
    for m in result["matches"]:
        if not m["date"]:
            continue                      # мачове без дата не влизат в track record-а
        rows.append({
            **{k: m[k] for k in KEY},
            "p_home": m["p_home"], "p_draw": m["p_draw"], "p_away": m["p_away"],
            "over_2.5": m["over_2.5"], "btts_yes": m["btts_yes"],
            "c_over_9.5": m.get("c_over_9.5"), "pick": m["pick"],
            "odds_h": m["odds"]["odds_h"], "odds_d": m["odds"]["odds_d"], "odds_a": m["odds"]["odds_a"],
        })
        for v in m["value_bets"]:
            bets.append({**{k: m[k] for k in KEY}, "market": v["market"],
                         "selection": v["selection"], "odds": v["odds"], "p": v["p"]})
    keys = {tuple(str(r[k]) for k in KEY) for r in rows}
    if rows:
        _save_merged(PRED_LOG, pd.DataFrame(rows), today, keys)
    if rows and (bets or BET_LOG.exists()):
        _save_merged(BET_LOG, pd.DataFrame(bets, columns=KEY + ["market", "selection", "odds", "p"]), today, keys)


def _settle(row) -> float | None:
    hg, ag = row["hg"], row["ag"]
    if pd.isna(hg):
        return None
    if row["market"] == "1X2":
        res = "1" if hg > ag else "X" if hg == ag else "2"
        win = res == row["selection"]
    else:
        over = hg + ag > 2.5
        win = over if row["selection"].startswith("Над") else not over
    return row["odds"] - 1 if win else -1.0


def _stats(df: pd.DataFrame, b: pd.DataFrame | None) -> dict:
    res = np.where(df.hg > df.ag, "1", np.where(df.hg == df.ag, "X", "2"))
    P = df[["p_home", "p_draw", "p_away"]].to_numpy()
    y = np.stack([res == "1", res == "X", res == "2"], 1).astype(float)
    out = {
        "n": int(len(df)),
        "acc_1x2": float((df["pick"].astype(str) == res).mean()),
        "logloss_1x2": float(-np.mean(np.log(np.clip((P * y).sum(1), 1e-9, 1)))),
        "brier_1x2": float(np.mean(((P - y) ** 2).sum(1))),
        "acc_ou25": float(((df["over_2.5"] > 0.5) == (df.hg + df.ag > 2.5)).mean()),
        "acc_btts": float(((df["btts_yes"] > 0.5) == ((df.hg > 0) & (df.ag > 0))).mean()),
    }
    c = df.dropna(subset=["c_over_9.5", "hc"])
    if len(c):
        out["acc_corners95"] = float(((c["c_over_9.5"] > 0.5) == (c.hc + c.ac > 9.5)).mean())
    if b is not None and len(b):
        out.update({"bets": int(len(b)), "profit": float(b.profit.sum()),
                    "roi": float(b.profit.mean()), "bet_hit": float((b.profit > 0).mean())})
    return out


def evaluate(hist: pd.DataFrame) -> dict:
    """Обща статистика + разбивка по състезания (ключ "by_group")."""
    preds = _load(PRED_LOG)
    if preds.empty or hist.empty:
        return {}
    h = hist[["date", "div", "home", "away", "hg", "ag", "hc", "ac"]].copy()
    h["date"] = h["date"].dt.strftime("%Y-%m-%d")
    df = preds.merge(h, on=KEY, how="inner").dropna(subset=["hg"])
    if df.empty:
        return {"n": 0}
    bets = _load(BET_LOG)
    b = None
    if len(bets):
        b = bets.merge(h, on=KEY, how="inner")
        b["profit"] = b.apply(_settle, axis=1)
        b = b.dropna(subset=["profit"])
    out = _stats(df, b)
    group = lambda d: d["div"].map(lambda x: config.DIV_NAMES.get(x, x) if x in (config.CL_CODE, config.NL_CODE)
                                   else config.DIV_COUNTRY.get(x, x))
    df["group"] = group(df)
    if b is not None and len(b):
        b["group"] = group(b)
    out["by_group"] = {g: _stats(d, b[b["group"] == g] if b is not None and len(b) else None)
                       for g, d in df.groupby("group")}
    return out


def bets_history(hist: pd.DataFrame) -> list[dict]:
    """Всички публикувани value залози с резултата им – за таб „Банка“.
    profit е печалбата при залог 1 единица; None = мачът още не е изигран."""
    bets = _load(BET_LOG)
    if bets.empty:
        return []
    if len(hist):
        h = hist[["date", "div", "home", "away", "hg", "ag"]].copy()
        h["date"] = h["date"].dt.strftime("%Y-%m-%d")
        bets = bets.merge(h.drop_duplicates(KEY), on=KEY, how="left")
    else:
        bets["hg"] = bets["ag"] = np.nan
    bets["profit"] = bets.apply(_settle, axis=1)
    out = []
    for r in bets.sort_values("date", kind="stable").itertuples(index=False):
        out.append({
            "date": str(r.date), "league": config.DIV_NAMES.get(r.div, r.div),
            "home": names.display(r.home), "away": names.display(r.away),
            "market": r.market, "selection": r.selection, "odds": float(r.odds), "p": float(r.p),
            "profit": None if pd.isna(r.profit) else float(r.profit),
            "score": None if pd.isna(r.hg) else f"{int(r.hg)}-{int(r.ag)}",
        })
    return out


TOURNAMENTS_BG = {
    "Friendly": "Приятелски", "UEFA Nations League": "Лига на нациите",
    "FIFA World Cup qualification": "Квалификации за СП", "UEFA Euro qualification": "Квалификации за Евро",
    "FIFA World Cup": "Световно първенство", "UEFA Euro": "Европейско първенство",
}


def _past_row(date, div, league, home, away, hg, ag, hc, ac, ph, pd_, pa, over, btts, cover, pick, retro=False):
    f = lambda v: None if v is None or pd.isna(v) else float(v)
    return {
        "date": date, "div": div, "league": league, "country": config.DIV_COUNTRY.get(div, ""),
        "home": names.display(home), "away": names.display(away), "hg": int(hg), "ag": int(ag),
        "corners": None if hc is None or pd.isna(hc) or pd.isna(ac) else int(hc + ac),
        "p_home": f(ph), "p_draw": f(pd_), "p_away": f(pa), "over": f(over), "btts": f(btts), "c_over": f(cover),
        "pick": str(pick), "retro": retro,
    }


def nations_retro(hist: pd.DataFrame, since: str, skip: set) -> list[dict]:
    """Прогнози за изиграните мачове между европейски национални отбори, които
    не са в лога (напр. изиграни преди сайтът да започне да ги записва).
    Изчисляват се след мача, но моделът се обучава САМО с мачовете преди
    съответния прозорец за национални отбори – както би било преди мача."""
    from src import markets
    from src.models import DixonColes
    intl = hist[hist["div"].isin([config.NL_CODE, "INT"])].dropna(subset=["hg", "ag"])
    if len(intl) < 300:
        return []
    europe = set(names.NATIONS_BG)
    tgt = intl[(intl["date"] >= since) & intl["home"].isin(europe) & intl["away"].isin(europe)].copy()
    tgt = tgt[[(d.strftime("%Y-%m-%d"), h, a) not in skip for d, h, a in zip(tgt["date"], tgt["home"], tgt["away"])]]
    if tgt.empty:
        return []
    # прозорци: мачове с разлика до 5 дни делят един модел
    dates = sorted(tgt["date"].unique())
    windows, start = {}, dates[0]
    for i, d in enumerate(dates):
        if i and (d - dates[i - 1]).days > 5:
            start = d
        windows[d] = start
    out, models = [], {}
    for r in tgt.sort_values("date").itertuples(index=False):
        w = windows[r.date]
        if w not in models:
            models[w] = DixonColes(xi=config.INTL_TIME_DECAY_XI).fit(intl[intl["date"] < w], w)
        dc = models[w]
        neutral = bool(r.neutral) if pd.notna(r.neutral) else False
        lam, mu = dc.rates(r.home, r.away, neutral=neutral)
        g = markets.goal_markets(markets.score_matrix(lam, mu, dc.rho), lam, mu)
        probs = {"1": g["p_home"], "X": g["p_draw"], "2": g["p_away"]}
        league = TOURNAMENTS_BG.get(getattr(r, "tournament", ""), getattr(r, "tournament", "") or
                                    config.DIV_NAMES.get(r.div, r.div))
        out.append(_past_row(r.date.strftime("%Y-%m-%d"), r.div, league, r.home, r.away, r.hg, r.ag,
                             None, None, g["p_home"], g["p_draw"], g["p_away"], g["over_2.5"],
                             g["btts_yes"], None, max(probs, key=probs.get), retro=True))
    return out


def past_results(hist: pd.DataFrame, days: int = 60, days_nations: int | None = None) -> list[dict]:
    """Изиграните мачове: последната ни прогноза + резултатът. За националните
    отбори се добавят и мачовете, които не са в лога (виж nations_retro)."""
    if hist.empty:
        return []
    preds = _load(PRED_LOG)
    since = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    since_n = (pd.Timestamp.today() - pd.Timedelta(days=days_nations or days)).strftime("%Y-%m-%d")
    out, skip = [], set()
    if not preds.empty:
        h = hist[["date", "div", "home", "away", "hg", "ag", "hc", "ac"]
                 + (["tournament"] if "tournament" in hist else [])].copy()
        h["date"] = h["date"].dt.strftime("%Y-%m-%d")
        preds["date"] = preds["date"].astype(str)
        preds = preds.rename(columns={"over_2.5": "over25", "c_over_9.5": "cover95"})
        nat = preds["div"].isin([config.NL_CODE, "INT"])
        preds = preds[(~nat & (preds["date"] >= since)) | (nat & (preds["date"] >= since_n))]
        df = preds.merge(h.drop_duplicates(KEY), on=KEY, how="inner").dropna(subset=["hg"])
        for r in df.itertuples(index=False):
            g = lambda k: getattr(r, k, None)
            t = g("tournament")
            league = TOURNAMENTS_BG.get(t, t) if isinstance(t, str) and t and t != "nan" \
                else config.DIV_NAMES.get(r.div, r.div)
            out.append(_past_row(r.date, r.div, league, r.home, r.away, r.hg, r.ag, r.hc, r.ac,
                                 g("p_home"), g("p_draw"), g("p_away"), g("over25"), g("btts_yes"),
                                 g("cover95"), r.pick))
            skip.add((r.date, r.home, r.away))
    out += nations_retro(hist, since_n, skip)
    return sorted(out, key=lambda m: (m["date"], m["div"]), reverse=True)
