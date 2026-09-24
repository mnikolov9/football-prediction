"""Track record: пази публикуваните прогнози и ги оценява след мачовете."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config

PRED_LOG = config.LOG_DIR / "predictions.csv"
BET_LOG = config.LOG_DIR / "value_bets.csv"
KEY = ["date", "div", "home", "away"]


def _load(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _save_merged(path, new: pd.DataFrame, today: str):
    old = _load(path)
    if len(old):
        # прогнозите за още неизиграни мачове се заменят с най-новите
        keys_new = set(map(tuple, new[KEY].astype(str).to_numpy())) if len(new) else set()
        stale = old["date"].astype(str).ge(today) & old[KEY].astype(str).apply(tuple, axis=1).isin(keys_new)
        old = old[~stale]
    out = pd.concat([old, new], ignore_index=True) if len(old) else new
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
    if rows:
        _save_merged(PRED_LOG, pd.DataFrame(rows), today)
    if bets:
        _save_merged(BET_LOG, pd.DataFrame(bets), today)


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
