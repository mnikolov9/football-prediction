"""Настройка на модела срещу бектеста.

1. Модел: за всяка държава се търси комбинацията от времево тегло (xi) и
   регуляризация (l2) с най-нисък log loss при walk-forward прогнози.
2. Value: прогнозите от най-добрия модел се делят по време на две половини.
   Правилата (тегло на модела спрямо пазара и минимално предимство) се избират
   по първата половина и се ПРОВЕРЯВАТ върху втората. Пазар остава включен само
   ако печели и в двете половини – иначе value залозите за него се изключват.
"""
from __future__ import annotations

import datetime as dt
import itertools
import json

import numpy as np
import pandas as pd

import config
from src import data, params
from src.backtest import logloss_1x2, value_profits, walk_forward

XI_GRID = [0.0010, 0.0019, 0.0030, 0.0045]
L2_GRID = [0.3, 1.0, 3.0]
WEIGHT_GRID = [0.2, 0.35, 0.5, 0.7]
EDGE_GRID = [0.03, 0.05, 0.08, 0.12]
MIN_BETS_TUNE = 30
MIN_BETS_VALID = 20
TEST_DAYS = 300


def tune_model(hist: pd.DataFrame) -> tuple[float, float, float]:
    best = None
    for xi, l2 in itertools.product(XI_GRID, L2_GRID):
        df = walk_forward(hist, xi=xi, l2=l2, test_days=TEST_DAYS, step=14)
        if df.empty:
            continue
        ll = logloss_1x2(df)
        print(f"    xi={xi:<7} l2={l2:<4} log loss={ll:.4f}")
        if best is None or ll < best[2]:
            best = (xi, l2, ll)
    return best


def tune_value(df: pd.DataFrame, market: str) -> dict:
    df = df.dropna(subset=["oh", "od", "oa"] if market == "1X2" else ["oo", "ou"]).sort_values("date")
    if len(df) < 100:
        return {"enabled": False, "reason": "малко мачове с коефициенти"}
    half = df["date"].iloc[len(df) // 2]
    tune, valid = df[df["date"] < half], df[df["date"] >= half]
    best = None
    for w, e in itertools.product(WEIGHT_GRID, EDGE_GRID):
        p = value_profits(tune, market, {"weight": w, "min_edge": e})
        if len(p) >= MIN_BETS_TUNE and (best is None or p.mean() > best[2]):
            best = (w, e, float(p.mean()), len(p))
    if best is None:
        return {"enabled": False, "reason": "твърде малко залози"}
    w, e, roi_t, n_t = best
    pv = value_profits(valid, market, {"weight": w, "min_edge": e})
    roi_v = float(pv.mean()) if len(pv) else None
    enabled = roi_t > 0 and roi_v is not None and roi_v > 0 and len(pv) >= MIN_BETS_VALID
    return {"enabled": bool(enabled), "weight": w, "min_edge": e,
            "tune_roi": roi_t, "tune_bets": n_t, "valid_roi": roi_v, "valid_bets": int(len(pv))}


def run(offline: bool = False) -> dict:
    out = params.load()
    for country, divs in config.COUNTRIES.items():
        print(f"Настройка: {country}")
        hist = data.load_history(list(divs), offline=offline)
        if len(hist) < 600:
            continue
        xi, l2, ll = tune_model(hist)
        base = walk_forward(hist, xi=config.TIME_DECAY_XI, l2=config.L2_PENALTY,
                            test_days=TEST_DAYS, step=14)
        preds = walk_forward(hist, xi=xi, l2=l2, test_days=TEST_DAYS, step=7).reset_index(drop=True)
        res = {"xi": xi, "l2": l2, "logloss": ll,
               "logloss_default": logloss_1x2(base) if len(base) else None}
        mk = preds.dropna(subset=["oh", "od", "oa"])
        if len(mk):
            inv = 1 / mk[["oh", "od", "oa"]].to_numpy()
            M = inv / inv.sum(1, keepdims=True)
            r = np.where(mk.hg > mk.ag, 0, np.where(mk.hg == mk.ag, 1, 2))
            res["logloss_market"] = float(-np.mean(np.log(M[np.arange(len(r)), r])))
        res["value"] = {m: tune_value(preds, m) for m in ("1X2", "OU")}
        print(f"  -> xi={xi}, l2={l2}, log loss {ll:.4f} (преди {res['logloss_default']:.4f}, "
              f"пазар {res.get('logloss_market', float('nan')):.4f})")
        for m, v in res["value"].items():
            print(f"     value {m}: {'ВКЛ' if v['enabled'] else 'ИЗКЛ'} {v}")
        out[country] = res
    out["_generated"] = dt.date.today().isoformat()
    params.TUNED.parent.mkdir(parents=True, exist_ok=True)
    params.TUNED.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    return out
